import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class GradientReverseFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = float(alpha)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class WarmStartGRL(nn.Module):
    def __init__(self, alpha=1.0, lo=0.0, hi=1.0, max_iters=1000,
                 auto_step=False):
        super().__init__()
        self.alpha = float(alpha)
        self.lo = float(lo)
        self.hi = float(hi)
        self.max_iters = int(max_iters)
        self.auto_step = bool(auto_step)
        self.iter_num = 0

    def forward(self, x, alpha=None):
        if alpha is None:
            coeff = (
                2.0 * (self.hi - self.lo)
                / (1.0 + math.exp(-self.alpha * self.iter_num / max(1, self.max_iters)))
                - (self.hi - self.lo)
                + self.lo
            )
        else:
            coeff = float(alpha)
        if self.auto_step:
            self.iter_num += 1
        return GradientReverseFunction.apply(x, coeff)


class MultiScaleTemporalConv(nn.Module):
    def __init__(self, in_channels, hidden_dim, kernel_length=16,
                 scale_factors=(1, 2, 4), dropout=0.5):
        super().__init__()
        self.branches = nn.ModuleList()
        for scale in scale_factors:
            kernel = max(3, int(kernel_length) // int(scale))
            if kernel % 2 == 0:
                kernel += 1
            self.branches.append(nn.Sequential(
                nn.Conv1d(int(in_channels), int(hidden_dim),
                          kernel_size=kernel, padding=kernel // 2),
                nn.BatchNorm1d(int(hidden_dim)),
                nn.LeakyReLU(0.2, inplace=True),
                nn.AdaptiveAvgPool1d(1),
            ))
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x):
        tokens = []
        for branch in self.branches:
            tokens.append(branch(x).squeeze(-1))
        return self.dropout(torch.stack(tokens, dim=1))


class DynamicTemporalAttention(nn.Module):
    def __init__(self, feature_dim, num_heads=4):
        super().__init__()
        heads = int(num_heads)
        if int(feature_dim) % heads != 0:
            heads = 1
        self.attn = nn.MultiheadAttention(
            embed_dim=int(feature_dim),
            num_heads=heads,
            batch_first=True,
        )
        self.out = nn.Linear(int(feature_dim), int(feature_dim))

    def forward(self, tokens):
        attended, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        return self.out(attended)


class ContextualGateLayer(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.fc_g = nn.Linear(int(feature_dim) * 2, int(feature_dim))

    def forward(self, z):
        context = torch.mean(z, dim=1, keepdim=True).expand_as(z)
        gate = torch.sigmoid(self.fc_g(torch.cat([z, context], dim=-1)))
        l1_loss = torch.mean(torch.abs(gate)) if self.training else z.new_tensor(0.0)
        return z * gate, l1_loss


class MDTNFeatureExtractor(nn.Module):
    def __init__(self, in_channels, hidden_dim, kernel_length=16,
                 num_heads=4, dropout=0.5):
        super().__init__()
        self.ms_conv = MultiScaleTemporalConv(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            kernel_length=kernel_length,
            dropout=dropout,
        )
        self.attention = DynamicTemporalAttention(hidden_dim, num_heads=num_heads)
        self.ctx_gate = ContextualGateLayer(hidden_dim)
        self.fc_final = nn.Sequential(
            nn.LayerNorm(int(hidden_dim)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(float(dropout)),
        )

    def forward(self, x):
        tokens = self.ms_conv(x)
        z = self.attention(tokens)
        z, l1_loss = self.ctx_gate(z)
        pooled = torch.mean(z, dim=1)
        return self.fc_final(pooled), l1_loss


class ChebyNetLayer(nn.Module):
    def __init__(self, order, in_features, out_features):
        super().__init__()
        self.order = int(order)
        self.weight = nn.Parameter(torch.empty(self.order, int(in_features), int(out_features)))
        self.bias = nn.Parameter(torch.zeros(int(out_features)))
        nn.init.xavier_normal_(self.weight)

    def forward(self, x, laplacian):
        polys = [x]
        if self.order > 1:
            polys.append(torch.bmm(laplacian, x))
        for _ in range(2, self.order):
            polys.append(2.0 * torch.bmm(laplacian, polys[-1]) - polys[-2])
        stacked = torch.stack(polys, dim=0)
        out = torch.einsum("kbni,kio->bno", stacked, self.weight)
        return F.relu(out + self.bias)


class ChebyDiscriminator(nn.Module):
    def __init__(self, num_nodes, feature_dim, num_classes, order=3, dropout=0.5):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.adj_param = nn.Parameter(torch.empty(self.num_nodes, self.num_nodes))
        nn.init.uniform_(self.adj_param, 0.0, 1.0)
        self.cheby_layer = ChebyNetLayer(order, feature_dim, feature_dim)
        self.dropout = nn.Dropout(float(dropout))
        self.fc_domain = nn.Linear(int(feature_dim), 1)
        self.fc_class = nn.Linear(int(feature_dim), int(num_classes))

    def get_adj(self):
        adj = F.relu(self.adj_param + self.adj_param.t())
        eye = torch.eye(self.num_nodes, device=adj.device, dtype=adj.dtype)
        return adj + eye

    def get_laplacian(self, adj):
        degree = torch.sum(adj, dim=1)
        inv_sqrt = torch.diag(torch.pow(degree + 1e-5, -0.5))
        return -torch.mm(torch.mm(inv_sqrt, adj), inv_sqrt)

    def forward(self, features):
        adj = self.get_adj()
        laplacian = self.get_laplacian(adj)
        if features.dim() == 2:
            features = features.unsqueeze(1).expand(-1, self.num_nodes, -1)
        laplacian = laplacian.unsqueeze(0).expand(features.size(0), -1, -1)
        node_hidden = self.dropout(self.cheby_layer(features, laplacian))
        graph_hidden = torch.mean(node_hidden, dim=1)
        domain_pred = torch.sigmoid(self.fc_domain(graph_hidden))
        class_pred = self.fc_class(graph_hidden)
        return domain_pred, class_pred, graph_hidden, adj


class MDTNGMDAModel(nn.Module):
    requires_domain = False

    def __init__(self, in_channels, hidden_dim, num_classes, num_nodes,
                 kernel_length=16, num_heads=4, cheby_order=3, dropout=0.5,
                 max_iter=1000):
        super().__init__()
        self.feature_extractor = MDTNFeatureExtractor(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            kernel_length=kernel_length,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.classifier = nn.Linear(int(hidden_dim), int(num_classes))
        self.grl = WarmStartGRL(max_iters=max_iter)
        self.discriminator = ChebyDiscriminator(
            num_nodes=num_nodes,
            feature_dim=hidden_dim,
            num_classes=num_classes,
            order=cheby_order,
            dropout=dropout,
        )

    def forward(self, x_s, x_t=None, alpha=None):
        f_s, l1_s = self.feature_extractor(x_s)
        y_s = self.classifier(f_s)
        if self.training and x_t is not None:
            f_t, l1_t = self.feature_extractor(x_t)
            y_t = self.classifier(f_t)
            f_all = torch.cat([f_s, f_t], dim=0)
            domain_pred, graph_cls, graph_hidden, adj = self.discriminator(
                self.grl(f_all, alpha=alpha)
            )
            l1_loss = 0.5 * (l1_s + l1_t)
            return y_s, y_t, domain_pred, graph_cls, graph_hidden, adj, l1_loss
        return y_s


class MDTNGMDALoss(nn.Module):
    def __init__(self, lambda_match=0.1, alpha=0.01, beta=0.01,
                 l1_weight=0.01):
        super().__init__()
        self.lambda_match = float(lambda_match)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.l1_weight = float(l1_weight)
        self.bce = nn.BCELoss()
        self.ce = nn.CrossEntropyLoss()

    @staticmethod
    def _mmd(source, target):
        return torch.sum((source.mean(0) - target.mean(0)) ** 2)

    def _conditional_mmd(self, source, target, source_labels, target_labels,
                         num_classes):
        loss = source.new_tensor(0.0)
        for cls in range(int(num_classes)):
            s_mask = source_labels == cls
            t_mask = target_labels == cls
            if bool(torch.any(s_mask)) and bool(torch.any(t_mask)):
                loss = loss + self._mmd(source[s_mask], target[t_mask])
        return loss

    @staticmethod
    def _graph_similarity_loss(h_s, h_t):
        common = min(int(h_s.shape[0]), int(h_t.shape[0]))
        if common < 2:
            return h_s.new_tensor(0.0)
        h_s = h_s[:common]
        h_t = h_t[:common]
        a_s = torch.mm(h_s, h_s.t())
        a_t = torch.mm(h_t, h_t.t())
        a_s = F.normalize(a_s, p=2, dim=1)
        a_t = F.normalize(a_t, p=2, dim=1)
        return torch.norm(a_s - a_t, p="fro") ** 2

    def forward(self, y_s, y_t, domain_pred, graph_cls, graph_hidden,
                source_labels, l1_loss):
        source_labels = source_labels.long()
        n_source = int(y_s.shape[0])
        n_target = int(y_t.shape[0])
        num_classes = int(y_s.shape[1])

        class_loss = self.ce(y_s, source_labels)
        domain_labels = torch.cat([
            torch.ones(n_source, 1, device=y_s.device),
            torch.zeros(n_target, 1, device=y_s.device),
        ], dim=0)
        domain_loss = self.bce(domain_pred, domain_labels)

        h_s, h_t = torch.split(graph_hidden, [n_source, n_target], dim=0)
        marginal = self._mmd(h_s, h_t)
        target_pseudo = torch.argmax(y_t.detach(), dim=1)
        conditional = self._conditional_mmd(
            h_s, h_t, source_labels, target_pseudo, num_classes
        )
        graph_cls_s = graph_cls[:n_source]
        graph_cls_loss = self.ce(graph_cls_s, source_labels)
        similarity = self._graph_similarity_loss(h_s, h_t)
        match_loss = graph_cls_loss + self.lambda_match * similarity

        total = (
            class_loss
            + domain_loss
            + match_loss
            + self.alpha * marginal
            + self.beta * conditional
            + self.l1_weight * l1_loss
        )
        return total, {
            "class": class_loss.detach(),
            "domain": domain_loss.detach(),
            "match": match_loss.detach(),
            "marginal": marginal.detach(),
            "conditional": conditional.detach(),
        }
