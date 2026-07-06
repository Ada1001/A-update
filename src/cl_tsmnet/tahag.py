import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = float(alpha)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


def _bn_init(bn, scale=1.0):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0.0)


def _conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode="fan_out")
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0.0)


def gaussian_mmd(source, target, kernel_mul=2.0, kernel_num=5):
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0)
    total1 = total.unsqueeze(1)
    l2_distance = ((total0 - total1) ** 2).sum(2)
    n_samples = int(total.size(0))
    denom = max(1, n_samples ** 2 - n_samples)
    bandwidth = torch.sum(l2_distance.detach()) / float(denom)
    bandwidth = torch.clamp(bandwidth, min=1e-6)
    bandwidth = bandwidth / (kernel_mul ** (kernel_num // 2))
    kernels = 0.0
    for i in range(kernel_num):
        kernels = kernels + torch.exp(-l2_distance / (bandwidth * (kernel_mul ** i)))

    ns = int(source.size(0))
    nt = int(target.size(0))
    xx = kernels[:ns, :ns].mean()
    yy = kernels[ns:, ns:].mean()
    xy = kernels[:ns, ns:].mean()
    yx = kernels[ns:, :ns].mean()
    return xx + yy - xy - yx


class Residual1x1(nn.Module):
    def __init__(self, in_feats, out_feats, residual=True):
        super().__init__()
        if not residual:
            self.res = None
        elif in_feats != out_feats:
            self.res = nn.Sequential(
                nn.Conv1d(in_feats, out_feats, kernel_size=1),
                nn.BatchNorm1d(out_feats),
            )
        else:
            self.res = nn.Identity()
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                _conv_init(module)
            elif isinstance(module, nn.BatchNorm1d):
                _bn_init(module)

    def forward(self, x):
        if self.res is None:
            return 0
        return self.res(x)


class AdaptiveGraphConv(nn.Module):
    def __init__(self, in_feats, out_feats, adj, coff_emb=4,
                 adaptive=True, attention=True, residual=True):
        super().__init__()
        inter_feats = max(1, int(out_feats) // int(coff_emb))
        self.adaptive = bool(adaptive)
        self.attention = bool(attention)
        self.conv_d = nn.Conv1d(in_feats, out_feats, kernel_size=1)
        if self.adaptive:
            self.pa = nn.Parameter(adj.float().clone(), requires_grad=True)
            self.alpha = nn.Parameter(torch.zeros(1), requires_grad=True)
            self.conv_a = nn.Conv1d(in_feats, inter_feats, kernel_size=1)
            self.conv_b = nn.Conv1d(in_feats, inter_feats, kernel_size=1)
        else:
            self.register_buffer("adj", adj.float().clone())

        if self.attention:
            reduction = 2
            hidden = max(1, out_feats // reduction)
            self.fc1_fa = nn.Linear(out_feats, hidden)
            self.fc2_fa = nn.Linear(hidden, out_feats)
            nn.init.kaiming_normal_(self.fc1_fa.weight)
            nn.init.constant_(self.fc1_fa.bias, 0.0)
            nn.init.constant_(self.fc2_fa.weight, 0.0)
            nn.init.constant_(self.fc2_fa.bias, 0.0)

        self.residual = Residual1x1(in_feats, out_feats, residual=True)
        self.global_residual = Residual1x1(in_feats, out_feats, residual=residual)
        self.bn = nn.BatchNorm1d(out_feats)
        self.global_bn = nn.BatchNorm1d(out_feats)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                _conv_init(module)
            elif isinstance(module, nn.BatchNorm1d):
                _bn_init(module)
        _bn_init(self.bn, 1e-6)

    def forward(self, x):
        if self.adaptive:
            a_base = self.pa
            a1 = self.conv_a(x).permute(0, 2, 1)
            a2 = self.conv_b(x)
            adaptive_adj = self.tanh(torch.matmul(a1, a2) / max(1, a1.size(-1)))
            adj = a_base + adaptive_adj * self.alpha
        else:
            adj = self.adj
        y = self.conv_d(torch.matmul(x, adj))
        y = self.bn(y)
        y = self.relu(y + self.residual(x))

        if self.attention:
            fe = y.mean(-1)
            gate = self.relu(self.fc1_fa(fe))
            gate = self.sigmoid(self.fc2_fa(gate))
            y = y * gate.unsqueeze(-1) + y

        y = self.global_bn(y)
        y = self.relu(y + self.global_residual(x))
        return y


class TAHAGSharedNet(nn.Module):
    def __init__(self, in_feats, adj, hidden_dims=(32, 64, 128),
                 adaptive=True, attention=True):
        super().__init__()
        self.data_bn = nn.BatchNorm1d(in_feats)
        _bn_init(self.data_bn)
        h1, h2, h3 = [int(v) for v in hidden_dims]
        self.l1 = AdaptiveGraphConv(in_feats, h1, adj, adaptive=adaptive,
                                    attention=attention, residual=True)
        self.l2 = AdaptiveGraphConv(h1, h2, adj, adaptive=adaptive,
                                    attention=attention, residual=True)
        self.l3 = AdaptiveGraphConv(h2, h3, adj, adaptive=adaptive,
                                    attention=attention, residual=True)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.data_bn(x)
        x = self.l1(x)
        hid1 = x.flatten(start_dim=1)
        x = self.l2(x)
        hid2 = x.flatten(start_dim=1)
        x = self.l3(x)
        feat = x.mean(-1)
        return feat, hid1, hid2


class TAHAGModel(nn.Module):
    requires_domain = False
    is_tahag = True

    def __init__(self, nchannels, in_feats, nclasses, adj=None, dropout=0.25,
                 adaptive=True, attention=True, hidden_dims=(32, 64, 128)):
        super().__init__()
        if adj is None:
            adj = torch.eye(int(nchannels), dtype=torch.float32)
        else:
            adj = torch.as_tensor(adj, dtype=torch.float32)
        last_dim = int(hidden_dims[-1])
        self.shared_net = TAHAGSharedNet(
            in_feats=int(in_feats),
            adj=adj,
            hidden_dims=hidden_dims,
            adaptive=adaptive,
            attention=attention,
        )
        self.domain_classifier = nn.Linear(last_dim, 2)
        self.fc = nn.Linear(last_dim, int(nclasses))
        nn.init.normal_(self.fc.weight, 0.0, math.sqrt(2.0 / float(nclasses)))
        self.dropout = nn.Dropout(float(dropout)) if dropout else nn.Identity()

    def forward(self, x_src, x_tgt=None, alpha=0.0):
        src_feat, src_hid1, src_hid2 = self.shared_net(x_src)
        mmd_loss = x_src.new_tensor(0.0)
        domain_out = None
        if self.training and x_tgt is not None:
            tgt_feat, tgt_hid1, tgt_hid2 = self.shared_net(x_tgt)
            mmd_loss = gaussian_mmd(src_hid1, tgt_hid1) + gaussian_mmd(src_hid2, tgt_hid2)
            src_rev = ReverseLayerF.apply(src_feat, alpha)
            tgt_rev = ReverseLayerF.apply(tgt_feat, alpha)
            domain_out = torch.cat([
                self.domain_classifier(src_rev),
                self.domain_classifier(tgt_rev),
            ], dim=0)
        logits = self.fc(self.dropout(src_feat))
        return logits, domain_out, mmd_loss
