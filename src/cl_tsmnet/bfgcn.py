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


def normalize_A(A):
    A = F.relu(A)
    d = torch.sum(A, dim=1)
    d = 1.0 / torch.sqrt(d + 1e-10)
    D = torch.diag_embed(d)
    return torch.matmul(torch.matmul(D, A), D)


def generate_cheby_adj(A, K):
    support = []
    eye = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
    for i in range(int(K)):
        if i == 0:
            support.append(eye.expand(A.shape[0], -1, -1))
        elif i == 1:
            support.append(A)
        else:
            support.append(torch.matmul(support[-1], A))
    return support


class GraphConvolution(nn.Module):
    def __init__(self, num_in, num_out, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(int(num_in), int(num_out)))
        nn.init.kaiming_normal_(self.weight)
        self.bias = nn.Parameter(torch.zeros(int(num_out))) if bias else None

    def forward(self, x, adj):
        out = torch.matmul(adj, x)
        out = torch.matmul(out, self.weight)
        return out + self.bias if self.bias is not None else out


class Chebynet(nn.Module):
    def __init__(self, xdim, K, num_out, dropout):
        super().__init__()
        self.K = int(K)
        self.gc1 = nn.ModuleList(
            [GraphConvolution(xdim[1], num_out) for _ in range(self.K)]
        )
        self.dp = nn.Dropout(float(dropout))

    def forward(self, x, L):
        adj = generate_cheby_adj(L, self.K)
        result = None
        for i, layer in enumerate(self.gc1):
            current = layer(x, adj[i])
            result = current if result is None else result + current
        return F.relu(result)


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(int(in_size), int(hidden_size)),
            nn.Tanh(),
            nn.Linear(int(hidden_size), 1, bias=False),
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta


class AttentionAdj(nn.Module):
    def __init__(self, in_size, hidden_size=62):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(int(in_size), int(hidden_size)),
            nn.Tanh(),
            nn.Linear(int(hidden_size), 1, bias=False),
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta


class SFGCN(nn.Module):
    def __init__(self, xdim, kadj, num_out, att_hidden, att_plv_hidden,
                 dropout, avgpool):
        super().__init__()
        self.SGCN1 = Chebynet(xdim, kadj, num_out, dropout)
        self.SGCN2 = Chebynet(xdim, kadj, num_out, dropout)
        self.CGCN = Chebynet(xdim, kadj, num_out, dropout)
        self.BN1 = nn.BatchNorm1d(xdim[1])
        self.attention = Attention(num_out, att_hidden)
        self.attentionadj = AttentionAdj(xdim[0], att_plv_hidden)
        self.bn = nn.BatchNorm1d(xdim[0])
        self.mp = nn.AvgPool2d(int(avgpool))
        self.A = nn.Parameter(torch.empty(int(xdim[0]), int(xdim[0])))
        nn.init.kaiming_normal_(self.A)

    def forward(self, x, fadj):
        x = self.BN1(x.transpose(1, 2)).transpose(1, 2)
        fadj = fadj.permute(0, 3, 1, 2)
        fadj, adj_att = self.attentionadj(fadj)
        fadj = normalize_A(fadj)
        static_adj = self.A + torch.eye(self.A.shape[0], device=x.device, dtype=x.dtype)
        sadj = normalize_A(static_adj.unsqueeze(0).expand(x.shape[0], -1, -1))

        emb1 = self.SGCN1(x, sadj)
        com1 = self.CGCN(x, sadj)
        com2 = self.CGCN(x, fadj)
        emb2 = self.SGCN2(x, fadj)
        xcom = (com1 + com2) / 2.0

        emb = torch.stack([emb1, emb2, xcom], dim=1)
        output, branch_att = self.attention(emb)
        output_att = output
        output = F.relu(self.bn(output))
        output = self.mp(output)
        output = output.reshape(output.shape[0], -1)
        return output, branch_att, emb1, com1, com2, emb2, emb, xcom, output_att, adj_att


class BFGCN(nn.Module):
    requires_domain = False
    is_bfgcn = True

    def __init__(self, nclass, xdim, kadj=2, num_out=16, att_hidden=16,
                 att_plv_hidden=None, classifier_hidden=32, avgpool=2,
                 dropout=0.0):
        super().__init__()
        if att_plv_hidden is None:
            att_plv_hidden = xdim[0]
        self.feature = SFGCN(
            xdim=xdim,
            kadj=kadj,
            num_out=num_out,
            att_hidden=att_hidden,
            att_plv_hidden=att_plv_hidden,
            dropout=dropout,
            avgpool=avgpool,
        )
        with torch.no_grad():
            self.feature.eval()
            dummy_x = torch.zeros(2, int(xdim[0]), int(xdim[1]))
            dummy_adj = torch.eye(int(xdim[0])).view(1, int(xdim[0]), int(xdim[0]), 1)
            dummy_adj = dummy_adj.repeat(2, 1, 1, 4)
            n_features = int(self.feature(dummy_x, dummy_adj)[0].shape[1])
            self.feature.train()
        self.class_classifier = nn.Sequential(
            nn.Linear(n_features, int(classifier_hidden)),
            nn.BatchNorm1d(int(classifier_hidden)),
            nn.Dropout(float(dropout)),
            nn.ReLU(),
        )
        self.output = nn.Linear(int(classifier_hidden), int(nclass))
        self.domain_classifier = nn.Sequential(
            nn.Linear(n_features, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100, 2),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, input_data, fadj, alpha=0.0):
        feature, branch_att, emb1, com1, com2, emb2, emb, xcom, output_att, adj_att = (
            self.feature(input_data, fadj)
        )
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        hidden = self.class_classifier(feature)
        class_output = self.output(hidden)
        domain_output = self.domain_classifier(reverse_feature)
        return (class_output, domain_output, emb1, com1, com2, emb2, emb,
                hidden, xcom, output_att, branch_att, adj_att)
