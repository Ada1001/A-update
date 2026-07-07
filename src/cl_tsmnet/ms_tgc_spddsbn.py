import torch
import torch.nn as nn
import torch.nn.functional as F

from .mdtn_gmda import ChebyNetLayer, MDTNFeatureExtractor


class ChebyGraphFeatureEncoder(nn.Module):
    def __init__(self, num_nodes, feature_dim, out_dim=None, order=3,
                 dropout=0.5):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.cheby = ChebyNetLayer(order, feature_dim, feature_dim)
        self.dropout = nn.Dropout(float(dropout))
        self.out = nn.Linear(int(feature_dim), int(out_dim or feature_dim))
        self.adj_param = nn.Parameter(torch.empty(self.num_nodes, self.num_nodes))
        nn.init.uniform_(self.adj_param, 0.0, 1.0)

    def _adjacency(self):
        adj = F.relu(self.adj_param + self.adj_param.t())
        eye = torch.eye(self.num_nodes, device=adj.device, dtype=adj.dtype)
        return adj + eye

    @staticmethod
    def _laplacian(adj):
        degree = torch.sum(adj, dim=1)
        inv_sqrt = torch.diag(torch.pow(degree + 1e-5, -0.5))
        return -torch.mm(torch.mm(inv_sqrt, adj), inv_sqrt)

    def forward(self, features):
        if features.dim() == 2:
            features = features.unsqueeze(1).expand(-1, self.num_nodes, -1)
        adj = self._adjacency()
        laplacian = self._laplacian(adj).unsqueeze(0).expand(
            features.size(0), -1, -1
        )
        node_hidden = self.dropout(self.cheby(features, laplacian))
        graph_hidden = torch.mean(node_hidden, dim=1)
        return self.out(graph_hidden)


class MSTGCSPDDSBN(nn.Module):
    requires_domain = True

    def __init__(self, spd_branch, spd_latent_dim, nchannels, nclasses,
                 temporal_hidden=64, graph_hidden=64, fusion_dim=128,
                 kernel_length=16, num_heads=4, cheby_order=3, dropout=0.5,
                 num_nodes=0):
        super().__init__()
        self.spd_branch = spd_branch
        self.nclasses = int(nclasses)
        self.graph_device = torch.device("cpu")
        graph_nodes = int(num_nodes) if int(num_nodes) > 0 else int(nchannels)

        self.temporal = MDTNFeatureExtractor(
            in_channels=int(nchannels),
            hidden_dim=int(temporal_hidden),
            kernel_length=int(kernel_length),
            num_heads=int(num_heads),
            dropout=float(dropout),
        )
        self.graph = ChebyGraphFeatureEncoder(
            num_nodes=graph_nodes,
            feature_dim=int(temporal_hidden),
            out_dim=int(graph_hidden),
            order=int(cheby_order),
            dropout=float(dropout),
        )
        self.spd_project = nn.Sequential(
            nn.LayerNorm(int(spd_latent_dim)),
            nn.Linear(int(spd_latent_dim), int(fusion_dim)),
            nn.GELU(),
        )
        self.graph_project = nn.Sequential(
            nn.LayerNorm(int(graph_hidden)),
            nn.Linear(int(graph_hidden), int(fusion_dim)),
            nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.Linear(int(fusion_dim) * 2, int(fusion_dim)),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(float(dropout)),
            nn.Linear(int(fusion_dim), int(nclasses)),
        )

    @property
    def spddsbnorm(self):
        return self.spd_branch.spddsbnorm

    def to(self, device=None, dtype=None, non_blocking=False):
        if device is not None:
            self.graph_device = torch.device(device)
            self.spd_branch.to(device=device)
            self.temporal.to(device=device, dtype=dtype, non_blocking=non_blocking)
            self.graph.to(device=device, dtype=dtype, non_blocking=non_blocking)
            self.spd_project.to(device=device, dtype=dtype, non_blocking=non_blocking)
            self.graph_project.to(device=device, dtype=dtype, non_blocking=non_blocking)
            self.gate.to(device=device, dtype=dtype, non_blocking=non_blocking)
            self.classifier.to(device=device, dtype=dtype, non_blocking=non_blocking)
            return self
        return super().to(device=device, dtype=dtype, non_blocking=non_blocking)

    def _spd_latent(self, x, d):
        out = self.spd_branch(x, d, return_latent=True)
        if not isinstance(out, (tuple, list)) or len(out) < 2:
            raise RuntimeError("TSMNet SPD branch did not return latent features.")
        return out[1].to(device=self.graph_device, dtype=torch.float32)

    def forward(self, x, d):
        spd_latent = self._spd_latent(x, d)
        x_graph = x.to(device=self.graph_device, dtype=torch.float32)
        temporal_latent, _ = self.temporal(x_graph)
        graph_latent = self.graph(temporal_latent)

        spd_feature = self.spd_project(spd_latent)
        graph_feature = self.graph_project(graph_latent)
        gate = self.gate(torch.cat([spd_feature, graph_feature], dim=1))
        fused = gate * spd_feature + (1.0 - gate) * graph_feature
        return self.classifier(fused)

    def domainadapt_finetune(self, x, y, d, target_domains):
        self.spd_branch.domainadapt_finetune(
            x=x, y=y, d=d, target_domains=target_domains
        )

    def finetune(self, x, y, d):
        if hasattr(self.spd_branch, "finetune"):
            self.spd_branch.finetune(x=x, y=y, d=d)
