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


class DomainBatchNorm1d(nn.Module):
    def __init__(self, feature_dim, domains, momentum=0.1):
        super().__init__()
        self.feature_dim = int(feature_dim)
        domain_values = sorted(int(d) for d in set(int(v) for v in domains))
        if not domain_values:
            domain_values = [0]
        self.domain_keys = domain_values
        self.layers = nn.ModuleDict({
            str(domain): nn.BatchNorm1d(
                self.feature_dim, momentum=float(momentum), affine=True
            )
            for domain in domain_values
        })

    def _layer(self, domain):
        key = str(int(domain))
        if key not in self.layers:
            self.layers[key] = nn.BatchNorm1d(
                self.feature_dim, momentum=0.1, affine=True
            ).to(next(self.parameters()).device)
        return self.layers[key]

    def forward(self, x, domains):
        domains = domains.detach().cpu().long()
        out = torch.empty_like(x)
        for domain in torch.unique(domains):
            mask = domains == domain
            idx = mask.to(device=x.device)
            layer = self._layer(int(domain))
            values = x[idx]
            if values.shape[0] < 2 and layer.training:
                out[idx] = F.batch_norm(
                    values,
                    layer.running_mean,
                    layer.running_var,
                    layer.weight,
                    layer.bias,
                    training=False,
                    eps=layer.eps,
                )
            else:
                out[idx] = layer(values)
        return out

    def refit_domain_stats(self, features, domains):
        was_training = self.training
        self.train()
        with torch.no_grad():
            for domain in torch.unique(domains.detach().cpu().long()):
                mask = domains.detach().cpu().long() == domain
                values = features[mask.to(device=features.device)]
                if values.shape[0] < 2:
                    continue
                layer = self._layer(int(domain))
                layer.reset_running_stats()
                layer.train()
                layer(values)
        self.train(was_training)


class ChannelSummaryFeatureExtractor(nn.Module):
    def __init__(self, hidden_dim, dropout=0.5):
        super().__init__()
        self.node_project = nn.Sequential(
            nn.Linear(2, int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )

    def forward(self, x):
        mean = torch.mean(x, dim=-1)
        std = torch.std(x, dim=-1, unbiased=False)
        node_features = torch.stack([mean, std], dim=-1)
        return self.node_project(node_features)


class MSTGCSPDDSBN(nn.Module):
    requires_domain = True

    def __init__(self, spd_branch, spd_latent_dim, nchannels, nclasses,
                 temporal_hidden=64, graph_hidden=64, fusion_dim=128,
                 kernel_length=16, num_heads=4, cheby_order=3, dropout=0.5,
                 num_nodes=0, use_dta=True, use_cheb=True,
                 euclidean_dsbn=False, domains=None):
        super().__init__()
        self.spd_branch = spd_branch
        self.use_spd = spd_branch is not None
        self.use_dta = bool(use_dta)
        self.use_cheb = bool(use_cheb)
        self.use_euclidean_dsbn = bool(euclidean_dsbn)
        self.nclasses = int(nclasses)
        self.graph_device = torch.device("cpu")
        if self.use_dta:
            graph_nodes = int(num_nodes) if int(num_nodes) > 0 else int(nchannels)
        else:
            graph_nodes = int(nchannels)

        if self.use_dta:
            self.temporal = MDTNFeatureExtractor(
                in_channels=int(nchannels),
                hidden_dim=int(temporal_hidden),
                kernel_length=int(kernel_length),
                num_heads=int(num_heads),
                dropout=float(dropout),
            )
            temporal_out_dim = int(temporal_hidden)
        else:
            self.temporal = ChannelSummaryFeatureExtractor(
                hidden_dim=int(temporal_hidden), dropout=float(dropout)
            )
            temporal_out_dim = int(temporal_hidden)

        if self.use_cheb:
            self.graph = ChebyGraphFeatureEncoder(
                num_nodes=graph_nodes,
                feature_dim=temporal_out_dim,
                out_dim=int(graph_hidden),
                order=int(cheby_order),
                dropout=float(dropout),
            )
            feature_dim = int(graph_hidden)
        else:
            self.graph = None
            feature_dim = temporal_out_dim

        self.eudsbnorm = None
        if self.use_euclidean_dsbn:
            bn_domains = [0] if domains is None else domains
            self.eudsbnorm = DomainBatchNorm1d(feature_dim, bn_domains)

        if self.use_spd:
            self.spd_project = nn.Sequential(
                nn.LayerNorm(int(spd_latent_dim)),
                nn.Linear(int(spd_latent_dim), int(fusion_dim)),
                nn.GELU(),
            )
            self.graph_project = nn.Sequential(
                nn.LayerNorm(int(feature_dim)),
                nn.Linear(int(feature_dim), int(fusion_dim)),
                nn.GELU(),
            )
            self.gate = nn.Sequential(
                nn.Linear(int(fusion_dim) * 2, int(fusion_dim)),
                nn.Sigmoid(),
            )
            classifier_dim = int(fusion_dim)
        else:
            self.spd_project = None
            self.graph_project = nn.Sequential(
                nn.LayerNorm(int(feature_dim)),
                nn.Linear(int(feature_dim), int(fusion_dim)),
                nn.GELU(),
            )
            self.gate = None
            classifier_dim = int(fusion_dim)

        self.classifier = nn.Sequential(
            nn.Dropout(float(dropout)),
            nn.Linear(classifier_dim, int(nclasses)),
        )

    @property
    def spddsbnorm(self):
        if self.spd_branch is not None and hasattr(self.spd_branch, "spddsbnorm"):
            return self.spd_branch.spddsbnorm
        raise AttributeError("This MS-TGC variant has no SPDDSBN layer")

    def to(self, device=None, dtype=None, non_blocking=False):
        if device is not None:
            self.graph_device = torch.device(device)
            if self.spd_branch is not None:
                self.spd_branch.to(device=device)
            self.temporal.to(device=device, dtype=dtype, non_blocking=non_blocking)
            if self.graph is not None:
                self.graph.to(device=device, dtype=dtype, non_blocking=non_blocking)
            if self.eudsbnorm is not None:
                self.eudsbnorm.to(device=device, dtype=dtype, non_blocking=non_blocking)
            if self.spd_project is not None:
                self.spd_project.to(device=device, dtype=dtype, non_blocking=non_blocking)
            self.graph_project.to(device=device, dtype=dtype, non_blocking=non_blocking)
            if self.gate is not None:
                self.gate.to(device=device, dtype=dtype, non_blocking=non_blocking)
            self.classifier.to(device=device, dtype=dtype, non_blocking=non_blocking)
            return self
        return super().to(device=device, dtype=dtype, non_blocking=non_blocking)

    def _spd_latent(self, x, d):
        out = self.spd_branch(x, d, return_latent=True)
        if not isinstance(out, (tuple, list)) or len(out) < 2:
            raise RuntimeError("TSMNet SPD branch did not return latent features.")
        return out[1].to(device=self.graph_device, dtype=torch.float32)

    def _temporal_graph_latent(self, x, d=None, apply_dsbn=True):
        x_graph = x.to(device=self.graph_device, dtype=torch.float32)
        if self.use_dta:
            temporal_latent, _ = self.temporal(x_graph)
        else:
            temporal_latent = self.temporal(x_graph)
        graph_latent = self.graph(temporal_latent) if self.graph is not None else temporal_latent
        if graph_latent.dim() == 3:
            graph_latent = torch.mean(graph_latent, dim=1)
        if self.eudsbnorm is not None and apply_dsbn:
            graph_latent = self.eudsbnorm(graph_latent, d.to(self.graph_device))
        return graph_latent

    def forward(self, x, d):
        graph_latent = self._temporal_graph_latent(x, d=d, apply_dsbn=True)
        graph_feature = self.graph_project(graph_latent)
        if not self.use_spd:
            return self.classifier(graph_feature)
        spd_latent = self._spd_latent(x, d)
        spd_feature = self.spd_project(spd_latent)
        gate = self.gate(torch.cat([spd_feature, graph_feature], dim=1))
        return self.classifier(gate * spd_feature + (1.0 - gate) * graph_feature)

    def domainadapt_finetune(self, x, y, d, target_domains):
        if self.spd_branch is not None and hasattr(self.spd_branch, "domainadapt_finetune"):
            self.spd_branch.domainadapt_finetune(
                x=x, y=y, d=d, target_domains=target_domains
            )
        if self.eudsbnorm is not None:
            was_training = self.training
            self.eval()
            with torch.no_grad():
                features = self._temporal_graph_latent(x, d=d, apply_dsbn=False)
                self.eudsbnorm.refit_domain_stats(features, d.to(self.graph_device))
            self.train(was_training)

    def finetune(self, x, y, d):
        if self.spd_branch is not None and hasattr(self.spd_branch, "finetune"):
            self.spd_branch.finetune(x=x, y=y, d=d)
