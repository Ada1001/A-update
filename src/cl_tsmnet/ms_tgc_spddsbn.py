import torch
import torch.nn as nn
import torch.nn.functional as F

from .mdtn_gmda import ContextualGateLayer, DynamicTemporalAttention


class SharedMultiScaleChannelTemporal(nn.Module):
    """Apply one shared multi-scale temporal encoder to every EEG channel."""

    def __init__(self, hidden_dim, kernel_length=16, num_heads=4,
                 dropout=0.5, scale_factors=(1, 2, 4), output_samples=64):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.output_samples = int(output_samples)
        self.branches = nn.ModuleList()
        for scale in scale_factors:
            kernel = max(3, int(kernel_length) // int(scale))
            if kernel % 2 == 0:
                kernel += 1
            self.branches.append(nn.Sequential(
                nn.Conv1d(1, self.hidden_dim, kernel_size=kernel,
                          padding=kernel // 2, bias=False),
                nn.BatchNorm1d(self.hidden_dim),
                nn.GELU(),
            ))
        self.scale_attention = DynamicTemporalAttention(
            self.hidden_dim, num_heads=num_heads
        )
        self.context_gate = ContextualGateLayer(self.hidden_dim)
        self.scale_score = nn.Linear(self.hidden_dim, 1)
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x):
        batch, channels, samples = x.shape
        channel_signals = x.reshape(batch * channels, 1, samples)
        maps = torch.stack(
            [branch(channel_signals) for branch in self.branches], dim=1
        )
        scale_tokens = torch.mean(maps, dim=-1)
        attended = self.scale_attention(scale_tokens)
        attended, _ = self.context_gate(attended)
        scale_weights = torch.softmax(self.scale_score(attended), dim=1)
        fused = torch.sum(maps * scale_weights.unsqueeze(-1), dim=1)
        if self.output_samples > 0 and fused.shape[-1] != self.output_samples:
            fused = F.adaptive_avg_pool1d(fused, self.output_samples)
            samples = self.output_samples
        fused = fused.transpose(1, 2)
        fused = self.dropout(self.output_norm(fused)).transpose(1, 2)
        return fused.reshape(batch, channels, self.hidden_dim, samples), scale_weights


class SharedSingleScaleChannelTemporal(nn.Module):
    """Single-scale counterpart used by the w/o DTA ablation."""

    def __init__(self, hidden_dim, kernel_length=16, dropout=0.5,
                 output_samples=64):
        super().__init__()
        kernel = max(3, int(kernel_length))
        if kernel % 2 == 0:
            kernel += 1
        self.hidden_dim = int(hidden_dim)
        self.output_samples = int(output_samples)
        self.encoder = nn.Sequential(
            nn.Conv1d(1, self.hidden_dim, kernel_size=kernel,
                      padding=kernel // 2, bias=False),
            nn.BatchNorm1d(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )

    def forward(self, x):
        batch, channels, samples = x.shape
        maps = self.encoder(x.reshape(batch * channels, 1, samples))
        if self.output_samples > 0 and maps.shape[-1] != self.output_samples:
            maps = F.adaptive_avg_pool1d(maps, self.output_samples)
            samples = self.output_samples
        maps = maps.reshape(batch, channels, self.hidden_dim, samples)
        return maps, None


class ChebyGraphSequenceLayer(nn.Module):
    def __init__(self, order, in_features, out_features):
        super().__init__()
        self.order = int(order)
        self.weight = nn.Parameter(torch.empty(
            self.order, int(in_features), int(out_features)
        ))
        self.bias = nn.Parameter(torch.zeros(int(out_features)))
        nn.init.xavier_normal_(self.weight)

    @staticmethod
    def _propagate(laplacian, features):
        return torch.einsum("nm,bmft->bnft", laplacian, features)

    @staticmethod
    def _project(features, weight):
        return torch.einsum("bnft,fo->bnot", features, weight)

    def forward(self, features, laplacian):
        t0 = features
        output = self._project(t0, self.weight[0])
        if self.order > 1:
            t1 = self._propagate(laplacian, t0)
            output = output + self._project(t1, self.weight[1])
        for order in range(2, self.order):
            t2 = 2.0 * self._propagate(laplacian, t1) - t0
            output = output + self._project(t2, self.weight[order])
            t0, t1 = t1, t2
        return F.relu(output + self.bias[None, None, :, None])


class ChebyGraphFeatureEncoder(nn.Module):
    """Propagate channel features at every time point with shared graph weights."""

    def __init__(self, num_nodes, feature_dim, out_dim=None, order=3,
                 dropout=0.5, graph_mode="adaptive", adjacencies=None,
                 neighbors=4):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.graph_mode = str(graph_mode)
        self.neighbors = min(max(1, int(neighbors)), max(1, self.num_nodes - 1))
        self.cheby = ChebyGraphSequenceLayer(
            order, feature_dim, int(out_dim or feature_dim)
        )
        self.dropout = nn.Dropout(float(dropout))
        if self.graph_mode == "adaptive":
            self.adj_param = nn.Parameter(
                torch.empty(self.num_nodes, self.num_nodes)
            )
            nn.init.uniform_(self.adj_param, 0.0, 1.0)
            self.register_buffer("fixed_adjacencies", torch.empty(0))
            self.graph_logits = None
        else:
            fixed = torch.as_tensor(adjacencies, dtype=torch.float32)
            if fixed.dim() == 2:
                fixed = fixed.unsqueeze(0)
            expected = (self.num_nodes, self.num_nodes)
            if fixed.dim() != 3 or tuple(fixed.shape[1:]) != expected:
                raise ValueError(
                    "Fixed graph shape must be [graphs, {}, {}], got {}".format(
                        self.num_nodes, self.num_nodes, tuple(fixed.shape)
                    )
                )
            self.register_parameter("adj_param", None)
            self.register_buffer("fixed_adjacencies", fixed)
            self.graph_logits = nn.Parameter(torch.zeros(fixed.shape[0])) \
                if fixed.shape[0] > 1 else None

    def _adjacencies(self):
        if self.graph_mode == "adaptive":
            adj = F.softplus(self.adj_param + self.adj_param.t())
            adj = adj - torch.diag_embed(torch.diagonal(adj))
            _, indices = torch.topk(adj, self.neighbors, dim=1)
            mask = torch.zeros_like(adj).scatter_(1, indices, 1.0)
            mask = torch.maximum(mask, mask.t())
            return (adj * mask).unsqueeze(0)
        return torch.clamp(self.fixed_adjacencies, min=0.0)

    @staticmethod
    def _scaled_laplacian(adj):
        degree = torch.sum(adj, dim=1)
        inv_sqrt = torch.diag(torch.pow(degree + 1e-5, -0.5))
        # For normalized L with lambda_max approximated by 2: 2L/2-I = -D^-1/2 A D^-1/2.
        return -torch.mm(torch.mm(inv_sqrt, adj), inv_sqrt)

    def forward(self, features):
        squeeze_time = features.dim() == 3
        if squeeze_time:
            features = features.unsqueeze(-1)
        if features.dim() != 4 or features.shape[1] != self.num_nodes:
            raise ValueError(
                "Cheb graph input must be [batch, {}, features, time], got {}"
                .format(self.num_nodes, tuple(features.shape))
            )
        batch, nodes, _, _ = features.shape
        eye = torch.eye(nodes, device=features.device, dtype=features.dtype)
        graph_outputs = []
        for adj in self._adjacencies():
            adj = 0.5 * (adj + adj.t()) + eye
            laplacian = self._scaled_laplacian(adj)
            graph_outputs.append(self.dropout(self.cheby(features, laplacian)))
        stacked = torch.stack(graph_outputs, dim=1)
        if self.graph_logits is None:
            graph_hidden = stacked[:, 0]
        else:
            weights = torch.softmax(self.graph_logits, dim=0)
            graph_hidden = torch.sum(
                stacked * weights[None, :, None, None, None], dim=1
            )
        return graph_hidden.squeeze(-1) if squeeze_time else graph_hidden


class DomainBatchNorm1d(nn.Module):
    def __init__(self, feature_dim, domains, momentum=0.1):
        super().__init__()
        self.feature_dim = int(feature_dim)
        domain_values = sorted(int(d) for d in set(int(v) for v in domains))
        if not domain_values:
            domain_values = [0]
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
            idx = (domains == domain).to(device=x.device)
            layer = self._layer(int(domain))
            values = x[idx]
            if values.shape[0] < 2 and layer.training:
                out[idx] = F.batch_norm(
                    values, layer.running_mean, layer.running_var,
                    layer.weight, layer.bias, training=False, eps=layer.eps,
                )
            else:
                out[idx] = layer(values)
        return out

    def refit_domain_stats(self, features, domains):
        was_training = self.training
        self.train()
        with torch.no_grad():
            cpu_domains = domains.detach().cpu().long()
            for domain in torch.unique(cpu_domains):
                values = features[(cpu_domains == domain).to(features.device)]
                if values.shape[0] < 2:
                    continue
                layer = self._layer(int(domain))
                layer.reset_running_stats()
                layer.train()
                layer(values)
        self.train(was_training)


class GraphSPDManifoldHead(nn.Module):
    """Embed graph-temporal observations as a first/second-order SPD matrix."""

    def __init__(self, feature_dim, subspacedims, bnorm, domains,
                 shrinkage=0.1, covariance_epsilon=1e-5):
        super().__init__()
        import spdnets.batchnorm as bn
        import spdnets.modules as modules

        self.feature_dim = int(feature_dim)
        self.augmented_dim = self.feature_dim + 1
        self.subspacedims = int(min(subspacedims, self.augmented_dim))
        if self.subspacedims < 1:
            raise ValueError("MS-TGC SPD subspace dimension must be positive")
        self.shrinkage = float(shrinkage)
        if not 0.0 <= self.shrinkage <= 1.0:
            raise ValueError("MS-TGC covariance shrinkage must be in [0, 1]")
        self.covariance_epsilon = float(covariance_epsilon)
        if self.covariance_epsilon <= 0.0:
            raise ValueError("MS-TGC covariance epsilon must be positive")
        self.bnorm = bnorm
        self.spd_device = torch.device("cpu")
        self.spdnet = nn.Sequential(
            modules.BiMap(
                (1, self.augmented_dim, self.subspacedims),
                dtype=torch.double, device=self.spd_device,
            ),
            modules.ReEig(threshold=1e-4),
        )
        if bnorm == "spddsbn":
            self.spddsbnorm = bn.AdaMomDomainSPDBatchNorm(
                (1, self.subspacedims, self.subspacedims), batchdim=0,
                domains=torch.as_tensor(sorted(set(int(d) for d in domains))),
                learn_mean=False, learn_std=True,
                dispersion=bn.BatchNormDispersion.SCALAR,
                eta=1.0, eta_test=0.1, dtype=torch.double,
                device=self.spd_device,
            )
        elif bnorm == "spdbn":
            self.spdbnorm = bn.AdaMomSPDBatchNorm(
                (1, self.subspacedims, self.subspacedims), batchdim=0,
                learn_mean=False, learn_std=True,
                dispersion=bn.BatchNormDispersion.SCALAR,
                eta=1.0, eta_test=0.1, dtype=torch.double,
                device=self.spd_device,
            )
        elif bnorm is not None:
            raise ValueError("Unknown graph SPD normalization: {}".format(bnorm))
        self.logeig = nn.Sequential(
            modules.LogEig(self.subspacedims),
            nn.Flatten(start_dim=1),
        )

    @property
    def latent_dim(self):
        return self.subspacedims * (self.subspacedims + 1) // 2

    def build_augmented_spd(self, maps):
        if maps.dim() != 4 or maps.shape[2] != self.feature_dim:
            raise ValueError(
                "Graph SPD input must be [batch, channels, {}, time], got {}"
                .format(self.feature_dim, tuple(maps.shape))
            )
        batch, channels, features, time_points = maps.shape
        observations = maps.permute(0, 2, 1, 3).reshape(
            batch, features, channels * time_points
        ).to(device=self.spd_device, dtype=torch.double)
        count = observations.shape[-1]
        if count < 2:
            raise ValueError("Graph SPD covariance needs at least two observations")

        mean = observations.mean(dim=-1, keepdim=True)
        centered = observations - mean
        covariance = torch.bmm(centered, centered.transpose(1, 2)) / float(count - 1)
        eye = torch.eye(
            features, device=covariance.device, dtype=covariance.dtype
        ).unsqueeze(0)
        trace_scale = torch.diagonal(
            covariance, dim1=-2, dim2=-1
        ).sum(dim=-1).view(batch, 1, 1) / float(features)
        covariance = (
            (1.0 - self.shrinkage) * covariance
            + self.shrinkage * trace_scale * eye
            + self.covariance_epsilon * eye
        )

        second_moment = covariance + torch.bmm(mean, mean.transpose(1, 2))
        top = torch.cat([second_moment, mean], dim=2)
        bottom = torch.cat([
            mean.transpose(1, 2),
            torch.ones(batch, 1, 1, device=mean.device, dtype=mean.dtype),
        ], dim=2)
        augmented = torch.cat([top, bottom], dim=1)
        augmented = 0.5 * (augmented + augmented.transpose(1, 2))
        return augmented.unsqueeze(1)

    def forward(self, maps, domains):
        augmented = self.build_augmented_spd(maps)
        latent = self.spdnet(augmented)
        if hasattr(self, "spdbnorm"):
            latent = self.spdbnorm(latent)
        if hasattr(self, "spddsbnorm"):
            latent = self.spddsbnorm(
                latent, domains.to(device=self.spd_device)
            )
        return self.logeig(latent)

    def refit_domains(self, maps, domains):
        if not hasattr(self, "spddsbnorm"):
            return
        import spdnets.batchnorm as bn

        self.spddsbnorm.set_test_stats_mode(bn.BatchNormTestStatsMode.REFIT)
        with torch.no_grad():
            for domain in domains.unique():
                self.forward(maps[domains == domain], domains[domains == domain])
        self.spddsbnorm.set_test_stats_mode(bn.BatchNormTestStatsMode.BUFFER)

    def refit_global(self, maps, domains):
        if not hasattr(self, "spdbnorm"):
            return
        import spdnets.batchnorm as bn

        self.spdbnorm.set_test_stats_mode(bn.BatchNormTestStatsMode.REFIT)
        with torch.no_grad():
            self.forward(maps, domains)
        self.spdbnorm.set_test_stats_mode(bn.BatchNormTestStatsMode.BUFFER)


class MSTGCSPDDSBN(nn.Module):
    requires_domain = True

    def __init__(self, spd_branch, spd_latent_dim, nchannels, nclasses,
                 temporal_hidden=64, graph_hidden=64, fusion_dim=128,
                 kernel_length=16, num_heads=4, cheby_order=3, dropout=0.5,
                 num_nodes=0, use_dta=True, use_cheb=True,
                 euclidean_dsbn=False, domains=None, graph_mode="adaptive",
                 graph_adjacencies=None, graph_neighbors=4,
                 graph_time_points=64, channel_weight_epsilon=1e-6):
        super().__init__()
        self.spd_branch = spd_branch
        self.use_spd = spd_branch is not None
        self.use_dta = bool(use_dta)
        self.use_cheb = bool(use_cheb)
        self.use_euclidean_dsbn = bool(euclidean_dsbn)
        self.graph_mode = str(graph_mode)
        self.graph_device = torch.device("cpu")
        self.channel_weight_epsilon = float(channel_weight_epsilon)
        if self.channel_weight_epsilon <= 0.0:
            raise ValueError("Channel-weight epsilon must be positive")
        graph_nodes = int(num_nodes) if int(num_nodes) > 0 else int(nchannels)
        if self.use_cheb and graph_nodes != int(nchannels):
            raise ValueError(
                "MS-TGC electrode graphs require num_nodes to equal the EEG "
                "channel count ({}), got {}".format(nchannels, graph_nodes)
            )

        if self.use_dta:
            self.temporal = SharedMultiScaleChannelTemporal(
                hidden_dim=temporal_hidden, kernel_length=kernel_length,
                num_heads=num_heads, dropout=dropout,
                output_samples=graph_time_points,
            )
        else:
            self.temporal = SharedSingleScaleChannelTemporal(
                hidden_dim=temporal_hidden, kernel_length=kernel_length,
                dropout=dropout, output_samples=graph_time_points,
            )

        if self.use_cheb:
            self.graph = ChebyGraphFeatureEncoder(
                num_nodes=graph_nodes, feature_dim=temporal_hidden,
                out_dim=graph_hidden, order=cheby_order, dropout=dropout,
                graph_mode=self.graph_mode, adjacencies=graph_adjacencies,
                neighbors=graph_neighbors,
            )
            feature_dim = int(graph_hidden)
        else:
            self.graph = None
            feature_dim = int(temporal_hidden)

        self.feature_dim = feature_dim
        self.channel_score = nn.Linear(feature_dim, 1)
        self.eudsbnorm = None
        if self.use_euclidean_dsbn:
            self.eudsbnorm = DomainBatchNorm1d(
                feature_dim, [0] if domains is None else domains
            )

        readout_dim = (
            int(getattr(spd_branch, "latent_dim", spd_latent_dim))
            if self.use_spd else feature_dim
        )
        self.readout = nn.Sequential(
            nn.LayerNorm(readout_dim),
            nn.Linear(readout_dim, int(fusion_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.classifier = nn.Linear(int(fusion_dim), int(nclasses))

    @property
    def spddsbnorm(self):
        if self.spd_branch is not None and hasattr(self.spd_branch, "spddsbnorm"):
            return self.spd_branch.spddsbnorm
        raise AttributeError("This MS-TGC variant has no SPDDSBN layer")

    def to(self, device=None, dtype=None, non_blocking=False):
        if device is None:
            return super().to(device=device, dtype=dtype, non_blocking=non_blocking)
        self.graph_device = torch.device(device)
        self.temporal.to(device=device, dtype=dtype, non_blocking=non_blocking)
        if self.graph is not None:
            self.graph.to(device=device, dtype=dtype, non_blocking=non_blocking)
        self.channel_score.to(device=device, dtype=dtype, non_blocking=non_blocking)
        if self.eudsbnorm is not None:
            self.eudsbnorm.to(device=device, dtype=dtype, non_blocking=non_blocking)
        self.readout.to(device=device, dtype=dtype, non_blocking=non_blocking)
        self.classifier.to(device=device, dtype=dtype, non_blocking=non_blocking)
        return self

    def _weighted_graph_maps(self, x, return_weights=False):
        maps, _ = self.temporal(x.to(self.graph_device, dtype=torch.float32))
        if self.graph is not None:
            maps = self.graph(maps)
        channel_summary = torch.mean(maps, dim=-1)
        channel_weights = torch.softmax(
            self.channel_score(channel_summary), dim=1
        )
        reliability = torch.sqrt(
            channel_weights + self.channel_weight_epsilon
        ).unsqueeze(-1)
        weighted_maps = maps * reliability
        if return_weights:
            return weighted_maps, channel_weights
        return weighted_maps

    def _first_order_readout(self, maps, domains=None, apply_dsbn=True):
        observations = maps.permute(0, 2, 1, 3).reshape(
            maps.shape[0], maps.shape[2], -1
        )
        latent = torch.mean(observations, dim=-1)
        if self.eudsbnorm is not None and apply_dsbn:
            latent = self.eudsbnorm(latent, domains.to(self.graph_device))
        return latent

    def forward(self, x, d):
        maps = self._weighted_graph_maps(x)
        if self.use_spd:
            latent = self.spd_branch(maps, d).to(
                self.graph_device, dtype=torch.float32
            )
        else:
            latent = self._first_order_readout(maps, d, apply_dsbn=True)
        return self.classifier(self.readout(latent))

    def domainadapt_finetune(self, x, y, d, target_domains):
        was_training = self.training
        self.eval()
        with torch.no_grad():
            maps = self._weighted_graph_maps(x)
            if self.spd_branch is not None:
                self.spd_branch.refit_domains(maps, d)
            if self.eudsbnorm is not None:
                features = self._first_order_readout(
                    maps, d, apply_dsbn=False
                )
                self.eudsbnorm.refit_domain_stats(features, d)
        self.train(was_training)

    def finetune(self, x, y, d):
        if self.spd_branch is None:
            return
        was_training = self.training
        self.eval()
        with torch.no_grad():
            maps = self._weighted_graph_maps(x)
            self.spd_branch.refit_global(maps, d)
        self.train(was_training)
