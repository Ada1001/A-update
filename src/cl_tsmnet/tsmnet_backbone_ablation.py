"""Independent TSMNet-CNN ablations; existing AGMNet modules are not modified."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .training import build_tsmnet
from .ms_tgc_spddsbn import DomainBatchNorm1d, GraphSPDManifoldHead

VARIANTS = ("mean-ce", "mean-eudsbn", "augspd-spdbn", "tsmnet", "augspd-spddsbn")


class TSMBackboneAblation(nn.Module):
    """CPU float32 author CNN; float64 descriptors and linear classifiers.

    tsmnet means author covariance TSMNet + SPDDSBN, NOT augmented SPD.
    EuDSBN shares affine parameters between domains; only moments are specific.
    """
    def __init__(self, variant, nchannels, nsamples, nclasses, domains,
                 spatial_filters=40, subspacedims=20, temporal_filters=4, temp_kernel=25):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.variant = variant
        base = build_tsmnet(str(Path(__file__).resolve().parents[2]), nchannels, nsamples,
            nclasses, domains, bnorm="spddsbn", temporal_filters=temporal_filters,
            spatial_filters=spatial_filters, subspacedims=subspacedims,
            temp_kernel=temp_kernel, device=torch.device("cpu"))
        if variant == "tsmnet":
            self.original = base
        else:
            self.cnn = base.cnn
            if variant.startswith("mean"):
                self.classifier = nn.Linear(spatial_filters, nclasses).double()
                if variant == "mean-eudsbn":
                    self.eudsbn = DomainBatchNorm1d(spatial_filters, domains).double()
                    # Independent domain moments, shared trainable affine transform.
                    for key in list(self.eudsbn.layers):
                        self.eudsbn.layers[key] = nn.BatchNorm1d(spatial_filters, affine=False).double()
                    self.affine_weight = nn.Parameter(torch.ones(spatial_filters, dtype=torch.double))
                    self.affine_bias = nn.Parameter(torch.zeros(spatial_filters, dtype=torch.double))
            else:
                self.head = GraphSPDManifoldHead(spatial_filters, subspacedims,
                    "spddsbn" if variant.endswith("spddsbn") else "spdbn", domains,
                    representation="augmented", shrinkage=.1)
                self.classifier = nn.Linear(self.head.latent_dim, nclasses).double()
        from spdnets.batchnorm import ConstantMomentumBatchNormScheduler
        from types import SimpleNamespace
        scheduler = ConstantMomentumBatchNormScheduler(.1, .1).initialize()
        scheduler.on_train_begin(SimpleNamespace(module_=self))

    @property
    def adaptive(self):
        return self.variant in ("mean-eudsbn", "tsmnet", "augspd-spddsbn")

    def forward(self, x, d, intermediates=False):
        if self.variant == "tsmnet":
            logits, features, post, pre = self.original(x, d, return_latent=True,
                                                        return_prebn=True, return_postbn=True)
            values = dict(features=features, pre=pre, post=post)
        else:
            h = self.cnn(x[:, None])
            if self.variant.startswith("mean"):
                features = h.mean(-1).double()
                if self.variant == "mean-eudsbn":
                    features = self.eudsbn(features, d)*self.affine_weight+self.affine_bias
                values = dict(features=features)
            else:
                features, spd = self.head(h[:, None], d, return_intermediates=True)
                values = dict(features=features, pre=spd["spd_pre_bn"], post=spd["spd_post_bn"])
            logits = self.classifier(features)
        return (logits, values) if intermediates else logits

    @torch.no_grad()
    def refit(self, x, d, batch_size=16):
        """Only supplied domains, no labels; caller restricts training/validation/test."""
        import spdnets.batchnorm as bn
        self.eval()
        if self.variant == "mean-ce":
            return
        groups = [np.arange(len(x))] if self.variant == "augspd-spdbn" else [
            np.flatnonzero(d.numpy()==int(domain)) for domain in d.unique()]
        for ix in groups:
            parts=[]
            for start in range(0,len(ix),batch_size):
                batch=ix[start:start+batch_size]
                if self.variant == "tsmnet":
                    _, pre=self.original(x[batch],d[batch],return_latent=False,return_prebn=True)
                else:
                    h=self.cnn(x[batch,None])
                    pre=h.mean(-1).double() if self.variant == "mean-eudsbn" else self.head.manifold_features(h[:,None])
                parts.append(pre)
            pre=torch.cat(parts)
            if self.variant == "mean-eudsbn":
                self.eudsbn.refit_domain_stats(pre,d[ix])
                continue
            layer=(self.original.spddsbnorm if self.variant == "tsmnet" else
                   self.head.spddsbnorm if self.variant == "augspd-spddsbn" else self.head.spdbnorm)
            layer.set_test_stats_mode(bn.BatchNormTestStatsMode.REFIT)
            try:
                layer(pre,d[ix]) if self.adaptive else layer(pre)
            finally:
                layer.set_test_stats_mode(bn.BatchNormTestStatsMode.BUFFER)
