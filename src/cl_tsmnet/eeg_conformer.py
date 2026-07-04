import math

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """EEG-Conformer convolutional patch embedding adapted to dynamic EEG shapes."""

    def __init__(self, nchannels, nsamples, emb_size=40, temporal_kernel=25,
                 dropout=0.5):
        super().__init__()
        kernel = int(min(max(3, temporal_kernel), nsamples))
        pool_size = int(max(4, min(75, nsamples // 8)))
        pool_stride = int(max(1, pool_size // 5))

        self.shallownet = nn.Sequential(
            nn.Conv2d(1, emb_size, kernel_size=(1, kernel), stride=(1, 1)),
            nn.Conv2d(emb_size, emb_size, kernel_size=(nchannels, 1), stride=(1, 1)),
            nn.BatchNorm2d(emb_size),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, pool_size), stride=(1, pool_stride)),
            nn.Dropout(dropout),
        )
        self.projection = nn.Conv2d(emb_size, emb_size, kernel_size=(1, 1))

    def forward(self, x):
        # x: batch x channels x samples
        x = x.unsqueeze(1)
        x = self.shallownet(x)
        x = self.projection(x)
        return x.squeeze(2).transpose(1, 2)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, emb_size=40, num_heads=5, drop_p=0.5,
                 forward_expansion=4, forward_drop_p=0.5):
        super().__init__()
        self.att_norm = nn.LayerNorm(emb_size)
        self.attn = nn.MultiheadAttention(
            embed_dim=emb_size,
            num_heads=num_heads,
            dropout=drop_p,
            batch_first=True,
        )
        self.att_drop = nn.Dropout(drop_p)
        self.ff_norm = nn.LayerNorm(emb_size)
        self.ff = nn.Sequential(
            nn.Linear(emb_size, forward_expansion * emb_size),
            nn.GELU(),
            nn.Dropout(forward_drop_p),
            nn.Linear(forward_expansion * emb_size, emb_size),
        )
        self.ff_drop = nn.Dropout(drop_p)

    def forward(self, x):
        h = self.att_norm(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.att_drop(attn_out)
        h = self.ff_norm(x)
        x = x + self.ff_drop(self.ff(h))
        return x


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth=6, emb_size=40, num_heads=5, drop_p=0.5):
        blocks = [TransformerEncoderBlock(emb_size=emb_size, num_heads=num_heads,
                                          drop_p=drop_p)
                  for _ in range(depth)]
        super().__init__(*blocks)


class EEGConformer(nn.Module):
    """Convolutional Transformer baseline from EEG-Conformer.

    The original repository hard-codes channel counts and classifier input sizes
    for BCI/SEED datasets. This version keeps the same core design while making
    the channel count, time samples, and class count configurable.
    """

    requires_domain = False

    def __init__(self, nchannels, nsamples, nclasses, emb_size=40, depth=6,
                 num_heads=5, temporal_kernel=25, dropout=0.5,
                 classifier_hidden=256):
        super().__init__()
        if emb_size % num_heads != 0:
            raise ValueError("emb_size must be divisible by num_heads")
        self.patch_embedding = PatchEmbedding(
            nchannels=nchannels,
            nsamples=nsamples,
            emb_size=emb_size,
            temporal_kernel=temporal_kernel,
            dropout=dropout,
        )
        self.transformer = TransformerEncoder(
            depth=depth,
            emb_size=emb_size,
            num_heads=num_heads,
            drop_p=dropout,
        )
        with torch.no_grad():
            dummy = torch.zeros(1, nchannels, nsamples)
            n_features = int(self.transformer(self.patch_embedding(dummy)).numel())
        self.classifier = nn.Sequential(
            nn.Linear(n_features, classifier_hidden),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, nclasses),
        )

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.transformer(x)
        features = x.contiguous().view(x.size(0), -1)
        return self.classifier(features)
