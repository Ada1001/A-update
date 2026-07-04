import torch
import torch.nn as nn


class EEGNet(nn.Module):
    """Dynamic EEGNet baseline adapted from the local EEGNet/ implementation.

    The original EEGNet folder uses synthetic data and expects input shaped as
    batch x time x channels. This version keeps the same convolutional design
    while accepting the project-standard batch x channels x samples tensors.
    """

    requires_domain = False

    def __init__(self, nchannels, nsamples, nclasses, num_temporal_filts=8,
                 num_spatial_filts=2, dropout=0.5, avgpool_factor=4):
        super().__init__()
        self.nchannels = int(nchannels)
        self.nsamples = int(nsamples)
        self.nclasses = int(nclasses)
        self.f1 = int(num_temporal_filts)
        self.d = int(num_spatial_filts)
        self.f2 = self.f1 * self.d
        self.avgpool_factor = int(max(1, min(avgpool_factor, self.nsamples)))

        kernel1 = int(min(9, self.nsamples))
        if kernel1 % 2 == 0:
            kernel1 = max(1, kernel1 - 1)
        pad1 = kernel1 // 2

        pooled_samples = max(1, self.nsamples // self.avgpool_factor)
        kernel2 = int(2 * (pooled_samples // 2) + 1)
        kernel2 = max(1, min(kernel2, pooled_samples if pooled_samples % 2 else pooled_samples - 1))
        pad2 = kernel2 // 2

        self.block1 = nn.Sequential(
            nn.Conv2d(1, self.f1, kernel_size=(1, kernel1), padding=(0, pad1), bias=True),
            nn.Conv2d(self.f1, self.f2, kernel_size=(self.nchannels, 1),
                      groups=self.f1, bias=True),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, self.avgpool_factor)),
            nn.Dropout(float(dropout)),
        )
        self.block2_features = nn.Sequential(
            nn.Conv2d(self.f2, self.f2, kernel_size=(1, kernel2),
                      padding=(0, pad2), groups=self.f2, bias=True),
            nn.Conv2d(self.f2, self.f2, kernel_size=(1, 1), bias=True),
            nn.ELU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, self.nchannels, self.nsamples)
            n_features = int(self._features(dummy).numel())
        self.classifier = nn.Linear(n_features, self.nclasses)

    def _features(self, x):
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2_features(x)
        return x.flatten(start_dim=1)

    def forward(self, x):
        return self.classifier(self._features(x))
