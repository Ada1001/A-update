import torch
from torch import nn
import torch.nn.functional as F


def squash(x, dim=-1, eps=1e-8):
    squared_norm = torch.sum(x * x, dim=dim, keepdim=True)
    scale = squared_norm / (1.0 + squared_norm)
    return scale * x / torch.sqrt(squared_norm + eps)


class DigitCapsules(nn.Module):
    def __init__(self, num_routes, in_dim, num_capsules, out_dim, routing_iters=3):
        super().__init__()
        self.num_routes = int(num_routes)
        self.in_dim = int(in_dim)
        self.num_capsules = int(num_capsules)
        self.out_dim = int(out_dim)
        self.routing_iters = int(routing_iters)
        self.weight = nn.Parameter(
            0.01 * torch.randn(1, self.num_routes, self.num_capsules, self.out_dim, self.in_dim)
        )

    def forward(self, x):
        batch = x.shape[0]
        weight = self.weight.expand(batch, -1, -1, -1, -1)
        u_hat = torch.matmul(weight, x[:, :, None, :, None]).squeeze(-1)
        logits = x.new_zeros(batch, self.num_routes, self.num_capsules)
        for step in range(self.routing_iters):
            coeff = F.softmax(logits, dim=2)
            s = torch.sum(coeff[:, :, :, None] * u_hat, dim=1)
            v = squash(s, dim=-1)
            if step < self.routing_iters - 1:
                logits = logits + torch.sum(u_hat * v[:, None, :, :], dim=-1)
        return v


class LSCCN(nn.Module):
    """Latent Space Coding Capsule Network for fused EEG power/connectivity features."""

    requires_domain = False

    def __init__(self, nchannels, nfeatures, nclasses, latent_dim=200,
                 conv_filters=16, primary_filters=32, primary_dim=4,
                 digit_dim=16, routing_iters=3):
        super().__init__()
        self.nchannels = int(nchannels)
        self.nfeatures = int(nfeatures)
        self.nclasses = int(nclasses)
        self.latent_dim = int(latent_dim)
        self.primary_dim = int(primary_dim)

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
        )
        with torch.no_grad():
            probe = torch.zeros(1, 1, self.nchannels, self.nfeatures)
            encoded_dim = int(self.encoder(probe).shape[1])
        self.fc_mu = nn.Linear(encoded_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(encoded_dim, self.latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.nchannels * self.nfeatures),
            nn.Sigmoid(),
        )

        self.conv = nn.Conv1d(1, int(conv_filters), kernel_size=9, stride=1)
        self.primary = nn.Conv1d(
            int(conv_filters), int(primary_filters), kernel_size=9, stride=2
        )
        with torch.no_grad():
            z_probe = torch.zeros(1, 1, self.latent_dim)
            primary_out = self.primary(self.conv(z_probe))
            num_routes = int(primary_out.shape[1] // self.primary_dim * primary_out.shape[2])
        self.digit_caps = DigitCapsules(
            num_routes=num_routes,
            in_dim=self.primary_dim,
            num_capsules=self.nclasses,
            out_dim=int(digit_dim),
            routing_iters=int(routing_iters),
        )

    def encode(self, x):
        h = self.encoder(x[:, None, :, :])
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z).view(-1, self.nchannels, self.nfeatures)

        features = F.relu(self.conv(z[:, None, :]))
        primary = self.primary(features)
        batch, channels, length = primary.shape
        usable = (channels // self.primary_dim) * self.primary_dim
        primary = primary[:, :usable, :]
        primary = primary.view(batch, usable // self.primary_dim, self.primary_dim, length)
        primary = primary.permute(0, 1, 3, 2).contiguous()
        primary = primary.view(batch, -1, self.primary_dim)
        primary = squash(primary, dim=-1)
        digit = self.digit_caps(primary)
        logits = torch.norm(digit, dim=-1)
        return logits, recon, mu, logvar
