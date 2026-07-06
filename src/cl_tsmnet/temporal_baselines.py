import torch
from torch import nn


class LSTMClassifier(nn.Module):
    requires_domain = False

    def __init__(self, nchannels, nclasses, hidden_size=64, num_layers=1,
                 dropout=0.5, bidirectional=False):
        super().__init__()
        self.bidirectional = bool(bidirectional)
        recurrent_dropout = float(dropout) if int(num_layers) > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=int(nchannels),
            hidden_size=int(hidden_size),
            num_layers=int(num_layers),
            dropout=recurrent_dropout,
            bidirectional=self.bidirectional,
        )
        out_dim = int(hidden_size) * (2 if self.bidirectional else 1)
        self.dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Linear(out_dim, int(nclasses))

    def forward(self, x):
        x = x.permute(2, 0, 1).contiguous()
        out, _ = self.lstm(x)
        last = out[-1]
        return self.classifier(self.dropout(last))


class TransformerClassifier(nn.Module):
    requires_domain = False

    def __init__(self, nchannels, nsamples, nclasses, d_model=64, num_heads=4,
                 num_layers=2, dim_feedforward=128, dropout=0.2):
        super().__init__()
        d_model = int(d_model)
        num_heads = int(num_heads)
        if d_model % num_heads != 0:
            raise ValueError("transformer d_model must be divisible by num_heads")
        self.input_proj = nn.Linear(int(nchannels), d_model)
        self.positional = nn.Parameter(torch.zeros(int(nsamples), 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(num_layers))
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Linear(d_model, int(nclasses))
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.positional, mean=0.0, std=0.02)

    def forward(self, x):
        x = x.permute(2, 0, 1).contiguous()
        z = self.input_proj(x) + self.positional[:x.shape[0]]
        z = self.encoder(z)
        z = self.norm(z.mean(dim=0))
        return self.classifier(self.dropout(z))


class ShallowCNN(nn.Module):
    requires_domain = False

    def __init__(self, nchannels, nsamples, nclasses, filters=40,
                 temporal_kernel=25, pool_size=25, dropout=0.5):
        super().__init__()
        self.temporal = nn.Conv2d(
            1, int(filters), kernel_size=(1, int(temporal_kernel)), bias=False
        )
        self.spatial = nn.Conv2d(
            int(filters), int(filters), kernel_size=(int(nchannels), 1), bias=False
        )
        self.bn = nn.BatchNorm2d(int(filters))
        self.pool = nn.AvgPool2d(kernel_size=(1, int(pool_size)),
                                 stride=(1, max(1, int(pool_size) // 2)))
        self.dropout = nn.Dropout(float(dropout))
        with torch.no_grad():
            probe = torch.zeros(1, int(nchannels), int(nsamples))
            flat_dim = int(self._features(probe).shape[1])
        self.classifier = nn.Linear(flat_dim, int(nclasses))

    def _features(self, x):
        x = x[:, None, :, :]
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.bn(x)
        x = torch.clamp(x * x, min=1e-6, max=1e6)
        x = self.pool(x)
        x = torch.log(torch.clamp(x, min=1e-6))
        x = self.dropout(x)
        return x.flatten(start_dim=1)

    def forward(self, x):
        return self.classifier(self._features(x))
