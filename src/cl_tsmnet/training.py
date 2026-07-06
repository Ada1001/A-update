import copy
import os
import sys
import warnings

import numpy as np
import torch
from scipy import signal
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.svm import LinearSVC, SVC
from torch.utils.data import DataLoader, Dataset


BFGCN_FEATURE_BANDS = [(1.0, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 45.0)]
BFGCN_PLV_BANDS = [(4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 45.0)]
LSCCN_CONNECTIVITY_BAND = (30.0, 45.0)


class EEGWindowDataset(Dataset):
    def __init__(self, x, y, d, indices, augment=False, noise_std=0.03,
                 shift_samples=8, channel_dropout=0.05, normalizer=None):
        self.x = x
        self.y = y
        self.d = d
        self.indices = np.asarray(indices, dtype=np.int64)
        self.augment = bool(augment)
        self.noise_std = float(noise_std)
        self.shift_samples = int(shift_samples)
        self.channel_dropout = float(channel_dropout)
        self.normalizer = normalizer

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        idx = self.indices[item]
        x = np.array(self.x[idx], copy=True)
        if self.normalizer is not None:
            x = self.normalizer.transform_window(x)
        if self.augment:
            if self.shift_samples > 0:
                shift = np.random.randint(-self.shift_samples, self.shift_samples + 1)
                x = np.roll(x, shift, axis=-1)
            if self.noise_std > 0:
                x = x + np.random.normal(0.0, self.noise_std, size=x.shape).astype(np.float32)
            if self.channel_dropout > 0:
                mask = np.random.rand(x.shape[0]) < self.channel_dropout
                x[mask, :] = 0.0
        return (torch.from_numpy(x.astype(np.float32)),
                torch.tensor(int(self.y[idx]), dtype=torch.long),
                torch.tensor(int(self.d[idx]), dtype=torch.long))


class RobustSourceNormalizer:
    def __init__(self, center, scale):
        self.center = np.asarray(center, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)

    def transform_window(self, x):
        return (x - self.center[:, None]) / self.scale[:, None]

    def transform_array(self, x):
        return (x - self.center[None, :, None]) / self.scale[None, :, None]


def fit_source_normalizer(x, indices):
    idx = np.asarray(indices, dtype=np.int64)
    train_x = np.asarray(x[idx], dtype=np.float32)
    center = np.median(train_x, axis=(0, 2))
    mad = np.median(np.abs(train_x - center[None, :, None]), axis=(0, 2))
    scale = (1.4826 * mad).astype(np.float32)
    small = scale < 1e-8
    if np.any(small):
        fallback = np.std(train_x, axis=(0, 2)).astype(np.float32)
        scale[small] = fallback[small]
    scale[scale < 1e-8] = 1.0
    return RobustSourceNormalizer(center, scale)


def _filter_artifact_windows(x, indices, normalizer, artifact_z):
    idx = np.asarray(indices, dtype=np.int64)
    if artifact_z is None:
        return idx
    keep = []
    threshold = float(artifact_z)
    for i in idx:
        w = normalizer.transform_window(x[i])
        if np.isfinite(w).all() and float(np.max(np.abs(w))) <= threshold:
            keep.append(int(i))
    return np.asarray(keep, dtype=np.int64)


def _metrics_from_arrays(y_true, y_pred, y_prob):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    if len(y_true) == 0:
        return {
            "accuracy": np.nan,
            "balanced_accuracy": np.nan,
            "f1": np.nan,
            "auc": np.nan,
        }
    auc = np.nan
    try:
        if y_prob.shape[1] == 2:
            auc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            auc = roc_auc_score(y_true, y_prob, multi_class="ovr",
                                average="macro")
    except (TypeError, ValueError):
        auc = np.nan
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        f1 = f1_score(y_true, y_pred, average="macro")
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1,
        "auc": auc,
    }


def _ensure_tsmnet_on_path(project_root):
    tsmnet_path = os.path.join(project_root, "TSMNet")
    if tsmnet_path not in sys.path:
        sys.path.insert(0, tsmnet_path)


def build_tsmnet(project_root, nchannels, nsamples, nclasses, domains,
                 bnorm="spddsbn", temporal_filters=4, spatial_filters=40,
                 subspacedims=20, temp_kernel=25, device=None):
    _ensure_tsmnet_on_path(project_root)
    from spdnets.models import TSMNet

    domains_tensor = torch.as_tensor(np.unique(domains), dtype=torch.long)
    if bnorm in ("none", "null", "None"):
        bnorm = None
    model = TSMNet(
        temporal_filters=int(temporal_filters),
        spatial_filters=int(spatial_filters),
        subspacedims=int(min(subspacedims, spatial_filters)),
        temp_cnn_kernel=int(temp_kernel),
        bnorm=bnorm,
        bnorm_dispersion="SCALAR" if bnorm else None,
        nclasses=int(nclasses),
        nchannels=int(nchannels),
        nsamples=int(nsamples),
        domains=domains_tensor,
        device=device or torch.device("cpu"),
    )
    return model


def build_eegconformer(nchannels, nsamples, nclasses, temporal_kernel=25,
                       emb_size=40, depth=6, num_heads=5, dropout=0.5,
                       classifier_hidden=256):
    from .eeg_conformer import EEGConformer

    return EEGConformer(
        nchannels=int(nchannels),
        nsamples=int(nsamples),
        nclasses=int(nclasses),
        emb_size=int(emb_size),
        depth=int(depth),
        num_heads=int(num_heads),
        temporal_kernel=int(temporal_kernel),
        dropout=float(dropout),
        classifier_hidden=int(classifier_hidden),
    )


def build_eegnet(nchannels, nsamples, nclasses, temporal_filters=64,
                 spatial_filters=4, dropout=0.5, avgpool_factor=2):
    from .eegnet import EEGNet

    return EEGNet(
        nchannels=int(nchannels),
        nsamples=int(nsamples),
        nclasses=int(nclasses),
        num_temporal_filts=int(temporal_filters),
        num_spatial_filts=int(spatial_filters),
        dropout=float(dropout),
        avgpool_factor=int(avgpool_factor),
    )


def build_bfgcn(nchannels, nclasses, kadj=2, num_out=16, att_hidden=16,
                classifier_hidden=32, avgpool=2, dropout=0.0):
    from .bfgcn import BFGCN

    return BFGCN(
        nclass=int(nclasses),
        xdim=[int(nchannels), len(BFGCN_FEATURE_BANDS)],
        kadj=int(kadj),
        num_out=int(num_out),
        att_hidden=int(att_hidden),
        att_plv_hidden=int(nchannels),
        classifier_hidden=int(classifier_hidden),
        avgpool=int(avgpool),
        dropout=float(dropout),
    )


def build_tahag(nchannels, nclasses, adj=None, dropout=0.25,
                adaptive=True, attention=True, hidden_dims=(32, 64, 128)):
    from .tahag import TAHAGModel

    return TAHAGModel(
        nchannels=int(nchannels),
        in_feats=len(BFGCN_FEATURE_BANDS),
        nclasses=int(nclasses),
        adj=adj,
        dropout=float(dropout),
        adaptive=bool(adaptive),
        attention=bool(attention),
        hidden_dims=hidden_dims,
    )


def build_lsccn(nchannels, nfeatures, nclasses, latent_dim=200,
                routing_iters=3, conv_filters=16, primary_filters=32,
                primary_dim=4, digit_dim=16):
    from .lsccn import LSCCN

    return LSCCN(
        nchannels=int(nchannels),
        nfeatures=int(nfeatures),
        nclasses=int(nclasses),
        latent_dim=int(latent_dim),
        conv_filters=int(conv_filters),
        primary_filters=int(primary_filters),
        primary_dim=int(primary_dim),
        digit_dim=int(digit_dim),
        routing_iters=int(routing_iters),
    )


def build_temporal_baseline(model_type, nchannels, nsamples, nclasses,
                            recurrent_hidden=64, recurrent_layers=1,
                            recurrent_dropout=0.5,
                            transformer_d_model=64, transformer_heads=4,
                            transformer_layers=2, transformer_ff=128,
                            transformer_dropout=0.2,
                            shallow_filters=40, shallow_kernel=25,
                            shallow_pool=25, shallow_dropout=0.5):
    from .temporal_baselines import LSTMClassifier, ShallowCNN, TransformerClassifier

    if model_type in ["lstm", "bilstm"]:
        return LSTMClassifier(
            nchannels=int(nchannels),
            nclasses=int(nclasses),
            hidden_size=int(recurrent_hidden),
            num_layers=int(recurrent_layers),
            dropout=float(recurrent_dropout),
            bidirectional=(model_type == "bilstm"),
        )
    if model_type == "transformer":
        return TransformerClassifier(
            nchannels=int(nchannels),
            nsamples=int(nsamples),
            nclasses=int(nclasses),
            d_model=int(transformer_d_model),
            num_heads=int(transformer_heads),
            num_layers=int(transformer_layers),
            dim_feedforward=int(transformer_ff),
            dropout=float(transformer_dropout),
        )
    if model_type == "shallowcnn":
        return ShallowCNN(
            nchannels=int(nchannels),
            nsamples=int(nsamples),
            nclasses=int(nclasses),
            filters=int(shallow_filters),
            temporal_kernel=int(shallow_kernel),
            pool_size=int(shallow_pool),
            dropout=float(shallow_dropout),
        )
    raise ValueError("Unknown temporal baseline: {}".format(model_type))


def make_optimizer(model, lr=1e-3, weight_decay=1e-4, model_type="tsmnet"):
    if model_type != "tsmnet":
        return torch.optim.Adam(model.parameters(), lr=float(lr),
                                weight_decay=float(weight_decay))
    try:
        import geoopt
        optim_cls = geoopt.optim.RiemannianAdam
    except Exception:
        optim_cls = torch.optim.Adam

    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "spdnet" in name and name.endswith("W"):
            no_decay.append(param)
        elif "mean" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": float(weight_decay)})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return optim_cls(groups, lr=float(lr))


def _forward_logits(model, xb, db):
    if getattr(model, "requires_domain", True):
        out = model(xb, db)
    else:
        out = model(xb)
    return out[0] if isinstance(out, (tuple, list)) else out


def evaluate(model, loader, device, return_predictions=False):
    model.eval()
    losses, y_true, y_pred, y_prob = [], [], [], []
    loss_fn = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for xb, yb, db in loader:
            xb = xb.to(device)
            db = db.to(device)
            logits = _forward_logits(model, xb, db)
            loss = loss_fn(logits, yb.to(logits.device))
            losses.append(float(loss.detach().cpu().item()))
            logits_cpu = logits.detach().cpu()
            pred = torch.argmax(logits_cpu, dim=1).numpy()
            prob = torch.softmax(logits_cpu, dim=1).numpy()
            y_pred.extend(pred.tolist())
            y_prob.extend(prob.tolist())
            y_true.extend(yb.numpy().tolist())
    metrics = _metrics_from_arrays(y_true, y_pred, y_prob)
    metrics.update({
        "loss": float(np.mean(losses)) if losses else np.nan,
    })
    if return_predictions:
        metrics["y_true"] = np.asarray(y_true, dtype=np.int64)
        metrics["y_pred"] = np.asarray(y_pred, dtype=np.int64)
        metrics["y_prob"] = np.asarray(y_prob, dtype=np.float32)
        metrics["indices"] = np.asarray(loader.dataset.indices, dtype=np.int64)
    return metrics


def _valid_band(band, fs):
    low, high = float(band[0]), float(band[1])
    nyq = float(fs) / 2.0
    high = min(high, nyq - 1e-4)
    return low, high


def _bfgcn_bandpower_features(windows, fs):
    windows = np.asarray(windows, dtype=np.float32)
    freqs = np.fft.rfftfreq(windows.shape[-1], d=1.0 / float(fs))
    spectrum = np.fft.rfft(windows, axis=-1)
    power = (np.abs(spectrum) ** 2).astype(np.float32)
    features = []
    for band in BFGCN_FEATURE_BANDS:
        low, high = _valid_band(band, fs)
        mask = (freqs >= low) & (freqs < high)
        if not np.any(mask):
            values = np.mean(power, axis=-1)
        else:
            values = np.mean(power[..., mask], axis=-1)
        features.append(np.log(values + 1e-6))
    return np.stack(features, axis=-1).astype(np.float32)


def _safe_bandpass_batch(windows, fs, band):
    low, high = _valid_band(band, fs)
    nyq = float(fs) / 2.0
    if low >= high:
        return windows
    sos = signal.butter(3, [low / nyq, high / nyq], btype="bandpass", output="sos")
    if windows.shape[-1] < 24:
        return signal.sosfilt(sos, windows, axis=-1)
    return signal.sosfiltfilt(sos, windows, axis=-1)


def _bfgcn_plv(windows, fs):
    windows = np.asarray(windows, dtype=np.float32)
    adjs = []
    for band in BFGCN_PLV_BANDS:
        filtered = _safe_bandpass_batch(windows, fs, band)
        analytic = signal.hilbert(filtered, axis=-1)
        phase = analytic / (np.abs(analytic) + 1e-8)
        plv = np.abs(np.einsum("nct,ndt->ncd", phase, np.conj(phase)) / phase.shape[-1])
        diag = np.arange(plv.shape[1])
        plv[:, diag, diag] = 1.0
        adjs.append(plv.astype(np.float32))
    return np.stack(adjs, axis=-1).astype(np.float32)


def _lsccn_fused_features(windows, fs):
    windows = np.asarray(windows, dtype=np.float32)
    power = _bfgcn_bandpower_features(windows, fs)
    power_min = np.min(power, axis=(1, 2), keepdims=True)
    power_max = np.max(power, axis=(1, 2), keepdims=True)
    power = (power - power_min) / (power_max - power_min + 1e-6)

    filtered = _safe_bandpass_batch(windows, fs, LSCCN_CONNECTIVITY_BAND)
    analytic = signal.hilbert(filtered, axis=-1)
    phase = analytic / (np.abs(analytic) + 1e-8)
    plv = np.abs(np.einsum("nct,ndt->ncd", phase, np.conj(phase)) / phase.shape[-1])
    diag = np.arange(plv.shape[1])
    plv[:, diag, diag] = 0.0
    return np.concatenate([plv.astype(np.float32), power.astype(np.float32)], axis=-1)


class BFGCNCollator:
    def __init__(self, fs):
        self.fs = float(fs)

    def __call__(self, batch):
        xs, ys, ds = zip(*batch)
        windows = torch.stack(xs).numpy().astype(np.float32)
        features = _bfgcn_bandpower_features(windows, self.fs)
        plv = _bfgcn_plv(windows, self.fs)
        return (
            torch.from_numpy(features),
            torch.from_numpy(plv),
            torch.stack(ys).long(),
            torch.stack(ds).long(),
        )


class TAHAGCollator:
    def __init__(self, fs):
        self.fs = float(fs)

    def __call__(self, batch):
        xs, ys, ds = zip(*batch)
        windows = torch.stack(xs).numpy().astype(np.float32)
        features = _bfgcn_bandpower_features(windows, self.fs)
        return (
            torch.from_numpy(features),
            torch.stack(ys).long(),
            torch.stack(ds).long(),
        )


class LSCCNCollator:
    def __init__(self, fs):
        self.fs = float(fs)

    def __call__(self, batch):
        xs, ys, ds = zip(*batch)
        windows = torch.stack(xs).numpy().astype(np.float32)
        features = _lsccn_fused_features(windows, self.fs)
        return (
            torch.from_numpy(features),
            torch.stack(ys).long(),
            torch.stack(ds).long(),
        )


def _bfgcn_forward_logits(model, batch, device, alpha=0.0):
    xb, adjb, yb, db = batch
    xb = xb.to(device)
    adjb = adjb.to(device)
    class_logits, domain_logits = model(xb, adjb, alpha=alpha)[:2]
    return class_logits, domain_logits, yb.to(device), db.to(device)


def _tahag_forward_logits(model, batch, device, target_batch=None, alpha=0.0):
    xb, yb, db = batch
    xb = xb.to(device)
    yb = yb.to(device)
    db = db.to(device)
    if target_batch is None:
        logits, domain_logits, mmd_loss = model(xb, None, alpha=alpha)
    else:
        xt = target_batch[0].to(device)
        logits, domain_logits, mmd_loss = model(xb, xt, alpha=alpha)
    return logits, domain_logits, mmd_loss, yb, db


def _lsccn_forward(model, batch, device):
    xb, yb, db = batch
    xb = xb.to(device)
    yb = yb.to(device)
    return model(xb), xb, yb, db.to(device)


def _lsccn_margin_loss(scores, labels, nclasses, m_plus=0.9, m_minus=0.1,
                       down_weight=0.5):
    target = torch.zeros(scores.shape[0], int(nclasses), device=scores.device)
    target.scatter_(1, labels[:, None], 1.0)
    positive = target * torch.clamp(float(m_plus) - scores, min=0.0).pow(2)
    negative = (1.0 - target) * torch.clamp(scores - float(m_minus), min=0.0).pow(2)
    return torch.mean(torch.sum(positive + float(down_weight) * negative, dim=1))


def _lsccn_loss(outputs, xb, yb, nclasses, recon_weight=1e-5,
                kl_weight=0.1):
    scores, recon, mu, logvar = outputs
    margin = _lsccn_margin_loss(scores, yb, nclasses)
    recon_loss = torch.nn.functional.binary_cross_entropy(
        recon.clamp(1e-6, 1.0 - 1e-6), xb.clamp(0.0, 1.0), reduction="mean"
    )
    kl_loss = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
    return margin + float(recon_weight) * (recon_loss + float(kl_weight) * kl_loss)


def evaluate_bfgcn(model, loader, device, return_predictions=False):
    model.eval()
    losses, y_true, y_pred, y_prob = [], [], [], []
    loss_fn = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in loader:
            logits, _, yb, _ = _bfgcn_forward_logits(model, batch, device, alpha=0.0)
            loss = loss_fn(logits, yb.to(logits.device))
            losses.append(float(loss.detach().cpu().item()))
            logits_cpu = logits.detach().cpu()
            pred = torch.argmax(logits_cpu, dim=1).numpy()
            prob = torch.softmax(logits_cpu, dim=1).numpy()
            y_pred.extend(pred.tolist())
            y_prob.extend(prob.tolist())
            y_true.extend(yb.detach().cpu().numpy().tolist())
    metrics = _metrics_from_arrays(y_true, y_pred, y_prob)
    metrics.update({"loss": float(np.mean(losses)) if losses else np.nan})
    if return_predictions:
        metrics["y_true"] = np.asarray(y_true, dtype=np.int64)
        metrics["y_pred"] = np.asarray(y_pred, dtype=np.int64)
        metrics["y_prob"] = np.asarray(y_prob, dtype=np.float32)
        metrics["indices"] = np.asarray(loader.dataset.indices, dtype=np.int64)
    return metrics


def evaluate_tahag(model, loader, device, return_predictions=False):
    model.eval()
    losses, y_true, y_pred, y_prob = [], [], [], []
    loss_fn = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in loader:
            logits, _, _, yb, _ = _tahag_forward_logits(model, batch, device, target_batch=None)
            loss = loss_fn(logits, yb.to(logits.device))
            losses.append(float(loss.detach().cpu().item()))
            logits_cpu = logits.detach().cpu()
            pred = torch.argmax(logits_cpu, dim=1).numpy()
            prob = torch.softmax(logits_cpu, dim=1).numpy()
            y_pred.extend(pred.tolist())
            y_prob.extend(prob.tolist())
            y_true.extend(yb.detach().cpu().numpy().tolist())
    metrics = _metrics_from_arrays(y_true, y_pred, y_prob)
    metrics.update({"loss": float(np.mean(losses)) if losses else np.nan})
    if return_predictions:
        metrics["y_true"] = np.asarray(y_true, dtype=np.int64)
        metrics["y_pred"] = np.asarray(y_pred, dtype=np.int64)
        metrics["y_prob"] = np.asarray(y_prob, dtype=np.float32)
        metrics["indices"] = np.asarray(loader.dataset.indices, dtype=np.int64)
    return metrics


def evaluate_lsccn(model, loader, device, nclasses, recon_weight=1e-5,
                   kl_weight=0.1, return_predictions=False):
    model.eval()
    losses, y_true, y_pred, y_prob = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            outputs, xb, yb, _ = _lsccn_forward(model, batch, device)
            scores = outputs[0]
            loss = _lsccn_loss(outputs, xb, yb, nclasses,
                               recon_weight=recon_weight,
                               kl_weight=kl_weight)
            losses.append(float(loss.detach().cpu().item()))
            scores_cpu = scores.detach().cpu()
            pred = torch.argmax(scores_cpu, dim=1).numpy()
            prob = torch.softmax(scores_cpu, dim=1).numpy()
            y_pred.extend(pred.tolist())
            y_prob.extend(prob.tolist())
            y_true.extend(yb.detach().cpu().numpy().tolist())
    metrics = _metrics_from_arrays(y_true, y_pred, y_prob)
    metrics.update({"loss": float(np.mean(losses)) if losses else np.nan})
    if return_predictions:
        metrics["y_true"] = np.asarray(y_true, dtype=np.int64)
        metrics["y_pred"] = np.asarray(y_pred, dtype=np.int64)
        metrics["y_prob"] = np.asarray(y_prob, dtype=np.float32)
        metrics["indices"] = np.asarray(loader.dataset.indices, dtype=np.int64)
    return metrics


def _svm_features(x, indices, normalizer):
    idx = np.asarray(indices, dtype=np.int64)
    values = normalizer.transform_array(np.asarray(x[idx], dtype=np.float32))
    return values.reshape(values.shape[0], -1)


def _align_class_columns(values, classes, labels):
    values = np.asarray(values, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    classes = np.asarray(classes, dtype=np.int64)
    if values.ndim == 1:
        aligned = np.zeros((values.shape[0], len(labels)), dtype=np.float32)
        if len(classes) == 2:
            first = np.flatnonzero(labels == classes[0])
            second = np.flatnonzero(labels == classes[1])
            if len(first):
                aligned[:, first[0]] = -values
            if len(second):
                aligned[:, second[0]] = values
        elif len(classes) == 1:
            col = np.flatnonzero(labels == classes[0])
            if len(col):
                aligned[:, col[0]] = values
        return aligned
    aligned = np.zeros((values.shape[0], len(labels)), dtype=np.float32)
    for col, cls in enumerate(classes):
        matches = np.flatnonzero(labels == int(cls))
        if len(matches):
            aligned[:, matches[0]] = values[:, col]
    return aligned


def _evaluate_svm(model, x, y, indices, normalizer, labels):
    idx = np.asarray(indices, dtype=np.int64)
    features = _svm_features(x, idx, normalizer)
    y_true = np.asarray(y[idx], dtype=np.int64)
    y_pred = model.predict(features)
    has_probability = hasattr(model, "predict_proba")
    try:
        raw_scores = model.predict_proba(features) if has_probability else model.decision_function(features)
        score_kind = "probability" if has_probability else "decision"
    except Exception:
        raw_scores = model.decision_function(features)
        score_kind = "decision"
    y_score = _align_class_columns(raw_scores, model.classes_, labels)
    metrics = _metrics_from_arrays(y_true, y_pred, y_score)
    try:
        if score_kind == "probability":
            metrics["loss"] = float(log_loss(y_true, y_score, labels=labels))
        else:
            metrics["loss"] = float(1.0 - metrics["balanced_accuracy"])
    except ValueError:
        metrics["loss"] = float(1.0 - metrics["balanced_accuracy"])
    return metrics


def _cycle_loader(loader):
    while True:
        for batch in loader:
            yield batch


def refit_batchnorm(model, x, y, domains, train_idx, test_idx, device,
                    target_adapt=True, normalizer=None):
    model.eval()
    with torch.no_grad():
        if hasattr(model, "domainadapt_finetune") and hasattr(model, "spddsbnorm"):
            idx = np.concatenate([train_idx, test_idx]) if target_adapt else train_idx
            xt = x[idx]
            if normalizer is not None:
                xt = normalizer.transform_array(xt)
            model.domainadapt_finetune(
                torch.from_numpy(xt).float().to(device),
                torch.from_numpy(y[idx]).long(),
                torch.from_numpy(domains[idx]).long().to(device),
                target_domains=np.unique(domains[test_idx]),
            )
        elif hasattr(model, "finetune"):
            xt = x[train_idx]
            if normalizer is not None:
                xt = normalizer.transform_array(xt)
            model.finetune(
                torch.from_numpy(xt).float().to(device),
                torch.from_numpy(y[train_idx]).long(),
                torch.from_numpy(domains[train_idx]).long().to(device),
            )


def train_one_split(dataset, domains, split, project_root, output_dir=None,
                    epochs=30, batch_size=64, lr=1e-3, weight_decay=1e-4,
                    bnorm="spddsbn", augment=False, patience=8,
                    model_type="tsmnet",
                    temporal_filters=4, spatial_filters=40, subspacedims=20,
                    temp_kernel=25, seed=42, target_adapt=True,
                    conformer_emb_size=40, conformer_depth=6,
                    conformer_num_heads=5, conformer_dropout=0.5,
                    conformer_classifier_hidden=256,
                    artifact_z=None, eegnet_temporal_filters=64,
                    eegnet_spatial_filters=4, eegnet_dropout=0.5,
                    eegnet_avgpool_factor=2, bfgcn_kadj=2,
                    bfgcn_num_out=16, bfgcn_att_hidden=16,
                    bfgcn_classifier_hidden=32, bfgcn_avgpool=2,
                    bfgcn_dropout=0.0, bfgcn_domain_weight=1.0,
                    svm_estimator="linear-svc", svm_kernel="rbf",
                    svm_c=1.0, svm_gamma="scale",
                    svm_class_weight="balanced", svm_probability=False,
                    svm_max_iter=5000, tahag_dropout=0.25,
                    tahag_domain_weight=1.0, tahag_mmd_weight=1.0,
                    tahag_adaptive=True, tahag_attention=True,
                    lsccn_latent_dim=200, lsccn_routing_iters=3,
                    lsccn_recon_weight=1e-5, lsccn_kl_weight=0.1,
                    recurrent_hidden=64, recurrent_layers=1,
                    recurrent_dropout=0.5,
                    transformer_d_model=64, transformer_heads=4,
                    transformer_layers=2, transformer_ff=128,
                    transformer_dropout=0.2,
                    shallow_filters=40, shallow_kernel=25,
                    shallow_pool=25, shallow_dropout=0.5):
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x, y = dataset["x"], dataset["y"]
    normalizer = fit_source_normalizer(x, split["train"])
    split = {
        "train": _filter_artifact_windows(x, split["train"], normalizer, artifact_z),
        "val": _filter_artifact_windows(x, split["val"], normalizer, artifact_z),
        "test": _filter_artifact_windows(x, split["test"], normalizer, artifact_z),
    }
    if len(split["train"]) == 0 or len(split["val"]) == 0 or len(split["test"]) == 0:
        raise RuntimeError("Artifact rejection removed an entire split: train={}, val={}, test={}".format(
            len(split["train"]), len(split["val"]), len(split["test"])))
    selected = np.concatenate([split["train"], split["val"], split["test"]])
    nclasses = int(np.max(y[selected]) + 1)
    class_labels = np.arange(nclasses, dtype=np.int64)
    if model_type == "svm":
        target_adapt = False
        class_weight = None if str(svm_class_weight).lower() in ["none", "null", ""] else svm_class_weight
        gamma = svm_gamma
        if str(gamma).lower() not in ["scale", "auto"]:
            gamma = float(gamma)
        train_features = _svm_features(x, split["train"], normalizer)
        train_labels = np.asarray(y[split["train"]], dtype=np.int64)
        if str(svm_estimator) == "linear-svc":
            model = LinearSVC(
                C=float(svm_c),
                class_weight=class_weight,
                dual=bool(train_features.shape[0] < train_features.shape[1]),
                max_iter=int(svm_max_iter),
                random_state=int(seed),
            )
        elif str(svm_estimator) == "svc":
            model = SVC(
                C=float(svm_c),
                kernel=str(svm_kernel),
                gamma=gamma,
                class_weight=class_weight,
                probability=bool(svm_probability),
                random_state=int(seed),
            )
        else:
            raise ValueError("Unknown svm_estimator: {}".format(svm_estimator))
        model.fit(train_features, train_labels)
        train_metrics = _evaluate_svm(model, x, y, split["train"], normalizer, class_labels)
        val_metrics = _evaluate_svm(model, x, y, split["val"], normalizer, class_labels)
        test_metrics = _evaluate_svm(model, x, y, split["test"], normalizer, class_labels)
        if output_dir:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            import joblib
            joblib.dump(model, os.path.join(output_dir, "model.joblib"))
        return {
            "history": [{
                "epoch": 1,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "val_bacc": val_metrics["balanced_accuracy"],
            }],
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
            "epochs_ran": 1,
            "best_epoch": 1,
            "best_val_loss": float(val_metrics["loss"]),
            "target_adapt": False,
            "n_train": int(len(split["train"])),
            "n_val": int(len(split["val"])),
            "n_test": int(len(split["test"])),
            "artifact_z": "" if artifact_z is None else float(artifact_z),
        }
    if model_type == "tsmnet":
        model = build_tsmnet(project_root, x.shape[1], x.shape[2], nclasses,
                             domains[selected], bnorm=bnorm,
                             temporal_filters=temporal_filters,
                             spatial_filters=spatial_filters,
                             subspacedims=subspacedims,
                             temp_kernel=temp_kernel,
                             device=device)
    elif model_type == "eegconformer":
        model = build_eegconformer(x.shape[1], x.shape[2], nclasses,
                                   temporal_kernel=temp_kernel,
                                   emb_size=conformer_emb_size,
                                   depth=conformer_depth,
                                   num_heads=conformer_num_heads,
                                   dropout=conformer_dropout,
                                   classifier_hidden=conformer_classifier_hidden).to(device)
        target_adapt = False
    elif model_type == "eegnet":
        model = build_eegnet(x.shape[1], x.shape[2], nclasses,
                             temporal_filters=eegnet_temporal_filters,
                             spatial_filters=eegnet_spatial_filters,
                             dropout=eegnet_dropout,
                             avgpool_factor=eegnet_avgpool_factor).to(device)
        target_adapt = False
    elif model_type == "bfgcn":
        model = build_bfgcn(x.shape[1], nclasses,
                            kadj=bfgcn_kadj,
                            num_out=bfgcn_num_out,
                            att_hidden=bfgcn_att_hidden,
                            classifier_hidden=bfgcn_classifier_hidden,
                            avgpool=bfgcn_avgpool,
                            dropout=bfgcn_dropout).to(device)
    elif model_type == "tahag":
        model = build_tahag(
            x.shape[1], nclasses,
            dropout=tahag_dropout,
            adaptive=tahag_adaptive,
            attention=tahag_attention,
        ).to(device)
    elif model_type == "lsccn":
        model = build_lsccn(
            x.shape[1], x.shape[1] + len(BFGCN_FEATURE_BANDS), nclasses,
            latent_dim=lsccn_latent_dim,
            routing_iters=lsccn_routing_iters,
        ).to(device)
        target_adapt = False
    elif model_type in ["lstm", "bilstm", "transformer", "shallowcnn"]:
        model = build_temporal_baseline(
            model_type,
            x.shape[1], x.shape[2], nclasses,
            recurrent_hidden=recurrent_hidden,
            recurrent_layers=recurrent_layers,
            recurrent_dropout=recurrent_dropout,
            transformer_d_model=transformer_d_model,
            transformer_heads=transformer_heads,
            transformer_layers=transformer_layers,
            transformer_ff=transformer_ff,
            transformer_dropout=transformer_dropout,
            shallow_filters=shallow_filters,
            shallow_kernel=shallow_kernel,
            shallow_pool=shallow_pool,
            shallow_dropout=shallow_dropout,
        ).to(device)
        target_adapt = False
    else:
        raise ValueError("Unknown model_type: {}".format(model_type))
    optimizer = make_optimizer(model, lr=lr, weight_decay=weight_decay,
                               model_type=model_type)
    loss_fn = torch.nn.CrossEntropyLoss()

    train_ds = EEGWindowDataset(x, y, domains, split["train"], augment=augment,
                                normalizer=normalizer)
    train_eval_ds = EEGWindowDataset(x, y, domains, split["train"], augment=False,
                                     normalizer=normalizer)
    val_ds = EEGWindowDataset(x, y, domains, split["val"], augment=False,
                              normalizer=normalizer)
    test_ds = EEGWindowDataset(x, y, domains, split["test"], augment=False,
                               normalizer=normalizer)
    if model_type in ["bfgcn", "tahag"]:
        if model_type == "bfgcn":
            collate = BFGCNCollator(dataset["fs"])
        else:
            collate = TAHAGCollator(dataset["fs"])
        target_domain_ds = EEGWindowDataset(
            x, y, domains, split["test"], augment=False, normalizer=normalizer
        )
        domain_batch_size = int(min(int(batch_size), len(train_ds), len(target_domain_ds)))
        domain_batch_size = max(1, domain_batch_size)
        drop_source = len(train_ds) > domain_batch_size
        drop_target = len(target_domain_ds) > domain_batch_size
        train_loader = DataLoader(
            train_ds, batch_size=domain_batch_size, shuffle=True,
            drop_last=drop_source, collate_fn=collate,
        )
        target_domain_loader = DataLoader(
            target_domain_ds, batch_size=domain_batch_size, shuffle=True,
            drop_last=drop_target, collate_fn=collate,
        )
        train_eval_loader = DataLoader(
            train_eval_ds, batch_size=batch_size, shuffle=False, collate_fn=collate
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate
        )
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate
        )
    elif model_type == "lsccn":
        collate = LSCCNCollator(dataset["fs"])
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, drop_last=False,
            collate_fn=collate,
        )
        train_eval_loader = DataLoader(
            train_eval_ds, batch_size=batch_size, shuffle=False, collate_fn=collate
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate
        )
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate
        )
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
        train_eval_loader = DataLoader(train_eval_ds, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    best_state, best_loss, best_epoch, bad_epochs = None, float("inf"), None, 0
    history = []
    target_iter = _cycle_loader(target_domain_loader) if model_type in ["bfgcn", "tahag"] else None
    domain_loss_fn = torch.nn.NLLLoss()
    for epoch in range(1, int(epochs) + 1):
        model.train()
        batch_losses = []
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            if model_type == "bfgcn":
                progress = (float(epoch - 1) + float(step) / max(1, len(train_loader))) / max(1, int(epochs))
                alpha = float(2.0 / (1.0 + np.exp(-10.0 * progress)) - 1.0) if target_adapt else 0.0
                logits, domain_src, yb, _ = _bfgcn_forward_logits(
                    model, batch, device, alpha=alpha
                )
                class_loss = loss_fn(logits, yb.to(logits.device))
                loss = class_loss
                if target_adapt:
                    target_batch = next(target_iter)
                    _, domain_tgt, _, _ = _bfgcn_forward_logits(
                        model, target_batch, device, alpha=alpha
                    )
                    domain_logits = torch.cat([domain_src, domain_tgt], dim=0)
                    domain_labels = torch.cat([
                        torch.zeros(domain_src.shape[0], dtype=torch.long, device=device),
                        torch.ones(domain_tgt.shape[0], dtype=torch.long, device=device),
                    ])
                    domain_loss = domain_loss_fn(domain_logits, domain_labels)
                    loss = class_loss + float(bfgcn_domain_weight) * domain_loss
            elif model_type == "tahag":
                progress = (float(epoch - 1) + float(step) / max(1, len(train_loader))) / max(1, int(epochs))
                alpha = float(2.0 / (1.0 + np.exp(-10.0 * progress)) - 1.0) if target_adapt else 0.0
                target_batch = next(target_iter) if target_adapt else None
                logits, domain_logits, mmd_loss, yb, _ = _tahag_forward_logits(
                    model, batch, device, target_batch=target_batch, alpha=alpha
                )
                class_loss = loss_fn(logits, yb.to(logits.device))
                loss = class_loss
                if target_adapt:
                    target_y = target_batch[1].to(device)
                    domain_labels = torch.cat([
                        torch.zeros(yb.shape[0], dtype=torch.long, device=device),
                        torch.ones(target_y.shape[0], dtype=torch.long, device=device),
                    ])
                    domain_loss = loss_fn(domain_logits, domain_labels)
                    loss = (class_loss
                            + float(tahag_domain_weight) * domain_loss
                            + float(tahag_mmd_weight) * alpha * mmd_loss)
            elif model_type == "lsccn":
                outputs, xb, yb, _ = _lsccn_forward(model, batch, device)
                loss = _lsccn_loss(
                    outputs, xb, yb, nclasses,
                    recon_weight=lsccn_recon_weight,
                    kl_weight=lsccn_kl_weight,
                )
            else:
                xb, yb, db = batch
                xb = xb.to(device)
                yb = yb.to(device)
                db = db.to(device)
                logits = _forward_logits(model, xb, db)
                loss = loss_fn(logits, yb.to(logits.device))
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu().item()))
        if model_type == "bfgcn":
            val_metrics = evaluate_bfgcn(model, val_loader, device)
        elif model_type == "tahag":
            val_metrics = evaluate_tahag(model, val_loader, device)
        elif model_type == "lsccn":
            val_metrics = evaluate_lsccn(
                model, val_loader, device, nclasses,
                recon_weight=lsccn_recon_weight,
                kl_weight=lsccn_kl_weight,
            )
        else:
            val_metrics = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": float(np.mean(batch_losses)),
               "val_loss": val_metrics["loss"],
               "val_bacc": val_metrics["balanced_accuracy"]}
        history.append(row)
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= int(patience):
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    if model_type == "tsmnet":
        refit_batchnorm(model, x, y, domains, split["train"], split["test"], device,
                        target_adapt=target_adapt, normalizer=normalizer)
    if model_type == "bfgcn":
        train_metrics = evaluate_bfgcn(model, train_eval_loader, device)
        val_metrics = evaluate_bfgcn(model, val_loader, device)
        test_metrics = evaluate_bfgcn(model, test_loader, device)
    elif model_type == "tahag":
        train_metrics = evaluate_tahag(model, train_eval_loader, device)
        val_metrics = evaluate_tahag(model, val_loader, device)
        test_metrics = evaluate_tahag(model, test_loader, device)
    elif model_type == "lsccn":
        train_metrics = evaluate_lsccn(
            model, train_eval_loader, device, nclasses,
            recon_weight=lsccn_recon_weight,
            kl_weight=lsccn_kl_weight,
        )
        val_metrics = evaluate_lsccn(
            model, val_loader, device, nclasses,
            recon_weight=lsccn_recon_weight,
            kl_weight=lsccn_kl_weight,
        )
        test_metrics = evaluate_lsccn(
            model, test_loader, device, nclasses,
            recon_weight=lsccn_recon_weight,
            kl_weight=lsccn_kl_weight,
        )
    else:
        train_metrics = evaluate(model, train_eval_loader, device)
        val_metrics = evaluate(model, val_loader, device)
        test_metrics = evaluate(model, test_loader, device)

    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))

    return {
        "history": history,
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "epochs_ran": len(history),
        "best_epoch": int(best_epoch) if best_epoch is not None else len(history),
        "best_val_loss": float(best_loss),
        "target_adapt": bool(target_adapt),
        "n_train": int(len(split["train"])),
        "n_val": int(len(split["val"])),
        "n_test": int(len(split["test"])),
        "artifact_z": "" if artifact_z is None else float(artifact_z),
    }
