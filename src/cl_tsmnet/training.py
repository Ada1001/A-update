import copy
import os
import sys

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset


class EEGWindowDataset(Dataset):
    def __init__(self, x, y, d, indices, augment=False, noise_std=0.03,
                 shift_samples=8, channel_dropout=0.05):
        self.x = x
        self.y = y
        self.d = d
        self.indices = np.asarray(indices, dtype=np.int64)
        self.augment = bool(augment)
        self.noise_std = float(noise_std)
        self.shift_samples = int(shift_samples)
        self.channel_dropout = float(channel_dropout)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        idx = self.indices[item]
        x = np.array(self.x[idx], copy=True)
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


def make_optimizer(model, lr=1e-3, weight_decay=1e-4):
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
    out = model(xb, db)
    return out[0] if isinstance(out, (tuple, list)) else out


def evaluate(model, loader, device):
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
    auc = np.nan
    if y_true:
        y_true_arr = np.asarray(y_true)
        y_prob_arr = np.asarray(y_prob)
        try:
            if y_prob_arr.shape[1] == 2:
                auc = roc_auc_score(y_true_arr, y_prob_arr[:, 1])
            else:
                auc = roc_auc_score(y_true_arr, y_prob_arr, multi_class="ovr",
                                    average="macro")
        except ValueError:
            auc = np.nan
    return {
        "loss": float(np.mean(losses)) if losses else np.nan,
        "accuracy": accuracy_score(y_true, y_pred) if y_true else np.nan,
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred) if y_true else np.nan,
        "f1": f1_score(y_true, y_pred, average="macro") if y_true else np.nan,
        "auc": auc,
    }


def refit_batchnorm(model, x, y, domains, train_idx, test_idx, device,
                    target_adapt=True):
    xt = torch.from_numpy(x).float()
    yt = torch.from_numpy(y).long()
    dt = torch.from_numpy(domains).long()
    model.eval()
    with torch.no_grad():
        if hasattr(model, "domainadapt_finetune") and hasattr(model, "spddsbnorm"):
            idx = np.concatenate([train_idx, test_idx]) if target_adapt else train_idx
            model.domainadapt_finetune(xt[idx].to(device), yt[idx], dt[idx].to(device),
                                       target_domains=np.unique(domains[test_idx]))
        elif hasattr(model, "finetune"):
            model.finetune(xt[train_idx].to(device), yt[train_idx],
                           dt[train_idx].to(device))


def train_one_split(dataset, domains, split, project_root, output_dir=None,
                    epochs=30, batch_size=64, lr=1e-3, weight_decay=1e-4,
                    bnorm="spddsbn", augment=False, patience=8,
                    temporal_filters=4, spatial_filters=40, subspacedims=20,
                    temp_kernel=25, seed=42, target_adapt=True):
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x, y = dataset["x"], dataset["y"]
    selected = np.concatenate([split["train"], split["val"], split["test"]])
    nclasses = int(len(np.unique(y[selected])))
    model = build_tsmnet(project_root, x.shape[1], x.shape[2], nclasses,
                         domains[selected], bnorm=bnorm,
                         temporal_filters=temporal_filters,
                         spatial_filters=spatial_filters,
                         subspacedims=subspacedims,
                         temp_kernel=temp_kernel,
                         device=device)
    optimizer = make_optimizer(model, lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()

    train_ds = EEGWindowDataset(x, y, domains, split["train"], augment=augment)
    val_ds = EEGWindowDataset(x, y, domains, split["val"], augment=False)
    test_ds = EEGWindowDataset(x, y, domains, split["test"], augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    best_state, best_loss, best_epoch, bad_epochs = None, float("inf"), None, 0
    history = []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        batch_losses = []
        for xb, yb, db in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            db = db.to(device)
            optimizer.zero_grad()
            logits = _forward_logits(model, xb, db)
            loss = loss_fn(logits, yb.to(logits.device))
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu().item()))
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

    refit_batchnorm(model, x, y, domains, split["train"], split["test"], device,
                    target_adapt=target_adapt)
    train_metrics = evaluate(model, train_loader, device)
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
    }
