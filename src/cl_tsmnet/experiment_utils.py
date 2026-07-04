import os
from datetime import datetime

import numpy as np
import pandas as pd

from .datasets import DEFAULT_TARGET_FS


def dataset_tag(dataset, cog_paradigm="nback"):
    if dataset == "cog-bci":
        return "cog_{}".format(cog_paradigm)
    return dataset


def default_target_fs(dataset, target_fs=None):
    if target_fs is not None:
        return float(target_fs)
    return float(DEFAULT_TARGET_FS[dataset])


def format_fs(fs):
    value = float(fs)
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return str(value).replace(".", "p")


def subject_scope(subject, protocol):
    if subject is None or protocol == "loso":
        return "all"
    return "sub{:02d}".format(int(subject))


def session_scope(dataset, protocol):
    if dataset != "cog-bci":
        return "s1"
    if protocol == "single_session":
        return "s1"
    return "s123"


def default_cache_path(dataset, protocol, cog_paradigm="nback", subject=None,
                       target_fs=None, cache_root=os.path.join("outputs", "cache")):
    fs = default_target_fs(dataset, target_fs)
    name = "{}_{}_{}_{}_{}hz_1s.npz".format(
        dataset_tag(dataset, cog_paradigm),
        protocol,
        subject_scope(subject, protocol),
        session_scope(dataset, protocol),
        format_fs(fs),
    )
    return os.path.join(cache_root, name)


def run_tag(model, bnorm):
    return bnorm if model == "tsmnet" else "eegconformer"


def run_directory_name(dataset_name, protocol, model, bnorm):
    return "{}_{}_{}".format(dataset_name, protocol, run_tag(model, bnorm))


def metric_mean_std(rows, column):
    values = np.asarray([row.get(column, np.nan) for row in rows], dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return mean, std


def append_master_summary(rows, path, run_info):
    if not rows:
        return
    acc_mean, acc_std = metric_mean_std(rows, "test_acc")
    bacc_mean, bacc_std = metric_mean_std(rows, "test_bacc")
    f1_mean, f1_std = metric_mean_std(rows, "test_f1")
    auc_mean, auc_std = metric_mean_std(rows, "test_auc")
    train_mean, train_std = metric_mean_std(rows, "train_acc")
    val_mean, val_std = metric_mean_std(rows, "val_acc")

    first = rows[0]
    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": first["dataset"],
        "model": first["model"],
        "model_type": first["model_type"],
        "protocol": first["protocol"],
        "n": int(len(rows)),
        "accuracy_mean": acc_mean,
        "accuracy_std": acc_std,
        "balanced_accuracy_mean": bacc_mean,
        "balanced_accuracy_std": bacc_std,
        "f1_mean": f1_mean,
        "f1_std": f1_std,
        "auc_mean": auc_mean,
        "auc_std": auc_std,
        "train_accuracy_mean": train_mean,
        "train_accuracy_std": train_std,
        "val_accuracy_mean": val_mean,
        "val_accuracy_std": val_std,
    }
    record.update(run_info)

    dirname = os.path.dirname(path)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname)
    frame = pd.DataFrame([record])
    if os.path.exists(path):
        old = pd.read_csv(path)
        for col in frame.columns:
            if col not in old.columns:
                old[col] = ""
        for col in old.columns:
            if col not in frame.columns:
                frame[col] = ""
        frame = frame[old.columns]
        pd.concat([old, frame], ignore_index=True).to_csv(path, index=False)
    else:
        frame.to_csv(path, index=False)
