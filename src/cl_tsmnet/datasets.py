import glob
import os
import re
import shutil
import tempfile
import zipfile

import numpy as np
import pandas as pd

from .preprocessing import make_windows, preprocess_eeg, useful_eeg_channels


STEW_CHANNELS = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
                 "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
EEGMAT_DIRNAME = "eeg-during-mental-arithmetic-tasks-1.0.0"
COG_TASKS = {
    "nback": [("zeroBACK", 0), ("oneBACK", 1), ("twoBACK", 2)],
    "matb": [("MATBeasy", 0), ("MATBmed", 1), ("MATBdiff", 2)],
}
DEFAULT_TARGET_FS = {
    "stew": 128.0,
    "eegmat": 250.0,
    "cog-bci": 250.0,
}


def _append(all_x, all_rows, x, rows):
    if len(rows):
        all_x.append(x)
        all_rows.extend(rows)


def _pack_dataset(name, all_x, all_rows, channels, fs, label_names):
    if not all_x:
        raise RuntimeError("No windows were produced for dataset {}".format(name))
    x = np.concatenate(all_x, axis=0).astype(np.float32)
    columns = ["label", "subject", "session", "paradigm", "task", "start_sample"]
    meta = pd.DataFrame(all_rows, columns=columns)
    return {
        "name": name,
        "x": x,
        "y": meta["label"].values.astype(np.int64),
        "meta": meta,
        "channels": list(channels),
        "fs": float(fs),
        "label_names": dict(label_names),
    }


def _subject_allowed(subject, subjects):
    return subjects is None or int(subject) in set(int(s) for s in subjects)


def _crop_eegmat_segment(data, fs, rec):
    crop_samples = int(round(60.0 * float(fs)))
    if data.shape[-1] <= crop_samples:
        return data
    if int(rec) == 1:
        return data[:, -crop_samples:]
    return data[:, :crop_samples]


def load_stew(data_root, target_fs=128.0, window_sec=1.0, stride_sec=1.0,
              reject_z=8.0, subjects=None, sessions=None):
    root = os.path.join(data_root, "STEW Dataset")
    all_x, all_rows = [], []
    for path in sorted(glob.glob(os.path.join(root, "sub*_*.txt"))):
        match = re.search(r"sub(\d+)_(lo|hi)\.txt$", os.path.basename(path))
        if match is None:
            continue
        subject = int(match.group(1))
        if not _subject_allowed(subject, subjects):
            continue
        task = match.group(2)
        label = 0 if task == "lo" else 1
        data = np.loadtxt(path).T
        data, fs = preprocess_eeg(data, fs=128.0, target_fs=target_fs)
        x, rows = make_windows(data, fs, label, subject, 1, "stew", task,
                               window_sec, stride_sec, reject_z)
        _append(all_x, all_rows, x, rows)
    return _pack_dataset("stew", all_x, all_rows, STEW_CHANNELS, target_fs,
                         {0: "low workload", 1: "high workload"})


def load_eegmat(data_root, target_fs=250.0, window_sec=1.0, stride_sec=1.0,
                reject_z=8.0, subjects=None, sessions=None):
    import mne

    root = os.path.join(data_root, EEGMAT_DIRNAME)
    all_x, all_rows = [], []
    channels = None
    label_names = {0: "resting/background EEG", 1: "mental arithmetic"}
    for path in sorted(glob.glob(os.path.join(root, "Subject*_*.edf"))):
        match = re.search(r"Subject(\d+)_(1|2)\.edf$", os.path.basename(path))
        if match is None:
            continue
        subject = int(match.group(1))
        if not _subject_allowed(subject, subjects):
            continue
        rec = int(match.group(2))
        label = 0 if rec == 1 else 1
        task = "baseline" if rec == 1 else "arithmetic"
        raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
        picks, picked_names = useful_eeg_channels(raw.ch_names)
        channels = picked_names
        raw_fs = float(raw.info["sfreq"])
        data = raw.get_data(picks=picks)
        data = _crop_eegmat_segment(data, raw_fs, rec)
        data, fs = preprocess_eeg(data, fs=raw_fs,
                                  target_fs=target_fs)
        x, rows = make_windows(data, fs, label, subject, 1, "eegmat", task,
                               window_sec, stride_sec, reject_z)
        _append(all_x, all_rows, x, rows)
    return _pack_dataset("eegmat", all_x, all_rows, channels, target_fs,
                         label_names)


def _extract_eeglab_pair(zip_file, set_name, workdir):
    fdt_name = set_name[:-4] + ".fdt"
    zip_file.extract(set_name, workdir)
    zip_file.extract(fdt_name, workdir)
    return os.path.join(workdir, set_name.replace("/", os.sep))


def load_cog_bci(data_root, paradigm="nback", sessions=(1, 2, 3),
                 target_fs=250.0, window_sec=1.0, stride_sec=1.0,
                 reject_z=8.0, subjects=None):
    import mne

    if paradigm not in COG_TASKS:
        raise ValueError("paradigm must be one of {}".format(sorted(COG_TASKS)))

    root = os.path.join(data_root, "COG-BCI")
    all_x, all_rows = [], []
    channels = None
    label_names = ({0: "0-back", 1: "1-back", 2: "2-back"}
                   if paradigm == "nback"
                   else {0: "MAT-B easy", 1: "MAT-B medium", 2: "MAT-B difficult"})

    for zip_path in sorted(glob.glob(os.path.join(root, "sub-*.zip"))):
        sub_match = re.search(r"sub-(\d+)\.zip$", os.path.basename(zip_path))
        if sub_match is None:
            continue
        subject = int(sub_match.group(1))
        if not _subject_allowed(subject, subjects):
            continue
        with zipfile.ZipFile(zip_path) as zf:
            for session in sessions:
                ses_name = "ses-S{}".format(int(session))
                for task, label in COG_TASKS[paradigm]:
                    set_name = "sub-{0:02d}/{1}/eeg/{2}.set".format(subject, ses_name, task)
                    if set_name not in zf.namelist():
                        continue
                    tmpdir = tempfile.mkdtemp(prefix="cog_bci_")
                    try:
                        set_path = _extract_eeglab_pair(zf, set_name, tmpdir)
                        raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose="ERROR")
                        picks, picked_names = useful_eeg_channels(raw.ch_names)
                        channels = picked_names
                        data = raw.get_data(picks=picks)
                        data, fs = preprocess_eeg(data, fs=float(raw.info["sfreq"]),
                                                  target_fs=target_fs)
                        x, rows = make_windows(data, fs, label, subject, session,
                                               paradigm, task, window_sec,
                                               stride_sec, reject_z)
                        _append(all_x, all_rows, x, rows)
                    finally:
                        shutil.rmtree(tmpdir, ignore_errors=True)
    return _pack_dataset("cog-bci-{}".format(paradigm), all_x, all_rows,
                         channels, target_fs, label_names)


def save_npz(dataset, path):
    dirname = os.path.dirname(path)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname)
    np.savez_compressed(
        path,
        x=dataset["x"],
        y=dataset["y"],
        label=dataset["meta"]["label"].values,
        subject=dataset["meta"]["subject"].values,
        session=dataset["meta"]["session"].values,
        paradigm=dataset["meta"]["paradigm"].values.astype("U32"),
        task=dataset["meta"]["task"].values.astype("U32"),
        start_sample=dataset["meta"]["start_sample"].values,
        channels=np.asarray(dataset["channels"]).astype("U32"),
        fs=np.asarray([dataset["fs"]], dtype=np.float32),
        name=np.asarray([dataset["name"]]).astype("U64"),
    )


def load_npz(path):
    data = np.load(path, allow_pickle=False)
    meta = pd.DataFrame({
        "label": data["label"].astype(np.int64),
        "subject": data["subject"].astype(np.int64),
        "session": data["session"].astype(np.int64),
        "paradigm": data["paradigm"].astype(str),
        "task": data["task"].astype(str),
        "start_sample": data["start_sample"].astype(np.int64),
    })
    return {
        "name": str(data["name"][0]),
        "x": data["x"].astype(np.float32),
        "y": data["y"].astype(np.int64),
        "meta": meta,
        "channels": [str(c) for c in data["channels"]],
        "fs": float(data["fs"][0]),
        "label_names": {},
    }


def load_dataset(name, data_root="data", cache=None, rebuild_cache=False,
                 cog_paradigm="nback", **kwargs):
    if cache and os.path.exists(cache) and not rebuild_cache:
        return load_npz(cache)
    if kwargs.get("target_fs") is None:
        kwargs["target_fs"] = DEFAULT_TARGET_FS[name]
    if name == "stew":
        dataset = load_stew(data_root, **kwargs)
    elif name == "eegmat":
        dataset = load_eegmat(data_root, **kwargs)
    elif name == "cog-bci":
        dataset = load_cog_bci(data_root, paradigm=cog_paradigm, **kwargs)
    else:
        raise ValueError("Unknown dataset: {}".format(name))
    if cache:
        save_npz(dataset, cache)
    return dataset
