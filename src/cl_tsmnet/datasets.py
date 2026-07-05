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
        "preprocess_standardized": False,
    }


def _subject_allowed(subject, subjects):
    return subjects is None or int(subject) in set(int(s) for s in subjects)


def discover_cog_bci_zip_subjects(data_root):
    root = os.path.join(data_root, "COG-BCI")
    if not os.path.isdir(root):
        return []
    subjects = []
    for fname in os.listdir(root):
        match = re.search(r"sub-(\d+)\.zip$", fname, flags=re.IGNORECASE)
        if match is not None:
            subjects.append(int(match.group(1)))
    return sorted(set(subjects))


def discover_cog_bci_recording_subjects(data_root, paradigm="nback", sessions=(1, 2, 3)):
    if paradigm not in COG_TASKS:
        raise ValueError("paradigm must be one of {}".format(sorted(COG_TASKS)))
    subjects = []
    for subject, zip_path in _discover_cog_bci_zip_paths(data_root):
        with zipfile.ZipFile(zip_path) as zf:
            archive_names = {name.replace("\\", "/"): name for name in zf.namelist()}
            found = False
            for session in sessions:
                for task, _ in COG_TASKS[paradigm]:
                    if _find_cog_eeg_pair(archive_names, session, task) is not None:
                        subjects.append(subject)
                        found = True
                        break
                if found:
                    break
    return sorted(set(subjects))


def _discover_cog_bci_zip_paths(data_root):
    root = os.path.join(data_root, "COG-BCI")
    paths = []
    if not os.path.isdir(root):
        return paths
    for fname in os.listdir(root):
        match = re.search(r"sub-(\d+)\.zip$", fname, flags=re.IGNORECASE)
        if match is not None:
            paths.append((int(match.group(1)), os.path.join(root, fname)))
    return sorted(paths, key=lambda item: item[0])


def _find_cog_eeg_pair(archive_names, session, task):
    suffix = "/ses-S{}/eeg/{}.set".format(int(session), task)
    candidates = sorted(name for name in archive_names if name.endswith(suffix))
    if not candidates:
        return None
    set_name = candidates[0]
    fdt_name = set_name[:-4] + ".fdt"
    if fdt_name not in archive_names:
        return None
    return archive_names[set_name], archive_names[fdt_name]


def _crop_eegmat_segment(data, fs, rec):
    crop_samples = int(round(60.0 * float(fs)))
    if data.shape[-1] <= crop_samples:
        return data
    if int(rec) == 1:
        return data[:, -crop_samples:]
    return data[:, :crop_samples]


def load_stew(data_root, target_fs=128.0, window_sec=1.0, stride_sec=1.0,
              reject_z=None, subjects=None, sessions=None):
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
                reject_z=None, subjects=None, sessions=None):
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


def _extract_eeglab_pair(zip_file, set_arcname, fdt_arcname, workdir):
    zip_file.extract(set_arcname, workdir)
    zip_file.extract(fdt_arcname, workdir)
    return os.path.join(workdir, set_arcname.replace("/", os.sep))


def load_cog_bci(data_root, paradigm="nback", sessions=(1, 2, 3),
                 target_fs=250.0, window_sec=1.0, stride_sec=1.0,
                 reject_z=None, subjects=None):
    import mne

    if paradigm not in COG_TASKS:
        raise ValueError("paradigm must be one of {}".format(sorted(COG_TASKS)))

    all_x, all_rows = [], []
    channels = None
    label_names = ({0: "0-back", 1: "1-back", 2: "2-back"}
                   if paradigm == "nback"
                   else {0: "MAT-B easy", 1: "MAT-B medium", 2: "MAT-B difficult"})

    for subject, zip_path in _discover_cog_bci_zip_paths(data_root):
        if not _subject_allowed(subject, subjects):
            continue
        with zipfile.ZipFile(zip_path) as zf:
            archive_names = {name.replace("\\", "/"): name for name in zf.namelist()}
            for session in sessions:
                for task, label in COG_TASKS[paradigm]:
                    eeg_pair = _find_cog_eeg_pair(archive_names, session, task)
                    if eeg_pair is None:
                        continue
                    tmpdir = tempfile.mkdtemp(prefix="cog_bci_")
                    try:
                        set_path = _extract_eeglab_pair(zf, eeg_pair[0], eeg_pair[1], tmpdir)
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
        preprocess_standardized=np.asarray(
            [bool(dataset.get("preprocess_standardized", False))], dtype=np.bool_
        ),
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
    preprocess_standardized = None
    if "preprocess_standardized" in data.files:
        preprocess_standardized = bool(data["preprocess_standardized"][0])
    return {
        "name": str(data["name"][0]),
        "x": data["x"].astype(np.float32),
        "y": data["y"].astype(np.int64),
        "meta": meta,
        "channels": [str(c) for c in data["channels"]],
        "fs": float(data["fs"][0]),
        "label_names": {},
        "preprocess_standardized": preprocess_standardized,
    }


def _validate_cache_scope(dataset, name, data_root, subjects=None, sessions=None,
                          cog_paradigm="nback"):
    meta = dataset["meta"]
    cached_subjects = set(int(s) for s in np.unique(meta["subject"].values))
    if subjects is not None:
        requested_subjects = set(int(s) for s in subjects)
        missing = sorted(requested_subjects - cached_subjects)
        if missing:
            raise ValueError(
                "Cache subject-scope mismatch: cache has subjects {}, but requested "
                "subjects {}. Missing {}. Rebuild this cache or use the matching "
                "subject-specific cache.".format(
                    sorted(cached_subjects), sorted(requested_subjects), missing
                )
            )
    elif name == "cog-bci":
        available = set(discover_cog_bci_recording_subjects(
            data_root, paradigm=cog_paradigm, sessions=sessions or (1, 2, 3)
        ))
        missing = sorted(available - cached_subjects)
        if missing:
            preview = missing[:10]
            suffix = "..." if len(missing) > len(preview) else ""
            raise ValueError(
                "Cache subject-scope mismatch: COG-BCI data root contains {} subjects "
                "with requested {} recordings, but cache contains only {} subjects. "
                "Missing subjects: {}{}. "
                "Rebuild the cache with --rebuild-cache or remove the stale cache.".format(
                    len(available), cog_paradigm, len(cached_subjects), preview, suffix
                )
            )
    if sessions is not None:
        requested_sessions = set(int(s) for s in sessions)
        cached_sessions = set(int(s) for s in np.unique(meta["session"].values))
        missing_sessions = sorted(requested_sessions - cached_sessions)
        if missing_sessions:
            raise ValueError(
                "Cache session-scope mismatch: requested sessions {}, but cache has "
                "sessions {}. Missing {}. Rebuild this cache for the requested protocol.".format(
                    sorted(requested_sessions), sorted(cached_sessions), missing_sessions
                )
            )


def load_dataset(name, data_root="data", cache=None, rebuild_cache=False,
                 cog_paradigm="nback", **kwargs):
    if cache and os.path.exists(cache) and not rebuild_cache:
        dataset = load_npz(cache)
        expected_name = "cog-bci-{}".format(cog_paradigm) if name == "cog-bci" else name
        if dataset["name"] != expected_name:
            raise ValueError("Cache dataset mismatch: expected {}, found {} in {}".format(
                expected_name, dataset["name"], cache))
        requested_fs = kwargs.get("target_fs")
        if requested_fs is not None and abs(float(dataset["fs"]) - float(requested_fs)) > 1e-6:
            raise ValueError("Cache sampling-rate mismatch: requested {} Hz, found {} Hz in {}. "
                             "Use the matching cache or pass --rebuild-cache.".format(
                                 requested_fs, dataset["fs"], cache))
        if dataset.get("preprocess_standardized") is not False:
            raise ValueError(
                "Cache {} was produced by an older preprocessing pipeline or contains "
                "record-level standardization. Rebuild it with --rebuild-cache for the "
                "strict source-only normalization protocol.".format(cache)
            )
        _validate_cache_scope(
            dataset, name, data_root,
            subjects=kwargs.get("subjects"), sessions=kwargs.get("sessions"),
            cog_paradigm=cog_paradigm,
        )
        return dataset
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
