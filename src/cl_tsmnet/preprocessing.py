import math

import numpy as np
from scipy import signal


def canonical_channel_name(name):
    """Normalize channel names across the three datasets."""
    ch = str(name).strip()
    if ch.upper().startswith("EEG "):
        ch = ch[4:].strip()
    aliases = {
        "T3": "T7",
        "T4": "T8",
        "T5": "P7",
        "T6": "P8",
    }
    return aliases.get(ch, ch)


def useful_eeg_channels(ch_names):
    """Return indices and canonical names for scalp EEG channels only."""
    picks = []
    names = []
    reject = set(["ECG", "ECG1", "ECG ECG", "A2-A1", "NAS", "LHJ", "RHJ"])
    for idx, ch in enumerate(ch_names):
        clean = canonical_channel_name(ch)
        if clean.upper() in reject:
            continue
        if clean.upper().startswith("ECG"):
            continue
        picks.append(idx)
        names.append(clean)
    return picks, names


def _safe_sosfiltfilt(sos, data):
    if data.shape[-1] < 16:
        return data
    return signal.sosfiltfilt(sos, data, axis=-1)


def preprocess_eeg(data, fs, target_fs=128.0, band=(1.0, 45.0), notch=50.0):
    """Band-pass, notch, and resample EEG without fitting data statistics.

    Input and output shapes are channels x samples. Split-dependent robust
    normalization is fitted later from the source training windows only, so
    cached windows do not contain target-domain recording statistics.
    """
    x = np.asarray(data, dtype=np.float64)
    x = np.nan_to_num(x)
    x = signal.detrend(x, axis=-1, type="constant")

    nyq = fs / 2.0
    if notch is not None and nyq > notch + 2.0:
        low = max((notch - 1.0) / nyq, 1e-5)
        high = min((notch + 1.0) / nyq, 0.999)
        sos = signal.butter(2, [low, high], btype="bandstop", output="sos")
        x = _safe_sosfiltfilt(sos, x)

    if band is not None:
        low_hz, high_hz = band
        low = max(float(low_hz) / nyq, 1e-5)
        high = min(float(high_hz) / nyq, 0.999)
        if low < high:
            sos = signal.butter(4, [low, high], btype="bandpass", output="sos")
            x = _safe_sosfiltfilt(sos, x)

    out_fs = float(fs)
    if target_fs is not None and abs(float(fs) - float(target_fs)) > 1e-6:
        gcd = math.gcd(int(round(target_fs)), int(round(fs)))
        up = int(round(target_fs)) // gcd
        down = int(round(fs)) // gcd
        x = signal.resample_poly(x, up=up, down=down, axis=-1)
        out_fs = float(target_fs)

    return x.astype(np.float32), out_fs


def make_windows(data, fs, label, subject, session, paradigm, task,
                 window_sec=1.0, stride_sec=1.0, reject_z=None):
    samples = int(round(float(window_sec) * float(fs)))
    stride = int(round(float(stride_sec) * float(fs)))
    if samples <= 0 or stride <= 0:
        raise ValueError("window_sec and stride_sec must be positive")

    xs = []
    rows = []
    n_total = data.shape[-1]
    for start in range(0, n_total - samples + 1, stride):
        w = data[:, start:start + samples]
        if not np.isfinite(w).all():
            continue
        if reject_z is not None and np.max(np.abs(w)) > reject_z:
            continue
        if float(np.min(np.std(w, axis=-1))) < 1e-6:
            continue
        xs.append(w)
        rows.append((int(label), int(subject), int(session), str(paradigm), str(task), int(start)))

    if not xs:
        return np.empty((0, data.shape[0], samples), dtype=np.float32), rows
    return np.stack(xs).astype(np.float32), rows
