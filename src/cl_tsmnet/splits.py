import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def _stratified_split(indices, y, test_size, seed):
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) == 0:
        raise ValueError("Cannot split an empty index set")
    labels = y[indices]
    if len(np.unique(labels)) < 2:
        cut = int(round(len(indices) * (1.0 - test_size)))
        return indices[:cut], indices[cut:]
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size,
                                      random_state=seed)
    tr_rel, te_rel = next(splitter.split(np.zeros(len(indices)), labels))
    return indices[tr_rel], indices[te_rel]


def _record_block_split(indices, meta, test_size, val_size):
    """Contiguous per-record split to reduce adjacent-window leakage."""
    train_parts, val_parts, test_parts = [], [], []
    frame = meta.iloc[np.asarray(indices, dtype=np.int64)].copy()
    frame["_idx"] = np.asarray(indices, dtype=np.int64)
    group_cols = ["subject", "session", "paradigm", "task"]
    for _, group in frame.groupby(group_cols, sort=True):
        ordered = group.sort_values("start_sample")["_idx"].values.astype(np.int64)
        n = len(ordered)
        if n < 3:
            train_parts.append(ordered)
            continue
        n_test = max(1, int(round(n * float(test_size))))
        n_test = min(n_test, n - 2)
        train_val = ordered[:-n_test]
        test_parts.append(ordered[-n_test:])
        n_val = max(1, int(round(len(train_val) * float(val_size))))
        n_val = min(n_val, len(train_val) - 1)
        train_parts.append(train_val[:-n_val])
        val_parts.append(train_val[-n_val:])
    def concat(parts):
        if not parts:
            return np.asarray([], dtype=np.int64)
        return np.concatenate(parts).astype(np.int64)
    return concat(train_parts), concat(val_parts), concat(test_parts)


def _subject_holdout_validation_split(source, meta, val_size, seed):
    source = np.asarray(source, dtype=np.int64)
    subjects = np.unique(meta.iloc[source]["subject"].values.astype(np.int64))
    if len(subjects) < 2:
        raise ValueError(
            "LOSO subject-level validation needs at least two source subjects; got {}. "
            "This usually means the dataset/cache contains too few subjects for LOSO "
            "after holding out the target subject.".format(len(subjects))
        )
    rng = np.random.RandomState(seed)
    shuffled = np.array(subjects, copy=True)
    rng.shuffle(shuffled)
    n_val = max(1, int(np.ceil(len(shuffled) * float(val_size))))
    n_val = min(n_val, len(shuffled) - 1)
    val_subjects = set(int(s) for s in shuffled[:n_val])
    subject_arr = meta["subject"].values.astype(np.int64)
    val = source[np.isin(subject_arr[source], np.asarray(sorted(val_subjects)))]
    train = source[~np.isin(subject_arr[source], np.asarray(sorted(val_subjects)))]
    if len(train) == 0 or len(val) == 0:
        raise RuntimeError("Invalid LOSO subject validation split")
    return train, val


def iter_eval_subjects(meta, protocol, dataset_name):
    subjects = sorted(int(s) for s in np.unique(meta["subject"].values))
    return subjects


def make_split(dataset, protocol, eval_subject, seed=42, val_size=0.2,
               test_size=0.2):
    meta = dataset["meta"]
    y = dataset["y"]
    subject = meta["subject"].values.astype(np.int64)
    session = meta["session"].values.astype(np.int64)
    name = dataset["name"]

    if protocol == "single_session":
        sess = 1
        selected = np.flatnonzero((subject == int(eval_subject)) & (session == sess))
        train, val, test = _record_block_split(selected, meta, test_size, val_size)
    elif protocol == "cog_multi_session":
        if not name.startswith("cog-bci"):
            raise ValueError("cog_multi_session is only defined for COG-BCI")
        train = np.flatnonzero((subject == int(eval_subject)) & (session == 1))
        val = np.flatnonzero((subject == int(eval_subject)) & (session == 2))
        test = np.flatnonzero((subject == int(eval_subject)) & (session == 3))
        if len(train) == 0 or len(val) == 0:
            source = np.flatnonzero((subject == int(eval_subject)) &
                                    np.isin(session, np.asarray([1, 2])))
            train, val = _stratified_split(source, y, val_size, seed + 1)
    elif protocol == "loso":
        test = np.flatnonzero(subject == int(eval_subject))
        source = np.flatnonzero(subject != int(eval_subject))
        train, val = _subject_holdout_validation_split(source, meta, val_size, seed + 1)
    else:
        raise ValueError("Unknown protocol: {}".format(protocol))

    if len(train) == 0 or len(val) == 0 or len(test) == 0:
        raise RuntimeError("Invalid split: train={}, val={}, test={}".format(
            len(train), len(val), len(test)))
    return {"train": train, "val": val, "test": test}


def make_splits(dataset, protocol, eval_subject, seed=42, val_size=0.2,
                test_size=0.2):
    return [make_split(dataset, protocol, eval_subject, seed=seed,
                       val_size=val_size, test_size=test_size)]


def split_label_counts(y, indices):
    labels, counts = np.unique(y[np.asarray(indices, dtype=np.int64)], return_counts=True)
    return {int(label): int(count) for label, count in zip(labels, counts)}


def split_summary(dataset, split, eval_subject):
    y = dataset["y"]
    meta = dataset["meta"]
    train = split["train"]
    val = split["val"]
    test = split["test"]
    source = np.concatenate([train, val])
    total = len(source) + len(test)
    return {
        "subject": int(eval_subject),
        "n_source": int(len(source)),
        "n_target": int(len(test)),
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "n_test": int(len(test)),
        "train_pct_total": float(len(train) / total) if total else 0.0,
        "val_pct_total": float(len(val) / total) if total else 0.0,
        "test_pct_total": float(len(test) / total) if total else 0.0,
        "train_labels": split_label_counts(y, train),
        "val_labels": split_label_counts(y, val),
        "test_labels": split_label_counts(y, test),
        "train_subjects": int(meta.iloc[train]["subject"].nunique()),
        "val_subjects": int(meta.iloc[val]["subject"].nunique()),
        "test_subjects": int(meta.iloc[test]["subject"].nunique()),
    }


def domain_ids(dataset, protocol):
    meta = dataset["meta"]
    subject = meta["subject"].values.astype(np.int64)
    session = meta["session"].values.astype(np.int64)
    if protocol == "loso":
        return subject
    return subject * 1000 + session
