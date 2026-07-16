import numpy as np


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


def _record_train_validation_split(indices, meta, val_size):
    """Contiguous per-record train/validation split for source-only sessions."""
    train_parts, val_parts = [], []
    frame = meta.iloc[np.asarray(indices, dtype=np.int64)].copy()
    frame["_idx"] = np.asarray(indices, dtype=np.int64)
    group_cols = ["subject", "session", "paradigm", "task"]
    for _, group in frame.groupby(group_cols, sort=True):
        ordered = group.sort_values("start_sample")["_idx"].values.astype(np.int64)
        n = len(ordered)
        if n < 2:
            train_parts.append(ordered)
            continue
        n_val = max(1, int(round(n * float(val_size))))
        n_val = min(n_val, n - 1)
        train_parts.append(ordered[:-n_val])
        val_parts.append(ordered[-n_val:])

    def concat(parts):
        if not parts:
            return np.asarray([], dtype=np.int64)
        return np.concatenate(parts).astype(np.int64)

    return concat(train_parts), concat(val_parts)


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


def _validate_loso_isolation(train, val, test, meta):
    """Fail closed if a future LOSO change mixes indices or subjects."""
    split_indices = {
        "train": set(int(v) for v in np.asarray(train, dtype=np.int64)),
        "val": set(int(v) for v in np.asarray(val, dtype=np.int64)),
        "test": set(int(v) for v in np.asarray(test, dtype=np.int64)),
    }
    names = list(split_indices)
    for pos, left in enumerate(names):
        for right in names[pos + 1:]:
            overlap = split_indices[left].intersection(split_indices[right])
            if overlap:
                raise RuntimeError(
                    "LOSO index leakage between {} and {}: {} windows".format(
                        left, right, len(overlap)
                    )
                )

    subjects = meta["subject"].values.astype(np.int64)
    split_subjects = {
        name: set(int(v) for v in np.unique(subjects[list(indices)]))
        for name, indices in split_indices.items()
    }
    for pos, left in enumerate(names):
        for right in names[pos + 1:]:
            overlap = split_subjects[left].intersection(split_subjects[right])
            if overlap:
                raise RuntimeError(
                    "LOSO subject leakage between {} and {}: {}".format(
                        left, right, sorted(overlap)
                    )
                )
    if len(split_subjects["test"]) != 1:
        raise RuntimeError(
            "LOSO test split must contain exactly one subject; got {}".format(
                sorted(split_subjects["test"])
            )
        )


def _cog_s1s2_to_s3_split(dataset, eval_subject, val_size):
    meta = dataset["meta"]
    subject = meta["subject"].values.astype(np.int64)
    session = meta["session"].values.astype(np.int64)
    subj_mask = subject == int(eval_subject)
    available_sessions = sorted(int(s) for s in np.unique(session[subj_mask]))
    required_sessions = [1, 2, 3]
    if not set(required_sessions).issubset(set(available_sessions)):
        raise ValueError(
            "COG-BCI S1+S2 -> S3 cross-session needs sessions {} for "
            "subject {}; got {}".format(
                required_sessions, int(eval_subject), available_sessions
            )
        )
    train_sessions = (1, 2)
    test_session = 3
    source = np.flatnonzero(subj_mask & np.isin(session, np.asarray(train_sessions)))
    test = np.flatnonzero(subj_mask & (session == int(test_session)))
    train, val = _record_train_validation_split(source, meta, val_size)
    if len(train) == 0 or len(val) == 0 or len(test) == 0:
        raise RuntimeError(
            "Invalid COG-BCI S1+S2 -> S3 split for subject {}: "
            "train={}, val={}, test={}".format(
                int(eval_subject), len(train), len(val), len(test)
            )
        )
    return {
        "train": train,
        "val": val,
        "test": test,
        "train_sessions": train_sessions,
        "test_session": int(test_session),
    }


def iter_eval_subjects(meta, protocol, dataset_name):
    subjects = sorted(int(s) for s in np.unique(meta["subject"].values))
    return subjects


def make_split(dataset, protocol, eval_subject, seed=42, val_size=0.2,
               test_size=0.2):
    meta = dataset["meta"]
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
        return _cog_s1s2_to_s3_split(dataset, eval_subject, val_size)
    elif protocol == "loso":
        test = np.flatnonzero(subject == int(eval_subject))
        source = np.flatnonzero(subject != int(eval_subject))
        train, val = _subject_holdout_validation_split(source, meta, val_size, seed + 1)
        _validate_loso_isolation(train, val, test, meta)
    else:
        raise ValueError("Unknown protocol: {}".format(protocol))

    if len(train) == 0 or len(val) == 0 or len(test) == 0:
        raise RuntimeError("Invalid split: train={}, val={}, test={}".format(
            len(train), len(val), len(test)))
    return {"train": train, "val": val, "test": test}


def make_splits(dataset, protocol, eval_subject, seed=42, val_size=0.2,
                test_size=0.2):
    if protocol == "cog_multi_session":
        if not dataset["name"].startswith("cog-bci"):
            raise ValueError("cog_multi_session is only defined for COG-BCI")
        return [_cog_s1s2_to_s3_split(dataset, eval_subject, val_size)]
    return [make_split(dataset, protocol, eval_subject, seed=seed,
                       val_size=val_size, test_size=test_size)]


def split_label_counts(y, indices):
    labels, counts = np.unique(y[np.asarray(indices, dtype=np.int64)], return_counts=True)
    return {int(label): int(count) for label, count in zip(labels, counts)}


def split_validation_issues(dataset, split, min_windows=2, min_class_windows=2,
                            require_all_classes=True):
    y = dataset["y"]
    all_labels = sorted(int(label) for label in np.unique(y))
    issues = []
    for name in ["train", "val", "test"]:
        indices = np.asarray(split[name], dtype=np.int64)
        if len(indices) < int(min_windows):
            issues.append("{} has {} windows < {}".format(
                name, len(indices), int(min_windows)
            ))
        counts = split_label_counts(y, indices)
        if require_all_classes:
            missing = [label for label in all_labels if counts.get(label, 0) == 0]
            if missing:
                issues.append("{} missing labels {}".format(name, missing))
        low = {
            label: counts.get(label, 0)
            for label in all_labels
            if counts.get(label, 0) < int(min_class_windows)
        }
        if low:
            issues.append("{} label counts below {}: {}".format(
                name, int(min_class_windows), low
            ))
    return issues


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
        "train_sessions": ",".join(str(s) for s in split.get("train_sessions", [])),
        "test_session": split.get("test_session", ""),
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
