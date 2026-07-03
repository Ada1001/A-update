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
        train_val, test = _stratified_split(selected, y, test_size, seed)
        train, val = _stratified_split(train_val, y, val_size, seed + 1)
    elif protocol == "cog_multi_session":
        if not name.startswith("cog-bci"):
            raise ValueError("cog_multi_session is only defined for COG-BCI")
        source = np.flatnonzero((subject == int(eval_subject)) &
                                np.isin(session, np.asarray([1, 2])))
        test = np.flatnonzero((subject == int(eval_subject)) & (session == 3))
        train, val = _stratified_split(source, y, val_size, seed + 1)
    elif protocol == "loso":
        test = np.flatnonzero(subject == int(eval_subject))
        source = np.flatnonzero(subject != int(eval_subject))
        train, val = _stratified_split(source, y, val_size, seed + 1)
    else:
        raise ValueError("Unknown protocol: {}".format(protocol))

    if len(train) == 0 or len(val) == 0 or len(test) == 0:
        raise RuntimeError("Invalid split: train={}, val={}, test={}".format(
            len(train), len(val), len(test)))
    return {"train": train, "val": val, "test": test}


def split_label_counts(y, indices):
    labels, counts = np.unique(y[np.asarray(indices, dtype=np.int64)], return_counts=True)
    return {int(label): int(count) for label, count in zip(labels, counts)}


def split_summary(dataset, split, eval_subject):
    y = dataset["y"]
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
    }


def domain_ids(dataset, protocol):
    meta = dataset["meta"]
    subject = meta["subject"].values.astype(np.int64)
    session = meta["session"].values.astype(np.int64)
    if protocol == "loso":
        return subject
    return subject * 1000 + session
