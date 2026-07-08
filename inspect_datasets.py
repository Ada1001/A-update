import argparse

import pandas as pd

from src.cl_tsmnet.datasets import (
    discover_cog_bci_recording_subjects,
    discover_cog_bci_zip_subjects,
    load_dataset,
)
from src.cl_tsmnet.experiment_utils import default_cache_path, default_target_fs
from src.cl_tsmnet.splits import (
    iter_eval_subjects,
    make_split,
    split_summary,
    split_validation_issues,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["stew", "eegmat", "cog-bci"], required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--cog-paradigm", choices=["nback", "matb"], default="nback")
    parser.add_argument("--cache", default=None)
    parser.add_argument("--cache-root", default="outputs/cache")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--protocol", choices=["single_session", "cog_multi_session", "loso"],
                        default=None)
    parser.add_argument("--subject", type=int, default=None)
    parser.add_argument("--target-fs", type=float, default=None,
                        help="Target sampling rate. Default: STEW=128 Hz, EEGMAT/COG-BCI=250 Hz.")
    parser.add_argument("--val-size", type=float, default=0.2,
                        help="Validation fraction for LOSO source subjects and COG fallback splits.")
    parser.add_argument("--single-val-size", type=float, default=0.125,
                        help="Validation fraction inside the single_session train+val block. "
                             "Default 0.125 gives train/val/test = 0.7/0.1/0.2.")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Target/test fraction for single_session sequential time-block split.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-split-windows", type=int, default=2,
                        help="Warn when any train/validation/test split has fewer windows than this.")
    parser.add_argument("--min-class-windows", type=int, default=2,
                        help="Warn when any class count in a split is below this value.")
    args = parser.parse_args()

    subjects = None
    if args.subject is not None and args.protocol in [None, "single_session", "cog_multi_session"]:
        subjects = [args.subject]
    sessions = (1,)
    if args.dataset == "cog-bci" and args.protocol != "single_session":
        sessions = (1, 2, 3)
    elif args.dataset == "cog-bci" and args.protocol == "single_session":
        sessions = (1,)
    target_fs = default_target_fs(args.dataset, args.target_fs)
    cache = args.cache or default_cache_path(
        args.dataset,
        args.protocol or "all",
        cog_paradigm=args.cog_paradigm,
        subject=args.subject,
        target_fs=target_fs,
        cache_root=args.cache_root,
    )
    print("cache:", cache)
    ds = load_dataset(args.dataset, data_root=args.data_root, cache=cache,
                      rebuild_cache=args.rebuild_cache,
                      cog_paradigm=args.cog_paradigm, subjects=subjects,
                      sessions=sessions, target_fs=target_fs)
    meta = ds["meta"]
    print("dataset:", ds["name"])
    print("x shape:", ds["x"].shape)
    print("fs:", ds["fs"])
    print("channels({}): {}".format(len(ds["channels"]), ", ".join(ds["channels"])))
    print("labels:", pd.Series(ds["y"]).value_counts().sort_index().to_dict())
    print("subjects:", meta["subject"].nunique())
    if args.dataset == "cog-bci":
        zip_subjects = discover_cog_bci_zip_subjects(args.data_root)
        recording_subjects = discover_cog_bci_recording_subjects(
            args.data_root, paradigm=args.cog_paradigm, sessions=sessions
        )
        cached_subjects = sorted(int(s) for s in meta["subject"].unique())
        cache_recording_subjects = ds.get("recording_subjects", [])
        subjects_without_windows = ds.get("subjects_without_windows", [])
        print("available COG-BCI zip subjects:", len(zip_subjects), zip_subjects)
        print("available COG-BCI recording subjects:",
              len(recording_subjects), recording_subjects)
        print("cache window subjects:", len(cached_subjects), cached_subjects)
        if len(cache_recording_subjects):
            print("cache recording subjects:",
                  len(cache_recording_subjects), cache_recording_subjects)
        if len(subjects_without_windows):
            print("cache subjects without usable windows:",
                  len(subjects_without_windows), subjects_without_windows)
        raw_missing = sorted(set(recording_subjects) - set(cached_subjects))
        if raw_missing:
            print("COG-BCI recording subjects absent from window cache:", raw_missing)
    print(meta.groupby(["subject", "session", "task"]).size().head(30))

    if args.protocol is not None:
        eval_subjects = ([args.subject] if args.subject is not None
                         else iter_eval_subjects(meta, args.protocol, ds["name"]))
        rows = []
        for subject in eval_subjects:
            split_val_size = args.single_val_size if args.protocol == "single_session" else args.val_size
            split = make_split(ds, args.protocol, subject, seed=args.seed,
                               val_size=split_val_size, test_size=args.test_size)
            row = split_summary(ds, split, subject)
            row["split_issues"] = "; ".join(split_validation_issues(
                ds, split,
                min_windows=args.min_split_windows,
                min_class_windows=args.min_class_windows,
                require_all_classes=True,
            ))
            rows.append(row)
        table = pd.DataFrame(rows)
        display_cols = ["subject", "n_source", "n_target", "n_train", "n_val", "n_test",
                        "train_subjects", "val_subjects", "test_subjects",
                        "train_pct_total", "val_pct_total", "test_pct_total",
                        "train_labels", "val_labels", "test_labels"]
        print("\nsplit protocol:", args.protocol)
        print(table[display_cols].to_string(index=False))
        numeric = table[["n_source", "n_target", "n_train", "n_val", "n_test"]]
        print("\nsplit count mean:")
        print(numeric.mean().round(2).to_string())
        small = table[
            (table["n_train"] < int(args.min_split_windows)) |
            (table["n_val"] < int(args.min_split_windows)) |
            (table["n_test"] < int(args.min_split_windows))
        ]
        if len(small):
            print("\nWARNING: splits with very small window counts:")
            print(small[display_cols].to_string(index=False))
        issue_rows = table[table["split_issues"].astype(str) != ""]
        if len(issue_rows):
            print("\nWARNING: split label/window quality issues:")
            print(issue_rows[["subject", "split_issues"]].to_string(index=False))
        if args.dataset == "cog-bci" and args.protocol == "cog_multi_session":
            print("\nCOG-BCI per-subject/session/task window counts:")
            counts = (meta.groupby(["subject", "session", "task"])
                      .size().rename("n_windows").reset_index())
            print(counts.to_string(index=False))
            session_counts = (meta.groupby(["subject", "session"])
                              .size().rename("n_windows").reset_index())
            small_sessions = session_counts[
                session_counts["n_windows"] < int(args.min_split_windows)
            ]
            if len(small_sessions):
                print("\nWARNING: subject/session blocks below --min-split-windows:")
                print(small_sessions.to_string(index=False))
        if args.protocol == "single_session" and args.subject is not None:
            split_val_size = args.single_val_size
            split = make_split(ds, args.protocol, args.subject, seed=args.seed,
                               val_size=split_val_size, test_size=args.test_size)
            print("\nsingle_session time-block ranges:")
            for name in ["train", "val", "test"]:
                idx = split[name]
                ranges = meta.iloc[idx].groupby(
                    ["subject", "session", "paradigm", "task"]
                )["start_sample"].agg(["min", "max", "count"])
                print("\n{}:".format(name))
                print(ranges.to_string())


if __name__ == "__main__":
    main()
