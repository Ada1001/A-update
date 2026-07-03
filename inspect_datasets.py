import argparse

import pandas as pd

from src.cl_tsmnet.datasets import load_dataset
from src.cl_tsmnet.splits import iter_eval_subjects, make_split, split_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["stew", "eegmat", "cog-bci"], required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--cog-paradigm", choices=["nback", "matb"], default="nback")
    parser.add_argument("--cache", default=None)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--protocol", choices=["single_session", "cog_multi_session", "loso"],
                        default=None)
    parser.add_argument("--subject", type=int, default=None)
    parser.add_argument("--target-fs", type=float, default=None,
                        help="Target sampling rate. Default: STEW=128 Hz, EEGMAT/COG-BCI=250 Hz.")
    parser.add_argument("--val-size", type=float, default=0.2,
                        help="Validation fraction inside the source domain.")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Target/test fraction for single_session only.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    subjects = None
    if args.subject is not None and args.protocol in [None, "single_session", "cog_multi_session"]:
        subjects = [args.subject]
    sessions = (1, 2, 3)
    if args.dataset == "cog-bci" and args.protocol == "single_session":
        sessions = (1,)
    ds = load_dataset(args.dataset, data_root=args.data_root, cache=args.cache,
                      rebuild_cache=args.rebuild_cache,
                      cog_paradigm=args.cog_paradigm, subjects=subjects,
                      sessions=sessions, target_fs=args.target_fs)
    meta = ds["meta"]
    print("dataset:", ds["name"])
    print("x shape:", ds["x"].shape)
    print("fs:", ds["fs"])
    print("channels({}): {}".format(len(ds["channels"]), ", ".join(ds["channels"])))
    print("labels:", pd.Series(ds["y"]).value_counts().sort_index().to_dict())
    print("subjects:", meta["subject"].nunique())
    print(meta.groupby(["subject", "session", "task"]).size().head(30))

    if args.protocol is not None:
        eval_subjects = ([args.subject] if args.subject is not None
                         else iter_eval_subjects(meta, args.protocol, ds["name"]))
        rows = []
        for subject in eval_subjects:
            split = make_split(ds, args.protocol, subject, seed=args.seed,
                               val_size=args.val_size, test_size=args.test_size)
            rows.append(split_summary(ds, split, subject))
        table = pd.DataFrame(rows)
        display_cols = ["subject", "n_source", "n_target", "n_train", "n_val", "n_test",
                        "train_pct_total", "val_pct_total", "test_pct_total",
                        "train_labels", "val_labels", "test_labels"]
        print("\nsplit protocol:", args.protocol)
        print(table[display_cols].to_string(index=False))
        numeric = table[["n_source", "n_target", "n_train", "n_val", "n_test"]]
        print("\nsplit count mean:")
        print(numeric.mean().round(2).to_string())


if __name__ == "__main__":
    main()
