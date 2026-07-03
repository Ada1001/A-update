import argparse

import pandas as pd

from src.cl_tsmnet.datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["stew", "eegmat", "cog-bci"], required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--cog-paradigm", choices=["nback", "matb"], default="nback")
    parser.add_argument("--cache", default=None)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--subject", type=int, default=None)
    parser.add_argument("--target-fs", type=float, default=None,
                        help="Target sampling rate. Default: STEW=128 Hz, EEGMAT/COG-BCI=250 Hz.")
    args = parser.parse_args()

    subjects = [args.subject] if args.subject is not None else None
    ds = load_dataset(args.dataset, data_root=args.data_root, cache=args.cache,
                      rebuild_cache=args.rebuild_cache,
                      cog_paradigm=args.cog_paradigm, subjects=subjects,
                      target_fs=args.target_fs)
    meta = ds["meta"]
    print("dataset:", ds["name"])
    print("x shape:", ds["x"].shape)
    print("fs:", ds["fs"])
    print("channels({}): {}".format(len(ds["channels"]), ", ".join(ds["channels"])))
    print("labels:", pd.Series(ds["y"]).value_counts().sort_index().to_dict())
    print("subjects:", meta["subject"].nunique())
    print(meta.groupby(["subject", "session", "task"]).size().head(30))


if __name__ == "__main__":
    main()
