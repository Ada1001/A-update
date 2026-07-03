import argparse
import csv
import os

import pandas as pd

from src.cl_tsmnet.datasets import load_dataset
from src.cl_tsmnet.splits import domain_ids, iter_eval_subjects, make_split
from src.cl_tsmnet.training import train_one_split


def parse_args():
    parser = argparse.ArgumentParser(description="Run TSMNet on cognitive-load EEG datasets.")
    parser.add_argument("--dataset", choices=["stew", "eegmat", "cog-bci"], required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--cog-paradigm", choices=["nback", "matb"], default="nback")
    parser.add_argument("--protocol", choices=["single_session", "cog_multi_session", "loso"],
                        required=True)
    parser.add_argument("--subject", type=int, default=None,
                        help="Evaluate one subject only. Default: run all subjects.")
    parser.add_argument("--cache", default=None,
                        help="Optional .npz cache for preprocessed 1 s windows.")
    parser.add_argument("--target-fs", type=float, default=None,
                        help="Target sampling rate. Default: STEW=128 Hz, EEGMAT/COG-BCI=250 Hz.")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--bnorm", choices=["spddsbn", "spdbn", "none"], default="spddsbn")
    parser.add_argument("--temporal-filters", type=int, default=4)
    parser.add_argument("--spatial-filters", type=int, default=40)
    parser.add_argument("--subspacedims", type=int, default=20)
    parser.add_argument("--temp-kernel", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable light train-time augmentation for STEW/EEGMAT.")
    parser.add_argument("--output", default="outputs/tsmnet")
    return parser.parse_args()


def main():
    args = parse_args()
    load_subjects = None
    if args.subject is not None and args.protocol in ["single_session", "cog_multi_session"]:
        load_subjects = [args.subject]
    load_sessions = (1, 2, 3)
    if args.dataset == "cog-bci" and args.protocol == "single_session":
        load_sessions = (1,)
    dataset = load_dataset(
        args.dataset,
        data_root=args.data_root,
        cache=args.cache,
        rebuild_cache=args.rebuild_cache,
        cog_paradigm=args.cog_paradigm,
        subjects=load_subjects,
        sessions=load_sessions,
        target_fs=args.target_fs,
        window_sec=1.0,
        stride_sec=1.0,
    )
    domains = domain_ids(dataset, args.protocol)
    subjects = [args.subject] if args.subject is not None else iter_eval_subjects(
        dataset["meta"], args.protocol, dataset["name"])
    augment = (args.dataset in ["stew", "eegmat"]) and (not args.no_augment)

    run_name = "{}_{}_{}".format(dataset["name"], args.protocol, args.bnorm)
    out_root = os.path.join(args.output, run_name)
    if not os.path.exists(out_root):
        os.makedirs(out_root)

    results = []
    project_root = os.path.abspath(os.path.dirname(__file__))
    for subject in subjects:
        split = make_split(dataset, args.protocol, subject, seed=args.seed)
        fold_dir = os.path.join(out_root, "subject_{:02d}".format(int(subject)))
        res = train_one_split(
            dataset=dataset,
            domains=domains,
            split=split,
            project_root=project_root,
            output_dir=fold_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            bnorm=args.bnorm,
            augment=augment,
            temporal_filters=args.temporal_filters,
            spatial_filters=args.spatial_filters,
            subspacedims=args.subspacedims,
            temp_kernel=args.temp_kernel,
            seed=args.seed + int(subject),
        )
        row = {
            "dataset": dataset["name"],
            "protocol": args.protocol,
            "subject": int(subject),
            "bnorm": args.bnorm,
            "epochs_ran": res["epochs_ran"],
            "train_bacc": res["train"]["balanced_accuracy"],
            "val_bacc": res["val"]["balanced_accuracy"],
            "test_bacc": res["test"]["balanced_accuracy"],
            "train_acc": res["train"]["accuracy"],
            "val_acc": res["val"]["accuracy"],
            "test_acc": res["test"]["accuracy"],
            "n_train": len(split["train"]),
            "n_val": len(split["val"]),
            "n_test": len(split["test"]),
        }
        results.append(row)
        pd.DataFrame(res["history"]).to_csv(os.path.join(fold_dir, "history.csv"), index=False)
        print(row)

    result_path = os.path.join(out_root, "summary.csv")
    with open(result_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print("Saved:", result_path)


if __name__ == "__main__":
    main()
