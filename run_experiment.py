import argparse
import csv
import os

import numpy as np
import pandas as pd

from src.cl_tsmnet.datasets import load_dataset
from src.cl_tsmnet.splits import domain_ids, iter_eval_subjects, make_split
from src.cl_tsmnet.training import train_one_split


def _model_label(args):
    if args.model_name:
        return args.model_name
    if args.model == "eegconformer":
        return "eegconformer"
    if args.bnorm == "spddsbn":
        return "tsmnet_spddsbn"
    if args.bnorm == "spdbn":
        return "tsmnet_spdbn"
    return "tsmnet"


def _fmt_mean_std(values):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return "nan +/- nan"
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return "{:.4f} +/- {:.4f}".format(mean, std)


def write_aggregate_summary(rows, path):
    df = pd.DataFrame(rows)
    agg_rows = []
    for (dataset, model, protocol), group in df.groupby(["dataset", "model", "protocol"]):
        agg_rows.append({
            "dataset": dataset,
            "model": model,
            "protocol": protocol,
            "n": int(len(group)),
            "accuracy": _fmt_mean_std(group["test_acc"].values),
            "balanced_accuracy": _fmt_mean_std(group["test_bacc"].values),
            "f1": _fmt_mean_std(group["test_f1"].values),
            "auc": _fmt_mean_std(group["test_auc"].values),
        })
    columns = ["dataset", "model", "protocol", "n", "accuracy",
               "balanced_accuracy", "f1", "auc"]
    pd.DataFrame(agg_rows, columns=columns).to_csv(path, index=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Run TSMNet on cognitive-load EEG datasets.")
    parser.add_argument("--dataset", choices=["stew", "eegmat", "cog-bci"], required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--cog-paradigm", choices=["nback", "matb"], default="nback")
    parser.add_argument("--protocol", choices=["single_session", "cog_multi_session", "loso"],
                        required=True)
    parser.add_argument("--model", choices=["tsmnet", "eegconformer"], default="tsmnet")
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
    parser.add_argument("--model-name", default=None,
                        help="Model name written to result CSV files.")
    parser.add_argument("--temporal-filters", type=int, default=4)
    parser.add_argument("--spatial-filters", type=int, default=40)
    parser.add_argument("--subspacedims", type=int, default=20)
    parser.add_argument("--temp-kernel", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=float, default=0.2,
                        help="Validation fraction inside the source domain.")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Target/test fraction for single_session only.")
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable light train-time augmentation for STEW/EEGMAT.")
    parser.add_argument("--no-target-adapt", action="store_true",
                        help="Disable unlabeled target-domain BN refit for SPDDSBN.")
    parser.add_argument("--artifact-z", type=float, default=None,
                        help="Reject windows whose source-normalized absolute amplitude exceeds this value.")
    parser.add_argument("--no-artifact-reject", action="store_true",
                        help="Disable artifact-window rejection after source-only normalization.")
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
    target_adapt = (not args.no_target_adapt) and args.model == "tsmnet"
    artifact_z = None if args.no_artifact_reject else args.artifact_z

    run_tag = args.bnorm if args.model == "tsmnet" else "eegconformer"
    run_name = "{}_{}_{}".format(dataset["name"], args.protocol, run_tag)
    out_root = os.path.join(args.output, run_name)
    if not os.path.exists(out_root):
        os.makedirs(out_root)

    results = []
    model_name = _model_label(args)
    project_root = os.path.abspath(os.path.dirname(__file__))
    for subject in subjects:
        split = make_split(dataset, args.protocol, subject, seed=args.seed,
                           val_size=args.val_size, test_size=args.test_size)
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
            model_type=args.model,
            temporal_filters=args.temporal_filters,
            spatial_filters=args.spatial_filters,
            subspacedims=args.subspacedims,
            temp_kernel=args.temp_kernel,
            seed=args.seed + int(subject),
            target_adapt=target_adapt,
            artifact_z=artifact_z,
        )
        row = {
            "dataset": dataset["name"],
            "model": model_name,
            "protocol": args.protocol,
            "subject": int(subject),
            "model_type": args.model,
            "bnorm": args.bnorm if args.model == "tsmnet" else "",
            "epochs_ran": res["epochs_ran"],
            "best_epoch": res["best_epoch"],
            "best_val_loss": res["best_val_loss"],
            "target_adapt": res["target_adapt"],
            "artifact_z": res["artifact_z"],
            "train_bacc": res["train"]["balanced_accuracy"],
            "val_bacc": res["val"]["balanced_accuracy"],
            "test_bacc": res["test"]["balanced_accuracy"],
            "train_acc": res["train"]["accuracy"],
            "val_acc": res["val"]["accuracy"],
            "test_acc": res["test"]["accuracy"],
            "train_f1": res["train"]["f1"],
            "val_f1": res["val"]["f1"],
            "test_f1": res["test"]["f1"],
            "train_auc": res["train"]["auc"],
            "val_auc": res["val"]["auc"],
            "test_auc": res["test"]["auc"],
            "n_train": res["n_train"],
            "n_val": res["n_val"],
            "n_test": res["n_test"],
            "val_size": args.val_size,
            "test_size": args.test_size if args.protocol == "single_session" else "",
        }
        results.append(row)
        history = pd.DataFrame(res["history"])
        if len(history):
            history["is_best_epoch"] = history["epoch"] == res["best_epoch"]
        history.to_csv(os.path.join(fold_dir, "history.csv"), index=False)
        print(row)

    result_path = os.path.join(out_root, "summary.csv")
    with open(result_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print("Saved:", result_path)
    aggregate_path = os.path.join(out_root, "aggregate_summary.csv")
    write_aggregate_summary(results, aggregate_path)
    print("Saved:", aggregate_path)


if __name__ == "__main__":
    main()
