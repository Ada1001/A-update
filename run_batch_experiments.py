import argparse
import subprocess
import sys


def _split_csv(value):
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run multiple cognitive-load EEG experiments with automatic cache/output names."
    )
    parser.add_argument("--datasets", default="stew,eegmat,cog-bci",
                        help="Comma-separated: stew,eegmat,cog-bci")
    parser.add_argument("--protocols", default="single_session,loso",
                        help="Comma-separated: single_session,cog_multi_session,loso")
    parser.add_argument("--models", default="tsmnet,eegconformer",
                        help="Comma-separated: tsmnet,eegconformer")
    parser.add_argument("--cog-paradigms", default="nback,matb",
                        help="Comma-separated COG-BCI paradigms.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--cache-root", default="outputs/cache")
    parser.add_argument("--master-summary", default="outputs/master_summary.csv")
    parser.add_argument("--target-fs", default=None,
                        help="Optional sampling rate applied to all runs.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--single-val-size", type=float, default=0.125)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--bnorm", default="spddsbn", choices=["spddsbn", "spdbn", "none"])
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--no-target-adapt", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    datasets = _split_csv(args.datasets)
    protocols = _split_csv(args.protocols)
    models = _split_csv(args.models)
    cog_paradigms = _split_csv(args.cog_paradigms)

    commands = []
    for dataset in datasets:
        paradigms = cog_paradigms if dataset == "cog-bci" else [None]
        for paradigm in paradigms:
            for protocol in protocols:
                if protocol == "cog_multi_session" and dataset != "cog-bci":
                    continue
                for model in models:
                    cmd = [
                        sys.executable, "run_experiment.py",
                        "--dataset", dataset,
                        "--protocol", protocol,
                        "--model", model,
                        "--data-root", args.data_root,
                        "--output", args.output,
                        "--cache-root", args.cache_root,
                        "--master-summary", args.master_summary,
                        "--epochs", str(args.epochs),
                        "--batch-size", str(args.batch_size),
                        "--lr", str(args.lr),
                        "--weight-decay", str(args.weight_decay),
                        "--seed", str(args.seed),
                        "--val-size", str(args.val_size),
                        "--single-val-size", str(args.single_val_size),
                        "--test-size", str(args.test_size),
                    ]
                    if args.target_fs is not None:
                        cmd.extend(["--target-fs", str(args.target_fs)])
                    if dataset == "cog-bci":
                        cmd.extend(["--cog-paradigm", paradigm])
                    if model == "tsmnet":
                        cmd.extend(["--bnorm", args.bnorm])
                    if args.rebuild_cache:
                        cmd.append("--rebuild-cache")
                    if args.no_augment:
                        cmd.append("--no-augment")
                    if args.no_target_adapt:
                        cmd.append("--no-target-adapt")
                    commands.append(cmd)

    for cmd in commands:
        print(" ".join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
