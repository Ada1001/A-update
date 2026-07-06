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
    parser.add_argument("--models", default="tsmnet,eegconformer,eegnet,bfgcn,mdtn-gmda",
                        help="Comma-separated: tsmnet,eegconformer,eegnet,bfgcn,mdtn-gmda")
    parser.add_argument("--cog-paradigms", default="nback,matb",
                        help="Comma-separated COG-BCI paradigms.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--cache-root", default="outputs/cache")
    parser.add_argument("--master-summary", default="outputs/master_summary.csv")
    parser.add_argument("--target-fs", default=None,
                        help="Optional sampling rate applied to all runs.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--single-val-size", type=float, default=0.125)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--bnorm", default="spddsbn", choices=["spddsbn", "spdbn", "none"])
    parser.add_argument("--conformer-emb-size", type=int, default=40)
    parser.add_argument("--conformer-depth", type=int, default=6)
    parser.add_argument("--conformer-num-heads", type=int, default=5)
    parser.add_argument("--conformer-dropout", type=float, default=0.5)
    parser.add_argument("--conformer-classifier-hidden", type=int, default=256)
    parser.add_argument("--eegnet-temporal-filters", type=int, default=8)
    parser.add_argument("--eegnet-spatial-filters", type=int, default=2)
    parser.add_argument("--eegnet-dropout", type=float, default=0.5)
    parser.add_argument("--eegnet-avgpool-factor", type=int, default=4)
    parser.add_argument("--bfgcn-kadj", type=int, default=2)
    parser.add_argument("--bfgcn-num-out", type=int, default=16)
    parser.add_argument("--bfgcn-att-hidden", type=int, default=16)
    parser.add_argument("--bfgcn-classifier-hidden", type=int, default=32)
    parser.add_argument("--bfgcn-avgpool", type=int, default=2)
    parser.add_argument("--bfgcn-dropout", type=float, default=0.0)
    parser.add_argument("--bfgcn-domain-weight", type=float, default=1.0)
    parser.add_argument("--mdtn-hidden-dim", type=int, default=64)
    parser.add_argument("--mdtn-k-length", type=int, default=16)
    parser.add_argument("--mdtn-graph-k", type=int, default=3)
    parser.add_argument("--mdtn-num-heads", type=int, default=4)
    parser.add_argument("--mdtn-dropout", type=float, default=0.5)
    parser.add_argument("--mdtn-lambda-match", type=float, default=0.1)
    parser.add_argument("--mdtn-alpha", type=float, default=0.01)
    parser.add_argument("--mdtn-beta", type=float, default=0.01)
    parser.add_argument("--mdtn-l1-weight", type=float, default=0.01)
    parser.add_argument("--mdtn-max-iter", type=int, default=1000)
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
                        "--patience", str(args.patience),
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
                    if model == "eegconformer":
                        cmd.extend([
                            "--conformer-emb-size", str(args.conformer_emb_size),
                            "--conformer-depth", str(args.conformer_depth),
                            "--conformer-num-heads", str(args.conformer_num_heads),
                            "--conformer-dropout", str(args.conformer_dropout),
                            "--conformer-classifier-hidden", str(args.conformer_classifier_hidden),
                        ])
                    if model == "eegnet":
                        cmd.extend([
                            "--eegnet-temporal-filters", str(args.eegnet_temporal_filters),
                            "--eegnet-spatial-filters", str(args.eegnet_spatial_filters),
                            "--eegnet-dropout", str(args.eegnet_dropout),
                            "--eegnet-avgpool-factor", str(args.eegnet_avgpool_factor),
                        ])
                    if model == "bfgcn":
                        cmd.extend([
                            "--bfgcn-kadj", str(args.bfgcn_kadj),
                            "--bfgcn-num-out", str(args.bfgcn_num_out),
                            "--bfgcn-att-hidden", str(args.bfgcn_att_hidden),
                            "--bfgcn-classifier-hidden", str(args.bfgcn_classifier_hidden),
                            "--bfgcn-avgpool", str(args.bfgcn_avgpool),
                            "--bfgcn-dropout", str(args.bfgcn_dropout),
                            "--bfgcn-domain-weight", str(args.bfgcn_domain_weight),
                        ])
                    if model == "mdtn-gmda":
                        cmd.extend([
                            "--mdtn-hidden-dim", str(args.mdtn_hidden_dim),
                            "--mdtn-k-length", str(args.mdtn_k_length),
                            "--mdtn-graph-k", str(args.mdtn_graph_k),
                            "--mdtn-num-heads", str(args.mdtn_num_heads),
                            "--mdtn-dropout", str(args.mdtn_dropout),
                            "--mdtn-lambda-match", str(args.mdtn_lambda_match),
                            "--mdtn-alpha", str(args.mdtn_alpha),
                            "--mdtn-beta", str(args.mdtn_beta),
                            "--mdtn-l1-weight", str(args.mdtn_l1_weight),
                            "--mdtn-max-iter", str(args.mdtn_max_iter),
                        ])
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
