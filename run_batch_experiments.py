import argparse
import subprocess
import sys


def _split_csv(value):
    return [item.strip() for item in str(value).split(",") if item.strip()]


MSTGC_ABLATION_MODELS = [
    "mstgc_graph_prior",
    "mstgc_graph_plv",
    "mstgc_graph_multigraph",
    "mstgc_dta_ce",
    "mstgc_dta_cheb_ce",
    "mstgc_dta_cheb_eudsbn",
    "mstgc_dta_cheb_spdmbn",
    "mstgc_dta_cheb_spdbn",
    "ms_tgc_spddsbn",
    "mstgc_wo_dta",
    "mstgc_wo_cheb",
    "mstgc_wo_spddsbn",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run multiple cognitive-load EEG experiments with automatic cache/output names."
    )
    parser.add_argument("--datasets", default="stew,eegmat,cog-bci",
                        help="Comma-separated: stew,eegmat,cog-bci")
    parser.add_argument("--protocols", default="single_session,loso",
                        help="Comma-separated: single_session,cog_multi_session,loso")
    parser.add_argument("--models", default="tsmnet,eegconformer,eegnet,bfgcn,tahag,mdtn,ms_tgc_spddsbn,svm,lsccn,lstm,bilstm,transformer,shallowcnn",
                        help="Comma-separated model names.")
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
    parser.add_argument("--min-split-windows", type=int, default=2)
    parser.add_argument("--min-class-windows", type=int, default=2)
    parser.add_argument("--allow-incomplete-splits", action="store_true")
    parser.add_argument("--bnorm", default="spddsbn", choices=["spddsbn", "spdbn", "none"])
    parser.add_argument("--conformer-emb-size", type=int, default=40)
    parser.add_argument("--conformer-depth", type=int, default=6)
    parser.add_argument("--conformer-num-heads", type=int, default=5)
    parser.add_argument("--conformer-dropout", type=float, default=0.5)
    parser.add_argument("--conformer-classifier-hidden", type=int, default=256)
    parser.add_argument("--eegnet-temporal-filters", type=int, default=64)
    parser.add_argument("--eegnet-spatial-filters", type=int, default=4)
    parser.add_argument("--eegnet-dropout", type=float, default=0.5)
    parser.add_argument("--eegnet-avgpool-factor", type=int, default=2)
    parser.add_argument("--bfgcn-kadj", type=int, default=2)
    parser.add_argument("--bfgcn-num-out", type=int, default=16)
    parser.add_argument("--bfgcn-att-hidden", type=int, default=16)
    parser.add_argument("--bfgcn-classifier-hidden", type=int, default=32)
    parser.add_argument("--bfgcn-avgpool", type=int, default=2)
    parser.add_argument("--bfgcn-dropout", type=float, default=0.0)
    parser.add_argument("--bfgcn-domain-weight", type=float, default=1.0)
    parser.add_argument("--tahag-dropout", type=float, default=0.25)
    parser.add_argument("--tahag-domain-weight", type=float, default=1.0)
    parser.add_argument("--tahag-mmd-weight", type=float, default=1.0)
    parser.add_argument("--no-tahag-adaptive", action="store_true")
    parser.add_argument("--no-tahag-attention", action="store_true")
    parser.add_argument("--mdtn-hidden-dim", type=int, default=64)
    parser.add_argument("--mdtn-num-nodes", type=int, default=0)
    parser.add_argument("--mdtn-kernel-length", type=int, default=16)
    parser.add_argument("--mdtn-num-heads", type=int, default=4)
    parser.add_argument("--mdtn-cheby-order", type=int, default=3)
    parser.add_argument("--mdtn-dropout", type=float, default=0.5)
    parser.add_argument("--mdtn-lambda-match", type=float, default=0.1)
    parser.add_argument("--mdtn-marginal-weight", type=float, default=0.01)
    parser.add_argument("--mdtn-conditional-weight", type=float, default=0.01)
    parser.add_argument("--mdtn-l1-weight", type=float, default=0.01)
    parser.add_argument("--mstgc-temporal-hidden", type=int, default=64)
    parser.add_argument("--mstgc-graph-hidden", type=int, default=64)
    parser.add_argument("--mstgc-fusion-dim", type=int, default=128)
    parser.add_argument("--mstgc-kernel-length", type=int, default=16)
    parser.add_argument("--mstgc-num-heads", type=int, default=4)
    parser.add_argument("--mstgc-cheby-order", type=int, default=3)
    parser.add_argument("--mstgc-dropout", type=float, default=0.5)
    parser.add_argument("--mstgc-num-nodes", type=int, default=0)
    parser.add_argument("--mstgc-graph-k", type=int, default=4)
    parser.add_argument("--mstgc-time-points", type=int, default=64)
    parser.add_argument("--mstgc-shrinkage", type=float, default=0.1)
    parser.add_argument("--svm-estimator", default="linear-svc",
                        choices=["linear-svc", "svc"])
    parser.add_argument("--svm-kernel", default="rbf",
                        choices=["linear", "poly", "rbf", "sigmoid"])
    parser.add_argument("--svm-c", type=float, default=1.0)
    parser.add_argument("--svm-gamma", default="scale")
    parser.add_argument("--svm-class-weight", default="balanced")
    parser.add_argument("--svm-probability", action="store_true")
    parser.add_argument("--svm-max-iter", type=int, default=5000)
    parser.add_argument("--lsccn-latent-dim", type=int, default=200)
    parser.add_argument("--lsccn-routing-iters", type=int, default=3)
    parser.add_argument("--lsccn-recon-weight", type=float, default=1e-5)
    parser.add_argument("--lsccn-kl-weight", type=float, default=0.1)
    parser.add_argument("--recurrent-hidden", type=int, default=64)
    parser.add_argument("--recurrent-layers", type=int, default=1)
    parser.add_argument("--recurrent-dropout", type=float, default=0.5)
    parser.add_argument("--transformer-d-model", type=int, default=64)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-ff", type=int, default=128)
    parser.add_argument("--transformer-dropout", type=float, default=0.2)
    parser.add_argument("--shallow-filters", type=int, default=40)
    parser.add_argument("--shallow-kernel", type=int, default=25)
    parser.add_argument("--shallow-pool", type=int, default=25)
    parser.add_argument("--shallow-dropout", type=float, default=0.5)
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
                        "--min-split-windows", str(args.min_split_windows),
                        "--min-class-windows", str(args.min_class_windows),
                    ]
                    if args.allow_incomplete_splits:
                        cmd.append("--allow-incomplete-splits")
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
                    if model == "tahag":
                        cmd.extend([
                            "--tahag-dropout", str(args.tahag_dropout),
                            "--tahag-domain-weight", str(args.tahag_domain_weight),
                            "--tahag-mmd-weight", str(args.tahag_mmd_weight),
                        ])
                        if args.no_tahag_adaptive:
                            cmd.append("--no-tahag-adaptive")
                        if args.no_tahag_attention:
                            cmd.append("--no-tahag-attention")
                    if model == "mdtn":
                        cmd.extend([
                            "--mdtn-hidden-dim", str(args.mdtn_hidden_dim),
                            "--mdtn-num-nodes", str(args.mdtn_num_nodes),
                            "--mdtn-kernel-length", str(args.mdtn_kernel_length),
                            "--mdtn-num-heads", str(args.mdtn_num_heads),
                            "--mdtn-cheby-order", str(args.mdtn_cheby_order),
                            "--mdtn-dropout", str(args.mdtn_dropout),
                            "--mdtn-lambda-match", str(args.mdtn_lambda_match),
                            "--mdtn-marginal-weight", str(args.mdtn_marginal_weight),
                            "--mdtn-conditional-weight", str(args.mdtn_conditional_weight),
                            "--mdtn-l1-weight", str(args.mdtn_l1_weight),
                        ])
                    if model in MSTGC_ABLATION_MODELS:
                        cmd.extend([
                            "--mstgc-temporal-hidden", str(args.mstgc_temporal_hidden),
                            "--mstgc-graph-hidden", str(args.mstgc_graph_hidden),
                            "--mstgc-fusion-dim", str(args.mstgc_fusion_dim),
                            "--mstgc-kernel-length", str(args.mstgc_kernel_length),
                            "--mstgc-num-heads", str(args.mstgc_num_heads),
                            "--mstgc-cheby-order", str(args.mstgc_cheby_order),
                            "--mstgc-dropout", str(args.mstgc_dropout),
                            "--mstgc-num-nodes", str(args.mstgc_num_nodes),
                            "--mstgc-graph-k", str(args.mstgc_graph_k),
                            "--mstgc-time-points", str(args.mstgc_time_points),
                            "--mstgc-shrinkage", str(args.mstgc_shrinkage),
                        ])
                    if model == "svm":
                        cmd.extend([
                            "--svm-estimator", str(args.svm_estimator),
                            "--svm-kernel", str(args.svm_kernel),
                            "--svm-c", str(args.svm_c),
                            "--svm-gamma", str(args.svm_gamma),
                            "--svm-class-weight", str(args.svm_class_weight),
                            "--svm-max-iter", str(args.svm_max_iter),
                        ])
                        if args.svm_probability:
                            cmd.append("--svm-probability")
                    if model == "lsccn":
                        cmd.extend([
                            "--lsccn-latent-dim", str(args.lsccn_latent_dim),
                            "--lsccn-routing-iters", str(args.lsccn_routing_iters),
                            "--lsccn-recon-weight", str(args.lsccn_recon_weight),
                            "--lsccn-kl-weight", str(args.lsccn_kl_weight),
                        ])
                    if model in ["lstm", "bilstm"]:
                        cmd.extend([
                            "--recurrent-hidden", str(args.recurrent_hidden),
                            "--recurrent-layers", str(args.recurrent_layers),
                            "--recurrent-dropout", str(args.recurrent_dropout),
                        ])
                    if model == "transformer":
                        cmd.extend([
                            "--transformer-d-model", str(args.transformer_d_model),
                            "--transformer-heads", str(args.transformer_heads),
                            "--transformer-layers", str(args.transformer_layers),
                            "--transformer-ff", str(args.transformer_ff),
                            "--transformer-dropout", str(args.transformer_dropout),
                        ])
                    if model == "shallowcnn":
                        cmd.extend([
                            "--shallow-filters", str(args.shallow_filters),
                            "--shallow-kernel", str(args.shallow_kernel),
                            "--shallow-pool", str(args.shallow_pool),
                            "--shallow-dropout", str(args.shallow_dropout),
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
