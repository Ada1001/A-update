import argparse
import csv
import os

import numpy as np
import pandas as pd

from src.cl_tsmnet.datasets import load_dataset
from src.cl_tsmnet.experiment_utils import (
    append_master_summary,
    default_cache_path,
    default_target_fs,
    run_directory_name,
)
from src.cl_tsmnet.splits import (
    domain_ids,
    iter_eval_subjects,
    make_split,
    split_validation_issues,
)
from src.cl_tsmnet.training import train_one_split


def _model_label(args):
    if args.model_name:
        return args.model_name
    if args.model == "eegconformer":
        return "eegconformer"
    if args.model == "eegnet":
        return "eegnet"
    if args.model == "bfgcn":
        return "bfgcn"
    if args.model == "tahag":
        return "tahag"
    if args.model == "mdtn":
        return "mdtn_gmda"
    if args.model == "ms_tgc_spddsbn":
        return "ms_tgc_spddsbn"
    if args.model == "svm":
        return "svm"
    if args.model == "lsccn":
        return "lsccn"
    if args.model in ["lstm", "bilstm", "transformer", "shallowcnn"]:
        return args.model
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
    parser = argparse.ArgumentParser(description="Run cognitive-load EEG model experiments.")
    parser.add_argument("--dataset", choices=["stew", "eegmat", "cog-bci"], required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--cog-paradigm", choices=["nback", "matb"], default="nback")
    parser.add_argument("--protocol", choices=["single_session", "cog_multi_session", "loso"],
                        required=True)
    parser.add_argument("--model", choices=[
        "tsmnet", "eegconformer", "eegnet", "bfgcn", "tahag", "svm",
        "mdtn", "ms_tgc_spddsbn", "lsccn", "lstm", "bilstm",
        "transformer", "shallowcnn",
    ],
                        default="tsmnet")
    parser.add_argument("--subject", type=int, default=None,
                        help="Evaluate one subject only. Default: run all subjects.")
    parser.add_argument("--cache", default=None,
                        help="Optional .npz cache for preprocessed 1 s windows. "
                             "Default: automatically named under outputs/cache/.")
    parser.add_argument("--cache-root", default=os.path.join("outputs", "cache"),
                        help="Directory used when --cache is omitted.")
    parser.add_argument("--target-fs", type=float, default=None,
                        help="Target sampling rate. Default: STEW=128 Hz, EEGMAT/COG-BCI=250 Hz.")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8,
                        help="Early-stopping patience measured in epochs.")
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
    parser.add_argument("--mdtn-num-nodes", type=int, default=0,
                        help="Cheby discriminator graph nodes. 0 means use EEG channel count.")
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
    parser.add_argument("--mstgc-num-nodes", type=int, default=0,
                        help="Cheby graph nodes for MS_TGC_SPDDSBN. 0 means use EEG channel count.")
    parser.add_argument("--svm-estimator", default="linear-svc",
                        choices=["linear-svc", "svc"],
                        help="linear-svc is the fast default; svc enables kernel SVM.")
    parser.add_argument("--svm-kernel", default="rbf",
                        choices=["linear", "poly", "rbf", "sigmoid"])
    parser.add_argument("--svm-c", type=float, default=1.0)
    parser.add_argument("--svm-gamma", default="scale")
    parser.add_argument("--svm-class-weight", default="balanced",
                        help="SVM class_weight; use 'none' to disable.")
    parser.add_argument("--svm-probability", action="store_true",
                        help="Enable SVC probability calibration. Slow; only used with --svm-estimator svc.")
    parser.add_argument("--svm-max-iter", type=int, default=5000,
                        help="Maximum iterations for LinearSVC.")
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=float, default=0.2,
                        help="Validation fraction for LOSO source subjects and COG fallback splits.")
    parser.add_argument("--single-val-size", type=float, default=0.125,
                        help="Validation fraction inside the single_session train+val block. "
                             "Default 0.125 gives train/val/test = 0.7/0.1/0.2.")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Target/test fraction for single_session sequential time-block split.")
    parser.add_argument("--min-split-windows", type=int, default=2,
                        help="Minimum split windows required for cog_multi_session quality checks.")
    parser.add_argument("--min-class-windows", type=int, default=2,
                        help="Minimum per-class windows required for cog_multi_session quality checks.")
    parser.add_argument("--allow-incomplete-splits", action="store_true",
                        help="Do not skip incomplete cog_multi_session subjects.")
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable light train-time augmentation for STEW/EEGMAT.")
    parser.add_argument("--no-target-adapt", action="store_true",
                        help="Disable unlabeled target-domain adaptation for TSMNet, BF-GCN, TAHAG, MDTN-GMDA, and MS_TGC_SPDDSBN.")
    parser.add_argument("--artifact-z", type=float, default=None,
                        help="Reject windows whose source-normalized absolute amplitude exceeds this value.")
    parser.add_argument("--no-artifact-reject", action="store_true",
                        help="Disable artifact-window rejection after source-only normalization.")
    parser.add_argument("--output", default="outputs",
                        help="Root output directory. The run subdirectory is named automatically.")
    parser.add_argument("--master-summary", default=os.path.join("outputs", "master_summary.csv"),
                        help="CSV table appended once per completed run. Pass empty string to disable.")
    return parser.parse_args()


def main():
    args = parse_args()
    load_subjects = None
    if args.subject is not None and args.protocol in ["single_session", "cog_multi_session"]:
        load_subjects = [args.subject]
    load_sessions = (1,)
    if args.dataset == "cog-bci" and args.protocol != "single_session":
        load_sessions = (1, 2, 3)
    elif args.dataset == "cog-bci" and args.protocol == "single_session":
        load_sessions = (1,)
    target_fs = default_target_fs(args.dataset, args.target_fs)
    cache = args.cache or default_cache_path(
        args.dataset,
        args.protocol,
        cog_paradigm=args.cog_paradigm,
        subject=args.subject,
        target_fs=target_fs,
        cache_root=args.cache_root,
    )
    print("Using cache:", cache)

    dataset = load_dataset(
        args.dataset,
        data_root=args.data_root,
        cache=cache,
        rebuild_cache=args.rebuild_cache,
        cog_paradigm=args.cog_paradigm,
        subjects=load_subjects,
        sessions=load_sessions,
        target_fs=target_fs,
        window_sec=1.0,
        stride_sec=1.0,
    )
    domains = domain_ids(dataset, args.protocol)
    subjects = [args.subject] if args.subject is not None else iter_eval_subjects(
        dataset["meta"], args.protocol, dataset["name"])
    augment = (args.dataset in ["stew", "eegmat"]) and (not args.no_augment)
    target_adapt = (not args.no_target_adapt) and args.model in [
        "tsmnet", "bfgcn", "tahag", "mdtn", "ms_tgc_spddsbn"
    ]
    artifact_z = None if args.no_artifact_reject else args.artifact_z

    run_name = run_directory_name(dataset["name"], args.protocol, args.model, args.bnorm)
    out_root = os.path.join(args.output, run_name)
    if not os.path.exists(out_root):
        os.makedirs(out_root)

    results = []
    model_name = _model_label(args)
    project_root = os.path.abspath(os.path.dirname(__file__))
    for subject in subjects:
        split_val_size = args.single_val_size if args.protocol == "single_session" else args.val_size
        split = make_split(dataset, args.protocol, subject, seed=args.seed,
                           val_size=split_val_size, test_size=args.test_size)
        split_issues = []
        if args.protocol == "cog_multi_session":
            split_issues = split_validation_issues(
                dataset,
                split,
                min_windows=args.min_split_windows,
                min_class_windows=args.min_class_windows,
                require_all_classes=True,
            )
            if split_issues and not args.allow_incomplete_splits:
                print("Skipping subject {} due to incomplete cog_multi_session split: {}".format(
                    int(subject), "; ".join(split_issues)
                ))
                continue
        subject_dir = os.path.join(out_root, "subject_{:02d}".format(int(subject)))
        res = train_one_split(
            dataset=dataset,
            domains=domains,
            split=split,
            project_root=project_root,
            output_dir=subject_dir,
            epochs=args.epochs,
            patience=args.patience,
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
            conformer_emb_size=args.conformer_emb_size,
            conformer_depth=args.conformer_depth,
            conformer_num_heads=args.conformer_num_heads,
            conformer_dropout=args.conformer_dropout,
            conformer_classifier_hidden=args.conformer_classifier_hidden,
            seed=args.seed + int(subject),
            target_adapt=target_adapt,
            artifact_z=artifact_z,
            eegnet_temporal_filters=args.eegnet_temporal_filters,
            eegnet_spatial_filters=args.eegnet_spatial_filters,
            eegnet_dropout=args.eegnet_dropout,
            eegnet_avgpool_factor=args.eegnet_avgpool_factor,
            bfgcn_kadj=args.bfgcn_kadj,
            bfgcn_num_out=args.bfgcn_num_out,
            bfgcn_att_hidden=args.bfgcn_att_hidden,
            bfgcn_classifier_hidden=args.bfgcn_classifier_hidden,
            bfgcn_avgpool=args.bfgcn_avgpool,
            bfgcn_dropout=args.bfgcn_dropout,
            bfgcn_domain_weight=args.bfgcn_domain_weight,
            tahag_dropout=args.tahag_dropout,
            tahag_domain_weight=args.tahag_domain_weight,
            tahag_mmd_weight=args.tahag_mmd_weight,
            tahag_adaptive=not args.no_tahag_adaptive,
            tahag_attention=not args.no_tahag_attention,
            svm_estimator=args.svm_estimator,
            svm_kernel=args.svm_kernel,
            svm_c=args.svm_c,
            svm_gamma=args.svm_gamma,
            svm_class_weight=args.svm_class_weight,
            svm_probability=args.svm_probability,
            svm_max_iter=args.svm_max_iter,
            lsccn_latent_dim=args.lsccn_latent_dim,
            lsccn_routing_iters=args.lsccn_routing_iters,
            lsccn_recon_weight=args.lsccn_recon_weight,
            lsccn_kl_weight=args.lsccn_kl_weight,
            mdtn_hidden_dim=args.mdtn_hidden_dim,
            mdtn_num_nodes=args.mdtn_num_nodes,
            mdtn_kernel_length=args.mdtn_kernel_length,
            mdtn_num_heads=args.mdtn_num_heads,
            mdtn_cheby_order=args.mdtn_cheby_order,
            mdtn_dropout=args.mdtn_dropout,
            mdtn_lambda_match=args.mdtn_lambda_match,
            mdtn_marginal_weight=args.mdtn_marginal_weight,
            mdtn_conditional_weight=args.mdtn_conditional_weight,
            mdtn_l1_weight=args.mdtn_l1_weight,
            mstgc_temporal_hidden=args.mstgc_temporal_hidden,
            mstgc_graph_hidden=args.mstgc_graph_hidden,
            mstgc_fusion_dim=args.mstgc_fusion_dim,
            mstgc_kernel_length=args.mstgc_kernel_length,
            mstgc_num_heads=args.mstgc_num_heads,
            mstgc_cheby_order=args.mstgc_cheby_order,
            mstgc_dropout=args.mstgc_dropout,
            mstgc_num_nodes=args.mstgc_num_nodes,
            recurrent_hidden=args.recurrent_hidden,
            recurrent_layers=args.recurrent_layers,
            recurrent_dropout=args.recurrent_dropout,
            transformer_d_model=args.transformer_d_model,
            transformer_heads=args.transformer_heads,
            transformer_layers=args.transformer_layers,
            transformer_ff=args.transformer_ff,
            transformer_dropout=args.transformer_dropout,
            shallow_filters=args.shallow_filters,
            shallow_kernel=args.shallow_kernel,
            shallow_pool=args.shallow_pool,
            shallow_dropout=args.shallow_dropout,
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
            "decision_threshold": res.get("decision_threshold", ""),
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
            "val_size": split_val_size,
            "test_size": args.test_size if args.protocol == "single_session" else "",
            "split_issues": "; ".join(split_issues),
        }
        results.append(row)
        history = pd.DataFrame(res["history"])
        if len(history):
            history["is_best_epoch"] = history["epoch"] == res["best_epoch"]
        history.to_csv(os.path.join(subject_dir, "history.csv"), index=False)
        print(row)

    result_path = os.path.join(out_root, "summary.csv")
    if not results:
        raise RuntimeError("No valid subjects were evaluated. Check split quality settings.")
    with open(result_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print("Saved:", result_path)
    aggregate_path = os.path.join(out_root, "aggregate_summary.csv")
    write_aggregate_summary(results, aggregate_path)
    print("Saved:", aggregate_path)
    if args.master_summary:
        append_master_summary(results, args.master_summary, {
            "cog_paradigm": args.cog_paradigm if args.dataset == "cog-bci" else "",
            "target_fs": target_fs,
            "cache": cache,
            "output_dir": out_root,
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "target_adapt": target_adapt,
            "augment": augment,
            "artifact_z": "" if artifact_z is None else artifact_z,
            "eegnet_temporal_filters": args.eegnet_temporal_filters if args.model == "eegnet" else "",
            "eegnet_spatial_filters": args.eegnet_spatial_filters if args.model == "eegnet" else "",
            "eegnet_dropout": args.eegnet_dropout if args.model == "eegnet" else "",
            "eegnet_avgpool_factor": args.eegnet_avgpool_factor if args.model == "eegnet" else "",
            "bfgcn_kadj": args.bfgcn_kadj if args.model == "bfgcn" else "",
            "bfgcn_num_out": args.bfgcn_num_out if args.model == "bfgcn" else "",
            "bfgcn_att_hidden": args.bfgcn_att_hidden if args.model == "bfgcn" else "",
            "bfgcn_classifier_hidden": args.bfgcn_classifier_hidden if args.model == "bfgcn" else "",
            "bfgcn_avgpool": args.bfgcn_avgpool if args.model == "bfgcn" else "",
            "bfgcn_dropout": args.bfgcn_dropout if args.model == "bfgcn" else "",
            "bfgcn_domain_weight": args.bfgcn_domain_weight if args.model == "bfgcn" else "",
            "tahag_dropout": args.tahag_dropout if args.model == "tahag" else "",
            "tahag_domain_weight": args.tahag_domain_weight if args.model == "tahag" else "",
            "tahag_mmd_weight": args.tahag_mmd_weight if args.model == "tahag" else "",
            "tahag_adaptive": (not args.no_tahag_adaptive) if args.model == "tahag" else "",
            "tahag_attention": (not args.no_tahag_attention) if args.model == "tahag" else "",
            "mdtn_hidden_dim": args.mdtn_hidden_dim if args.model == "mdtn" else "",
            "mdtn_num_nodes": args.mdtn_num_nodes if args.model == "mdtn" else "",
            "mdtn_kernel_length": args.mdtn_kernel_length if args.model == "mdtn" else "",
            "mdtn_num_heads": args.mdtn_num_heads if args.model == "mdtn" else "",
            "mdtn_cheby_order": args.mdtn_cheby_order if args.model == "mdtn" else "",
            "mdtn_dropout": args.mdtn_dropout if args.model == "mdtn" else "",
            "mdtn_lambda_match": args.mdtn_lambda_match if args.model == "mdtn" else "",
            "mdtn_marginal_weight": args.mdtn_marginal_weight if args.model == "mdtn" else "",
            "mdtn_conditional_weight": args.mdtn_conditional_weight if args.model == "mdtn" else "",
            "mdtn_l1_weight": args.mdtn_l1_weight if args.model == "mdtn" else "",
            "mstgc_temporal_hidden": args.mstgc_temporal_hidden if args.model == "ms_tgc_spddsbn" else "",
            "mstgc_graph_hidden": args.mstgc_graph_hidden if args.model == "ms_tgc_spddsbn" else "",
            "mstgc_fusion_dim": args.mstgc_fusion_dim if args.model == "ms_tgc_spddsbn" else "",
            "mstgc_kernel_length": args.mstgc_kernel_length if args.model == "ms_tgc_spddsbn" else "",
            "mstgc_num_heads": args.mstgc_num_heads if args.model == "ms_tgc_spddsbn" else "",
            "mstgc_cheby_order": args.mstgc_cheby_order if args.model == "ms_tgc_spddsbn" else "",
            "mstgc_dropout": args.mstgc_dropout if args.model == "ms_tgc_spddsbn" else "",
            "mstgc_num_nodes": args.mstgc_num_nodes if args.model == "ms_tgc_spddsbn" else "",
            "svm_estimator": args.svm_estimator if args.model == "svm" else "",
            "svm_kernel": args.svm_kernel if args.model == "svm" else "",
            "svm_c": args.svm_c if args.model == "svm" else "",
            "svm_gamma": args.svm_gamma if args.model == "svm" else "",
            "svm_class_weight": args.svm_class_weight if args.model == "svm" else "",
            "svm_probability": args.svm_probability if args.model == "svm" else "",
            "svm_max_iter": args.svm_max_iter if args.model == "svm" else "",
            "lsccn_latent_dim": args.lsccn_latent_dim if args.model == "lsccn" else "",
            "lsccn_routing_iters": args.lsccn_routing_iters if args.model == "lsccn" else "",
            "lsccn_recon_weight": args.lsccn_recon_weight if args.model == "lsccn" else "",
            "lsccn_kl_weight": args.lsccn_kl_weight if args.model == "lsccn" else "",
            "recurrent_hidden": args.recurrent_hidden if args.model in ["lstm", "bilstm"] else "",
            "recurrent_layers": args.recurrent_layers if args.model in ["lstm", "bilstm"] else "",
            "recurrent_dropout": args.recurrent_dropout if args.model in ["lstm", "bilstm"] else "",
            "transformer_d_model": args.transformer_d_model if args.model == "transformer" else "",
            "transformer_heads": args.transformer_heads if args.model == "transformer" else "",
            "transformer_layers": args.transformer_layers if args.model == "transformer" else "",
            "transformer_ff": args.transformer_ff if args.model == "transformer" else "",
            "transformer_dropout": args.transformer_dropout if args.model == "transformer" else "",
            "shallow_filters": args.shallow_filters if args.model == "shallowcnn" else "",
            "shallow_kernel": args.shallow_kernel if args.model == "shallowcnn" else "",
            "shallow_pool": args.shallow_pool if args.model == "shallowcnn" else "",
            "shallow_dropout": args.shallow_dropout if args.model == "shallowcnn" else "",
            "val_size": args.val_size,
            "single_val_size": args.single_val_size,
            "test_size": args.test_size,
            "min_split_windows": args.min_split_windows,
            "min_class_windows": args.min_class_windows,
            "allow_incomplete_splits": args.allow_incomplete_splits,
        })
        print("Updated:", args.master_summary)


if __name__ == "__main__":
    main()
