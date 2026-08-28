"""Generate paper figures B3/B4 in one leakage-safe common SPD tangent space."""

import argparse
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch

from src.cl_tsmnet.datasets import load_dataset
from src.cl_tsmnet.experiment_utils import default_cache_path, default_target_fs
from src.cl_tsmnet.spd_pca import (
    airm_karcher_mean,
    alignment_metrics,
    balanced_plot_manifest,
    choose_median_fold,
    common_tangent_vectors,
    migrate_legacy_spddsbn_buffers,
    validate_spd_matrices,
)
from src.cl_tsmnet.spd_visualization_adapters import (
    SUPPORTED_SPD_VISUALIZATION_MODELS,
    extract_spd_intermediates,
    visualization_model_metadata,
)
from src.cl_tsmnet.splits import domain_ids, make_split
from src.cl_tsmnet.training import (
    _filter_artifact_windows,
    build_ms_tgc_spddsbn,
    build_tsmnet,
    fit_source_normalizer,
)


DOMAIN_COLORS = {"source": "#4C78A8", "target": "#E45756"}
CLASS_COLORS = ["#3B75AF", "#D94F45", "#E3A12F", "#6F4E9C", "#2A9D8F"]
MODEL_TYPES = SUPPORTED_SPD_VISUALIZATION_MODELS
DEFAULT_MODEL_CONFIG = {
    "temporal_filters": 4,
    "spatial_filters": 40,
    "subspacedims": 20,
    "temp_kernel": 25,
    "mstgc_temporal_hidden": 64,
    "mstgc_graph_hidden": 64,
    "mstgc_fusion_dim": 128,
    "mstgc_kernel_length": 16,
    "mstgc_num_heads": 4,
    "mstgc_cheby_order": 3,
    "mstgc_dropout": 0.5,
    "mstgc_num_nodes": 0,
    "mstgc_graph_k": 4,
    "mstgc_graph_density": None,
    "mstgc_time_points": 64,
    "mstgc_shrinkage": 0.1,
    "seed": 42,
    "val_size": 0.2,
    "test_size": 0.2,
    "artifact_z": None,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize TSMNet/MS-TGC SPD features before/after SPDDSBN."
    )
    parser.add_argument("--dataset", choices=["stew", "eegmat", "cog-bci"], required=True)
    parser.add_argument("--cog-paradigm", choices=["nback", "matb"], default="nback")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--cache", default=None)
    parser.add_argument("--cache-root", default=os.path.join("outputs", "cache"))
    parser.add_argument("--target-fs", type=float, default=None)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--results-file", default=None)
    parser.add_argument("--master-summary", default=None)
    parser.add_argument("--output-dir", default=os.path.join(
        "results", "figures", "manifold_alignment"
    ))
    parser.add_argument("--mode", choices=["auto-median-fold", "target-subject"],
                        default="auto-median-fold")
    parser.add_argument("--target-subject", type=int, default=None)
    parser.add_argument("--model", choices=sorted(MODEL_TYPES), default="ms_tgc_spddsbn")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--plot-seed", type=int, default=2026)
    parser.add_argument("--max-points-per-class-domain", type=int, default=500)
    parser.add_argument("--annotate-metrics", action="store_true")
    parser.add_argument("--force-reextract", action="store_true")
    parser.add_argument("--no-ellipses", action="store_true")
    parser.add_argument("--karcher-tol", type=float, default=1e-7)
    parser.add_argument("--karcher-max-iter", type=int, default=50)
    parser.add_argument("--riemann-batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--val-size", type=float, default=None)
    parser.add_argument("--test-size", type=float, default=None)
    parser.add_argument("--artifact-z", type=float, default=None)
    parser.add_argument("--temporal-filters", type=int, default=None)
    parser.add_argument("--spatial-filters", type=int, default=None)
    parser.add_argument("--subspacedims", type=int, default=None)
    parser.add_argument("--temp-kernel", type=int, default=None)
    parser.add_argument("--mstgc-temporal-hidden", type=int, default=None)
    parser.add_argument("--mstgc-graph-hidden", type=int, default=None)
    parser.add_argument("--mstgc-fusion-dim", type=int, default=None)
    parser.add_argument("--mstgc-kernel-length", type=int, default=None)
    parser.add_argument("--mstgc-num-heads", type=int, default=None)
    parser.add_argument("--mstgc-cheby-order", type=int, default=None)
    parser.add_argument("--mstgc-dropout", type=float, default=None)
    parser.add_argument("--mstgc-num-nodes", type=int, default=None)
    parser.add_argument("--mstgc-graph-k", type=int, default=None)
    parser.add_argument("--mstgc-graph-density", type=float, default=None)
    parser.add_argument("--mstgc-time-points", type=int, default=None)
    parser.add_argument("--mstgc-shrinkage", type=float, default=None)
    return parser.parse_args()


def _json_dump(payload, path):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)


def _normalize_path(path):
    return os.path.normcase(os.path.abspath(os.path.normpath(str(path))))


def _read_csv(path, purpose):
    if not path or not os.path.exists(path):
        raise FileNotFoundError("{} was not found: {}".format(purpose, path))
    return pd.read_csv(path)


def _candidate_master_summary(checkpoint_root, explicit):
    if explicit:
        return explicit
    root = checkpoint_root if os.path.isdir(checkpoint_root) else os.path.dirname(checkpoint_root)
    candidates = [
        os.path.join(os.path.dirname(root), "master_summary.csv"),
        os.path.join("outputs", "master_summary.csv"),
    ]
    return next((path for path in candidates if os.path.exists(path)), None)


def _matching_run_record(master_path, checkpoint_root, dataset_name, model):
    if not master_path or not os.path.exists(master_path):
        return {}
    frame = pd.read_csv(master_path)
    if "protocol" in frame:
        frame = frame[frame["protocol"].astype(str) == "loso"]
    if "model_type" in frame:
        frame = frame[frame["model_type"].astype(str) == model]
    if "dataset" in frame:
        frame = frame[frame["dataset"].astype(str) == dataset_name]
    if frame.empty:
        return {}
    if "output_dir" in frame:
        exact = frame[frame["output_dir"].astype(str).map(
            lambda value: _normalize_path(value) == _normalize_path(checkpoint_root)
        )]
        if not exact.empty:
            frame = exact
    return frame.iloc[-1].dropna().to_dict()


def _coerce(value, default):
    if value is None or (isinstance(value, str) and not value.strip()):
        return default
    if isinstance(default, bool):
        return str(value).lower() in {"1", "true", "yes"}
    if default is None:
        return value
    return type(default)(value)


def _resolve_config(args, run_record, selected_result):
    config = dict(DEFAULT_MODEL_CONFIG)
    for key in config:
        cli_value = getattr(args, key, None)
        source_value = cli_value if cli_value is not None else run_record.get(key)
        if key == "artifact_z" and source_value is None:
            source_value = selected_result.get("artifact_z")
        config[key] = _coerce(source_value, config[key])
    if config["mstgc_graph_density"] is not None:
        config["mstgc_graph_density"] = float(config["mstgc_graph_density"])
    if config["artifact_z"] is not None:
        config["artifact_z"] = float(config["artifact_z"])
    config["seed"] = int(config["seed"])
    config["val_size"] = float(config["val_size"])
    config["test_size"] = float(config["test_size"])
    return config


def _select_fold(args, dataset_name):
    results_path = args.results_file or os.path.join(args.checkpoint_root, "summary.csv")
    if args.mode == "auto-median-fold":
        results = _read_csv(results_path, "Per-subject LOSO results")
        if "protocol" in results:
            results = results[results["protocol"].astype(str) == "loso"]
        if "dataset" in results:
            results = results[results["dataset"].astype(str) == dataset_name]
        if "model_type" in results:
            results = results[results["model_type"].astype(str) == args.model]
        if results.empty:
            raise ValueError(
                "No matching {} LOSO rows were found in {}".format(
                    args.model, results_path
                )
            )
        selection = choose_median_fold(results)
        selection["dataset"] = dataset_name
        selection["results_file"] = os.path.abspath(results_path)
        subject = selection["selected_target_subject"]
        selected_rows = results[pd.to_numeric(results["subject"]) == subject]
        selected_result = selected_rows.iloc[-1].dropna().to_dict()
        return subject, selection, selected_result
    if args.target_subject is None:
        raise ValueError("--mode target-subject requires --target-subject")
    selection = {
        "dataset": dataset_name,
        "selected_target_subject": int(args.target_subject),
        "subject_balanced_accuracy": None,
        "all_subject_median_balanced_accuracy": None,
        "selection_rule": "explicit_target_subject",
        "results_file": os.path.abspath(results_path) if os.path.exists(results_path) else None,
    }
    selected_result = {}
    if os.path.exists(results_path):
        results = pd.read_csv(results_path)
        rows = results[pd.to_numeric(results["subject"], errors="coerce") == args.target_subject]
        if not rows.empty:
            selected_result = rows.iloc[-1].dropna().to_dict()
    return int(args.target_subject), selection, selected_result


def _checkpoint_path(root, subject):
    if os.path.isfile(root):
        return os.path.abspath(root)
    candidates = [
        os.path.join(root, "subject_{:02d}".format(subject), "model.pt"),
        os.path.join(root, "subject_{}".format(subject), "model.pt"),
        os.path.join(root, "model.pt"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)
    raise FileNotFoundError(
        "No checkpoint was found for target subject {} under {}. Expected {}"
        .format(subject, root, candidates[:2])
    )


def _load_state(path):
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError("Checkpoint does not contain a PyTorch state_dict")
    return state


def _dataset_name(args):
    return "cog-bci-{}".format(args.cog_paradigm) if args.dataset == "cog-bci" else args.dataset


def _feature_signature(checkpoint, cache, subject, config, model_type):
    stat = os.stat(checkpoint)
    return {
        "checkpoint_path": os.path.abspath(checkpoint),
        "checkpoint_size": int(stat.st_size),
        "checkpoint_mtime_ns": int(stat.st_mtime_ns),
        "cache_path": os.path.abspath(cache),
        "target_subject": int(subject),
        "model_type": str(model_type),
        "model_config": config,
        "split_seed": int(config["seed"]),
        "val_size": float(config["val_size"]),
        "test_size": float(config["test_size"]),
        "artifact_z": config["artifact_z"],
    }


def _extract_features(model, model_type, dataset, indices, domains, normalizer,
                      split_name, batch_size, device):
    pre_parts, post_parts = [], []
    indices = np.asarray(indices, dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), int(batch_size)):
            batch_indices = indices[start:start + int(batch_size)]
            windows = normalizer.transform_array(dataset["x"][batch_indices])
            xb = torch.from_numpy(windows).to(device=device, dtype=torch.float32)
            db = torch.from_numpy(domains[batch_indices]).to(device=device, dtype=torch.long)
            _, intermediates = extract_spd_intermediates(
                model, xb, db, model_type
            )
            pre_parts.append(intermediates["spd_pre_bn"].detach().cpu().double().numpy())
            post_parts.append(intermediates["spd_post_bn"].detach().cpu().double().numpy())
    pre = np.concatenate(pre_parts, axis=0)
    post = np.concatenate(post_parts, axis=0)
    if pre.shape != post.shape or len(pre) != len(indices):
        raise RuntimeError("{} pre/post feature alignment failed".format(split_name))
    return pre, post


def _extract_or_load(args, dataset, split, domains, normalizer, model, checkpoint,
                     cache, subject, config, device):
    feature_path = os.path.join(args.output_dir, "spd_intermediates.npz")
    metadata_path = os.path.join(args.output_dir, "spd_intermediates_metadata.csv")
    signature_path = os.path.join(args.output_dir, "extraction_signature.json")
    signature = _feature_signature(
        checkpoint, cache, subject, config, args.model
    )
    can_reuse = all(os.path.exists(path) for path in [
        feature_path, metadata_path, signature_path
    ]) and not args.force_reextract
    if can_reuse:
        with open(signature_path, encoding="utf-8") as handle:
            can_reuse = json.load(handle) == signature
    if can_reuse:
        with np.load(feature_path) as saved:
            pre = saved["spd_pre_bn"].astype(np.float64)
            post = saved["spd_post_bn"].astype(np.float64)
            sample_ids = saved["sample_id"].astype(np.int64)
        metadata = pd.read_csv(metadata_path)
        if not np.array_equal(sample_ids, metadata["sample_id"].to_numpy(dtype=np.int64)):
            raise RuntimeError("Cached feature sample IDs do not match metadata")
        return pre, post, metadata, True

    source_ids = np.asarray(split["train"], dtype=np.int64)
    target_ids = np.asarray(split["test"], dtype=np.int64)
    if np.intersect1d(source_ids, target_ids).size:
        raise RuntimeError("Source-train and target-test sample IDs overlap")
    source_pre, source_post = _extract_features(
        model, args.model, dataset, source_ids, domains, normalizer, "source_train",
        args.batch_size, device,
    )
    target_pre, target_post = _extract_features(
        model, args.model, dataset, target_ids, domains, normalizer, "target_test",
        args.batch_size, device,
    )
    sample_ids = np.concatenate([source_ids, target_ids])
    pre = np.concatenate([source_pre, target_pre], axis=0)
    post = np.concatenate([source_post, target_post], axis=0)
    meta = dataset["meta"].iloc[sample_ids]
    labels = dataset["y"][sample_ids].astype(np.int64)
    label_names = dataset.get("label_names", {})
    metadata = pd.DataFrame({
        "sample_id": sample_ids,
        "subject_id": meta["subject"].to_numpy(dtype=np.int64),
        "domain_id": domains[sample_ids].astype(np.int64),
        "domain": np.where(np.isin(sample_ids, source_ids), "source", "target"),
        "class_id": labels,
        "class_name": [str(label_names.get(int(v), "class {}".format(v))) for v in labels],
        "split": np.where(np.isin(sample_ids, source_ids), "source_train", "target_test"),
        "session": meta["session"].to_numpy(dtype=np.int64),
        "task": meta["task"].astype(str).to_numpy(),
        "start_sample": meta["start_sample"].to_numpy(dtype=np.int64),
    })
    np.savez_compressed(
        feature_path,
        sample_id=sample_ids,
        spd_pre_bn=pre,
        spd_post_bn=post,
    )
    metadata.to_csv(metadata_path, index=False)
    _json_dump(signature, signature_path)
    return pre, post, metadata, False


def _font_family():
    for family in ["Arial", "Helvetica", "DejaVu Sans"]:
        try:
            font_manager.findfont(family, fallback_to_default=False)
            return family
        except ValueError:
            continue
    return "DejaVu Sans"


def _axis_limits(pre_coordinates, post_coordinates):
    all_coordinates = np.vstack([pre_coordinates, post_coordinates])
    limits = []
    for axis in range(2):
        low, high = np.percentile(all_coordinates[:, axis], [1.0, 99.0])
        span = float(high - low)
        if span <= 0.0:
            span = max(1.0, abs(float(low)) * 0.1)
        limits.append((float(low - 0.08 * span), float(high + 0.08 * span)))
    return limits


def _ellipse(ax, points, color):
    if len(points) < 5:
        return
    covariance = np.cov(points, rowvar=False)
    if not np.isfinite(covariance).all():
        return
    values, vectors = np.linalg.eigh(covariance)
    if np.min(values) <= 0.0:
        return
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    angle = np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0]))
    scale = np.sqrt(4.605170186)
    patch = Ellipse(
        points.mean(axis=0), width=2 * scale * np.sqrt(values[0]),
        height=2 * scale * np.sqrt(values[1]), angle=angle,
        facecolor=color, edgecolor=color, linewidth=0.7, alpha=0.08,
        zorder=1,
    )
    ax.add_patch(patch)


def _draw_panel(ax, coordinates, metadata, color_mode, title, panel_label,
                axis_limits, variance, ellipses, metrics=None):
    point_size = 22 if len(metadata) <= 1000 else 18
    if color_mode == "domain":
        groups = [(value, value.title(), DOMAIN_COLORS[value],
                   "o" if value == "source" else "^")
                  for value in ["source", "target"]]
        field = "domain"
    else:
        class_rows = metadata[["class_id", "class_name"]].drop_duplicates().sort_values("class_id")
        groups = [
            (int(row.class_id), str(row.class_name), CLASS_COLORS[index % len(CLASS_COLORS)], "o")
            for index, row in enumerate(class_rows.itertuples(index=False))
        ]
        field = "class_id"
    handles = []
    for value, label, color, marker in groups:
        mask = metadata[field].to_numpy() == value
        points = coordinates[mask]
        if ellipses:
            _ellipse(ax, points, color)
        ax.scatter(
            points[:, 0], points[:, 1], s=point_size, marker=marker,
            c=color, alpha=0.66, edgecolors="white", linewidths=0.2,
            zorder=2,
        )
        center = points.mean(axis=0)
        ax.scatter(
            center[0], center[1], s=58, marker="X", c=color,
            edgecolors="white", linewidths=0.6, zorder=4,
        )
        handles.append(Line2D(
            [0], [0], marker=marker, linestyle="None", label=label,
            markerfacecolor=color, markeredgecolor="white", markersize=5.5,
        ))
    ax.set_title(title, fontsize=9.5, pad=5)
    ax.text(-0.13, 1.06, panel_label, transform=ax.transAxes,
            fontsize=9.5, fontweight="bold", va="top")
    ax.set_xlim(*axis_limits[0])
    ax.set_ylim(*axis_limits[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("PC1 ({:.1f}% variance)".format(100.0 * variance[0]))
    ax.set_ylabel("PC2 ({:.1f}% variance)".format(100.0 * variance[1]))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(width=0.7, length=3, labelsize=7.5)
    ax.legend(handles=handles, frameon=False, fontsize=7.5, ncol=len(handles),
              loc="upper center", bbox_to_anchor=(0.5, 1.02))
    if metrics:
        ax.text(0.02, 0.02, metrics, transform=ax.transAxes, fontsize=7,
                va="bottom", ha="left")


def _save_figure(fig, stem):
    metadata = {"Creator": "visualize_spddsbn_pca.py"}
    fig.savefig(stem + ".pdf", bbox_inches="tight", metadata=metadata)
    fig.savefig(stem + ".svg", bbox_inches="tight", metadata=metadata)
    fig.savefig(stem + ".png", dpi=600, bbox_inches="tight", metadata=metadata)
    plt.close(fig)


def _make_figures(output_dir, pre_coordinates, post_coordinates, metadata,
                  manifest, variance, metrics, annotate, ellipses):
    order = metadata.set_index("sample_id")
    positions = {int(sample_id): index for index, sample_id in enumerate(metadata["sample_id"])}
    selected_positions = np.asarray([
        positions[int(value)] for value in manifest["sample_id"]
    ], dtype=np.int64)
    plot_meta = order.loc[manifest["sample_id"].to_numpy()].reset_index()
    pre = pre_coordinates[selected_positions]
    post = post_coordinates[selected_positions]
    if not np.array_equal(
            plot_meta["sample_id"].to_numpy(), manifest["sample_id"].to_numpy()):
        raise RuntimeError("B3/B4 sample manifest order mismatch")
    limits = _axis_limits(pre, post)
    clipped = {
        "pre": int(np.sum(
            (pre[:, 0] < limits[0][0]) | (pre[:, 0] > limits[0][1])
            | (pre[:, 1] < limits[1][0]) | (pre[:, 1] > limits[1][1])
        )),
        "post": int(np.sum(
            (post[:, 0] < limits[0][0]) | (post[:, 0] > limits[0][1])
            | (post[:, 1] < limits[1][0]) | (post[:, 1] > limits[1][1])
        )),
    }
    domain_annotations = None
    class_annotations = None
    if annotate:
        domain_annotations = [
            "Domain distance: {:.3f}".format(metrics["domain_centroid_distance_pre"]),
            "Domain distance: {:.3f}".format(metrics["domain_centroid_distance_post"]),
        ]
        class_annotations = [
            "Fisher ratio: {:.3f}".format(metrics["fisher_ratio_pre"]),
            "Fisher ratio: {:.3f}".format(metrics["fisher_ratio_post"]),
        ]

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.2), constrained_layout=True)
    _draw_panel(axes[0], pre, plot_meta, "domain", "Before SPDDSBN", "(a)",
                limits, variance, ellipses, None if not annotate else domain_annotations[0])
    _draw_panel(axes[1], post, plot_meta, "domain", "After SPDDSBN", "(b)",
                limits, variance, ellipses, None if not annotate else domain_annotations[1])
    _save_figure(fig, os.path.join(output_dir, "fig_B3_domain_alignment"))

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.2), constrained_layout=True)
    _draw_panel(axes[0], pre, plot_meta, "class", "Before SPDDSBN", "(a)",
                limits, variance, ellipses, None if not annotate else class_annotations[0])
    _draw_panel(axes[1], post, plot_meta, "class", "After SPDDSBN", "(b)",
                limits, variance, ellipses, None if not annotate else class_annotations[1])
    _save_figure(fig, os.path.join(output_dir, "fig_B4_class_structure"))

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), constrained_layout=True)
    settings = [
        (axes[0, 0], pre, "domain", "Before SPDDSBN", "(a)", domain_annotations, 0),
        (axes[0, 1], post, "domain", "After SPDDSBN", "(b)", domain_annotations, 1),
        (axes[1, 0], pre, "class", "Before SPDDSBN", "(c)", class_annotations, 0),
        (axes[1, 1], post, "class", "After SPDDSBN", "(d)", class_annotations, 1),
    ]
    for ax, coords, mode, title, label, annotations, index in settings:
        annotation = None if not annotate else annotations[index]
        _draw_panel(ax, coords, plot_meta, mode, title, label, limits,
                    variance, ellipses, annotation)
    _save_figure(fig, os.path.join(output_dir, "fig_B3_B4_combined"))
    return limits, clipped


def _git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], universal_newlines=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    args = parse_args()
    if args.max_points_per_class_domain < 1:
        raise ValueError("--max-points-per-class-domain must be positive")
    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.plot_seed)
    np.random.seed(args.plot_seed)
    torch.manual_seed(args.plot_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.plot_seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    matplotlib.rcParams.update({
        "font.family": _font_family(),
        "font.size": 8.0,
        "axes.labelsize": 8.0,
        "axes.titlesize": 9.5,
        "axes.linewidth": 0.8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "svg.hashsalt": "spddsbn-pca-2026",
    })
    print("Matplotlib font family:", matplotlib.rcParams["font.family"])

    dataset_name = _dataset_name(args)
    subject, selection, selected_result = _select_fold(args, dataset_name)
    checkpoint = _checkpoint_path(args.checkpoint_root, subject)
    master_path = _candidate_master_summary(args.checkpoint_root, args.master_summary)
    run_record = _matching_run_record(master_path, args.checkpoint_root,
                                      dataset_name, args.model)
    config = _resolve_config(args, run_record, selected_result)
    if selected_result and str(selected_result.get("target_adapt", "True")).lower() in {
            "false", "0", "no"}:
        raise ValueError("Selected checkpoint was evaluated without target SPDDSBN adaptation")
    if (
        args.model == "tsmnet"
        and selected_result.get("bnorm") not in [None, "", "spddsbn"]
    ):
        raise ValueError(
            "TSMNet B3/B4 requires an SPDDSBN checkpoint, but summary.csv "
            "reports bnorm={!r}".format(selected_result.get("bnorm"))
        )
    if selected_result.get("target_refit_scope") not in [None, "", "target_only"]:
        raise ValueError(
            "Selected checkpoint has unexpected target_refit_scope={!r}".format(
                selected_result.get("target_refit_scope")
            )
        )
    _json_dump(selection, os.path.join(args.output_dir, "representative_fold.json"))

    target_fs = args.target_fs
    if target_fs is None and run_record.get("target_fs") not in [None, ""]:
        target_fs = float(run_record["target_fs"])
    target_fs = default_target_fs(args.dataset, target_fs)
    cache = args.cache or default_cache_path(
        args.dataset, "loso", cog_paradigm=args.cog_paradigm,
        target_fs=target_fs, cache_root=args.cache_root,
    )
    sessions = (1, 2, 3) if args.dataset == "cog-bci" else (1,)
    dataset = load_dataset(
        args.dataset, data_root=args.data_root, cache=cache,
        rebuild_cache=args.rebuild_cache, cog_paradigm=args.cog_paradigm,
        sessions=sessions, target_fs=target_fs,
    )
    if subject not in set(int(v) for v in dataset["meta"]["subject"].unique()):
        raise ValueError("Target subject {} is absent from {}".format(subject, cache))
    split = make_split(
        dataset, "loso", subject, seed=config["seed"],
        val_size=config["val_size"], test_size=config["test_size"],
    )
    normalizer = fit_source_normalizer(dataset["x"], split["train"])
    split = dict(split)
    for name in ["train", "val", "test"]:
        split[name] = _filter_artifact_windows(
            dataset["x"], split[name], normalizer, config["artifact_z"]
        )
        if len(split[name]) == 0:
            raise RuntimeError("Artifact filtering removed all {} windows".format(name))
    domains = domain_ids(dataset, "loso")
    selected = np.concatenate([split["train"], split["val"], split["test"]])
    train_labels = np.unique(dataset["y"][split["train"]]).astype(np.int64)
    if not np.array_equal(train_labels, np.arange(len(train_labels))):
        raise ValueError("Source training labels are not contiguous 0..K-1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model == "tsmnet":
        model = build_tsmnet(
            os.getcwd(), dataset["x"].shape[1], dataset["x"].shape[2],
            len(train_labels), domains[selected], bnorm="spddsbn",
            temporal_filters=config["temporal_filters"],
            spatial_filters=config["spatial_filters"],
            subspacedims=config["subspacedims"],
            temp_kernel=config["temp_kernel"], device=device,
        ).to(device)
    else:
        kernel_samples = max(3, int(round(
            float(config["mstgc_kernel_length"]) * float(dataset["fs"]) / 128.0
        )))
        model = build_ms_tgc_spddsbn(
            os.getcwd(), dataset["x"].shape[1], dataset["x"].shape[2],
            len(train_labels), domains[selected],
            subspacedims=config["subspacedims"], device=device,
            temporal_hidden=config["mstgc_temporal_hidden"],
            graph_hidden=config["mstgc_graph_hidden"],
            fusion_dim=config["mstgc_fusion_dim"],
            kernel_length=kernel_samples,
            num_heads=config["mstgc_num_heads"],
            cheby_order=config["mstgc_cheby_order"],
            dropout=config["mstgc_dropout"],
            num_nodes=config["mstgc_num_nodes"],
            variant=args.model,
            graph_mode="adaptive",
            graph_neighbors=config["mstgc_graph_k"],
            graph_density=config["mstgc_graph_density"],
            graph_time_points=config["mstgc_time_points"],
            covariance_shrinkage=config["mstgc_shrinkage"],
        ).to(device)
    checkpoint_state, checkpoint_migrations = migrate_legacy_spddsbn_buffers(
        _load_state(checkpoint), model.state_dict()
    )
    try:
        model.load_state_dict(checkpoint_state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint architecture mismatch. Supply the exact model CLI "
            "hyperparameters used for training. Original error:\n{}".format(exc)
        ) from exc
    model.eval()

    pre, post, metadata, reused = _extract_or_load(
        args, dataset, split, domains, normalizer, model, checkpoint,
        cache, subject, config, device,
    )
    if pre.shape != post.shape or len(pre) != len(metadata):
        raise RuntimeError("Extracted pre/post samples are not one-to-one")
    checks = {
        "pre": validate_spd_matrices(pre, "spd_pre_bn"),
        "post": validate_spd_matrices(post, "spd_post_bn"),
    }
    source = metadata["domain"].to_numpy() == "source"
    reference, karcher = airm_karcher_mean(
        pre[source], tolerance=args.karcher_tol,
        max_iterations=args.karcher_max_iter,
        batch_size=args.riemann_batch_size,
    )
    pre_tangent = common_tangent_vectors(
        pre, reference, batch_size=args.riemann_batch_size
    )
    post_tangent = common_tangent_vectors(
        post, reference, batch_size=args.riemann_batch_size
    )
    expected_dimension = pre.shape[-1] * (pre.shape[-1] + 1) // 2
    if pre_tangent.shape[1] != expected_dimension or post_tangent.shape != pre_tangent.shape:
        raise RuntimeError("Common tangent-space dimension/alignment check failed")
    scaler = StandardScaler().fit(pre_tangent[source])
    pre_scaled = scaler.transform(pre_tangent)
    post_scaled = scaler.transform(post_tangent)
    pca = PCA(n_components=2, svd_solver="full").fit(pre_scaled[source])
    pre_coordinates = pca.transform(pre_scaled)
    post_coordinates = pca.transform(post_scaled)
    metrics = alignment_metrics(
        pre_tangent, post_tangent,
        metadata["domain"].to_numpy(), metadata["class_id"].to_numpy(),
    )
    metrics["target_labels_usage"] = "offline class-conditional metric only"
    manifest = balanced_plot_manifest(
        metadata, max_per_class_domain=args.max_points_per_class_domain,
        seed=args.plot_seed,
    )
    manifest.to_csv(os.path.join(args.output_dir, "plot_sample_manifest.csv"), index=False)

    coordinates = metadata.copy()
    coordinates["pre_pc1"] = pre_coordinates[:, 0]
    coordinates["pre_pc2"] = pre_coordinates[:, 1]
    coordinates["post_pc1"] = post_coordinates[:, 0]
    coordinates["post_pc2"] = post_coordinates[:, 1]
    coordinates.to_csv(os.path.join(args.output_dir, "pca_coordinates.csv"), index=False)
    np.savez_compressed(
        os.path.join(args.output_dir, "common_tangent_pca_artifacts.npz"),
        reference=reference,
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        pca_components=pca.components_,
        pca_mean=pca.mean_,
        pca_explained_variance=pca.explained_variance_,
        pca_explained_variance_ratio=pca.explained_variance_ratio_,
        eeg_normalizer_center=normalizer.center,
        eeg_normalizer_scale=normalizer.scale,
        sample_id=metadata["sample_id"].to_numpy(dtype=np.int64),
        pre_tangent=pre_tangent,
        post_tangent=post_tangent,
        pre_coordinates=pre_coordinates,
        post_coordinates=post_coordinates,
    )
    _json_dump(metrics, os.path.join(args.output_dir, "alignment_metrics.json"))
    pd.DataFrame([metrics]).to_csv(
        os.path.join(args.output_dir, "alignment_metrics.csv"), index=False
    )
    axis_limits, clipped = _make_figures(
        args.output_dir, pre_coordinates, post_coordinates, metadata,
        manifest, pca.explained_variance_ratio_, metrics,
        args.annotate_metrics, not args.no_ellipses,
    )
    checks["karcher"] = karcher
    checks["tangent_dimension"] = int(expected_dimension)
    checks["pca_explained_variance_ratio"] = [
        float(v) for v in pca.explained_variance_ratio_
    ]
    checks["axis_limits"] = axis_limits
    checks["clipped_plot_points"] = clipped
    checks["source_train_count"] = int(source.sum())
    checks["target_test_count"] = int((~source).sum())
    checks["fit_provenance"] = {
        "reference": "source_train_spd_pre_bn_only",
        "scaler": "source_train_pre_tangent_only",
        "pca": "source_train_pre_tangent_scaled_only",
        "target_labels": "not used for reference/scaler/PCA/model statistics",
    }
    _json_dump(checks, os.path.join(args.output_dir, "numerical_checks.json"))

    model_metadata = visualization_model_metadata(args.model)
    run_config = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "command": " ".join(sys.argv),
        "dataset": dataset["name"],
        "cache": os.path.abspath(cache),
        "target_subject": int(subject),
        "checkpoint": checkpoint,
        "model_type": args.model,
        "model_class": model_metadata["model_class"],
        "feature_locations": {
            "pre": model_metadata["pre"],
            "post": model_metadata["post"],
        },
        "model_config": config,
        "plot_seed": int(args.plot_seed),
        "feature_cache_reused": bool(reused),
        "checkpoint_buffer_migrations": checkpoint_migrations,
        "font_family": matplotlib.rcParams["font.family"],
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "sklearn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    _json_dump(run_config, os.path.join(args.output_dir, "run_config.json"))
    print("Selected target subject:", subject)
    print("Checkpoint:", checkpoint)
    print("Legacy SPDDSBN buffers migrated:", len(checkpoint_migrations))
    print("SPD shape:", pre.shape)
    print("Minimum eigenvalue before SPDDSBN:", checks["pre"]["minimum_eigenvalue"])
    print("Minimum eigenvalue after SPDDSBN:", checks["post"]["minimum_eigenvalue"])
    print("Karcher iterations/residual:", karcher["iterations"], karcher["residual"])
    print("PCA explained variance ratio:", pca.explained_variance_ratio_.tolist())
    print("Alignment metrics:", metrics)
    print("Saved figures and artifacts:", os.path.abspath(args.output_dir))


if __name__ == "__main__":
    main()
