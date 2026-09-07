"""Publication-quality analysis of learned EEG channel interaction patterns.

The default MS-TGC mode analyzes the actual adaptive adjacency used by the
Chebyshev layer. A TSMNet reference mode is also supported, but is explicitly
reported as a spatial-filter channel-similarity proxy because TSMNet has no
message-passing graph. No random or synthetic values are accepted by main().
"""

import argparse
import json
import math
import os
import platform
import random
import sys
import warnings
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import matplotlib.tri as mtri
import networkx as nx
import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.cl_tsmnet.datasets import load_dataset
from src.cl_tsmnet.experiment_utils import default_cache_path, default_target_fs
from src.cl_tsmnet.spd_pca import choose_median_fold, migrate_legacy_spddsbn_buffers
from src.cl_tsmnet.spd_visualization_adapters import (
    extract_graph_analysis_features,
    extract_static_channel_interaction,
)
from src.cl_tsmnet.splits import domain_ids, make_split
from src.cl_tsmnet.training import (
    _filter_artifact_windows,
    build_ms_tgc_spddsbn,
    build_tsmnet,
    fit_source_normalizer,
)


RANDOM_SEED = 2026
FIGURE_SIZE = (7.16, 5.25)
MODEL_DEFAULTS = {
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
}
CLASS_COLORS = ["#377EB8", "#E69F00", "#009E73", "#CC79A7"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate IEEE Fig. 6 learned channel interaction analysis."
    )
    parser.add_argument(
        "--datasets", default="stew",
        help="Comma-separated: stew,eegmat,cog-bci:nback,cog-bci:matb.",
    )
    parser.add_argument(
        "--dataset-labels", default="STEW",
        help="Comma-separated display names matching --datasets.",
    )
    parser.add_argument(
        "--model", choices=["ms_tgc_spddsbn", "tsmnet"],
        default="ms_tgc_spddsbn",
    )
    parser.add_argument("--bnorm", choices=["none", "spdbn", "spddsbn"], default="spddsbn")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--cache-root", default=os.path.join("outputs", "cache"))
    parser.add_argument("--master-summary", default=os.path.join("outputs", "master_summary.csv"))
    parser.add_argument("--output-dir", default="results")
    parser.add_argument(
        "--feature-cache-dir",
        default=os.path.join("results", "fig6_graph_pattern_cache"),
    )
    parser.add_argument("--target-fs-stew", type=float, default=None)
    parser.add_argument("--target-fs-eegmat", type=float, default=None)
    parser.add_argument("--target-fs-cog-bci", type=float, default=None)
    parser.add_argument(
        "--target-subjects", default="",
        help="Optional dataset=subject pairs for response maps.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dynamic-windows-per-fold", type=int, default=120)
    parser.add_argument("--response-windows-per-class", type=int, default=150)
    parser.add_argument("--max-stability-pairs", type=int, default=4000)
    parser.add_argument("--top-edge-fraction", type=float, default=0.10)
    parser.add_argument("--min-edge-prevalence", type=float, default=0.70)
    parser.add_argument(
        "--tsmnet-proxy-neighbors", type=int, default=4,
        help="Per-node sparsity used only for the TSMNet interaction proxy.",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--force-reextract", action="store_true")
    parser.add_argument("--allow-missing-master-config", action="store_true")
    parser.add_argument("--allow-legacy-refit", action="store_true")
    return parser.parse_args()


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(value, path):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_jsonable(value), handle, indent=2, ensure_ascii=False)


def _font_family():
    for family in ["Arial", "Helvetica", "DejaVu Sans"]:
        try:
            font_manager.findfont(family, fallback_to_default=False)
            return family
        except ValueError:
            continue
    return "DejaVu Sans"


def _usable(value):
    if value is None:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return value


def _parse_datasets(values, labels):
    raw = [item.strip() for item in values.split(",") if item.strip()]
    display = [item.strip() for item in labels.split(",") if item.strip()]
    if not raw or len(raw) != len(display):
        raise ValueError("--datasets and --dataset-labels must have equal nonzero length")
    result = []
    for value, label in zip(raw, display):
        parts = value.split(":", 1)
        dataset = parts[0].lower()
        paradigm = parts[1].lower() if len(parts) == 2 else "nback"
        if dataset not in {"stew", "eegmat", "cog-bci"}:
            raise ValueError("Unsupported dataset: {}".format(value))
        if dataset == "cog-bci" and paradigm not in {"nback", "matb"}:
            raise ValueError("COG-BCI paradigm must be nback or matb")
        result.append({
            "dataset": dataset, "paradigm": paradigm,
            "name": "cog-bci-{}".format(paradigm) if dataset == "cog-bci" else dataset,
            "display_name": label,
        })
    return result


def _target_overrides(text):
    result = {}
    for item in [value.strip() for value in text.split(",") if value.strip()]:
        if "=" not in item:
            raise ValueError("Invalid target-subject item: {}".format(item))
        key, value = item.split("=", 1)
        result[key.strip()] = int(value)
    return result


def _target_fs(spec, args):
    value = getattr(args, "target_fs_{}".format(spec["dataset"].replace("-", "_")))
    return default_target_fs(spec["dataset"], value)


def _run_candidates(spec, args):
    if args.model == "ms_tgc_spddsbn":
        return [os.path.join(args.output_root, "{}_loso_ms_tgc_spddsbn".format(spec["name"]))]
    return [
        os.path.join(args.output_root, "{}_loso_tsmnet_{}".format(spec["name"], args.bnorm)),
        os.path.join(args.output_root, "{}_loso_{}".format(spec["name"], args.bnorm)),
    ]


def _resolve_run(spec, args, master):
    candidates = _run_candidates(spec, args)
    run_dir = next((path for path in candidates if os.path.isdir(path)), None)
    if run_dir is None:
        raise FileNotFoundError(
            "Missing {} LOSO output for {}. Tried:\n  {}".format(
                args.model, spec["display_name"],
                "\n  ".join(os.path.abspath(path) for path in candidates),
            )
        )
    summary_path = os.path.join(run_dir, "summary.csv")
    if not os.path.exists(summary_path):
        raise FileNotFoundError("Missing per-fold summary: {}".format(summary_path))
    summary = pd.read_csv(summary_path)
    if "protocol" in summary:
        summary = summary[summary["protocol"].astype(str) == "loso"]
    if "model_type" in summary:
        summary = summary[summary["model_type"].astype(str) == args.model]
    if summary.empty:
        raise ValueError("No matching LOSO rows in {}".format(summary_path))

    record = {}
    if master is not None and not master.empty:
        frame = master.copy()
        for column, value in [
            ("dataset", spec["name"]), ("protocol", "loso"),
            ("model_type", args.model),
        ]:
            if column in frame:
                frame = frame[frame[column].astype(str) == str(value)]
        if args.model == "tsmnet" and "model" in frame:
            exact = frame[frame["model"].astype(str) == "tsmnet_{}".format(args.bnorm)]
            if not exact.empty:
                frame = exact
        if not frame.empty:
            record = frame.iloc[-1].dropna().to_dict()
    if not record and not args.allow_missing_master_config:
        raise ValueError(
            "No matching master-summary configuration for {}. Exact model "
            "reconstruction is required.".format(spec["name"])
        )
    return {
        "run_dir": os.path.abspath(run_dir),
        "summary_path": os.path.abspath(summary_path),
        "summary": summary,
        "record": record,
    }


def _value(record, key, default, cast):
    value = _usable(record.get(key))
    return default if value is None else cast(value)


def _model_config(record):
    integer = {
        "temporal_filters", "spatial_filters", "subspacedims", "temp_kernel",
        "mstgc_temporal_hidden", "mstgc_graph_hidden", "mstgc_fusion_dim",
        "mstgc_kernel_length", "mstgc_num_heads", "mstgc_cheby_order",
        "mstgc_num_nodes", "mstgc_graph_k", "mstgc_time_points",
    }
    return {
        key: _value(record, key, default, int if key in integer else float)
        for key, default in MODEL_DEFAULTS.items()
    }


def _split_config(run_info):
    record, summary = run_info["record"], run_info["summary"]
    artifact = _usable(record.get("artifact_z"))
    if artifact is None and "artifact_z" in summary:
        values = summary["artifact_z"].dropna()
        artifact = _usable(values.iloc[0]) if len(values) else None
    return {
        "seed": _value(record, "seed", 42, int),
        "val_size": _value(record, "val_size", 0.2, float),
        "test_size": _value(record, "test_size", 0.2, float),
        "artifact_z": None if artifact is None else float(artifact),
    }


def _validate_run(run_info, args):
    summary = run_info["summary"]
    if "target_adapt" not in summary:
        raise ValueError("summary.csv lacks target_adapt audit data")
    expects_adapt = args.model == "ms_tgc_spddsbn" or (
        args.model == "tsmnet" and args.bnorm == "spddsbn"
    )
    actual = summary["target_adapt"].astype(str).str.lower().isin(["true", "1", "yes"])
    if expects_adapt and not bool(actual.all()):
        raise ValueError("Selected model was not evaluated with target adaptation")
    if expects_adapt:
        scopes = set(summary.get("target_refit_scope", pd.Series(dtype=str)).dropna().astype(str))
        if scopes != {"target_only"}:
            if not args.allow_legacy_refit:
                raise ValueError(
                    "Current Fig. 6 requires target_refit_scope=target_only; found {}"
                    .format(sorted(scopes))
                )
            warnings.warn("Using legacy refit scope {}".format(sorted(scopes)), UserWarning)


def _checkpoint(run_dir, subject):
    candidates = [
        os.path.join(run_dir, "subject_{:02d}".format(int(subject)), "model.pt"),
        os.path.join(run_dir, "subject_{}".format(int(subject)), "model.pt"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)
    raise FileNotFoundError("Missing checkpoint for subject {}; tried {}".format(subject, candidates))


def _load_state(path):
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError("Checkpoint is not a state_dict: {}".format(path))
    return state


def _build_model(dataset, domains, selected, source_train, config, args):
    shape = dataset["x"].shape
    nclasses = int(len(np.unique(dataset["y"][source_train])))
    if args.model == "tsmnet":
        bnorm = None if args.bnorm == "none" else args.bnorm
        model = build_tsmnet(
            PROJECT_ROOT, shape[1], shape[2], nclasses, domains[selected],
            bnorm=bnorm, temporal_filters=config["temporal_filters"],
            spatial_filters=config["spatial_filters"],
            subspacedims=config["subspacedims"],
            temp_kernel=config["temp_kernel"], device=args.device_object,
        )
    else:
        kernel = max(3, int(round(
            config["mstgc_kernel_length"] * float(dataset["fs"]) / 128.0
        )))
        model = build_ms_tgc_spddsbn(
            PROJECT_ROOT, shape[1], shape[2], nclasses, domains[selected],
            subspacedims=config["subspacedims"], device=args.device_object,
            temporal_hidden=config["mstgc_temporal_hidden"],
            graph_hidden=config["mstgc_graph_hidden"],
            fusion_dim=config["mstgc_fusion_dim"], kernel_length=kernel,
            num_heads=config["mstgc_num_heads"],
            cheby_order=config["mstgc_cheby_order"],
            dropout=config["mstgc_dropout"], num_nodes=config["mstgc_num_nodes"],
            variant="ms_tgc_spddsbn", graph_mode="adaptive",
            graph_neighbors=config["mstgc_graph_k"],
            graph_density=config["mstgc_graph_density"],
            graph_time_points=config["mstgc_time_points"],
            covariance_shrinkage=config["mstgc_shrinkage"],
        )
    return model.to(args.device_object).eval()


def _sparsify_top_edges(matrix, edge_count):
    matrix = np.asarray(matrix, dtype=np.float64)
    matrix = np.maximum(0.0, 0.5 * (matrix + matrix.T))
    np.fill_diagonal(matrix, 0.0)
    upper = np.triu_indices(len(matrix), k=1)
    keep = min(max(1, int(edge_count)), len(upper[0]))
    selected = np.argpartition(matrix[upper], -keep)[-keep:]
    result = np.zeros_like(matrix)
    rows, cols = upper[0][selected], upper[1][selected]
    result[rows, cols] = matrix[rows, cols]
    result[cols, rows] = matrix[rows, cols]
    return result


def _topk_per_node(matrix, neighbors):
    matrix = np.asarray(matrix, dtype=np.float64)
    matrix = np.maximum(0.0, 0.5 * (matrix + matrix.T))
    np.fill_diagonal(matrix, 0.0)
    count = len(matrix)
    k = min(max(1, int(neighbors)), count - 1)
    mask = np.zeros_like(matrix, dtype=bool)
    for row in range(count):
        selected = np.argpartition(matrix[row], -k)[-k:]
        mask[row, selected] = True
    mask |= mask.T
    return matrix * mask


def _dynamic_graphs(temporal_maps, edge_count):
    values = temporal_maps.reshape(
        temporal_maps.shape[0], temporal_maps.shape[1], -1
    ).astype(np.float64)
    values -= values.mean(axis=2, keepdims=True)
    values /= np.linalg.norm(values, axis=2, keepdims=True) + 1e-12
    correlations = np.abs(np.matmul(values, np.swapaxes(values, 1, 2)))
    return np.stack([
        _sparsify_top_edges(matrix, edge_count) for matrix in correlations
    ]).astype(np.float32)


def _balanced_target_ids(dataset, target_ids, per_class, seed):
    rng = np.random.RandomState(int(seed))
    labels = dataset["y"][target_ids]
    class_ids = sorted(int(value) for value in np.unique(labels))
    groups = [np.asarray(target_ids)[labels == class_id].copy() for class_id in class_ids]
    count = min(int(per_class), min(len(group) for group in groups))
    if count < 1:
        raise ValueError("Balanced target selection found an empty class")
    selected = []
    for ids in groups:
        rng.shuffle(ids)
        selected.extend(ids[:count].tolist())
    return np.asarray(selected, dtype=np.int64)


def _extract_maps(model, dataset, ids, domains, normalizer, args):
    temporal_parts, response_parts = [], []
    feature_metadata = None
    with torch.no_grad():
        for start in range(0, len(ids), int(args.batch_size)):
            batch_ids = ids[start:start + int(args.batch_size)]
            windows = normalizer.transform_array(dataset["x"][batch_ids])
            xb = torch.from_numpy(windows).to(args.device_object, dtype=torch.float32)
            db = torch.from_numpy(domains[batch_ids]).to(args.device_object, dtype=torch.long)
            temporal, response, current = extract_graph_analysis_features(
                model, xb, db, args.model
            )
            if feature_metadata is None:
                feature_metadata = current
            elif feature_metadata != current:
                raise RuntimeError("Graph feature location changed between batches")
            temporal_parts.append(temporal.detach().cpu().numpy().astype(np.float32))
            response_parts.append(torch.sqrt(
                torch.mean(response.square(), dim=(2, 3)) + 1e-12
            ).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(temporal_parts), np.concatenate(response_parts), feature_metadata


def _fold_analysis(context, run_info, subject, representative, config, split_config, args):
    dataset, domains = context["dataset"], context["domains"]
    checkpoint = _checkpoint(run_info["run_dir"], subject)
    cache_dir = os.path.join(
        args.feature_cache_dir, context["spec"]["name"],
        args.model + ("_" + args.bnorm if args.model == "tsmnet" else ""),
        "subject_{:02d}".format(int(subject)),
    )
    data_path = os.path.join(cache_dir, "fold_graph_analysis.npz")
    signature_path = os.path.join(cache_dir, "signature.json")
    signature = {
        "analysis_cache_schema": 2,
        "checkpoint": checkpoint, "checkpoint_size": int(os.path.getsize(checkpoint)),
        "checkpoint_mtime_ns": int(os.stat(checkpoint).st_mtime_ns),
        "dataset_cache": context["cache"], "subject": int(subject),
        "representative": bool(representative), "model": args.model,
        "bnorm": args.bnorm, "model_config": config,
        "split_config": split_config,
        "dynamic_windows_per_fold": int(args.dynamic_windows_per_fold),
        "response_windows_per_class": int(args.response_windows_per_class),
        "tsmnet_proxy_neighbors": int(args.tsmnet_proxy_neighbors),
    }
    if not args.force_reextract and os.path.exists(data_path) and os.path.exists(signature_path):
        with open(signature_path, encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("signature") == _jsonable(signature):
            with np.load(data_path, allow_pickle=False) as saved:
                return {key: saved[key] for key in saved.files}, payload

    split = make_split(
        dataset, "loso", int(subject), seed=int(split_config["seed"]),
        val_size=float(split_config["val_size"]),
        test_size=float(split_config["test_size"]),
    )
    normalizer = fit_source_normalizer(dataset["x"], split["train"])
    filtered = {}
    for name in ["train", "val", "test"]:
        filtered[name] = _filter_artifact_windows(
            dataset["x"], split[name], normalizer, split_config["artifact_z"]
        )
        if len(filtered[name]) == 0:
            raise RuntimeError("No {} windows remain for subject {}".format(name, subject))
    selected = np.concatenate([filtered["train"], filtered["val"], filtered["test"]])
    model = _build_model(
        dataset, domains, selected, filtered["train"], config, args
    )
    state, migrations = migrate_legacy_spddsbn_buffers(
        _load_state(checkpoint), model.state_dict()
    )
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint architecture mismatch for subject {}.\n{}".format(subject, exc)
        ) from exc
    adjacency_tensor, interaction_metadata = extract_static_channel_interaction(
        model, args.model
    )
    adjacency = adjacency_tensor.detach().cpu().numpy().astype(np.float64)
    if args.model == "tsmnet":
        adjacency = _topk_per_node(adjacency, args.tsmnet_proxy_neighbors)
    scale = float(np.max(adjacency))
    if scale <= 0.0:
        raise ValueError("Learned channel interaction matrix has no positive edge")
    adjacency /= scale
    edge_count = int(np.count_nonzero(np.triu(adjacency, k=1)))

    target_class_count = len(np.unique(dataset["y"][filtered["test"]]))
    dynamic_per_class = max(
        1, int(math.ceil(args.dynamic_windows_per_fold / float(target_class_count)))
    )
    dynamic_ids = _balanced_target_ids(
        dataset, filtered["test"], dynamic_per_class, args.seed + int(subject)
    )
    temporal, _, feature_metadata = _extract_maps(
        model, dataset, dynamic_ids, domains, normalizer, args
    )
    dynamic_graphs = _dynamic_graphs(temporal, edge_count)
    response_ids = (
        _balanced_target_ids(
            dataset, filtered["test"], args.response_windows_per_class,
            args.seed + int(subject),
        )
        if representative else dynamic_ids
    )
    _, target_response, _ = _extract_maps(
        model, dataset, response_ids, domains, normalizer, args
    )
    result = {
        "adjacency": adjacency.astype(np.float32),
        "dynamic_graphs": dynamic_graphs,
        "dynamic_ids": dynamic_ids.astype(np.int64),
        "target_ids": response_ids.astype(np.int64),
        "target_labels": dataset["y"][response_ids].astype(np.int64),
        "target_response": target_response.astype(np.float32),
        "channels": np.asarray(dataset["channels"], dtype="U32"),
    }
    if representative:
        source_ids = np.asarray(filtered["train"], dtype=np.int64)
        _, source_response, _ = _extract_maps(
            model, dataset, source_ids, domains, normalizer, args
        )
        result["source_response_mean"] = source_response.mean(axis=0).astype(np.float32)
        result["source_response_std"] = (
            source_response.std(axis=0, ddof=1) + 1e-6
        ).astype(np.float32)
    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(data_path, **result)
    payload = {
        "signature": signature, "interaction": interaction_metadata,
        "features": feature_metadata, "checkpoint_buffer_migrations": migrations,
        "edge_count": edge_count,
    }
    _write_json(payload, signature_path)
    return result, payload


def load_channel_info(channels):
    """Load standard EEG positions, with a deterministic matplotlib fallback."""
    channels = [str(channel) for channel in channels]
    try:
        import mne
        montage = mne.channels.make_standard_montage("standard_1020")
        positions = {
            str(name).lower(): value
            for name, value in montage.get_positions()["ch_pos"].items()
        }
        missing = [channel for channel in channels if channel.lower() not in positions]
        if missing:
            raise ValueError("standard_1020 lacks {}".format(missing))
        xy = np.stack([positions[channel.lower()][:2] for channel in channels])
        source = "MNE standard_1020"
    except (ImportError, ValueError) as exc:
        warnings.warn(
            "MNE montage unavailable ({}); using deterministic approximate "
            "10-20 positions.".format(exc), UserWarning,
        )
        y_by_region = {
            "FP": 0.92, "AF": 0.76, "F": 0.55, "FT": 0.32,
            "FC": 0.27, "T": 0.02, "C": 0.02, "TP": -0.26,
            "CP": -0.28, "P": -0.53, "PO": -0.73, "O": -0.90,
        }
        lateral = {1: 0.18, 2: 0.18, 3: 0.38, 4: 0.38,
                   5: 0.61, 6: 0.61, 7: 0.82, 8: 0.82,
                   9: 0.94, 10: 0.94}
        values = []
        for channel in channels:
            upper = channel.upper()
            prefix = next((key for key in sorted(y_by_region, key=len, reverse=True)
                           if upper.startswith(key)), "C")
            suffix = upper[len(prefix):]
            if suffix == "Z" or not suffix:
                x = 0.0
            else:
                number = int("".join(character for character in suffix if character.isdigit()) or 1)
                x = lateral.get(number, 0.9)
                if number % 2 == 1:
                    x *= -1.0
            values.append([x, y_by_region[prefix]])
        xy = np.asarray(values, dtype=np.float64)
        source = "approximate bundled 10-20 fallback"
    radius = np.max(np.linalg.norm(xy, axis=1))
    xy = 0.92 * xy / max(radius, 1e-12)
    return pd.DataFrame({
        "channel": channels, "x": xy[:, 0], "y": xy[:, 1],
        "position_source": source,
    })


def load_adjacency_matrices(paths, expected_channels=None):
    """Load one symmetric channel interaction matrix from each fold cache/state."""
    matrices, channels = [], None
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError("Adjacency input missing: {}".format(path))
        with np.load(path, allow_pickle=False) as saved:
            if "adjacency" in saved.files:
                matrix = saved["adjacency"]
            elif "adjacencies" in saved.files:
                values = saved["adjacencies"]
                weights = saved["graph_weights"] if "graph_weights" in saved.files else np.ones(len(values))
                matrix = np.tensordot(weights / np.sum(weights), values, axes=(0, 0))
            else:
                raise ValueError("{} contains no adjacency data".format(path))
            current_channels = [str(value) for value in saved["channels"]]
        matrix = np.asarray(matrix, dtype=np.float64)
        matrix = np.maximum(0.0, 0.5 * (matrix + matrix.T))
        np.fill_diagonal(matrix, 0.0)
        if not np.isfinite(matrix).all() or matrix.shape != (len(current_channels), len(current_channels)):
            raise ValueError("Invalid adjacency matrix in {}".format(path))
        if channels is None:
            channels = current_channels
        elif channels != current_channels:
            raise ValueError("Channel order differs across fold adjacency files")
        scale = float(np.max(matrix))
        matrices.append(matrix / max(scale, 1e-12))
    if expected_channels is not None and channels != [str(value) for value in expected_channels]:
        raise ValueError("Saved graph channel order does not match the dataset")
    return np.stack(matrices), channels


def compute_mean_adjacency(adjacencies):
    """Symmetrize, max-normalize per fold, and average learned interactions."""
    values = np.asarray(adjacencies, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] != values.shape[2]:
        raise ValueError("Adjacencies must have shape [folds, channels, channels]")
    normalized = []
    for matrix in values:
        matrix = np.maximum(0.0, 0.5 * (matrix + matrix.T))
        np.fill_diagonal(matrix, 0.0)
        normalized.append(matrix / max(float(np.max(matrix)), 1e-12))
    mean = np.mean(normalized, axis=0)
    return 0.5 * (mean + mean.T)


def _pair_indices(count, max_pairs, seed):
    pairs = np.asarray([
        (left, right) for left in range(count) for right in range(left + 1, count)
    ], dtype=np.int64)
    if len(pairs) > int(max_pairs):
        rng = np.random.RandomState(int(seed))
        pairs = pairs[rng.choice(len(pairs), int(max_pairs), replace=False)]
    return pairs


def compute_edge_stability(adjacencies, max_pairs=4000, seed=RANDOM_SEED):
    """Return pairwise upper-edge Jaccard similarities."""
    values = np.asarray(adjacencies)
    upper = np.triu_indices(values.shape[1], k=1)
    edges = values[:, upper[0], upper[1]] > 0.0
    scores = []
    for left, right in _pair_indices(len(values), max_pairs, seed):
        union = np.logical_or(edges[left], edges[right]).sum()
        intersection = np.logical_and(edges[left], edges[right]).sum()
        scores.append(1.0 if union == 0 else intersection / float(union))
    return np.asarray(scores, dtype=np.float64)


def compute_weight_stability(adjacencies, max_pairs=4000, seed=RANDOM_SEED):
    """Return pairwise cosine similarity of weighted upper triangles."""
    values = np.asarray(adjacencies, dtype=np.float64)
    upper = np.triu_indices(values.shape[1], k=1)
    vectors = values[:, upper[0], upper[1]]
    norms = np.linalg.norm(vectors, axis=1)
    scores = []
    for left, right in _pair_indices(len(values), max_pairs, seed):
        denominator = norms[left] * norms[right]
        scores.append(0.0 if denominator <= 1e-12 else
                      float(np.dot(vectors[left], vectors[right]) / denominator))
    return np.asarray(scores, dtype=np.float64)


def _per_fold_shared_stability(adjacencies, function):
    rows = []
    for index in range(len(adjacencies)):
        scores = []
        for other in range(len(adjacencies)):
            if other == index:
                continue
            scores.extend(function(adjacencies[[index, other]], max_pairs=1, seed=0))
        rows.append(float(np.mean(scores)))
    return np.asarray(rows)


def _stable_edges(adjacencies, fraction, prevalence_threshold):
    mean = compute_mean_adjacency(adjacencies)
    prevalence = np.mean(np.asarray(adjacencies) > 0.0, axis=0)
    upper = np.triu_indices(len(mean), k=1)
    keep = max(1, int(math.ceil(float(fraction) * len(upper[0]))))
    top = np.argpartition(mean[upper], -keep)[-keep:]
    mask = np.zeros_like(mean, dtype=bool)
    rows, cols = upper[0][top], upper[1][top]
    valid = prevalence[rows, cols] >= float(prevalence_threshold)
    mask[rows[valid], cols[valid]] = True
    mask |= mask.T
    return mean, prevalence, mask


def _draw_head(ax):
    theta = np.linspace(0.0, 2.0 * np.pi, 240)
    ax.plot(np.cos(theta), np.sin(theta), color="#333333", linewidth=0.65)
    ax.plot([-0.10, 0.0, 0.10], [0.995, 1.10, 0.995], color="#333333", linewidth=0.65)
    ax.plot([-1.00, -1.07, -1.00], [0.18, 0.0, -0.18], color="#333333", linewidth=0.55)
    ax.plot([1.00, 1.07, 1.00], [0.18, 0.0, -0.18], color="#333333", linewidth=0.55)
    ax.set_xlim(-1.12, 1.12)
    ax.set_ylim(-1.08, 1.13)
    ax.set_aspect("equal")
    ax.axis("off")


def _topomap(ax, positions, values, cmap, vmin, vmax):
    xy = positions[["x", "y"]].to_numpy(dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    artist = None
    try:
        triangulation = mtri.Triangulation(xy[:, 0], xy[:, 1])
        artist = ax.tricontourf(
            triangulation, values, levels=np.linspace(vmin, vmax, 80),
            cmap=cmap, vmin=vmin, vmax=vmax, extend="both",
        )
    except (RuntimeError, ValueError):
        artist = ax.scatter(
            xy[:, 0], xy[:, 1], c=values, cmap=cmap,
            vmin=vmin, vmax=vmax, s=30,
        )
    circle = plt.Circle((0, 0), 1.0, transform=ax.transData)
    for collection in getattr(artist, "collections", []):
        collection.set_clip_path(circle)
    ax.scatter(xy[:, 0], xy[:, 1], s=3.5, c="#333333", linewidths=0, zorder=5)
    _draw_head(ax)
    return artist


def plot_adjacency_heatmap(ax, mean_adjacency, channels, title):
    image = ax.imshow(mean_adjacency, cmap="cividis", vmin=0.0,
                      vmax=max(float(np.max(mean_adjacency)), 1e-6), aspect="equal")
    step = max(1, int(math.ceil(len(channels) / 16.0)))
    ticks = np.arange(0, len(channels), step)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([channels[index] for index in ticks], rotation=90, fontsize=4.7)
    ax.set_yticklabels([channels[index] for index in ticks], fontsize=4.7)
    ax.tick_params(length=1.5, width=0.45, pad=1.0)
    ax.set_title(title, fontsize=7.5, pad=4.0, fontweight="semibold")
    ax.text(0.01, 1.02, "(a)", transform=ax.transAxes, fontsize=7.0,
            fontweight="semibold", va="bottom")
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.025)
    colorbar.ax.tick_params(labelsize=5.2, width=0.4, length=2)
    colorbar.set_label("Normalized weight", fontsize=5.6)
    return image


def plot_scalp_graph(ax, mean_adjacency, stable_mask, positions, node_strength,
                     title="Stable learned channel interactions"):
    xy = positions[["x", "y"]].to_numpy(dtype=np.float64)
    graph = nx.Graph()
    graph.add_nodes_from(range(len(xy)))
    rows, cols = np.where(np.triu(stable_mask, k=1))
    for row, col in zip(rows, cols):
        graph.add_edge(int(row), int(col), weight=float(mean_adjacency[row, col]))
    layout = {index: tuple(xy[index]) for index in range(len(xy))}
    weights = np.asarray([data["weight"] for _, _, data in graph.edges(data=True)])
    if len(weights):
        normalized = (weights - weights.min()) / max(float(np.ptp(weights)), 1e-12)
        nx.draw_networkx_edges(
            graph, layout, ax=ax, width=0.45 + 1.7 * normalized,
            edge_color=weights, edge_cmap=plt.cm.cividis,
            edge_vmin=float(weights.min()), edge_vmax=float(weights.max()),
            alpha=0.72,
        )
    else:
        ax.text(0.5, 0.08, "No edges satisfy the prespecified criterion",
                transform=ax.transAxes, ha="center", fontsize=5.3, color="#555555")
    nx.draw_networkx_nodes(
        graph, layout, ax=ax, node_size=18, node_color=node_strength,
        cmap=plt.cm.viridis, linewidths=0.35, edgecolors="white",
    )
    _draw_head(ax)
    ax.set_title(title, fontsize=7.3, pad=4.0, fontweight="semibold")
    ax.text(0.01, 1.02, "(b1)", transform=ax.transAxes, fontsize=7.0,
            fontweight="semibold", va="bottom")


def plot_node_strength_topomap(ax, positions, node_strength):
    artist = _topomap(
        ax, positions, node_strength, "viridis",
        float(np.min(node_strength)), float(np.max(node_strength)),
    )
    ax.set_title("Node strength distribution", fontsize=7.3, pad=4.0,
                 fontweight="semibold")
    ax.text(0.01, 1.02, "(b2)", transform=ax.transAxes, fontsize=7.0,
            fontweight="semibold", va="bottom")
    colorbar = ax.figure.colorbar(artist, ax=ax, fraction=0.046, pad=0.02)
    colorbar.ax.tick_params(labelsize=5.2, width=0.4, length=2)
    colorbar.set_label("Strength", fontsize=5.6)


def plot_node_response_maps(container, positions, responses, class_names, figure):
    count = len(responses)
    grid = container.subgridspec(1, count, wspace=0.10)
    limit = max(float(np.max(np.abs(responses))), 1e-6)
    axes, artist = [], None
    for index, (response, name) in enumerate(zip(responses, class_names)):
        ax = figure.add_subplot(grid[0, index])
        artist = _topomap(ax, positions, response, "RdBu_r", -limit, limit)
        ax.set_title(name, fontsize=6.8, pad=2.5, fontweight="semibold")
        axes.append(ax)
    axes[0].text(-0.05, 1.10, "(c) Task-related node response",
                 transform=axes[0].transAxes, fontsize=7.2,
                 fontweight="semibold", va="bottom")
    colorbar = figure.colorbar(
        artist, ax=axes, orientation="horizontal", fraction=0.065,
        pad=0.08, aspect=35,
    )
    colorbar.ax.tick_params(labelsize=5.2, width=0.4, length=2)
    colorbar.set_label("Source-standardized node response", fontsize=5.6)
    return axes


def plot_graph_stability(ax, stability):
    metrics = ["edge_jaccard", "weight_cosine"]
    labels = ["Edge-set\nJaccard", "Weight\ncosine"]
    graph_types = ["Dynamic graph"] + [
        value for value in stability["graph_type"].drop_duplicates().astype(str)
        if value != "Dynamic graph"
    ]
    if len(graph_types) != 2:
        raise ValueError("Stability plot requires dynamic and one shared graph type")
    colors = ["#8A8A8A", "#3B75AF"]
    hatches = ["////", ""]
    x = np.arange(len(metrics), dtype=np.float64)
    width = 0.32
    for index, graph_type in enumerate(graph_types):
        means, errors = [], []
        for metric in metrics:
            values = stability.loc[
                stability["graph_type"] == graph_type, metric
            ].to_numpy(dtype=np.float64)
            means.append(float(np.mean(values)))
            errors.append(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)
        positions = x + (index - 0.5) * width
        bars = ax.bar(
            positions, means, width=width, yerr=errors, color=colors[index],
            edgecolor="#333333", linewidth=0.45, hatch=hatches[index],
            capsize=2.0, error_kw={"linewidth": 0.65}, label=graph_type,
            zorder=3,
        )
        for bar, value in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.035,
                    "{:.2f}".format(value), ha="center", va="bottom", fontsize=5.4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.0)
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("Similarity", fontsize=6.2)
    ax.set_title("(d) Graph stability comparison", fontsize=7.3, loc="left",
                 pad=4.0, fontweight="semibold")
    ax.grid(axis="y", linewidth=0.35, color="#D8D8D8", alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=5.8, width=0.5, length=2.5)
    ax.legend(frameon=False, fontsize=5.6, loc="lower right",
              handlelength=1.5, borderpad=0.1)


def _save_figure(fig, output_dir):
    stem = os.path.join(output_dir, "fig6_graph_pattern_analysis")
    pdf_metadata = {
        "Title": "Analysis of Learned Channel Interaction Patterns",
        "Author": "Generated from real LOSO checkpoints",
        "Subject": "IEEE Transactions Fig. 6",
        "Keywords": "EEG, channel interaction graph, stability, node response",
    }
    fig.savefig(stem + ".pdf", dpi=600, bbox_inches="tight", metadata=pdf_metadata)
    fig.savefig(stem + ".png", dpi=600, bbox_inches="tight",
                metadata={"Description": pdf_metadata["Subject"]})
    fig.savefig(stem + ".svg", dpi=600, bbox_inches="tight", metadata={
        "Title": pdf_metadata["Title"], "Description": pdf_metadata["Subject"],
        "Creator": pdf_metadata["Author"],
        "Keywords": pdf_metadata["Keywords"].split(", "),
    })
    return [os.path.abspath(stem + suffix) for suffix in [".pdf", ".png", ".svg"]]


def _dataset_context(spec, args):
    fs = _target_fs(spec, args)
    cache = default_cache_path(
        spec["dataset"], "loso", cog_paradigm=spec["paradigm"],
        target_fs=fs, cache_root=args.cache_root,
    )
    sessions = (1, 2, 3) if spec["dataset"] == "cog-bci" else (1,)
    dataset = load_dataset(
        spec["dataset"], data_root=args.data_root, cache=cache,
        rebuild_cache=False, cog_paradigm=spec["paradigm"], sessions=sessions,
        target_fs=fs, window_sec=1.0, stride_sec=1.0,
    )
    return {
        "spec": spec, "dataset": dataset,
        "domains": domain_ids(dataset, "loso"), "cache": os.path.abspath(cache),
    }


def _class_display_names(class_ids):
    if len(class_ids) == 2:
        return ["Low", "High"]
    if len(class_ids) == 3:
        return ["Low", "Medium", "High"]
    return ["Class {}".format(value) for value in class_ids]


def _output_directory(root, spec, multiple):
    return os.path.join(root, spec["name"]) if multiple else root


def _run_dataset(spec, args, master, overrides, multiple):
    output_dir = _output_directory(args.output_dir, spec, multiple)
    os.makedirs(output_dir, exist_ok=True)
    context = _dataset_context(spec, args)
    run_info = _resolve_run(spec, args, master)
    _validate_run(run_info, args)
    config = _model_config(run_info["record"])
    split_config = _split_config(run_info)
    completed = sorted(set(pd.to_numeric(
        run_info["summary"]["subject"], errors="coerce"
    ).dropna().astype(int)))
    if len(completed) < 2:
        raise ValueError("Graph stability needs at least two completed LOSO folds")
    if spec["name"] in overrides:
        representative = overrides[spec["name"]]
        selection = {"selected_target_subject": representative, "selection_rule": "explicit"}
    elif spec["dataset"] in overrides:
        representative = overrides[spec["dataset"]]
        selection = {"selected_target_subject": representative, "selection_rule": "explicit"}
    else:
        selection = choose_median_fold(run_info["summary"])
        representative = int(selection["selected_target_subject"])
    if representative not in completed:
        raise ValueError("Representative subject is absent from completed LOSO folds")

    fold_results, payloads, cache_paths = [], [], []
    for subject in completed:
        print("Extracting {} fold {}/{}".format(spec["display_name"], subject, len(completed)))
        result, payload = _fold_analysis(
            context, run_info, subject, subject == representative,
            config, split_config, args,
        )
        fold_results.append(result)
        payloads.append(payload)
        cache_paths.append(os.path.join(
            args.feature_cache_dir, spec["name"],
            args.model + ("_" + args.bnorm if args.model == "tsmnet" else ""),
            "subject_{:02d}".format(int(subject)), "fold_graph_analysis.npz",
        ))

    adjacencies, channels = load_adjacency_matrices(
        cache_paths, expected_channels=context["dataset"]["channels"]
    )
    mean_adjacency, prevalence, stable_mask = _stable_edges(
        adjacencies, args.top_edge_fraction, args.min_edge_prevalence
    )
    stable_count = int(np.count_nonzero(np.triu(stable_mask, k=1)))
    if stable_count == 0:
        warnings.warn(
            "No edge meets both top-{:.0f}% mean weight and {:.0f}% fold prevalence. "
            "The prespecified thresholds are retained.".format(
                100 * args.top_edge_fraction, 100 * args.min_edge_prevalence
            ), UserWarning,
        )

    shared_edge = _per_fold_shared_stability(adjacencies, compute_edge_stability)
    shared_weight = _per_fold_shared_stability(adjacencies, compute_weight_stability)
    dynamic_edge, dynamic_weight = [], []
    for index, result in enumerate(fold_results):
        dynamic_edge.append(float(np.mean(compute_edge_stability(
            result["dynamic_graphs"], args.max_stability_pairs,
            args.seed + completed[index],
        ))))
        dynamic_weight.append(float(np.mean(compute_weight_stability(
            result["dynamic_graphs"], args.max_stability_pairs,
            args.seed + completed[index],
        ))))
    stability = pd.concat([
        pd.DataFrame({
            "subject": completed, "graph_type": "Dynamic graph",
            "edge_jaccard": dynamic_edge, "weight_cosine": dynamic_weight,
        }),
        pd.DataFrame({
            "subject": completed, "graph_type": "Shared adaptive graph",
            "edge_jaccard": shared_edge, "weight_cosine": shared_weight,
        }),
    ], ignore_index=True)
    if args.model == "tsmnet":
        stability.loc[stability["graph_type"] == "Shared adaptive graph", "graph_type"] = "Shared model proxy"

    representative_index = completed.index(representative)
    representative_result = fold_results[representative_index]
    if "source_response_mean" not in representative_result:
        raise RuntimeError("Representative fold cache lacks source response statistics")
    target_z = (
        representative_result["target_response"]
        - representative_result["source_response_mean"][None, :]
    ) / representative_result["source_response_std"][None, :]
    labels = representative_result["target_labels"].astype(np.int64)
    class_ids = sorted(int(value) for value in np.unique(labels))
    responses = np.stack([target_z[labels == value].mean(axis=0) for value in class_ids])
    class_names = _class_display_names(class_ids)
    node_strength = mean_adjacency.sum(axis=1)
    positions = load_channel_info(channels)

    np.save(os.path.join(output_dir, "fig6_mean_adjacency.npy"), mean_adjacency)
    np.savez_compressed(
        os.path.join(output_dir, "fig6_fold_adjacencies.npz"),
        subjects=np.asarray(completed, dtype=np.int64),
        channels=np.asarray(channels, dtype="U32"), adjacencies=adjacencies,
    )
    edge_rows = []
    for row, col in zip(*np.triu_indices(len(channels), k=1)):
        edge_rows.append({
            "channel_1": channels[row], "channel_2": channels[col],
            "mean_weight": mean_adjacency[row, col],
            "fold_prevalence": prevalence[row, col],
            "stable_edge": bool(stable_mask[row, col]),
        })
    pd.DataFrame(edge_rows).to_csv(
        os.path.join(output_dir, "fig6_edge_statistics.csv"), index=False
    )
    stability.to_csv(os.path.join(output_dir, "fig6_stability_by_fold.csv"), index=False)
    response_frame = pd.DataFrame(responses, columns=channels)
    response_frame.insert(0, "class_id", class_ids)
    response_frame.insert(1, "class_name", class_names)
    response_frame.to_csv(os.path.join(output_dir, "fig6_node_responses.csv"), index=False)
    positions.to_csv(os.path.join(output_dir, "fig6_channel_positions.csv"), index=False)

    fig = plt.figure(figsize=FIGURE_SIZE)
    outer = fig.add_gridspec(
        2, 3, width_ratios=[1.28, 1.0, 1.05], height_ratios=[1.0, 0.90],
        left=0.055, right=0.985, bottom=0.09, top=0.94,
        wspace=0.32, hspace=0.35,
    )
    heatmap_ax = fig.add_subplot(outer[0, 0])
    graph_ax = fig.add_subplot(outer[0, 1])
    strength_ax = fig.add_subplot(outer[0, 2])
    title = (
        "Mean learned adjacency matrix"
        if args.model == "ms_tgc_spddsbn"
        else "Mean spatial-filter interaction proxy"
    )
    plot_adjacency_heatmap(heatmap_ax, mean_adjacency, channels, title)
    plot_scalp_graph(
        graph_ax, mean_adjacency, stable_mask, positions, node_strength,
        "Stable learned channel interactions" if args.model == "ms_tgc_spddsbn"
        else "Stable spatial-filter interactions",
    )
    plot_node_strength_topomap(strength_ax, positions, node_strength)
    plot_node_response_maps(outer[1, :2], positions, responses, class_names, fig)
    stability_ax = fig.add_subplot(outer[1, 2])
    plot_graph_stability(stability_ax, stability)
    if args.model == "tsmnet":
        handles, labels_ = stability_ax.get_legend_handles_labels()
        labels_ = ["Shared model proxy" if value == "Shared adaptive graph" else value for value in labels_]
        stability_ax.legend(handles, labels_, frameon=False, fontsize=5.6, loc="lower right")
    fig.text(0.5, 0.025, spec["display_name"], ha="center", fontsize=6.5,
             color="#4A4A4A", fontweight="semibold")
    paths = _save_figure(fig, output_dir)
    plt.close(fig)

    dynamic_means = stability[stability["graph_type"] == "Dynamic graph"]
    shared_label = "Shared adaptive graph" if args.model == "ms_tgc_spddsbn" else "Shared model proxy"
    shared_means = stability[stability["graph_type"] == shared_label]
    if (
        shared_means["edge_jaccard"].mean() <= dynamic_means["edge_jaccard"].mean()
        or shared_means["weight_cosine"].mean() <= dynamic_means["weight_cosine"].mean()
    ):
        warnings.warn(
            "The prespecified shared representation is not more stable on both "
            "metrics. Results are retained without threshold tuning.", UserWarning,
        )
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv), "dataset": spec,
        "model": args.model, "bnorm": args.bnorm if args.model == "tsmnet" else None,
        "interaction_semantics": payloads[0]["interaction"],
        "feature_locations": payloads[representative_index]["features"],
        "representative_fold": selection, "completed_loso_subjects": completed,
        "checkpoint_root": run_info["run_dir"], "summary": run_info["summary_path"],
        "master_summary": os.path.abspath(args.master_summary), "dataset_cache": context["cache"],
        "model_config": config, "split_config": split_config,
        "stable_edge_rule": {
            "top_mean_weight_fraction": float(args.top_edge_fraction),
            "minimum_fold_prevalence": float(args.min_edge_prevalence),
            "stable_edge_count": stable_count,
        },
        "dynamic_graph_definition": {
            "features": "shared temporal node maps before graph/spatial interaction",
            "weights": "absolute within-window Pearson correlation",
            "sparsity": "same undirected edge count as the fold shared interaction",
            "requested_windows_per_fold": int(args.dynamic_windows_per_fold),
            "windows_per_fold_upper_bound": int(args.dynamic_windows_per_fold)
            + int(len(np.unique(representative_result["target_labels"]))) - 1,
            "actual_windows_by_fold": {
                str(subject): int(len(result["dynamic_ids"]))
                for subject, result in zip(completed, fold_results)
            },
        },
        "stability_definition": {
            "dynamic": "mean across-window pair similarity within each target fold",
            "shared": "mean similarity of each fold interaction to all other folds",
            "paired_fold_sample_size": int(len(completed)),
            "edge_metric": "Jaccard similarity of nonzero upper-triangle edge sets",
            "weight_metric": "cosine similarity of weighted upper triangles",
            "error_bars": "standard deviation across LOSO folds",
        },
        "node_response_definition": {
            "mstgc": "RMS post-Chebyshev node map before channel reliability and SPD pooling",
            "tsmnet": "RMS temporal response weighted by spatial-kernel channel norm",
            "normalization": "per-channel source-train mean and standard deviation",
            "grouping": "representative target-test windows grouped by true class after training",
        },
        "target_label_usage": (
            "offline class-balanced window selection and response grouping only; "
            "never model reconstruction, graph learning, or graph-weight estimation"
        ),
        "position_source": str(positions["position_source"].iloc[0]),
        "outputs": paths,
        "versions": {
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "torch": torch.__version__,
            "matplotlib": matplotlib.__version__, "networkx": nx.__version__,
        },
    }
    _write_json(metadata, os.path.join(output_dir, "fig6_graph_pattern_metadata.json"))
    print("Representative target subject:", representative)
    print("Stable edge count:", stable_count)
    print("Stability means:")
    print(stability.groupby("graph_type")[["edge_jaccard", "weight_cosine"]].mean())
    print("Saved:", output_dir)
    return metadata


def main():
    args = parse_args()
    if args.batch_size < 1 or args.dynamic_windows_per_fold < 2:
        raise ValueError("Batch size must be positive and dynamic windows at least 2")
    if args.response_windows_per_class < 1 or args.max_stability_pairs < 1:
        raise ValueError("Response windows and stability pairs must be positive")
    if not 0.0 < args.top_edge_fraction <= 1.0:
        raise ValueError("--top-edge-fraction must be in (0,1]")
    if not 0.0 <= args.min_edge_prevalence <= 1.0:
        raise ValueError("--min-edge-prevalence must be in [0,1]")
    datasets = _parse_datasets(args.datasets, args.dataset_labels)
    overrides = _target_overrides(args.target_subjects)
    if not os.path.exists(args.master_summary):
        if not args.allow_missing_master_config:
            raise FileNotFoundError(
                "Master summary not found: {}".format(args.master_summary)
            )
        master = pd.DataFrame()
    else:
        master = pd.read_csv(args.master_summary)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    args.device_object = torch.device("cpu" if device == "auto" else device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    matplotlib.rcParams.update({
        "font.family": _font_family(), "font.size": 7.0,
        "axes.linewidth": 0.6, "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.facecolor": "white",
        "svg.hashsalt": "fig6-channel-interaction-2026",
    })
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.feature_cache_dir, exist_ok=True)
    metadata = []
    for spec in datasets:
        metadata.append(_run_dataset(
            spec, args, master, overrides, multiple=len(datasets) > 1
        ))
    return metadata


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit("Fig. 6 input error: {}".format(exc))
