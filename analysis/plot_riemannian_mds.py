"""Joint AIRM-MDS visualization of class Frechet centers before/after SPDDSBN."""

import argparse
import inspect
import json
import os
import platform
import random
import sys
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.linalg import eigvalsh, orthogonal_procrustes
from scipy.spatial.distance import pdist, squareform
import sklearn
from sklearn.manifold import MDS
import torch


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.cl_tsmnet.datasets import load_dataset
from src.cl_tsmnet.experiment_utils import (
    default_cache_path,
    default_target_fs,
    run_directory_name,
)
from src.cl_tsmnet.spd_pca import migrate_legacy_spddsbn_buffers
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


MAIN_MDS_SEED = 2026
CLASS_COLORS = {
    "low": "#3B75AF",
    "medium": "#D9A62E",
    "high": "#D9534F",
}
FALLBACK_COLORS = ["#3B75AF", "#D9534F", "#D9A62E", "#6F4E9C", "#2A9D8F"]
MODEL_CONFIG_DEFAULTS = {
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


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize subject-by-class SPD Frechet centers with one joint "
            "AIRM metric-MDS embedding."
        )
    )
    parser.add_argument("--dataset", choices=["stew", "eegmat", "cog-bci"], required=True)
    parser.add_argument("--cog-paradigm", choices=["nback", "matb"], default="nback")
    parser.add_argument("--model", choices=sorted(SUPPORTED_SPD_VISUALIZATION_MODELS),
                        default="ms_tgc_spddsbn")
    parser.add_argument("--checkpoint-root", default=None)
    parser.add_argument("--results-file", default=None)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--master-summary", default=None)
    parser.add_argument("--target-subject", type=int, default=None)
    parser.add_argument(
        "--feature-cache-dir", default=None,
        help=(
            "Optional directory containing previously extracted "
            "spd_intermediates.npz and metadata. By default the script reads "
            "the original LOSO summary/checkpoint under --output-root."
        ),
    )
    parser.add_argument("--output-dir", default=os.path.join(
        "results", "figures", "riemannian_mds"
    ))
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--cache", default=None)
    parser.add_argument("--cache-root", default=os.path.join("outputs", "cache"))
    parser.add_argument("--target-fs", type=float, default=None)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--val-size", type=float, default=None)
    parser.add_argument("--test-size", type=float, default=None)
    parser.add_argument("--artifact-z", type=float, default=None)
    parser.add_argument("--karcher-tol", type=float, default=1e-7)
    parser.add_argument("--karcher-max-iter", type=int, default=50)
    parser.add_argument("--stability-seeds", default="7,42,2026,3407,9001")
    parser.add_argument("--stress-3d-threshold", type=float, default=0.15)
    parser.add_argument("--annotate-distances", action="store_true")
    parser.add_argument(
        "--reuse-centroids", action="store_true",
        help=(
            "Resume from riemannian_mds_centroids.npz and the matching "
            "coordinate metadata in --output-dir. This is explicit to prevent "
            "accidental reuse across runs."
        ),
    )
    parser.add_argument(
        "--reuse-mds", action="store_true",
        help=(
            "Resume plotting from the existing distance matrix, coordinates, "
            "and metadata in --output-dir after validating center IDs."
        ),
    )
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


def _write_json(payload, path):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, indent=2, ensure_ascii=False,
                  allow_nan=False)


def _read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_feature_cache_dir(path, output_dir):
    """Validate an explicitly supplied extracted-feature directory."""
    if path is None:
        return None
    path = os.path.abspath(path)
    if os.path.isdir(path):
        return path
    raise FileNotFoundError(
        "--feature-cache-dir is not an extracted feature directory: {}. "
        "Remove this argument to extract from the original LOSO checkpoint."
        .format(path)
    )


def _dataset_name(args):
    if args.dataset == "cog-bci":
        return "cog-bci-{}".format(args.cog_paradigm)
    return args.dataset


def _normalize_path(path):
    return os.path.normcase(os.path.abspath(os.path.normpath(str(path))))


def _usable_value(value):
    if value is None:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return value


def _matching_master_record(args):
    master_path = args.master_summary or os.path.join(
        args.output_root, "master_summary.csv"
    )
    args.master_summary = master_path
    if not os.path.exists(master_path):
        return {}
    frame = pd.read_csv(master_path)
    if "protocol" in frame:
        frame = frame[frame["protocol"].astype(str) == "loso"]
    if "dataset" in frame:
        frame = frame[frame["dataset"].astype(str) == _dataset_name(args)]
    if "model_type" in frame:
        frame = frame[frame["model_type"].astype(str) == args.model]
    if frame.empty:
        return {}
    if "output_dir" in frame:
        populated = frame[frame["output_dir"].map(_usable_value).notna()]
        exact = populated[populated["output_dir"].astype(str).map(
            lambda value: _normalize_path(value) == _normalize_path(
                args.checkpoint_root
            )
        )]
        if not exact.empty:
            frame = exact
        elif not populated.empty:
            return {}
    return frame.iloc[-1].dropna().to_dict()


def _resolve_original_output_inputs(args):
    if args.checkpoint_root is None:
        run_name = run_directory_name(
            _dataset_name(args), "loso", args.model, "spddsbn"
        )
        args.checkpoint_root = os.path.join(args.output_root, run_name)
    if args.results_file is None:
        args.results_file = os.path.join(args.checkpoint_root, "summary.csv")

    run_record = _matching_master_record(args)
    args.run_record = run_record
    for key, default in MODEL_CONFIG_DEFAULTS.items():
        current = getattr(args, key)
        if current is None:
            value = _usable_value(run_record.get(key))
            setattr(args, key, default if value is None else value)
    split_defaults = {"seed": 42, "val_size": 0.2, "test_size": 0.2}
    for key, default in split_defaults.items():
        current = getattr(args, key)
        if current is None:
            value = _usable_value(run_record.get(key))
            setattr(args, key, default if value is None else value)
    if args.target_fs is None:
        value = _usable_value(run_record.get("target_fs"))
        if value is not None:
            args.target_fs = float(value)
    if args.cache is None:
        value = _usable_value(run_record.get("cache"))
        if value is not None and os.path.exists(str(value)):
            args.cache = str(value)
    if args.artifact_z is None:
        value = _usable_value(run_record.get("artifact_z"))
        if value is not None:
            args.artifact_z = float(value)

    if args.feature_cache_dir is None and not os.path.isdir(args.checkpoint_root):
        raise FileNotFoundError(
            "Original LOSO output directory was not found: {}. Expected "
            "summary.csv and subject_XX/model.pt under this directory. Use "
            "--checkpoint-root only if the run was saved elsewhere."
            .format(os.path.abspath(args.checkpoint_root))
        )
    print("LOSO output directory:", os.path.abspath(args.checkpoint_root))
    print("Per-subject results:", os.path.abspath(args.results_file))
    if run_record:
        print("Training configuration: matched", os.path.abspath(args.master_summary))
    else:
        print("Training configuration: defaults/explicit CLI (no matching master row)")


def _validate_selected_result(args, selection):
    results_file = selection.get("results_file")
    if not results_file or not os.path.exists(results_file):
        return {"status": "unavailable_explicit_subject"}
    frame = pd.read_csv(results_file)
    rows = frame[pd.to_numeric(
        frame["subject"], errors="coerce"
    ) == int(selection["selected_target_subject"])]
    if "protocol" in rows:
        rows = rows[rows["protocol"].astype(str) == "loso"]
    if "dataset" in rows:
        rows = rows[rows["dataset"].astype(str) == _dataset_name(args)]
    if "model_type" in rows:
        rows = rows[rows["model_type"].astype(str) == args.model]
    if rows.empty:
        raise ValueError("Selected subject/model is absent from summary.csv")
    required = {"target_adapt", "target_refit_scope"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(
            "Original summary lacks required SPDDSBN audit columns: {}. "
            "Use results produced by the current training pipeline."
            .format(sorted(missing))
        )
    enabled = rows["target_adapt"].astype(str).str.lower().isin(
        ["true", "1", "yes"]
    )
    if not bool(enabled.all()):
        raise ValueError(
            "Riemannian alignment visualization requires a checkpoint "
            "evaluated with unlabeled target SPDDSBN adaptation"
        )
    scopes = sorted(set(
        value for value in rows["target_refit_scope"].dropna().astype(str)
        if value
    ))
    if scopes != ["target_only"]:
        raise ValueError(
            "Expected target_refit_scope=target_only, found {}".format(scopes)
        )
    if args.model == "tsmnet" and "bnorm" in rows:
        bnorms = set(rows["bnorm"].dropna().astype(str))
        if bnorms != {"spddsbn"}:
            raise ValueError("TSMNet visualization requires bnorm=spddsbn")
    return {
        "status": "validated_from_summary",
        "row_count": int(len(rows)),
        "target_adapt": True,
        "target_refit_scope": scopes,
        "best_epochs": (
            sorted(pd.to_numeric(rows["best_epoch"], errors="coerce")
                   .dropna().astype(int).unique().tolist())
            if "best_epoch" in rows else []
        ),
    }


def _select_representative(results_file, dataset_name, model_type):
    if not results_file or not os.path.exists(results_file):
        raise FileNotFoundError(
            "Per-subject LOSO results were not found. Supply --target-subject "
            "explicitly; a representative fold will not be guessed."
        )
    frame = pd.read_csv(results_file)
    if "protocol" in frame:
        frame = frame[frame["protocol"].astype(str) == "loso"]
    if "dataset" in frame:
        frame = frame[frame["dataset"].astype(str) == dataset_name]
    if "model_type" in frame:
        frame = frame[frame["model_type"].astype(str) == model_type]
    metric = "test_bacc" if "test_bacc" in frame else "balanced_accuracy"
    required = {"subject", metric}
    missing = required - set(frame.columns)
    if missing or frame.empty:
        raise ValueError(
            "No matching LOSO subject rows with {} exist in {}".format(
                metric, results_file
            )
        )
    clean = frame[["subject", metric]].copy()
    clean["subject"] = pd.to_numeric(clean["subject"], errors="coerce")
    clean[metric] = pd.to_numeric(clean[metric], errors="coerce")
    clean = clean.dropna()
    grouped = clean.groupby("subject", sort=True)[metric].agg(["mean", "count"])
    median = float(np.median(grouped["mean"].to_numpy(dtype=np.float64)))
    grouped["distance"] = np.abs(grouped["mean"] - median)
    grouped = grouped.reset_index().sort_values(["distance", "subject"])
    row = grouped.iloc[0]
    return {
        "selected_target_subject": int(row["subject"]),
        "subject_balanced_accuracy": float(row["mean"]),
        "all_subject_median_balanced_accuracy": median,
        "subject_seed_rows_averaged": int(row["count"]),
        "selection_rule": "closest_to_median_then_lowest_subject_id",
        "results_file": os.path.abspath(results_file),
    }


def _selection_from_args(args):
    dataset_name = _dataset_name(args)
    results_file = args.results_file
    if results_file is None and args.checkpoint_root:
        candidate = os.path.join(args.checkpoint_root, "summary.csv")
        results_file = candidate if os.path.exists(candidate) else None
    if args.target_subject is None:
        return _select_representative(results_file, dataset_name, args.model)
    selection = {
        "selected_target_subject": int(args.target_subject),
        "subject_balanced_accuracy": None,
        "all_subject_median_balanced_accuracy": None,
        "subject_seed_rows_averaged": None,
        "selection_rule": "explicit_target_subject",
        "results_file": os.path.abspath(results_file) if results_file else None,
    }
    if args.feature_cache_dir:
        evidence_path = os.path.join(args.feature_cache_dir, "representative_fold.json")
        if os.path.exists(evidence_path):
            evidence = _read_json(evidence_path)
            if int(evidence.get("selected_target_subject", -1)) != int(args.target_subject):
                raise ValueError(
                    "Explicit target subject conflicts with cached representative-fold metadata"
                )
            selection["cached_selection_evidence"] = evidence
    return selection


def _symmetrize(matrices):
    matrices = np.asarray(matrices, dtype=np.float64)
    return 0.5 * (matrices + np.swapaxes(matrices, -1, -2))


def validate_spd(matrices, name, symmetry_atol=1e-8):
    matrices = np.asarray(matrices, dtype=np.float64)
    if matrices.ndim != 3 or matrices.shape[-1] != matrices.shape[-2]:
        raise ValueError("{} must be [N,d,d], got {}".format(name, matrices.shape))
    if not np.isfinite(matrices).all():
        raise ValueError("{} contains NaN or Inf".format(name))
    symmetry_error = float(np.max(np.abs(
        matrices - np.swapaxes(matrices, -1, -2)
    )))
    if symmetry_error > float(symmetry_atol):
        raise ValueError(
            "{} symmetry error {:.3e} exceeds {:.3e}".format(
                name, symmetry_error, symmetry_atol
            )
        )
    eigenvalues = np.linalg.eigvalsh(_symmetrize(matrices))
    minimum = float(np.min(eigenvalues))
    invalid = int(np.sum(np.min(eigenvalues, axis=1) <= 0.0))
    if invalid:
        raise ValueError(
            "{} contains {} non-SPD matrices; minimum eigenvalue {:.3e}".format(
                name, invalid, minimum
            )
        )
    return {
        "shape": [int(value) for value in matrices.shape],
        "minimum_eigenvalue": minimum,
        "maximum_symmetry_error": symmetry_error,
        "non_finite_values": 0,
        "non_positive_matrices": 0,
    }


def airm_distance(left, right):
    """Affine-invariant distance using generalized SPD eigenvalues."""
    left = _symmetrize(left)
    right = _symmetrize(right)
    values = eigvalsh(right, left, check_finite=False)
    if not np.isfinite(values).all() or float(np.min(values)) <= 0.0:
        raise ValueError("AIRM distance received a non-SPD matrix pair")
    return float(np.linalg.norm(np.log(values)))


def pairwise_airm(matrices):
    matrices = np.asarray(matrices, dtype=np.float64)
    count = len(matrices)
    distances = np.zeros((count, count), dtype=np.float64)
    for left in range(count):
        for right in range(left + 1, count):
            value = airm_distance(matrices[left], matrices[right])
            distances[left, right] = value
            distances[right, left] = value
    return distances


def frechet_mean(matrices, tolerance=1e-7, max_iterations=50):
    tsmnet_root = os.path.join(PROJECT_ROOT, "TSMNet")
    if tsmnet_root not in sys.path:
        sys.path.insert(0, tsmnet_root)
    from spdnets.functionals import spd_mean_kracher_flow

    matrices = _symmetrize(matrices)
    validate_spd(matrices, "Frechet_mean_input")
    tensor = torch.from_numpy(matrices).to(dtype=torch.double, device="cpu")
    with torch.no_grad():
        mean, diagnostics = spd_mean_kracher_flow(
            tensor, maxiter=int(max_iterations), dim=0,
            tolerance=float(tolerance), return_info=True,
        )
    mean = _symmetrize(mean.squeeze(0).detach().cpu().numpy())
    if not diagnostics["converged"]:
        raise RuntimeError(
            "Frechet mean failed to converge after {} iterations (residual {:.3e})"
            .format(diagnostics["iterations"], diagnostics["residual"])
        )
    validate_spd(mean[None], "Frechet_mean_output")
    return mean, diagnostics


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


def _checkpoint_path(root, subject):
    if not root:
        raise ValueError(
            "--checkpoint-root is required when a reusable SPD feature cache is absent"
        )
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
    raise FileNotFoundError("No checkpoint found; tried {}".format(candidates))


def _model_config(args):
    graph_density = args.mstgc_graph_density
    if graph_density is not None:
        graph_density = float(graph_density)
    return {
        "temporal_filters": int(args.temporal_filters),
        "spatial_filters": int(args.spatial_filters),
        "subspacedims": int(args.subspacedims),
        "temp_kernel": int(args.temp_kernel),
        "mstgc_temporal_hidden": int(args.mstgc_temporal_hidden),
        "mstgc_graph_hidden": int(args.mstgc_graph_hidden),
        "mstgc_fusion_dim": int(args.mstgc_fusion_dim),
        "mstgc_kernel_length": int(args.mstgc_kernel_length),
        "mstgc_num_heads": int(args.mstgc_num_heads),
        "mstgc_cheby_order": int(args.mstgc_cheby_order),
        "mstgc_dropout": float(args.mstgc_dropout),
        "mstgc_num_nodes": int(args.mstgc_num_nodes),
        "mstgc_graph_k": int(args.mstgc_graph_k),
        "mstgc_graph_density": graph_density,
        "mstgc_time_points": int(args.mstgc_time_points),
        "mstgc_shrinkage": float(args.mstgc_shrinkage),
    }


def _build_model(args, dataset, domains, selected_indices, nclasses, device):
    config = _model_config(args)
    shape = dataset["x"].shape
    if args.model == "tsmnet":
        return build_tsmnet(
            PROJECT_ROOT, shape[1], shape[2], nclasses,
            domains[selected_indices], bnorm="spddsbn",
            temporal_filters=config["temporal_filters"],
            spatial_filters=config["spatial_filters"],
            subspacedims=config["subspacedims"],
            temp_kernel=config["temp_kernel"], device=device,
        ).to(device), config
    kernel_samples = max(3, int(round(
        config["mstgc_kernel_length"] * float(dataset["fs"]) / 128.0
    )))
    return build_ms_tgc_spddsbn(
        PROJECT_ROOT, shape[1], shape[2], nclasses,
        domains[selected_indices], subspacedims=config["subspacedims"],
        device=device, temporal_hidden=config["mstgc_temporal_hidden"],
        graph_hidden=config["mstgc_graph_hidden"],
        fusion_dim=config["mstgc_fusion_dim"], kernel_length=kernel_samples,
        num_heads=config["mstgc_num_heads"],
        cheby_order=config["mstgc_cheby_order"],
        dropout=config["mstgc_dropout"], num_nodes=config["mstgc_num_nodes"],
        variant=args.model, graph_mode="adaptive",
        graph_neighbors=config["mstgc_graph_k"],
        graph_density=config["mstgc_graph_density"],
        graph_time_points=config["mstgc_time_points"],
        covariance_shrinkage=config["mstgc_shrinkage"],
    ).to(device), config


def _extract_partition(model, model_type, dataset, indices, domains,
                       normalizer, batch_size, device):
    pre_parts, post_parts = [], []
    indices = np.asarray(indices, dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), int(batch_size)):
            ids = indices[start:start + int(batch_size)]
            windows = normalizer.transform_array(dataset["x"][ids])
            xb = torch.from_numpy(windows).to(device=device, dtype=torch.float32)
            db = torch.from_numpy(domains[ids]).to(device=device, dtype=torch.long)
            _, features = extract_spd_intermediates(model, xb, db, model_type)
            pre_parts.append(features["spd_pre_bn"].detach().cpu().double().numpy())
            post_parts.append(features["spd_post_bn"].detach().cpu().double().numpy())
    return np.concatenate(pre_parts), np.concatenate(post_parts)


def _extract_from_checkpoint(args, selection):
    subject = int(selection["selected_target_subject"])
    checkpoint = _checkpoint_path(args.checkpoint_root, subject)
    print("Selected best checkpoint:", checkpoint)
    target_fs = default_target_fs(args.dataset, args.target_fs)
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
    split = make_split(
        dataset, "loso", subject, seed=args.seed,
        val_size=args.val_size, test_size=args.test_size,
    )
    normalizer = fit_source_normalizer(dataset["x"], split["train"])
    split = dict(split)
    for name in ["train", "val", "test"]:
        split[name] = _filter_artifact_windows(
            dataset["x"], split[name], normalizer, args.artifact_z
        )
        if len(split[name]) == 0:
            raise RuntimeError("Artifact filtering removed all {} windows".format(name))
    source_ids = np.asarray(split["train"], dtype=np.int64)
    target_ids = np.asarray(split["test"], dtype=np.int64)
    if np.intersect1d(source_ids, target_ids).size:
        raise RuntimeError("Source-train and target-test sample IDs overlap")
    domains = domain_ids(dataset, "loso")
    selected = np.concatenate([split["train"], split["val"], split["test"]])
    labels = np.unique(dataset["y"][split["train"]]).astype(np.int64)
    if not np.array_equal(labels, np.arange(len(labels))):
        raise ValueError("Source training labels are not contiguous 0..K-1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = _build_model(
        args, dataset, domains, selected, len(labels), device
    )
    state, migrations = migrate_legacy_spddsbn_buffers(
        _load_state(checkpoint), model.state_dict()
    )
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint architecture mismatch; pass the exact training "
            "hyperparameters. Original error:\n{}".format(exc)
        ) from exc
    source_pre, source_post = _extract_partition(
        model, args.model, dataset, source_ids, domains, normalizer,
        args.batch_size, device,
    )
    target_pre, target_post = _extract_partition(
        model, args.model, dataset, target_ids, domains, normalizer,
        args.batch_size, device,
    )
    sample_ids = np.concatenate([source_ids, target_ids])
    pre = np.concatenate([source_pre, target_pre])
    post = np.concatenate([source_post, target_post])
    selected_meta = dataset["meta"].iloc[sample_ids]
    class_ids = dataset["y"][sample_ids].astype(np.int64)
    label_names = dataset.get("label_names", {})
    metadata = pd.DataFrame({
        "sample_id": sample_ids,
        "subject_id": selected_meta["subject"].to_numpy(dtype=np.int64),
        "domain_id": domains[sample_ids].astype(np.int64),
        "domain": np.where(np.isin(sample_ids, source_ids), "source", "target"),
        "class_id": class_ids,
        "class_name": [
            str(label_names.get(int(value), "class {}".format(value)))
            for value in class_ids
        ],
        "split": np.where(
            np.isin(sample_ids, source_ids), "source_train", "target_test"
        ),
        "session": selected_meta["session"].to_numpy(dtype=np.int64),
        "task": selected_meta["task"].astype(str).to_numpy(),
        "start_sample": selected_meta["start_sample"].to_numpy(dtype=np.int64),
    })
    feature_dir = os.path.join(args.output_dir, "feature_cache")
    os.makedirs(feature_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(feature_dir, "spd_intermediates.npz"),
        sample_id=sample_ids, spd_pre_bn=pre, spd_post_bn=post,
    )
    metadata.to_csv(
        os.path.join(feature_dir, "spd_intermediates_metadata.csv"), index=False
    )
    provenance = {
        "source": "extracted_from_checkpoint",
        "checkpoint": checkpoint,
        "loso_results_file": os.path.abspath(args.results_file),
        "master_summary": (
            os.path.abspath(args.master_summary)
            if args.master_summary and os.path.exists(args.master_summary)
            else None
        ),
        "matched_master_record": getattr(args, "run_record", {}),
        "dataset_cache": os.path.abspath(cache),
        "model_type": args.model,
        "model_config": config,
        "checkpoint_buffer_migrations": migrations,
        "feature_locations": visualization_model_metadata(args.model),
        "source_partition": "source_train_only",
        "target_partition": "target_test_only",
    }
    _write_json(provenance, os.path.join(feature_dir, "extraction_provenance.json"))
    return pre, post, metadata, provenance


def _validate_cached_model(feature_dir, args, subject):
    run_config_path = os.path.join(feature_dir, "run_config.json")
    signature_path = os.path.join(feature_dir, "extraction_signature.json")
    provenance = {
        "source": "reused_validated_feature_cache",
        "feature_cache_dir": os.path.abspath(feature_dir),
        "model_type": args.model,
    }
    if os.path.exists(run_config_path):
        run_config = _read_json(run_config_path)
        provenance["run_config"] = run_config
        if str(run_config.get("dataset")) != _dataset_name(args):
            raise ValueError("Feature cache dataset does not match --dataset")
        if int(run_config.get("target_subject", -1)) != int(subject):
            raise ValueError("Feature cache target subject does not match selection")
        cached_type = run_config.get("model_type")
        if cached_type and str(cached_type) != args.model:
            raise ValueError("Feature cache contains a different model type")
        model_class = str(run_config.get("model_class", ""))
        expected_class = visualization_model_metadata(args.model)["model_class"]
        if model_class and model_class != expected_class:
            raise ValueError(
                "Feature cache model class {!r} does not match {!r}".format(
                    model_class, expected_class
                )
            )
        locations = run_config.get("feature_locations", {})
        if "BiMap" not in str(locations.get("pre", "")):
            raise ValueError("Cache does not document a BiMap/ReEig pre-SPDDSBN feature")
        if "LogEig" not in str(locations.get("post", "")):
            raise ValueError("Cache does not document a post-SPDDSBN/pre-LogEig feature")
    if os.path.exists(signature_path):
        signature = _read_json(signature_path)
        provenance["extraction_signature"] = signature
        if int(signature.get("target_subject", -1)) != int(subject):
            raise ValueError("Extraction signature target subject mismatch")
    return provenance


def _validate_cached_partition(args, metadata, subject, provenance):
    signature = provenance.get("extraction_signature", {})
    target_fs = default_target_fs(args.dataset, args.target_fs)
    dataset_cache = args.cache or default_cache_path(
        args.dataset, "loso", cog_paradigm=args.cog_paradigm,
        target_fs=target_fs, cache_root=args.cache_root,
    )
    if not os.path.exists(dataset_cache):
        return {
            "status": "dataset_cache_unavailable",
            "expected_cache": os.path.abspath(dataset_cache),
        }
    sessions = (1, 2, 3) if args.dataset == "cog-bci" else (1,)
    dataset = load_dataset(
        args.dataset, data_root=args.data_root, cache=dataset_cache,
        rebuild_cache=False, cog_paradigm=args.cog_paradigm,
        sessions=sessions, target_fs=target_fs,
    )
    split_seed = int(signature.get("split_seed", args.seed))
    val_size = float(signature.get("val_size", args.val_size))
    test_size = float(signature.get("test_size", args.test_size))
    artifact_z = signature.get("artifact_z", args.artifact_z)
    split = make_split(
        dataset, "loso", subject, seed=split_seed,
        val_size=val_size, test_size=test_size,
    )
    normalizer = fit_source_normalizer(dataset["x"], split["train"])
    filtered = {}
    for name in ["train", "val", "test"]:
        filtered[name] = _filter_artifact_windows(
            dataset["x"], split[name], normalizer, artifact_z
        )
    expected_ids = np.concatenate([filtered["train"], filtered["test"]])
    actual_ids = metadata["sample_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(expected_ids, actual_ids):
        raise ValueError(
            "Reusable feature cache sample IDs do not exactly reproduce the LOSO split"
        )
    expected_classes = dataset["y"][expected_ids].astype(np.int64)
    if not np.array_equal(
            expected_classes, metadata["class_id"].to_numpy(dtype=np.int64)):
        raise ValueError("Reusable feature-cache class IDs do not match the dataset")
    expected_subjects = dataset["meta"].iloc[expected_ids]["subject"].to_numpy(
        dtype=np.int64
    )
    if not np.array_equal(
            expected_subjects, metadata["subject_id"].to_numpy(dtype=np.int64)):
        raise ValueError("Reusable feature-cache subject IDs do not match the dataset")
    source_subjects = sorted(set(int(value) for value in expected_subjects[
        :len(filtered["train"])
    ]))
    validation_subjects = sorted(set(int(value) for value in dataset["meta"].iloc[
        filtered["val"]
    ]["subject"].to_numpy(dtype=np.int64)))
    return {
        "status": "exact_sample_id_class_subject_match",
        "dataset_cache": os.path.abspath(dataset_cache),
        "split_seed": split_seed,
        "val_size": val_size,
        "test_size": test_size,
        "artifact_z": artifact_z,
        "source_train_subjects": source_subjects,
        "excluded_source_validation_subjects": validation_subjects,
        "target_test_subject": int(subject),
        "source_train_windows": int(len(filtered["train"])),
        "source_validation_windows_excluded": int(len(filtered["val"])),
        "target_test_windows": int(len(filtered["test"])),
        "exclusion_reason": (
            "Validation-domain refit statistics are temporary during model "
            "selection and are not retained in the final target-refit checkpoint."
        ),
    }


def _load_or_extract_features(args, selection):
    if not args.feature_cache_dir:
        return _extract_from_checkpoint(args, selection)
    feature_path = os.path.join(args.feature_cache_dir, "spd_intermediates.npz")
    metadata_path = os.path.join(
        args.feature_cache_dir, "spd_intermediates_metadata.csv"
    )
    if not os.path.exists(feature_path) or not os.path.exists(metadata_path):
        return _extract_from_checkpoint(args, selection)
    subject = int(selection["selected_target_subject"])
    provenance = _validate_cached_model(args.feature_cache_dir, args, subject)
    with np.load(feature_path, allow_pickle=False) as saved:
        required = {"sample_id", "spd_pre_bn", "spd_post_bn"}
        if not required.issubset(saved.files):
            raise ValueError("Feature cache is missing {}".format(required - set(saved.files)))
        sample_ids = saved["sample_id"].astype(np.int64)
        pre = saved["spd_pre_bn"].astype(np.float64)
        post = saved["spd_post_bn"].astype(np.float64)
    metadata = pd.read_csv(metadata_path)
    required_columns = {
        "sample_id", "subject_id", "class_id", "class_name", "domain", "split"
    }
    if not required_columns.issubset(metadata.columns):
        raise ValueError(
            "Feature metadata is missing {}".format(
                required_columns - set(metadata.columns)
            )
        )
    metadata_ids = metadata["sample_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(sample_ids, metadata_ids):
        raise ValueError("Feature-cache sample IDs do not match metadata order")
    if pre.shape != post.shape or len(pre) != len(metadata):
        raise ValueError("Pre/post matrices and metadata are not one-to-one")
    target_subjects = set(metadata.loc[
        metadata["domain"].astype(str) == "target", "subject_id"
    ].astype(int))
    if target_subjects != {subject}:
        raise ValueError(
            "Feature cache target subjects {} do not equal selected subject {}"
            .format(sorted(target_subjects), subject)
        )
    allowed_splits = {"source_train", "target_test"}
    if not set(metadata["split"].astype(str)).issubset(allowed_splits):
        raise ValueError("Only source_train and target_test features may enter this analysis")
    provenance["split_validation"] = _validate_cached_partition(
        args, metadata, subject, provenance
    )
    return pre, post, metadata, provenance


def build_frechet_centers(pre, post, metadata, tolerance, max_iterations):
    records = []
    matrices = []
    diagnostics = []
    stages = [("pre", pre), ("post", post)]
    group_columns = ["subject_id", "class_id", "class_name", "domain", "split"]
    for stage, features in stages:
        grouped = metadata.reset_index().groupby(group_columns, sort=True)
        for values, group in grouped:
            subject, class_id, class_name, domain, split = values
            positions = group["index"].to_numpy(dtype=np.int64)
            center, info = frechet_mean(
                features[positions], tolerance=tolerance,
                max_iterations=max_iterations,
            )
            center_id = "subject_{:03d}_class_{}_{}".format(
                int(subject), int(class_id), stage
            )
            records.append({
                "center_id": center_id,
                "center_type": "subject_class",
                "subject_id": int(subject),
                "class_id": int(class_id),
                "class_name": str(class_name),
                "domain": str(domain),
                "split": str(split),
                "stage": stage,
                "n_windows": int(len(positions)),
            })
            matrices.append(center)
            diagnostics.append({
                "center_id": center_id,
                "iterations": int(info["iterations"]),
                "residual": float(info["residual"]),
                "converged": bool(info["converged"]),
            })

    center_frame = pd.DataFrame(records)
    center_matrices = np.asarray(matrices, dtype=np.float64)
    for stage in ["pre", "post"]:
        stage_mask = center_frame["stage"].astype(str) == stage
        source = center_frame[stage_mask & (
            center_frame["domain"].astype(str) == "source"
        )]
        for class_id, class_rows in source.groupby("class_id", sort=True):
            positions = class_rows.index.to_numpy(dtype=np.int64)
            reference, info = frechet_mean(
                center_matrices[positions], tolerance=tolerance,
                max_iterations=max_iterations,
            )
            class_name = str(class_rows.iloc[0]["class_name"])
            center_id = "source_reference_class_{}_{}".format(int(class_id), stage)
            records.append({
                "center_id": center_id,
                "center_type": "source_class_reference",
                "subject_id": -1,
                "class_id": int(class_id),
                "class_name": class_name,
                "domain": "source_reference",
                "split": "source_train_equal_subject_weight",
                "stage": stage,
                "n_windows": int(class_rows["n_windows"].sum()),
            })
            matrices.append(reference)
            diagnostics.append({
                "center_id": center_id,
                "iterations": int(info["iterations"]),
                "residual": float(info["residual"]),
                "converged": bool(info["converged"]),
                "n_subject_centers": int(len(positions)),
            })
    center_frame = pd.DataFrame(records)
    center_matrices = np.asarray(matrices, dtype=np.float64)
    if len(center_frame) != len(center_matrices):
        raise RuntimeError("Frechet center metadata/matrix alignment failed")
    return center_matrices, center_frame, diagnostics


def load_saved_centers(output_dir):
    center_path = os.path.join(output_dir, "riemannian_mds_centroids.npz")
    coordinate_path = os.path.join(output_dir, "riemannian_mds_coordinates.csv")
    if not os.path.exists(center_path) or not os.path.exists(coordinate_path):
        raise FileNotFoundError(
            "--reuse-centroids requires existing centroid NPZ and coordinate CSV"
        )
    with np.load(center_path, allow_pickle=False) as saved:
        center_ids = saved["center_id"].astype(str)
        matrices = saved["center_matrix"].astype(np.float64)
    frame = pd.read_csv(coordinate_path)
    if "center_id" not in frame:
        raise ValueError("Saved coordinate metadata has no center_id")
    if not np.array_equal(center_ids, frame["center_id"].astype(str).to_numpy()):
        raise ValueError("Saved centroid matrices and metadata IDs do not match")
    coordinate_columns = [
        column for column in frame.columns
        if column.startswith("mds")
    ]
    frame = frame.drop(columns=coordinate_columns)
    required = {
        "center_id", "center_type", "subject_id", "class_id", "class_name",
        "domain", "split", "stage", "n_windows",
    }
    if not required.issubset(frame.columns):
        raise ValueError("Saved center metadata is incomplete")
    validate_spd(matrices, "reused_Frechet_centers")
    return matrices, frame, [{
        "status": "reused_explicitly_after_validation",
        "centroid_file": os.path.abspath(center_path),
    }]


def _mds_kwargs(dimensions, seed):
    kwargs = {
        "n_components": int(dimensions),
        "metric": True,
        "dissimilarity": "precomputed",
        "n_init": 20,
        "max_iter": 1000,
        "eps": 1e-9,
        "random_state": int(seed),
    }
    if "normalized_stress" in inspect.signature(MDS).parameters:
        kwargs["normalized_stress"] = False
    return kwargs


def _normalized_stress(distance_matrix, coordinates):
    original = squareform(distance_matrix, checks=False)
    embedded = pdist(coordinates, metric="euclidean")
    denominator = float(np.sum(original ** 2))
    if denominator <= 0.0:
        return 0.0
    return float(np.sqrt(np.sum((original - embedded) ** 2) / denominator))


def fit_metric_mds(distance_matrix, dimensions, seed):
    estimator = MDS(**_mds_kwargs(dimensions, seed))
    coordinates = estimator.fit_transform(distance_matrix)
    iterations = int(getattr(estimator, "n_iter_", -1))
    return coordinates, {
        "seed": int(seed),
        "dimensions": int(dimensions),
        "raw_stress": float(estimator.stress_),
        "normalized_stress_1": _normalized_stress(distance_matrix, coordinates),
        "iterations": iterations,
        "reached_max_iter": iterations >= 1000,
        "parameters": _mds_kwargs(dimensions, seed),
    }


def _stability_against(reference, candidate, distance_matrix):
    reference_centered = reference - reference.mean(axis=0, keepdims=True)
    candidate_centered = candidate - candidate.mean(axis=0, keepdims=True)
    rotation, _ = orthogonal_procrustes(candidate_centered, reference_centered)
    aligned = candidate_centered @ rotation
    denominator = float(np.sum(aligned ** 2))
    scale = (
        float(np.sum(reference_centered * aligned)) / denominator
        if denominator > 0.0 else 1.0
    )
    aligned *= scale
    reference_norm = max(float(np.linalg.norm(reference_centered)), 1e-12)
    distance_reference = pdist(reference)
    distance_candidate = pdist(candidate)
    correlation = float(np.corrcoef(distance_reference, distance_candidate)[0, 1])
    return {
        "procrustes_normalized_rmse": float(
            np.linalg.norm(reference_centered - aligned) / reference_norm
        ),
        "embedded_distance_correlation": correlation,
        "normalized_stress_1": _normalized_stress(distance_matrix, candidate),
    }


def load_saved_mds(output_dir, center_frame, distance_matrix):
    coordinate_path = os.path.join(output_dir, "riemannian_mds_coordinates.csv")
    distance_path = os.path.join(output_dir, "riemannian_mds_distance_matrix.npy")
    metadata_path = os.path.join(output_dir, "riemannian_mds_metadata.json")
    if not all(os.path.exists(path) for path in [
            coordinate_path, distance_path, metadata_path]):
        raise FileNotFoundError(
            "--reuse-mds requires existing coordinates, distance matrix, and metadata"
        )
    saved_distances = np.load(distance_path)
    if saved_distances.shape != distance_matrix.shape or not np.allclose(
            saved_distances, distance_matrix, atol=1e-12, rtol=1e-12):
        raise ValueError("Saved MDS distance matrix does not match current centers")
    saved_frame = pd.read_csv(coordinate_path)
    if not np.array_equal(
            saved_frame["center_id"].astype(str).to_numpy(),
            center_frame["center_id"].astype(str).to_numpy()):
        raise ValueError("Saved MDS coordinate center IDs do not match")
    metadata = _read_json(metadata_path)
    if int(metadata.get("target_subject", -1)) != int(
            center_frame.loc[center_frame["domain"] == "target", "subject_id"].iloc[0]):
        raise ValueError("Saved MDS target subject does not match current centers")
    coordinates_2d = saved_frame[["mds_x", "mds_y"]].to_numpy(dtype=np.float64)
    coordinates_3d = None
    if metadata.get("mds_3d_generated"):
        columns = ["mds3_x", "mds3_y", "mds3_z"]
        if not set(columns).issubset(saved_frame.columns):
            raise ValueError("Saved metadata reports 3D MDS but coordinates are absent")
        coordinates_3d = saved_frame[columns].to_numpy(dtype=np.float64)
    stress_2d = dict(metadata["mds_2d"])
    stress_2d.setdefault(
        "reached_max_iter",
        int(stress_2d.get("iterations", -1)) >= int(
            stress_2d.get("parameters", {}).get("max_iter", 1000)
        ),
    )
    stability = []
    for saved in metadata.get("stability_runs", []):
        record = dict(saved)
        record.setdefault(
            "reached_max_iter",
            int(record.get("iterations", -1)) >= int(
                record.get("parameters", {}).get("max_iter", 1000)
            ),
        )
        stability.append(record)
    return (
        coordinates_2d,
        stress_2d,
        stability,
        coordinates_3d,
        metadata.get("mds_3d"),
    )


def _class_color_map(center_frame):
    classes = center_frame[["class_id", "class_name"]].drop_duplicates()
    classes = classes.sort_values("class_id")
    mapping = {}
    used = set()
    for index, row in enumerate(classes.itertuples(index=False)):
        name = str(row.class_name).lower()
        key = next((candidate for candidate in ["low", "medium", "high"]
                    if candidate in name), None)
        color = CLASS_COLORS.get(key, FALLBACK_COLORS[index % len(FALLBACK_COLORS)])
        if color in used:
            color = FALLBACK_COLORS[index % len(FALLBACK_COLORS)]
        mapping[int(row.class_id)] = color
        used.add(color)
    return mapping


def _font_family():
    for family in ["Arial", "Helvetica", "DejaVu Sans"]:
        try:
            font_manager.findfont(family, fallback_to_default=False)
            return family
        except ValueError:
            continue
    return "DejaVu Sans"


def _axis_limits(coordinates):
    limits = []
    for axis in range(2):
        low, high = np.min(coordinates[:, axis]), np.max(coordinates[:, axis])
        span = float(high - low)
        if span <= 0.0:
            span = 1.0
        limits.append((float(low - 0.10 * span), float(high + 0.14 * span)))
    return limits


def _short_class_name(name):
    value = str(name).replace("workload", "").strip()
    return value.title() if value else str(name)


def _class_and_domain_legends(fig, frame, colors, y=1.015):
    class_rows = frame[["class_id", "class_name"]].drop_duplicates().sort_values("class_id")
    class_handles = [
        Line2D([0], [0], marker="o", linestyle="None", markersize=5.5,
               markerfacecolor=colors[int(row.class_id)], markeredgecolor="none",
               label=str(row.class_name))
        for row in class_rows.itertuples(index=False)
    ]
    domain_handles = [
        Line2D([0], [0], marker="o", linestyle="None", markersize=4.5,
               markerfacecolor="#777777", markeredgecolor="none", label="Source subject"),
        Line2D([0], [0], marker="^", linestyle="None", markersize=7,
               markerfacecolor="white", markeredgecolor="#333333", label="Target subject"),
        Line2D([0], [0], marker="*", linestyle="None", markersize=9,
               markerfacecolor="#777777", markeredgecolor="white",
               label="Source class Frechet center"),
    ]
    legend_classes = fig.legend(
        handles=class_handles, title="Class", frameon=False, ncol=len(class_handles),
        loc="upper left", bbox_to_anchor=(0.02, y), fontsize=7.5,
        title_fontsize=8,
    )
    fig.add_artist(legend_classes)
    fig.legend(
        handles=domain_handles, title="Domain / center", frameon=False,
        ncol=len(domain_handles), loc="upper right", bbox_to_anchor=(0.98, y),
        fontsize=7.5, title_fontsize=8,
    )


def _style_axis(ax, limits):
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Riemannian MDS 1")
    ax.set_ylabel("Riemannian MDS 2")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=0.7, length=3, labelsize=7.5)


def _save_figure(fig, stem):
    metadata = {"Creator": "analysis/plot_riemannian_mds.py"}
    fig.savefig(stem + ".pdf", bbox_inches="tight", metadata=metadata)
    fig.savefig(stem + ".svg", bbox_inches="tight", metadata=metadata)
    fig.savefig(stem + ".png", dpi=600, bbox_inches="tight", metadata=metadata)
    plt.close(fig)


def plot_pre_post(frame, coordinates, output_dir, target_subject, stress_text,
                  annotate_distances, distance_changes):
    colors = _class_color_map(frame)
    limits = _axis_limits(coordinates)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.45), constrained_layout=False)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.14, top=0.78, wspace=0.22)
    for ax, stage, title in zip(
            axes, ["pre", "post"], ["Before SPDDSBN", "After SPDDSBN"]):
        stage_rows = frame[frame["stage"].astype(str) == stage]
        class_order = {
            int(value): index for index, value in enumerate(
                sorted(stage_rows["class_id"].astype(int).unique())
            )
        }
        for row in stage_rows.itertuples():
            x, y = coordinates[row.Index]
            color = colors[int(row.class_id)]
            if row.center_type == "source_class_reference":
                ax.scatter(x, y, s=110, marker="*", c=color, edgecolors="white",
                           linewidths=0.8, zorder=7)
                order = class_order[int(row.class_id)]
                source_offset = (
                    (5, -10 - 8 * order) if stage == "pre"
                    else (-5, -10 - 8 * order)
                )
                ax.annotate(
                    "Src {}".format(_short_class_name(row.class_name)),
                    (x, y), xytext=source_offset,
                    textcoords="offset points", fontsize=6.3, color=color,
                    ha="left" if stage == "pre" else "right", zorder=8,
                )
            elif row.domain == "target":
                ax.scatter(x, y, s=78, marker="^", facecolors="none",
                           edgecolors=color, linewidths=1.7, zorder=10)
                order = class_order[int(row.class_id)]
                ax.annotate(
                    "T{} {}".format(
                        target_subject, _short_class_name(row.class_name)
                    ), (x, y), xytext=(5, 5 + 10 * order),
                    textcoords="offset points", fontsize=6.8, color=color,
                    zorder=11,
                )
            else:
                ax.scatter(x, y, s=17, marker="o", c=color, alpha=0.58,
                           edgecolors="white", linewidths=0.2, zorder=3)
        ax.set_title(title, fontsize=9.5, pad=5)
        _style_axis(ax, limits)
    axes[0].text(-0.14, 1.08, "(a)", transform=axes[0].transAxes,
                 fontsize=9.5, fontweight="bold")
    axes[1].text(-0.14, 1.08, "(b)", transform=axes[1].transAxes,
                 fontsize=9.5, fontweight="bold")
    fig.text(0.5, 0.015, stress_text, ha="center", va="bottom", fontsize=7)
    if annotate_distances:
        lines = [
            "{}: {:.3f} -> {:.3f}".format(
                row["class_name"], row["distance_pre"], row["distance_post"]
            )
            for row in distance_changes
        ]
        axes[1].text(0.02, 0.02, "\n".join(lines), transform=axes[1].transAxes,
                     fontsize=6.8, va="bottom")
    _class_and_domain_legends(fig, frame, colors)
    _save_figure(
        fig, os.path.join(output_dir, "fig_riemannian_mds_pre_post")
    )


def plot_trajectory(frame, coordinates, output_dir, target_subject, stress_text):
    colors = _class_color_map(frame)
    limits = _axis_limits(coordinates)
    fig, ax = plt.subplots(figsize=(7.1, 4.65), constrained_layout=False)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.12, top=0.79)
    subject_rows = frame[frame["center_type"] == "subject_class"]
    class_order = {
        int(value): index for index, value in enumerate(
            sorted(subject_rows["class_id"].astype(int).unique())
        )
    }
    for (_, _), pair in subject_rows.groupby(["subject_id", "class_id"], sort=True):
        if set(pair["stage"].astype(str)) != {"pre", "post"}:
            raise RuntimeError("A subject-class center is missing pre or post")
        pre_row = pair[pair["stage"] == "pre"].iloc[0]
        post_row = pair[pair["stage"] == "post"].iloc[0]
        pre_xy = coordinates[int(pre_row.name)]
        post_xy = coordinates[int(post_row.name)]
        target = str(pre_row["domain"]) == "target"
        color = colors[int(pre_row["class_id"])]
        arrow_color = color if target else "#A9A9A9"
        ax.annotate(
            "", xy=post_xy, xytext=pre_xy,
            arrowprops={
                "arrowstyle": "->", "color": arrow_color,
                "lw": 1.65 if target else 0.55,
                "alpha": 0.95 if target else 0.28,
                "shrinkA": 2.5, "shrinkB": 2.5,
            },
            zorder=8 if target else 1,
        )
        marker = "^" if target else "o"
        size = 72 if target else 16
        ax.scatter(*pre_xy, s=size, marker=marker, facecolors="white",
                   edgecolors=color, linewidths=1.5 if target else 0.65,
                   alpha=1.0 if target else 0.58, zorder=10 if target else 3)
        ax.scatter(*post_xy, s=size, marker=marker, facecolors=color,
                   edgecolors="white", linewidths=0.45,
                   alpha=1.0 if target else 0.58, zorder=11 if target else 4)
        if target:
            order = class_order[int(pre_row["class_id"])]
            ax.annotate(
                "T{} {}".format(
                    target_subject, _short_class_name(pre_row["class_name"])
                ),
                post_xy, xytext=(6, -11 + 18 * order), textcoords="offset points",
                fontsize=7, color=color, fontweight="bold", zorder=12,
            )
    references = frame[frame["center_type"] == "source_class_reference"]
    for class_id, pair in references.groupby("class_id", sort=True):
        color = colors[int(class_id)]
        order = class_order[int(class_id)]
        for _, row in pair.iterrows():
            xy = coordinates[int(row.name)]
            filled = str(row["stage"]) == "post"
            ax.scatter(*xy, s=125, marker="*",
                       facecolors=color if filled else "white",
                       edgecolors=color, linewidths=1.2, zorder=9)
            ax.annotate(
                "Src {} {}".format(
                    _short_class_name(row["class_name"]), row["stage"]
                ),
                xy,
                xytext=(-6, (-19 + 12 * order) if filled else (8 + 12 * order)),
                textcoords="offset points", ha="right",
                fontsize=6.5, color=color, zorder=10,
            )
    _style_axis(ax, limits)
    ax.set_title("Subject-class geometric trajectories: pre to post SPDDSBN",
                 fontsize=9.5, pad=5)
    fig.text(0.5, 0.015, stress_text, ha="center", va="bottom", fontsize=7)
    _class_and_domain_legends(fig, frame, colors)
    stage_handles = [
        Line2D([0], [0], marker="o", linestyle="None", markersize=5,
               markerfacecolor="white", markeredgecolor="#555555", label="Pre"),
        Line2D([0], [0], marker="o", linestyle="None", markersize=5,
               markerfacecolor="#555555", markeredgecolor="white", label="Post"),
    ]
    ax.legend(handles=stage_handles, title="Stage", frameon=False, ncol=2,
              loc="upper right", fontsize=7.5, title_fontsize=8)
    _save_figure(
        fig, os.path.join(output_dir, "fig_riemannian_mds_trajectory")
    )


def plot_3d(frame, coordinates, output_dir, stress_text):
    colors = _class_color_map(frame)
    fig = plt.figure(figsize=(7.1, 5.2))
    ax = fig.add_subplot(111, projection="3d")
    for row in frame.itertuples():
        color = colors[int(row.class_id)]
        if row.center_type == "source_class_reference":
            marker, size, face, edge, alpha = "*", 105, color, "white", 1.0
        elif row.domain == "target":
            marker, size, face, edge, alpha = "^", 70, "none", color, 1.0
        else:
            marker, size, face, edge, alpha = "o", 14, color, "white", 0.55
        ax.scatter(
            coordinates[row.Index, 0], coordinates[row.Index, 1],
            coordinates[row.Index, 2], marker=marker, s=size,
            facecolors=face, edgecolors=edge, linewidths=0.6, alpha=alpha,
            depthshade=False,
        )
    ax.set_xlabel("Riemannian MDS 1")
    ax.set_ylabel("Riemannian MDS 2")
    ax.set_zlabel("Riemannian MDS 3")
    ax.set_title("Supplementary 3D AIRM-MDS\n{}".format(stress_text), fontsize=9.5)
    ax.grid(False)
    _save_figure(fig, os.path.join(output_dir, "fig_riemannian_mds_3d"))


def target_source_distance_changes(center_matrices, center_frame, target_subject):
    rows = []
    classes = center_frame[["class_id", "class_name"]].drop_duplicates()
    for class_row in classes.sort_values("class_id").itertuples(index=False):
        values = {"class_id": int(class_row.class_id), "class_name": str(class_row.class_name)}
        for stage in ["pre", "post"]:
            target_mask = (
                (center_frame["center_type"] == "subject_class")
                & (center_frame["subject_id"] == int(target_subject))
                & (center_frame["class_id"] == int(class_row.class_id))
                & (center_frame["stage"] == stage)
            )
            source_mask = (
                (center_frame["center_type"] == "source_class_reference")
                & (center_frame["class_id"] == int(class_row.class_id))
                & (center_frame["stage"] == stage)
            )
            if int(target_mask.sum()) != 1 or int(source_mask.sum()) != 1:
                raise RuntimeError("Target/source class reference lookup is not unique")
            target_position = int(np.flatnonzero(target_mask.to_numpy())[0])
            source_position = int(np.flatnonzero(source_mask.to_numpy())[0])
            values["distance_{}".format(stage)] = airm_distance(
                center_matrices[target_position], center_matrices[source_position]
            )
        values["change_post_minus_pre"] = values["distance_post"] - values["distance_pre"]
        values["relative_change_percent"] = (
            100.0 * values["change_post_minus_pre"] / values["distance_pre"]
            if values["distance_pre"] > 0.0 else None
        )
        rows.append(values)
    return rows


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.karcher_tol <= 0.0 or args.karcher_max_iter < 1:
        raise ValueError("Invalid Karcher-flow convergence settings")
    if args.stress_3d_threshold <= 0.0:
        raise ValueError("--stress-3d-threshold must be positive")
    os.makedirs(args.output_dir, exist_ok=True)
    args.feature_cache_dir = _resolve_feature_cache_dir(
        args.feature_cache_dir, args.output_dir
    )
    _resolve_original_output_inputs(args)
    random.seed(MAIN_MDS_SEED)
    np.random.seed(MAIN_MDS_SEED)
    torch.manual_seed(MAIN_MDS_SEED)
    matplotlib.rcParams.update({
        "font.family": _font_family(),
        "font.size": 8.0,
        "axes.labelsize": 8.0,
        "axes.titlesize": 9.5,
        "axes.linewidth": 0.8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "svg.hashsalt": "riemannian-mds-2026",
    })

    selection = _selection_from_args(args)
    selection["checkpoint_protocol_validation"] = _validate_selected_result(
        args, selection
    )
    target_subject = int(selection["selected_target_subject"])
    pre, post, sample_metadata, feature_provenance = _load_or_extract_features(
        args, selection
    )
    checks = {
        "pre": validate_spd(pre, "P_pre"),
        "post": validate_spd(post, "P_post"),
    }
    if pre.shape != post.shape or len(pre) != len(sample_metadata):
        raise RuntimeError("P_pre/P_post sample alignment failed")
    if sample_metadata["sample_id"].duplicated().any():
        raise RuntimeError("Sample IDs are not unique")

    if args.reuse_centroids:
        center_matrices, center_frame, center_diagnostics = load_saved_centers(
            args.output_dir
        )
    else:
        center_matrices, center_frame, center_diagnostics = build_frechet_centers(
            pre, post, sample_metadata, args.karcher_tol, args.karcher_max_iter
        )
    checks["centers"] = validate_spd(center_matrices, "Frechet_centers")
    distances = pairwise_airm(center_matrices)
    if not np.allclose(distances, distances.T, atol=1e-12):
        raise RuntimeError("AIRM distance matrix is not symmetric")
    if not np.allclose(np.diag(distances), 0.0, atol=1e-12):
        raise RuntimeError("AIRM distance-matrix diagonal is not zero")

    if args.reuse_mds:
        (
            coordinates_2d, stress_2d, stability,
            coordinates_3d, stress_3d,
        ) = load_saved_mds(args.output_dir, center_frame, distances)
    else:
        coordinates_2d, stress_2d = fit_metric_mds(
            distances, dimensions=2, seed=MAIN_MDS_SEED
        )
        seeds = sorted(set(
            int(value.strip()) for value in args.stability_seeds.split(",")
            if value.strip()
        ) | {MAIN_MDS_SEED})
        stability = []
        for seed in seeds:
            if seed == MAIN_MDS_SEED:
                candidate = coordinates_2d
                candidate_stress = stress_2d
            else:
                candidate, candidate_stress = fit_metric_mds(
                    distances, dimensions=2, seed=seed
                )
            record = dict(candidate_stress)
            record.update(_stability_against(coordinates_2d, candidate, distances))
            stability.append(record)

        coordinates_3d = None
        stress_3d = None
        if stress_2d["normalized_stress_1"] > float(args.stress_3d_threshold):
            coordinates_3d, stress_3d = fit_metric_mds(
                distances, dimensions=3, seed=MAIN_MDS_SEED
            )
    changes = target_source_distance_changes(
        center_matrices, center_frame, target_subject
    )

    np.savez_compressed(
        os.path.join(args.output_dir, "riemannian_mds_centroids.npz"),
        center_id=center_frame["center_id"].astype(str).to_numpy(dtype="U"),
        center_matrix=center_matrices,
        subject_id=center_frame["subject_id"].to_numpy(dtype=np.int64),
        class_id=center_frame["class_id"].to_numpy(dtype=np.int64),
        stage=center_frame["stage"].astype(str).to_numpy(dtype="U"),
        center_type=center_frame["center_type"].astype(str).to_numpy(dtype="U"),
    )
    np.save(
        os.path.join(args.output_dir, "riemannian_mds_distance_matrix.npy"),
        distances,
    )
    coordinate_frame = center_frame.copy()
    coordinate_frame["mds_x"] = coordinates_2d[:, 0]
    coordinate_frame["mds_y"] = coordinates_2d[:, 1]
    if coordinates_3d is not None:
        coordinate_frame["mds3_x"] = coordinates_3d[:, 0]
        coordinate_frame["mds3_y"] = coordinates_3d[:, 1]
        coordinate_frame["mds3_z"] = coordinates_3d[:, 2]
    coordinate_frame.to_csv(
        os.path.join(args.output_dir, "riemannian_mds_coordinates.csv"), index=False
    )
    pd.DataFrame(changes).to_csv(
        os.path.join(args.output_dir, "riemannian_mds_target_distances.csv"),
        index=False,
    )
    stress_text = "2D metric MDS: raw stress={:.3f}, normalized Stress-1={:.3f}".format(
        stress_2d["raw_stress"], stress_2d["normalized_stress_1"]
    )
    plot_pre_post(
        center_frame, coordinates_2d, args.output_dir, target_subject,
        stress_text, args.annotate_distances, changes,
    )
    plot_trajectory(
        center_frame, coordinates_2d, args.output_dir, target_subject, stress_text
    )
    if coordinates_3d is not None:
        plot_3d(
            center_frame, coordinates_3d, args.output_dir,
            "raw stress={:.3f}, normalized Stress-1={:.3f}".format(
                stress_3d["raw_stress"], stress_3d["normalized_stress_1"]
            ),
        )

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "dataset": _dataset_name(args),
        "model_type": args.model,
        "target_subject": target_subject,
        "representative_fold_selection": selection,
        "original_output_inputs": {
            "loso_output_directory": os.path.abspath(args.checkpoint_root),
            "per_subject_results": os.path.abspath(args.results_file),
            "master_summary": (
                os.path.abspath(args.master_summary)
                if args.master_summary and os.path.exists(args.master_summary)
                else None
            ),
            "training_configuration_source": (
                "matched_master_summary" if getattr(args, "run_record", {})
                else "explicit_cli_or_project_defaults"
            ),
        },
        "feature_provenance": feature_provenance,
        "feature_locations": visualization_model_metadata(args.model),
        "sample_counts": {
            "total": int(len(sample_metadata)),
            "source_train": int((sample_metadata["domain"] == "source").sum()),
            "target_test": int((sample_metadata["domain"] == "target").sum()),
            "source_subjects": int(sample_metadata.loc[
                sample_metadata["domain"] == "source", "subject_id"
            ].nunique()),
            "target_subjects": int(sample_metadata.loc[
                sample_metadata["domain"] == "target", "subject_id"
            ].nunique()),
        },
        "class_names": {
            str(int(row.class_id)): str(row.class_name)
            for row in sample_metadata[["class_id", "class_name"]]
            .drop_duplicates().sort_values("class_id").itertuples(index=False)
        },
        "spd_dimension": int(pre.shape[-1]),
        "center_count": int(len(center_frame)),
        "subject_class_center_count": int(
            (center_frame["center_type"] == "subject_class").sum()
        ),
        "source_reference_center_count": int(
            (center_frame["center_type"] == "source_class_reference").sum()
        ),
        "distance_metric": (
            "AIRM: Frobenius norm of log eigenvalues of "
            "P^(-1/2) Q P^(-1/2)"
        ),
        "mds_joint_stages": ["pre", "post"],
        "mds_reused_explicitly": bool(args.reuse_mds),
        "mds_2d": stress_2d,
        "stability_runs": stability,
        "stress_3d_threshold": float(args.stress_3d_threshold),
        "mds_3d_generated": coordinates_3d is not None,
        "mds_3d": stress_3d,
        "target_source_class_distances": changes,
        "spd_checks": checks,
        "frechet_diagnostics": center_diagnostics,
        "target_labels_usage": (
            "offline grouping, plotting, and class-conditional distances only"
        ),
        "frechet_convergence_enforced": (
            "Every center is produced by TSMNet Karcher flow; the run stops "
            "before saving centers if any mean is non-SPD or fails tolerance."
        ),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "sklearn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    _write_json(
        metadata,
        os.path.join(args.output_dir, "riemannian_mds_metadata.json"),
    )
    print("Target subject:", target_subject)
    print("Center count / SPD dimension:", len(center_frame), pre.shape[-1])
    print("2D MDS stress:", stress_2d)
    if stress_3d is not None:
        print("3D MDS stress:", stress_3d)
    print("Target-to-source class distances:", changes)
    print("Saved:", os.path.abspath(args.output_dir))


if __name__ == "__main__":
    main()
