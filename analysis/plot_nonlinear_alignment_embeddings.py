"""Stable UMAP and joint t-SNE supplements for SPDDSBN alignment.

The script extracts paired SPD matrices from one representative LOSO checkpoint,
maps them into one source-pre-fitted common tangent space, and treats nonlinear
embeddings as visualization only. High-dimensional metrics remain the evidence.
"""

import argparse
import inspect
import json
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
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
import sklearn
from sklearn.manifold import TSNE
try:
    from sklearn.manifold import trustworthiness
except ImportError:  # scikit-learn < 0.22
    from sklearn.manifold.t_sne import trustworthiness
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
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
from src.cl_tsmnet.spd_pca import (
    airm_karcher_mean,
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


MAIN_SEED = 2026
DEFAULT_UMAP_SEEDS = (2026, 2027, 2028, 2029, 2030)
CLASS_COLORS = ["#3B75AF", "#D9534F", "#D9A62E", "#2A9D8F", "#7A5195"]
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate source-pre-fitted UMAP and jointly fitted t-SNE supplements."
    )
    parser.add_argument("--dataset", choices=["stew", "eegmat", "cog-bci"], required=True)
    parser.add_argument("--cog-paradigm", choices=["nback", "matb"], default="nback")
    parser.add_argument("--model", choices=sorted(SUPPORTED_SPD_VISUALIZATION_MODELS),
                        default="ms_tgc_spddsbn")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--checkpoint-root", default=None)
    parser.add_argument("--results-file", default=None)
    parser.add_argument("--master-summary", default=None)
    parser.add_argument("--target-subject", type=int, default=None)
    parser.add_argument("--allow-legacy-source-target-refit", action="store_true")
    parser.add_argument("--feature-cache-dir", default=None)
    parser.add_argument("--force-reextract", action="store_true")
    parser.add_argument("--output-dir", default=os.path.join(
        "results", "figures", "nonlinear_embeddings"
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
    parser.add_argument("--max-points-per-class-domain", type=int, default=500)
    parser.add_argument("--sampling-seed", type=int, default=2026)
    parser.add_argument("--karcher-tol", type=float, default=1e-7)
    parser.add_argument("--karcher-max-iter", type=int, default=50)
    parser.add_argument("--riemann-batch-size", type=int, default=512)
    parser.add_argument("--umap-neighbors", type=int, default=15)
    parser.add_argument("--umap-min-dist", type=float, default=0.15)
    parser.add_argument("--umap-metric", default="euclidean")
    parser.add_argument("--umap-seeds", default="2026,2027,2028,2029,2030")
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument("--tsne-max-iter", type=int, default=2000)
    parser.add_argument("--quality-neighbors", type=int, default=15)
    parser.add_argument("--no-ellipses", action="store_true")
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


def _dataset_name(args):
    return "cog-bci-{}".format(args.cog_paradigm) if args.dataset == "cog-bci" else args.dataset


def _usable(value):
    if value is None:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return value


def _normal_path(path):
    return os.path.normcase(os.path.abspath(os.path.normpath(str(path))))


def _resolve_inputs(args):
    if args.checkpoint_root is None:
        args.checkpoint_root = os.path.join(
            args.output_root,
            run_directory_name(_dataset_name(args), "loso", args.model, "spddsbn"),
        )
    if args.results_file is None:
        args.results_file = os.path.join(args.checkpoint_root, "summary.csv")
    if args.master_summary is None:
        args.master_summary = os.path.join(args.output_root, "master_summary.csv")
    if not os.path.exists(args.results_file):
        raise FileNotFoundError("LOSO per-subject summary was not found: {}".format(
            os.path.abspath(args.results_file)
        ))

    run_record = {}
    if os.path.exists(args.master_summary):
        master = pd.read_csv(args.master_summary)
        if "protocol" in master:
            master = master[master["protocol"].astype(str) == "loso"]
        if "dataset" in master:
            master = master[master["dataset"].astype(str) == _dataset_name(args)]
        if "model_type" in master:
            master = master[master["model_type"].astype(str) == args.model]
        if "output_dir" in master and not master.empty:
            exact = master[master["output_dir"].astype(str).map(
                lambda value: _normal_path(value) == _normal_path(args.checkpoint_root)
            )]
            master = exact
        if not master.empty:
            run_record = master.iloc[-1].dropna().to_dict()
    args.run_record = run_record

    for key, default in MODEL_DEFAULTS.items():
        value = getattr(args, key)
        if value is None:
            value = _usable(run_record.get(key))
            setattr(args, key, default if value is None else value)
    for key, default in {"seed": 42, "val_size": 0.2, "test_size": 0.2}.items():
        value = getattr(args, key)
        if value is None:
            value = _usable(run_record.get(key))
            setattr(args, key, default if value is None else value)
    if args.target_fs is None and _usable(run_record.get("target_fs")) is not None:
        args.target_fs = float(run_record["target_fs"])
    if args.cache is None and _usable(run_record.get("cache")) is not None:
        candidate = str(run_record["cache"])
        if os.path.exists(candidate):
            args.cache = candidate
    if args.artifact_z is None and _usable(run_record.get("artifact_z")) is not None:
        args.artifact_z = float(run_record["artifact_z"])
    return run_record


def _select_fold(args):
    results = pd.read_csv(args.results_file)
    if "protocol" in results:
        results = results[results["protocol"].astype(str) == "loso"]
    if "dataset" in results:
        results = results[results["dataset"].astype(str) == _dataset_name(args)]
    if "model_type" in results:
        results = results[results["model_type"].astype(str) == args.model]
    if results.empty:
        raise ValueError("No matching LOSO rows exist in {}".format(args.results_file))
    if args.target_subject is None:
        selection = choose_median_fold(results)
    else:
        selection = {
            "selected_target_subject": int(args.target_subject),
            "subject_balanced_accuracy": None,
            "all_subject_median_balanced_accuracy": None,
            "selection_rule": "explicit_target_subject",
            "subject_seed_rows_averaged": None,
        }
    subject = int(selection["selected_target_subject"])
    rows = results[pd.to_numeric(results["subject"], errors="coerce") == subject]
    if rows.empty:
        raise ValueError("Selected target subject is absent from the LOSO summary")
    if "target_adapt" not in rows:
        raise ValueError("summary.csv lacks target_adapt audit metadata")
    enabled = rows["target_adapt"].astype(str).str.lower().isin(["true", "1", "yes"])
    if not bool(enabled.all()):
        raise ValueError("Selected checkpoint was evaluated without target adaptation")
    if args.model == "tsmnet" and "bnorm" in rows:
        if set(rows["bnorm"].dropna().astype(str)) != {"spddsbn"}:
            raise ValueError("TSMNet nonlinear visualization requires bnorm=spddsbn")

    scopes = []
    if "target_refit_scope" in rows:
        scopes = sorted(set(v for v in rows["target_refit_scope"].dropna().astype(str) if v))
    scope_source = "summary.csv"
    if not scopes and _usable(args.run_record.get("target_refit_scope")) is not None:
        scopes = sorted(set(
            v.strip() for v in str(args.run_record["target_refit_scope"]).split(",")
            if v.strip()
        ))
        scope_source = "master_summary.csv"
    legacy = False
    if not scopes:
        if not args.allow_legacy_source_target_refit:
            raise ValueError(
                "This result predates target_refit_scope metadata and used the "
                "historical unlabeled source+target SPDDSBN refit. Re-run with "
                "the current target-only pipeline or pass "
                "--allow-legacy-source-target-refit to disclose the legacy scope."
            )
        legacy = True
        scopes = ["source_plus_target_unlabeled"]
        scope_source = "historical_contract_before_d9d0ca0"
        warnings.warn("Using a legacy unlabeled source+target SPDDSBN checkpoint")
    if not legacy and scopes != ["target_only"]:
        raise ValueError("Expected target_refit_scope=target_only, found {}".format(scopes))
    selection.update({
        "results_file": os.path.abspath(args.results_file),
        "target_refit_scope": scopes,
        "target_refit_scope_evidence": scope_source,
        "publication_warning": (
            "Legacy source+target unlabeled SPDDSBN refit; not target-only."
            if legacy else None
        ),
    })
    return subject, selection


def _checkpoint_path(root, subject):
    candidates = [
        root if os.path.isfile(root) else None,
        os.path.join(root, "subject_{:02d}".format(subject), "model.pt"),
        os.path.join(root, "subject_{}".format(subject), "model.pt"),
        os.path.join(root, "model.pt"),
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return os.path.abspath(path)
    raise FileNotFoundError("No checkpoint found; tried {}".format(candidates[1:]))


def _load_state(path):
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError("Checkpoint does not contain a state_dict")
    return state


def _model_config(args):
    density = args.mstgc_graph_density
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
        "mstgc_graph_density": None if density is None else float(density),
        "mstgc_time_points": int(args.mstgc_time_points),
        "mstgc_shrinkage": float(args.mstgc_shrinkage),
    }


def _build_model(args, dataset, domains, selected, classes, device):
    config = _model_config(args)
    channels, samples = dataset["x"].shape[1:]
    if args.model == "tsmnet":
        model = build_tsmnet(
            PROJECT_ROOT, channels, samples, classes, domains[selected],
            bnorm="spddsbn", temporal_filters=config["temporal_filters"],
            spatial_filters=config["spatial_filters"],
            subspacedims=config["subspacedims"],
            temp_kernel=config["temp_kernel"], device=device,
        )
    else:
        kernel = max(3, int(round(
            config["mstgc_kernel_length"] * float(dataset["fs"]) / 128.0
        )))
        model = build_ms_tgc_spddsbn(
            PROJECT_ROOT, channels, samples, classes, domains[selected],
            subspacedims=config["subspacedims"], device=device,
            temporal_hidden=config["mstgc_temporal_hidden"],
            graph_hidden=config["mstgc_graph_hidden"],
            fusion_dim=config["mstgc_fusion_dim"], kernel_length=kernel,
            num_heads=config["mstgc_num_heads"],
            cheby_order=config["mstgc_cheby_order"],
            dropout=config["mstgc_dropout"], num_nodes=config["mstgc_num_nodes"],
            variant=args.model, graph_mode="adaptive",
            graph_neighbors=config["mstgc_graph_k"],
            graph_density=config["mstgc_graph_density"],
            graph_time_points=config["mstgc_time_points"],
            covariance_shrinkage=config["mstgc_shrinkage"],
        )
    return model.to(device), config


def _extract_partition(model, model_type, dataset, indices, domains, normalizer,
                       batch_size, device):
    pre_parts, post_parts = [], []
    indices = np.asarray(indices, dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), int(batch_size)):
            batch_ids = indices[start:start + int(batch_size)]
            windows = normalizer.transform_array(dataset["x"][batch_ids])
            xb = torch.from_numpy(windows).to(device=device, dtype=torch.float32)
            db = torch.from_numpy(domains[batch_ids]).to(device=device, dtype=torch.long)
            _, values = extract_spd_intermediates(model, xb, db, model_type)
            pre_parts.append(values["spd_pre_bn"].detach().cpu().double().numpy())
            post_parts.append(values["spd_post_bn"].detach().cpu().double().numpy())
    return np.concatenate(pre_parts), np.concatenate(post_parts)


def _feature_files(directory):
    return (
        os.path.join(directory, "spd_intermediates.npz"),
        os.path.join(directory, "spd_intermediates_metadata.csv"),
        os.path.join(directory, "extraction_signature.json"),
    )


def _load_feature_files(directory):
    feature_path, metadata_path, _ = _feature_files(directory)
    if not os.path.exists(feature_path) or not os.path.exists(metadata_path):
        raise FileNotFoundError("Feature cache is incomplete: {}".format(directory))
    with np.load(feature_path, allow_pickle=False) as saved:
        required = {"sample_id", "spd_pre_bn", "spd_post_bn"}
        if not required.issubset(saved.files):
            raise ValueError("Feature cache lacks {}".format(required - set(saved.files)))
        ids = saved["sample_id"].astype(np.int64)
        pre = saved["spd_pre_bn"].astype(np.float64)
        post = saved["spd_post_bn"].astype(np.float64)
    metadata = pd.read_csv(metadata_path)
    required_columns = {
        "sample_id", "subject_id", "domain", "class_id", "class_name", "split"
    }
    missing = required_columns - set(metadata.columns)
    if missing:
        raise ValueError("Feature metadata lacks {}".format(sorted(missing)))
    if not np.array_equal(ids, metadata["sample_id"].to_numpy(dtype=np.int64)):
        raise ValueError("Feature-cache sample order does not match metadata")
    if pre.shape != post.shape or len(pre) != len(metadata):
        raise ValueError("Feature-cache pre/post alignment failed")
    if metadata["sample_id"].duplicated().any():
        raise ValueError("Feature cache contains duplicate sample IDs")
    if set(metadata["domain"].astype(str)) != {"source", "target"}:
        raise ValueError("Feature cache must contain source and target domains only")
    if not set(metadata["split"].astype(str)).issubset({"source_train", "target_test"}):
        raise ValueError("Feature cache contains validation or unknown split rows")
    return pre, post, metadata


def _extract_or_load(args, subject, checkpoint):
    if args.feature_cache_dir:
        pre, post, metadata = _load_feature_files(args.feature_cache_dir)
        targets = set(metadata.loc[metadata["domain"] == "target", "subject_id"].astype(int))
        if targets != {subject}:
            raise ValueError("Feature cache target subjects do not match selected fold")
        return pre, post, metadata, {
            "source": "explicit_validated_feature_cache",
            "directory": os.path.abspath(args.feature_cache_dir),
        }

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
        dataset, "loso", subject, seed=int(args.seed),
        val_size=float(args.val_size), test_size=float(args.test_size),
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
    labels = np.unique(dataset["y"][source_ids]).astype(np.int64)
    if not np.array_equal(labels, np.arange(len(labels))):
        raise ValueError("Source labels must be contiguous 0..K-1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = _build_model(args, dataset, domains, selected, len(labels), device)
    state, migrations = migrate_legacy_spddsbn_buffers(
        _load_state(checkpoint), model.state_dict()
    )
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError("Checkpoint architecture mismatch:\n{}".format(exc)) from exc

    signature = {
        "checkpoint": checkpoint,
        "checkpoint_size": int(os.path.getsize(checkpoint)),
        "checkpoint_mtime_ns": int(os.stat(checkpoint).st_mtime_ns),
        "dataset_cache": os.path.abspath(cache),
        "target_subject": int(subject),
        "model_type": args.model,
        "model_config": config,
        "split_seed": int(args.seed),
        "val_size": float(args.val_size),
        "test_size": float(args.test_size),
        "artifact_z": args.artifact_z,
    }
    feature_dir = os.path.join(args.output_dir, "feature_cache")
    feature_path, metadata_path, signature_path = _feature_files(feature_dir)
    if (
        not args.force_reextract
        and all(os.path.exists(path) for path in [feature_path, metadata_path, signature_path])
    ):
        with open(signature_path, encoding="utf-8") as handle:
            if json.load(handle) == _jsonable(signature):
                pre, post, metadata = _load_feature_files(feature_dir)
                return pre, post, metadata, {
                    "source": "reused_signature_matched_feature_cache",
                    "directory": os.path.abspath(feature_dir),
                    "signature": signature,
                }

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
    meta = dataset["meta"].iloc[sample_ids]
    class_ids = dataset["y"][sample_ids].astype(np.int64)
    names = dataset.get("label_names", {})
    metadata = pd.DataFrame({
        "sample_id": sample_ids,
        "subject_id": meta["subject"].to_numpy(dtype=np.int64),
        "domain_id": domains[sample_ids].astype(np.int64),
        "domain": np.where(np.isin(sample_ids, source_ids), "source", "target"),
        "class_id": class_ids,
        "class_name": [str(names.get(int(v), "class {}".format(v))) for v in class_ids],
        "split": np.where(np.isin(sample_ids, source_ids), "source_train", "target_test"),
        "session": meta["session"].to_numpy(dtype=np.int64),
        "task": meta["task"].astype(str).to_numpy(),
        "start_sample": meta["start_sample"].to_numpy(dtype=np.int64),
    })
    os.makedirs(feature_dir, exist_ok=True)
    np.savez_compressed(
        feature_path, sample_id=sample_ids, spd_pre_bn=pre, spd_post_bn=post
    )
    metadata.to_csv(metadata_path, index=False)
    _write_json(signature, signature_path)
    return pre, post, metadata, {
        "source": "extracted_from_best_checkpoint",
        "checkpoint": checkpoint,
        "dataset_cache": os.path.abspath(cache),
        "signature": signature,
        "checkpoint_buffer_migrations": migrations,
    }


def _manifest_positions(metadata, manifest):
    positions = {int(value): index for index, value in enumerate(metadata["sample_id"])}
    selected = np.asarray([positions[int(value)] for value in manifest["sample_id"]],
                          dtype=np.int64)
    if len(np.unique(selected)) != len(selected):
        raise RuntimeError("Balanced manifest contains duplicate sample IDs")
    return selected


def _effective_neighbors(requested, sample_count):
    return max(1, min(int(requested), int(sample_count) - 1, (int(sample_count) - 1) // 2))


def knn_overlap(high, low, neighbors):
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    if len(high) != len(low):
        raise ValueError("High/low coordinate counts differ")
    k = _effective_neighbors(neighbors, len(high))
    def neighbor_ids(values):
        raw = NearestNeighbors(n_neighbors=k + 1).fit(values).kneighbors(
            values, return_distance=False
        )
        cleaned = []
        for row, candidates in enumerate(raw):
            selected = [int(value) for value in candidates if int(value) != row][:k]
            if len(selected) != k:
                raise RuntimeError("Could not construct a self-excluded kNN set")
            cleaned.append(selected)
        return np.asarray(cleaned, dtype=np.int64)

    high_ids = neighbor_ids(high)
    low_ids = neighbor_ids(low)
    overlap = [len(set(a).intersection(set(b))) / float(k) for a, b in zip(high_ids, low_ids)]
    return {"k": int(k), "mean_overlap": float(np.mean(overlap)),
            "std_overlap": float(np.std(overlap, ddof=1)) if len(overlap) > 1 else 0.0}


def embedding_quality(high, low, neighbors):
    k = _effective_neighbors(neighbors, len(high))
    return {
        "trustworthiness": float(trustworthiness(high, low, n_neighbors=k)),
        "knn_overlap": knn_overlap(high, low, k),
    }


def high_dimensional_metrics(pre, post, metadata, epsilon=1e-12):
    domains = metadata["domain"].astype(str).to_numpy()
    classes = metadata["class_id"].to_numpy(dtype=np.int64)
    source, target = domains == "source", domains == "target"

    def stage(values):
        source_mean = values[source].mean(axis=0)
        target_mean = values[target].mean(axis=0)
        conditional = {}
        conditional_values = []
        centers, within = [], []
        for class_id in sorted(int(value) for value in np.unique(classes)):
            source_class = source & (classes == class_id)
            target_class = target & (classes == class_id)
            distance = float(np.linalg.norm(
                values[source_class].mean(axis=0) - values[target_class].mean(axis=0)
            ))
            conditional[str(class_id)] = distance
            conditional_values.append(distance)
            class_mask = classes == class_id
            center = values[class_mask].mean(axis=0)
            centers.append(center)
            within.append(float(np.mean(np.sum((values[class_mask] - center) ** 2, axis=1))))
        between = [
            float(np.sum((centers[left] - centers[right]) ** 2))
            for left in range(len(centers)) for right in range(left + 1, len(centers))
        ]
        return {
            "domain_centroid_distance": float(np.linalg.norm(source_mean - target_mean)),
            "class_conditional_domain_distances": conditional,
            "class_balanced_domain_distance": float(np.mean(conditional_values)),
            "fisher_ratio": float(np.mean(between) / (np.mean(within) + epsilon)),
        }

    result = {"pre": stage(pre), "post": stage(post),
              "feature_dimension": int(pre.shape[1])}
    result["changes_post_minus_pre"] = {
        "domain_centroid_distance": (
            result["post"]["domain_centroid_distance"]
            - result["pre"]["domain_centroid_distance"]
        ),
        "class_balanced_domain_distance": (
            result["post"]["class_balanced_domain_distance"]
            - result["pre"]["class_balanced_domain_distance"]
        ),
        "fisher_ratio": result["post"]["fisher_ratio"] - result["pre"]["fisher_ratio"],
    }
    return result


def _stability(main_coordinates, candidate_coordinates):
    main = np.asarray(main_coordinates, dtype=np.float64)
    candidate = np.asarray(candidate_coordinates, dtype=np.float64)
    correlation = spearmanr(pdist(main), pdist(candidate)).correlation
    left = main - main.mean(axis=0)
    right = candidate - candidate.mean(axis=0)
    left /= max(np.linalg.norm(left), 1e-12)
    right /= max(np.linalg.norm(right), 1e-12)
    rotation, _ = orthogonal_procrustes(right, left)
    aligned = right @ rotation
    return {
        "pairwise_distance_spearman": float(correlation),
        "procrustes_rmse": float(np.sqrt(np.mean((left - aligned) ** 2))),
    }


def _import_umap():
    try:
        from umap import UMAP
    except ImportError as exc:
        raise ImportError(
            "UMAP requires the optional analysis dependency. Install it with "
            "`pip install umap-learn` in the experiment environment."
        ) from exc
    return UMAP


def fit_umap_runs(fit_source_pre, pre, post, metadata, args):
    seeds = tuple(int(value.strip()) for value in args.umap_seeds.split(",") if value.strip())
    if MAIN_SEED not in seeds or len(set(seeds)) < 5:
        raise ValueError("--umap-seeds must contain 2026 and at least five fixed seeds")
    seeds = tuple(dict.fromkeys(seeds))
    UMAP = _import_umap()
    neighbors = min(int(args.umap_neighbors), len(fit_source_pre) - 1)
    if neighbors < 2:
        raise ValueError("UMAP needs at least three source-pre samples")
    outputs, quality = {}, {}
    for seed in seeds:
        reducer = UMAP(
            n_components=2, n_neighbors=neighbors,
            min_dist=float(args.umap_min_dist), metric=args.umap_metric,
            random_state=int(seed), transform_seed=int(seed), n_jobs=1,
        )
        reducer.fit(fit_source_pre)
        pre_coordinates = reducer.transform(pre)
        post_coordinates = reducer.transform(post)
        outputs[int(seed)] = (pre_coordinates, post_coordinates)
        quality[str(seed)] = {
            "pre": embedding_quality(pre, pre_coordinates, args.quality_neighbors),
            "post": embedding_quality(post, post_coordinates, args.quality_neighbors),
        }
        frame = _coordinate_frame(metadata, pre_coordinates, post_coordinates)
        frame.to_csv(os.path.join(
            args.output_dir, "umap_coordinates_seed-{}.csv".format(seed)
        ), index=False)
    stability = []
    main_pre, main_post = outputs[MAIN_SEED]
    for seed in seeds:
        candidate_pre, candidate_post = outputs[seed]
        stability.append({
            "seed": int(seed),
            "pre": _stability(main_pre, candidate_pre),
            "post": _stability(main_post, candidate_post),
        })
    parameters = {
        "n_components": 2, "n_neighbors": int(neighbors),
        "min_dist": float(args.umap_min_dist), "metric": args.umap_metric,
        "main_random_state": MAIN_SEED, "main_transform_seed": MAIN_SEED,
        "n_jobs": 1,
        "stability_seeds": [int(seed) for seed in seeds],
        "fit_partition": "all_source_train_pre_only",
        "transformed_partitions": ["source_pre", "target_pre", "source_post", "target_post"],
    }
    return outputs, quality, stability, parameters


def _effective_perplexity(requested, sample_count):
    if sample_count < 4:
        raise ValueError("Joint t-SNE requires at least four observations")
    recommended_max = max(1.0, (float(sample_count) - 1.0) / 3.0)
    return float(min(float(requested), recommended_max, float(sample_count) - 1e-6))


def fit_joint_tsne(pre, post, metadata, args):
    paired = np.vstack([pre, post])
    perplexity = _effective_perplexity(args.tsne_perplexity, len(paired))
    signature = inspect.signature(TSNE).parameters
    kwargs = {
        "n_components": 2, "init": "pca", "perplexity": perplexity,
        "random_state": MAIN_SEED,
    }
    version_parts = tuple(
        int(part) for part in sklearn.__version__.split(".")[:2]
        if part.isdigit()
    )
    supports_auto_learning_rate = version_parts >= (1, 2)
    if "max_iter" in signature:
        kwargs["max_iter"] = int(args.tsne_max_iter)
    else:
        kwargs["n_iter"] = int(args.tsne_max_iter)
    if supports_auto_learning_rate:
        kwargs["learning_rate"] = "auto"
        actual_learning_rate = "auto"
    else:
        actual_learning_rate = max(float(len(paired)) / 48.0, 50.0)
        kwargs["learning_rate"] = actual_learning_rate
    coordinates = TSNE(**kwargs).fit_transform(paired)
    pre_coordinates = coordinates[:len(pre)]
    post_coordinates = coordinates[len(pre):]
    frame = _coordinate_frame(metadata, pre_coordinates, post_coordinates)
    frame.to_csv(os.path.join(
        args.output_dir, "tsne_joint_coordinates_seed-2026.csv"
    ), index=False)
    quality = {
        "joint": embedding_quality(paired, coordinates, args.quality_neighbors),
        "pre": embedding_quality(pre, pre_coordinates, args.quality_neighbors),
        "post": embedding_quality(post, post_coordinates, args.quality_neighbors),
    }
    parameters = {
        "n_components": 2, "init": "pca",
        "learning_rate": actual_learning_rate, "perplexity": perplexity,
        "learning_rate_compatibility": (
            "native_auto" if supports_auto_learning_rate
            else "numeric equivalent fallback for scikit-learn < 1.2"
        ),
        "requested_perplexity": float(args.tsne_perplexity),
        "max_iter": int(args.tsne_max_iter), "random_state": MAIN_SEED,
        "fit_order": "[all balanced pre features; all paired balanced post features]",
        "joint_fit": True,
        "caption": "t-SNE was jointly fitted to paired pre- and post-alignment representations.",
    }
    return pre_coordinates, post_coordinates, quality, parameters


def _coordinate_frame(metadata, pre, post):
    frames = []
    for stage, coordinates in [("pre", pre), ("post", post)]:
        frame = metadata.copy()
        frame["stage"] = stage
        frame["embedding_x"] = coordinates[:, 0]
        frame["embedding_y"] = coordinates[:, 1]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _font_family():
    for family in ["Arial", "Helvetica", "DejaVu Sans"]:
        try:
            font_manager.findfont(family, fallback_to_default=False)
            return family
        except ValueError:
            continue
    return "DejaVu Sans"


def _class_color(index, class_count):
    if int(class_count) == 2:
        palette = ["#3B75AF", "#D9534F"]
    elif int(class_count) == 3:
        palette = ["#3B75AF", "#D9A62E", "#D9534F"]
    else:
        palette = CLASS_COLORS
    return palette[int(index) % len(palette)]


def _limits(pre, post):
    combined = np.vstack([pre, post])
    output = []
    for axis in range(2):
        low, high = float(combined[:, axis].min()), float(combined[:, axis].max())
        span = max(high - low, 1e-6)
        output.append((low - 0.05 * span, high + 0.05 * span))
    return output


def _ellipse(ax, points, color):
    if len(points) < 5:
        return
    covariance = np.cov(points, rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    if not np.isfinite(values).all() or float(np.min(values)) <= 0.0:
        return
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    angle = np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0]))
    scale = np.sqrt(-2.0 * np.log(1.0 - 0.80))
    ax.add_patch(Ellipse(
        points.mean(axis=0), 2 * scale * np.sqrt(values[0]),
        2 * scale * np.sqrt(values[1]), angle=angle,
        facecolor=color, edgecolor=color, linewidth=0.7, alpha=0.07, zorder=1,
    ))


def _draw_embedding(ax, coordinates, metadata, title, limits, ellipses=True):
    class_rows = metadata[["class_id", "class_name"]].drop_duplicates().sort_values("class_id")
    for index, row in enumerate(class_rows.itertuples(index=False)):
        class_id = int(row.class_id)
        color = _class_color(index, len(class_rows))
        class_mask = metadata["class_id"].to_numpy(dtype=np.int64) == class_id
        if ellipses:
            _ellipse(ax, coordinates[class_mask], color)
        source = class_mask & (metadata["domain"].astype(str).to_numpy() == "source")
        target = class_mask & (metadata["domain"].astype(str).to_numpy() == "target")
        ax.scatter(coordinates[source, 0], coordinates[source, 1], s=10,
                   marker="o", c=color, edgecolors="none", alpha=0.34, zorder=2)
        ax.scatter(coordinates[target, 0], coordinates[target, 1], s=34,
                   marker="^", facecolors="none", edgecolors=color,
                   linewidths=0.8, alpha=0.95, zorder=5)
        center = coordinates[class_mask].mean(axis=0)
        ax.scatter(center[0], center[1], s=48, marker="X", c=color,
                   edgecolors="white", linewidths=0.6, zorder=6)
    ax.set_title(title, pad=5)
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_xlabel("Embedding dimension 1")
    ax.set_ylabel("Embedding dimension 2")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=0.7, length=3, labelsize=7.2)


def _legend_handles(metadata):
    class_rows = metadata[["class_id", "class_name"]].drop_duplicates().sort_values("class_id")
    class_handles = [
        Line2D([0], [0], marker="o", linestyle="None", label=str(row.class_name),
               markerfacecolor=_class_color(index, len(class_rows)),
               markeredgecolor="none", markersize=5)
        for index, row in enumerate(class_rows.itertuples(index=False))
    ]
    domain_handles = [
        Line2D([0], [0], marker="o", linestyle="None", label="Source",
               markerfacecolor="#777777", markeredgecolor="none", markersize=4.5),
        Line2D([0], [0], marker="^", linestyle="None", label="Target",
               markerfacecolor="none", markeredgecolor="#333333", markersize=5.5),
        Line2D([0], [0], marker="X", linestyle="None", label="Class center",
               markerfacecolor="#777777", markeredgecolor="white", markersize=5.5),
    ]
    return class_handles, domain_handles


def _save_figure(fig, stem, description):
    creator = "plot_nonlinear_alignment_embeddings.py"
    fig.savefig(stem + ".pdf", bbox_inches="tight", metadata={
        "Creator": creator, "Subject": description,
    })
    fig.savefig(stem + ".svg", bbox_inches="tight", metadata={
        "Creator": creator, "Description": description,
    })
    fig.savefig(stem + ".png", dpi=600, bbox_inches="tight", metadata={
        "Creator": creator, "Description": description,
    })
    plt.close(fig)


def plot_pair(pre, post, metadata, method, output_dir, ellipses, caption):
    limits = _limits(pre, post)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.15))
    _draw_embedding(axes[0], pre, metadata, "Before SPDDSBN", limits, ellipses)
    _draw_embedding(axes[1], post, metadata, "After SPDDSBN", limits, ellipses)
    axes[0].text(-0.12, 1.04, "(a)", transform=axes[0].transAxes, fontweight="bold")
    axes[1].text(-0.12, 1.04, "(b)", transform=axes[1].transAxes, fontweight="bold")
    class_handles, domain_handles = _legend_handles(metadata)
    first = fig.legend(class_handles, [h.get_label() for h in class_handles],
                       loc="upper center", bbox_to_anchor=(0.34, 1.01),
                       ncol=len(class_handles), frameon=False, title="Class",
                       fontsize=7.2, title_fontsize=7.4)
    fig.add_artist(first)
    fig.legend(domain_handles, [h.get_label() for h in domain_handles],
               loc="upper center", bbox_to_anchor=(0.79, 1.01), ncol=3,
               frameon=False, title="Domain", fontsize=7.2, title_fontsize=7.4)
    fig.text(0.5, 0.01, caption, ha="center", va="bottom", fontsize=6.8)
    fig.subplots_adjust(top=0.78, bottom=0.18, left=0.08, right=0.98, wspace=0.24)
    _save_figure(
        fig, os.path.join(output_dir, "fig_{}_alignment".format(method)), caption
    )
    return limits


def plot_stability(umap_outputs, metadata, stability, output_dir):
    seeds = list(umap_outputs)
    fig, axes = plt.subplots(1, len(seeds), figsize=(7.1, 1.85))
    if len(seeds) == 1:
        axes = [axes]
    stability_by_seed = {int(row["seed"]): row for row in stability}
    for ax, seed in zip(axes, seeds):
        _, post = umap_outputs[seed]
        limits = _limits(post, post)
        _draw_embedding(ax, post, metadata, "Seed {}".format(seed), limits, False)
        rho = stability_by_seed[seed]["post"]["pairwise_distance_spearman"]
        ax.text(0.5, -0.20, "rho={:.3f}".format(rho), transform=ax.transAxes,
                ha="center", va="top", fontsize=6.5)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    fig.suptitle("UMAP stability: After SPDDSBN (reference seed 2026)", y=0.98,
                 fontsize=8.5)
    fig.subplots_adjust(top=0.76, bottom=0.19, left=0.02, right=0.99, wspace=0.18)
    _save_figure(
        fig, os.path.join(output_dir, "fig_nonlinear_stability"),
        "Fixed-seed UMAP stability; seed 2026 is the prespecified main result.",
    )


def main():
    args = parse_args()
    if args.batch_size < 1 or args.max_points_per_class_domain < 1:
        raise ValueError("Batch size and balanced sample limit must be positive")
    if args.umap_neighbors < 2 or not 0.0 <= args.umap_min_dist <= 1.0:
        raise ValueError("Invalid UMAP neighborhood/min_dist")
    if args.tsne_max_iter < 250 or args.quality_neighbors < 1:
        raise ValueError("Invalid t-SNE iteration or quality-neighbor setting")
    _import_umap()
    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(MAIN_SEED)
    np.random.seed(MAIN_SEED)
    torch.manual_seed(MAIN_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(MAIN_SEED)
    matplotlib.rcParams.update({
        "font.family": _font_family(), "font.size": 8.0,
        "axes.labelsize": 7.8, "axes.titlesize": 9.0,
        "axes.linewidth": 0.8, "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.facecolor": "white",
        "svg.hashsalt": "nonlinear-spddsbn-2026",
    })

    run_record = _resolve_inputs(args)
    subject, selection = _select_fold(args)
    checkpoint = _checkpoint_path(args.checkpoint_root, subject)
    pre_spd, post_spd, metadata, provenance = _extract_or_load(
        args, subject, checkpoint
    )
    checks = {
        "pre": validate_spd_matrices(pre_spd, "P_pre"),
        "post": validate_spd_matrices(post_spd, "P_post"),
    }
    if pre_spd.shape != post_spd.shape or len(pre_spd) != len(metadata):
        raise RuntimeError("P_pre/P_post sample pairing failed")
    if metadata["sample_id"].duplicated().any():
        raise RuntimeError("Sample IDs are not unique")

    source = metadata["domain"].astype(str).to_numpy() == "source"
    reference, karcher = airm_karcher_mean(
        pre_spd[source], tolerance=args.karcher_tol,
        max_iterations=args.karcher_max_iter,
        batch_size=args.riemann_batch_size,
    )
    pre_tangent = common_tangent_vectors(
        pre_spd, reference, batch_size=args.riemann_batch_size
    )
    post_tangent = common_tangent_vectors(
        post_spd, reference, batch_size=args.riemann_batch_size
    )
    scaler = StandardScaler().fit(pre_tangent[source])
    pre_scaled = scaler.transform(pre_tangent)
    post_scaled = scaler.transform(post_tangent)
    expected_dimension = pre_spd.shape[-1] * (pre_spd.shape[-1] + 1) // 2
    if pre_scaled.shape[1] != expected_dimension or post_scaled.shape != pre_scaled.shape:
        raise RuntimeError("Common tangent feature dimensions do not match")

    manifest = balanced_plot_manifest(
        metadata, max_per_class_domain=args.max_points_per_class_domain,
        seed=args.sampling_seed,
    )
    manifest.to_csv(os.path.join(args.output_dir, "plot_sample_manifest.csv"), index=False)
    positions = _manifest_positions(metadata, manifest)
    balanced_metadata = metadata.iloc[positions].reset_index(drop=True)
    if not np.array_equal(
        balanced_metadata["sample_id"].to_numpy(dtype=np.int64),
        manifest["sample_id"].to_numpy(dtype=np.int64),
    ):
        raise RuntimeError("Manifest/sample ordering failed")
    balanced_pre = pre_scaled[positions]
    balanced_post = post_scaled[positions]

    high_metrics = high_dimensional_metrics(
        balanced_pre, balanced_post, balanced_metadata
    )
    umap_outputs, umap_quality, stability, umap_parameters = fit_umap_runs(
        pre_scaled[source], balanced_pre, balanced_post, balanced_metadata, args
    )
    tsne_pre, tsne_post, tsne_quality, tsne_parameters = fit_joint_tsne(
        balanced_pre, balanced_post, balanced_metadata, args
    )
    umap_pre, umap_post = umap_outputs[MAIN_SEED]

    umap_limits = plot_pair(
        umap_pre, umap_post, balanced_metadata, "umap", args.output_dir,
        not args.no_ellipses,
        "UMAP fitted on source-pre only; seed 2026 is prespecified.",
    )
    tsne_limits = plot_pair(
        tsne_pre, tsne_post, balanced_metadata, "tsne_joint", args.output_dir,
        not args.no_ellipses,
        "t-SNE was jointly fitted to paired pre- and post-alignment representations.",
    )
    plot_stability(umap_outputs, balanced_metadata, stability, args.output_dir)

    np.savez_compressed(
        os.path.join(args.output_dir, "nonlinear_embedding_features.npz"),
        sample_id=balanced_metadata["sample_id"].to_numpy(dtype=np.int64),
        tangent_reference=reference, scaler_mean=scaler.mean_, scaler_scale=scaler.scale_,
        pre_features=balanced_pre, post_features=balanced_post,
    )
    stability_correlations = [
        row[stage]["pairwise_distance_spearman"]
        for row in stability if row["seed"] != MAIN_SEED for stage in ["pre", "post"]
    ]
    metrics = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "dataset": _dataset_name(args), "model_type": args.model,
        "representative_fold": selection, "checkpoint": checkpoint,
        "feature_locations": visualization_model_metadata(args.model),
        "feature_provenance": provenance,
        "common_tangent_space": {
            "reference_fit": "all source-train P_pre only",
            "reference_method": "AIRM Karcher/Frechet mean",
            "standardizer_fit": "all source-train pre tangent vectors only",
            "input_spd_dimension": int(pre_spd.shape[-1]),
            "feature_dimension": int(expected_dimension),
            "karcher_diagnostics": karcher,
        },
        "balanced_sampling": {
            "seed": int(args.sampling_seed), "sample_count": int(len(manifest)),
            "paired_stage_observations": int(2 * len(manifest)),
            "counts": manifest.groupby(["domain", "class_name"]).size().to_dict(),
            "target_labels_usage": (
                "offline domain-by-class balanced sampling, grouping, plotting, "
                "and class-conditional quality metrics only"
            ),
        },
        "umap": {
            "parameters": umap_parameters, "quality_by_seed": umap_quality,
            "stability_against_seed_2026": stability,
            "stability_summary": {
                "minimum_pairwise_distance_spearman": float(min(stability_correlations)),
                "mean_pairwise_distance_spearman": float(np.mean(stability_correlations)),
            },
            "axis_limits_main": umap_limits,
        },
        "tsne": {
            "parameters": tsne_parameters, "quality": tsne_quality,
            "axis_limits": tsne_limits,
        },
        "high_dimensional_evidence": high_metrics,
        "interpretation_guardrail": (
            "UMAP/t-SNE are nonlinear visualization supplements only. Formal "
            "alignment/class conclusions must use these high-dimensional metrics "
            "or the separate AIRM/Frechet analysis, never 2D centroid distances."
        ),
        "numerical_checks": checks,
        "run_record": run_record,
        "versions": {
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "torch": torch.__version__,
            "sklearn": sklearn.__version__, "matplotlib": matplotlib.__version__,
            "umap": getattr(sys.modules.get("umap"), "__version__", None),
        },
    }
    _write_json(metrics, os.path.join(args.output_dir, "nonlinear_embedding_metrics.json"))

    print("Representative target subject:", subject)
    print("Common tangent feature dimension:", expected_dimension)
    print("Balanced samples / paired t-SNE observations:", len(manifest), 2 * len(manifest))
    print("UMAP parameters:", umap_parameters)
    print("UMAP seed-2026 quality:", umap_quality[str(MAIN_SEED)])
    print("UMAP stability summary:", metrics["umap"]["stability_summary"])
    print("Joint t-SNE parameters:", tsne_parameters)
    print("Joint t-SNE quality:", tsne_quality)
    print("High-dimensional evidence:", high_metrics)
    print("Saved:", os.path.abspath(args.output_dir))


if __name__ == "__main__":
    main()
