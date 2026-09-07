"""IEEE-style cross-subject representation-alignment composite figure.

The default four columns are the controlled MS-TGC ablations requested for
Fig. 5. Embeddings are descriptive only. Quantitative panels are computed in
the source-standardized high-dimensional feature space and can aggregate all
common LOSO folds. Target labels are used only after training for balanced
sampling and class-conditional reporting.
"""

import argparse
import inspect
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
import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import torch


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.cl_tsmnet.datasets import load_dataset
from src.cl_tsmnet.experiment_utils import default_cache_path, default_target_fs
from src.cl_tsmnet.spd_pca import (
    balanced_plot_manifest,
    choose_median_fold,
    migrate_legacy_spddsbn_buffers,
)
from src.cl_tsmnet.spd_visualization_adapters import (
    SUPPORTED_ALIGNMENT_MODELS,
    extract_alignment_representation,
)
from src.cl_tsmnet.splits import domain_ids, make_split
from src.cl_tsmnet.training import (
    _filter_artifact_windows,
    build_ms_tgc_spddsbn,
    build_tsmnet,
    fit_source_normalizer,
)


RANDOM_SEED = 2026
FIGURE_SIZE = (7.16, 4.95)
CLASS_COLORS = ["#377EB8", "#E69F00", "#009E73", "#CC79A7", "#7F7F7F"]
DATASET_STYLES = [
    ("#4C78A8", "o", "-"),
    ("#8F5B78", "s", "--"),
    ("#4B7F52", "D", "-."),
]
DEFAULT_METHODS = [
    {
        "label": "Mean-CE",
        "short_label": "Mean\nCE",
        "model_type": "mstgc_mean_ce",
        "bnorm": None,
    },
    {
        "label": "Mean-EuDSBN",
        "short_label": "Mean\nEuDSBN",
        "model_type": "mstgc_dta_cheb_eudsbn",
        "bnorm": None,
    },
    {
        "label": "AugSPD-SPDBN",
        "short_label": "AugSPD\nSPDBN",
        "model_type": "mstgc_dta_cheb_spdbn",
        "bnorm": None,
    },
    {
        "label": "AugSPD-SPDDSBN",
        "short_label": "AugSPD\nSPDDSBN",
        "model_type": "ms_tgc_spddsbn",
        "bnorm": None,
    },
]
EXPECTED_MSTGC_PROVENANCE = {
    "mstgc_mean_ce": {
        "architecture": "shared_channel_graph_mean_v3",
        "representation": "mean",
    },
    "mstgc_dta_cheb_eudsbn": {
        "architecture": "shared_channel_graph_mean_v3",
        "representation": "mean",
    },
    "mstgc_dta_cheb_spdbn": {
        "architecture": "shared_channel_graph_augmented_spd_v3",
        "representation": "augmented",
    },
    "ms_tgc_spddsbn": {
        "architecture": "shared_channel_graph_augmented_spd_v3",
        "representation": "augmented",
    },
}
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
        description="Generate IEEE Fig. 5 cross-subject alignment panels."
    )
    parser.add_argument(
        "--datasets", default="stew,cog-bci:nback",
        help="One or two entries: stew,eegmat,cog-bci:nback,cog-bci:matb.",
    )
    parser.add_argument(
        "--dataset-labels", default="STEW,N-Back",
        help="Comma-separated display names in the same order as --datasets.",
    )
    parser.add_argument(
        "--method-manifest", default=None,
        help=(
            "Optional JSON file containing exactly four method objects with "
            "label, short_label, model_type, optional bnorm/run_dir. This is "
            "the generic interface for MSTGC and TSMNet checkpoints."
        ),
    )
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--cache-root", default=os.path.join("outputs", "cache"))
    parser.add_argument("--master-summary", default=os.path.join("outputs", "master_summary.csv"))
    parser.add_argument("--output-dir", default="results")
    parser.add_argument(
        "--feature-cache-dir",
        default=os.path.join("results", "fig5_representation_cache"),
    )
    parser.add_argument("--target-fs-stew", type=float, default=None)
    parser.add_argument("--target-fs-eegmat", type=float, default=None)
    parser.add_argument("--target-fs-cog-bci", type=float, default=None)
    parser.add_argument(
        "--target-subjects", default="",
        help="Optional dataset=subject pairs, e.g. stew=6,cog-bci-nback=12.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-points-per-group", type=int, default=200)
    parser.add_argument(
        "--metric-scope", choices=["all", "representative"], default="all",
        help="Publication default aggregates high-dimensional metrics over all common LOSO folds.",
    )
    parser.add_argument("--reducer", choices=["auto", "umap", "tsne"], default="auto")
    parser.add_argument("--pca-dim", type=int, default=30)
    parser.add_argument("--umap-neighbors", type=int, default=15)
    parser.add_argument("--umap-min-dist", type=float, default=0.15)
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument("--tsne-max-iter", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--force-reextract", action="store_true")
    parser.add_argument("--allow-missing-master-config", action="store_true")
    parser.add_argument("--allow-legacy-refit", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
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


def _usable(value):
    if value is None:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return value


def _as_bool(value):
    return str(value).strip().lower() in {"true", "1", "yes"}


def _font_family():
    for family in ["Arial", "Helvetica", "DejaVu Sans"]:
        try:
            font_manager.findfont(family, fallback_to_default=False)
            return family
        except ValueError:
            continue
    return "DejaVu Sans"


def _parse_datasets(values, labels):
    raw = [item.strip() for item in values.split(",") if item.strip()]
    display = [item.strip() for item in labels.split(",") if item.strip()]
    if len(raw) not in {1, 2} or len(display) != len(raw):
        raise ValueError(
            "Fig. 5 requires one or two datasets and the same number of labels"
        )
    parsed = []
    for value, label in zip(raw, display):
        parts = value.split(":", 1)
        dataset = parts[0].lower()
        paradigm = parts[1].lower() if len(parts) == 2 else "nback"
        if dataset not in {"stew", "eegmat", "cog-bci"}:
            raise ValueError("Unsupported dataset specification: {}".format(value))
        if dataset == "cog-bci" and paradigm not in {"nback", "matb"}:
            raise ValueError("COG-BCI paradigm must be nback or matb")
        name = "cog-bci-{}".format(paradigm) if dataset == "cog-bci" else dataset
        normalized_label = "".join(
            character for character in label.lower() if character.isalnum()
        )
        reserved_labels = {
            "stew": "stew",
            "eegmat": "eegmat",
            "nback": "cog-bci-nback",
            "cogbcinback": "cog-bci-nback",
            "matb": "cog-bci-matb",
            "cogbcimatb": "cog-bci-matb",
        }
        claimed_dataset = reserved_labels.get(normalized_label)
        if claimed_dataset is not None and claimed_dataset != name:
            raise ValueError(
                "Display label {!r} identifies {}, but the corresponding "
                "dataset is {}".format(label, claimed_dataset, name)
            )
        parsed.append({
            "dataset": dataset,
            "paradigm": paradigm,
            "name": name,
            "display_name": label,
        })
    return parsed


def _load_methods(path):
    if path is None:
        methods = [dict(item) for item in DEFAULT_METHODS]
    else:
        if not os.path.exists(path):
            raise FileNotFoundError("Method manifest not found: {}".format(path))
        with open(path, encoding="utf-8") as handle:
            methods = json.load(handle)
    if not isinstance(methods, list) or len(methods) != 4:
        raise ValueError("The method manifest must contain exactly four methods")
    required = {"label", "model_type"}
    for index, method in enumerate(methods):
        missing = required - set(method)
        if missing:
            raise ValueError("Method {} lacks {}".format(index + 1, sorted(missing)))
        if method["model_type"] not in SUPPORTED_ALIGNMENT_MODELS:
            raise ValueError(
                "Unsupported alignment model {!r}; supported: {}".format(
                    method["model_type"], sorted(SUPPORTED_ALIGNMENT_MODELS)
                )
            )
        method.setdefault("short_label", str(method["label"]).replace("-", "\n", 1))
        method.setdefault("bnorm", "spddsbn" if method["model_type"] == "tsmnet" else None)
    return methods


def _target_subject_overrides(text):
    result = {}
    for item in [value.strip() for value in text.split(",") if value.strip()]:
        if "=" not in item:
            raise ValueError("Invalid --target-subjects item: {}".format(item))
        name, value = item.split("=", 1)
        result[name.strip()] = int(value)
    return result


def _target_fs(dataset_spec, args):
    override = getattr(args, "target_fs_{}".format(
        dataset_spec["dataset"].replace("-", "_")
    ))
    return default_target_fs(dataset_spec["dataset"], override)


def _candidate_run_dirs(dataset_name, method, output_root):
    if method.get("run_dir"):
        rendered = str(method["run_dir"]).format(
            dataset=dataset_name, model=method["model_type"],
            bnorm=method.get("bnorm") or "none",
        )
        return [rendered if os.path.isabs(rendered) else os.path.join(output_root, rendered)]
    model_type = method["model_type"]
    if model_type == "tsmnet":
        bnorm = method.get("bnorm") or "none"
        return [
            os.path.join(output_root, "{}_loso_tsmnet_{}".format(dataset_name, bnorm)),
            os.path.join(output_root, "{}_loso_{}".format(dataset_name, bnorm)),
        ]
    return [os.path.join(output_root, "{}_loso_{}".format(dataset_name, model_type))]


def _matching_master_row(master, dataset_name, method, run_dir):
    if master is None or master.empty:
        return {}
    frame = master.copy()
    for column, value in [
        ("dataset", dataset_name), ("protocol", "loso"),
        ("model_type", method["model_type"]),
    ]:
        if column in frame:
            frame = frame[frame[column].astype(str) == str(value)]
    if method["model_type"] == "tsmnet" and "model" in frame:
        expected = "tsmnet_{}".format(method.get("bnorm") or "none")
        exact = frame[frame["model"].astype(str) == expected]
        if not exact.empty:
            frame = exact
    if "output_dir" in frame:
        normalized = os.path.normcase(os.path.abspath(run_dir))
        exact = frame[frame["output_dir"].astype(str).map(
            lambda value: os.path.normcase(os.path.abspath(value)) == normalized
        )]
        if exact.empty:
            return {}
        frame = exact
    return {} if frame.empty else frame.iloc[-1].dropna().to_dict()


def _resolve_run(dataset_spec, method, args, master):
    candidates = _candidate_run_dirs(dataset_spec["name"], method, args.output_root)
    run_dir = next((path for path in candidates if os.path.isdir(path)), None)
    if run_dir is None:
        raise FileNotFoundError(
            "Missing LOSO output for {} / {}. Tried:\n  {}\nRun the model "
            "experiment first or set run_dir in --method-manifest.".format(
                dataset_spec["display_name"], method["label"],
                "\n  ".join(os.path.abspath(path) for path in candidates),
            )
        )
    summary_path = os.path.join(run_dir, "summary.csv")
    if not os.path.exists(summary_path):
        raise FileNotFoundError("Missing per-subject summary: {}".format(summary_path))
    summary = pd.read_csv(summary_path)
    if "protocol" in summary:
        summary = summary[summary["protocol"].astype(str) == "loso"]
    if "model_type" in summary:
        summary = summary[summary["model_type"].astype(str) == method["model_type"]]
    if summary.empty:
        raise ValueError("No matching LOSO rows in {}".format(summary_path))
    record = _matching_master_row(master, dataset_spec["name"], method, run_dir)
    if not record and not args.allow_missing_master_config:
        raise ValueError(
            "No matching training configuration in {} for {} / {}. Exact "
            "checkpoint reconstruction is required for publication output; "
            "use --allow-missing-master-config only for audited legacy runs."
            .format(args.master_summary, dataset_spec["name"], method["label"])
        )
    return {
        "run_dir": os.path.abspath(run_dir),
        "summary_path": os.path.abspath(summary_path),
        "summary": summary,
        "record": record,
    }


def _record_value(record, key, default, cast):
    value = _usable(record.get(key))
    if value is None:
        return default
    return cast(value)


def _model_config(record):
    integer_keys = {
        "temporal_filters", "spatial_filters", "subspacedims", "temp_kernel",
        "mstgc_temporal_hidden", "mstgc_graph_hidden", "mstgc_fusion_dim",
        "mstgc_kernel_length", "mstgc_num_heads", "mstgc_cheby_order",
        "mstgc_num_nodes", "mstgc_graph_k", "mstgc_time_points",
    }
    config = {}
    for key, default in MODEL_DEFAULTS.items():
        cast = int if key in integer_keys else float
        config[key] = _record_value(record, key, default, cast)
    return config


def _split_config(run_info):
    record = run_info["record"]
    summary = run_info["summary"]
    artifact = _usable(record.get("artifact_z"))
    if artifact is None and "artifact_z" in summary:
        values = summary["artifact_z"].dropna()
        artifact = _usable(values.iloc[0]) if len(values) else None
    return {
        "seed": _record_value(record, "seed", 42, int),
        "val_size": _record_value(record, "val_size", 0.2, float),
        "test_size": _record_value(record, "test_size", 0.2, float),
        "artifact_z": None if artifact is None else float(artifact),
    }


def _validate_comparable_runs(run_infos, methods, args):
    reference = _split_config(run_infos[-1])
    keys = ["seed", "val_size", "test_size", "artifact_z"]
    frontend_keys = [
        "mstgc_temporal_hidden", "mstgc_graph_hidden", "mstgc_fusion_dim",
        "mstgc_kernel_length", "mstgc_num_heads", "mstgc_cheby_order",
        "mstgc_dropout", "mstgc_num_nodes", "mstgc_graph_k",
        "mstgc_time_points", "mstgc_graph_density",
    ]
    reference_frontend = _model_config(run_infos[-1]["record"])
    errors = []
    for info, method in zip(run_infos, methods):
        current = _split_config(info)
        differences = [key for key in keys if current[key] != reference[key]]
        if differences:
            errors.append(
                "{} has different split/preprocessing fields: {}"
                .format(method["label"], ", ".join(differences))
            )
        if method["model_type"] in EXPECTED_MSTGC_PROVENANCE:
            expected = EXPECTED_MSTGC_PROVENANCE[method["model_type"]]
            architecture = _usable(info["record"].get("mstgc_architecture"))
            representation = _usable(info["record"].get("mstgc_representation"))
            if architecture != expected["architecture"]:
                errors.append(
                    "{} architecture is {!r}, expected {!r}; this checkpoint "
                    "predates or differs from the current v3 comparison"
                    .format(method["label"], architecture, expected["architecture"])
                )
            if representation != expected["representation"]:
                errors.append(
                    "{} representation is {!r}, expected {!r}"
                    .format(method["label"], representation, expected["representation"])
                )
            current_frontend = _model_config(info["record"])
            changed_frontend = [
                key for key in frontend_keys
                if current_frontend[key] != reference_frontend[key]
            ]
            if changed_frontend:
                errors.append(
                    "{} uses different shared-front-end fields: {}"
                    .format(method["label"], ", ".join(changed_frontend))
                )
        summary = info["summary"]
        expects_adapt = method["model_type"] in {
            "mstgc_dta_cheb_eudsbn", "ms_tgc_spddsbn",
            "mstgc_augspd_spddsbn",
        } or (
            method["model_type"] == "tsmnet"
            and method.get("bnorm") == "spddsbn"
        )
        if "target_adapt" not in summary:
            errors.append("{} summary lacks target_adapt audit data".format(method["label"]))
            actual = pd.Series(dtype=bool)
        else:
            actual = summary["target_adapt"].map(_as_bool)
        if expects_adapt and (actual.empty or not bool(actual.all())):
            errors.append("{} checkpoint was not target-adapted".format(method["label"]))
        if not expects_adapt and not actual.empty and bool(actual.any()):
            errors.append("{} unexpectedly reports target adaptation".format(method["label"]))
        if expects_adapt:
            scopes = set(summary.get("target_refit_scope", pd.Series(dtype=str)).dropna().astype(str))
            scope_evidence = "summary.csv"
            if not scopes:
                master_scope = _usable(info["record"].get("target_refit_scope"))
                if master_scope is not None:
                    scopes = {str(master_scope)}
                    scope_evidence = "master_summary.csv"
            if scopes != {"target_only"}:
                if not args.allow_legacy_refit:
                    errors.append(
                        "{} must report target_refit_scope=target_only; found {}"
                        .format(method["label"], sorted(scopes))
                    )
                else:
                    warnings.warn(
                        "{} uses unaudited legacy refit scope {}"
                        .format(method["label"], sorted(scopes)), UserWarning,
                    )
            info["refit_scope_audit"] = {
                "values": sorted(scopes), "evidence": scope_evidence,
            }
    if errors:
        raise ValueError(
            "Fig. 5 run audit failed:\n- " + "\n- ".join(errors)
            + "\nRerun the listed methods with the current training pipeline; "
              "do not mix legacy and v3 checkpoints in the publication figure."
        )
    return reference


def _checkpoint_path(run_dir, subject):
    candidates = [
        os.path.join(run_dir, "subject_{:02d}".format(int(subject)), "model.pt"),
        os.path.join(run_dir, "subject_{}".format(int(subject)), "model.pt"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)
    raise FileNotFoundError("No checkpoint for subject {}; tried {}".format(subject, candidates))


def _load_state(path):
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError("Checkpoint is not a PyTorch state_dict: {}".format(path))
    return state


def _build_model(dataset, domains, selected, source_train, method, config, device):
    shape = dataset["x"].shape
    labels = np.unique(dataset["y"][source_train]).astype(np.int64)
    nclasses = int(len(labels))
    if method["model_type"] == "tsmnet":
        model = build_tsmnet(
            PROJECT_ROOT, shape[1], shape[2], nclasses, domains[selected],
            bnorm=method.get("bnorm"),
            temporal_filters=config["temporal_filters"],
            spatial_filters=config["spatial_filters"],
            subspacedims=config["subspacedims"],
            temp_kernel=config["temp_kernel"], device=device,
        )
    else:
        kernel = max(3, int(round(
            config["mstgc_kernel_length"] * float(dataset["fs"]) / 128.0
        )))
        model = build_ms_tgc_spddsbn(
            PROJECT_ROOT, shape[1], shape[2], nclasses, domains[selected],
            subspacedims=config["subspacedims"], device=device,
            temporal_hidden=config["mstgc_temporal_hidden"],
            graph_hidden=config["mstgc_graph_hidden"],
            fusion_dim=config["mstgc_fusion_dim"], kernel_length=kernel,
            num_heads=config["mstgc_num_heads"],
            cheby_order=config["mstgc_cheby_order"],
            dropout=config["mstgc_dropout"], num_nodes=config["mstgc_num_nodes"],
            variant=method["model_type"], graph_mode="adaptive",
            graph_neighbors=config["mstgc_graph_k"],
            graph_density=config["mstgc_graph_density"],
            graph_time_points=config["mstgc_time_points"],
            covariance_shrinkage=config["mstgc_shrinkage"],
        )
    return model.to(device).eval()


def _feature_metadata(dataset, ids, domains, source_ids):
    rows = dataset["meta"].iloc[ids]
    class_ids = dataset["y"][ids].astype(np.int64)
    names = dataset.get("label_names", {})
    return pd.DataFrame({
        "sample_id": ids,
        "subject_id": rows["subject"].to_numpy(dtype=np.int64),
        "domain_id": domains[ids].astype(np.int64),
        "domain": np.where(np.isin(ids, source_ids), "source", "target"),
        "class_id": class_ids,
        "class_name": [str(names.get(int(value), "class {}".format(value))) for value in class_ids],
        "split": np.where(np.isin(ids, source_ids), "source_train", "target_test"),
        "session": rows["session"].to_numpy(dtype=np.int64),
        "task": rows["task"].astype(str).to_numpy(),
        "start_sample": rows["start_sample"].to_numpy(dtype=np.int64),
    })


def _extract_vectors(model, method, dataset, ids, domains, normalizer, batch_size, device):
    parts = []
    extraction_metadata = None
    with torch.no_grad():
        for start in range(0, len(ids), int(batch_size)):
            batch_ids = ids[start:start + int(batch_size)]
            windows = normalizer.transform_array(dataset["x"][batch_ids])
            xb = torch.from_numpy(windows).to(device=device, dtype=torch.float32)
            db = torch.from_numpy(domains[batch_ids]).to(device=device, dtype=torch.long)
            values, current = extract_alignment_representation(
                model, xb, db, method["model_type"]
            )
            if values.ndim != 2 or not torch.isfinite(values).all():
                raise RuntimeError("Extracted representation is not finite [N,D]")
            if extraction_metadata is None:
                extraction_metadata = current
            elif current != extraction_metadata:
                raise RuntimeError("Feature-location metadata changed between batches")
            parts.append(values.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(parts, axis=0), extraction_metadata


def load_feature_set(dataset_context, method, run_info, subject, split_context, args):
    """Load or extract one method's real pre-classifier LOSO representation."""
    dataset = dataset_context["dataset_object"]
    domains = dataset_context["domains"]
    checkpoint = _checkpoint_path(run_info["run_dir"], subject)
    config = _model_config(run_info["record"])
    cache_dir = os.path.join(
        args.feature_cache_dir, dataset_context["spec"]["name"],
        method["model_type"] + ("_" + str(method.get("bnorm")) if method["model_type"] == "tsmnet" else ""),
        "subject_{:02d}".format(int(subject)),
    )
    feature_path = os.path.join(cache_dir, "representations.npz")
    metadata_path = os.path.join(cache_dir, "metadata.csv")
    signature_path = os.path.join(cache_dir, "signature.json")
    signature = {
        "checkpoint": checkpoint,
        "checkpoint_size": int(os.path.getsize(checkpoint)),
        "checkpoint_mtime_ns": int(os.stat(checkpoint).st_mtime_ns),
        "dataset_cache": dataset_context["cache"],
        "subject": int(subject),
        "method": method,
        "model_config": config,
        "split_config": split_context["config"],
        "source_ids_checksum": int(np.sum(split_context["source_ids"], dtype=np.int64)),
        "target_ids_checksum": int(np.sum(split_context["target_ids"], dtype=np.int64)),
    }
    if not args.force_reextract and all(os.path.exists(path) for path in [
        feature_path, metadata_path, signature_path
    ]):
        with open(signature_path, encoding="utf-8") as handle:
            saved_payload = json.load(handle)
        if saved_payload.get("signature") == _jsonable(signature):
            with np.load(feature_path, allow_pickle=False) as saved:
                ids = saved["sample_id"].astype(np.int64)
                features = saved["features"].astype(np.float32)
            metadata = pd.read_csv(metadata_path)
            if np.array_equal(ids, metadata["sample_id"].to_numpy(dtype=np.int64)):
                return features, metadata, saved_payload.get("feature_location", {})

    selected = np.concatenate([
        split_context["source_ids"], split_context["val_ids"],
        split_context["target_ids"],
    ]).astype(np.int64)
    model = _build_model(
        dataset, domains, selected, split_context["source_ids"],
        method, config, args.device_object
    )
    state, migrations = migrate_legacy_spddsbn_buffers(
        _load_state(checkpoint), model.state_dict()
    )
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint architecture mismatch for {} / subject {}. Verify the "
            "master-summary hyperparameters.\n{}".format(method["label"], subject, exc)
        ) from exc
    ids = np.concatenate([
        split_context["source_ids"], split_context["target_ids"]
    ]).astype(np.int64)
    features, location = _extract_vectors(
        model, method, dataset, ids, domains, split_context["normalizer"],
        args.batch_size, args.device_object,
    )
    metadata = _feature_metadata(
        dataset, ids, domains, split_context["source_ids"]
    )
    if len(features) != len(metadata) or metadata["sample_id"].duplicated().any():
        raise RuntimeError("Feature/sample metadata alignment failed")
    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(feature_path, sample_id=ids, features=features)
    metadata.to_csv(metadata_path, index=False)
    _write_json({
        "signature": signature,
        "feature_location": location,
        "checkpoint_buffer_migrations": migrations,
    }, signature_path)
    return features, metadata, location


def _make_split_context(dataset_context, subject, split_config):
    dataset = dataset_context["dataset_object"]
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
            raise RuntimeError(
                "Artifact filtering removed all {} windows for subject {}"
                .format(name, subject)
            )
    if np.intersect1d(filtered["train"], filtered["test"]).size:
        raise RuntimeError("Source-train and target-test sample IDs overlap")
    return {
        "source_ids": np.asarray(filtered["train"], dtype=np.int64),
        "val_ids": np.asarray(filtered["val"], dtype=np.int64),
        "target_ids": np.asarray(filtered["test"], dtype=np.int64),
        "normalizer": normalizer,
        "config": split_config,
    }


def balanced_sample(metadata, max_points_per_group=200, seed=RANDOM_SEED):
    """Return one class-domain-balanced, source-subject-balanced manifest."""
    return balanced_plot_manifest(
        metadata, max_per_class_domain=int(max_points_per_group), seed=int(seed)
    )


def _manifest_positions(metadata, manifest):
    position = {
        int(sample_id): index
        for index, sample_id in enumerate(metadata["sample_id"].to_numpy())
    }
    try:
        indices = np.asarray([
            position[int(sample_id)] for sample_id in manifest["sample_id"]
        ], dtype=np.int64)
    except KeyError as exc:
        raise RuntimeError("A shared plot sample is missing from one method") from exc
    return indices


def _source_standardize(features, metadata):
    source = metadata["domain"].astype(str).to_numpy() == "source"
    if not source.any():
        raise ValueError("Source-only feature standardization has no source samples")
    scaler = StandardScaler().fit(features[source])
    transformed = scaler.transform(features)
    if not np.isfinite(transformed).all():
        raise ValueError("Source-standardized features contain NaN/Inf")
    return transformed, scaler


def _import_umap():
    try:
        from umap import UMAP
        return UMAP
    except ImportError:
        return None


def reduce_to_2d(features, metadata, reducer="auto", pca_dim=30,
                 seed=RANDOM_SEED, umap_neighbors=15, umap_min_dist=0.15,
                 tsne_perplexity=30.0, tsne_max_iter=2000):
    """Reduce balanced features with source-fitted PCA and UMAP/one joint t-SNE."""
    features = np.asarray(features, dtype=np.float64)
    source = metadata["domain"].astype(str).to_numpy() == "source"
    if features.ndim != 2 or len(features) != len(metadata):
        raise ValueError("Features and metadata must align as [N,D]")
    transformed = features
    pca_metadata = {"applied": False, "input_dimension": int(features.shape[1])}
    maximum = min(int(pca_dim), int(source.sum()) - 1, features.shape[1])
    if features.shape[1] > maximum and maximum >= 2:
        pca = PCA(n_components=maximum, random_state=int(seed))
        pca.fit(features[source])
        transformed = pca.transform(features)
        pca_metadata = {
            "applied": True,
            "input_dimension": int(features.shape[1]),
            "output_dimension": int(maximum),
            "fit_partition": "balanced_source_only",
            "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        }

    UMAP = _import_umap()
    selected = reducer
    if reducer == "auto":
        selected = "umap" if UMAP is not None else "tsne"
    if selected == "umap" and UMAP is None:
        if reducer == "umap":
            raise ImportError("--reducer umap requires `pip install umap-learn`")
        selected = "tsne"
    if selected == "umap":
        neighbors = min(int(umap_neighbors), int(source.sum()) - 1)
        if neighbors < 2:
            raise ValueError("UMAP needs at least three balanced source samples")
        model = UMAP(
            n_components=2, n_neighbors=neighbors,
            min_dist=float(umap_min_dist), metric="euclidean",
            random_state=int(seed), transform_seed=int(seed), n_jobs=1,
        )
        model.fit(transformed[source])
        coordinates = model.transform(transformed)
        reduction = {
            "method": "UMAP", "fit_partition": "balanced_source_only",
            "target_used_to_fit": False, "n_neighbors": int(neighbors),
            "min_dist": float(umap_min_dist), "metric": "euclidean",
            "random_state": int(seed), "transform_seed": int(seed),
        }
    else:
        count = len(transformed)
        if count < 4:
            raise ValueError("t-SNE requires at least four balanced samples")
        perplexity = min(float(tsne_perplexity), (count - 1.0) / 3.0, count - 1e-6)
        version_parts = tuple(
            int(part) for part in sklearn.__version__.split(".")[:2]
            if part.isdigit()
        )
        learning_rate = (
            "auto" if version_parts >= (1, 2)
            else max(float(count) / 12.0, 50.0)
        )
        kwargs = {
            "n_components": 2, "init": "pca", "learning_rate": learning_rate,
            "perplexity": float(perplexity), "random_state": int(seed),
        }
        signature = inspect.signature(TSNE).parameters
        if "max_iter" in signature:
            kwargs["max_iter"] = int(tsne_max_iter)
        else:
            kwargs["n_iter"] = int(tsne_max_iter)
        coordinates = TSNE(**kwargs).fit_transform(transformed)
        reduction = {
            "method": "t-SNE", "fit_partition": "balanced_source_and_target_joint",
            "target_used_to_fit": True, "target_labels_used_to_fit": False,
            "perplexity": float(perplexity), "init": "pca",
            "learning_rate": kwargs["learning_rate"],
            "max_iter": int(tsne_max_iter), "random_state": int(seed),
        }
        if reducer == "auto" and UMAP is None:
            warnings.warn(
                "umap-learn is unavailable; using one joint unlabeled t-SNE fit. "
                "Install umap-learn for source-only out-of-sample transformation.",
                UserWarning,
            )
    return np.asarray(coordinates, dtype=np.float64), {
        "pca": pca_metadata, "reduction": reduction,
    }


def _rms_distance(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    return float(np.linalg.norm(left - right) / math.sqrt(left.size))


def high_dimensional_metrics(features, metadata, epsilon=1e-12):
    """Class-conditional domain discrepancy and class/domain ratio."""
    values = np.asarray(features, dtype=np.float64)
    domains = metadata["domain"].astype(str).to_numpy()
    labels = metadata["class_id"].to_numpy(dtype=np.int64)
    source, target = domains == "source", domains == "target"
    class_ids = sorted(int(value) for value in np.unique(labels))
    domain_distances, pooled_centers, within = [], [], []
    for class_id in class_ids:
        source_class = source & (labels == class_id)
        target_class = target & (labels == class_id)
        if not source_class.any() or not target_class.any():
            raise ValueError("Class {} is missing in source or target".format(class_id))
        source_center = values[source_class].mean(axis=0)
        target_center = values[target_class].mean(axis=0)
        domain_distances.append(_rms_distance(source_center, target_center))
        center = 0.5 * (source_center + target_center)
        pooled_centers.append(center)
        class_values = values[labels == class_id]
        within.append(float(np.mean(
            np.sum((class_values - center) ** 2, axis=1) / values.shape[1]
        )))
    class_distances = [
        _rms_distance(pooled_centers[left], pooled_centers[right])
        for left in range(len(pooled_centers))
        for right in range(left + 1, len(pooled_centers))
    ]
    discrepancy = float(np.mean(domain_distances))
    separation = float(np.mean(class_distances))
    return {
        "domain_discrepancy": discrepancy,
        "class_separation": separation,
        "separation_ratio": float(separation / (discrepancy + epsilon)),
        "fisher_ratio": float(separation ** 2 / (np.mean(within) + epsilon)),
        "feature_dimension": int(values.shape[1]),
        "class_conditional_domain_distances": domain_distances,
    }


def _class_color(class_id, class_count):
    if class_count == 2:
        index = 0 if int(class_id) == 0 else 2
    else:
        index = int(class_id)
    return CLASS_COLORS[index % len(CLASS_COLORS)]


def plot_embedding_panel(ax, coordinates, metadata, panel_label,
                         method_title=None, dataset_label=None):
    """Draw one compact, print-safe domain/class embedding panel."""
    labels = sorted(int(value) for value in metadata["class_id"].unique())
    for class_id in labels:
        color = _class_color(class_id, len(labels))
        for domain, marker, size, alpha, zorder in [
            ("source", "o", 8.0, 0.64, 2),
            ("target", "^", 18.0, 0.84, 4),
        ]:
            mask = (
                (metadata["class_id"].to_numpy(dtype=np.int64) == class_id)
                & (metadata["domain"].astype(str).to_numpy() == domain)
            )
            ax.scatter(
                coordinates[mask, 0], coordinates[mask, 1], s=size,
                c=color, marker=marker, alpha=alpha,
                edgecolors="white" if domain == "source" else "#303030",
                linewidths=0.18 if domain == "source" else 0.35,
                rasterized=True, zorder=zorder,
            )
    if method_title:
        ax.set_title(method_title, fontsize=7.5, pad=4.0, fontweight="semibold")
    if dataset_label:
        ax.set_ylabel(dataset_label, fontsize=7.5, fontweight="semibold", labelpad=5.0)
    ax.text(
        0.025, 0.97, panel_label, transform=ax.transAxes,
        ha="left", va="top", fontsize=6.6, fontweight="semibold",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.55)
        spine.set_color("#707070")
    ax.set_facecolor("white")
    ax.margins(0.08)


def _metric_aggregate(metric_rows, metric):
    frame = pd.DataFrame(metric_rows)
    rows = []
    for (dataset, method), group in frame.groupby(["dataset", "method"], sort=False):
        values = group[metric].to_numpy(dtype=np.float64)
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        ci95 = float(1.96 * std / math.sqrt(len(values))) if len(values) > 1 else 0.0
        rows.append({
            "dataset": dataset, "method": method, "metric": metric,
            "mean": mean, "std": std, "ci95": ci95, "n_folds": int(len(values)),
        })
    return pd.DataFrame(rows)


def plot_metric_panel(ax, aggregate, methods, datasets, metric, title,
                      higher_is_better):
    """Draw fold-aggregated high-dimensional metrics with 95% CIs."""
    x = np.arange(len(methods), dtype=np.float64)
    for index, dataset in enumerate(datasets):
        color, marker, linestyle = DATASET_STYLES[index % len(DATASET_STYLES)]
        rows = aggregate[
            (aggregate["dataset"] == dataset["display_name"])
            & (aggregate["metric"] == metric)
        ].set_index("method")
        means = np.asarray([rows.loc[item["label"], "mean"] for item in methods])
        errors = np.asarray([rows.loc[item["label"], "ci95"] for item in methods])
        ax.errorbar(
            x, means, yerr=errors, color=color, marker=marker,
            linestyle=linestyle, linewidth=1.0, markersize=3.6,
            markeredgecolor="white", markeredgewidth=0.35,
            capsize=2.0, elinewidth=0.7, label=dataset["display_name"],
            zorder=3,
        )
        best = int(np.argmax(means) if higher_is_better else np.argmin(means))
        ax.annotate(
            "{:.2f}".format(means[best]), (x[best], means[best]),
            xytext=(0, 5), textcoords="offset points", ha="center",
            fontsize=5.8, color=color,
        )
    ax.set_title(title, fontsize=7.3, loc="left", pad=4.0, fontweight="semibold")
    ax.set_xticks(x)
    ax.set_xticklabels([item["short_label"] for item in methods], fontsize=5.4)
    ax.set_xlim(-0.25, len(methods) - 0.75)
    ax.set_ylim(bottom=0.0)
    ax.tick_params(axis="y", labelsize=5.8, width=0.55, length=2.5)
    ax.tick_params(axis="x", width=0.55, length=0)
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.35, alpha=0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)


def _save_figure(fig, output_dir):
    stem = os.path.join(output_dir, "fig5_representation_alignment")
    metadata = {
        "Title": "Cross-Subject Representation Alignment",
        "Author": "Generated from real LOSO checkpoints",
        "Subject": "IEEE Transactions Fig. 5",
        "Keywords": "EEG, domain alignment, SPDDSBN, UMAP",
    }
    fig.savefig(stem + ".pdf", dpi=600, bbox_inches="tight", metadata=metadata)
    fig.savefig(stem + ".png", dpi=600, bbox_inches="tight",
                metadata={"Description": metadata["Subject"]})
    fig.savefig(stem + ".svg", dpi=600, bbox_inches="tight", metadata={
        "Title": metadata["Title"],
        "Description": metadata["Subject"],
        "Creator": metadata["Author"],
        "Keywords": metadata["Keywords"].split(", "),
    })
    return [os.path.abspath(stem + suffix) for suffix in [".pdf", ".png", ".svg"]]


def _load_dataset_context(spec, args):
    fs = _target_fs(spec, args)
    cache = default_cache_path(
        spec["dataset"], "loso", cog_paradigm=spec["paradigm"],
        target_fs=fs, cache_root=args.cache_root,
    )
    sessions = (1, 2, 3) if spec["dataset"] == "cog-bci" else (1,)
    dataset = load_dataset(
        spec["dataset"], data_root=args.data_root, cache=cache,
        rebuild_cache=False, cog_paradigm=spec["paradigm"],
        sessions=sessions, target_fs=fs, window_sec=1.0, stride_sec=1.0,
    )
    return {
        "spec": spec, "dataset_object": dataset,
        "domains": domain_ids(dataset, "loso"),
        "cache": os.path.abspath(cache),
    }


def _common_subjects(run_infos):
    subjects = None
    for info in run_infos:
        values = set(pd.to_numeric(
            info["summary"]["subject"], errors="coerce"
        ).dropna().astype(int))
        subjects = values if subjects is None else subjects & values
    if not subjects:
        raise ValueError("The four methods have no common completed LOSO subjects")
    return sorted(subjects)


def _select_subject(dataset, full_run, overrides):
    if dataset["name"] in overrides:
        subject = int(overrides[dataset["name"]])
        evidence = {"selection_rule": "explicit", "selected_target_subject": subject}
    elif dataset["dataset"] in overrides:
        subject = int(overrides[dataset["dataset"]])
        evidence = {"selection_rule": "explicit", "selected_target_subject": subject}
    else:
        evidence = choose_median_fold(full_run["summary"])
        subject = int(evidence["selected_target_subject"])
    return subject, evidence


def main():
    args = parse_args()
    if args.batch_size < 1 or args.max_points_per_group < 1:
        raise ValueError("Batch size and max points must be positive")
    if args.pca_dim < 2 or args.umap_neighbors < 2:
        raise ValueError("PCA dimension and UMAP neighbors must be at least 2")
    datasets = _parse_datasets(args.datasets, args.dataset_labels)
    methods = _load_methods(args.method_manifest)
    overrides = _target_subject_overrides(args.target_subjects)
    if not os.path.exists(args.master_summary):
        if not args.allow_missing_master_config:
            raise FileNotFoundError(
                "Master summary not found: {}. It is required to reconstruct "
                "the exact training architecture.".format(args.master_summary)
            )
        master = pd.DataFrame()
    else:
        master = pd.read_csv(args.master_summary)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    args.device_object = torch.device(device_name)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.feature_cache_dir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    matplotlib.rcParams.update({
        "font.family": _font_family(), "font.size": 7.0,
        "axes.labelsize": 6.5, "axes.titlesize": 7.5,
        "axes.linewidth": 0.6, "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.facecolor": "white",
        "svg.hashsalt": "fig5-representation-alignment-2026",
    })

    embedding_records = []
    metric_records = []
    manifest_records = []
    provenance = {"datasets": {}, "methods": methods}
    panel_data = {}

    for dataset_index, dataset_spec in enumerate(datasets):
        print("Preparing dataset:", dataset_spec["display_name"])
        context = _load_dataset_context(dataset_spec, args)
        run_infos = [
            _resolve_run(dataset_spec, method, args, master) for method in methods
        ]
        split_config = _validate_comparable_runs(run_infos, methods, args)
        common_subjects = _common_subjects(run_infos)
        representative, selection = _select_subject(
            dataset_spec, run_infos[-1], overrides
        )
        if representative not in common_subjects:
            raise ValueError(
                "Representative subject {} is not complete for all four methods"
                .format(representative)
            )
        metric_subjects = common_subjects if args.metric_scope == "all" else [representative]
        provenance["datasets"][dataset_spec["name"]] = {
            "display_name": dataset_spec["display_name"],
            "cache": context["cache"], "split_config": split_config,
            "representative_fold": selection,
            "common_loso_subjects": common_subjects,
            "metric_subjects": metric_subjects,
            "runs": [
                {"method": method["label"], "run_dir": info["run_dir"],
                 "summary": info["summary_path"], "record": info["record"],
                 "refit_scope_audit": info.get("refit_scope_audit")}
                for method, info in zip(methods, run_infos)
            ],
        }

        representative_features = {}
        representative_metadata = None
        for subject in metric_subjects:
            split_context = _make_split_context(context, subject, split_config)
            subject_sets = []
            for method, run_info in zip(methods, run_infos):
                features, metadata, location = load_feature_set(
                    context, method, run_info, subject, split_context, args
                )
                subject_sets.append((features, metadata, location))
            base_ids = subject_sets[0][1]["sample_id"].to_numpy(dtype=np.int64)
            for _, metadata, _ in subject_sets[1:]:
                if not np.array_equal(
                    base_ids, metadata["sample_id"].to_numpy(dtype=np.int64)
                ):
                    raise RuntimeError("Methods do not share identical LOSO sample IDs")
            metric_manifest = balanced_sample(
                subject_sets[0][1], args.max_points_per_group,
                seed=args.seed + int(subject),
            )
            for method_index, (method, item) in enumerate(zip(methods, subject_sets)):
                features, metadata, location = item
                standardized, _ = _source_standardize(features, metadata)
                positions = _manifest_positions(metadata, metric_manifest)
                values = standardized[positions]
                balanced_metadata = metadata.iloc[positions].reset_index(drop=True)
                metrics = high_dimensional_metrics(values, balanced_metadata)
                metric_records.append({
                    "dataset": dataset_spec["display_name"],
                    "dataset_id": dataset_spec["name"],
                    "method": method["label"],
                    "model_type": method["model_type"],
                    "target_subject": int(subject),
                    **metrics,
                })
                if subject == representative:
                    representative_features[method_index] = values
                    panel_data[(dataset_index, method_index)] = {
                        "features": values,
                        "metadata": balanced_metadata,
                        "location": location,
                    }
                    representative_metadata = balanced_metadata
            if subject == representative:
                saved_manifest = metric_manifest.copy()
                saved_manifest.insert(0, "dataset", dataset_spec["display_name"])
                saved_manifest.insert(1, "target_subject", int(subject))
                manifest_records.append(saved_manifest)

        if representative_metadata is None:
            raise RuntimeError("Representative fold was not extracted")
        for method_index, method in enumerate(methods):
            item = panel_data[(dataset_index, method_index)]
            coordinates, reduction_metadata = reduce_to_2d(
                item["features"], item["metadata"], reducer=args.reducer,
                pca_dim=args.pca_dim, seed=args.seed,
                umap_neighbors=args.umap_neighbors,
                umap_min_dist=args.umap_min_dist,
                tsne_perplexity=args.tsne_perplexity,
                tsne_max_iter=args.tsne_max_iter,
            )
            item["coordinates"] = coordinates
            item["reduction"] = reduction_metadata
            frame = item["metadata"].copy()
            frame.insert(0, "dataset", dataset_spec["display_name"])
            frame.insert(1, "method", method["label"])
            frame["embedding_x"] = coordinates[:, 0]
            frame["embedding_y"] = coordinates[:, 1]
            embedding_records.append(frame)

    metric_frame = pd.DataFrame(metric_records)
    discrepancy = _metric_aggregate(metric_records, "domain_discrepancy")
    ratio = _metric_aggregate(metric_records, "separation_ratio")
    aggregate = pd.concat([discrepancy, ratio], ignore_index=True)
    metric_frame.to_csv(
        os.path.join(args.output_dir, "fig5_alignment_metrics_by_fold.csv"), index=False
    )
    aggregate.to_csv(
        os.path.join(args.output_dir, "fig5_alignment_metrics_aggregate.csv"), index=False
    )
    pd.concat(embedding_records, ignore_index=True).to_csv(
        os.path.join(args.output_dir, "fig5_embedding_coordinates.csv"), index=False
    )
    pd.concat(manifest_records, ignore_index=True).to_csv(
        os.path.join(args.output_dir, "fig5_plot_sample_manifest.csv"), index=False
    )

    single_dataset = len(datasets) == 1
    figure_size = (FIGURE_SIZE[0], 2.85) if single_dataset else FIGURE_SIZE
    fig = plt.figure(figsize=figure_size)
    outer = fig.add_gridspec(
        len(datasets), 5, width_ratios=[1, 1, 1, 1, 1.38],
        left=0.055, right=0.995,
        bottom=0.16 if single_dataset else 0.105,
        top=0.78 if single_dataset else 0.84,
        wspace=0.18, hspace=0.18,
    )
    for dataset_index, dataset in enumerate(datasets):
        for method_index, method in enumerate(methods):
            ax = fig.add_subplot(outer[dataset_index, method_index])
            item = panel_data[(dataset_index, method_index)]
            panel_number = dataset_index * len(methods) + method_index + 1
            plot_embedding_panel(
                ax, item["coordinates"], item["metadata"],
                "(a{})".format(panel_number),
                method_title=method["label"] if dataset_index == 0 else None,
                dataset_label=dataset["display_name"] if method_index == 0 else None,
            )
    metric_grid = outer[:, 4].subgridspec(2, 1, hspace=0.48)
    domain_ax = fig.add_subplot(metric_grid[0, 0])
    ratio_ax = fig.add_subplot(metric_grid[1, 0])
    plot_metric_panel(
        domain_ax, aggregate, methods, datasets, "domain_discrepancy",
        r"(b) Domain discrepancy $\downarrow$", False,
    )
    plot_metric_panel(
        ratio_ax, aggregate, methods, datasets, "separation_ratio",
        r"(c) Class/domain separation $\uparrow$", True,
    )
    if len(datasets) > 1:
        domain_ax.legend(
            loc="upper right", frameon=False, fontsize=5.8,
            handlelength=1.5, borderpad=0.1, labelspacing=0.25,
        )

    class_count = max(
        panel_data[key]["metadata"]["class_id"].nunique() for key in panel_data
    )
    class_names = ["Low", "Medium", "High"] if class_count == 3 else ["Low", "High"]
    class_handles = [
        Line2D([0], [0], marker="o", linestyle="None", markersize=4.2,
               markerfacecolor=_class_color(index, class_count), markeredgecolor="none",
               label=name)
        for index, name in enumerate(class_names)
    ]
    domain_handles = [
        Line2D([0], [0], marker="o", linestyle="None", markersize=4.2,
               markerfacecolor="#777777", markeredgecolor="white", label="Source"),
        Line2D([0], [0], marker="^", linestyle="None", markersize=4.8,
               markerfacecolor="#777777", markeredgecolor="#303030", label="Target"),
    ]
    legend1 = fig.legend(
        class_handles, [handle.get_label() for handle in class_handles],
        title="Class", loc="upper center", bbox_to_anchor=(0.34, 0.985),
        ncol=len(class_handles), frameon=False, fontsize=6.4,
        title_fontsize=6.4, handletextpad=0.35, columnspacing=0.9,
    )
    fig.add_artist(legend1)
    fig.legend(
        domain_handles, [handle.get_label() for handle in domain_handles],
        title="Domain", loc="upper center", bbox_to_anchor=(0.66, 0.985),
        ncol=2, frameon=False, fontsize=6.4, title_fontsize=6.4,
        handletextpad=0.35, columnspacing=0.9,
    )
    fig.text(
        0.47, 0.035,
        "Embeddings are descriptive; quantitative panels use source-standardized high-dimensional features.",
        ha="center", va="center", fontsize=5.7, color="#555555",
    )
    paths = _save_figure(fig, args.output_dir)
    plt.close(fig)

    provenance.update({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv), "random_seed": int(args.seed),
        "metric_scope": args.metric_scope,
        "metric_definition": {
            "domain_discrepancy": (
                "mean class-conditional source-target centroid RMS Euclidean "
                "distance in source-standardized high-dimensional features"
            ),
            "separation_ratio": (
                "mean pairwise pooled class-centroid RMS distance divided by "
                "class-conditional domain discrepancy"
            ),
            "error_bars": "95% normal-approximation CI across target-subject LOSO folds",
        },
        "balanced_sampling": {
            "max_points_per_class_domain": int(args.max_points_per_group),
            "seed_rule": "global_seed + target_subject",
            "same_sample_ids_across_methods": True,
        },
        "target_label_usage": (
            "offline class-domain balancing, color encoding, and "
            "class-conditional metrics only; never model fitting, source "
            "normalization, PCA fitting, or UMAP fitting"
        ),
        "embedding_guardrail": (
            "No trend is enforced. Panels are independently reduced with "
            "identical hyperparameters; formal evidence is high-dimensional."
        ),
        "panel_reductions": {
            "{}::{}".format(datasets[i]["name"], methods[j]["label"]):
                panel_data[(i, j)]["reduction"]
            for i in range(len(datasets)) for j in range(len(methods))
        },
        "outputs": paths,
        "versions": {
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "torch": torch.__version__,
            "sklearn": sklearn.__version__, "matplotlib": matplotlib.__version__,
        },
    })
    _write_json(provenance, os.path.join(
        args.output_dir, "fig5_representation_alignment_metadata.json"
    ))
    print("Representative subjects:", {
        name: value["representative_fold"]["selected_target_subject"]
        for name, value in provenance["datasets"].items()
    })
    print("Metric folds:", {
        name: len(value["metric_subjects"])
        for name, value in provenance["datasets"].items()
    })
    print("Saved:")
    for path in paths:
        print(" ", path)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit("Fig. 5 input error: {}".format(exc))
