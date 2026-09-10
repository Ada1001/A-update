"""Offline fixed-weight diagnostics; never overwrites a training checkpoint."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from analysis import fig5_representation_alignment as fig5
from src.cl_tsmnet.spd_pca import migrate_legacy_spddsbn_buffers
from src.cl_tsmnet.spd_visualization_adapters import extract_spd_intermediates


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def extract(model, dataset, domains, ids, normalizer, method, batch_size, device):
    """Capture both feature locations and logits from the SAME eval forward."""
    parts = {"representation": [], "classifier_input": [], "logits": []}
    hooks = []
    def capture(name):
        def hook(module, inputs):
            parts[name].append(inputs[0].detach().cpu().numpy().copy())
        return hook
    if method == "tsmnet":
        hooks.append(model.classifier.register_forward_pre_hook(capture("representation")))
    else:
        hooks.append(model.readout.register_forward_pre_hook(capture("representation")))
    hooks.append(model.classifier.register_forward_pre_hook(capture("classifier_input")))
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(ids), batch_size):
                ix = ids[start:start + batch_size]
                x = torch.from_numpy(normalizer.transform_array(dataset["x"][ix])).to(device)
                d = torch.from_numpy(domains[ix]).long().to(device)
                output = model(x, d)
                logits = output[0] if isinstance(output, tuple) else output
                parts["logits"].append(logits.detach().cpu().numpy())
    finally:
        for hook in hooks:
            hook.remove()
    result = {key: np.concatenate(value) for key, value in parts.items()}
    if any(not np.isfinite(value).all() for value in result.values()):
        raise ValueError("Non-finite diagnostic features/logits")
    return result


def refit_source(model, dataset, domains, ids, normalizer, method, batch_size, device):
    """Refit one source domain at a time; extract compact features in batches."""
    model.eval()
    if not hasattr(model, "spddsbnorm") and getattr(model, "eudsbnorm", None) is None:
        return "not_applicable_no_domain_bn"
    with torch.no_grad():
        for domain in np.unique(domains[ids]):
            domain_ids = ids[domains[ids] == domain]
            chunks = []
            for start in range(0, len(domain_ids), batch_size):
                ix = domain_ids[start:start + batch_size]
                x = torch.from_numpy(normalizer.transform_array(dataset["x"][ix])).to(device)
                d = torch.from_numpy(domains[ix]).long().to(device)
                if method == "tsmnet":
                    _, intermediate = extract_spd_intermediates(model, x, d, "tsmnet")
                    features = intermediate["spd_pre_bn"]
                else:
                    maps = model._weighted_graph_maps(x)
                    features = (model.spd_branch.manifold_features(maps)
                                if model.spd_branch is not None else
                                model._first_order_readout(maps, d, apply_dsbn=False))
                chunks.append(features.detach().cpu())
            features = torch.cat(chunks)
            d = torch.full((len(features),), int(domain), dtype=torch.long)
            if method == "tsmnet":
                import spdnets.batchnorm as bn
                model.spddsbnorm.set_test_stats_mode(bn.BatchNormTestStatsMode.REFIT)
                try:
                    model.spddsbnorm(features.to(model.spd_device_), d.to(model.spd_device_))
                finally:
                    model.spddsbnorm.set_test_stats_mode(bn.BatchNormTestStatsMode.BUFFER)
            elif model.spd_branch is not None:
                model.spd_branch.refit_domain_features(features, d)
            else:
                model.eudsbnorm.refit_domain_stats(features.to(device), d.to(device))
    return "source_train_domains_only"


def audit_state(before, model, source_domains):
    allowed = set()
    for key, _ in model.named_buffers(remove_duplicate=False):
        if any((".batchnorm.dom {}.".format(int(d)) in key or
                key.startswith("eudsbnorm.layers.{}.".format(int(d)))) for d in source_domains):
            allowed.add(key)
    after = model.state_dict()
    if before.keys() != after.keys():
        raise AssertionError("Refit changed model state keys")
    changed = [key for key in before if not torch.equal(before[key], after[key].detach().cpu())]
    illegal = sorted(set(changed) - allowed)
    if illegal:
        raise AssertionError("Refit changed non-source buffers or weights: {}".format(illegal))
    return changed


def class_metrics(features, labels):
    centers, within = [], []
    for label in np.unique(labels):
        values = features[labels == label]
        center = values.mean(0)
        centers.append(center)
        within.append(np.mean(np.sum((values - center) ** 2, axis=1) / features.shape[1]))
    if len(centers) < 2:
        return {"class_distance": np.nan, "within_variance": np.nan, "fisher_ratio": np.nan}
    distance = np.mean([fig5._rms_distance(a, b) for i, a in enumerate(centers) for b in centers[i+1:]])
    return {"class_distance": float(distance), "within_variance": float(np.mean(within)),
            "fisher_ratio": float(distance ** 2 / (np.mean(within) + 1e-12))}


def diagnose_fold(context, info, method, subject, args):
    out = Path(args.output_dir) / method["model_type"] / ("subject_%02d" % subject)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "audit.json").exists():
        raise FileExistsError("Use a new output directory; diagnostic already exists: " + str(out))
    dataset, domains = context["dataset_object"], context["domains"]
    split = fig5._make_split_context(context, subject, fig5._split_config(info))
    source, target = split["source_ids"], split["target_ids"]
    if np.intersect1d(domains[source], domains[target]).size:
        raise ValueError("Diagnostic requires disjoint source and target domains")
    ids = np.concatenate([source, target])
    selected = np.concatenate([ids, split["val_ids"]])
    checkpoint = fig5._checkpoint_path(info["run_dir"], subject)
    sha = digest(checkpoint)
    model = fig5._build_model(dataset, domains, selected, source, method,
                             fig5._model_config(info["record"]), args.device_object)
    state, migrations = migrate_legacy_spddsbn_buffers(fig5._load_state(checkpoint), model.state_dict())
    model.load_state_dict(state, strict=True)
    before_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    baseline = extract(model, dataset, domains, ids, split["normalizer"], method["model_type"],
                       args.batch_size, args.device_object)
    copy_model = copy.deepcopy(model)
    del model
    status = refit_source(copy_model, dataset, domains, source, split["normalizer"],
                          method["model_type"], args.batch_size, args.device_object)
    changed = audit_state(before_state, copy_model, np.unique(domains[source]))
    after = extract(copy_model, dataset, domains, ids, split["normalizer"], method["model_type"],
                    args.batch_size, args.device_object)
    target_mask = np.arange(len(ids)) >= len(source)
    delta = float(np.max(np.abs(baseline["logits"][target_mask] - after["logits"][target_mask])))
    identical_predictions = bool(np.array_equal(baseline["logits"][target_mask].argmax(1),
                                                after["logits"][target_mask].argmax(1)))
    audit = {"checkpoint": checkpoint, "checkpoint_sha256": sha, "method": method,
             "subject": subject, "split_config": split["config"], "status": status,
             "changed_source_buffer_keys": changed, "weights_and_non_source_buffers_unchanged": True,
             "target_logit_max_abs_delta": delta, "target_predictions_identical": identical_predictions,
             "target_logits_allclose": bool(np.allclose(baseline["logits"][target_mask],
                  after["logits"][target_mask], rtol=1e-6, atol=1e-6)), "migrations": migrations,
             "checkpoint_unchanged": sha == digest(checkpoint), "seed": args.seed,
             "scale_policy": "saved-buffer source-only scaler reused for both conditions",
             "classifier_input_note": "TSMNet equals tangent; MSTGC after readout GELU, eval dropout",
             "data_cache": context["cache"], "run_record": info["record"]}
    fig5._write_json(audit, str(out / "audit.json"))
    if not (audit["target_logits_allclose"] and identical_predictions and audit["checkpoint_unchanged"]):
        raise AssertionError("Invariant failed; inspect " + str(out / "audit.json"))
    metadata = fig5._feature_metadata(dataset, ids, domains, source)
    metadata.to_csv(out / "samples.csv", index=False)
    manifest = fig5.balanced_sample(metadata, args.max_points_per_group, args.seed + subject)
    positions = fig5._manifest_positions(metadata, manifest)
    manifest.to_csv(out / "plot_manifest.csv", index=False)
    scores, per_subject, alignment, separation = [], [], [], []
    archive = {"sample_id": ids, "plot_positions": positions}
    for condition, values in [("saved", baseline), ("source_refit", after)]:
        prediction = values["logits"].argmax(1)
        y = dataset["y"][ids]
        prediction_frame = metadata.copy()
        prediction_frame["prediction"] = prediction
        for k in range(values["logits"].shape[1]):
            prediction_frame["logit_%d" % k] = values["logits"][:, k]
        prediction_frame.to_csv(out / (condition + "_predictions.csv"), index=False)
        for partition in ["source", "target"]:
            mask = metadata.domain.to_numpy() == partition
            scores.append({"condition": condition, "partition": partition, "n": int(mask.sum()),
                           "accuracy": accuracy_score(y[mask], prediction[mask]),
                           "bacc": balanced_accuracy_score(y[mask], prediction[mask]),
                           "macro_f1": f1_score(y[mask], prediction[mask], average="macro", zero_division=0)})
        for location in ["representation", "classifier_input"]:
            scaler = StandardScaler().fit(baseline[location][~target_mask])
            features = scaler.transform(values[location])
            archive[condition + "_" + location] = features.astype(np.float32)
            alignment.append({"condition": condition, "location": location,
                **fig5.high_dimensional_metrics(features[positions], metadata.iloc[positions])})
            for partition in ["source", "target"]:
                mask = metadata.domain.to_numpy() == partition
                separation.append({"condition": condition, "location": location, "partition": partition,
                                   **class_metrics(features[mask], y[mask])})
            for who in metadata.subject_id.unique():
                mask = metadata.subject_id.to_numpy() == who
                per_subject.append({"condition": condition, "location": location, "subject": int(who),
                    "partition": metadata.loc[mask, "domain"].iloc[0],
                    "bacc": balanced_accuracy_score(y[mask], prediction[mask]),
                    **class_metrics(features[mask], y[mask])})
    for name, rows in [("classification", scores), ("alignment", alignment),
                       ("class_separation", separation), ("by_subject", per_subject)]:
        pd.DataFrame(rows).to_csv(out / (name + ".csv"), index=False)
    np.savez_compressed(out / "features.npz", **archive)
    print(pd.DataFrame(scores).to_string(index=False), flush=True)
    print("Target invariants passed; saved", out, flush=True)


def embed_directory(directory, reducers):
    """Use previously extracted fixed features/samples; never loads a model."""
    files = sorted(Path(directory).rglob("features.npz"))
    if not files:
        raise FileNotFoundError("No diagnostic features.npz under " + str(directory))
    for file in files:
        with np.load(file) as data:
            metadata = pd.read_csv(file.parent / "samples.csv")
            positions = data["plot_positions"]
            rows = metadata.iloc[positions].reset_index(drop=True)
            audit = json.loads((file.parent / "audit.json").read_text(encoding="utf-8"))
            for location in ["representation", "classifier_input"]:
                for reducer in reducers:
                    name = location + "_" + reducer
                    output = file.parent / (name + ".png")
                    if output.exists():
                        raise FileExistsError("Embedding exists: " + str(output))
                    fig, axes = fig5.plt.subplots(2, 2, figsize=(9, 8))
                    coordinates, details = [], {}
                    for col, condition in enumerate(["saved", "source_refit"]):
                        values = data[condition + "_" + location][positions]
                        mode = "umap" if reducer.startswith("umap") else "tsne"
                        pca_dim = 30 if reducer == "umap_pca30" else values.shape[1]
                        xy, detail = fig5.reduce_to_2d(values, rows, reducer=mode, pca_dim=pca_dim,
                                                      seed=audit["seed"])
                        details[condition] = detail
                        fig5.plot_embedding_panel(axes[0, col], xy, rows, condition,
                                                  method_title=condition)
                        for domain, marker in [("source", "o"), ("target", "^")]:
                            mask = rows.domain.to_numpy() == domain
                            axes[1, col].scatter(xy[mask, 0], xy[mask, 1],
                                c=rows.subject_id.to_numpy()[mask], cmap="turbo", vmin=0,
                                vmax=metadata.subject_id.max(), marker=marker, s=14)
                        axes[1, col].set_title("Color = subject; triangle = target")
                        table = rows.copy()
                        table["condition"], table["x"], table["y"] = condition, xy[:, 0], xy[:, 1]
                        coordinates.append(table)
                    fig.suptitle(name + " | top: blue Low / green High; independent projections")
                    fig.tight_layout()
                    fig.savefig(output, dpi=180)
                    fig5.plt.close(fig)
                    pd.concat(coordinates).to_csv(file.parent / (name + ".csv"), index=False)
                    fig5._write_json(details, str(file.parent / (name + ".json")))
                    print("Saved", output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["extract", "embed"], default="extract")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--datasets", default="stew")
    parser.add_argument("--dataset-labels", default="STEW")
    parser.add_argument("--models", default="ms_tgc_spddsbn")
    parser.add_argument("--subjects", default="21", help="Comma-separated subjects, or all")
    parser.add_argument("--output-root", default="outputs/fig5_stew_v3")
    parser.add_argument("--master-summary", default="outputs/fig5_stew_v3/master_summary.csv")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--cache-root", default="outputs/cache")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-points-per-group", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--reducers", default="umap_pca30,umap_no_pca,tsne")
    args = parser.parse_args()
    if args.stage == "embed":
        reducers = args.reducers.split(",")
        if not set(reducers) <= {"umap_pca30", "umap_no_pca", "tsne"}:
            raise ValueError("Unknown reducer")
        embed_directory(args.output_dir, reducers)
        return
    if args.batch_size < 1 or args.max_points_per_group < 1:
        raise ValueError("Batch size and sample cap must be positive")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.device_object = torch.device(args.device)
    args.target_fs_stew = args.target_fs_eegmat = args.target_fs_cog_bci = None
    args.allow_missing_master_config = args.allow_legacy_refit = False
    specs = fig5._parse_datasets(args.datasets, args.dataset_labels)
    if len(specs) != 1:
        raise ValueError("Run one dataset per diagnostic output directory")
    methods = {m["model_type"]: m for m in fig5._load_methods(None)}
    methods["tsmnet"] = fig5._load_methods(None, fourth_model="tsmnet")[-1]
    requested = args.models.split(",")
    if not set(requested) <= set(methods):
        raise ValueError("Supported models: " + ",".join(methods))
    master = pd.read_csv(args.master_summary)
    context = fig5._load_dataset_context(specs[0], args)
    for name in requested:
        method = methods[name]
        info = fig5._resolve_run(specs[0], method, args, master)
        fig5._validate_comparable_runs([info], [method], args)
        completed = set(info["summary"].subject.astype(int))
        subjects = sorted(completed) if args.subjects == "all" else list(map(int, args.subjects.split(",")))
        if not set(subjects) <= completed:
            raise ValueError("Requested subject absent from summary")
        for subject in subjects:
            print("Diagnosing", name, subject, flush=True)
            diagnose_fold(context, info, method, subject, args)


if __name__ == "__main__":
    main()
