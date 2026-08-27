"""Leakage-safe common-tangent-space utilities for SPDDSBN visualization."""

import math
import os
import sys

import numpy as np
import pandas as pd
import torch


def _symmetrize(matrices):
    matrices = np.asarray(matrices, dtype=np.float64)
    return 0.5 * (matrices + np.swapaxes(matrices, -1, -2))


def validate_spd_matrices(matrices, name, symmetry_atol=1e-8):
    matrices = np.asarray(matrices, dtype=np.float64)
    if matrices.ndim != 3 or matrices.shape[-1] != matrices.shape[-2]:
        raise ValueError(
            "{} must have shape [samples, d, d], got {}".format(
                name, tuple(matrices.shape)
            )
        )
    nonfinite = int(np.size(matrices) - np.isfinite(matrices).sum())
    if nonfinite:
        raise ValueError("{} contains {} NaN/Inf values".format(name, nonfinite))
    symmetry_error = float(np.max(np.abs(
        matrices - np.swapaxes(matrices, -1, -2)
    )))
    if symmetry_error > float(symmetry_atol):
        raise ValueError(
            "{} is not symmetric: max error {:.3e} > {:.3e}".format(
                name, symmetry_error, float(symmetry_atol)
            )
        )
    eigenvalues = np.linalg.eigvalsh(_symmetrize(matrices))
    minimum = float(np.min(eigenvalues))
    nonpositive = int(np.sum(np.min(eigenvalues, axis=1) <= 0.0))
    if nonpositive:
        raise ValueError(
            "{} contains {} non-positive matrices (minimum eigenvalue {:.3e})"
            .format(name, nonpositive, minimum)
        )
    return {
        "name": str(name),
        "shape": [int(v) for v in matrices.shape],
        "minimum_eigenvalue": minimum,
        "non_positive_matrices": nonpositive,
        "non_finite_values": nonfinite,
        "maximum_symmetry_error": symmetry_error,
    }


def _symmetric_function(matrix, function, context):
    matrix = _symmetrize(matrix)
    values, vectors = np.linalg.eigh(matrix)
    if not np.isfinite(values).all() or float(np.min(values)) <= 0.0:
        raise ValueError(
            "{} requires an SPD matrix; minimum eigenvalue is {}".format(
                context, float(np.min(values))
            )
        )
    transformed = function(values)
    return _symmetrize((vectors * transformed[None, :]) @ vectors.T)


def _batch_log_whitened(matrices, inverse_sqrt, batch_size):
    matrices = np.asarray(matrices, dtype=np.float64)
    outputs = []
    for start in range(0, len(matrices), int(batch_size)):
        batch = _symmetrize(matrices[start:start + int(batch_size)])
        whitened = np.matmul(
            np.matmul(inverse_sqrt[None, :, :], batch),
            inverse_sqrt[None, :, :],
        )
        whitened = _symmetrize(whitened)
        values, vectors = np.linalg.eigh(whitened)
        minimum = float(np.min(values))
        if not np.isfinite(values).all() or minimum <= 0.0:
            raise ValueError(
                "Tangent logarithm received a non-SPD matrix; minimum "
                "eigenvalue is {:.3e}".format(minimum)
            )
        logged = np.matmul(
            vectors * np.log(values)[:, None, :],
            np.swapaxes(vectors, -1, -2),
        )
        outputs.append(_symmetrize(logged))
    return np.concatenate(outputs, axis=0)


def airm_karcher_mean(matrices, tolerance=1e-7, max_iterations=50,
                      batch_size=512):
    """Compute the AIRM Frechet mean with TSMNet's SPDDSBN Karcher flow."""
    matrices = _symmetrize(np.asarray(matrices, dtype=np.float64))
    validate_spd_matrices(matrices, "source_pre_for_reference")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    tsmnet_root = os.path.join(project_root, "TSMNet")
    if tsmnet_root not in sys.path:
        sys.path.insert(0, tsmnet_root)
    from spdnets.functionals import spd_mean_kracher_flow

    tensor = torch.from_numpy(matrices).to(dtype=torch.double, device="cpu")
    with torch.no_grad():
        mean, diagnostics = spd_mean_kracher_flow(
            tensor, maxiter=int(max_iterations), dim=0,
            tolerance=float(tolerance), return_info=True,
        )
    mean = _symmetrize(mean.squeeze(0).detach().cpu().numpy())
    diagnostics["batch_size_argument"] = int(batch_size)
    diagnostics["implementation"] = "TSMNet.spdnets.functionals.spd_mean_kracher_flow"
    if not diagnostics["converged"]:
        raise RuntimeError(
            "AIRM Karcher mean did not converge in {} iterations; residual "
            "{:.3e} exceeds tolerance {:.3e}".format(
                diagnostics["iterations"], diagnostics["residual"],
                float(tolerance)
            )
        )
    validate_spd_matrices(mean[None], "Karcher_reference")
    return mean, diagnostics


def tangent_vectorize(symmetric_matrices):
    matrices = _symmetrize(np.asarray(symmetric_matrices, dtype=np.float64))
    if matrices.ndim != 3 or matrices.shape[-1] != matrices.shape[-2]:
        raise ValueError("Tangent matrices must have shape [samples, d, d]")
    dimension = matrices.shape[-1]
    rows, cols = np.triu_indices(dimension)
    vectors = matrices[:, rows, cols].copy()
    vectors[:, rows != cols] *= math.sqrt(2.0)
    expected = dimension * (dimension + 1) // 2
    if vectors.shape[1] != expected:
        raise RuntimeError("Unexpected tangent-vector dimension")
    return vectors


def common_tangent_vectors(matrices, reference, batch_size=512):
    reference = _symmetrize(np.asarray(reference, dtype=np.float64))
    validate_spd_matrices(reference[None], "common_reference")
    inverse_sqrt = _symmetric_function(
        reference, lambda values: 1.0 / np.sqrt(values),
        "common reference inverse square root",
    )
    tangents = _batch_log_whitened(matrices, inverse_sqrt, batch_size)
    return tangent_vectorize(tangents)


def alignment_metrics(pre_vectors, post_vectors, domains, class_ids,
                      epsilon=1e-12):
    pre_vectors = np.asarray(pre_vectors, dtype=np.float64)
    post_vectors = np.asarray(post_vectors, dtype=np.float64)
    domains = np.asarray(domains).astype(str)
    class_ids = np.asarray(class_ids, dtype=np.int64)
    if pre_vectors.shape != post_vectors.shape:
        raise ValueError("Pre/post tangent features must have identical shapes")

    source = domains == "source"
    target = domains == "target"
    if not source.any() or not target.any():
        raise ValueError("Alignment metrics require source and target samples")

    def domain_distance(vectors):
        return float(np.linalg.norm(
            vectors[source].mean(axis=0) - vectors[target].mean(axis=0)
        ))

    def conditional_distance(vectors):
        distances = []
        for class_id in sorted(int(v) for v in np.unique(class_ids)):
            source_class = source & (class_ids == class_id)
            target_class = target & (class_ids == class_id)
            if source_class.any() and target_class.any():
                distances.append(np.linalg.norm(
                    vectors[source_class].mean(axis=0)
                    - vectors[target_class].mean(axis=0)
                ))
        if not distances:
            return float("nan")
        return float(np.mean(distances))

    def fisher_ratio(vectors):
        centers = []
        within = []
        for class_id in sorted(int(v) for v in np.unique(class_ids)):
            mask = class_ids == class_id
            if not mask.any():
                continue
            center = vectors[mask].mean(axis=0)
            centers.append(center)
            within.append(np.mean(np.sum((vectors[mask] - center) ** 2, axis=1)))
        pairwise = []
        for left in range(len(centers)):
            for right in range(left + 1, len(centers)):
                pairwise.append(np.sum((centers[left] - centers[right]) ** 2))
        if not pairwise:
            return float("nan")
        return float(np.mean(pairwise) / (np.mean(within) + float(epsilon)))

    return {
        "domain_centroid_distance_pre": domain_distance(pre_vectors),
        "domain_centroid_distance_post": domain_distance(post_vectors),
        "class_balanced_domain_distance_pre": conditional_distance(pre_vectors),
        "class_balanced_domain_distance_post": conditional_distance(post_vectors),
        "fisher_ratio_pre": fisher_ratio(pre_vectors),
        "fisher_ratio_post": fisher_ratio(post_vectors),
        "feature_dimension": int(pre_vectors.shape[1]),
    }


def _uniform_subject_sample(frame, count, rng):
    groups = {}
    for subject, group in frame.groupby("subject_id", sort=True):
        values = group["sample_id"].to_numpy(dtype=np.int64).copy()
        rng.shuffle(values)
        groups[int(subject)] = list(values)
    selected = []
    subjects = sorted(groups)
    while len(selected) < int(count):
        progressed = False
        for subject in subjects:
            if groups[subject] and len(selected) < int(count):
                selected.append(groups[subject].pop())
                progressed = True
        if not progressed:
            break
    return selected


def balanced_plot_manifest(metadata, max_per_class_domain=500, seed=2026):
    metadata = metadata.copy()
    required = {"sample_id", "subject_id", "domain", "class_id"}
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ValueError("Plot metadata is missing columns {}".format(missing))
    rng = np.random.RandomState(int(seed))
    selected = []
    classes = sorted(int(v) for v in metadata["class_id"].unique())
    for class_id in classes:
        source = metadata[
            (metadata["domain"] == "source")
            & (metadata["class_id"] == class_id)
        ]
        target = metadata[
            (metadata["domain"] == "target")
            & (metadata["class_id"] == class_id)
        ]
        count = min(int(max_per_class_domain), len(source), len(target))
        if count < 1:
            raise ValueError(
                "Class {} has no balanced source/target samples".format(class_id)
            )
        target_ids = target["sample_id"].to_numpy(dtype=np.int64).copy()
        rng.shuffle(target_ids)
        selected.extend((int(v), "target", class_id) for v in target_ids[:count])
        source_ids = _uniform_subject_sample(source, count, rng)
        if len(source_ids) != count:
            raise RuntimeError("Could not draw the requested balanced source sample")
        selected.extend((int(v), "source", class_id) for v in source_ids)

    order = {int(sample_id): position for position, (sample_id, _, _) in enumerate(selected)}
    manifest = metadata[metadata["sample_id"].isin(order)].copy()
    manifest["plot_order"] = manifest["sample_id"].map(order).astype(np.int64)
    manifest = manifest.sort_values("plot_order").reset_index(drop=True)
    counts = manifest.groupby(["class_id", "domain"]).size().unstack(fill_value=0)
    if not np.array_equal(counts.get("source", 0), counts.get("target", 0)):
        raise RuntimeError("Source/target plot counts are not class balanced")
    return manifest


def choose_median_fold(results, subject_column="subject",
                       metric_column="test_bacc"):
    frame = pd.DataFrame(results).copy()
    if subject_column not in frame or metric_column not in frame:
        raise ValueError(
            "Per-subject results require columns {!r} and {!r}".format(
                subject_column, metric_column
            )
        )
    frame[subject_column] = pd.to_numeric(frame[subject_column], errors="raise").astype(int)
    frame[metric_column] = pd.to_numeric(frame[metric_column], errors="coerce")
    frame = frame[np.isfinite(frame[metric_column])]
    if frame.empty:
        raise ValueError("No finite per-subject balanced accuracy values were found")
    per_subject = frame.groupby(subject_column, sort=True)[metric_column].mean()
    median = float(per_subject.median())
    distances = (per_subject - median).abs()
    minimum = float(distances.min())
    tied = sorted(int(v) for v in distances.index[
        np.isclose(distances, minimum, rtol=0.0, atol=1e-12)
    ])
    subject = tied[0]
    return {
        "selected_target_subject": int(subject),
        "subject_balanced_accuracy": float(per_subject.loc[subject]),
        "all_subject_median_balanced_accuracy": median,
        "selection_rule": "closest_to_median",
        "subject_seed_rows_averaged": int((frame[subject_column] == subject).sum()),
    }
