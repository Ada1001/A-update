import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from analysis.plot_nonlinear_alignment_embeddings import (
    _coordinate_frame,
    _effective_perplexity,
    _stability,
    embedding_quality,
    fit_umap_runs,
    fit_joint_tsne,
    high_dimensional_metrics,
    knn_overlap,
    plot_pair,
    plot_stability,
)


class NonlinearEmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.metadata = pd.DataFrame({
            "sample_id": np.arange(12),
            "subject_id": [1] * 6 + [2] * 6,
            "domain": ["source"] * 6 + ["target"] * 6,
            "class_id": [0, 0, 0, 1, 1, 1] * 2,
            "class_name": ["Low"] * 3 + ["High"] * 3
                          + ["Low"] * 3 + ["High"] * 3,
            "split": ["source_train"] * 6 + ["target_test"] * 6,
        })
        base = np.arange(12, dtype=np.float64)[:, None]
        self.pre = np.hstack([base, np.sin(base), np.cos(base), base % 3])
        self.post = self.pre.copy()
        self.post[6:] = 0.75 * self.post[6:] + 0.25 * self.post[:6]

    def test_knn_overlap_is_one_for_identical_geometry(self):
        result = knn_overlap(self.pre, self.pre.copy(), 3)
        self.assertEqual(result["k"], 3)
        self.assertAlmostEqual(result["mean_overlap"], 1.0)

    def test_embedding_quality_returns_bounded_scores(self):
        result = embedding_quality(self.pre, self.pre[:, :2], 3)
        self.assertGreaterEqual(result["trustworthiness"], 0.0)
        self.assertLessEqual(result["trustworthiness"], 1.0)
        self.assertGreaterEqual(result["knn_overlap"]["mean_overlap"], 0.0)
        self.assertLessEqual(result["knn_overlap"]["mean_overlap"], 1.0)

    def test_high_dimensional_metrics_are_stage_specific(self):
        result = high_dimensional_metrics(self.pre, self.post, self.metadata)
        self.assertEqual(result["feature_dimension"], 4)
        self.assertIn("class_conditional_domain_distances", result["pre"])
        self.assertTrue(np.isfinite(result["post"]["fisher_ratio"]))

    def test_stability_is_exact_for_identical_coordinates(self):
        result = _stability(self.pre[:, :2], self.pre[:, :2].copy())
        self.assertAlmostEqual(result["pairwise_distance_spearman"], 1.0)
        self.assertAlmostEqual(result["procrustes_rmse"], 0.0, places=12)

    def test_perplexity_is_legal_and_sample_aware(self):
        self.assertLess(_effective_perplexity(100.0, 20), 20)
        self.assertAlmostEqual(_effective_perplexity(30.0, 301), 30.0)

    def test_coordinate_frame_keeps_paired_sample_ids(self):
        coordinates = _coordinate_frame(
            self.metadata, self.pre[:, :2], self.post[:, :2]
        )
        self.assertEqual(len(coordinates), 24)
        pre = coordinates[coordinates["stage"] == "pre"]["sample_id"].to_numpy()
        post = coordinates[coordinates["stage"] == "post"]["sample_id"].to_numpy()
        self.assertTrue(np.array_equal(pre, post))

    def test_tsne_is_one_joint_fit_with_paired_output(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                tsne_perplexity=5.0, tsne_max_iter=250,
                quality_neighbors=3, output_dir=directory,
            )
            pre, post, quality, parameters = fit_joint_tsne(
                self.pre, self.post, self.metadata, args
            )
            self.assertEqual(pre.shape, (12, 2))
            self.assertEqual(post.shape, (12, 2))
            self.assertTrue(parameters["joint_fit"])
            self.assertEqual(
                parameters["fit_order"],
                "[all balanced pre features; all paired balanced post features]",
            )
            saved = pd.read_csv(os.path.join(
                directory, "tsne_joint_coordinates_seed-2026.csv"
            ))
            self.assertEqual(len(saved), 24)
            self.assertIn("joint", quality)

    def test_umap_fits_source_pre_once_per_fixed_seed_and_saves_all_runs(self):
        class FakeUMAP:
            fit_shapes = []

            def __init__(self, **kwargs):
                self.seed = int(kwargs["random_state"])

            def fit(self, values):
                self.fit_shapes.append(tuple(values.shape))
                return self

            def transform(self, values):
                angle = (self.seed - 2026) * 0.1
                rotation = np.asarray([
                    [np.cos(angle), -np.sin(angle)],
                    [np.sin(angle), np.cos(angle)],
                ])
                return values[:, :2] @ rotation

        args = SimpleNamespace(
            umap_seeds="2026,2027,2028,2029,2030",
            umap_neighbors=3, umap_min_dist=0.15,
            umap_metric="euclidean", quality_neighbors=3,
        )
        with tempfile.TemporaryDirectory() as directory:
            args.output_dir = directory
            with patch(
                "analysis.plot_nonlinear_alignment_embeddings._import_umap",
                return_value=FakeUMAP,
            ):
                outputs, quality, stability, parameters = fit_umap_runs(
                    self.pre[:6], self.pre, self.post, self.metadata, args
                )
            self.assertEqual(sorted(outputs), [2026, 2027, 2028, 2029, 2030])
            self.assertEqual(FakeUMAP.fit_shapes, [(6, 4)] * 5)
            self.assertEqual(parameters["fit_partition"], "all_source_train_pre_only")
            self.assertEqual(len(stability), 5)
            self.assertIn("2026", quality)
            for seed in outputs:
                self.assertTrue(os.path.exists(os.path.join(
                    directory, "umap_coordinates_seed-{}.csv".format(seed)
                )))

    def test_publication_figures_write_vector_and_600dpi_outputs(self):
        pre = self.pre[:, :2]
        post = self.post[:, :2]
        outputs = {seed: (pre, post) for seed in range(2026, 2031)}
        stability = [{
            "seed": seed,
            "pre": {"pairwise_distance_spearman": 1.0, "procrustes_rmse": 0.0},
            "post": {"pairwise_distance_spearman": 1.0, "procrustes_rmse": 0.0},
        } for seed in range(2026, 2031)]
        with tempfile.TemporaryDirectory() as directory:
            plot_pair(
                pre, post, self.metadata, "umap", directory, True,
                "UMAP fitted on source-pre only; seed 2026 is prespecified.",
            )
            plot_stability(outputs, self.metadata, stability, directory)
            for stem in ["fig_umap_alignment", "fig_nonlinear_stability"]:
                for suffix in ["pdf", "svg", "png"]:
                    path = os.path.join(directory, "{}.{}".format(stem, suffix))
                    self.assertTrue(os.path.exists(path))
                    self.assertGreater(os.path.getsize(path), 0)


if __name__ == "__main__":
    unittest.main()
