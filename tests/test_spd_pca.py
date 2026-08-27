import os
import unittest

import numpy as np
import pandas as pd
import torch

from src.cl_tsmnet.spd_pca import (
    airm_karcher_mean,
    alignment_metrics,
    balanced_plot_manifest,
    choose_median_fold,
    common_tangent_vectors,
    tangent_vectorize,
    validate_spd_matrices,
)
from src.cl_tsmnet.training import build_ms_tgc_spddsbn


class SPDPCAUtilitiesTests(unittest.TestCase):
    def test_diagonal_airm_mean_and_common_tangent_dimension(self):
        matrices = np.asarray([
            np.diag([1.0, 4.0, 9.0]),
            np.diag([4.0, 9.0, 16.0]),
            np.diag([9.0, 16.0, 25.0]),
        ], dtype=np.float64)
        reference, diagnostics = airm_karcher_mean(
            matrices, tolerance=1e-10, max_iterations=20, batch_size=2
        )
        expected = np.diag(np.exp(np.mean(np.log(np.diagonal(matrices, axis1=1, axis2=2)), axis=0)))
        self.assertTrue(np.allclose(reference, expected, atol=1e-9))
        self.assertTrue(diagnostics["converged"])
        vectors = common_tangent_vectors(matrices, reference, batch_size=2)
        self.assertEqual(vectors.shape, (3, 6))

    def test_tangent_vectorization_preserves_frobenius_norm(self):
        rng = np.random.RandomState(9)
        raw = rng.randn(5, 4, 4)
        symmetric = 0.5 * (raw + raw.transpose(0, 2, 1))
        vectors = tangent_vectorize(symmetric)
        matrix_norms = np.sum(symmetric ** 2, axis=(1, 2))
        vector_norms = np.sum(vectors ** 2, axis=1)
        self.assertTrue(np.allclose(matrix_norms, vector_norms, atol=1e-12))

    def test_spd_validation_fails_closed(self):
        valid = np.stack([np.eye(3), 2.0 * np.eye(3)])
        result = validate_spd_matrices(valid, "valid")
        self.assertGreater(result["minimum_eigenvalue"], 0.0)
        invalid = valid.copy()
        invalid[0, 0, 0] = -1.0
        with self.assertRaisesRegex(ValueError, "non-positive"):
            validate_spd_matrices(invalid, "invalid")

    def test_balanced_manifest_is_deterministic_and_subject_balanced(self):
        rows = []
        sample_id = 0
        for class_id in [0, 1]:
            for domain in ["source", "target"]:
                subjects = [1, 2, 3] if domain == "source" else [9]
                for subject in subjects:
                    for _ in range(4):
                        rows.append({
                            "sample_id": sample_id,
                            "subject_id": subject,
                            "domain": domain,
                            "class_id": class_id,
                            "class_name": str(class_id),
                        })
                        sample_id += 1
        metadata = pd.DataFrame(rows)
        first = balanced_plot_manifest(metadata, 6, seed=2026)
        second = balanced_plot_manifest(metadata, 6, seed=2026)
        self.assertTrue(first.equals(second))
        counts = first.groupby(["class_id", "domain"]).size().unstack()
        self.assertTrue(np.array_equal(counts["source"], counts["target"]))
        for class_id in [0, 1]:
            source = first[(first["class_id"] == class_id) & (first["domain"] == "source")]
            self.assertEqual(source["subject_id"].nunique(), 3)

    def test_median_fold_averages_seed_rows_and_breaks_ties_by_subject(self):
        rows = pd.DataFrame({
            "subject": [1, 1, 2, 3],
            "test_bacc": [0.6, 0.8, 0.7, 0.9],
        })
        selected = choose_median_fold(rows)
        self.assertEqual(selected["selected_target_subject"], 1)
        self.assertAlmostEqual(selected["subject_balanced_accuracy"], 0.7)
        self.assertEqual(selected["subject_seed_rows_averaged"], 2)

    def test_alignment_metrics_use_full_tangent_dimension(self):
        pre = np.asarray([[0., 0.], [1., 0.], [2., 0.], [3., 0.]])
        post = np.asarray([[0., 0.], [1., 0.], [1., 0.], [2., 0.]])
        domains = np.asarray(["source", "source", "target", "target"])
        classes = np.asarray([0, 1, 0, 1])
        metrics = alignment_metrics(pre, post, domains, classes)
        self.assertEqual(metrics["feature_dimension"], 2)
        self.assertLess(
            metrics["domain_centroid_distance_post"],
            metrics["domain_centroid_distance_pre"],
        )


class SPDPCAForwardContractTests(unittest.TestCase):
    def test_intermediate_forward_preserves_default_logits(self):
        model = build_ms_tgc_spddsbn(
            os.getcwd(), nchannels=4, nsamples=32, nclasses=2,
            domains=np.asarray([0, 1]), temporal_hidden=4,
            graph_hidden=4, fusion_dim=6, kernel_length=5,
            num_heads=2, cheby_order=2, dropout=0.0,
            graph_time_points=8, subspacedims=3,
            covariance_shrinkage=0.1, variant="ms_tgc_spddsbn",
        ).eval()
        windows = torch.randn(4, 4, 32)
        domains = torch.tensor([0, 0, 1, 1])
        with torch.no_grad():
            default_logits = model(windows, domains)
            logits, intermediates = model(
                windows, domains, return_intermediates=True
            )
        self.assertTrue(torch.equal(default_logits, logits))
        self.assertEqual(tuple(intermediates["spd_pre_bn"].shape), (4, 3, 3))
        self.assertEqual(tuple(intermediates["spd_post_bn"].shape), (4, 3, 3))
        self.assertTrue(torch.all(
            torch.linalg.eigvalsh(intermediates["spd_pre_bn"]) > 0
        ))
        self.assertTrue(torch.all(
            torch.linalg.eigvalsh(intermediates["spd_post_bn"]) > 0
        ))


if __name__ == "__main__":
    unittest.main()
