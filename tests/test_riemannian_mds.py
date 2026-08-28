import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from analysis.plot_riemannian_mds import (
    _select_representative,
    airm_distance,
    fit_metric_mds,
    pairwise_airm,
    validate_spd,
)


class RiemannianMDSAnalysisTests(unittest.TestCase):
    def test_airm_distance_matches_diagonal_closed_form(self):
        left = np.diag([1.0, 4.0, 9.0])
        right = np.diag([4.0, 9.0, 36.0])
        expected = np.linalg.norm(np.log(np.asarray([4.0, 2.25, 4.0])))
        self.assertAlmostEqual(airm_distance(left, right), expected, places=12)

    def test_pairwise_airm_is_symmetric_with_zero_diagonal(self):
        matrices = np.stack([
            np.eye(3), 2.0 * np.eye(3), np.diag([1.0, 2.0, 4.0]),
        ])
        distances = pairwise_airm(matrices)
        self.assertTrue(np.allclose(distances, distances.T, atol=1e-12))
        self.assertTrue(np.allclose(np.diag(distances), 0.0, atol=1e-12))

    def test_representative_fold_averages_seeds_and_breaks_tie_by_subject(self):
        frame = pd.DataFrame({
            "dataset": ["stew"] * 6,
            "model_type": ["ms_tgc_spddsbn"] * 6,
            "protocol": ["loso"] * 6,
            "subject": [1, 1, 2, 2, 3, 3],
            "test_bacc": [0.6, 0.8, 0.65, 0.75, 0.8, 0.8],
        })
        handle, path = tempfile.mkstemp(suffix=".csv")
        os.close(handle)
        try:
            frame.to_csv(path, index=False)
            selected = _select_representative(
                path, "stew", "ms_tgc_spddsbn"
            )
        finally:
            os.remove(path)
        self.assertEqual(selected["selected_target_subject"], 1)
        self.assertAlmostEqual(selected["subject_balanced_accuracy"], 0.7)
        self.assertEqual(selected["subject_seed_rows_averaged"], 2)

    def test_metric_mds_is_deterministic_for_fixed_seed(self):
        matrices = np.stack([
            np.eye(2), 2.0 * np.eye(2), 3.0 * np.eye(2), 4.0 * np.eye(2),
        ])
        validate_spd(matrices, "test")
        distances = pairwise_airm(matrices)
        first, first_stress = fit_metric_mds(distances, 2, 2026)
        second, second_stress = fit_metric_mds(distances, 2, 2026)
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(first_stress["raw_stress"], second_stress["raw_stress"])


if __name__ == "__main__":
    unittest.main()
