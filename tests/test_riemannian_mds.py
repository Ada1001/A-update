import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from analysis.plot_riemannian_mds import (
    _resolve_feature_cache_dir,
    _resolve_original_output_inputs,
    _select_representative,
    _validate_selected_result,
    airm_distance,
    fit_metric_mds,
    pairwise_airm,
    validate_spd,
)


class RiemannianMDSAnalysisTests(unittest.TestCase):
    def test_extracted_feature_cache_directory_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(
                _resolve_feature_cache_dir(directory, "unused"),
                os.path.abspath(directory),
            )

    def test_missing_explicit_feature_cache_fails_at_the_cache_path(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = os.path.join(directory, "missing")
            with self.assertRaisesRegex(FileNotFoundError, "not an extracted"):
                _resolve_feature_cache_dir(missing, directory)

    def test_original_loso_output_is_resolved_without_feature_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = os.path.join(directory, "stew_loso_ms_tgc_spddsbn")
            os.makedirs(run_dir)
            args = self._original_output_args(directory)
            _resolve_original_output_inputs(args)
            self.assertEqual(os.path.abspath(args.checkpoint_root), run_dir)
            self.assertEqual(
                os.path.abspath(args.results_file),
                os.path.join(run_dir, "summary.csv"),
            )
            self.assertEqual(args.seed, 42)
            self.assertEqual(args.mstgc_cheby_order, 3)

    def test_master_summary_restores_exact_run_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = os.path.join(directory, "stew_loso_ms_tgc_spddsbn")
            os.makedirs(run_dir)
            pd.DataFrame([{
                "dataset": "stew",
                "model_type": "ms_tgc_spddsbn",
                "protocol": "loso",
                "output_dir": run_dir,
                "seed": 17,
                "val_size": 0.25,
                "test_size": 0.2,
                "target_fs": 128,
                "mstgc_cheby_order": 4,
                "mstgc_graph_k": 7,
            }]).to_csv(os.path.join(directory, "master_summary.csv"), index=False)
            args = self._original_output_args(directory)
            _resolve_original_output_inputs(args)
            self.assertEqual(args.seed, 17)
            self.assertEqual(args.val_size, 0.25)
            self.assertEqual(args.mstgc_cheby_order, 4)
            self.assertEqual(args.mstgc_graph_k, 7)
            self.assertEqual(args.target_fs, 128.0)
            self.assertTrue(args.run_record)

    def test_selected_result_requires_unlabeled_target_only_refit(self):
        with tempfile.TemporaryDirectory() as directory:
            result_path = os.path.join(directory, "summary.csv")
            pd.DataFrame([{
                "dataset": "stew",
                "model_type": "ms_tgc_spddsbn",
                "protocol": "loso",
                "subject": 6,
                "target_adapt": True,
                "target_refit_scope": "target_only",
            }]).to_csv(result_path, index=False)
            args = SimpleNamespace(
                dataset="stew", cog_paradigm="nback",
                model="ms_tgc_spddsbn", run_record={},
                allow_legacy_source_target_refit=False,
            )
            validation = _validate_selected_result(args, {
                "selected_target_subject": 6,
                "results_file": result_path,
            })
            self.assertEqual(validation["status"], "validated_target_only_refit")
            self.assertEqual(validation["target_refit_scope"], ["target_only"])

    def test_missing_scope_uses_matching_master_summary_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            result_path = os.path.join(directory, "summary.csv")
            pd.DataFrame([{
                "dataset": "stew", "model_type": "tsmnet",
                "protocol": "loso", "subject": 6,
                "target_adapt": True, "bnorm": "spddsbn",
            }]).to_csv(result_path, index=False)
            args = SimpleNamespace(
                dataset="stew", cog_paradigm="nback", model="tsmnet",
                run_record={"target_refit_scope": "target_only"},
                allow_legacy_source_target_refit=False,
            )
            validation = _validate_selected_result(args, {
                "selected_target_subject": 6, "results_file": result_path,
            })
            self.assertEqual(validation["target_refit_scope"], ["target_only"])
            self.assertEqual(
                validation["target_refit_scope_evidence"],
                "matched_master_summary",
            )

    def test_legacy_scope_requires_explicit_opt_in_and_is_disclosed(self):
        with tempfile.TemporaryDirectory() as directory:
            result_path = os.path.join(directory, "summary.csv")
            pd.DataFrame([{
                "dataset": "stew", "model_type": "tsmnet",
                "protocol": "loso", "subject": 6,
                "target_adapt": True, "bnorm": "spddsbn",
            }]).to_csv(result_path, index=False)
            args = SimpleNamespace(
                dataset="stew", cog_paradigm="nback", model="tsmnet",
                run_record={}, allow_legacy_source_target_refit=False,
            )
            selection = {
                "selected_target_subject": 6, "results_file": result_path,
            }
            with self.assertRaisesRegex(ValueError, "historical pipeline"):
                _validate_selected_result(args, selection)
            args.allow_legacy_source_target_refit = True
            with self.assertWarnsRegex(UserWarning, "legacy checkpoint"):
                validation = _validate_selected_result(args, selection)
            self.assertEqual(
                validation["status"],
                "validated_legacy_source_target_unlabeled_refit",
            )
            self.assertTrue(validation["publication_warning"])

    @staticmethod
    def _original_output_args(output_root):
        values = {
            "dataset": "stew",
            "cog_paradigm": "nback",
            "model": "ms_tgc_spddsbn",
            "checkpoint_root": None,
            "results_file": None,
            "output_root": output_root,
            "master_summary": None,
            "feature_cache_dir": None,
            "cache": None,
            "target_fs": None,
            "artifact_z": None,
            "seed": None,
            "val_size": None,
            "test_size": None,
        }
        values.update({key: None for key in [
            "temporal_filters", "spatial_filters", "subspacedims", "temp_kernel",
            "mstgc_temporal_hidden", "mstgc_graph_hidden", "mstgc_fusion_dim",
            "mstgc_kernel_length", "mstgc_num_heads", "mstgc_cheby_order",
            "mstgc_dropout", "mstgc_num_nodes", "mstgc_graph_k",
            "mstgc_graph_density", "mstgc_time_points", "mstgc_shrinkage",
        ]})
        return SimpleNamespace(**values)

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
