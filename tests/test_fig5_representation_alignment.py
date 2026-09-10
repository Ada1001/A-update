import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.fig5_representation_alignment import (
    _load_methods,
    _parse_datasets,
    _save_figure,
    _validate_comparable_runs,
    balanced_sample,
    high_dimensional_metrics,
    reduce_to_2d,
)


class Fig5RepresentationAlignmentTests(unittest.TestCase):
    def test_fourth_panel_tsmnet_preset_preserves_first_three_methods(self):
        baseline = _load_methods(None)
        methods = _load_methods(None, fourth_model="tsmnet", fourth_run_dir="custom_tsmnet")
        self.assertEqual(methods[:3], baseline[:3])
        self.assertEqual(methods[3]["model_type"], "tsmnet")
        self.assertEqual(methods[3]["bnorm"], "spddsbn")
        self.assertEqual(methods[3]["run_dir"], "custom_tsmnet")
        self.assertEqual(methods[3]["label"], "TSMNet-SPDDSBN")

    def test_mixed_models_compare_only_mstgc_frontends(self):
        _, infos = self._comparable_run_infos()
        methods = _load_methods(None, fourth_model="tsmnet")
        for info in infos[:3]:
            info["record"]["mstgc_temporal_hidden"] = 32
        # Unrelated/default MSTGC metadata on TSMNet must not be a reference.
        infos[3]["record"]["mstgc_temporal_hidden"] = 64
        _validate_comparable_runs(infos, methods, SimpleNamespace(allow_legacy_refit=False))
        infos[0]["record"]["mstgc_temporal_hidden"] = 16
        with self.assertRaisesRegex(ValueError, "different shared-front-end"):
            _validate_comparable_runs(infos, methods, SimpleNamespace(allow_legacy_refit=False))

    def test_manifest_and_fourth_preset_conflict_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "either"):
            _load_methods("unused.json", fourth_model="tsmnet")

    def _metadata(self, samples_per_group=8):
        rows = []
        sample_id = 0
        for class_id in [0, 1]:
            for domain in ["source", "target"]:
                for index in range(samples_per_group):
                    rows.append({
                        "sample_id": sample_id,
                        "subject_id": index % 3 + (0 if domain == "source" else 9),
                        "domain": domain,
                        "class_id": class_id,
                    })
                    sample_id += 1
        return pd.DataFrame(rows)

    def test_balanced_sample_uses_equal_class_domain_counts(self):
        metadata = self._metadata(samples_per_group=9)
        manifest = balanced_sample(metadata, max_points_per_group=5, seed=2026)
        counts = manifest.groupby(["class_id", "domain"]).size()
        self.assertTrue((counts == 5).all())
        self.assertFalse(manifest["sample_id"].duplicated().any())

    def test_high_dimensional_metrics_reward_alignment_not_plot_coordinates(self):
        metadata = self._metadata(samples_per_group=10)
        rng = np.random.RandomState(7)
        labels = metadata["class_id"].to_numpy(dtype=np.int64)
        target = metadata["domain"].to_numpy() == "target"
        base = np.column_stack([labels * 3.0, labels * 1.5])
        aligned = base + rng.normal(0.0, 0.08, size=base.shape)
        shifted = aligned.copy()
        shifted[target] += np.asarray([2.0, -1.0])
        aligned_metrics = high_dimensional_metrics(aligned, metadata)
        shifted_metrics = high_dimensional_metrics(shifted, metadata)
        self.assertLess(
            aligned_metrics["domain_discrepancy"],
            shifted_metrics["domain_discrepancy"],
        )
        self.assertGreater(
            aligned_metrics["separation_ratio"],
            shifted_metrics["separation_ratio"],
        )

    def test_tsne_fallback_is_finite_and_reproducible(self):
        metadata = self._metadata(samples_per_group=6)
        rng = np.random.RandomState(11)
        features = rng.normal(size=(len(metadata), 8))
        first, details = reduce_to_2d(
            features, metadata, reducer="tsne", pca_dim=5,
            seed=2026, tsne_perplexity=5, tsne_max_iter=300,
        )
        second, _ = reduce_to_2d(
            features, metadata, reducer="tsne", pca_dim=5,
            seed=2026, tsne_perplexity=5, tsne_max_iter=300,
        )
        self.assertEqual(first.shape, (len(metadata), 2))
        self.assertTrue(np.isfinite(first).all())
        self.assertTrue(np.allclose(first, second))
        self.assertEqual(details["reduction"]["method"], "t-SNE")
        self.assertFalse(details["reduction"]["target_labels_used_to_fit"])

    def test_method_manifest_accepts_tsmnet_adapter_entries(self):
        methods = [
            {"label": "T{}".format(index), "short_label": "T{}".format(index),
             "model_type": "tsmnet", "bnorm": normalization}
            for index, normalization in enumerate(
                [None, "spdbn", "spddsbn", "spddsbn"], start=1
            )
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "methods.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(methods, handle)
            loaded = _load_methods(path)
        self.assertEqual(len(loaded), 4)
        self.assertTrue(all(item["model_type"] == "tsmnet" for item in loaded))

    def test_reserved_dataset_label_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "identifies eegmat"):
            _parse_datasets("stew,cog-bci:nback", "STEW,EEGMAT")

    def test_single_dataset_figure_input_is_supported(self):
        parsed = _parse_datasets("cog-bci:nback", "N-Back")
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["name"], "cog-bci-nback")

    def _comparable_run_infos(self):
        methods = _load_methods(None)
        expected = [
            ("shared_channel_graph_mean_v3", "mean", False),
            ("shared_channel_graph_mean_v3", "mean", True),
            ("shared_channel_graph_augmented_spd_v3", "augmented", False),
            ("shared_channel_graph_augmented_spd_v3", "augmented", True),
        ]
        infos = []
        for index, (architecture, representation, adapted) in enumerate(expected):
            record = {
                "seed": 42, "val_size": 0.2, "test_size": 0.2,
                "mstgc_architecture": architecture,
                "mstgc_representation": representation,
            }
            if adapted:
                record["target_refit_scope"] = "target_only"
            infos.append({
                "record": record,
                "summary": pd.DataFrame({"target_adapt": [adapted, adapted]}),
                "run_dir": "run_{}".format(index),
                "summary_path": "summary_{}.csv".format(index),
            })
        return methods, infos

    def test_run_audit_accepts_master_scope_fallback(self):
        methods, infos = self._comparable_run_infos()
        result = _validate_comparable_runs(
            infos, methods, SimpleNamespace(allow_legacy_refit=False)
        )
        self.assertEqual(result["seed"], 42)
        self.assertEqual(
            infos[1]["refit_scope_audit"]["evidence"], "master_summary.csv"
        )

    def test_run_audit_rejects_missing_v3_provenance(self):
        methods, infos = self._comparable_run_infos()
        infos[1]["record"].pop("mstgc_architecture")
        with self.assertRaisesRegex(ValueError, "predates or differs"):
            _validate_comparable_runs(
                infos, methods, SimpleNamespace(allow_legacy_refit=True)
            )

    def test_figure_writer_emits_pdf_png_and_svg(self):
        with tempfile.TemporaryDirectory() as directory:
            fig, ax = plt.subplots(figsize=(2, 1.5))
            ax.plot([0, 1], [0, 1])
            paths = _save_figure(fig, directory)
            plt.close(fig)
            self.assertTrue(all(os.path.exists(path) for path in paths))


if __name__ == "__main__":
    unittest.main()
