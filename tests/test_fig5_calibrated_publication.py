from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import torch
from analysis import fig5_representation_alignment as fig5
from analysis.diagnose_fig5_alignment import digest
from src.cl_tsmnet.training import fit_source_normalizer


@pytest.mark.parametrize("name,adapted", [
    ("mstgc_mean_ce", False), ("mstgc_dta_cheb_eudsbn", True),
    ("mstgc_dta_cheb_spdbn", False), ("ms_tgc_spddsbn", True), ("tsmnet", True),
])
def test_publication_calibration_features_cache_and_target_invariance(tmp_path, name, adapted):
    dataset = {"x": np.random.RandomState(9).randn(24, 4, 32).astype(np.float32),
               "y": np.tile([0, 1], 12), "fs": 128.,
               "meta": pd.DataFrame({"subject": np.repeat([1, 2, 3], 8),
                                     "session": 1, "task": "test", "start_sample": np.arange(24) * 32})}
    domains = np.repeat([1, 2, 3], 8)
    source, target, val = np.arange(8), np.arange(16, 24), np.arange(8, 16)
    split = {"source_ids": source, "target_ids": target, "val_ids": val,
             "normalizer": fit_source_normalizer(dataset["x"], source), "config": {"seed": 42}}
    method = next(m for m in fig5._load_methods(None) + fig5._load_methods(None, "tsmnet")
                  if m["model_type"] == name)
    record = {"mstgc_temporal_hidden": 4, "mstgc_graph_hidden": 4, "mstgc_fusion_dim": 7,
              "mstgc_num_heads": 2, "mstgc_time_points": 8, "subspacedims": 3,
              "spatial_filters": 6, "temp_kernel": 5, "mstgc_dropout": 0.}
    model = fig5._build_model(dataset, domains, np.arange(24), source, method,
                             fig5._model_config(record), torch.device("cpu"))
    checkpoint = tmp_path / "run" / "subject_03" / "model.pt"
    checkpoint.parent.mkdir(parents=True)
    torch.save(model.state_dict(), checkpoint)
    checksum = digest(checkpoint)
    info = {"run_dir": str(checkpoint.parent.parent), "record": record}
    context = {"dataset_object": dataset, "domains": domains, "cache": "synthetic", "spec": {"name": "stew"}}
    args = SimpleNamespace(feature_cache_dir=str(tmp_path / "features"), force_reextract=False,
                           source_calibration="refit", feature_location="classifier_input",
                           batch_size=4, device_object=torch.device("cpu"))
    first, metadata, location = fig5.load_feature_set(context, method, info, 3, split, args)
    assert first.shape == (16, 6 if name == "tsmnet" else 7)
    audit = location["calibration_audit"]
    assert audit["target_prediction_identical"] and audit["weights_and_non_source_buffers_unchanged"]
    assert audit["target_logit_max_abs_delta"] == 0
    assert bool(audit["changed_source_buffer_keys"]) == adapted
    cached, _, cached_location = fig5.load_feature_set(context, method, info, 3, split, args)
    np.testing.assert_array_equal(first, cached)
    assert cached_location == location
    args.source_calibration = "saved"
    saved, _, _ = fig5.load_feature_set(context, method, info, 3, split, args)
    np.testing.assert_array_equal(first[8:], saved[8:])
    if not adapted:
        np.testing.assert_array_equal(first, saved)
    args.feature_location = "representation"
    original, _, _ = fig5.load_feature_set(context, method, info, 3, split, args)
    assert original.shape == (16, 4 if "mean" in name or "eudsbn" in name else 6)
    assert len(list((tmp_path / "features").rglob("signature.json"))) == 3
    assert digest(checkpoint) == checksum
