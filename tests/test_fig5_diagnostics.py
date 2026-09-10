from types import SimpleNamespace
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
import torch
from analysis import diagnose_fig5_alignment as diag
from src.cl_tsmnet.training import fit_source_normalizer


@pytest.mark.parametrize("name", ["ms_tgc_spddsbn", "mstgc_dta_cheb_eudsbn", "tsmnet"])
def test_diagnostic_end_to_end_preserves_target_and_checkpoint(tmp_path, monkeypatch, name):
    rng = np.random.RandomState(5)
    dataset = {"x": rng.randn(24, 4, 32).astype(np.float32),
               "y": np.tile([0, 1], 12), "fs": 128.,
               "meta": pd.DataFrame({"subject": np.repeat([1, 2, 3], 8),
                                     "session": 1, "task": "test", "start_sample": np.arange(24) * 32})}
    domains = np.repeat([1, 2, 3], 8)
    source, target, val = np.arange(8), np.arange(16, 24), np.arange(8, 16)
    normalizer = fit_source_normalizer(dataset["x"], source)
    split = {"source_ids": source, "target_ids": target, "val_ids": val,
             "normalizer": normalizer, "config": {"seed": 42}}
    monkeypatch.setattr(diag.fig5, "_make_split_context", lambda *args: split)
    method = next(m for m in (diag.fig5._load_methods(None) +
                 diag.fig5._load_methods(None, fourth_model="tsmnet")) if m["model_type"] == name)
    record = {"mstgc_temporal_hidden": 4, "mstgc_graph_hidden": 4,
              "mstgc_fusion_dim": 6, "mstgc_num_heads": 2, "mstgc_time_points": 8,
              "subspacedims": 3, "spatial_filters": 6, "temp_kernel": 5,
              "mstgc_dropout": 0.}
    model = diag.fig5._build_model(dataset, domains, np.arange(24), source, method,
                                  diag.fig5._model_config(record), torch.device("cpu"))
    checkpoint = tmp_path / "run" / "subject_03" / "model.pt"
    checkpoint.parent.mkdir(parents=True)
    torch.save(model.state_dict(), checkpoint)
    info = {"run_dir": str(checkpoint.parent.parent), "record": record,
            "summary": pd.DataFrame({"subject": [3]})}
    context = {"dataset_object": dataset, "domains": domains, "cache": "synthetic"}
    args = SimpleNamespace(output_dir=str(tmp_path / "diagnostic"), device_object=torch.device("cpu"),
                           batch_size=4, seed=2026, max_points_per_group=4)
    diag.diagnose_fold(context, info, method, 3, args)
    out = Path(args.output_dir) / name / "subject_03"
    audit = json.loads((out / "audit.json").read_text())
    assert audit["checkpoint_unchanged"] and audit["target_predictions_identical"]
    assert audit["changed_source_buffer_keys"]
    assert audit["target_logit_max_abs_delta"] == 0
    with np.load(out / "features.npz") as f:
        assert np.allclose(f["saved_representation"][8:], f["source_refit_representation"][8:])
        assert not np.allclose(f["saved_representation"][:8], f["source_refit_representation"][:8])
    # Extraction locations have different meanings; TSMNet has no 128D head.
    if name == "tsmnet":
        with np.load(out / "features.npz") as f:
            np.testing.assert_array_equal(f["saved_representation"], f["saved_classifier_input"])
    if name == "ms_tgc_spddsbn":
        metrics_hash = diag.digest(out / "alignment.csv")
        features_hash = diag.digest(out / "features.npz")
        diag.embed_directory(args.output_dir, ["tsne"])
        assert (out / "representation_tsne.png").exists()
        assert (out / "classifier_input_tsne.png").exists()
        assert diag.digest(out / "alignment.csv") == metrics_hash
        assert diag.digest(out / "features.npz") == features_hash
        coordinates = pd.read_csv(out / "representation_tsne.csv")
        assert coordinates.groupby("condition").sample_id.apply(lambda v: tuple(v)).nunique() == 1


def test_state_audit_rejects_parameter_changes():
    model = torch.nn.Linear(2, 2)
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        model.weight.add_(1)
    with pytest.raises(AssertionError, match="non-source"):
        diag.audit_state(before, model, [1])
