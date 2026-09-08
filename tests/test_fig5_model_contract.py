"""Exercise Fig. 5 model semantics, feature exports and saved training state."""
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from src.cl_tsmnet.ms_tgc_spddsbn import DomainBatchNorm1d
from src.cl_tsmnet.training import (
    _mstgc_architecture_name, build_ms_tgc_spddsbn, train_one_split,
)

ROOT = str(Path(__file__).resolve().parents[1])
CASES = [
    ("mstgc_mean_ce", "mean", None, False, 64),
    ("mstgc_dta_cheb_eudsbn", "mean", "eudsbn", True, 64),
    ("mstgc_dta_cheb_spdbn", "augmented", "spdbn", False, 210),
    ("ms_tgc_spddsbn", "augmented", "spddsbn", True, 210),
]


def test_eudsbn_refit_replaces_moments_and_preserves_other_state():
    layer = DomainBatchNorm1d(2, [0, 1]).eval()
    values = torch.tensor([[8., 16.], [10., 20.], [12., 24.]])
    before = {key: value.clone() for key, value in layer.state_dict().items()}
    layer.refit_domain_stats(values, torch.ones(3, dtype=torch.long))
    target = layer.layers["1"]
    torch.testing.assert_close(target.running_mean, values.mean(0))
    torch.testing.assert_close(target.running_var, values.var(0))
    expected = (values - values.mean(0)) / torch.sqrt(values.var(0) + target.eps)
    torch.testing.assert_close(layer(values, torch.ones(3, dtype=torch.long)), expected)
    assert not layer.training and not target.training
    assert target.momentum == 0.1
    for key, value in layer.state_dict().items():
        if key.startswith("layers.0.") or key.endswith(("weight", "bias")):
            torch.testing.assert_close(value, before[key])
    layer.refit_domain_stats(values + 5, torch.ones(3, dtype=torch.long))
    torch.testing.assert_close(target.running_mean, values.mean(0) + 5)


@pytest.mark.parametrize("variant,representation,normalization,adapted,dimension", CASES)
def test_default_model_features_are_the_actual_readout_input(
        variant, representation, normalization, adapted, dimension):
    torch.manual_seed(42)
    model = build_ms_tgc_spddsbn(
        ROOT, nchannels=14, nsamples=128, nclasses=2,
        domains=np.array([0, 1]), variant=variant,
    ).eval()
    windows, domains = torch.randn(4, 14, 128), torch.tensor([0, 0, 1, 1])
    captured = []
    hook = model.readout.register_forward_pre_hook(
        lambda module, inputs: captured.append(inputs[0].detach().clone())
    )
    with torch.no_grad():
        logits = model(windows, domains)
        features, metadata = model.extract_alignment_representation(windows, domains)
    hook.remove()
    torch.testing.assert_close(features, captured[0])
    assert features.shape == (4, dimension)
    assert logits.shape == (4, 2) and torch.isfinite(logits).all()
    assert metadata["representation"] == representation
    assert metadata["normalization"] == normalization
    assert model.temporal_mode == "dta_gate"
    assert model.temporal.kernel_sizes == [17, 9, 5]
    assert model.graph.graph_mode == "adaptive" and model.graph.cheby.order == 3
    assert model.channel_score is not None
    assert _mstgc_architecture_name(model) == (
        "shared_channel_graph_mean_v3" if representation == "mean"
        else "shared_channel_graph_augmented_spd_v3"
    )
    assert (model.spd_branch is not None) == (representation == "augmented")
    if model.spd_branch is not None:
        assert model.spd_branch.spd_input_dim == 65
        assert model.spd_branch.subspacedims == 20


@pytest.mark.parametrize("variant,representation,normalization,adapted,dimension", CASES)
def test_training_saves_expected_architecture_and_adaptation(
        tmp_path, variant, representation, normalization, adapted, dimension):
    rng = np.random.RandomState(7)
    dataset = {
        "x": rng.randn(16, 4, 32).astype(np.float32),
        "y": np.tile([0, 1], 8), "fs": 128.,
        "channels": ["F3", "F4", "P3", "P4"],
    }
    domains = np.repeat([0, 1, 2, 3], 4)
    split = {"train": np.arange(8), "val": np.arange(8, 12), "test": np.arange(12, 16)}
    with patch("torch.cuda.is_available", return_value=False):
        result = train_one_split(
            dataset, domains, split, ROOT, output_dir=str(tmp_path),
            model_type=variant, epochs=1, batch_size=4, refit_batch_size=2,
            mstgc_temporal_hidden=4, mstgc_graph_hidden=4, mstgc_fusion_dim=6,
            mstgc_num_heads=2, mstgc_time_points=8, subspacedims=3,
            mstgc_dropout=0., augment=False,
        )
    assert result["mstgc_representation"] == representation
    assert result["mstgc_architecture"] == (
        "shared_channel_graph_mean_v3" if representation == "mean"
        else "shared_channel_graph_augmented_spd_v3"
    )
    assert result["target_adapt"] == adapted
    assert result["val_stat_refit"] == adapted
    assert result["target_refit_scope"] == ("target_only" if adapted else "none")
    state = torch.load(tmp_path / "model.pt", weights_only=True)
    assert any(key.startswith("spd_branch.") for key in state) == (representation == "augmented")
    if normalization == "eudsbn":
        assert state["eudsbnorm.layers.2.num_batches_tracked"].item() == 0
        assert state["eudsbnorm.layers.3.num_batches_tracked"].item() == 1
