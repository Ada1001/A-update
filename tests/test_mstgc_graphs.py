import unittest

import numpy as np
import torch

from src.cl_tsmnet.datasets import STEW_CHANNELS
from src.cl_tsmnet.ms_tgc_spddsbn import MSTGCSPDDSBN
from src.cl_tsmnet.ms_tgc_spddsbn import ChebyGraphSequenceLayer
from src.cl_tsmnet.mdtn_gmda import ChebyNetLayer
from src.cl_tsmnet.training import (
    _mstgc_spatial_prior,
    build_mstgc_graph_adjacencies,
    fit_source_normalizer,
)


class MSTGCGraphTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(13)
        self.dataset = {
            "x": rng.randn(10, len(STEW_CHANNELS), 128).astype(np.float32),
            "y": np.tile(np.asarray([0, 1]), 5),
            "channels": list(STEW_CHANNELS),
            "fs": 128.0,
        }
        self.train_indices = np.arange(6, dtype=np.int64)
        self.normalizer = fit_source_normalizer(
            self.dataset["x"], self.train_indices
        )

    def test_spatial_prior_is_symmetric_and_sparse(self):
        adjacency = _mstgc_spatial_prior(STEW_CHANNELS, neighbors=4)
        self.assertEqual(adjacency.shape, (14, 14))
        self.assertTrue(np.all(np.isfinite(adjacency)))
        self.assertTrue(np.allclose(adjacency, adjacency.T))
        self.assertTrue(np.allclose(np.diag(adjacency), 0.0))

    def test_plv_graph_uses_source_train_windows_only(self):
        first = build_mstgc_graph_adjacencies(
            self.dataset, self.train_indices, self.normalizer,
            graph_mode="multigraph", neighbors=4,
        )
        changed = dict(self.dataset)
        changed["x"] = self.dataset["x"].copy()
        changed["x"][6:] = 1e6
        second = build_mstgc_graph_adjacencies(
            changed, self.train_indices, self.normalizer,
            graph_mode="multigraph", neighbors=4,
        )
        self.assertTrue(np.array_equal(first, second))

    def test_multigraph_forward_and_backward(self):
        adjacencies = build_mstgc_graph_adjacencies(
            self.dataset, self.train_indices, self.normalizer,
            graph_mode="multigraph", neighbors=4,
        )
        model = MSTGCSPDDSBN(
            spd_branch=None,
            spd_latent_dim=0,
            nchannels=len(STEW_CHANNELS),
            nclasses=2,
            temporal_hidden=8,
            graph_hidden=8,
            fusion_dim=8,
            kernel_length=8,
            num_heads=2,
            cheby_order=2,
            dropout=0.0,
            graph_mode="multigraph",
            graph_adjacencies=adjacencies,
            graph_neighbors=4,
        )
        windows = self.normalizer.transform_array(
            self.dataset["x"][self.train_indices]
        )
        logits = model(
            torch.from_numpy(windows).float(),
            torch.zeros(len(windows), dtype=torch.long),
        )
        self.assertEqual(tuple(logits.shape), (len(windows), 2))
        logits.sum().backward()
        self.assertIsNotNone(model.graph.graph_logits.grad)

    def test_temporal_backbone_preserves_channel_and_time_axes(self):
        model = MSTGCSPDDSBN(
            spd_branch=None,
            spd_latent_dim=0,
            nchannels=len(STEW_CHANNELS),
            nclasses=2,
            temporal_hidden=8,
            graph_hidden=8,
            fusion_dim=8,
            kernel_length=8,
            num_heads=2,
            cheby_order=2,
            dropout=0.0,
            graph_mode="adaptive",
            graph_neighbors=4,
        )
        windows = torch.from_numpy(
            self.normalizer.transform_array(self.dataset["x"][:3])
        ).float()
        temporal_maps, scale_weights = model.temporal(windows)
        self.assertEqual(tuple(temporal_maps.shape), (3, 14, 8, 64))
        self.assertEqual(tuple(scale_weights.shape), (3 * 14, 3, 1))
        self.assertEqual(model.temporal.branches[0][0].in_channels, 1)
        graph_maps = model.graph(temporal_maps)
        self.assertEqual(tuple(graph_maps.shape), (3, 14, 8, 64))

    def test_sequence_chebyshev_matches_reference_layer(self):
        torch.manual_seed(5)
        reference = ChebyNetLayer(order=3, in_features=4, out_features=6)
        sequence = ChebyGraphSequenceLayer(
            order=3, in_features=4, out_features=6
        )
        with torch.no_grad():
            sequence.weight.copy_(reference.weight)
            sequence.bias.copy_(reference.bias)
        features = torch.randn(2, 5, 4, 7)
        adjacency = torch.rand(5, 5)
        adjacency = 0.5 * (adjacency + adjacency.t())
        degree = torch.sum(adjacency, dim=1)
        inv_sqrt = torch.diag(torch.pow(degree + 1e-5, -0.5))
        laplacian = -torch.mm(torch.mm(inv_sqrt, adjacency), inv_sqrt)
        actual = sequence(features, laplacian)
        flat = features.permute(0, 3, 1, 2).reshape(2 * 7, 5, 4)
        expanded = laplacian.unsqueeze(0).expand(flat.shape[0], -1, -1)
        expected = reference(flat, expanded)
        expected = expected.reshape(2, 7, 5, 6).permute(0, 2, 3, 1)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
