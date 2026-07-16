import os
import unittest

import numpy as np
import torch

from src.cl_tsmnet.datasets import STEW_CHANNELS
from src.cl_tsmnet.ms_tgc_spddsbn import GraphSPDManifoldHead, MSTGCSPDDSBN
from src.cl_tsmnet.ms_tgc_spddsbn import ChebyGraphSequenceLayer
from src.cl_tsmnet.mdtn_gmda import ChebyNetLayer
from src.cl_tsmnet.training import (
    _mstgc_spatial_prior,
    build_mstgc_graph_adjacencies,
    build_ms_tgc_spddsbn,
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

        weighted_maps, channel_weights = model._weighted_graph_maps(
            windows, return_weights=True
        )
        self.assertEqual(tuple(weighted_maps.shape), (3, 14, 8, 64))
        self.assertEqual(tuple(channel_weights.shape), (3, 14, 1))
        self.assertTrue(torch.allclose(
            channel_weights.sum(dim=1), torch.ones(3, 1), atol=1e-6
        ))

    def test_augmented_spd_is_positive_definite_and_uses_feature_plus_one(self):
        from src.cl_tsmnet.training import _ensure_tsmnet_on_path

        _ensure_tsmnet_on_path(os.getcwd())
        head = GraphSPDManifoldHead(
            feature_dim=8, subspacedims=4, bnorm=None,
            domains=[0, 1], shrinkage=0.1,
        )
        maps = torch.randn(3, 14, 8, 16)
        augmented = head.build_augmented_spd(maps)
        self.assertEqual(tuple(augmented.shape), (3, 1, 9, 9))
        self.assertTrue(torch.allclose(
            augmented, augmented.transpose(-1, -2), atol=1e-10
        ))
        eigenvalues = torch.linalg.eigvalsh(augmented[:, 0])
        self.assertGreater(float(eigenvalues.min()), 0.0)
        latent = head(maps, torch.tensor([0, 0, 1]))
        self.assertEqual(tuple(latent.shape), (3, 10))

    def test_full_augmented_spd_model_backpropagates_through_shared_backbone(self):
        model = build_ms_tgc_spddsbn(
            os.getcwd(), nchannels=14, nsamples=128, nclasses=2,
            domains=np.asarray([0, 1]), temporal_hidden=8,
            graph_hidden=8, fusion_dim=12, kernel_length=8,
            num_heads=2, cheby_order=2, dropout=0.0,
            graph_time_points=16, subspacedims=4,
            covariance_shrinkage=0.1, variant="ms_tgc_spddsbn",
        )
        windows = torch.randn(4, 14, 128)
        domains = torch.tensor([0, 0, 1, 1])
        logits = model(windows, domains)
        self.assertEqual(tuple(logits.shape), (4, 2))
        torch.nn.functional.cross_entropy(
            logits, torch.tensor([0, 1, 0, 1])
        ).backward()
        self.assertIsNotNone(model.temporal.branches[0][0].weight.grad)
        self.assertIsNotNone(model.graph.cheby.weight.grad)
        self.assertIsNotNone(model.spd_branch.spdnet[0].W.grad)

    def test_default_manifold_dimensions_are_65_to_20_and_210_to_128(self):
        model = build_ms_tgc_spddsbn(
            os.getcwd(), nchannels=14, nsamples=128, nclasses=2,
            domains=np.asarray([0, 1]), variant="ms_tgc_spddsbn",
        )
        self.assertEqual(model.spd_branch.augmented_dim, 65)
        self.assertEqual(tuple(model.spd_branch.spdnet[0].W.shape), (1, 65, 20))
        self.assertEqual(model.spd_branch.latent_dim, 210)
        self.assertEqual(model.readout[1].in_features, 210)
        self.assertEqual(model.readout[1].out_features, 128)

    def test_all_ablation_variants_share_compatible_forward_contract(self):
        variants = [
            "mstgc_mean_ce", "mstgc_cov_spddsbn",
            "mstgc_augspd_spddsbn",
            "mstgc_wo_channel_attention",
            "mstgc_dta_ce", "mstgc_dta_cheb_ce",
            "mstgc_dta_cheb_eudsbn", "mstgc_dta_cheb_spdmbn",
            "mstgc_dta_cheb_spdbn", "ms_tgc_spddsbn",
            "mstgc_wo_dta", "mstgc_wo_cheb", "mstgc_wo_spddsbn",
        ]
        windows = torch.randn(4, 4, 32)
        domains = torch.tensor([0, 0, 1, 1])
        for variant in variants:
            model = build_ms_tgc_spddsbn(
                os.getcwd(), nchannels=4, nsamples=32, nclasses=3,
                domains=np.asarray([0, 1]), temporal_hidden=4,
                graph_hidden=4, fusion_dim=6, kernel_length=4,
                num_heads=2, cheby_order=2, dropout=0.0,
                graph_time_points=8, subspacedims=3,
                covariance_shrinkage=0.1, variant=variant,
            )
            with self.subTest(variant=variant):
                self.assertEqual(tuple(model(windows, domains).shape), (4, 3))

    def test_without_channel_attention_uses_fixed_uniform_reliability(self):
        model = build_ms_tgc_spddsbn(
            os.getcwd(), nchannels=4, nsamples=32, nclasses=2,
            domains=np.asarray([0, 1]), temporal_hidden=4,
            graph_hidden=4, fusion_dim=6, kernel_length=4,
            num_heads=2, cheby_order=2, dropout=0.0,
            graph_time_points=8, subspacedims=3,
            variant="mstgc_wo_channel_attention",
        )
        _, weights = model._weighted_graph_maps(
            torch.randn(3, 4, 32), return_weights=True
        )
        self.assertFalse(model.use_channel_attention)
        self.assertIsNone(model.channel_score)
        self.assertTrue(torch.allclose(
            weights, torch.full_like(weights, 0.25)
        ))

    def test_representation_aliases_and_spd_input_dimensions(self):
        common = dict(
            project_root=os.getcwd(), nchannels=4, nsamples=32, nclasses=2,
            domains=np.asarray([0, 1]), temporal_hidden=4, graph_hidden=4,
            fusion_dim=6, kernel_length=4, num_heads=2, cheby_order=2,
            dropout=0.0, graph_time_points=8, subspacedims=3,
            covariance_shrinkage=0.1,
        )
        windows = torch.randn(4, 4, 32)
        domains = torch.tensor([0, 0, 1, 1])

        torch.manual_seed(31)
        mean_alias = build_ms_tgc_spddsbn(
            variant="mstgc_mean_ce", **common
        ).eval()
        torch.manual_seed(31)
        mean_original = build_ms_tgc_spddsbn(
            variant="mstgc_dta_cheb_ce", **common
        ).eval()
        self.assertTrue(torch.allclose(
            mean_alias(windows, domains), mean_original(windows, domains)
        ))

        torch.manual_seed(37)
        augmented_alias = build_ms_tgc_spddsbn(
            variant="mstgc_augspd_spddsbn", **common
        ).eval()
        torch.manual_seed(37)
        full = build_ms_tgc_spddsbn(
            variant="ms_tgc_spddsbn", **common
        ).eval()
        self.assertTrue(torch.allclose(
            augmented_alias(windows, domains), full(windows, domains)
        ))

        covariance = build_ms_tgc_spddsbn(
            variant="mstgc_cov_spddsbn", **common
        )
        self.assertEqual(covariance.representation, "covariance")
        self.assertEqual(covariance.spd_branch.spd_input_dim, 4)
        self.assertEqual(tuple(covariance.spd_branch.spdnet[0].W.shape), (1, 4, 3))
        self.assertEqual(full.representation, "augmented")
        self.assertEqual(full.spd_branch.spd_input_dim, 5)
        self.assertEqual(tuple(full.spd_branch.spdnet[0].W.shape), (1, 5, 3))
        self.assertEqual(covariance.spd_branch.latent_dim, full.spd_branch.latent_dim)

    def test_graph_parameters_follow_the_declared_update_policy(self):
        torch.manual_seed(41)
        features = torch.randn(4, 14, 8, 16)

        adaptive = MSTGCSPDDSBN(
            spd_branch=None, spd_latent_dim=0, nchannels=14, nclasses=2,
            temporal_hidden=8, graph_hidden=8, fusion_dim=8,
            kernel_length=8, num_heads=2, cheby_order=2, dropout=0.0,
            graph_mode="adaptive", graph_neighbors=4,
        )
        before_adaptive = adaptive.graph.adj_param.detach().clone()
        optimizer = torch.optim.SGD(adaptive.parameters(), lr=0.05)
        optimizer.zero_grad()
        adaptive.graph(features).square().mean().backward()
        optimizer.step()
        self.assertFalse(torch.equal(before_adaptive, adaptive.graph.adj_param))

        graphs = build_mstgc_graph_adjacencies(
            self.dataset, self.train_indices, self.normalizer,
            graph_mode="multigraph", neighbors=4,
        )
        multigraph = MSTGCSPDDSBN(
            spd_branch=None, spd_latent_dim=0, nchannels=14, nclasses=2,
            temporal_hidden=8, graph_hidden=8, fusion_dim=8,
            kernel_length=8, num_heads=2, cheby_order=2, dropout=0.0,
            graph_mode="multigraph", graph_adjacencies=graphs,
            graph_neighbors=4,
        )
        before_graphs = multigraph.graph.fixed_adjacencies.detach().clone()
        before_logits = multigraph.graph.graph_logits.detach().clone()
        before_cheby = multigraph.graph.cheby.weight.detach().clone()
        optimizer = torch.optim.SGD(multigraph.parameters(), lr=0.05)
        optimizer.zero_grad()
        multigraph.graph(features).square().mean().backward()
        optimizer.step()
        self.assertTrue(torch.equal(before_graphs, multigraph.graph.fixed_adjacencies))
        self.assertFalse(torch.equal(before_logits, multigraph.graph.graph_logits))
        self.assertFalse(torch.equal(before_cheby, multigraph.graph.cheby.weight))
        self.assertEqual(len([
            module for module in multigraph.graph.modules()
            if isinstance(module, ChebyGraphSequenceLayer)
        ]), 1)

    def test_graph_source_variants_keep_the_same_augmented_spd_head(self):
        graphs = build_mstgc_graph_adjacencies(
            self.dataset, self.train_indices, self.normalizer,
            graph_mode="multigraph", neighbors=4,
        )
        variants = [
            ("mstgc_graph_prior", "prior", graphs[:1]),
            ("mstgc_graph_plv", "plv", graphs[1:2]),
            ("mstgc_graph_multigraph", "multigraph", graphs),
        ]
        windows = torch.from_numpy(
            self.normalizer.transform_array(self.dataset["x"][:4])
        ).float()
        domains = torch.tensor([0, 0, 1, 1])
        for variant, mode, adjacencies in variants:
            model = build_ms_tgc_spddsbn(
                os.getcwd(), nchannels=14, nsamples=128, nclasses=2,
                domains=np.asarray([0, 1]), temporal_hidden=8,
                graph_hidden=8, fusion_dim=12, kernel_length=8,
                num_heads=2, cheby_order=2, dropout=0.0,
                graph_time_points=16, subspacedims=4,
                covariance_shrinkage=0.1, variant=variant,
                graph_mode=mode, graph_adjacencies=adjacencies,
            )
            with self.subTest(variant=variant):
                self.assertEqual(model.spd_branch.augmented_dim, 9)
                self.assertEqual(tuple(model(windows, domains).shape), (4, 2))

    def test_unlabelled_domain_refit_contract_for_spd_and_euclidean_dsbn(self):
        windows = torch.randn(4, 4, 32)
        domains = torch.tensor([0, 0, 1, 1])
        labels_are_ignored = torch.tensor([0, 1, 0, 1])
        for variant in [
                "ms_tgc_spddsbn", "mstgc_cov_spddsbn",
                "mstgc_augspd_spddsbn", "mstgc_dta_cheb_eudsbn"]:
            model = build_ms_tgc_spddsbn(
                os.getcwd(), nchannels=4, nsamples=32, nclasses=2,
                domains=np.asarray([0, 1]), temporal_hidden=4,
                graph_hidden=4, fusion_dim=6, kernel_length=4,
                num_heads=2, cheby_order=2, dropout=0.0,
                graph_time_points=8, subspacedims=3,
                covariance_shrinkage=0.1, variant=variant,
            )
            model.domainadapt_finetune(
                windows, labels_are_ignored, domains, target_domains=[1]
            )
            with self.subTest(variant=variant):
                self.assertEqual(tuple(model(windows, domains).shape), (4, 2))

    def test_batched_spd_refit_matches_full_refit_and_limits_gpu_batches(self):
        common = dict(
            project_root=os.getcwd(), nchannels=4, nsamples=32, nclasses=2,
            domains=np.asarray([0, 1]), temporal_hidden=4,
            graph_hidden=4, fusion_dim=6, kernel_length=4,
            num_heads=2, cheby_order=2, dropout=0.0,
            graph_time_points=8, subspacedims=3,
            covariance_shrinkage=0.1, variant="mstgc_cov_spddsbn",
        )
        torch.manual_seed(53)
        full = build_ms_tgc_spddsbn(**common)
        torch.manual_seed(53)
        batched = build_ms_tgc_spddsbn(**common)
        windows = torch.randn(12, 4, 32)
        domains = torch.tensor([0] * 6 + [1] * 6)
        ignored_labels = torch.full((12,), -1, dtype=torch.long)

        full.domainadapt_finetune(
            windows, ignored_labels, domains, target_domains=[1]
        )
        seen_batches = []
        hook = batched.temporal.register_forward_pre_hook(
            lambda module, inputs: seen_batches.append(int(inputs[0].shape[0]))
        )
        batched.domainadapt_finetune_batched(
            windows, domains, target_domains=[1], batch_size=3
        )
        hook.remove()
        self.assertEqual(sum(seen_batches), len(windows))
        self.assertLessEqual(max(seen_batches), 3)

        full_state = full.state_dict()
        batched_state = batched.state_dict()
        test_stat_keys = [
            key for key in full_state
            if "running_mean_test" in key or "running_var_test" in key
        ]
        self.assertTrue(test_stat_keys)
        for key in test_stat_keys:
            self.assertTrue(torch.allclose(
                full_state[key], batched_state[key], atol=1e-7, rtol=1e-6
            ), msg=key)

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
