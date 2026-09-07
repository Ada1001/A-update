import os
import tempfile
import unittest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.fig6_graph_pattern_analysis import (
    _dynamic_graphs,
    _save_figure,
    _stable_edges,
    compute_edge_stability,
    compute_mean_adjacency,
    compute_weight_stability,
    load_adjacency_matrices,
    plot_adjacency_heatmap,
    plot_graph_stability,
    plot_node_response_maps,
    plot_node_strength_topomap,
    plot_scalp_graph,
)


class Fig6GraphPatternAnalysisTests(unittest.TestCase):
    def _adjacencies(self):
        first = np.asarray([
            [0, 1.0, 0.5, 0],
            [1.0, 0, 0, 0.3],
            [0.5, 0, 0, 0.8],
            [0, 0.3, 0.8, 0],
        ], dtype=np.float64)
        second = first.copy()
        second[0, 2] = second[2, 0] = 0.0
        second[0, 3] = second[3, 0] = 0.5
        return np.stack([first, second])

    def test_mean_adjacency_is_symmetric_normalized_and_zero_diagonal(self):
        mean = compute_mean_adjacency(self._adjacencies())
        self.assertTrue(np.allclose(mean, mean.T))
        self.assertTrue(np.allclose(np.diag(mean), 0.0))
        self.assertLessEqual(float(mean.max()), 1.0)
        self.assertGreater(float(mean.max()), 0.0)

    def test_edge_and_weight_stability_have_expected_limits(self):
        first = self._adjacencies()[0]
        identical = np.stack([first, first])
        self.assertAlmostEqual(compute_edge_stability(identical)[0], 1.0)
        self.assertAlmostEqual(compute_weight_stability(identical)[0], 1.0)

        left = np.zeros((4, 4), dtype=np.float64)
        right = np.zeros((4, 4), dtype=np.float64)
        left[0, 1] = left[1, 0] = 1.0
        right[2, 3] = right[3, 2] = 1.0
        disjoint = np.stack([left, right])
        self.assertAlmostEqual(compute_edge_stability(disjoint)[0], 0.0)
        self.assertAlmostEqual(compute_weight_stability(disjoint)[0], 0.0)

    def test_dynamic_graphs_match_prescribed_edge_count(self):
        rng = np.random.RandomState(21)
        temporal = rng.normal(size=(6, 4, 3, 8)).astype(np.float32)
        graphs = _dynamic_graphs(temporal, edge_count=3)
        counts = [np.count_nonzero(np.triu(graph, k=1)) for graph in graphs]
        self.assertEqual(graphs.shape, (6, 4, 4))
        self.assertEqual(counts, [3] * 6)
        self.assertTrue(np.allclose(graphs, np.swapaxes(graphs, 1, 2)))

    def test_adjacency_loader_validates_channel_order(self):
        matrices = self._adjacencies()
        channels = np.asarray(["F3", "F4", "P3", "P4"])
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index, matrix in enumerate(matrices):
                path = os.path.join(directory, "fold_{}.npz".format(index))
                np.savez_compressed(path, adjacency=matrix, channels=channels)
                paths.append(path)
            loaded, names = load_adjacency_matrices(paths, channels)
        self.assertEqual(loaded.shape, matrices.shape)
        self.assertEqual(names, channels.tolist())

    def test_prespecified_stable_edge_rule_is_not_relaxed(self):
        matrices = self._adjacencies()
        _, prevalence, mask = _stable_edges(
            matrices, fraction=0.5, prevalence_threshold=1.0
        )
        self.assertTrue(np.all(prevalence[mask] >= 1.0))
        self.assertTrue(np.array_equal(mask, mask.T))

    def test_publication_panels_and_writers_render(self):
        adjacency = compute_mean_adjacency(self._adjacencies())
        mask = adjacency > 0.7
        np.fill_diagonal(mask, False)
        positions = pd.DataFrame({
            "channel": ["F3", "F4", "P3", "P4"],
            "x": [-0.45, 0.45, -0.35, 0.35],
            "y": [0.55, 0.55, -0.55, -0.55],
        })
        strength = adjacency.sum(axis=1)
        stability = pd.DataFrame({
            "subject": [1, 2, 1, 2],
            "graph_type": ["Dynamic graph", "Dynamic graph",
                           "Shared adaptive graph", "Shared adaptive graph"],
            "edge_jaccard": [0.3, 0.4, 0.8, 0.9],
            "weight_cosine": [0.4, 0.5, 0.85, 0.95],
        })
        with tempfile.TemporaryDirectory() as directory:
            fig = plt.figure(figsize=(7.16, 5.25))
            grid = fig.add_gridspec(2, 3)
            plot_adjacency_heatmap(
                fig.add_subplot(grid[0, 0]), adjacency,
                positions["channel"].tolist(), "Mean learned adjacency matrix",
            )
            plot_scalp_graph(
                fig.add_subplot(grid[0, 1]), adjacency, mask,
                positions, strength,
            )
            plot_node_strength_topomap(
                fig.add_subplot(grid[0, 2]), positions, strength
            )
            plot_node_response_maps(
                grid[1, :2], positions,
                np.stack([strength - strength.mean(), strength.mean() - strength]),
                ["Low", "High"], fig,
            )
            plot_graph_stability(fig.add_subplot(grid[1, 2]), stability)
            paths = _save_figure(fig, directory)
            plt.close(fig)
            self.assertTrue(all(os.path.exists(path) for path in paths))


if __name__ == "__main__":
    unittest.main()
