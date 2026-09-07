# Fig. 6 Analysis of Learned Channel Interaction Patterns

`analysis/fig6_graph_pattern_analysis.py` independently produces the graph and
node-response figure. It does not load or overwrite PCA, MDS, UMAP, t-SNE, or
Fig. 5 coordinates.

## Relationship to earlier figures

| Figure code | Evidence represented |
|---|---|
| `visualize_spddsbn_pca.py` | linear pre/post SPDDSBN representation alignment |
| `analysis/plot_riemannian_mds.py` | AIRM geometry of subject-class Frechet centers |
| `analysis/plot_nonlinear_alignment_embeddings.py` | nonlinear pre/post representation supplement |
| `analysis/fig5_representation_alignment.py` | four-method representation/calibration comparison |
| `analysis/fig6_graph_pattern_analysis.py` | learned channel interaction topology, response, and stability |

## Four panels

1. **Mean learned adjacency.** The actual sparse adaptive adjacency from every
   completed LOSO checkpoint is symmetrized, max-normalized per fold, and then
   averaged. Channel order is preserved.
2. **Stable scalp interactions and node strength.** An edge is drawn only when
   it belongs to the top 10% of mean weights and is nonzero in at least 70% of
   folds. The script does not relax these thresholds when no edge survives.
   Node strength is the row sum of the mean interaction matrix.
3. **Task-related node response.** For the target subject whose BAcc is closest
   to the LOSO median, node response is the RMS Chebyshev output before channel
   reliability weighting and SPD pooling. It is standardized by source-train
   channel statistics, then grouped by target class after training. All class
   maps share one color range.
4. **Stability comparison.** A sample-wise dynamic graph is the absolute
   Pearson correlation of each window's shared temporal channel maps, sparsified
   to exactly the same edge count as that fold's learned graph. Each LOSO fold
   contributes one dynamic stability score and one shared-graph reproducibility
   score for both edge Jaccard and weighted cosine similarity.

The term **learned channel interaction** is used throughout. The figure does
not claim that learned weights are physiological functional connectivity.

## Interpretation boundary

The shared graph is fixed across windows by model design. Its reported
stability is therefore **across-fold reproducibility**, while dynamic-graph
stability is **within-fold across-window consistency**. The comparison supports
a stability claim but does not by itself prove higher classification accuracy
or neurophysiological connectivity. If either shared metric is not higher, the
script warns and retains the result without changing thresholds.

Target labels are used only after training to balance the analyzed windows and
to form the final class-conditioned response maps. They are not used to train
or reconstruct the model, learn the adjacency, choose stable edges, calculate
correlation weights, or fit any normalization statistic.

## MS-TGC commands

Generate one STEW figure at the exact requested output names:

```bash
python analysis/fig6_graph_pattern_analysis.py \
  --datasets stew \
  --dataset-labels STEW \
  --model ms_tgc_spddsbn \
  --output-root outputs \
  --master-summary outputs/master_summary.csv \
  --output-dir results
```

Process all three datasets in one command. Because their channel counts differ,
the script creates one complete Fig. 6 per dataset:

```bash
python analysis/fig6_graph_pattern_analysis.py \
  --datasets stew,eegmat,cog-bci:nback \
  --dataset-labels STEW,EEGMAT,N-Back \
  --model ms_tgc_spddsbn \
  --output-root outputs \
  --master-summary outputs/master_summary.csv \
  --output-dir results/fig6_mstgc
```

The multi-dataset outputs are stored under `results/fig6_mstgc/stew/`,
`results/fig6_mstgc/eegmat/`, and `results/fig6_mstgc/cog-bci-nback/`.

## TSMNet reference mode

TSMNet has no adjacency matrix or graph message passing. The supported TSMNet
mode derives a spatial-filter channel-similarity **proxy** from its learned
spatial convolution weights and derives channel responses from temporal
activations weighted by channel-kernel norm. Every title and metadata record
identifies this distinction; it must not be reported as an adaptive graph.

```bash
python analysis/fig6_graph_pattern_analysis.py \
  --datasets stew \
  --dataset-labels STEW \
  --model tsmnet \
  --bnorm spddsbn \
  --output-root outputs \
  --output-dir results/fig6_tsmnet
```

## Outputs

```text
fig6_graph_pattern_analysis.pdf
fig6_graph_pattern_analysis.png
fig6_graph_pattern_analysis.svg
fig6_mean_adjacency.npy
fig6_fold_adjacencies.npz
fig6_edge_statistics.csv
fig6_stability_by_fold.csv
fig6_node_responses.csv
fig6_channel_positions.csv
fig6_graph_pattern_metadata.json
```

The PNG and all rasterized scatter/edge content are saved at 600 dpi. PDF and
SVG remain the preferred manuscript formats. Repeated extraction reuses a
checkpoint-signature-validated cache unless `--force-reextract` is passed.
