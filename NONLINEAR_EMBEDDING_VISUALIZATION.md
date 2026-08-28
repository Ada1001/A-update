# Stable UMAP and Joint t-SNE Supplement

`analysis/plot_nonlinear_alignment_embeddings.py` generates nonlinear
visualization supplements from one representative LOSO checkpoint. It does not
replace the AIRM/Fréchet analysis and does not import the PCA or MDS scripts.

## Feature Contract

- `P_pre`: output after `BiMap + ReEig`, immediately before SPDDSBN;
- `P_post`: SPDDSBN output, immediately before `LogEig`;
- pre/post rows have the same sample IDs and order;
- the common AIRM tangent reference is fitted from source-train `P_pre` only;
- the feature standardizer is fitted from source-train pre tangent vectors only;
- target labels are used after training only for balanced sampling, plotting,
  and class-conditional descriptive metrics.

The representative target is the subject whose LOSO Balanced Accuracy is
closest to the across-subject median. Multiple result rows for one subject are
averaged first, and subject ID breaks an exact tie.

## Installation

UMAP is an optional analysis dependency:

```bash
python -m pip install -r requirements-analysis.txt
```

## Main Command

For the complete MS_TGC_SPDDSBN model on STEW:

```bash
python analysis/plot_nonlinear_alignment_embeddings.py \
  --dataset stew \
  --model ms_tgc_spddsbn \
  --output-root outputs \
  --output-dir results/figures/nonlinear_embeddings/stew_ms_tgc
```

The original LOSO summary, matching master-summary row, raw window cache, and
selected `subject_XX/model.pt` are resolved automatically. No downloaded ZIP or
separately fitted PCA/MDS coordinates are used.

For current target-only TSMNet-SPDDSBN:

```bash
python analysis/plot_nonlinear_alignment_embeddings.py \
  --dataset stew \
  --model tsmnet \
  --output-root outputs_target_only \
  --master-summary outputs_target_only/master_summary.csv \
  --output-dir results/figures/nonlinear_embeddings/stew_tsmnet_target_only
```

Historical TSMNet checkpoints without `target_refit_scope` used unlabeled
source+target statistics. They have no target-label leakage but are not the
current target-only protocol. Such a checkpoint requires the explicit
`--allow-legacy-source-target-refit` option, and the limitation is written to
the metrics JSON.

## Embedding Protocol

The main UMAP is fixed to seed 2026 with `n_neighbors=15`, `min_dist=0.15`, and
Euclidean distance. UMAP is fitted only on all source-pre features. The same
fitted reducer transforms the balanced source-pre, target-pre, source-post, and
target-post samples. Seeds 2026-2030 are all run and saved; seed 2026 is always
the main figure.

t-SNE receives one ordered matrix containing `[balanced pre; paired balanced
post]`. It is fitted once with PCA initialization, automatic learning rate,
2,000 iterations, and seed 2026. The legal perplexity is derived from the
actual joint sample count. Its figure metadata states: “t-SNE was jointly
fitted to paired pre- and post-alignment representations.”

## Evidence and Outputs

The metrics JSON reports trustworthiness, kNN overlap, fixed-seed UMAP
stability, high-dimensional class-conditional domain distances, and the
high-dimensional Fisher ratio. Formal conclusions must use these
high-dimensional values or the separate AIRM/Fréchet results, not distances in
the two-dimensional plots.

Outputs are isolated under the requested nonlinear directory:

```text
umap_coordinates_seed-*.csv
tsne_joint_coordinates_seed-2026.csv
nonlinear_embedding_metrics.json
plot_sample_manifest.csv
nonlinear_embedding_features.npz
fig_umap_alignment.pdf/svg/png
fig_tsne_joint_alignment.pdf/svg/png
fig_nonlinear_stability.pdf/svg/png
```

PNG files use 600 dpi. PDF and SVG remain vector outputs. Before/after panels
share coordinate limits, classes use color, source/target use marker shape, and
target points retain the highest point-layer visibility.
