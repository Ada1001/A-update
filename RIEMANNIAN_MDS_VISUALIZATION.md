# AIRM Riemannian MDS Visualization

`analysis/plot_riemannian_mds.py` independently generates the joint
before/after SPDDSBN Riemannian MDS figures. It does not import the PCA
visualization script and never fits separate embeddings for the two stages.

The analysis uses the true model locations:

- `P_pre`: `BiMap + ReEig` output, immediately before SPDDSBN;
- `P_post`: SPDDSBN output, immediately before `LogEig`.

It may reuse a validated `spd_intermediates.npz` cache. If that cache is
missing, it reconstructs the LOSO split, fits the EEG normalizer on source
training windows only, strictly loads the selected best checkpoint, and
extracts both stages itself.

For every source-training subject and the target-test subject, one AIRM
Frechet mean is computed per class and stage. Source class reference centers
are then computed from the subject centers, giving every source subject equal
weight. All subject centers, source references, and both stages enter one AIRM
distance matrix and one metric MDS fit with seed 2026. Extra fixed seeds are
used only for stability reporting.

Target labels are used after model training only for class-conditional center
construction, plotting, and descriptive target-to-source distances. They do
not affect model parameters, SPDDSBN statistics, fold selection, or MDS seed
selection.

## Validated Feature Cache

```bash
python analysis/plot_riemannian_mds.py \
  --dataset stew \
  --model ms_tgc_spddsbn \
  --target-subject 6 \
  --feature-cache-dir outputs/_riemannian_mds_input/stew \
  --output-dir results/figures/riemannian_mds/stew_ms_tgc
```

`--target-subject` is required when the machine does not have the original
per-subject `summary.csv`. It must agree with any cached representative-fold
metadata.

## Direct Checkpoint Extraction

```bash
python analysis/plot_riemannian_mds.py \
  --dataset stew \
  --model ms_tgc_spddsbn \
  --checkpoint-root outputs/stew_loso_ms_tgc_spddsbn \
  --results-file outputs/stew_loso_ms_tgc_spddsbn/summary.csv \
  --output-dir results/figures/riemannian_mds/stew_ms_tgc
```

For original TSMNet, use `--model tsmnet` and its SPDDSBN checkpoint directory,
for example `outputs/stew_loso_spddsbn`. Non-default model dimensions must be
passed exactly as they were during training; checkpoint loading is strict.

The output directory contains the requested centroid, distance, coordinate,
metadata, PDF, SVG, and 600 dpi PNG artifacts. A supplementary 3D MDS is
generated automatically when 2D normalized Stress-1 exceeds the configured
threshold (default `0.15`).
