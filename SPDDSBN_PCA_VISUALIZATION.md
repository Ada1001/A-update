# SPDDSBN Common-Tangent-Space PCA (Figures B3/B4)

`visualize_spddsbn_pca.py` generates the paper-domain and workload-class
visualizations for one trained `MS_TGC_SPDDSBN` LOSO fold. It does not train or
modify the classifier.

The actual feature locations are in
`src/cl_tsmnet/ms_tgc_spddsbn.py::GraphSPDManifoldHead`:

- `spd_pre_bn`: output of `BiMap + ReEig`, before `AdaMomDomainSPDBatchNorm`
- `spd_post_bn`: output of `AdaMomDomainSPDBatchNorm`, before `LogEig`

Both are `[B, d, d]`; the default `d=20`. The model's ordinary
`model(x, domain)` call is unchanged. Intermediates are returned only with
`return_intermediates=True`.

## Leakage Controls

For a selected target subject, the script reconstructs the existing LOSO split
and uses only `source_train` windows to fit the robust EEG normalizer. It loads
the saved post-refit checkpoint and never calls training or SPDDSBN refitting.
It then uses only source-train pre-SPDDSBN matrices to compute:

1. one AIRM Karcher reference matrix;
2. one `StandardScaler`;
3. one full-SVD two-component PCA.

The same reference, scaler, PCA, sample IDs, and plot manifest are used for
source/target and pre/post features in both B3 and B4. Target labels are used
only after these unsupervised transforms, for class coloring, balanced display
subsampling, and descriptive class-conditional metrics.

Automatic fold selection reads the run's `summary.csv`, averages repeated seed
rows per subject, and selects the target balanced accuracy closest to the LOSO
median. If no per-subject summary exists, use explicit target mode; the script
will not invent a representative fold.

## Command

```bash
python visualize_spddsbn_pca.py \
  --dataset stew \
  --mode auto-median-fold \
  --checkpoint-root outputs/stew_loso_ms_tgc_spddsbn \
  --output-dir results/figures/manifold_alignment/stew \
  --plot-seed 2026 \
  --max-points-per-class-domain 500
```

For COG-BCI, add `--cog-paradigm nback` or `--cog-paradigm matb`. When the
checkpoint directory has been copied without its `summary.csv`, use:

```bash
python visualize_spddsbn_pca.py \
  --dataset stew \
  --mode target-subject \
  --target-subject 12 \
  --checkpoint-root outputs/stew_loso_ms_tgc_spddsbn \
  --output-dir results/figures/manifold_alignment/stew
```

If the training run used non-default MS-TGC dimensions and its matching row is
not available in `outputs/master_summary.csv`, pass the same `--subspacedims`
and `--mstgc-*` arguments used for training. Checkpoint loading is strict and
stops on any architecture mismatch.

Older trained checkpoints may contain SPDDSBN running-statistic buffers with
one retained leading batch singleton, for example `[1,1,20,20]` instead of the
declared `[1,20,20]`. The loader records and removes only this exact legacy
singleton before strict loading; no parameter or target-domain statistic is
dropped. New training runs keep these registered buffer shapes invariant.

## Outputs

The output directory contains all three figures as PDF, SVG, and 600 dpi PNG,
plus:

- `representative_fold.json`
- `spd_intermediates.npz` and `spd_intermediates_metadata.csv`
- `common_tangent_pca_artifacts.npz`
- `pca_coordinates.csv`
- `plot_sample_manifest.csv`
- `alignment_metrics.json` and `.csv`
- `numerical_checks.json`
- `run_config.json`

The script stops instead of plotting if matrices are non-finite, asymmetric,
non-positive, if the Karcher mean fails to converge, or if checkpoint/model
dimensions do not match.
