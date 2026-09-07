# Fig. 5 Cross-Subject Representation Alignment

`analysis/fig5_representation_alignment.py` is the independent implementation
of the IEEE Transactions Fig. 5 method-progression comparison. It does not
overwrite or reuse coordinates from the earlier visualization scripts.

## Visualization inventory

| Script | Scientific question | Default output |
|---|---|---|
| `visualize_spddsbn_pca.py` | Where the same checkpoint moves from pre- to post-SPDDSBN in one common tangent/PCA space | `results/figures/spddsbn_pca/` |
| `analysis/plot_riemannian_mds.py` | How subject-by-class Frechet centers move under AIRM geometry | `results/figures/riemannian_mds/` |
| `analysis/plot_nonlinear_alignment_embeddings.py` | Whether pre/post nonlinear embeddings are stable under fixed seeds | `results/figures/nonlinear_embeddings/` |
| `analysis/fig5_representation_alignment.py` | How four trained representation/calibration strategies compare across datasets | `results/fig5_representation_alignment.*` |

The first three are before/after analyses of one model. Fig. 5 is a horizontal
comparison of four separately trained models and must not be interpreted as a
trajectory through one checkpoint.

## Default methods

| Figure label | Training model | Exported feature |
|---|---|---|
| Mean-CE | `mstgc_mean_ce` | first-order mean before the classifier head |
| Mean-EuDSBN | `mstgc_dta_cheb_eudsbn` | first-order mean after EuDSBN |
| AugSPD-SPDBN | `mstgc_dta_cheb_spdbn` | LogEig tangent vector after global SPDBN |
| AugSPD-SPDDSBN | `ms_tgc_spddsbn` | LogEig tangent vector after SPDDSBN |

All four retain the same multi-scale temporal, adaptive Chebyshev graph, and
channel-reliability front end. Their final feature dimensions can differ, so
each method is standardized with source-only statistics before comparison.

## Protocol safeguards

- The representative target subject is selected only from the full
  AugSPD-SPDDSBN LOSO results as the BAcc closest to the median. The same target
  subject, split, sample IDs, and balanced manifest are used in all four panels.
- PCA and UMAP are fitted on balanced source features only. Target features are
  transformed without target labels. If UMAP is unavailable, one joint,
  label-free t-SNE fit is used and recorded in metadata.
- Domain discrepancy and separation ratio are calculated in the
  source-standardized high-dimensional representation, never from 2D points.
- With `--metric-scope all` (default), metric curves report the mean and 95%
  CI over the intersection of completed LOSO target subjects for all methods.
- Target labels are used only after training for balanced plotting and
  class-conditional metrics.
- The script never modifies coordinates or axes to force a monotonic trend.

Domain discrepancy is the mean class-conditional source-target centroid RMS
distance. Separation ratio is the mean pairwise pooled class-centroid RMS
distance divided by domain discrepancy. Both are computed after fitting the
feature standardizer on source windows only.

## Publication command

Install the preferred optional reducer once:

```bash
pip install -r requirements-analysis.txt
```

Generate STEW and N-Back Fig. 5 from the original LOSO outputs:

```bash
python analysis/fig5_representation_alignment.py \
  --datasets stew,cog-bci:nback \
  --dataset-labels STEW,N-Back \
  --output-root outputs \
  --master-summary outputs/master_summary.csv \
  --metric-scope all \
  --max-points-per-group 200 \
  --reducer auto \
  --output-dir results
```

Replace the second row with EEGMAT without changing the plotting code:

```bash
python analysis/fig5_representation_alignment.py \
  --datasets stew,eegmat \
  --dataset-labels STEW,EEGMAT \
  --metric-scope all \
  --output-dir results
```

For a quick pipeline check, use `--metric-scope representative`. This is not
the recommended quantitative panel for the paper.

## TSMNet support

The shared feature adapter supports both `tsmnet` and MSTGC checkpoints. The
default Fig. 5 remains the specified four MSTGC methods because TSMNet does not
implement Mean-CE or Mean-EuDSBN. For a separate four-run TSMNet comparison,
provide a JSON `--method-manifest`; each entry has `label`, `short_label`,
`model_type: "tsmnet"`, `bnorm`, and optionally `run_dir`. The script rejects
missing checkpoints and never relabels a TSMNet normalization as a nonexistent
method.

## Outputs

```text
results/fig5_representation_alignment.pdf
results/fig5_representation_alignment.png
results/fig5_representation_alignment.svg
results/fig5_embedding_coordinates.csv
results/fig5_plot_sample_manifest.csv
results/fig5_alignment_metrics_by_fold.csv
results/fig5_alignment_metrics_aggregate.csv
results/fig5_representation_alignment_metadata.json
results/fig5_representation_cache/
```

The metadata records model runs, checkpoints, exact feature locations, split
configuration, representative-subject evidence, reducers, target-label usage,
and software versions.
