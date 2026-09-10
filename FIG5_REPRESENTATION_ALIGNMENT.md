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

The exported vectors are the inputs to `readout`, before its LayerNorm,
Linear projection to 128 dimensions, GELU, and dropout. They are not the
128-dimensional inputs to the final linear classifier.

This is a method comparison, not a normalization-only controlled ablation:
Mean-EuDSBN has per-domain trainable affine parameters, whereas SPDDSBN shares
its learned scalar dispersion across domains and fixes the output mean.
Unseen EuDSBN domains retain their initialized affine parameters because
unlabeled refitting updates statistics only. The Mean-to-AugSPD comparison
also changes representation dimension and introduces BiMap/ReEig/LogEig.

### EuDSBN refit correction

`DomainBatchNorm1d.refit_domain_stats` now replaces the supplied domains'
running mean and unbiased variance with full-domain estimates, preserving
affine parameters, other domains, and train/eval mode. Previously it reset
the buffers and made one update with momentum 0.1, retaining 90% of the
default mean/variance. For example, a target mean of 10 was saved as 1.
EuDSBN runs made before this correction should be retrained: the correction
also affects validation-domain normalization and hence checkpoint selection.
Changing CSV provenance or only rerunning the plotting script does not
repair those saved training results.

Model-contract checks, including one-epoch synthetic training of all four
variants, can be run from the project root:

```bash
python -m pytest tests/test_fig5_model_contract.py tests/test_mstgc_graphs.py tests/test_fig5_representation_alignment.py tests/test_loso_protocol.py tests/test_spd_domain_refit.py -q
```

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
- Default MS-TGC comparisons require the current v3 architecture provenance
  in `master_summary.csv`. Legacy checkpoints without architecture and
  representation records are rejected rather than silently mixed with v3.
- Domain-adapted methods require `target_refit_scope=target_only`. Evidence is
  read from each run's `summary.csv`, with its exactly matched master-summary
  row used only as a fallback. `--allow-legacy-refit` is intended for
  exploratory legacy plots, not the publication figure.
- The master-summary row must match the selected run's `output_dir` exactly;
  graph-density and Chebyshev sensitivity runs are never substituted for the
  standard model run.
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

Generate each dataset independently. Every command produces its own complete
1x4 embedding row and two quantitative panels, which can be composed later
without mixing data in the analysis script.

STEW:

```bash
python analysis/fig5_representation_alignment.py \
  --datasets stew \
  --dataset-labels STEW \
  --output-root outputs \
  --master-summary outputs/master_summary.csv \
  --metric-scope all \
  --max-points-per-group 200 \
  --reducer auto \
  --output-dir results/fig5_stew
```

N-Back:

```bash
python analysis/fig5_representation_alignment.py \
  --datasets cog-bci:nback \
  --dataset-labels N-Back \
  --output-root outputs \
  --master-summary outputs/master_summary.csv \
  --metric-scope all \
  --max-points-per-group 200 \
  --reducer auto \
  --output-dir results/fig5_nback
```

EEGMAT uses the same command with `--datasets eegmat`,
`--dataset-labels EEGMAT`, and `--output-dir results/fig5_eegmat`.
Two-dataset output remains supported for backward compatibility. Dataset
labels are checked against reserved names, so N-Back cannot accidentally be
published under the EEGMAT label.

The figure extractor reads each fold's saved best `model.pt`; it does not
retrain a valid run. Retraining is required only when a method is missing, was
trained with a different architecture/front end, or lacks auditable target
refit provenance. A checkpoint alone cannot recover which windows were used to
fit its saved domain statistics.

For a quick pipeline check, use `--metric-scope representative`. This is not
the recommended quantitative panel for the paper.

## TSMNet support

To retain the first three default MSTGC methods and replace only column four
with TSMNet-SPDDSBN, use `--fourth-model tsmnet`. Default run discovery tries
`<output-root>/<dataset>_loso_tsmnet_spddsbn` and then
`<output-root>/<dataset>_loso_spddsbn`. `--fourth-run-dir` overrides that path
(absolute, or relative to output-root; `{dataset}` is supported). The master
summary must contain an exactly matched record for each of the four runs.
Do not combine this preset with `--method-manifest`.

```bash
python analysis/fig5_representation_alignment.py \
  --datasets stew --dataset-labels STEW \
  --fourth-model tsmnet \
  --output-root outputs/fig5_stew_v3 \
  --master-summary outputs/fig5_stew_v3/master_summary.csv \
  --target-subjects stew=21 \
  --metric-scope all --max-points-per-group 200 \
  --reducer umap --batch-size 16 --device cpu \
  --output-dir results/fig5_stew_tsmnet_s21
```

The explicit subject retains subject 21 from the uploaded STEW figure.
Without this override, the representative is selected from column four's
LOSO median (TSMNet in this preset). The metadata records the reference method.
Only MSTGC runs are checked for an identical MSTGC front end; all four must
still share the split/preprocessing configuration. This mixed architecture
figure is a baseline comparison, not a four-step ablation of one architecture.
TSMNet exports its covariance-SPD tangent representation, not augmented SPD.

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
