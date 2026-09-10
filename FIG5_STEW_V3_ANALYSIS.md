# STEW Fig. 5 export audit

Input: `fig5_stew_v3.zip`, generated 2026-09-09 14:05 UTC. The ZIP contains
figures, per-fold/aggregate alignment metrics, embedding coordinates, sample
manifest, and metadata. It does not contain model weights, high-dimensional
feature caches, per-subject training summaries, histories, or predictions.
Classification aggregates below come from training records embedded in metadata.
They have not been independently recomputed from predictions.

## Observations

| Method | Test BAcc | Test macro F1 | Test AUC | Train evaluation accuracy | Validation accuracy |
|---|---:|---:|---:|---:|---:|
| Mean-CE | 78.73% | 77.92% | 88.21% | 83.21% | 76.85% |
| Mean-EuDSBN | 80.19% | 80.16% | 87.08% | 79.13% | 79.22% |
| AugSPD-SPDBN | 73.53% | 70.26% | 88.13% | 78.07% | 71.32% |
| AugSPD-SPDDSBN | 82.22% | 82.16% | 88.84% | 60.99% | 81.56% |

The full model is best on these aggregate test metrics: +3.49 percentage
points BAcc versus Mean-CE, +2.03 versus EuDSBN, +8.69 versus SPDBN. These
are descriptive differences, not evidence of statistical significance.
Per-subject scores are needed for paired inference; LOSO training sets overlap.
The reported full-model between-subject BAcc standard deviation is 9.23
percentage points, not a standard error or confidence interval.

All four alignment runs include the same 48 subjects. The displayed fold is
subject 21: full-model BAcc 84%, close to the LOSO median 83.67%. Each panel
uses the same 600 sample IDs: 150 per class/domain, despite the cap being 200.

| Method | Mean domain discrepancy | Mean class separation | Mean separation ratio | Mean Fisher ratio |
|---|---:|---:|---:|---:|
| Mean-CE | 0.538083 | 0.654926 | 1.329254 | 0.549361 |
| Mean-EuDSBN | 0.853690 | 0.651154 | 0.782894 | 0.336930 |
| AugSPD-SPDBN | 0.552165 | 0.484620 | 0.931085 | 0.280207 |
| AugSPD-SPDDSBN | 0.281987 | 0.086667 | 0.322982 | 0.014599 |

Full-model separation ratio is below Mean-CE in all 48 folds. This is not
just a visually unfavorable representative fold. Do not multiply aggregate
discrepancy and aggregate ratio to recover aggregate class separation:
the mean of a product is not generally the product of means.

For subject 21 the full-model discrepancy is 0.241364, class separation
0.091058, ratio 0.377265, Fisher ratio 0.019510. Its source UMAP coordinates
have two clearly separated islands (split at x=0):

| Island | Low | High |
|---|---:|---:|
| Left | 64 | 65 |
| Right | 86 | 85 |

36 of 37 sampled source subjects fall entirely within one island. Assigning
each subject to its majority island accounts for 299/300 source points.
Thus these islands track subjects much more than workload classes. This is
descriptive of these coordinates, not proof of the biological cause.
The sum of 2D coordinate variances is 57.632677 for source and 0.091626 for
target (ratio about 0.00159). This verifies target compression in the displayed
embedding, not collapse of the original 210-dimensional representation.

## Explanation and unresolved causes

1. Source and target statistics have different estimation histories. Training
   evaluation and figure extraction use saved source-domain EMA test buffers.
   Validation/test domains are refitted on all their own unlabeled windows
   with the selected model. Source train buffers are not refitted at the end.
   The source evaluation accuracy of 60.99% versus target 82.22%, plus
   subject-specific islands, makes this the first mechanism to investigate.
   The code establishes the asymmetry, but the ZIP cannot prove causation.
   Feature drift, augmentation, and few windows per domain per mixed batch
   are plausible contributors to inaccurate source buffers.
2. The metric pools many source subjects' class centers in standardized
   Euclidean coordinates. Subject variation can dominate variance and mask
   within-subject class directions. It is not a measure of classifier accuracy
   or full-distribution equivalence. The full model's nonlinear readout also
   follows the plotted tangent features (LayerNorm, projection, GELU).
3. PCA fits source samples only, then source-fitted UMAP maps target samples.
   Source dominant directions need not preserve target class information.
   PCA retains 90.11% of source variance for the full model, versus 99.57%
   for Mean-CE, 97.12% for EuDSBN and 86.53% for SPDBN. These percentages say
   nothing directly about retained target discriminative information.
4. SPDBN has high AUC (88.13%) but lower BAcc (73.53%): inspect per-subject
   confusion matrices, score distributions, source global BN buffers and
   validation calibration. This pattern suggests a decision/calibration
   issue is possible, but does not establish it from aggregate scores.
5. EuDSBN metadata proves target-only scope, not the numerical refit algorithm.
   There is no code hash/refit implementation version in the export. Confirm
   the server used the corrected full-domain moments implementation; the
   export cannot determine this from architecture labels alone.

## Prioritized experiments

- Keep the existing test result and export as the baseline.
- With fixed weights on subject 21, compare saved source buffers against a
  COPY of the model with only source-train domains refitted, using the same
  normalizer, samples and extraction point. Keep target buffers unchanged.
  Recompute source accuracy, within-subject separation and alignment metrics;
  verify target predictions do not change. If the source behavior recovers,
  repeat across all folds and report it as an explicit analysis protocol.
- Compare saved tangent features with post-readout 128D features and target-only
  class metrics. This is a different extraction point, so label it explicitly.
- Diagnose reduction using identical samples with no PCA pre-reduction
  (`--pca-dim 210` for defaults), and separately joint unlabeled t-SNE. Changing
  only the reducer/PCA must leave the high-dimensional metric panels unchanged.
- Only then tune training on source validation: consider domain-balanced
  batches/statistics estimation if the buffer experiment supports that cause.
  Do not choose a seed, subject, threshold or reducer to enforce a desired trend.

## Intended figure and TSMNet comparison

An ideal alignment plot preserves Low/High separation while mixing source
and target within each class. Separate islands are not required, and an
ordering of all four methods is not guaranteed. A convincing positive result
would combine lower domain discrepancy with preserved/improved class
separation and stronger held-out classification. This export supports better
test classification and reduced centroid discrepancy, but not improved pooled
class separation. Use that qualified conclusion.

The new `--fourth-model tsmnet` preset keeps the first three methods and uses
TSMNet-SPDDSBN as column four. Keep `--target-subjects stew=21` for comparison
with the original figure. Otherwise the representative is TSMNet's median
fold. This changes the question to a mixed-architecture baseline comparison;
TSMNet is not the final stage of the MSTGC ablation and is not guaranteed to
produce a better looking embedding. See FIG5_REPRESENTATION_ALIGNMENT.md for
the interface and output paths.
