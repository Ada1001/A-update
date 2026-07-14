# MS_TGC_SPDDSBN Model and Graph Ablation

## Model path

The proposed model is implemented in `src/cl_tsmnet/ms_tgc_spddsbn.py` and is
trained through the common split, normalization, early-stopping, evaluation,
and output pipeline in `src/cl_tsmnet/training.py`.

The full forward path is:

1. Every EEG channel is passed independently through the same three temporal
   convolution branches. Shared kernels preserve electrode identity while
   learning comparable multi-scale channel representations.
2. Scale attention and contextual gating fuse the three full temporal maps
   into `[batch, channels, temporal_hidden, time]`.
3. Chebyshev graph convolution propagates features over EEG channels at every
   time sample, directly coupling temporal and graph learning.
4. Learned channel attention pools the maps into one shared
   `[batch, graph_hidden, time]` graph-temporal sequence.
5. Its temporal mean forms the Euclidean graph latent. Its covariance enters
   BiMap, ReEig, SPDDSBN, and LogEig to form the SPD tangent-space latent.
6. Both latents therefore share the full temporal-graph backbone. They are
   projected to the same dimension and combined by a learned element-wise gate.

The model uses source cross-entropy only. It does not use MDTN-GMDA domain
adversarial, graph matching, or MMD losses. When target adaptation is enabled,
unlabeled target windows are used only for SPDDSBN statistics after restoring
the validation-selected checkpoint. Same-domain within-session refitting is
disabled. LOSO validation-domain refits are temporary and are restored after
each validation-loss calculation.

Earlier versions expanded one pooled DTA vector across nodes or used five
summary statistics as the channel-specific component. The current version
learns a complete multi-scale temporal sequence for every channel and applies
graph propagation before temporal pooling. Older results must not be mixed
with current results.

The fusion model no longer calls TSMNet's temporal and all-channel spatial
convolutions. It reuses the manifold operations in a private
`GraphSPDManifoldHead`. The standalone TSMNet and all baselines are unchanged.

## Default tensor shapes

For input `[B, C, T]`, the default full model uses:

| Stage | Shape |
|---|---|
| three shared temporal scales | three tensors `[B, C, 64, T]` |
| scale-attention fusion and temporal pooling | `[B, C, 64, 64]` |
| time-resolved Chebyshev propagation | `[B, C, 64, 64]` |
| learned channel pooling | `[B, 64, 64]` |
| Euclidean temporal mean | `[B, 64]` |
| covariance before BiMap | `[B, 64, 64]` |
| SPD subspace after BiMap/ReEig/SPDDSBN | `[B, 20, 20]` |
| LogEig tangent vector | `[B, 210]` |
| projected Euclidean and SPD latents | two tensors `[B, 128]` |
| gated fusion and classifier | `[B, classes]` |

Defaults are temporal hidden 64, graph hidden 64, SPD subspace 20, fusion
dimension 128, temporal base kernel 16 samples at the 128 Hz reference rate,
64 graph time points, four scale-attention heads, Chebyshev order 3, dropout
0.5, and graph k=4. The base kernel is sampling-rate adjusted before odd-kernel
construction: 128 Hz uses 17/9/5 and 250 Hz uses approximately 31/15/7.
The standalone TSMNet options `temporal_filters`, `spatial_filters`, and
`temp_kernel` no longer affect this fusion model.

## Graph ablation

All four groups use the same data split, source-fitted normalizer, node
features, Chebyshev order, SPDDSBN branch, fusion head, optimizer, and loss.
Only the graph source changes.

| Model | Graph | Graph data |
|---|---|---|
| `ms_tgc_spddsbn` | source-CE learned adaptive graph | source train labels through model optimization |
| `mstgc_graph_prior` | fixed 10-20 spatial k-nearest-neighbor graph | channel names and standard montage coordinates |
| `mstgc_graph_plv` | fixed mean four-band PLV graph | source train windows only |
| `mstgc_graph_multigraph` | one spatial graph plus theta/alpha/beta/gamma PLV graphs | montage plus source train windows only |

The adaptive graph is symmetrized and top-k sparsified on each forward pass.
Fixed graphs are also symmetrized and top-k sparsified. This keeps graph density
approximately controlled across groups. `--mstgc-graph-k` controls k and
defaults to 4. Self-loops are added inside the graph encoder for every group.

PLV bands are 4-8, 8-13, 13-30, and 30-45 Hz (clipped below Nyquist when
necessary). The single PLV variant averages the four source-train PLV graphs
and sparsifies the result. The multi-graph variant applies one shared Chebyshev
layer to all five graphs and learns only the graph-mixture logits. Sharing the
Chebyshev parameters prevents the multi-graph group from gaining five copies
of the graph-convolution capacity.

Validation and target/test windows never contribute to prior or PLV graph
estimation. Target labels are never read by graph construction or adaptation.

## Outputs

Each subject directory stores `graph_state.npz` beside `model.pt`. It contains:

- `graph_mode` and `graph_names`
- channel order
- final adaptive graph or fixed graph matrices
- learned multi-graph mixture weights
- the configured neighbor count

The graph mode is also recorded in `summary.csv` and `master_summary.csv`.

## Commands

Run the four graph groups on all applicable dataset/protocol combinations:

```powershell
python run_batch_experiments.py --datasets stew,eegmat,cog-bci --protocols single_session,loso,cog_multi_session --models ms_tgc_spddsbn,mstgc_graph_prior,mstgc_graph_plv,mstgc_graph_multigraph --epochs 30 --batch-size 64 --mstgc-graph-k 4
```

`run_batch_experiments.py` automatically skips `cog_multi_session` for STEW
and EEGMAT. It runs both N-back and MAT-B for COG-BCI unless
`--cog-paradigms` narrows the selection.

For a strict source-only sensitivity experiment, append
`--no-target-adapt`. Do not combine source-only and transductive SPDDSBN rows in
one primary comparison table.

## Reporting

Use at least the same random seed set for all four graph groups. Report mean
and standard deviation over evaluation subjects, and compare paired per-subject
scores rather than only comparing aggregate means. A useful secondary analysis
is `k` sensitivity with k in 2, 4, and 8, selected and reported without using
target/test labels. Keep `k=4` as the primary prespecified setting.

Run the graph-specific automated checks with:

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```
