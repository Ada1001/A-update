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
4. Channel attention computes one reliability value per electrode and applies
   `sqrt(alpha + epsilon)` weighting. Channels are not summed, so the output
   remains `[batch, channels, graph_hidden, time]`.
5. Channel and time axes become observations: `[B,C,F,L] -> [B,F,C*L]`.
   The model estimates the observation mean and a trace-target shrinkage
   covariance using `--mstgc-shrinkage` (default 0.1).
6. Mean and covariance form one Gaussian augmented SPD matrix
   `[[Sigma + mu*mu^T, mu], [mu^T, 1]]`. With `F=64`, its shape is 65 by 65.
7. The only full-model readout is BiMap 65-to-20, ReEig, SPDDSBN, LogEig/Vec
   210, Linear 210-to-128, and the classifier. There is no Euclidean/SPD gate.

The model uses source cross-entropy only. It does not use MDTN-GMDA domain
adversarial, graph matching, or MMD losses. When target adaptation is enabled,
unlabeled target windows are used only for SPDDSBN statistics after restoring
the validation-selected checkpoint. Same-domain within-session refitting is
disabled. LOSO validation-domain refits are temporary and are restored after
each validation-loss calculation.

Refitting uses `--refit-batch-size 16` by default. The temporal-graph backbone
extracts pre-BN SPD features in GPU mini-batches; compact `20 x 20` matrices are
then fitted once per domain on CPU. This preserves full-domain REFIT statistics
without materializing every raw EEG window and temporal scale on GPU at once.

Earlier versions expanded one pooled DTA vector across nodes or used five
summary statistics as the channel-specific component. The current version
learns a complete multi-scale temporal sequence for every channel and applies
graph propagation before temporal pooling. Older results must not be mixed
with current results.

The proposed model does not call TSMNet's temporal and all-channel spatial
convolutions. It reuses the manifold operations in a private
`GraphSPDManifoldHead`. The standalone TSMNet and all baselines are unchanged.

## Default tensor shapes

For input `[B, C, T]`, the default full model uses:

| Stage | Shape |
|---|---|
| three shared temporal scales | three tensors `[B, C, 64, T]` |
| scale-attention fusion and temporal pooling | `[B, C, 64, 64]` |
| time-resolved Chebyshev propagation | `[B, C, 64, 64]` |
| reliability-weighted maps, channels retained | `[B, C, 64, 64]` |
| channel-time observations | `[B, 64, C*64]` |
| first-order mean and shrinkage covariance | `[B,64,1]`, `[B,64,64]` |
| Gaussian augmented SPD before BiMap | `[B, 65, 65]` |
| SPD subspace after BiMap/ReEig/SPDDSBN | `[B, 20, 20]` |
| LogEig tangent vector | `[B, 210]` |
| tangent projection | `[B, 128]` |
| classifier | `[B, classes]` |

Defaults are temporal hidden 64, graph hidden 64, SPD subspace 20, fusion
dimension 128, temporal base kernel 16 samples at the 128 Hz reference rate,
64 graph time points, covariance shrinkage 0.1, four scale-attention heads,
Chebyshev order 3, dropout 0.5, and graph k=4. The base kernel is
sampling-rate adjusted before odd-kernel
construction: 128 Hz uses 17/9/5 and 250 Hz uses approximately 31/15/7.
The standalone TSMNet options `temporal_filters`, `spatial_filters`, and
`temp_kernel` no longer affect this fusion model.

## Graph ablation

All four groups use the same data split, source-fitted normalizer, node
features, Chebyshev order, augmented SPD head, optimizer, and loss.
Only the graph source changes.

| Model | Graph | Graph data |
|---|---|---|
| `ms_tgc_spddsbn` | source-CE learned adaptive graph | source train labels through model optimization |
| `mstgc_graph_prior` | fixed 10-20 spatial k-nearest-neighbor graph | channel names and standard montage coordinates |
| `mstgc_graph_plv` | fixed mean four-band PLV graph | source train windows only |
| `mstgc_graph_multigraph` | one spatial graph plus theta/alpha/beta/gamma PLV graphs | montage plus source train windows only |

### Update policy

- `ms_tgc_spddsbn` has one global adaptive adjacency shared by all windows in
  a split. Selected top-k edge weights and the shared Chebyshev parameters are
  updated by source classification loss.
- `mstgc_graph_prior` and `mstgc_graph_plv` store adjacency matrices as fixed
  buffers. They never receive gradients; only the common Chebyshev feature
  extractor is trained.
- `mstgc_graph_multigraph` also keeps all five adjacency matrices fixed. One
  Chebyshev layer is reused for every graph, and trainable global softmax logits
  combine their outputs. It therefore learns common graph filters and graph
  importance, not five independent graph networks.
- PLV and multigraph connectivity is estimated separately for every split from
  source training windows only. It is a source-population graph in LOSO and a
  source-session graph in cross-session experiments, not a target-informed or
  sample-specific graph.

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

## Statistical representation ablation

This group keeps the multi-scale temporal encoder, adaptive Chebyshev graph,
channel reliability, optimizer, split, and classifier width fixed.

| Reported model | Representation | Manifold input | Normalization |
|---|---|---|---|
| `mstgc_mean_ce` | first-order mean | none | none |
| `mstgc_cov_spddsbn` | shrinkage covariance | `64 x 64 -> 20 x 20` | SPDDSBN |
| `mstgc_augspd_spddsbn` | mean plus covariance | `65 x 65 -> 20 x 20` | SPDDSBN |

`mstgc_mean_ce` is exactly the existing `mstgc_dta_cheb_ce` model, and
`mstgc_augspd_spddsbn` is exactly the full `ms_tgc_spddsbn` model. Do not report
both names as separate independent experiments. The main inferential contrast
is covariance-only versus augmented SPDDSBN because both have the same
20-dimensional SPD output and 210-dimensional tangent classifier; only the
first-order mean embedding differs.

## Module ablation after the augmented-SPD redesign

The required primary module set is now:

| Model | Removed or replaced component |
|---|---|
| `mstgc_wo_dta` | three-scale attention replaced by one shared temporal scale |
| `mstgc_wo_cheb` | channel graph propagation removed |
| `mstgc_wo_channel_attention` | learned reliability replaced by fixed `1/C` reliability |
| `mstgc_wo_spddsbn` | SPDDSBN removed while augmented SPD is retained |

Use `--mstgc-shrinkage 0`, `0.05`, `0.1`, and `0.2` as a sensitivity analysis,
not as four extra named architectures. The prespecified primary value remains
0.1 and must not be selected with target/test performance.

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
