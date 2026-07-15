# MS_TGC_SPDDSBN Model Flowchart

## Full Forward Path

```mermaid
flowchart TD
    X[EEG window X<br/>B x C x T] --> TEMP

    subgraph TEMPORAL[Shared per-channel multi-scale temporal encoder]
        TEMP[Reshape to B*C x 1 x T] --> S1[Shared temporal Conv scale 1]
        TEMP --> S2[Shared temporal Conv scale 2]
        TEMP --> S3[Shared temporal Conv scale 3]
        S1 --> ATT[Cross-scale multi-head attention]
        S2 --> ATT
        S3 --> ATT
        ATT --> SG[Context gate and softmax scale fusion]
        SG --> HT[Channel temporal maps<br/>B x C x F x L]
    end

    HT --> CHEB
    subgraph GRAPH[Time-resolved EEG graph propagation]
        CHEB[Chebyshev graph convolution<br/>adaptive, prior, PLV, or multigraph]
        CHEB --> HG[Graph temporal maps<br/>B x C x F x L]
        HG --> SCORE[Channel score from temporal mean]
        SCORE --> ALPHA[Softmax reliability alpha<br/>B x C x 1]
        ALPHA --> WEIGHT[Multiply each channel by<br/>sqrt alpha + epsilon]
        HG --> WEIGHT
        WEIGHT --> HW[Weighted maps<br/>B x C x F x L<br/>no channel summation]
    end

    HW --> OBS[Permute and reshape<br/>Z: B x F x C*L]
    OBS --> MU[Mean mu<br/>B x F x 1]
    OBS --> COV[Sample covariance Sigma<br/>B x F x F]
    COV --> SHRINK[Trace-target shrinkage<br/>Sigma_lambda]
    MU --> AUG
    SHRINK --> AUG[Gaussian augmented SPD<br/>P_aug: B x F+1 x F+1]

    AUG --> BIMAP[BiMap F+1 to 20]
    BIMAP --> REEIG[ReEig]
    REEIG --> DSBN[SPDDSBN]
    DSBN --> LOG[LogEig and Vec<br/>B x 210]
    LOG --> PROJ[LayerNorm + Linear 210 to 128<br/>GELU + Dropout]
    PROJ --> CLS[Linear classifier]
    CLS --> Y[Class logits<br/>B x K]
```

For the default `F=64`, the augmented input to BiMap is `65 x 65`. BiMap
reduces it to `20 x 20`, so tangent vectorization remains
`20 * 21 / 2 = 210`.

## Mathematical Readout

For reliability-weighted graph maps `H`:

```text
alpha_c = softmax(q(mean_L(H_c)))
H_c     = sqrt(alpha_c + epsilon) * H_c
Z       = reshape(H) in R^(B x F x (C*L))
mu      = mean_observation(Z)
Sigma   = centered(Z) centered(Z)^T / (C*L - 1)
Sigma_l = (1-lambda) Sigma + lambda tr(Sigma)/F I + epsilon I
P_aug   = [[Sigma_l + mu mu^T, mu], [mu^T, 1]]
```

The square-root reliability factor makes a channel's second-order contribution
approximately proportional to `alpha_c`, instead of `alpha_c^2`. The shrinkage
term and diagonal epsilon keep the covariance positive definite before the
Gaussian augmentation.

## Training And Target Adaptation

```mermaid
flowchart LR
    SRC[Source train windows and labels] --> CE[Source cross entropy]
    CE --> OPT[Optimize temporal, graph, reliability, BiMap, and classifier]
    VAL[Source-only validation] --> BEST[Select lowest validation loss]
    OPT --> BEST
    BEST --> LOAD[Restore best checkpoint]
    LOAD --> REFIT
    TGT[Unlabelled target windows] --> REFIT[Refit target SPDDSBN statistics]
    REFIT --> TEST[Window-level target evaluation]
    LABEL[Target labels] -. evaluation only .-> TEST
```

No target labels, pseudo-label loss, domain-adversarial loss, MMD, or graph
matching loss is used by the full model. Same-domain single-session refitting
is disabled; `--no-target-adapt` disables target SPDDSBN refitting entirely.

## Consistent Ablations

| Model | Shared temporal | Cheb | Statistical readout | Normalization |
|---|---|---|---|---|
| `mstgc_dta_ce` | Multi-scale | No | First-order mean | None |
| `mstgc_dta_cheb_ce` | Multi-scale | Yes | First-order mean | None |
| `mstgc_dta_cheb_eudsbn` | Multi-scale | Yes | First-order mean | Euclidean DSBN |
| `mstgc_mean_ce` | Multi-scale | Yes | First-order mean | None; alias of `mstgc_dta_cheb_ce` |
| `mstgc_cov_spddsbn` | Multi-scale | Yes | Covariance SPD | SPDDSBN |
| `mstgc_augspd_spddsbn` | Multi-scale | Yes | Augmented SPD | SPDDSBN; full-model alias |
| `mstgc_dta_cheb_spdmbn` | Multi-scale | Yes | Augmented SPD | Shared SPD BN |
| `mstgc_dta_cheb_spdbn` | Multi-scale | Yes | Augmented SPD | Shared SPD BN alias |
| `ms_tgc_spddsbn` | Multi-scale | Yes | Augmented SPD | SPDDSBN |
| `mstgc_wo_dta` | Single-scale | Yes | Augmented SPD | SPDDSBN |
| `mstgc_wo_cheb` | Multi-scale | No | Augmented SPD | SPDDSBN |
| `mstgc_wo_channel_attention` | Multi-scale | Yes | Augmented SPD | SPDDSBN; uniform channel weights |
| `mstgc_wo_spddsbn` | Multi-scale | Yes | Augmented SPD | None |

The graph-source variants `mstgc_graph_prior`, `mstgc_graph_plv`, and
`mstgc_graph_multigraph` retain the complete augmented-SPD model and change
only the adjacency source.
