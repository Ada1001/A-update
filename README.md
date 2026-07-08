# TSMNet, EEG-Conformer, EEGNet, BF-GCN, TAHAG, MDTN-GMDA, MS_TGC_SPDDSBN, SVM, LSCCN, and Temporal Baseline Commands

This file lists the full commands for running TSMNet, EEG-Conformer, EEGNet, BF-GCN, TAHAG, MDTN-GMDA, MS_TGC_SPDDSBN, SVM, LSCCN, LSTM, BiLSTM, Transformer, and ShallowCNN experiments on STEW, EEGMAT, and COG-BCI.

Before formal training, rebuild strict caches once and inspect the split to confirm sampling rate, subject scope, train/validation/test counts, and subject-disjoint validation where applicable. Older caches with record-level standardization are rejected by the loader.
For COG-BCI, all protocols run split-quality checks before training. Subjects with unusable train/validation/test splits are skipped by default; use `--allow-incomplete-splits` only for diagnostics.

## Split Inspection

When `--cache` is omitted, the scripts automatically name caches under `outputs/cache/`.

```powershell
python inspect_datasets.py --dataset stew --protocol loso
```

Add `--rebuild-cache` the first time you regenerate the strict cache:

```powershell
python inspect_datasets.py --dataset stew --protocol loso --cache outputs/cache/stew_all_128hz_1s.npz --rebuild-cache
```

```powershell
python inspect_datasets.py --dataset eegmat --protocol single_session --cache outputs/cache/eegmat_all_250hz_1s.npz --target-fs 250
```

```powershell
python inspect_datasets.py --dataset cog-bci --cog-paradigm nback --protocol cog_multi_session --cache outputs/cache/cog_nback_all_250hz_1s.npz --target-fs 250
```

```powershell
python inspect_datasets.py --dataset cog-bci --cog-paradigm matb --protocol cog_multi_session --cache outputs/cache/cog_matb_all_250hz_1s.npz --target-fs 250
```

## STEW

Single-subject single-session:

```powershell
python run_experiment.py --dataset stew --protocol single_session --epochs 30 --batch-size 64
```

Leave-one-subject-out:

```powershell
python run_experiment.py --dataset stew --protocol loso --epochs 30 --batch-size 64
```

## EEGMAT

Single-subject single-session:

```powershell
python run_experiment.py --dataset eegmat --protocol single_session --cache outputs/cache/eegmat_all_250hz_1s.npz --target-fs 250 --output outputs/tsmnet --epochs 30 --batch-size 64
```

Leave-one-subject-out:

```powershell
python run_experiment.py --dataset eegmat --protocol loso --cache outputs/cache/eegmat_all_250hz_1s.npz --target-fs 250 --output outputs/tsmnet --epochs 30 --batch-size 64
```

## COG-BCI N-Back

Single-subject single-session, session 1 only:

```powershell
python run_experiment.py --dataset cog-bci --cog-paradigm nback --protocol single_session --cache outputs/cache/cog_nback_all_250hz_1s.npz --target-fs 250 --output outputs/tsmnet --epochs 30 --batch-size 64
```

Single-subject multi-session, sessions 1 and 2 as source and session 3 as target:

```powershell
python run_experiment.py --dataset cog-bci --cog-paradigm nback --protocol cog_multi_session --cache outputs/cache/cog_nback_all_250hz_1s.npz --target-fs 250 --output outputs/tsmnet --epochs 30 --batch-size 64
```

Leave-one-subject-out:

```powershell
python run_experiment.py --dataset cog-bci --cog-paradigm nback --protocol loso --cache outputs/cache/cog_nback_all_250hz_1s.npz --target-fs 250 --output outputs/tsmnet --epochs 30 --batch-size 64
```

## COG-BCI MAT-B

Single-subject single-session, session 1 only:

```powershell
python run_experiment.py --dataset cog-bci --cog-paradigm matb --protocol single_session --cache outputs/cache/cog_matb_all_250hz_1s.npz --target-fs 250 --output outputs/tsmnet --epochs 30 --batch-size 64
```

Single-subject multi-session, sessions 1 and 2 as source and session 3 as target:

```powershell
python run_experiment.py --dataset cog-bci --cog-paradigm matb --protocol cog_multi_session --cache outputs/cache/cog_matb_all_250hz_1s.npz --target-fs 250 --output outputs/tsmnet --epochs 30 --batch-size 64
```

Leave-one-subject-out:

```powershell
python run_experiment.py --dataset cog-bci --cog-paradigm matb --protocol loso --cache outputs/cache/cog_matb_all_250hz_1s.npz --target-fs 250 --output outputs/tsmnet --epochs 30 --batch-size 64
```

## Strict No-Target-Adapt Variant

By default, `spddsbn` uses TSMNet-style unsupervised target-domain BN refit with unlabeled target windows. To disable this for stricter no-target-feature experiments, append:

```powershell
--no-target-adapt
```

Example:

```powershell
python run_experiment.py --dataset eegmat --protocol loso --cache outputs/cache/eegmat_all_250hz_1s.npz --target-fs 250 --output outputs/tsmnet_no_target_adapt --epochs 30 --batch-size 64 --no-target-adapt
```

## EEG-Conformer Baseline

EEG-Conformer has no cross-domain adaptation in this project. Target-domain windows are used only for final testing.

Use `--model eegconformer` with any dataset/protocol command above.

## EEGNet Baseline

EEGNet has no cross-domain adaptation in this project. Target-domain windows are used only for final testing.

Use `--model eegnet` with any dataset/protocol command above. Examples:

```powershell
python run_experiment.py --model eegnet --dataset stew --protocol loso --epochs 30 --batch-size 64
```

```powershell
python run_experiment.py --model eegnet --dataset eegmat --protocol single_session --target-fs 250 --epochs 30 --batch-size 64
```

```powershell
python run_experiment.py --model eegnet --dataset cog-bci --cog-paradigm nback --protocol cog_multi_session --target-fs 250 --epochs 30 --batch-size 64
```

The default EEGNet command follows the local `EEGNet/trainEEGNet.py` example configuration: `--eegnet-temporal-filters 64 --eegnet-spatial-filters 4 --eegnet-avgpool-factor 2`. For ablations, sweep these parameters plus `--eegnet-dropout`, and consider longer `--epochs` with a larger `--patience`.

## BF-GCN Baseline

BF-GCN uses its own domain-adversarial transfer branch. Source-domain training windows use class labels; target-domain windows are used only with domain labels in the gradient-reversal loss and are still evaluated with their labels only after training.

Use `--model bfgcn` with any dataset/protocol command above. Examples:

```powershell
python run_experiment.py --model bfgcn --dataset stew --protocol loso --epochs 30 --batch-size 64
```

```powershell
python run_experiment.py --model bfgcn --dataset eegmat --protocol single_session --target-fs 250 --epochs 30 --batch-size 64
```

```powershell
python run_experiment.py --model bfgcn --dataset cog-bci --cog-paradigm nback --protocol cog_multi_session --target-fs 250 --epochs 30 --batch-size 64
```

For a no-target-feature ablation, append:

```powershell
--no-target-adapt
```

## TAHAG Baseline

TAHAG uses its own target-domain adaptation. Source training windows provide class labels; unlabeled target/test windows participate through gradient-reversal domain alignment and hidden-layer MMD. The adapter uses 5-band log-power graph node features from each normalized 1 s EEG window, with adaptive graph learning and feature attention.

Use `--model tahag` with any dataset/protocol command above. Examples:

```powershell
python run_experiment.py --model tahag --dataset stew --protocol loso --epochs 30 --batch-size 64
```

```powershell
python run_experiment.py --model tahag --dataset eegmat --protocol single_session --target-fs 250 --epochs 30 --batch-size 64
```

For a source-only ablation, append:

```powershell
--no-target-adapt
```

Useful tunable parameters are `--tahag-dropout`, `--tahag-domain-weight`, `--tahag-mmd-weight`, `--no-tahag-adaptive`, and `--no-tahag-attention`.

## MDTN-GMDA Baseline

MDTN-GMDA uses its own target-domain adaptation. Source windows provide class labels; target/test windows are used without labels through warm-start gradient reversal, a Chebyshev graph discriminator, graph matching, and marginal/conditional MMD. The adapter follows `MDTN-GMDA/Net.py` but fixes the reference MDTN attention dimensionality so the multi-scale temporal tokens can be trained inside this project.

Use `--model mdtn` with any dataset/protocol command above. Examples:

```powershell
python run_experiment.py --model mdtn --dataset stew --protocol loso --epochs 30 --batch-size 64
```

```powershell
python run_experiment.py --model mdtn --dataset eegmat --protocol single_session --target-fs 250 --epochs 30 --batch-size 64
```

```powershell
python run_experiment.py --model mdtn --dataset cog-bci --cog-paradigm nback --protocol cog_multi_session --target-fs 250 --epochs 30 --batch-size 64
```

For a source-only ablation, append:

```powershell
--no-target-adapt
```

Useful tunable parameters are `--mdtn-hidden-dim`, `--mdtn-num-nodes`, `--mdtn-kernel-length`, `--mdtn-num-heads`, `--mdtn-cheby-order`, `--mdtn-dropout`, `--mdtn-lambda-match`, `--mdtn-marginal-weight`, `--mdtn-conditional-weight`, and `--mdtn-l1-weight`.

## MS_TGC_SPDDSBN Fusion Model

MS_TGC_SPDDSBN combines the TSMNet-SPDDSBN SPD manifold branch with MDTN-style multi-scale temporal features and a Chebyshev graph-convolution feature encoder. The fusion is latent-level: the SPDDSBN log-Eig latent and graph-temporal latent are projected to a shared dimension and combined with a learnable gate before classification. It intentionally does not use MDTN-GMDA's domain-adversarial, MMD, or graph-matching losses. Training uses source-domain cross entropy only; target/test windows are used without labels only during the TSMNet-SPDDSBN statistics refit.

Use `--model ms_tgc_spddsbn` with any dataset/protocol command above. Examples:

```powershell
python run_experiment.py --model ms_tgc_spddsbn --dataset stew --protocol loso --epochs 30 --batch-size 64
```

```powershell
python run_experiment.py --model ms_tgc_spddsbn --dataset eegmat --protocol single_session --target-fs 250 --epochs 30 --batch-size 64
```

```powershell
python run_experiment.py --model ms_tgc_spddsbn --dataset cog-bci --cog-paradigm nback --protocol cog_multi_session --target-fs 250 --epochs 30 --batch-size 64
```

For a source-only ablation with no target-domain SPDDSBN refit, append:

```powershell
--no-target-adapt
```

Useful tunable parameters are `--mstgc-temporal-hidden`, `--mstgc-graph-hidden`, `--mstgc-fusion-dim`, `--mstgc-kernel-length`, `--mstgc-num-heads`, `--mstgc-cheby-order`, `--mstgc-dropout`, and `--mstgc-num-nodes`.

The ablation variants are implemented in the same model file and can be selected with `--model`:

| Model name | Meaning |
|---|---|
| `mstgc_dta_ce` | DTA + CE, multi-scale temporal branch only |
| `mstgc_dta_cheb_ce` | DTA + Cheb + CE |
| `mstgc_dta_cheb_eudsbn` | DTA + Cheb + Euclidean DSBN |
| `mstgc_dta_cheb_spdbn` | DTA + Cheb + SPDBN, SPD BN without domain-specific statistics |
| `ms_tgc_spddsbn` | DTA + Cheb + SPDDSBN, full fusion model |
| `mstgc_wo_dta` | Full model without DTA; channel summary features feed Cheb |
| `mstgc_wo_cheb` | Full model without Cheb; DTA features are fused directly |
| `mstgc_wo_spddsbn` | Full fusion backbone with the TSMNet SPD branch but no SPDDSBN alignment |

Run all ablations on the three datasets:

```powershell
python run_batch_experiments.py --datasets stew,eegmat,cog-bci --protocols single_session,loso,cog_multi_session --models mstgc_dta_ce,mstgc_dta_cheb_ce,mstgc_dta_cheb_eudsbn,mstgc_dta_cheb_spdbn,ms_tgc_spddsbn,mstgc_wo_dta,mstgc_wo_cheb,mstgc_wo_spddsbn --epochs 30 --batch-size 64
```

## SVM Baseline

SVM has no cross-domain adaptation in this project. It uses the same source-fitted robust normalization as the neural models and flattens each 1 s EEG window. The default is fast `LinearSVC` with balanced class weights; AUC is computed from the decision function. Kernel SVM is available for ablation with `--svm-estimator svc`.

Use `--model svm` with any dataset/protocol command above. Examples:

```powershell
python run_experiment.py --model svm --dataset stew --protocol single_session --epochs 1 --batch-size 64
```

```powershell
python run_experiment.py --model svm --dataset eegmat --protocol loso --target-fs 250 --epochs 1 --batch-size 64
```

Useful tunable parameters are `--svm-estimator`, `--svm-kernel`, `--svm-c`, `--svm-gamma`, `--svm-class-weight`, `--svm-max-iter`, and `--svm-probability`. `--epochs` is accepted for command compatibility but SVM training runs once. `--svm-probability` is intentionally off by default because SVC probability calibration can make LOSO runs several times slower.

## LSCCN Baseline

LSCCN has no cross-domain adaptation in this project. It follows the local paper's feature-fusion capsule design: each source-normalized 1 s EEG window is converted into 5-band log-power features plus gamma-band PLV connectivity, then classified by a VAE, 1D convolution, primary capsules, and dynamic-routing digit capsules. Target-domain windows are used only for final testing.
For binary LSCCN runs, the final capsule decision threshold is selected on validation windows only and saved as `decision_threshold`.

Use `--model lsccn` with any dataset/protocol command above. Examples:

```powershell
python run_experiment.py --model lsccn --dataset stew --protocol loso --epochs 30 --batch-size 64
```

```powershell
python run_experiment.py --model lsccn --dataset eegmat --protocol single_session --target-fs 250 --epochs 30 --batch-size 64
```

```powershell
python run_experiment.py --model lsccn --dataset cog-bci --cog-paradigm nback --protocol cog_multi_session --target-fs 250 --epochs 30 --batch-size 64
```

Useful tunable parameters are `--lsccn-latent-dim`, `--lsccn-routing-iters`, `--lsccn-recon-weight`, and `--lsccn-kl-weight`. Defaults are `200`, `3`, `1e-5`, and `0.1`, respectively.

## Temporal Baselines

LSTM, BiLSTM, Transformer, and ShallowCNN have no cross-domain adaptation in this project. They use the same source-fitted robust normalization and the same train/validation/test splits as all other models. Target-domain windows are used only for final testing.

Use `--model lstm`, `--model bilstm`, `--model transformer`, or `--model shallowcnn` with any dataset/protocol command above. Examples:

```powershell
python run_experiment.py --model lstm --dataset stew --protocol loso --epochs 30 --batch-size 64
```

```powershell
python run_experiment.py --model bilstm --dataset eegmat --protocol single_session --target-fs 250 --epochs 30 --batch-size 64
```

```powershell
python run_experiment.py --model transformer --dataset cog-bci --cog-paradigm nback --protocol cog_multi_session --target-fs 250 --epochs 30 --batch-size 64
```

```powershell
python run_experiment.py --model shallowcnn --dataset cog-bci --cog-paradigm matb --protocol loso --target-fs 250 --epochs 30 --batch-size 64
```

Useful LSTM/BiLSTM parameters are `--recurrent-hidden`, `--recurrent-layers`, and `--recurrent-dropout`. Useful Transformer parameters are `--transformer-d-model`, `--transformer-heads`, `--transformer-layers`, `--transformer-ff`, and `--transformer-dropout`. Useful ShallowCNN parameters are `--shallow-filters`, `--shallow-kernel`, `--shallow-pool`, and `--shallow-dropout`.

## Pre-run Audit Checklist

- `single_session` is the original sequential split: each task record is time-sorted, the last 20% is test, and the previous 80% is split into train/validation with `--single-val-size 0.125`.
- `cog_multi_session` uses COG-BCI sessions 1/2/3 as train/validation/test.
- `loso` holds out one target subject and selects validation from source subjects only.
- Cache construction performs filtering/resampling/windowing only. Robust normalization is fitted inside each split from source training windows only.
- COG-BCI caches store recording-subject coverage and known no-window exclusions. A known absent subject is skipped consistently; newly added recording subjects require rebuilding the cache.
- Train metrics are evaluated on non-augmented training windows; validation and test are also non-augmented.
- EEG-Conformer, EEGNet, BF-GCN, TAHAG, MDTN-GMDA, MS_TGC_SPDDSBN, SVM, LSCCN, LSTM, BiLSTM, Transformer, and ShallowCNN hyperparameters in `run_experiment.py` and `run_batch_experiments.py` are passed to the corresponding training code.
- Keep `data/`, `outputs/`, `__pycache__/`, and `*.pyc` out of the source release.

## Batch Runs

Run multiple datasets, protocols, models, and COG-BCI paradigms in one command:

```powershell
python run_batch_experiments.py --datasets stew,eegmat,cog-bci --protocols single_session,loso,cog_multi_session --models tsmnet,eegconformer,eegnet,bfgcn,tahag,mdtn,ms_tgc_spddsbn,svm,lsccn,lstm,bilstm,transformer,shallowcnn --epochs 30 --batch-size 64
```

Preview commands without running:

```powershell
python run_batch_experiments.py --datasets stew,eegmat --protocols single_session,loso --models eegconformer --dry-run
```

## Baseline Defaults

Baseline defaults are aligned with the local reference implementations where explicit examples are available: EEG-Conformer uses the 1 s Conformer setting family (`emb_size=40`, `depth=6`, `num_heads=5`), EEGNet follows `EEGNet/trainEEGNet.py` (`64/4/2`), and BF-GCN follows `BF-GCN/Simple_Demo.py` (`kadj=2`, `num_out=16`, `att_hidden=16`, `classifier_hidden=32`, `avgpool=2`). TAHAG follows the independent-transfer setting: adaptive graph learning, attention, source classification, GRL domain loss, and MMD loss. MDTN-GMDA follows the local `MDTN-GMDA/Net.py` components with hidden size 64, 4 attention heads, Cheby order 3, graph matching weight 0.1, and marginal/conditional MMD weights 0.01. MS_TGC_SPDDSBN uses the same TSMNet SPD branch defaults plus 64-dimensional temporal and graph features, 128-dimensional gated fusion, 4 attention heads, and Cheby order 3. SVM defaults to `LinearSVC` because flattened EEG windows are high-dimensional and LOSO repeatedly trains on thousands of windows; RBF `SVC` remains available as an explicitly reported slow ablation. LSCCN follows the local paper's 200-dimensional latent space and 3 dynamic-routing iterations, with PLV plus band-power feature fusion. LSTM/BiLSTM use hidden size 64, Transformer uses `d_model=64` with 4 heads and 2 layers, and ShallowCNN uses 40 temporal-spatial filters. For formal reporting, keep the default run and optionally add a small validation-only sweep or sensitivity analysis; do not tune on target/test labels.

STEW:

```powershell
python run_experiment.py --model eegconformer --dataset stew --protocol single_session --cache outputs/cache/stew_all_128hz_1s.npz --output outputs/eegconformer --epochs 30 --batch-size 64
```

```powershell
python run_experiment.py --model eegconformer --dataset stew --protocol loso --cache outputs/cache/stew_all_128hz_1s.npz --output outputs/eegconformer --epochs 30 --batch-size 64
```

EEGMAT:

```powershell
python run_experiment.py --model eegconformer --dataset eegmat --protocol single_session --cache outputs/cache/eegmat_all_250hz_1s.npz --target-fs 250 --output outputs/eegconformer --epochs 30 --batch-size 64
```

```powershell
python run_experiment.py --model eegconformer --dataset eegmat --protocol loso --cache outputs/cache/eegmat_all_250hz_1s.npz --target-fs 250 --output outputs/eegconformer --epochs 30 --batch-size 64
```

COG-BCI N-Back:

```powershell
python run_experiment.py --model eegconformer --dataset cog-bci --cog-paradigm nback --protocol single_session --cache outputs/cache/cog_nback_all_250hz_1s.npz --target-fs 250 --output outputs/eegconformer --epochs 30 --batch-size 64
```

```powershell
python run_experiment.py --model eegconformer --dataset cog-bci --cog-paradigm nback --protocol cog_multi_session --cache outputs/cache/cog_nback_all_250hz_1s.npz --target-fs 250 --output outputs/eegconformer --epochs 30 --batch-size 64
```

```powershell
python run_experiment.py --model eegconformer --dataset cog-bci --cog-paradigm nback --protocol loso --cache outputs/cache/cog_nback_all_250hz_1s.npz --target-fs 250 --output outputs/eegconformer --epochs 30 --batch-size 64
```

COG-BCI MAT-B:

```powershell
python run_experiment.py --model eegconformer --dataset cog-bci --cog-paradigm matb --protocol single_session --cache outputs/cache/cog_matb_all_250hz_1s.npz --target-fs 250 --output outputs/eegconformer --epochs 30 --batch-size 64
```

```powershell
python run_experiment.py --model eegconformer --dataset cog-bci --cog-paradigm matb --protocol cog_multi_session --cache outputs/cache/cog_matb_all_250hz_1s.npz --target-fs 250 --output outputs/eegconformer --epochs 30 --batch-size 64
```

```powershell
python run_experiment.py --model eegconformer --dataset cog-bci --cog-paradigm matb --protocol loso --cache outputs/cache/cog_matb_all_250hz_1s.npz --target-fs 250 --output outputs/eegconformer --epochs 30 --batch-size 64
```

## Notes

- Cache files are checked for dataset name, sampling rate, and strict preprocessing metadata. If you change `--target-fs`, use a matching cache name or pass `--rebuild-cache`.
- If `--cache` is omitted, cache names are generated automatically by dataset, paradigm, protocol, subject scope, sessions, and sampling rate.
- For COG-BCI, older caches without recording-subject metadata are usable as their own subject universe but should be rebuilt once before formal reporting so no-window exclusions are explicit.
- Cache construction does not use full-recording standardization. Robust normalization is fitted from source-domain training windows inside each split, then applied to validation and target/test windows.
- Artifact-window rejection is off by default for the formal protocol so the target/test set remains fixed. Use `--artifact-z <value>` only for an explicitly reported ablation.
- `single_session` uses the original contiguous sequential split within each task record: the last 20% is target/test, and the preceding source block is split into training/validation with `--single-val-size 0.125`, giving approximately train/validation/test = 70%/10%/20%.
- `cog_multi_session` uses COG-BCI S1/S2/S3 as train/validation/test; `loso` randomly selects `ceil(20% * source_subjects)` source subjects for validation and uses the remaining source subjects for training.
- `aggregate_summary.csv` reports window-level test metrics, which are the primary metrics for this project.
- `outputs/master_summary.csv` is append-only. Every completed run adds one row with model, dataset, protocol, settings, cache/output path, and numeric metric mean/std.
- Full-dataset COG-BCI caches can take time to build because each subject zip is decompressed and read from EEGLAB `.set/.fdt` files.
- Outputs are saved under `outputs/<dataset>_<protocol>_<model-or-bnorm>/` by default.
- `summary.csv` stores per-subject raw window-level train/validation/test results.
