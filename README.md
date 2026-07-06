# TSMNet, EEG-Conformer, EEGNet, BF-GCN, and MDTN-GMDA Experiment Commands

This file lists the full commands for running TSMNet, EEG-Conformer, EEGNet, BF-GCN, and MDTN-GMDA experiments on STEW, EEGMAT, and COG-BCI.

Before formal training, rebuild strict caches once and inspect the split to confirm sampling rate, subject scope, train/validation/test counts, and subject-disjoint validation where applicable. Older caches with record-level standardization are rejected by the loader.

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

For fair comparison against EEG-Conformer and EEGNet, report this no-target-adapt TSMNet variant alongside the default adaptive TSMNet result. The default adaptive result is transductive because it uses unlabeled target/test windows for model adaptation.

## EEG-Conformer Baseline

EEG-Conformer has no cross-domain adaptation in this project. Target-domain windows are used only for final testing.

Use `--model eegconformer` with any dataset/protocol command above.

Useful tunable parameters are exposed for formal sweeps: `--conformer-emb-size`, `--conformer-depth`, `--conformer-num-heads`, `--conformer-dropout`, `--conformer-classifier-hidden`, plus global `--epochs` and `--patience`.

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

The local EEGNet implementation is intentionally lightweight. For stronger EEGNet baselines, sweep `--eegnet-temporal-filters`, `--eegnet-spatial-filters`, `--eegnet-dropout`, and `--eegnet-avgpool-factor`, and consider longer `--epochs` with a larger `--patience`.

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

BF-GCN is graph-feature based rather than raw-waveform based. It may need dataset-specific tuning of `--bfgcn-domain-weight`, `--bfgcn-num-out`, `--bfgcn-classifier-hidden`, and `--bfgcn-dropout`, especially in single-session experiments where the source and target are time blocks from the same subject rather than different subjects.

## MDTN-GMDA Baseline

MDTN-GMDA uses its own transfer mechanism: source train windows provide labels, while unlabeled target/test windows participate through WarmStartGRL, graph discriminator, marginal MMD, conditional pseudo-label MMD, and graph matching losses.

Use `--model mdtn-gmda` with any dataset/protocol command above. Examples:

```powershell
python run_experiment.py --model mdtn-gmda --dataset stew --protocol loso --epochs 30 --batch-size 64
```

```powershell
python run_experiment.py --model mdtn-gmda --dataset eegmat --protocol single_session --target-fs 250 --epochs 30 --batch-size 64
```

```powershell
python run_experiment.py --model mdtn-gmda --dataset cog-bci --cog-paradigm nback --protocol cog_multi_session --target-fs 250 --epochs 30 --batch-size 64
```

Use `--no-target-adapt` for the source-only MDTN ablation. Useful tunable parameters are `--mdtn-hidden-dim`, `--mdtn-k-length`, `--mdtn-graph-k`, `--mdtn-num-heads`, `--mdtn-dropout`, `--mdtn-lambda-match`, `--mdtn-alpha`, and `--mdtn-beta`.

## Batch Runs

Run multiple datasets, protocols, models, and COG-BCI paradigms in one command:

```powershell
python run_batch_experiments.py --datasets stew,eegmat,cog-bci --protocols single_session,loso,cog_multi_session --models tsmnet,eegconformer,eegnet,bfgcn,mdtn-gmda --epochs 30 --batch-size 64
```

Preview commands without running:

```powershell
python run_batch_experiments.py --datasets stew,eegmat --protocols single_session,loso --models eegconformer --dry-run
```

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
- Cache construction does not use full-recording standardization. Robust normalization is fitted from source-domain training windows inside each split, then applied to validation and target/test windows.
- Artifact-window rejection is off by default for the formal protocol so the target/test set remains fixed. Use `--artifact-z <value>` only for an explicitly reported ablation.
- `single_session` uses contiguous time blocks within each task record with train/validation/test = 70%/10%/20%; `cog_multi_session` uses COG-BCI S1/S2/S3 as train/validation/test; `loso` randomly selects `ceil(20% * source_subjects)` source subjects for validation and uses the remaining source subjects for training.
- `aggregate_summary.csv` reports window-level test metrics, which are the primary metrics for this project.
- `outputs/master_summary.csv` is append-only. Every completed run adds one row with model, dataset, protocol, settings, cache/output path, and numeric metric mean/std.
- Full-dataset COG-BCI caches can take time to build because each subject zip is decompressed and read from EEGLAB `.set/.fdt` files.
- Outputs are saved under `outputs/<dataset>_<protocol>_<model-or-bnorm>/` by default.
- `summary.csv` stores per-subject/fold raw window-level train/validation/test results.

For single-subject reliability, the current default favors strict time-block generalization. A 5-fold single-subject cross-validation protocol can be added as a supplementary protocol if you need lower variance, but it should be reported separately from the stricter 70/10/20 time-block split.
