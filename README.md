# TSMNet, EEG-Conformer, EEGNet, BF-GCN, and SVM Experiment Commands

This file lists the full commands for running TSMNet, EEG-Conformer, EEGNet, BF-GCN, and SVM experiments on STEW, EEGMAT, and COG-BCI.

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

## SVM Baseline

SVM has no cross-domain adaptation in this project. It uses the same source-fitted robust normalization as the neural models, flattens each 1 s EEG window, and trains `SVC(probability=True)` with RBF kernel and balanced class weights by default.

Use `--model svm` with any dataset/protocol command above. Examples:

```powershell
python run_experiment.py --model svm --dataset stew --protocol single_session --epochs 1 --batch-size 64
```

```powershell
python run_experiment.py --model svm --dataset eegmat --protocol loso --target-fs 250 --epochs 1 --batch-size 64
```

Useful tunable parameters are `--svm-kernel`, `--svm-c`, `--svm-gamma`, and `--svm-class-weight`. `--epochs` is accepted for command compatibility but SVM training runs once.

## Batch Runs

Run multiple datasets, protocols, models, and COG-BCI paradigms in one command:

```powershell
python run_batch_experiments.py --datasets stew,eegmat,cog-bci --protocols single_session,loso,cog_multi_session --models tsmnet,eegconformer,eegnet,bfgcn,svm --epochs 30 --batch-size 64
```

Preview commands without running:

```powershell
python run_batch_experiments.py --datasets stew,eegmat --protocols single_session,loso --models eegconformer --dry-run
```

## Baseline Defaults

Baseline defaults are aligned with the local reference implementations where explicit examples are available: EEG-Conformer uses the 1 s Conformer setting family (`emb_size=40`, `depth=6`, `num_heads=5`), EEGNet follows `EEGNet/trainEEGNet.py` (`64/4/2`), and BF-GCN follows `BF-GCN/Simple_Demo.py` (`kadj=2`, `num_out=16`, `att_hidden=16`, `classifier_hidden=32`, `avgpool=2`). SVM uses a standard RBF kernel with `C=1.0`, `gamma=scale`, and balanced class weights. For formal reporting, keep the default run and optionally add a small validation-only sweep or sensitivity analysis; do not tune on target/test labels.

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
- `single_session` uses the original contiguous sequential split within each task record: the last 20% is target/test, and the preceding source block is split into training/validation with `--single-val-size 0.125`, giving approximately train/validation/test = 70%/10%/20%.
- `cog_multi_session` uses COG-BCI S1/S2/S3 as train/validation/test; `loso` randomly selects `ceil(20% * source_subjects)` source subjects for validation and uses the remaining source subjects for training.
- `aggregate_summary.csv` reports window-level test metrics, which are the primary metrics for this project.
- `outputs/master_summary.csv` is append-only. Every completed run adds one row with model, dataset, protocol, settings, cache/output path, and numeric metric mean/std.
- Full-dataset COG-BCI caches can take time to build because each subject zip is decompressed and read from EEGLAB `.set/.fdt` files.
- Outputs are saved under `outputs/<dataset>_<protocol>_<model-or-bnorm>/` by default.
- `summary.csv` stores per-subject raw window-level train/validation/test results.
