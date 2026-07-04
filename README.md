# TSMNet and EEG-Conformer Experiment Commands

This file lists the full commands for running TSMNet and EEG-Conformer experiments on STEW, EEGMAT, and COG-BCI.

Before formal training, rebuild strict caches once and inspect the split to confirm sampling rate, subject scope, train/validation/test counts, and subject-disjoint validation where applicable. Older caches with record-level standardization are rejected by the loader.

## Split Inspection

```powershell
python inspect_datasets.py --dataset stew --protocol loso --cache outputs/cache/stew_all_128hz_1s.npz
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
python run_experiment.py --dataset stew --protocol single_session --cache outputs/cache/stew_all_128hz_1s.npz --output outputs/tsmnet --epochs 30 --batch-size 64
```

Leave-one-subject-out:

```powershell
python run_experiment.py --dataset stew --protocol loso --cache outputs/cache/stew_all_128hz_1s.npz --output outputs/tsmnet --epochs 30 --batch-size 64
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
- Cache construction does not use full-recording standardization. Robust normalization is fitted from source-domain training windows inside each split, then applied to validation and target/test windows.
- Artifact-window rejection is off by default for the formal protocol so the target/test set remains fixed. Use `--artifact-z <value>` only for an explicitly reported ablation.
- `single_session` uses contiguous time blocks within each task record; `cog_multi_session` uses COG-BCI S1/S2/S3 as train/validation/test; `loso` randomly selects `ceil(20% * source_subjects)` source subjects for validation and uses the remaining source subjects for training.
- `aggregate_summary.csv` reports window-level test metrics, which are the primary metrics for this project.
- Full-dataset COG-BCI caches can take time to build because each subject zip is decompressed and read from EEGLAB `.set/.fdt` files.
- Outputs are saved under `outputs/<model>/<dataset>_<protocol>_<model-or-bnorm>/`.
- `summary.csv` stores per-subject/fold raw window-level train/validation/test results.
