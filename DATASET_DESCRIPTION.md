# Cognitive-load EEG Dataset Description

This project uses 1 s non-overlapping windows for all datasets. Cache construction performs channel selection, task/segment selection, 1-45 Hz band-pass filtering, 50 Hz notch filtering when the sampling rate allows it, and resampling. It does not fit normalization statistics on full recordings. During each experiment split, robust per-channel normalization is fitted from source-domain training windows only and then applied unchanged to validation and target/test windows. Artifact-window rejection is optional via `--artifact-z`; when enabled, it uses the same source-fitted normalization. The default formal protocol does not discard target windows after splitting. Sampling rates are dataset-aware: STEW remains at 128 Hz, while EEGMAT and COG-BCI are resampled from 500 Hz to 250 Hz by default. This keeps more temporal detail from the 500 Hz datasets than 128 Hz while still reducing computation and memory.

## STEW

- Location: `data/STEW Dataset`
- Files: 48 subjects, each with `sub##_lo.txt` and `sub##_hi.txt`
- Original shape: `19200 x 14`, interpreted as about 150 s at 128 Hz
- Channels retained: all 14 Emotiv channels: `AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4`
- Labels: `lo=0` low workload, `hi=1` high workload
- Sessions: one session, represented as session `1`
- Augmentation: enabled by default during training only, using light time shift, Gaussian noise, and channel dropout

## EEGMAT

- Location: `data/eeg-during-mental-arithmetic-tasks-1.0.0`
- Files: 36 subjects, two EDF recordings per subject
- Original sampling rate: 500 Hz. In the local EDF files, baseline recordings are usually about 180 s and arithmetic recordings about 60 s.
- Segment selection: for each subject, use the last 60 s of the resting/background recording and the first 60 s of the arithmetic recording, so the two classes contribute comparable usable duration.
- Retained channels: 19 scalp EEG channels after removing `ECG ECG` and `EEG A2-A1`; legacy temporal names are canonicalized (`T3/T4/T5/T6` to `T7/T8/P7/P8`)
- Labels: `_1=0` background/resting EEG, `_2=1` mental arithmetic EEG
- Sessions: one session, represented as session `1`
- Note: `subject-info.csv` contains count quality, but that is not used as the class label for cognitive-load recognition here
- Augmentation: enabled by default during training only

## COG-BCI

- Location: `data/COG-BCI`
- Files: 29 zipped subjects, each with sessions `S1/S2/S3`
- Original sampling rate: 500 Hz EEGLAB `.set/.fdt`
- Retained channels: scalp EEG only; `ECG1` is removed. The EEGLAB files expose 63 channels, giving 62 EEG channels after this removal.
- Used paradigms only: N-Back and MAT-B
- N-Back labels: `zeroBACK=0`, `oneBACK=1`, `twoBACK=2`
- MAT-B labels: `MATBeasy=0`, `MATBmed=1`, `MATBdiff=2`
- Excluded paradigms: Flanker, PVT, resting-state, behavioral-only files
- Session use: session 1 for single-session experiments; sessions 1 and 2 as source and session 3 as target for multi-session experiments

## Experiment Protocols

- `single_session`: for STEW and EEGMAT, use the single session of one subject; for COG-BCI, use session 1 only. Each task record is sorted by time and split into contiguous blocks. With default settings, the approximate proportions are train/validation/test = 64%/16%/20%. This avoids random adjacent-window mixing between train, validation, and test.
- `cog_multi_session`: COG-BCI only. For each subject, session 1 is the supervised training source, session 2 is the validation source, and session 3 is the target/test domain. Session 3 is never used for supervised training.
- `loso`: leave-one-subject-out. The held-out subject is the target/test domain; all other subjects are the source domain. Validation is also subject-disjoint from training: approximately `--val-size` of the source subjects are held out for validation.
- Default split parameters: `--test-size 0.2` for `single_session`; `--val-size 0.2` for single-session source validation blocks and LOSO source-subject validation. These parameters are exposed in both `run_experiment.py` and `inspect_datasets.py`.

## Evaluation Level

The raw model prediction is produced per 1 s window. For rigorous reporting, the project now also computes recording/task-level metrics. Windows sharing the same `(subject, session, paradigm, task)` are aggregated by averaging class probabilities, then the aggregated label is used to compute `*_group_acc`, `*_group_bacc`, `*_group_f1`, and `*_group_auc`. `aggregate_summary.csv` reports these recording/task-level test metrics by default. Window-level metrics remain in `summary.csv` as `test_acc`, `test_bacc`, `test_f1`, and `test_auc`, and `window_aggregate_summary.csv` summarizes them for comparison.

## Split Count Inspection

Exact sample counts depend on preprocessing, artifact-window rejection, chosen sampling rate, selected subject, and whether the cache contains one subject or all subjects. Use `inspect_datasets.py --protocol ...` to print the exact source/target/train/validation/test counts and class distributions before training.

Examples:

```powershell
python inspect_datasets.py --dataset stew --protocol single_session --subject 1 --cache outputs/cache/stew_all_128hz_1s.npz
python inspect_datasets.py --dataset eegmat --protocol single_session --subject 1 --cache outputs/cache/eegmat_sub01_250hz_1s.npz --target-fs 250
python inspect_datasets.py --dataset cog-bci --cog-paradigm nback --protocol cog_multi_session --subject 1 --cache outputs/cache/cog_nback_sub01_250hz_1s.npz --target-fs 250
python inspect_datasets.py --dataset stew --protocol loso --subject 1 --cache outputs/cache/stew_all_128hz_1s.npz
```

Representative counts should be regenerated after rebuilding strict caches, because older caches with record-level standardization are rejected and the split policy is now more conservative:

```powershell
python inspect_datasets.py --dataset stew --protocol loso --cache outputs/cache/stew_all_128hz_1s.npz --rebuild-cache
python inspect_datasets.py --dataset eegmat --protocol loso --cache outputs/cache/eegmat_all_250hz_1s.npz --target-fs 250 --rebuild-cache
```

## Running

Examples:

```powershell
python run_experiment.py --dataset stew --protocol loso --epochs 30 --cache outputs/cache/stew_all_128hz_1s.npz
python run_experiment.py --dataset eegmat --protocol single_session --subject 1 --epochs 30 --cache outputs/cache/eegmat_sub01_250hz_1s.npz
python run_experiment.py --dataset cog-bci --cog-paradigm nback --protocol cog_multi_session --subject 1 --epochs 30 --cache outputs/cache/cog_nback_sub01_250hz_1s.npz
python run_experiment.py --dataset cog-bci --cog-paradigm matb --protocol loso --subject 1 --epochs 30 --cache outputs/cache/cog_matb_all_250hz_1s.npz
```

The default model is the original TSMNet with SPD domain-specific batch normalization (`--bnorm spddsbn`). Use `--bnorm spdbn` for the non-domain-specific SPD BN variant or `--bnorm none` for the plain TSMNet configuration. Use `--target-fs 128`, `--target-fs 250`, or `--target-fs 500` for explicit sampling-rate comparisons; cache file names should include the chosen rate.

By default, `spddsbn` uses TSMNet-style unsupervised target-domain adaptation: target/test windows are used without labels to refit domain-specific BN statistics before evaluation. This is recorded as `target_adapt=True` in `summary.csv`. Use `--no-target-adapt` for a stricter no-target-feature baseline.

Use `--model eegconformer` for the EEG-Conformer baseline. The implementation keeps the original model idea, convolutional patch embedding followed by Transformer encoder and fully connected classifier, but makes channel count, window length, and class count dynamic for STEW, EEGMAT, and COG-BCI. EEG-Conformer does not perform target-domain adaptation; target-domain data is never used in training or normalization refitting.

Cache files should match the loaded scope. For example, do not reuse a `sub01` COG-BCI cache for a leave-one-subject-out run over all subjects. The loader checks dataset name, requested sampling rate, and strict preprocessing metadata. Older caches that contain record-level standardization must be rebuilt with `--rebuild-cache`.

## Output Files

- `summary.csv`: one raw row per evaluated subject/fold. It keeps split sizes after source-normalized artifact rejection, `epochs_ran`, `best_epoch`, `best_val_loss`, window-level metrics, and recording/task-level group metrics.
- `subject_##/history.csv`: epoch-level training history with `is_best_epoch=True` on the selected best validation epoch.
- `subject_##/model.pt`: the model state restored from the best validation epoch, then evaluated and saved.
- `aggregate_summary.csv`: protocol-level recording/task-level summary with columns `dataset, model, protocol, metric_level, n, accuracy, balanced_accuracy, f1, auc`. Metrics are formatted as `mean +/- std` with four decimals, for example `0.7539 +/- 0.0956`.
- `window_aggregate_summary.csv`: the same aggregate format using window-level test metrics for comparison and ablation reporting.
