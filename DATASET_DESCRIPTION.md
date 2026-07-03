# Cognitive-load EEG Dataset Description

This project uses 1 s non-overlapping windows for all datasets. Signals are filtered with a 1-45 Hz band-pass, a 50 Hz notch when the sampling rate allows it, robust-standardized per channel, and window-rejected when the standardized amplitude is extreme. Sampling rates are dataset-aware: STEW remains at 128 Hz, while EEGMAT and COG-BCI are resampled from 500 Hz to 250 Hz by default. This keeps more temporal detail from the 500 Hz datasets than 128 Hz while still reducing computation and memory.

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

- `single_session`: for STEW and EEGMAT, split the single session of one subject into train/validation/test; for COG-BCI, use session 1 only. Train and validation are source, test is target.
- `cog_multi_session`: COG-BCI only. For each subject, sessions 1 and 2 are source; session 3 is target.
- `loso`: leave-one-subject-out. The held-out subject is target; all other subjects are source. This applies to all three datasets.

## Running

Examples:

```powershell
python run_experiment.py --dataset stew --protocol loso --epochs 30 --cache outputs/cache/stew_all_128hz_1s.npz
python run_experiment.py --dataset eegmat --protocol single_session --subject 1 --epochs 30 --cache outputs/cache/eegmat_sub01_250hz_1s.npz
python run_experiment.py --dataset cog-bci --cog-paradigm nback --protocol cog_multi_session --subject 1 --epochs 30 --cache outputs/cache/cog_nback_sub01_250hz_1s.npz
python run_experiment.py --dataset cog-bci --cog-paradigm matb --protocol loso --subject 1 --epochs 30 --cache outputs/cache/cog_matb_all_250hz_1s.npz
```

The default model is the original TSMNet with SPD domain-specific batch normalization (`--bnorm spddsbn`). Use `--bnorm spdbn` for the non-domain-specific SPD BN variant or `--bnorm none` for the plain TSMNet configuration. Use `--target-fs 128`, `--target-fs 250`, or `--target-fs 500` for explicit sampling-rate comparisons; cache file names should include the chosen rate.

Cache files should match the loaded scope. For example, do not reuse a `sub01` COG-BCI cache for a leave-one-subject-out run over all subjects.
