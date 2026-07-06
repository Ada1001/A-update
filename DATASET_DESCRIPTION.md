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
- Zip layout handling: the loader accepts both `sub-xx/ses-Sx/eeg/...` and duplicated-root layouts such as `sub-xx/sub-xx/ses-Sx/eeg/...`, which occur in some local COG-BCI archives.
- Retained channels: a fixed 62-channel scalp EEG whitelist in a fixed order. The EEGLAB files usually expose 63 channels; auxiliary/non-scalp channels are ignored so every COG-BCI subject and task has the same input shape.
- Used paradigms only: N-Back and MAT-B
- N-Back labels: `zeroBACK=0`, `oneBACK=1`, `twoBACK=2`
- MAT-B labels: `MATBeasy=0`, `MATBmed=1`, `MATBdiff=2`
- Excluded paradigms: Flanker, PVT, resting-state, behavioral-only files
- Session use: session 1 for single-session experiments; sessions 1 and 2 as source and session 3 as target for multi-session experiments

## Experiment Protocols

- `single_session`: for STEW and EEGMAT, use the single session of one subject; for COG-BCI, use session 1 only. Each task record is sorted by time and split contiguously: the last `--test-size 0.2` is target/test, and the preceding 80% source block is further split into training and validation. This restores the original sequential 70%/10%/20% single-subject protocol and avoids random adjacent-window mixing between train, validation, and test.
- `cog_multi_session`: COG-BCI only. For each subject, session 1 is the supervised training source, session 2 is the validation source, and session 3 is the target/test domain. Session 3 is never used for supervised training.
- `loso`: leave-one-subject-out. The held-out subject is the target/test domain; all other subjects are the source domain. Validation is subject-disjoint from training: by default, `ceil(0.2 * number_of_source_subjects)` source subjects are randomly selected as validation subjects using the experiment seed, and the remaining source subjects form the training set.
- Default split parameters: `--test-size 0.2` and `--single-val-size 0.125` for `single_session`, giving approximately 70%/10%/20% train/validation/test because validation is taken from the remaining 80% source block. `--val-size 0.2` controls LOSO source-subject validation. These parameters are exposed in `run_experiment.py`, `run_batch_experiments.py`, and `inspect_datasets.py`.

## Evaluation Level

Metrics are computed at the 1 s window level, which is the standard reporting unit for many EEG workload-classification experiments using fixed-length windows. `summary.csv` stores per-subject window-level metrics, and `aggregate_summary.csv` reports the mean +/- standard deviation of window-level test metrics across evaluated subjects.

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

Use `--model eegconformer` for the EEG-Conformer baseline. The implementation keeps the original model idea, convolutional patch embedding followed by Transformer encoder and fully connected classifier, but makes channel count, window length, and class count dynamic for STEW, EEGMAT, and COG-BCI. Use `--model eegnet` for the local EEGNet baseline. The project implementation keeps the original EEGNet folder's temporal convolution, depthwise spatial convolution, average pooling/dropout, depthwise temporal convolution, pointwise convolution, and fully connected classifier, while adapting input shape, channel count, window length, and class count dynamically. Use `--model svm` for a classical SVM baseline over flattened, source-normalized 1 s EEG windows. Use `--model lsccn` for the LSCCN feature-fusion capsule baseline. Use `--model lstm`, `--model bilstm`, `--model transformer`, or `--model shallowcnn` for basic temporal baselines. EEG-Conformer, EEGNet, SVM, LSCCN, LSTM, BiLSTM, Transformer, and ShallowCNN do not perform target-domain adaptation; target-domain data is never used in training or normalization refitting.

Use `--model bfgcn` for the local BF-GCN baseline. The adapter follows the original BF-GCN input design by converting each 1 s window into 5-band log-power node features and 4-band PLV adjacency matrices, then training the graph branches with a gradient-reversal domain classifier. Source train windows provide class labels; unlabeled target/test windows contribute only to the domain-adversarial loss when `target_adapt=True`. This is separate from TSMNet's SPDDSBN target BN refit. Use `--no-target-adapt` to disable BF-GCN target-domain use for an ablation.

Use `--model tahag` for the local TAHAG baseline. The adapter follows the subject-independent transfer setting by converting each 1 s window into 5-band log-power graph node features, then training an adaptive graph network with source-label classification, gradient-reversal domain loss, and MMD alignment. Target/test labels are never used in training; unlabeled target/test windows are used only by the TAHAG transfer losses when `target_adapt=True`. Use `--no-target-adapt` for a source-only TAHAG ablation.

Use `--model lsccn` for the LSCCN baseline described in the local paper `Latent_Space_Coding_Capsule_Network_for_Mental_Workload_Classification.docx`. The cache remains raw-window based; during training/evaluation, each source-normalized 1 s window is converted to a fused feature matrix by concatenating gamma-band PLV connectivity with 5-band log-power features. The network uses the paper's VAE-to-capsule structure: two 3x3 convolutional encoder layers, a 200-dimensional latent code, a 1x9 convolutional block, primary capsules, dynamic-routing digit capsules, and margin plus VAE reconstruction/KL loss. Because the original LSCCN is not a domain-adaptation model, target/test windows are used only for final evaluation.

Use `--model lstm` and `--model bilstm` for recurrent temporal baselines. They treat each 1 s EEG window as a sequence of samples with channels as features. Defaults are hidden size 64, one recurrent layer, and dropout 0.5. Use `--model transformer` for a source-only Transformer encoder baseline with channel projection, learnable positional embeddings, mean temporal pooling, and a classifier. Defaults are `d_model=64`, 4 heads, 2 encoder layers, feed-forward width 128, and dropout 0.2. Use `--model shallowcnn` for a compact EEG ShallowConvNet-style baseline with temporal convolution, spatial convolution across channels, square/log nonlinearity, average pooling, and dropout. Defaults are 40 filters, temporal kernel 25, pooling size 25, and dropout 0.5. These baselines use the same train-time augmentation option and the same non-augmented train/validation/test evaluation policy as the other neural models.

Cache files should match the loaded scope. For example, do not reuse a `sub01` COG-BCI cache for a leave-one-subject-out run over all subjects. The loader checks dataset name, requested sampling rate, strict preprocessing metadata, requested sessions, and COG-BCI subject coverage for the requested paradigm. Older strict caches with object-array string metadata are migrated automatically to the current safe string-array format. Older caches that contain record-level standardization or incomplete subject coverage must be rebuilt with `--rebuild-cache`.

If `--cache` is omitted, the scripts automatically create a cache name under `outputs/cache/`, for example `stew_loso_all_s1_128hz_1s.npz` or `cog_nback_cog_multi_session_all_s123_250hz_1s.npz`.

## Output Files

- `summary.csv`: one raw row per evaluated subject. It keeps split sizes after optional artifact rejection, `epochs_ran`, `best_epoch`, `best_val_loss`, and window-level train/validation/test metrics.
- `subject_##/history.csv`: epoch-level training history with `is_best_epoch=True` on the selected best validation epoch.
- `subject_##/model.pt`: the neural model state restored from the best validation epoch, then evaluated and saved. SVM runs save `subject_##/model.joblib`.
- `aggregate_summary.csv`: protocol-level window-level summary with columns `dataset, model, protocol, n, accuracy, balanced_accuracy, f1, auc`. Metrics are formatted as `mean +/- std` with four decimals, for example `0.7539 +/- 0.0956`.
- `outputs/master_summary.csv`: project-level append-only table. Each completed run appends one row with dataset, model, protocol, output directory, cache path, settings, and numeric mean/std metrics.
