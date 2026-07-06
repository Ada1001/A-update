# TSMNet Cognitive-load Reproduction

The original TSMNet implementation is kept under `TSMNet/`. The new code in `src/cl_tsmnet/` adapts it and the local EEG-Conformer, EEGNet, BF-GCN, TAHAG, and SVM baselines to the three local cognitive-load EEG datasets.

Main entry points:

- `inspect_datasets.py`: build or load preprocessed windows and print dataset statistics
- `inspect_datasets.py --protocol ...`: print exact source/target/train/validation/test counts for a planned split
- `run_experiment.py`: run TSMNet, EEG-Conformer, EEGNet, BF-GCN, TAHAG, or SVM under `single_session`, `cog_multi_session`, or `loso`
- `run_batch_experiments.py`: run multiple datasets, protocols, models, and COG-BCI paradigms from one command
- `DATASET_DESCRIPTION.md`: detailed dataset parsing, channel filtering, labels, preprocessing, and protocol notes

The pipeline uses 1 s non-overlapping windows. Cache construction performs only channel selection, segment selection, filtering, and resampling; robust per-channel normalization is fitted inside each split from source-domain training windows only, then applied to validation and target/test windows. Optional artifact-window rejection can be enabled with `--artifact-z`, using the same source-only normalization. Sampling rates are dataset-aware: STEW stays at its native 128 Hz, while EEGMAT and COG-BCI are resampled from 500 Hz to 250 Hz by default. Use `--target-fs` to override this.

Default splits are leakage-conscious. `single_session` uses the original contiguous time-block split within each task record: the last 20% is target/test, and the preceding 80% is split into training and validation with `--single-val-size 0.125`, giving approximately train/validation/test = 70%/10%/20%. `cog_multi_session` uses COG-BCI session 1 for training, session 2 for validation, and session 3 for target/test. `loso` leaves one subject as target/test, then randomly selects `ceil(20% * source_subjects)` source subjects for validation by default; the remaining source subjects are used for training.

Outputs are written per protocol/model directory. If `--cache` is omitted, caches are automatically named under `outputs/cache/` by dataset, paradigm, protocol, subject scope, sessions, and sampling rate. `summary.csv` stores one raw row per evaluated subject, including `best_epoch` and window-level train/validation/test metrics. `aggregate_summary.csv` stores the protocol-level window-level test mean +/- standard deviation with four decimals. Every completed run also appends a numeric mean/std row to `outputs/master_summary.csv` by default.

By default, `spddsbn` performs the TSMNet-style unsupervised target-domain BN refit using target windows without labels. BF-GCN uses its own graph domain-adversarial branch. TAHAG uses its own source classification loss plus gradient-reversal domain loss and hidden-layer MMD on 5-band graph features. Pass `--no-target-adapt` to disable these target-domain mechanisms; the `target_adapt` column is saved in `summary.csv`.

Use `--model eegconformer`, `--model eegnet`, or `--model svm` to run non-adaptive baseline models. EEG-Conformer, EEGNet, and SVM have no cross-domain adaptation in this project, so target-domain windows are used only for final testing and `target_adapt` is always `False`.

Use `--model svm` for the classical SVM baseline. It uses the same source-fitted robust normalization as the neural models and flattens each 1 s EEG window. The default is fast `LinearSVC` with `class_weight=balanced`; use `--svm-estimator svc --svm-kernel rbf` only for the much slower kernel-SVM ablation, and add `--svm-probability` only when calibrated probabilities are explicitly needed.

Use `--model bfgcn` to run the BF-GCN baseline. The adapter computes 5-band log-power node features and 4-band PLV adjacency matrices from each 1 s EEG window, then trains the original-style graph branches and gradient-reversal domain classifier without changing the TSMNet adaptation path.

Use `--model tahag` to run the TAHAG baseline. The adapter converts each normalized 1 s EEG window into 5-band log-power node features, then trains an adaptive graph network with source-label classification, target-unlabeled gradient-reversal domain alignment, and MMD alignment. This is separate from both TSMNet SPDDSBN and BF-GCN transfer code.

Current validation notes:

- `single_session` is the original sequential 70%/10%/20% split, not 5-fold CV.
- Normalization is fitted only on source training windows inside each split.
- EEG-Conformer, EEGNet, BF-GCN, TAHAG, and SVM command-line hyperparameters are passed into their model/training code.
- `__pycache__/`, `*.pyc`, `outputs/`, and `data/` are generated/local artifacts and should not be part of the reproducible source package.

Cache files are literal filtered/resampled window datasets. Rebuild old caches once because older caches may contain record-level standardization and are intentionally rejected by the strict loader. Name caches by dataset, paradigm, subject scope, and sampling rate, for example `cog_nback_all_250hz_1s.npz` or `stew_all_128hz_1s.npz`.
