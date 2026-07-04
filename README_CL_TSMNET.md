# TSMNet Cognitive-load Reproduction

The original TSMNet implementation is kept under `TSMNet/`. The new code in `src/cl_tsmnet/` adapts it to the three local cognitive-load EEG datasets without changing the model architecture.

Main entry points:

- `inspect_datasets.py`: build or load preprocessed windows and print dataset statistics
- `inspect_datasets.py --protocol ...`: print exact source/target/train/validation/test counts for a planned split
- `run_experiment.py`: run TSMNet or EEG-Conformer under `single_session`, `cog_multi_session`, or `loso`
- `DATASET_DESCRIPTION.md`: detailed dataset parsing, channel filtering, labels, preprocessing, and protocol notes

The pipeline uses 1 s non-overlapping windows. Cache construction performs only channel selection, segment selection, filtering, and resampling; robust per-channel normalization is fitted inside each split from source-domain training windows only, then applied to validation and target/test windows. Optional artifact-window rejection can be enabled with `--artifact-z`, using the same source-only normalization. Sampling rates are dataset-aware: STEW stays at its native 128 Hz, while EEGMAT and COG-BCI are resampled from 500 Hz to 250 Hz by default. Use `--target-fs` to override this.

Default splits are leakage-conscious. `single_session` uses contiguous time blocks within each task record, approximately 64%/16%/20% train/validation/test. `cog_multi_session` uses COG-BCI session 1 for training, session 2 for validation, and session 3 for target/test. `loso` leaves one subject as target/test and holds out source subjects, not random source windows, for validation.

Outputs are written per protocol/model directory. `summary.csv` stores one raw row per evaluated subject/fold, including `best_epoch`, window-level `test_*` metrics, and recording/task-level `test_group_*` metrics. `aggregate_summary.csv` uses the stricter recording/task-level metrics by default and stores mean +/- standard deviation with four decimals. `window_aggregate_summary.csv` keeps the old window-level aggregate for comparison.

By default, `spddsbn` performs the TSMNet-style unsupervised target-domain BN refit using target windows without labels. Pass `--no-target-adapt` to disable this; the `target_adapt` column is saved in `summary.csv`.

Use `--model eegconformer` to run the EEG-Conformer baseline. EEG-Conformer has no cross-domain adaptation in this project, so target-domain windows are used only for final testing and `target_adapt` is always `False`.

Cache files are literal filtered/resampled window datasets. Rebuild old caches once because older caches may contain record-level standardization and are intentionally rejected by the strict loader. Name caches by dataset, paradigm, subject scope, and sampling rate, for example `cog_nback_all_250hz_1s.npz` or `stew_all_128hz_1s.npz`.
