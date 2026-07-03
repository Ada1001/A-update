# TSMNet Cognitive-load Reproduction

The original TSMNet implementation is kept under `TSMNet/`. The new code in `src/cl_tsmnet/` adapts it to the three local cognitive-load EEG datasets without changing the model architecture.

Main entry points:

- `inspect_datasets.py`: build or load preprocessed windows and print dataset statistics
- `inspect_datasets.py --protocol ...`: print exact source/target/train/validation/test counts for a planned split
- `run_experiment.py`: run TSMNet under `single_session`, `cog_multi_session`, or `loso`
- `DATASET_DESCRIPTION.md`: detailed dataset parsing, channel filtering, labels, preprocessing, and protocol notes

The pipeline uses 1 s non-overlapping windows, robust per-channel normalization, artifact-window rejection, and train-time augmentation for STEW/EEGMAT by default. Sampling rates are dataset-aware: STEW stays at its native 128 Hz, while EEGMAT and COG-BCI are resampled from 500 Hz to 250 Hz by default. Use `--target-fs` to override this.

Default splits are source/target = 8:2 for `single_session`, then source train/validation = 8:2, giving about 64%/16%/20% train/validation/test. For `cog_multi_session`, COG-BCI sessions 1-2 are source and session 3 is target. For `loso`, non-held-out subjects are source and the held-out subject is target.

Outputs are written per protocol/model directory. `summary.csv` stores one raw row per evaluated subject/fold, including `best_epoch`, `accuracy`, `balanced_accuracy`, `f1`, and `auc`. `aggregate_summary.csv` stores protocol-level mean +/- standard deviation with four decimals.

By default, `spddsbn` performs the TSMNet-style unsupervised target-domain BN refit using target windows without labels. Pass `--no-target-adapt` to disable this; the `target_adapt` column is saved in `summary.csv`.

Cache files are literal preprocessed datasets. Name them by dataset, paradigm, and subject scope, for example `cog_nback_sub01_1s.npz` for a single-subject COG-BCI cache and `stew_all_1s.npz` for all STEW subjects.
