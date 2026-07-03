# TSMNet Cognitive-load Reproduction

The original TSMNet implementation is kept under `TSMNet/`. The new code in `src/cl_tsmnet/` adapts it to the three local cognitive-load EEG datasets without changing the model architecture.

Main entry points:

- `inspect_datasets.py`: build or load preprocessed windows and print dataset statistics
- `run_experiment.py`: run TSMNet under `single_session`, `cog_multi_session`, or `loso`
- `DATASET_DESCRIPTION.md`: detailed dataset parsing, channel filtering, labels, preprocessing, and protocol notes

The pipeline uses 1 s non-overlapping windows, robust per-channel normalization, artifact-window rejection, and train-time augmentation for STEW/EEGMAT by default. Sampling rates are dataset-aware: STEW stays at its native 128 Hz, while EEGMAT and COG-BCI are resampled from 500 Hz to 250 Hz by default. Use `--target-fs` to override this.

Cache files are literal preprocessed datasets. Name them by dataset, paradigm, and subject scope, for example `cog_nback_sub01_1s.npz` for a single-subject COG-BCI cache and `stew_all_1s.npz` for all STEW subjects.
