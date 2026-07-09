# TSMNet Cognitive-load Reproduction

The original TSMNet implementation is kept under `TSMNet/`. The new code in `src/cl_tsmnet/` adapts it and the local EEG-Conformer, EEGNet, BF-GCN, TAHAG, MDTN-GMDA, MS_TGC_SPDDSBN, SVM, LSCCN, LSTM, BiLSTM, Transformer, and ShallowCNN baselines to the three local cognitive-load EEG datasets.

Main entry points:

- `inspect_datasets.py`: build or load preprocessed windows and print dataset statistics
- `inspect_datasets.py --protocol ...`: print exact source/target/train/validation/test counts for a planned split
- `run_experiment.py`: run TSMNet, EEG-Conformer, EEGNet, BF-GCN, TAHAG, MDTN-GMDA, MS_TGC_SPDDSBN, SVM, LSCCN, LSTM, BiLSTM, Transformer, or ShallowCNN under `single_session`, `cog_multi_session`, or `loso`
- `run_batch_experiments.py`: run multiple datasets, protocols, models, and COG-BCI paradigms from one command
- `DATASET_DESCRIPTION.md`: detailed dataset parsing, channel filtering, labels, preprocessing, and protocol notes

The pipeline uses 1 s non-overlapping windows. Cache construction performs only channel selection, segment selection, filtering, and resampling; robust per-channel normalization is fitted inside each split from source-domain training windows only, then applied to validation and target/test windows. Optional artifact-window rejection can be enabled with `--artifact-z`, using the same source-only normalization. Sampling rates are dataset-aware: STEW stays at its native 128 Hz, while EEGMAT and COG-BCI are resampled from 500 Hz to 250 Hz by default. Use `--target-fs` to override this.

Default splits are leakage-conscious. `single_session` uses the original contiguous time-block split within each task record: the last 20% is target/test, and the preceding 80% is split into training and validation with `--single-val-size 0.125`, giving approximately train/validation/test = 70%/10%/20%. `cog_multi_session` is the ordinary COG-BCI cross-session transfer protocol: sessions 1 and 2 are source sessions, session 3 is the held-out target/test session, and validation windows are taken only from sessions 1 and 2 by contiguous per-record/task blocks controlled by `--val-size`. Session 3 is never used for validation or supervised training. `loso` leaves one subject as target/test, then randomly selects `ceil(20% * source_subjects)` source subjects for validation by default; the remaining source subjects are used for training.

Outputs are written per protocol/model directory. If `--cache` is omitted, caches are automatically named under `outputs/cache/` by dataset, paradigm, protocol, subject scope, sessions, and sampling rate. `summary.csv` stores one raw row per evaluated subject, including `train_sessions`, `test_session`, `best_epoch`, and window-level train/validation/test metrics. `aggregate_summary.csv` stores the protocol-level window-level test mean +/- standard deviation with four decimals. Every completed run also appends a numeric mean/std row to `outputs/master_summary.csv` by default.

For COG-BCI, all protocols apply split-quality checks before training. Subjects with unusable splits are skipped by default; `--allow-incomplete-splits` is only for diagnostics. Full COG-BCI caches may also record subjects that had requested recordings but produced no usable windows, so those subjects are excluded consistently instead of causing later model runs to fail.

By default, `spddsbn` performs the TSMNet-style unsupervised target-domain BN refit using target windows without labels. BF-GCN uses its own graph domain-adversarial branch. TAHAG uses its own source classification loss plus gradient-reversal domain loss and hidden-layer MMD on 5-band graph features. MDTN-GMDA uses its own raw-window multi-scale temporal feature extractor, gradient-reversal Chebyshev graph discriminator, graph matching loss, and marginal/conditional MMD. MS_TGC_SPDDSBN uses only source-label cross entropy during training, then uses unlabeled target windows only for the embedded TSMNet-SPDDSBN statistics refit. Pass `--no-target-adapt` to disable these target-domain mechanisms; the `target_adapt` column is saved in `summary.csv`.

Use `--model eegconformer`, `--model eegnet`, `--model svm`, `--model lsccn`, `--model lstm`, `--model bilstm`, `--model transformer`, or `--model shallowcnn` to run non-adaptive baseline models. These models have no cross-domain adaptation in this project, so target-domain windows are used only for final testing and `target_adapt` is always `False`.

Use `--model svm` for the classical SVM baseline. It uses the same source-fitted robust normalization as the neural models and flattens each 1 s EEG window. The default is fast `LinearSVC` with `class_weight=balanced`; use `--svm-estimator svc --svm-kernel rbf` only for the much slower kernel-SVM ablation, and add `--svm-probability` only when calibrated probabilities are explicitly needed.

Use `--model bfgcn` to run the BF-GCN baseline. The adapter computes 5-band log-power node features and 4-band PLV adjacency matrices from each 1 s EEG window, then trains the original-style graph branches and gradient-reversal domain classifier without changing the TSMNet adaptation path.

Use `--model tahag` to run the TAHAG baseline. The adapter converts each normalized 1 s EEG window into 5-band log-power node features, then trains an adaptive graph network with source-label classification, target-unlabeled gradient-reversal domain alignment, and MMD alignment. This is separate from both TSMNet SPDDSBN and BF-GCN transfer code.

Use `--model mdtn` to run the MDTN-GMDA baseline from `MDTN-GMDA/Net.py`. The adapter preserves the intended components in that reference: multi-scale temporal convolution, dynamic attention, contextual gating, warm-start GRL, Chebyshev graph discriminator, graph matching, and marginal/conditional MMD. Target/test windows are used without labels for the MDTN-GMDA domain losses only. Tunable parameters include `--mdtn-hidden-dim`, `--mdtn-num-nodes`, `--mdtn-kernel-length`, `--mdtn-num-heads`, `--mdtn-cheby-order`, `--mdtn-dropout`, `--mdtn-lambda-match`, `--mdtn-marginal-weight`, `--mdtn-conditional-weight`, and `--mdtn-l1-weight`.

Use `--model ms_tgc_spddsbn` to run the fusion model. It combines TSMNet-SPDDSBN's SPD manifold branch with MDTN-style multi-scale temporal features and a Chebyshev graph-convolution feature encoder. The two branches are fused by projected latent features and a learnable gate before classification. It intentionally does not use MDTN-GMDA's domain-adversarial, MMD, or graph-matching losses; training uses source cross entropy only, and target data is used only by SPDDSBN refitting. Tunable parameters include `--mstgc-temporal-hidden`, `--mstgc-graph-hidden`, `--mstgc-fusion-dim`, `--mstgc-kernel-length`, `--mstgc-num-heads`, `--mstgc-cheby-order`, `--mstgc-dropout`, and `--mstgc-num-nodes`.

MS_TGC_SPDDSBN ablations are implemented in the same `src/cl_tsmnet/ms_tgc_spddsbn.py` file and use the same training/evaluation pipeline: `mstgc_dta_ce`, `mstgc_dta_cheb_ce`, `mstgc_dta_cheb_eudsbn`, `mstgc_dta_cheb_spdbn`, `ms_tgc_spddsbn`, `mstgc_wo_dta`, `mstgc_wo_cheb`, and `mstgc_wo_spddsbn`. Only Euclidean DSBN and SPDDSBN variants use unlabeled target windows for post-training BN-statistic refitting.

Use `--model lsccn` to run the LSCCN baseline from `Latent_Space_Coding_Capsule_Network_for_Mental_Workload_Classification.docx`. The adapter computes the paper-style fused feature matrix inside the DataLoader: 5-band log-power node features are concatenated with gamma-band PLV connectivity, then passed through a VAE encoder, 1D convolution, primary capsules, and dynamic-routing digit capsules. The default latent dimension is 200 and routing iterations are 3, matching the paper. LSCCN is source-only in this project; target windows are not used during training.
For binary LSCCN runs, the final capsule score decision threshold is calibrated on validation windows only and then fixed for train/validation/test reporting; the threshold is saved as `decision_threshold` in `summary.csv`.

Use `--model lstm`, `--model bilstm`, `--model transformer`, or `--model shallowcnn` for source-only temporal baselines over normalized 1 s EEG windows. LSTM/BiLSTM consume sequences shaped as time steps by channels; Transformer uses channel projection, learnable positional embeddings, and encoder blocks; ShallowCNN follows the classic temporal-convolution plus spatial-convolution EEG baseline pattern. Their parameters are exposed in both run scripts.

Current validation notes:

- `single_session` is the original sequential 70%/10%/20% split, not 5-fold CV.
- Normalization is fitted only on source training windows inside each split.
- EEG-Conformer, EEGNet, BF-GCN, TAHAG, MDTN-GMDA, MS_TGC_SPDDSBN, SVM, LSCCN, LSTM, BiLSTM, Transformer, and ShallowCNN command-line hyperparameters are passed into their model/training code.
- `__pycache__/`, `*.pyc`, `outputs/`, and `data/` are generated/local artifacts and should not be part of the reproducible source package.

Cache files are literal filtered/resampled window datasets. Rebuild old caches once because older caches may contain record-level standardization and are intentionally rejected by the strict loader. Current COG-BCI caches also store recording-subject coverage and known no-window exclusions. Name caches by dataset, paradigm, subject scope, and sampling rate, for example `cog_nback_all_250hz_1s.npz` or `stew_all_128hz_1s.npz`.
