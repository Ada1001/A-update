#!/usr/bin/env bash
# Run from the project root. Each checkpoint is newly trained for time_blocks_v1.
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
SUBJECT="${SUBJECT:-21}"
RUN_ROOT="${RUN_ROOT:-outputs/stew_single_adapt_v1}"
RESULT_ROOT="${RESULT_ROOT:-results/stew_single_adapt_v1}"
EPOCHS="${EPOCHS:-30}"
if [[ -e "$RUN_ROOT/master_summary.csv" ]]; then
  echo "Existing experiment found. Choose a fresh RUN_ROOT and RESULT_ROOT to preserve checkpoints." >&2
  exit 1
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
mkdir -p "$RUN_ROOT" "$RESULT_ROOT"
train_subject=(--subject "$SUBJECT")
folds=1
representative="$SUBJECT"
if [[ "$SUBJECT" == "all" ]]; then
  train_subject=()
  folds=48
  representative=21
fi
for model in mstgc_mean_ce mstgc_dta_cheb_eudsbn mstgc_dta_cheb_spdbn ms_tgc_spddsbn tsmnet; do
  "$PYTHON_BIN" -u run_experiment.py --dataset stew --protocol single_session \
    "${train_subject[@]}" --model "$model" --bnorm spddsbn \
    --tsmnet-bn-schedule constant --data-root data --cache-root outputs/cache \
    --output "$RUN_ROOT" --master-summary "$RUN_ROOT/master_summary.csv" \
    --epochs "$EPOCHS" --patience 8 --batch-size 16 --refit-batch-size 16 \
    --lr 0.001 --weight-decay 0.0001 --seed 42 --single-val-size 0.125 \
    --test-size 0.2 --no-augment
done
for model in ms_tgc_spddsbn tsmnet; do
  "$PYTHON_BIN" -u analysis/fig_spddsbn_paired_mechanism_analysis.py \
    --protocol single_session --model "$model" --stage all --subjects "$SUBJECT" \
    --expected-folds "$folds" --representative-subject "$representative" \
    --output-root "$RUN_ROOT" --master-summary "$RUN_ROOT/master_summary.csv" \
    --data-root data --cache-root outputs/cache --device cpu --batch-size 16 \
    --output-dir "$RESULT_ROOT/fig4_$model"
  "$PYTHON_BIN" -u analysis/fig5_representation_alignment.py \
    --protocol single_session --datasets stew --dataset-labels STEW \
    --fourth-model "$model" --target-subjects "stew=$representative" \
    --output-root "$RUN_ROOT" --master-summary "$RUN_ROOT/master_summary.csv" \
    --data-root data --cache-root outputs/cache --source-calibration refit \
    --metric-scope all --feature-location representation --reducer tsne --pca-dim 210 \
    --seed 42 --device cpu --batch-size 16 --max-points-per-group 200 \
    --output-dir "$RESULT_ROOT/fig5_$model" \
    --feature-cache-dir "$RESULT_ROOT/fig5_cache"
done
