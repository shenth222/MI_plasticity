#!/usr/bin/env bash
set -euo pipefail

# Unified GLUE harness runner (single run).
#
# Usage:
#   bash scripts/run_glue.sh <method> <task> [seed]
#     method : goodput | lora | gora | adalora
#     task   : rte | mrpc | stsb | cola | wnli | sst2 | qnli | qqp | mnli
#     seed   : integer (default 42)
#
# Env overrides:
#   CUDA_VISIBLE_DEVICES, MODEL_PATH, DATA_ROOT, EPOCHS, LR, LR_ADALORA,
#   TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE, CALIBRATION_SIZE,
#   TARGET_RANK, MAX_LORA_RANK, INITIAL_ACTIVE_RANK, WANDB_MODE
#
# Example:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_glue.sh goodput rte 42
#   CUDA_VISIBLE_DEVICES=1 bash scripts/run_glue.sh lora    rte 42

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

METHOD="${1:?method required: goodput|lora|gora|adalora}"
TASK="${2:?task required: rte|mrpc|stsb|cola|wnli|sst2|qnli|qqp|mnli}"
SEED="${3:-42}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MODEL_PATH="${MODEL_PATH:-/data/shenth/models/deberta/v3-base}"
DATA_ROOT="${DATA_ROOT:-/data/shenth/datasets/glue}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
CALIBRATION_SIZE="${CALIBRATION_SIZE:-128}"
TARGET_RANK="${TARGET_RANK:-6}"
MAX_LORA_RANK="${MAX_LORA_RANK:-12}"
# Goodput's warmup start rank. It MUST scale with the budget: a fixed value
# (previously 8) makes goodput start at 8*72 total rank, which at a low target
# budget (e.g. target=2 -> budget 144) is a 4x over-allocation during warmup —
# an unfair, budget-violating confound. Default to the per-module target so
# warmup spends exactly the budget, then reallocation only redistributes it.
INITIAL_ACTIVE_RANK="${INITIAL_ACTIVE_RANK:-${TARGET_RANK}}"
LR="${LR:-2e-4}"
WANDB_MODE="${WANDB_MODE:-disabled}"

# AdaLoRA's SVD reparam + orthogonal regularization needs a higher LR than
# vanilla LoRA to learn at all (at 2e-4 it collapses to the majority class on
# RTE; at 1e-3 it reaches ~88% on RTE). Use its own tuned LR; override with
# LR_ADALORA.
if [[ "${METHOD}" == "adalora" ]]; then
  LR="${LR_ADALORA:-1e-3}"
fi

# Per-task default epochs (small tasks need more epochs; large tasks fewer).
case "${TASK}" in
  rte|mrpc|stsb|cola|wnli) DEFAULT_EPOCHS=30 ;;
  sst2|qnli|qqp|mnli)      DEFAULT_EPOCHS=10 ;;
  *)                       DEFAULT_EPOCHS=20 ;;
esac
EPOCHS="${EPOCHS:-${DEFAULT_EPOCHS}}"

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
# OUT_PREFIX lets a sweep write to a separate results root (e.g. a low-budget
# experiment) so its runs don't mix with the default budget in outputs/.
OUT_PREFIX="${OUT_PREFIX:-outputs}"
OUT_DIR="${OUT_PREFIX}/${TASK}/${METHOD}/seed${SEED}_${RUN_TAG}"

# Flags shared by every method (same data / budget / eval / logging protocol).
BASE_ARGS=(
  --task_name "${TASK}"
  --dataset_path "${DATA_ROOT}"
  --model_name_or_path "${MODEL_PATH}"
  --output_dir "${OUT_DIR}"
  --max_length 256
  --bf16
  --per_device_train_batch_size "${TRAIN_BATCH_SIZE}"
  --per_device_eval_batch_size "${EVAL_BATCH_SIZE}"
  --learning_rate "${LR}"
  --weight_decay 0.01
  --num_train_epochs "${EPOCHS}"
  --warmup_ratio 0.1
  --seed "${SEED}"
  --max_lora_rank "${MAX_LORA_RANK}"
  --lora_alpha 16
  --lora_dropout 0.05
  --target_rank "${TARGET_RANK}"
  --tfinal_ratio 0.15
  --logging_steps 20
  --evaluation_strategy epoch
  --wandb_mode "${WANDB_MODE}"
  --wandb_project ipd-lora-harness
  --wandb_run_name "${METHOD}-${TASK}-seed${SEED}-${RUN_TAG}"
)
if [[ "${WANDB_MODE}" != "disabled" ]]; then
  BASE_ARGS+=(--report_to_wandb)
fi

echo "[run] method=${METHOD} task=${TASK} seed=${SEED} epochs=${EPOCHS} gpu=${CUDA_VISIBLE_DEVICES} -> ${OUT_DIR}"

case "${METHOD}" in
  goodput)
    python train_ipd_lora.py "${BASE_ARGS[@]}" \
      --method goodput --initial_active_rank "${INITIAL_ACTIVE_RANK}" --total_rank_budget 0 \
      --score_interval 100 --warmup_steps_for_ipd 100 \
      --calibration_size "${CALIBRATION_SIZE}" --calibration_resample_stride 9973 \
      --goodput_method proxy --goodput_every_n_scoring 1 --beta_G 0.9 --goodput_min_rank 1
    ;;
  gora)
    # gora reallocates by pre-train importance immediately, so init rank is only
    # the transient starting point; keep it consistent with the budget anyway.
    python train_ipd_lora.py "${BASE_ARGS[@]}" \
      --method gora --initial_active_rank "${INITIAL_ACTIVE_RANK}" --total_rank_budget 0 \
      --gora_num_batches 16 --goodput_min_rank 1
    ;;
  lora)
    # lora sets a uniform rank = target immediately, so init rank is irrelevant.
    python train_ipd_lora.py "${BASE_ARGS[@]}" \
      --method lora --initial_active_rank "${INITIAL_ACTIVE_RANK}" --total_rank_budget 0
    ;;
  adalora)
    python train_adalora.py "${BASE_ARGS[@]}" \
      --adalora_tinit_ratio 0.1 --adalora_deltaT 10 --adalora_orth_reg_weight 0.5
    ;;
  *)
    echo "Unknown method: ${METHOD} (expected goodput|lora|gora|adalora)" >&2
    exit 1
    ;;
esac
