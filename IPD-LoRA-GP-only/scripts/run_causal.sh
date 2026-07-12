#!/usr/bin/env bash
# Single causal-LM (LLaMA + QLoRA) run for one method on one GPU.
# Mirrors run_glue.sh dispatch style but for train_causal.py / train_adalora_causal.py.
#
# Usage:
#   bash scripts/run_causal.sh <method> <task> <seed> <gpu> [out_prefix]
#   method: goodput | lora | gora | adalora
#   task:   commonsense | gsm8k
#
# Env overrides (all optional):
#   MODEL_PATH MAX_TRAIN_SAMPLES EVAL_PER_TASK EVAL_SUBSET
#   TARGET_RANK MAX_LORA_RANK INITIAL_ACTIVE_RANK
#   BATCH EVAL_BATCH ACCUM EPOCHS LR MAX_NEW_TOKENS
set -uo pipefail

METHOD="${1:?method required}"
TASK="${2:?task required}"
SEED="${3:-42}"
GPU="${4:?gpu required}"
OUT_PREFIX="${5:-outputs_causal}"

PY="${PY:-/data/shenth/miniconda3/envs/MI/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data/shenth/models/llama/3.1-8b}"
DATASET_ROOT="${DATASET_ROOT:-/data/shenth/datasets}"

# budget defaults (avg rank 16 across all injected modules)
TARGET_RANK="${TARGET_RANK:-16}"
MAX_LORA_RANK="${MAX_LORA_RANK:-32}"
INITIAL_ACTIVE_RANK="${INITIAL_ACTIVE_RANK:-${TARGET_RANK}}"
LORA_ALPHA="${LORA_ALPHA:-32}"

BATCH="${BATCH:-8}"
EVAL_BATCH="${EVAL_BATCH:-16}"
ACCUM="${ACCUM:-2}"
EPOCHS="${EPOCHS:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"

# data sizing (0 = all). Keep eval bounded so generation stays cheap.
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-0}"
EVAL_PER_TASK="${EVAL_PER_TASK:-0}"
EVAL_SUBSET="${EVAL_SUBSET:-0}"

# per-method LR (AdaLoRA/SVD wants a higher LR).
if [[ "${METHOD}" == "adalora" ]]; then
  LR="${LR:-3e-4}"
  MAX_NEW_TOKENS="${MAX_NEW_TOKENS}"
else
  LR="${LR:-2e-4}"
fi

if [[ "${TASK}" == "gsm8k" ]]; then
  MAX_NEW_TOKENS="${MAX_NEW_TOKENS_GSM:-256}"
fi

OUT_DIR="${OUT_PREFIX}/${TASK}/${METHOD}/seed${SEED}"
mkdir -p "${OUT_DIR}"
echo "[run_causal] method=${METHOD} task=${TASK} seed=${SEED} gpu=${GPU} -> ${OUT_DIR}"

COMMON_ARGS=(
  --task_name "${TASK}"
  --dataset_root "${DATASET_ROOT}"
  --model_name_or_path "${MODEL_PATH}"
  --output_dir "${OUT_DIR}"
  --bf16 --load_in_4bit --gradient_checkpointing
  --per_device_train_batch_size "${BATCH}"
  --per_device_eval_batch_size "${EVAL_BATCH}"
  --gradient_accumulation_steps "${ACCUM}"
  --num_train_epochs "${EPOCHS}"
  --learning_rate "${LR}"
  --seed "${SEED}"
  --target_rank "${TARGET_RANK}"
  --max_lora_rank "${MAX_LORA_RANK}"
  --lora_alpha "${LORA_ALPHA}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --max_train_samples "${MAX_TRAIN_SAMPLES}"
  --eval_per_task "${EVAL_PER_TASK}"
  --eval_subset "${EVAL_SUBSET}"
)

if [[ "${METHOD}" == "adalora" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" train_adalora_causal.py "${COMMON_ARGS[@]}" \
    > "${OUT_DIR}/train.log" 2>&1
else
  CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" train_causal.py "${COMMON_ARGS[@]}" \
    --method "${METHOD}" \
    --initial_active_rank "${INITIAL_ACTIVE_RANK}" \
    --goodput_method proxy \
    > "${OUT_DIR}/train.log" 2>&1
fi

STATUS=$?
if [[ ${STATUS} -eq 0 ]]; then
  echo "[run_causal] DONE method=${METHOD} task=${TASK} seed=${SEED}"
else
  echo "[run_causal] FAILED (exit ${STATUS}) method=${METHOD} task=${TASK} seed=${SEED}; see ${OUT_DIR}/train.log"
fi
exit ${STATUS}
