#!/usr/bin/env bash
# Serial causal-LM pipeline: commonsense 4-method sweep -> aggregate -> gsm8k
# 4-method sweep -> aggregate. Each sweep runs its 4 methods in parallel across
# GPUS. Launch as ONE tool-managed background job (not nohup&, which gets reaped
# when the shell session ends).
#
# Usage: bash scripts/run_causal_queue.sh [seed] [out_prefix]
set -uo pipefail

SEED="${1:-42}"
OUT_PREFIX="${2:-outputs_causal}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${HERE}"
PY="${PY:-/data/shenth/miniconda3/envs/MI/bin/python}"
SWEEP_GPUS="${GPUS:-5 6 7 8}"

echo "[queue] === STAGE 1: commonsense sweep ==="
GPUS="${SWEEP_GPUS}" METHODS="goodput lora gora adalora" \
MAX_TRAIN_SAMPLES="${CS_TRAIN:-30000}" EVAL_PER_TASK="${CS_EVAL_PER_TASK:-300}" \
BATCH="${CS_BATCH:-8}" EVAL_BATCH="${CS_EVAL_BATCH:-16}" ACCUM="${CS_ACCUM:-2}" EPOCHS="${CS_EPOCHS:-1}" \
bash scripts/run_causal_sweep.sh commonsense "${SEED}" "${OUT_PREFIX}"

echo "[queue] aggregating commonsense ..."
"${PY}" scripts/aggregate_results.py --task commonsense --outputs "${OUT_PREFIX}" || true

echo "[queue] === STAGE 2: gsm8k sweep ==="
GPUS="${SWEEP_GPUS}" METHODS="goodput lora gora adalora" \
MAX_TRAIN_SAMPLES="${GSM_TRAIN:-0}" EVAL_SUBSET="${GSM_EVAL:-500}" \
BATCH="${GSM_BATCH:-4}" EVAL_BATCH="${GSM_EVAL_BATCH:-8}" ACCUM="${GSM_ACCUM:-4}" \
EPOCHS="${GSM_EPOCHS:-3}" MAX_NEW_TOKENS_GSM="${GSM_NEW_TOKENS:-256}" \
bash scripts/run_causal_sweep.sh gsm8k "${SEED}" "${OUT_PREFIX}"

echo "[queue] aggregating gsm8k ..."
"${PY}" scripts/aggregate_results.py --task gsm8k --outputs "${OUT_PREFIX}" || true

echo "[queue] ALL DONE"
