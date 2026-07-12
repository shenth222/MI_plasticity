#!/usr/bin/env bash
# Parallel causal-LM sweep: run several methods for one task/seed, one method
# per GPU. Fault-tolerant (a failed method does not abort the others).
#
# Usage:
#   bash scripts/run_causal_sweep.sh <task> <seed> [out_prefix]
# Env:
#   GPUS="5 6 7 8"                      # GPUs (one method each, round-robin)
#   METHODS="goodput lora gora adalora"
#   plus any env consumed by run_causal.sh (MAX_TRAIN_SAMPLES, EVAL_PER_TASK, ...)
set -uo pipefail

TASK="${1:?task required (commonsense|gsm8k)}"
SEED="${2:-42}"
OUT_PREFIX="${3:-outputs_causal}"

read -r -a GPUS <<< "${GPUS:-5 6 7 8}"
read -r -a METHODS <<< "${METHODS:-goodput lora gora adalora}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${HERE}"

echo "[sweep] task=${TASK} seed=${SEED} methods=(${METHODS[*]}) gpus=(${GPUS[*]})"
declare -a PIDS METHOD_OF
i=0
for m in "${METHODS[@]}"; do
  gpu="${GPUS[$(( i % ${#GPUS[@]} ))]}"
  echo "[sweep] launching ${m} on GPU ${gpu}"
  bash scripts/run_causal.sh "${m}" "${TASK}" "${SEED}" "${gpu}" "${OUT_PREFIX}" &
  PIDS+=($!)
  METHOD_OF+=("${m}")
  i=$(( i + 1 ))
  sleep 10   # stagger model loads to avoid disk/IO spikes
done

FAIL=0
for idx in "${!PIDS[@]}"; do
  if wait "${PIDS[$idx]}"; then
    echo "[sweep] OK: ${METHOD_OF[$idx]}"
  else
    echo "[sweep] FAILED: ${METHOD_OF[$idx]}"
    FAIL=$(( FAIL + 1 ))
  fi
done

echo "[sweep] finished task=${TASK} seed=${SEED} failures=${FAIL}"
exit 0
