#!/usr/bin/env bash
set -euo pipefail

# Multi-method x multi-seed GLUE sweep for one task, parallelized across GPUs.
#
# Usage:
#   bash scripts/run_multiseed.sh <task> [methods] [seeds] [gpus]
#     task    : rte | mrpc | ... | mnli   (required)
#     methods : space-separated, quoted   (default "lora gora goodput")
#     seeds   : space-separated, quoted   (default "42 1 2")
#     gpus    : space-separated, quoted   (default "0 1 2 3")
#
# Each (method, seed) job is pinned to one GPU round-robin and run in the
# background; the script waits for all jobs. Logs go to outputs/<task>/_logs/.
# A single failing job is recorded but does NOT abort the rest of the sweep.
#
# Example:
#   bash scripts/run_multiseed.sh rte "lora gora goodput adalora" "42 1 2" "0 1 2 3"
#
# IMPORTANT: do not edit this script or run_glue.sh while a sweep is running.
# bash reads scripts by byte offset, so editing a live script corrupts the
# running shell (it jumps to the wrong offset and errors at the tail).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

TASK="${1:?task required}"
METHODS="${2:-lora gora goodput}"
SEEDS="${3:-42 1 2}"
GPUS="${4:-0 1 2 3}"

read -r -a GPU_ARR <<< "${GPUS}"
N_GPU="${#GPU_ARR[@]}"
# Honor OUT_PREFIX (exported to run_glue.sh) so logs live alongside results.
export OUT_PREFIX="${OUT_PREFIX:-outputs}"
LOG_DIR="${OUT_PREFIX}/${TASK}/_logs"
mkdir -p "${LOG_DIR}"

fails=()

# Wait for every pid in the current batch; record failures but keep going.
drain() {
  local p rc
  for p in "$@"; do
    if wait "${p}"; then :; else
      rc=$?
      fails+=("pid=${p}:exit=${rc}")
      echo "[warn] a job (pid=${p}) exited with code ${rc}; continuing sweep."
    fi
  done
}

job_idx=0
pids=()
for method in ${METHODS}; do
  for seed in ${SEEDS}; do
    gpu="${GPU_ARR[$((job_idx % N_GPU))]}"
    log_file="${LOG_DIR}/${method}_seed${seed}_$(date +%Y%m%d_%H%M%S).log"
    echo "[launch] task=${TASK} method=${method} seed=${seed} gpu=${gpu} -> ${log_file}"
    CUDA_VISIBLE_DEVICES="${gpu}" bash scripts/run_glue.sh "${method}" "${TASK}" "${seed}" \
      > "${log_file}" 2>&1 &
    pids+=("$!")
    job_idx=$((job_idx + 1))
    # Throttle: never launch more concurrent jobs than available GPUs.
    if (( job_idx % N_GPU == 0 )); then
      drain "${pids[@]}"
      pids=()
    fi
  done
done

if (( ${#pids[@]} > 0 )); then
  drain "${pids[@]}"
fi

if (( ${#fails[@]} > 0 )); then
  echo "[done] task=${TASK} sweep finished with ${#fails[@]} failed job(s): ${fails[*]}"
  echo "       inspect logs in ${LOG_DIR}; re-run only the failed (method,seed) pairs."
else
  echo "[done] task=${TASK} sweep finished, all jobs OK. Logs in ${LOG_DIR}"
fi
