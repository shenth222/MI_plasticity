#!/usr/bin/env bash
# Sequential task queue for the unified GLUE harness.
#
# Waits for prerequisite sweeps (by PID) to release the GPUs, then runs a list
# of tasks one after another. Each task is a full 4-method x N-seed sweep across
# all provided GPUs, followed by aggregation. A failure in one task does NOT
# abort the queue.
#
# NOTE: not using `set -e` on purpose so a single failed task/job never kills
# the whole overnight queue. Do NOT edit run_multiseed.sh / run_glue.sh while
# this queue is running (bash reads scripts by byte offset).
#
# Usage:
#   WAIT_PIDS="123 456" bash scripts/run_queue.sh
#
# Env overrides:
#   GPUS     (default "4 5 6 7 8 9")
#   METHODS  (default "lora gora goodput adalora")
#   SEEDS    (default "42 1 2")
#   WAIT_PIDS(space-separated PIDs to wait on before starting; optional)
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

GPUS="${GPUS:-4 5 6 7 8 9}"
METHODS="${METHODS:-lora gora goodput adalora}"
SEEDS="${SEEDS:-42 1 2}"

# Task queue: "task:epochs" pairs, run in this order (small -> large).
TASK_QUEUE=(
  "stsb:30"
  "sst2:3"
  "qnli:3"
  "mnli:3"
  "qqp:3"
)

for pid in ${WAIT_PIDS:-}; do
  echo "[queue] waiting for prerequisite pid ${pid} to finish ..."
  while kill -0 "${pid}" 2>/dev/null; do sleep 30; done
  echo "[queue] prerequisite pid ${pid} finished."
done

echo "[queue] starting task queue on GPUs: ${GPUS}"
echo "[queue] methods=${METHODS} seeds=${SEEDS}"
START_TS=$(date +%s)

for entry in "${TASK_QUEUE[@]}"; do
  task="${entry%%:*}"
  epochs="${entry##*:}"
  echo "[queue] ===== START task=${task} epochs=${epochs} $(date '+%F %T') ====="
  EPOCHS="${epochs}" bash scripts/run_multiseed.sh "${task}" "${METHODS}" "${SEEDS}" "${GPUS}"
  echo "[queue] aggregating ${task} ..."
  python scripts/aggregate_results.py --task "${task}" || echo "[queue] aggregate failed for ${task}"
  echo "[queue] ===== DONE task=${task} $(date '+%F %T') ====="
done

ELAPSED=$(( $(date +%s) - START_TS ))
echo "[queue] ALL TASKS DONE in ${ELAPSED}s. Summaries: outputs/<task>/summary.md"
