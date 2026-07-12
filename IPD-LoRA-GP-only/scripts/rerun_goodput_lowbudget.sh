#!/usr/bin/env bash
# Clean re-run of the goodput method on the low-budget configs after fixing the
# initial_active_rank confound (it used to start at 8*72 total rank regardless of
# budget; now it starts at target*72 = exactly the budget).
#
# Only goodput is re-run. lora/gora/adalora low-budget results are valid (they do
# not use initial_active_rank for warmup) and are kept. Old (confounded) goodput
# dirs are deleted first so aggregation averages only the clean runs.
#
# NOTE: not using `set -e`; a single failed job never aborts the sweep.
# Do NOT edit run_multiseed.sh / run_glue.sh while this is running.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

GPUS="${GPUS:-0 1 2 3 4 7 8 9}"
SEEDS="${SEEDS:-42 1 2}"

# "prefix:target:max:tasks" — mirror run_budget_sweep.sh exactly.
CONFIGS=(
  "outputs/budget_r2:2:8:rte mrpc cola"
  "outputs/budget_r1:1:4:rte mrpc cola"
  "outputs/budget_r2:2:8:mnli"
)

START_TS=$(date +%s)
for cfg in "${CONFIGS[@]}"; do
  prefix="${cfg%%:*}"; rest="${cfg#*:}"
  target="${rest%%:*}"; rest="${rest#*:}"
  maxr="${rest%%:*}"; tasks="${rest#*:}"
  for task in ${tasks}; do
    if [ "${task}" = "mnli" ] || [ "${task}" = "qqp" ] || [ "${task}" = "qnli" ] || [ "${task}" = "sst2" ]; then
      ep=3
    else
      ep=30
    fi
    old="${prefix}/${task}/goodput"
    if [ -d "${old}" ]; then
      echo "[rerun] removing confounded ${old}"
      rm -rf "${old}"
    fi
    echo "[rerun] ===== prefix=${prefix} target=${target} max=${maxr} task=${task} epochs=${ep} init=${target} $(date '+%F %T') ====="
    OUT_PREFIX="${prefix}" TARGET_RANK="${target}" MAX_LORA_RANK="${maxr}" \
      INITIAL_ACTIVE_RANK="${target}" EPOCHS="${ep}" \
      bash scripts/run_multiseed.sh "${task}" "goodput" "${SEEDS}" "${GPUS}"
    python scripts/aggregate_results.py --task "${task}" --outputs "${prefix}" \
      || echo "[rerun] aggregate failed for ${task} under ${prefix}"
    echo "[rerun] ===== DONE prefix=${prefix} task=${task} $(date '+%F %T') ====="
  done
done
echo "[rerun] ALL GOODPUT LOW-BUDGET RE-RUNS DONE in $(( $(date +%s) - START_TS ))s"
