#!/usr/bin/env bash
# Low-budget controlled study (Plan B): does dynamic rank allocation beat static
# uniform LoRA when the rank budget is scarce?
#
# Each config runs a full 4-method x 3-seed sweep at a fixed budget, into a
# separate results root (OUT_PREFIX) so it never mixes with the default
# budget=432 results in outputs/. Aggregation runs after each task.
#
# Budget = TARGET_RANK * 72 modules. MAX_LORA_RANK is the per-module cap and is
# kept at ~4x target so dynamic methods have room to reallocate.
#
# NOTE: not using `set -e`; one failed job/task never aborts the whole sweep.
# Do NOT edit run_multiseed.sh / run_glue.sh while this is running.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

GPUS="${GPUS:-4 5 6 7 8 9}"
METHODS="${METHODS:-lora gora goodput adalora}"
SEEDS="${SEEDS:-42 1 2}"

# "prefix:target:max:tasks" — small tasks first (fast signal), MNLI last (slow).
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
    # Large tasks get fewer epochs (consistent with the full-budget MNLI run).
    if [ "${task}" = "mnli" ] || [ "${task}" = "qqp" ] || [ "${task}" = "qnli" ] || [ "${task}" = "sst2" ]; then
      ep=3
    else
      ep=30
    fi
    echo "[budget] ===== prefix=${prefix} target=${target} max=${maxr} task=${task} epochs=${ep} $(date '+%F %T') ====="
    OUT_PREFIX="${prefix}" TARGET_RANK="${target}" MAX_LORA_RANK="${maxr}" EPOCHS="${ep}" \
      bash scripts/run_multiseed.sh "${task}" "${METHODS}" "${SEEDS}" "${GPUS}"
    python scripts/aggregate_results.py --task "${task}" --outputs "${prefix}" \
      || echo "[budget] aggregate failed for ${task} under ${prefix}"
    echo "[budget] ===== DONE prefix=${prefix} task=${task} $(date '+%F %T') ====="
  done
done
echo "[budget] ALL BUDGET SWEEPS DONE in $(( $(date +%s) - START_TS ))s"
