#!/usr/bin/env bash
# Re-run AdaLoRA on the full-budget E1 table with checkpoint saving + true
# active_total_rank logging (old runs only saved eval_results / training_log with
# a constant rank-step proxy and no best_model adapter).
#
# Only adalora is re-run; lora/gora/goodput results are kept. Old adalora dirs
# are removed first so summaries aggregate only the new runs.
#
# NOTE: not using `set -e`; a single failed job never aborts the sweep.
# Do NOT edit run_multiseed.sh / run_glue.sh while this is running.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

GPUS="${GPUS:-4 7 8 9}"
SEEDS="${SEEDS:-42 1 2}"
OUT_PREFIX="${OUT_PREFIX:-outputs}"

# Small tasks first for fast signal; large tasks use fewer epochs (same as E1).
TASKS=(rte mrpc cola stsb sst2 qnli mnli qqp)

START_TS=$(date +%s)
for task in "${TASKS[@]}"; do
  if [ "${task}" = "mnli" ] || [ "${task}" = "qqp" ] || [ "${task}" = "qnli" ] || [ "${task}" = "sst2" ]; then
    ep=3
  else
    ep=30
  fi
  old="${OUT_PREFIX}/${task}/adalora"
  if [ -d "${old}" ]; then
    echo "[adalora-rerun] removing old ${old} (no checkpoint / wrong rank log)"
    rm -rf "${old}"
  fi
  echo "[adalora-rerun] ===== task=${task} epochs=${ep} $(date '+%F %T') ====="
  OUT_PREFIX="${OUT_PREFIX}" EPOCHS="${ep}" \
    bash scripts/run_multiseed.sh "${task}" "adalora" "${SEEDS}" "${GPUS}"
  python scripts/aggregate_results.py --task "${task}" --outputs "${OUT_PREFIX}" \
    || echo "[adalora-rerun] aggregate failed for ${task}"
  python scripts/efficiency_report.py --task "${task}" --outputs "${OUT_PREFIX}" \
    || echo "[adalora-rerun] efficiency report failed for ${task}"
  python scripts/effective_rank.py --task "${task}" --outputs "${OUT_PREFIX}" \
    || echo "[adalora-rerun] effective_rank failed for ${task}"
  echo "[adalora-rerun] ===== DONE task=${task} $(date '+%F %T') ====="
done
echo "[adalora-rerun] ALL ADALORA RE-RUNS DONE in $(( $(date +%s) - START_TS ))s"
