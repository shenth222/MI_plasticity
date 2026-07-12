#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Usage:
#   bash plot_gp_only_results.sh outputs/gp_only_lora_rte_probing_xxx
#   bash plot_gp_only_results.sh all
#
# Optional:
#   TOP_K_MODULES=24 bash plot_gp_only_results.sh all

TARGET="${1:-all}"
TOP_K_MODULES="${TOP_K_MODULES:-20}"

plot_one() {
  local run_dir="$1"
  if [[ ! -f "${run_dir}/training_log.jsonl" && ! -f "${run_dir}/module_scores.jsonl" ]]; then
    echo "[skip] ${run_dir}: missing training_log.jsonl/module_scores.jsonl"
    return 0
  fi
  echo "[plot] ${run_dir}"
  python plot_gp_only_results.py \
    --output_dir "${run_dir}" \
    --top_k_modules "${TOP_K_MODULES}"
}

if [[ "${TARGET}" == "all" ]]; then
  shopt -s nullglob
  run_dirs=(outputs/gp_only_lora_*)
  if [[ ${#run_dirs[@]} -eq 0 ]]; then
    echo "[warn] no outputs/gp_only_lora_* directories found"
    exit 0
  fi
  for run_dir in "${run_dirs[@]}"; do
    if [[ -d "${run_dir}" ]]; then
      plot_one "${run_dir}"
    fi
  done
else
  plot_one "${TARGET}"
fi
