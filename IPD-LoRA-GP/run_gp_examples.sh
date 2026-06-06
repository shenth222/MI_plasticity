#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# IPD-GP-LoRA (Goodput-aware IPD-LoRA) example runs.
#
# Usage:
#   bash run_gp_examples.sh proxy      # method A: low-cost online proxy goodput
#   bash run_gp_examples.sh probing    # method B: one-step probing goodput (main experiment)
#   bash run_gp_examples.sh baseline   # original IPD-LoRA quadrant allocation (no goodput)
#
# Optional overrides:
#   CUDA_VISIBLE_DEVICES=0 MODEL_PATH=/path DATA_ROOT=/path bash run_gp_examples.sh probing

export CUDA_VISIBLE_DEVICES=0
MODEL_PATH="${MODEL_PATH:-/data/shenth/models/deberta/v3-base}"
DATA_ROOT="${DATA_ROOT:-/data/shenth/datasets/glue}"

MODE="${1:-proxy}"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"

COMMON_ARGS=(
  --task_name rte
  --dataset_path "${DATA_ROOT}"
  --model_name_or_path "${MODEL_PATH}"
  --max_length 256
  --per_device_train_batch_size 32
  --per_device_eval_batch_size 32
  --learning_rate 2e-4
  --weight_decay 0.01
  --num_train_epochs 50
  --warmup_ratio 0.1
  --seed 42
  --max_lora_rank 12
  --initial_active_rank 8
  --lora_alpha 16
  --lora_dropout 0.05
  --score_interval 100
  --importance_update_interval 1
  --importance_exact_interval 4
  --importance_group_size 3
  --score_module_batch_size 12
  --min_importance_scores_per_module 2
  --warmup_steps_for_ipd 100
  --calibration_size 192
  --calibration_resample_stride 9973
  --total_rank_budget 0
  --target_rank 6
  --avoid_zero_rank
  --beta_I 0.9
  --beta_P 0.9
  --plasticity_task_weight 0.1
  --high_i_quantile 0.5
  --high_p_quantile 0.5
  --low_i_low_p_update_interval 32
  --tfinal_ratio 0.15
  --logging_steps 20
  --evaluation_strategy epoch
  --disable_module_early_stop
  --report_to_wandb
  --wandb_project ipd-gp-lora
)

if [[ "${MODE}" == "proxy" ]]; then
  python train_ipd_lora.py "${COMMON_ARGS[@]}" \
    --output_dir "outputs/ipd_gp_lora_rte_proxy_${RUN_TAG}" \
    --enable_goodput \
    --goodput_method proxy \
    --goodput_every_n_scoring 1 \
    --beta_G 0.9 \
    --goodput_alpha 1.0 \
    --goodput_beta 1.0 \
    --rank_alloc_mode goodput \
    --goodput_score_mode GxI \
    --goodput_min_rank 1 \
    --wandb_run_name "ipd-gp-lora-rte-proxy-${RUN_TAG}"
elif [[ "${MODE}" == "probing" ]]; then
  python train_ipd_lora.py "${COMMON_ARGS[@]}" \
    --output_dir "outputs/ipd_gp_lora_rte_probing_${RUN_TAG}" \
    --enable_goodput \
    --goodput_method probing \
    --goodput_every_n_scoring 2 \
    --beta_G 0.9 \
    --goodput_probe_max_batches 6 \
    --rank_alloc_mode goodput \
    --goodput_score_mode G \
    --goodput_min_rank 1 \
    --wandb_run_name "ipd-gp-lora-rte-probing-${RUN_TAG}"
elif [[ "${MODE}" == "baseline" ]]; then
  # Goodput is still measured/logged for comparison, but allocation stays quadrant-based.
  python train_ipd_lora.py "${COMMON_ARGS[@]}" \
    --output_dir "outputs/ipd_lora_rte_baseline_${RUN_TAG}" \
    --enable_goodput \
    --goodput_method proxy \
    --rank_alloc_mode quadrant \
    --wandb_run_name "ipd-lora-rte-baseline-${RUN_TAG}"
else
  echo "Unknown mode: ${MODE}"
  echo "Supported: proxy | probing | baseline"
  exit 1
fi
