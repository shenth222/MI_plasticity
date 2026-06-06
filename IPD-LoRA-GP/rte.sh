#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Optional overrides:
#   CUDA_VISIBLE_DEVICES=0 MODEL_PATH=/path/to/model DATA_ROOT=/path/to/glue bash run_train_examples.sh mnli
export CUDA_VISIBLE_DEVICES=1
MODEL_PATH="/data/shenth/models/deberta/v3-base"
DATA_ROOT="/data/shenth/datasets/glue"

# Usage:
#   bash run_train_examples.sh rte
#   bash run_train_examples.sh mnli
#   bash run_train_examples.sh local
#
# Note:
#   train_ipd_lora.py currently injects LoRA into DeBERTa-v3-base
#   q/k/v/o attention projections + FFN (intermediate/output) dense layers.

MODE="rte"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="outputs/ipd_lora_rte_seed42_v4_${RUN_TAG}"

python train_ipd_lora.py \
    --task_name rte \
    --dataset_path "${DATA_ROOT}" \
    --model_name_or_path "${MODEL_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --max_length 256 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --learning_rate 2e-4 \
    --weight_decay 0.01 \
    --num_train_epochs 50 \
    --warmup_ratio 0.1 \
    --seed 42 \
    --max_lora_rank 12 \
    --initial_active_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --score_interval 100 \
    --importance_update_interval 1 \
    --importance_exact_interval 4 \
    --importance_group_size 3 \
    --score_module_batch_size 12 \
    --min_importance_scores_per_module 2 \
    --warmup_steps_for_ipd 100 \
    --calibration_size 192 \
    --calibration_resample_stride 9973 \
    --total_rank_budget 0 \
    --target_rank 6 \
    --avoid_zero_rank \
    --beta_I 0.9 \
    --beta_P 0.9 \
    --plasticity_task_weight 0.1 \
    --high_i_quantile 0.5 \
    --high_p_quantile 0.5 \
    --low_i_low_p_update_interval 32 \
    --tfinal_ratio 0.15 \
    --logging_steps 20 \
    --evaluation_strategy epoch \
    --disable_module_early_stop \
    --report_to_wandb \
    --wandb_project ipd-lora \
    --wandb_run_name "ipd-lora-rte-seed42-v4-${RUN_TAG}"
