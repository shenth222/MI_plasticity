#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Optional overrides:
#   CUDA_VISIBLE_DEVICES=0 MODEL_PATH=/path/to/model DATA_ROOT=/path/to/glue bash run_train_examples.sh mnli
export CUDA_VISIBLE_DEVICES=0
MODEL_PATH="/data/shenth/models/deberta/v3-base"
DATA_ROOT="/data/shenth/datasets/glue"

# Usage:
#   bash run_train_examples.sh rte
#   bash run_train_examples.sh mnli
#   bash run_train_examples.sh local

MODE="${1:-mnli}"

if [[ "${MODE}" == "rte" ]]; then
  python train_ipd_lora.py \
    --task_name rte \
    --dataset_path "${DATA_ROOT}" \
    --model_name_or_path "${MODEL_PATH}" \
    --output_dir outputs/ipd_lora_rte_seed42_v2 \
    --max_length 128 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --learning_rate 2e-4 \
    --weight_decay 0.01 \
    --num_train_epochs 10 \
    --warmup_ratio 0.06 \
    --seed 42 \
    --max_lora_rank 12 \
    --initial_active_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --score_interval 100 \
    --importance_update_interval 2 \
    --importance_exact_interval 6 \
    --importance_group_size 4 \
    --score_module_batch_size 8 \
    --warmup_steps_for_ipd 100 \
    --calibration_size 256 \
    --calibration_resample_stride 9973 \
    --total_rank_budget 0 \
    --target_rank 4 \
    --avoid_zero_rank \
    --beta_I 0.9 \
    --beta_P 0.9 \
    --tfinal_ratio 0.15 \
    --logging_steps 20 \
    --evaluation_strategy epoch \
    --early_stop_patience 3 \
    --early_stop_i_tolerance 1e-4 \
    --early_stop_unfreeze_interval 3 \
    --early_stop_max_freeze_cycles 3 \
    --early_stop_unfreeze_rank 4 \
    --report_to_wandb \
    --wandb_project ipd-lora \
    --wandb_run_name ipd-lora-rte-seed42
elif [[ "${MODE}" == "mnli" ]]; then
  python train_ipd_lora.py \
    --task_name mnli \
    --dataset_path "${DATA_ROOT}" \
    --model_name_or_path "${MODEL_PATH}" \
    --output_dir outputs/ipd_lora_mnli_seed42_r8 \
    --max_length 128 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 64 \
    --learning_rate 2e-4 \
    --weight_decay 0.01 \
    --num_train_epochs 7 \
    --warmup_ratio 0.06 \
    --seed 42 \
    --max_lora_rank 12 \
    --initial_active_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --score_interval 500 \
    --importance_update_interval 2 \
    --importance_exact_interval 8 \
    --importance_group_size 6 \
    --score_module_batch_size 12 \
    --warmup_steps_for_ipd 500 \
    --calibration_size 1024 \
    --calibration_resample_stride 9973 \
    --total_rank_budget 0 \
    --target_rank 8 \
    --avoid_zero_rank \
    --beta_I 0.9 \
    --beta_P 0.9 \
    --tfinal_ratio 0.1 \
    --logging_steps 50 \
    --evaluation_strategy epoch \
    --early_stop_patience 3 \
    --early_stop_i_tolerance 1e-4 \
    --early_stop_unfreeze_interval 3 \
    --early_stop_max_freeze_cycles 2 \
    --early_stop_unfreeze_rank 1 \
    --report_to_wandb \
    --wandb_project ipd-lora \
    --wandb_run_name ipd-lora-mnli-seed42
elif [[ "${MODE}" == "local" ]]; then
  # Example 1: local dataset saved by datasets.save_to_disk(...)
  # python train_ipd_lora.py \
  #   --task_name rte \
  #   --dataset_path /path/to/local_dataset_disk \
  #   --local_train_split train \
  #   --local_eval_split validation \
  #   --text_column1 sentence1 \
  #   --text_column2 sentence2 \
  #   --label_column label \
  #   --model_name_or_path /path/to/local_or_hf_model \
  #   --output_dir outputs/ipd_lora_local
  #
  # Example 2: local csv/json/jsonl files
  python train_ipd_lora.py \
    --task_name rte \
    --train_file /path/to/train.jsonl \
    --validation_file /path/to/validation.jsonl \
    --text_column1 sentence1 \
    --text_column2 sentence2 \
    --label_column label \
    --model_name_or_path /path/to/local_or_hf_model \
    --output_dir outputs/ipd_lora_local \
    --max_length 128 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 32 \
    --learning_rate 2e-4 \
    --weight_decay 0.01 \
    --num_train_epochs 3 \
    --warmup_ratio 0.06 \
    --seed 42 \
    --max_lora_rank 16 \
    --initial_active_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --score_interval 100 \
    --importance_update_interval 2 \
    --importance_exact_interval 6 \
    --importance_group_size 4 \
    --score_module_batch_size 8 \
    --warmup_steps_for_ipd 100 \
    --calibration_size 256 \
    --calibration_resample_stride 9973 \
    --total_rank_budget 0 \
    --target_rank 4 \
    --avoid_zero_rank \
    --beta_I 0.9 \
    --beta_P 0.9 \
    --tfinal_ratio 0.1 \
    --logging_steps 20 \
    --eval_steps 100 \
    --early_stop_patience 3 \
    --early_stop_i_tolerance 1e-4 \
    --early_stop_unfreeze_interval 3 \
    --early_stop_max_freeze_cycles 2 \
    --early_stop_unfreeze_rank 1 \
    --report_to_wandb \
    --wandb_project ipd-lora \
    --wandb_run_name ipd-lora-local-seed42
else
  echo "Unknown mode: ${MODE}"
  echo "Supported: rte | mnli | local"
  exit 1
fi

