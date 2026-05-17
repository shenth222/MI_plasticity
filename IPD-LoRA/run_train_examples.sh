#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=3
set -euo pipefail

# Usage:
#   bash run_train_examples.sh rte
#   bash run_train_examples.sh mnli
#   bash run_train_examples.sh local

MODE="${1:-mnli}"

if [[ "${MODE}" == "rte" ]]; then
  python train_ipd_lora.py \
    --task_name rte \
    --dataset_name /data1/shenth/datasets/glue/rte \
    --model_name_or_path /data1/shenth/models/deberta/v3-base \
    --output_dir outputs/ipd_lora_rte_seed42 \
    --max_length 128 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --learning_rate 1.2e-3 \
    --weight_decay 0.01 \
    --num_train_epochs 50 \
    --warmup_ratio 0.06 \
    --seed 42 \
    --max_lora_rank 16 \
    --initial_active_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --score_interval 100 \
    --warmup_steps_for_ipd 100 \
    --calibration_size 256 \
    --total_rank_budget 96 \
    --beta_I 0.9 \
    --beta_P 0.9 \
    --logging_steps 20 \
    --eval_steps 100 \
    --report_to_wandb \
    --wandb_project ipd-lora \
    --wandb_run_name ipd-lora-rte-seed42
elif [[ "${MODE}" == "mnli" ]]; then
  python train_ipd_lora.py \
    --task_name mnli \
    --dataset_name /data1/shenth/datasets/glue/mnli \
    --model_name_or_path /data1/shenth/models/deberta/v3-base \
    --output_dir outputs/ipd_lora_mnli_seed42 \
    --max_length 128 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --learning_rate 5e-4 \
    --weight_decay 0.01 \
    --num_train_epochs 7 \
    --warmup_ratio 0.06 \
    --seed 42 \
    --max_lora_rank 16 \
    --initial_active_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --score_interval 500 \
    --warmup_steps_for_ipd 500 \
    --calibration_size 1024 \
    --total_rank_budget 192 \
    --beta_I 0.9 \
    --beta_P 0.9 \
    --logging_steps 50 \
    --eval_steps 1000 \
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
    --warmup_steps_for_ipd 100 \
    --calibration_size 256 \
    --total_rank_budget 96 \
    --beta_I 0.9 \
    --beta_P 0.9 \
    --logging_steps 20 \
    --eval_steps 100 \
    --report_to_wandb \
    --wandb_project ipd-lora \
    --wandb_run_name ipd-lora-local-seed42
else
  echo "Unknown mode: ${MODE}"
  echo "Supported: rte | mnli | local"
  exit 1
fi

