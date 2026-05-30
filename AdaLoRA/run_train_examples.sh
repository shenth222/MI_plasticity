#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
# 说明：
# 1) MODEL_PATH 与 DATASET_PATH 必须是本地路径；
# 2) DATASET_PATH 需为 datasets.load_from_disk 保存的 GLUE 数据（含 train/validation）。

MODEL_PATH="/data/shenth/models/deberta/v3-base"
DATASET_PATH="/data/shenth/datasets/glue"
TASK_NAME="${1:-mnli}"
OUTPUT_ROOT="./outputs"

# 可选目标 rank：2/4/8/12/16。可通过环境变量 TARGET_RANKS 覆盖，例如：TARGET_RANKS="4 8"
TARGET_RANKS=8
# AdaLoRA 初始 rank，通常大于等于目标 rank
INIT_RANK=12

# W&B 配置（可通过环境变量覆盖）
WANDB_PROJECT="${WANDB_PROJECT:-adalora-glue-local}"

for TARGET_RANK in ${TARGET_RANKS}; do
  RUN_NAME="deberta-v3-base-${TASK_NAME}-adalora-r${TARGET_RANK}-all"
  OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"

  echo "==> Start training: ${RUN_NAME}"
  python train_adalora_glue.py \
    --model_path "${MODEL_PATH}" \
    --dataset_path "${DATASET_PATH}" \
    --task_name "${TASK_NAME}" \
    --output_dir "${OUTPUT_DIR}" \
    --init_rank "${INIT_RANK}" \
    --target_rank "${TARGET_RANK}" \
    --lora_alpha "$((TARGET_RANK * 2))" \
    --lora_dropout 0.1 \
    --adalora_tinit 8000 \
    --adalora_tfinal 50000 \
    --adalora_delta_t 100 \
    --adalora_beta1 0.85 \
    --adalora_beta2 0.85 \
    --adalora_orth_reg_weight 0.1 \
    --learning_rate 5e-4 \
    --num_train_epochs 7 \
    --max_length 256 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --logging_steps 20 \
    --seed 42 \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_run_name "${RUN_NAME}"
done

echo "All runs finished."
