#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
# 说明：
# 1) MODEL_PATH 与 DATASET_PATH 必须是本地路径；
# 2) DATASET_PATH 需为 datasets.load_from_disk 保存的 GLUE 数据（含 train/validation）。

MODEL_PATH="/data/shenth/models/deberta/v3-base"
DATASET_PATH="/data/shenth/datasets/glue"
TASK_NAME="${1:-mnli}"
OUTPUT_ROOT="./outputs"

# 可选 rank：2/4/8/16。你可以通过环境变量 RANKS 覆盖，例如：RANKS="4 8"
RANKS=8

# W&B 配置（可通过环境变量覆盖）
WANDB_PROJECT="${WANDB_PROJECT:-lora-glue-local}"

for RANK in ${RANKS}; do
  RUN_NAME="deberta-v3-base-${TASK_NAME}-lora-r${RANK}-all"
  OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"

  echo "==> Start training: ${RUN_NAME}"
  python train_lora_glue.py \
    --model_path "${MODEL_PATH}" \
    --dataset_path "${DATASET_PATH}" \
    --task_name "${TASK_NAME}" \
    --output_dir "${OUTPUT_DIR}" \
    --rank "${RANK}" \
    --lora_alpha "$((RANK * 2))" \
    --lora_dropout 0.1 \
    --learning_rate 1e-4 \
    --num_train_epochs 5 \
    --max_length 256 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 32 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --logging_steps 20 \
    --seed 42 \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_run_name "${RUN_NAME}"
done

echo "All runs finished."
