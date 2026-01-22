#!/bin/bash
#
# 运行 MNLI 任务
# 用法: bash scripts/run_mnli.sh <signal_type> <seed>
#
# 示例:
#   bash scripts/run_mnli.sh baseline_adalora 42
#   bash scripts/run_mnli.sh importance_only 42
#

set -e

# 参数
SIGNAL_TYPE=${1:-"baseline_adalora"}
SEED=${2:-42}

# 配置
TASK="mnli"
MODEL_PATH="/data1/shenth/models/deberta-v3-base"  # 修改为实际路径
OUTPUT_DIR="./outputs"

# AdaLoRA 参数
INIT_R=12
TARGET_R=4
TINIT=200
TFINAL=200
DELTA_T=10

# 训练参数
EPOCHS=3
BATCH_SIZE=32
LR=1e-5

echo "========================================"
echo "Running MNLI Experiment"
echo "========================================"
echo "Signal type: ${SIGNAL_TYPE}"
echo "Seed: ${SEED}"
echo "Model: ${MODEL_PATH}"
echo "========================================"

# 运行训练
cd "$(dirname "$0")/.."

python src/main.py \
    --mode train \
    --task ${TASK} \
    --signal ${SIGNAL_TYPE} \
    --seed ${SEED} \
    --model_path ${MODEL_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --init_r ${INIT_R} \
    --target_r ${TARGET_R} \
    --tinit ${TINIT} \
    --tfinal ${TFINAL} \
    --deltaT ${DELTA_T} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --learning_rate ${LR} \
    --bf16

echo ""
echo "========================================"
echo "Training completed!"
echo "Output dir: ${OUTPUT_DIR}/${TASK}/${SIGNAL_TYPE}/seed${SEED}"
echo "========================================"
