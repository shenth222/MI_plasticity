#!/bin/bash
#
# 快速测试脚本（使用小数据集）
# 用法: bash scripts/quick_test.sh
#

set -e

echo "========================================"
echo "Quick Test Run"
echo "========================================"

cd "$(dirname "$0")/.."

# 测试 baseline_adalora（只跑 1 个 epoch，小 batch）
python src/main.py \
    --mode train \
    --task mnli \
    --signal baseline_adalora \
    --seed 42 \
    --model_path /data1/shenth/models/deberta-v3-base \
    --output_dir ./outputs/test \
    --init_r 8 \
    --target_r 4 \
    --tinit 10 \
    --tfinal 20 \
    --deltaT 5 \
    --epochs 1 \
    --batch_size 8 \
    --learning_rate 1e-5 \
    --bf16

echo ""
echo "========================================"
echo "Test completed!"
echo "Check outputs/test/mnli/baseline_adalora/seed42/"
echo "========================================"
