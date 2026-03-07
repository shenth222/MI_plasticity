#!/bin/bash
#
# 极简训练测试：验证 update_and_allocate 在真实训练中被调用
# 用法: bash scripts/minimal_update_and_allocate_test.sh
#

set -e

echo "========================================"
echo "Minimal update_and_allocate test"
echo "========================================"

cd "$(dirname "$0")/.."

OUT_DIR=./outputs/test_update_and_allocate

python src/main.py \
    --mode train \
    --task rte \
    --signal baseline_adalora \
    --seed 42 \
    --model_path /data1/shenth/models/deberta/v3-base \
    --output_dir "${OUT_DIR}" \
    --init_r 8 \
    --target_r 4 \
    --tinit 1 \
    --tfinal 3 \
    --deltaT 1 \
    --epochs 1 \
    --batch_size 128 \
    --learning_rate 1e-5 \
    --bf16

LOG_DIR="${OUT_DIR}/rte/baseline_adalora/seed42"

if ! grep -m 1 "AdaLoRA Update" "${LOG_DIR}/training.log" >/dev/null; then
    echo "ERROR: update_and_allocate seems not called (no log entry)."
    exit 1
fi

if [ ! -s "${LOG_DIR}/rank_pattern.jsonl" ]; then
    echo "ERROR: rank_pattern.jsonl missing or empty."
    exit 1
fi

echo ""
echo "========================================"
echo "Test passed: update_and_allocate called"
echo "Output: ${LOG_DIR}"
echo "========================================"
