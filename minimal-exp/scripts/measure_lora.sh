#!/bin/bash
# scripts/measure_lora.sh
# Usage: bash scripts/measure_lora.sh [seed] [task] [lora_r]

SEED=${1:-1}
TASK=${2:-"RTE"}
LORA_R=${3:-8}
OUT_DIR="outputs/LoRA/${TASK}/seed${SEED}_r${LORA_R}"
BASE_MODEL="${OUT_DIR}/ckpt_init"
LORA_CKPT="${OUT_DIR}/ckpt_final"

echo "======================================"
echo "Measuring LoRA Model (seed=${SEED}, task=${TASK}, r=${LORA_R})"
echo "======================================"

# 检查训练是否完成
if [ ! -d "${BASE_MODEL}" ] || [ ! -d "${LORA_CKPT}" ]; then
    echo "ERROR: Training not complete. Run scripts/run_lora.sh first."
    exit 1
fi

# Step 1: 创建固定的评估子集
echo ""
echo "[1/5] Creating fixed eval subset..."
python -c "
from src.data.glue import load_glue_dataset
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('${BASE_MODEL}', use_fast=True)
ds = load_glue_dataset('${TASK}', tok, max_len=256)
print(len(ds['eval_raw']))
" > /tmp/eval_size_lora_${SEED}.txt

EVAL_SIZE=$(cat /tmp/eval_size_lora_${SEED}.txt)
echo "Eval set size: ${EVAL_SIZE}"

python -m src.analysis.make_subset \
    --size ${EVAL_SIZE} \
    --n 1024 \
    --seed 999 \
    --out ${OUT_DIR}/eval_subset.json

echo "Saved: ${OUT_DIR}/eval_subset.json"

# Step 2: 重要性测量（微调前 - 基础模型）
echo ""
echo "[2/5] Measuring importance (pre-training = θ0, base model)..."
python -m src.measure.importance_ablation \
    --task ${TASK} \
    --ckpt_dir ${BASE_MODEL} \
    --subset_path ${OUT_DIR}/eval_subset.json \
    --out_jsonl ${OUT_DIR}/importance_pre.jsonl \
    --bsz 128 \
    --max_len 256

echo "Saved: ${OUT_DIR}/importance_pre.jsonl"

# Step 3: 梯度与Fisher测量（微调前）
echo ""
echo "[3/5] Measuring gradient/Fisher proxy (pre-training = θ0)..."
python -m src.measure.grad_fisher_gate \
    --task ${TASK} \
    --ckpt_dir ${BASE_MODEL} \
    --subset_path ${OUT_DIR}/eval_subset.json \
    --out_jsonl ${OUT_DIR}/gradfisher_pre.jsonl \
    --bsz 16 \
    --max_len 256

echo "Saved: ${OUT_DIR}/gradfisher_pre.jsonl"

# Step 4: 更新量测量（LoRA权重）
echo ""
echo "[4/5] Measuring LoRA update magnitude (θ1 - θ0)..."
python -m src.measure.update_magnitude_lora \
    --ckpt_init ${BASE_MODEL} \
    --ckpt_final ${LORA_CKPT} \
    --out_jsonl ${OUT_DIR}/update.jsonl

echo "Saved: ${OUT_DIR}/update.jsonl"

# Step 5: 重要性测量（微调后 - LoRA模型）
echo ""
echo "[5/5] Measuring importance (post-training = θ1, LoRA model)..."
python -m src.measure.importance_ablation_lora \
    --task ${TASK} \
    --base_model ${BASE_MODEL} \
    --lora_ckpt ${LORA_CKPT} \
    --subset_path ${OUT_DIR}/eval_subset.json \
    --out_jsonl ${OUT_DIR}/importance_post.jsonl \
    --bsz 16 \
    --max_len 256 \
    --merge_weights

echo "Saved: ${OUT_DIR}/importance_post.jsonl"

echo ""
echo "======================================"
echo "All LoRA measurements complete!"
echo "======================================"
