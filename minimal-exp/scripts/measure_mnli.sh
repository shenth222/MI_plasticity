#!/bin/bash
# scripts/measure_mnli.sh
# Usage: bash scripts/measure_mnli.sh [seed]

SEED=${1:-1}
TASK="MNLI"
OUT_DIR="outputs/${TASK}/seed${SEED}"

echo "======================================"
echo "Measuring Importance & Plasticity (seed=${SEED})"
echo "======================================"

# Check if training is done
if [ ! -d "${OUT_DIR}/ckpt_init" ]; then
    echo "ERROR: Training not complete. Run scripts/run_mnli.sh first."
    exit 1
fi

# Step 1: Create eval subset
echo ""
echo "[1/5] Creating fixed eval subset..."
# First, we need to get the eval size by loading the dataset
python -c "
from src.data.glue import load_glue_dataset
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('${OUT_DIR}/ckpt_init', use_fast=True)
ds = load_glue_dataset('${TASK}', tok, max_len=256)
print(len(ds['eval_raw']))
" > /tmp/eval_size_${SEED}.txt

EVAL_SIZE=$(cat /tmp/eval_size_${SEED}.txt)
echo "Eval set size: ${EVAL_SIZE}"

python -m src.analysis.make_subset \
    --size ${EVAL_SIZE} \
    --n 1024 \
    --seed 999 \
    --out ${OUT_DIR}/eval_subset.json

echo "Saved: ${OUT_DIR}/eval_subset.json"

# Step 2: Importance (pre)
echo ""
echo "[2/5] Measuring importance (pre-training = θ0)..."
python -m src.measure.importance_ablation \
    --task ${TASK} \
    --ckpt_dir ${OUT_DIR}/ckpt_init \
    --subset_path ${OUT_DIR}/eval_subset.json \
    --out_jsonl ${OUT_DIR}/importance_pre.jsonl \
    --bsz 16 \
    --max_len 256

echo "Saved: ${OUT_DIR}/importance_pre.jsonl"

# Step 3: Gradient & Fisher (pre)
echo ""
echo "[3/5] Measuring gradient/Fisher proxy (pre-training = θ0)..."
python -m src.measure.grad_fisher_gate \
    --task ${TASK} \
    --ckpt_dir ${OUT_DIR}/ckpt_init \
    --subset_path ${OUT_DIR}/eval_subset.json \
    --out_jsonl ${OUT_DIR}/gradfisher_pre.jsonl \
    --bsz 16 \
    --max_len 256

echo "Saved: ${OUT_DIR}/gradfisher_pre.jsonl"

# Step 4: Update magnitude (θ1 - θ0)
echo ""
echo "[4/5] Measuring update magnitude (θ1 - θ0)..."
python -m src.measure.update_magnitude \
    --ckpt_init ${OUT_DIR}/ckpt_init \
    --ckpt_final ${OUT_DIR}/ckpt_final \
    --out_jsonl ${OUT_DIR}/update.jsonl

echo "Saved: ${OUT_DIR}/update.jsonl"

# Step 5: Importance (post)
echo ""
echo "[5/5] Measuring importance (post-training = θ1)..."
python -m src.measure.importance_ablation \
    --task ${TASK} \
    --ckpt_dir ${OUT_DIR}/ckpt_final \
    --subset_path ${OUT_DIR}/eval_subset.json \
    --out_jsonl ${OUT_DIR}/importance_post.jsonl \
    --bsz 16 \
    --max_len 256

echo "Saved: ${OUT_DIR}/importance_post.jsonl"

echo ""
echo "======================================"
echo "All measurements complete!"
echo "======================================"
