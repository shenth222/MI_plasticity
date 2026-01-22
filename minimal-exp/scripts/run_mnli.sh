#!/bin/bash
# scripts/run_mnli.sh
# Usage: bash scripts/run_mnli.sh [seed]

SEED=${1:-1}
TASK="RTE"
MODEL="/data1/shenth/models/deberta/v3-base"
OUT_DIR="outputs/${TASK}/seed${SEED}"

echo "======================================"
echo "Training DeBERTa on ${TASK} (seed=${SEED})"
echo "======================================"

python -m src.train.finetune_glue \
    --task ${TASK} \
    --model_name ${MODEL} \
    --out_dir ${OUT_DIR} \
    --seed ${SEED} \
    --max_len 256 \
    --lr 2e-5 \
    --epochs 3 \
    --bsz 128

echo ""
echo "Training complete. Checkpoints saved to:"
echo "  θ0: ${OUT_DIR}/ckpt_init"
echo "  θ1: ${OUT_DIR}/ckpt_final"
