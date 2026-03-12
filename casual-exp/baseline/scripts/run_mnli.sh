#!/bin/bash
# scripts/run_mnli.sh
# Usage: bash scripts/run_mnli.sh [seed]

SEED=${1:-1}
TASK=${2:-"RTE"}
LR=${3:-"1e-5"}
MODEL="/data1/shenth/models/deberta/v3-base"
OUT_DIR="baseline/outputs/FFT/${TASK}/seed${SEED}/lr${LR}"

echo "======================================"
echo "Training DeBERTa with FFT on ${TASK} (seed=${SEED})"
echo "======================================"

accelerate launch --num_processes=4 -m baseline.train.finetune_glue \
    --task ${TASK} \
    --model_name ${MODEL} \
    --out_dir ${OUT_DIR} \
    --seed ${SEED} \
    --max_len 256 \
    --lr ${LR} \
    --epochs 10 \
    --bsz 64

echo ""
echo "Training complete. Checkpoints saved to:"
echo "  θ0: ${OUT_DIR}/ckpt_init"
echo "  θ1: ${OUT_DIR}/ckpt_final"
