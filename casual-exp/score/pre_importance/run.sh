#!/bin/bash
# scripts/run_mnli.sh
# Usage: bash scripts/run_mnli.sh [seed]

SEED=${1:-42}
TASK=${2:-"MNLI"}
LR=${3:-"1e-5"}
MODEL="/data1/shenth/models/deberta/v3-base"
OUT_DIR="score/pre_importance/outputs/FFT/${TASK}/seed${SEED}/lr${LR}"

echo "======================================"
echo "Training DeBERTa with FFT on ${TASK} (seed=${SEED})"
echo "======================================"

accelerate launch --num_processes=4 -m score.pre_importance.finetune_glue_I_pre \
    --task ${TASK} \
    --model_name ${MODEL} \
    --out_dir ${OUT_DIR} \
    --seed ${SEED} \
    --max_len 256 \
    --lr ${LR} \
    --epochs 10 \
    --bsz 64 \
    --pre_importance fisher,saliency,perturbation,singular_value,spectral_entropy \
    --pre_importance_batches 32 \
    --pre_importance_perturb_batches 4 \
    --pre_importance_head_granularity

echo ""
echo "Training complete. Checkpoints saved to:"
echo "  θ0: ${OUT_DIR}/ckpt_init"
echo "  θ1: ${OUT_DIR}/ckpt_final"
