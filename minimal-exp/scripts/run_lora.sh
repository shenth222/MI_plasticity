#!/bin/bash
# scripts/run_lora.sh
# Usage: bash scripts/run_lora.sh [seed] [task] [lora_r] [lora_alpha]

SEED=${1:-1}
TASK=${2:-"RTE"}
LORA_R=${3:-8}
LORA_ALPHA=${4:-16}
MODEL="/data1/shenth/models/deberta/v3-base"
OUT_DIR="outputs/LoRA/${TASK}/seed${SEED}_r${LORA_R}"

echo "======================================"
echo "Training DeBERTa with LoRA on ${TASK} (seed=${SEED})"
echo "LoRA Config: r=${LORA_R}, alpha=${LORA_ALPHA}"
echo "======================================"

python -m src.train.finetune_glue_lora \
    --task ${TASK} \
    --model_name ${MODEL} \
    --out_dir ${OUT_DIR} \
    --seed ${SEED} \
    --max_len 256 \
    --lr 2e-4 \
    --epochs 20 \
    --bsz 128 \
    --lora_r ${LORA_R} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_dropout 0.1 \
    --lora_target_modules "query_proj,key_proj,value_proj,dense"

echo ""
echo "======================================"
echo "LoRA Training complete. Checkpoints saved to:"
echo "  θ0 (base): ${OUT_DIR}/ckpt_init"
echo "  θ1 (LoRA): ${OUT_DIR}/ckpt_final"
echo "======================================"
