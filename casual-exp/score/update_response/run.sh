#!/bin/bash
# score/update_response/run.sh
# 运行更新响应预测（R_hat）实验
#
# 用法：bash score/update_response/run.sh [seed] [task] [lr]
# 示例：bash score/update_response/run.sh 1 MNLI 2e-5

SEED=${1:-1}
TASK=${2:-"MNLI"}
LR=${3:-"1e-5"}
MODEL="/data1/shenth/models/deberta/v3-base"
OUT_DIR="score/update_response/outputs/FFT/${TASK}/seed${SEED}/lr${LR}"

echo "======================================"
echo "Training DeBERTa with FFT on ${TASK}"
echo "seed=${SEED}  lr=${LR}"
echo "======================================"

accelerate launch --num_processes=4 -m score.update_response.finetune_glue_R_hat \
    --task          ${TASK} \
    --model_name    ${MODEL} \
    --out_dir       ${OUT_DIR} \
    --seed          ${SEED} \
    --max_len       256 \
    --lr            ${LR} \
    --epochs        10 \
    --bsz           64 \
    \
    --update_response def1,def2,def3,def4 \
    --ur_probe_steps  20 \
    --ur_num_batches  32 \
    --ur_T_early      100 \
    --ur_epsilon      1e-8

echo ""
echo "Done. Results saved to:"
echo "  θ₀:              ${OUT_DIR}/ckpt_init"
echo "  θ₁:              ${OUT_DIR}/ckpt_final"
echo "  update_response: ${OUT_DIR}/update_response/*.json"
