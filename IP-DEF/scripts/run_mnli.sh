#!/bin/bash
# scripts/run_mnli.sh
# Train DeBERTa-v3-base on MNLI with IP-DEF.
# Usage: bash scripts/run_mnli.sh [seed] [budget_ratio] [task]

set -euo pipefail
cd "$(dirname "$0")/.."

SEED=${1:-42}
B=${2:-0.3}
TASK=${3:-MNLI}
MODEL=${MODEL:-/data1/shenth/models/deberta/v3-base}
DATASET_PATH=${GLUE_DATA_PATH:-/data1/shenth/datasets/glue}
OUT_DIR=${OUT_DIR:-outputs/IPDEF/${TASK}/B${B}_seed${SEED}}

# wandb
export WANDB_PROJECT=${WANDB_PROJECT:-IP-DEF}

echo "======================================"
echo " IP-DEF  task=${TASK}  seed=${SEED}  B=${B}"
echo " model=${MODEL}"
echo " dataset=${DATASET_PATH}"
echo " out=${OUT_DIR}"
echo " wandb_project=${WANDB_PROJECT}"
echo "======================================"

python -m src.train.finetune_ipdef \
    --task         ${TASK} \
    --model_name   ${MODEL} \
    --dataset_path ${DATASET_PATH} \
    --out_dir      ${OUT_DIR} \
    --seed         ${SEED} \
    --max_len      256 \
    --bsz          32 \
    --eval_bsz     64 \
    --lr           1e-5 \
    --epochs       10 \
    --warmup_ratio 0.06 \
    --budget_ratio ${B} \
    --beta_I 0.95 --beta_P 0.95 \
    --T0 300 --K_c 100 --K_I 100 --M 2 \
    --alpha 0.5 --r_min 0.5 --r_max 2.0 \
    --lambda_calib 0.5 --calib_sample_ratio 0.10 --calib_group_size 4 \
    --log_every 20 \
    --eval_every_steps 500 \
    --save_signals_every 500 \
    --wandb_project ${WANDB_PROJECT} \
    --run_name "${TASK}-IPDEF-B${B}-seed${SEED}"
