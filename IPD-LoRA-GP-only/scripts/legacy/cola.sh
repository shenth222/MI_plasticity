#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# GP-only dynamic-rank LoRA example runs for GLUE CoLA.
#
# Usage:
#   bash cola.sh probing
#   bash cola.sh proxy
#
# Optional overrides:
#   CUDA_VISIBLE_DEVICES=1 MODEL_PATH=/path DATA_ROOT=/path bash cola.sh probing

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
MODEL_PATH="${MODEL_PATH:-/data/shenth/models/deberta/v3-base}"
DATA_ROOT="${DATA_ROOT:-/data/shenth/datasets/glue}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
CALIBRATION_SIZE="${CALIBRATION_SIZE:-128}"

MODE="${1:-probing}"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"

COMMON_ARGS=(
  --task_name cola
  --dataset_path "${DATA_ROOT}"
  --model_name_or_path "${MODEL_PATH}"
  --max_length 256
  --bf16
  --per_device_train_batch_size "${TRAIN_BATCH_SIZE}"
  --per_device_eval_batch_size "${EVAL_BATCH_SIZE}"
  --learning_rate 2e-4
  --weight_decay 0.01
  --num_train_epochs 50
  --warmup_ratio 0.1
  --seed 42
  --max_lora_rank 12
  --initial_active_rank 8
  --lora_alpha 16
  --lora_dropout 0.05
  --score_interval 100
  --warmup_steps_for_ipd 100
  --calibration_size "${CALIBRATION_SIZE}"
  --calibration_resample_stride 9973
  --total_rank_budget 0
  --target_rank 6
  --tfinal_ratio 0.15
  --logging_steps 20
  --evaluation_strategy epoch
  --report_to_wandb
  --wandb_project gp-only-lora
)

if [[ "${MODE}" == "probing" ]]; then
  python train_ipd_lora.py "${COMMON_ARGS[@]}" \
    --output_dir "outputs/gp_only_lora_cola_probing_${RUN_TAG}" \
    --goodput_method probing \
    --goodput_every_n_scoring 2 \
    --beta_G 0.9 \
    --goodput_probe_max_batches 6 \
    --goodput_min_rank 1 \
    --wandb_run_name "gp-only-lora-cola-probing-${RUN_TAG}"
elif [[ "${MODE}" == "proxy" ]]; then
  python train_ipd_lora.py "${COMMON_ARGS[@]}" \
    --output_dir "outputs/gp_only_lora_cola_proxy_${RUN_TAG}" \
    --goodput_method proxy \
    --goodput_every_n_scoring 1 \
    --beta_G 0.9 \
    --goodput_min_rank 1 \
    --wandb_run_name "gp-only-lora-cola-proxy-${RUN_TAG}"
else
  echo "Unknown mode: ${MODE}"
  echo "Supported: probing | proxy"
  exit 1
fi
