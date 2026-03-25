#!/bin/bash
# ============================================================
# Training-Induced Circuit Shift — Full Pipeline
# ============================================================
# Usage:
#   bash scripts/run.sh [SEED] [LR] [GPU]
#
# Example:
#   bash scripts/run.sh 42 1e-5 0
# ============================================================

set -e

SEED=${1:-42}
LR=${2:-"1e-5"}
GPU=${3:-0}

# Paths
CKPT_BASE="/data1/shenth/work/MI_plasticity/casual-exp/baseline/outputs/FFT/MNLI/seed${SEED}/lr${LR}"
THETA0="${CKPT_BASE}/ckpt_init"
THETA1="${CKPT_BASE}/ckpt_best"

# Use the MI conda environment which has transformers, torch, etc.
CONDA_ENV="MI"

PROJECT_ROOT="/data1/shenth/work/MI_plasticity"
SHIFT_DIR="${PROJECT_ROOT}/importance-shift"
OUT_BASE="${SHIFT_DIR}/outputs/seed${SEED}_lr${LR}"
DATASET_PATH="/data1/shenth/datasets/glue"

# Run python from within the importance-shift directory so imports work
PYTHON="conda run -n ${CONDA_ENV} --no-capture-output python"

NUM_BATCHES=64
BATCH_SIZE=32
MAX_LEN=256
SPLIT="validation_matched"

echo "======================================================"
echo "  Circuit Shift Analysis"
echo "  Seed=${SEED}  LR=${LR}  GPU=${GPU}"
echo "  θ0: ${THETA0}"
echo "  θ1: ${THETA1}"
echo "======================================================"

# Check checkpoints exist
if [ ! -d "${THETA0}" ]; then
    echo "ERROR: θ0 checkpoint not found: ${THETA0}"
    exit 1
fi
if [ ! -d "${THETA1}" ]; then
    echo "ERROR: θ1 checkpoint not found: ${THETA1}"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=${GPU}
export GLUE_DATA_PATH=${DATASET_PATH}

# Run everything from within the importance-shift directory
cd ${SHIFT_DIR}

# ── Step 1: Compute attribution scores for θ0 ──────────────────────────────
echo ""
echo "[Step 1/3] Computing activation patching scores for θ0 ..."
mkdir -p "${OUT_BASE}/theta0"
${PYTHON} run_patching.py \
    --model_path "${THETA0}" \
    --out_dir "${OUT_BASE}/theta0" \
    --dataset_path "${DATASET_PATH}" \
    --split "${SPLIT}" \
    --max_len ${MAX_LEN} \
    --batch_size ${BATCH_SIZE} \
    --num_batches ${NUM_BATCHES} \
    --method attribution \
    --seed ${SEED}
echo "[Step 1/3] Done. Results saved to ${OUT_BASE}/theta0"

# ── Step 2: Compute attribution scores for θ1 ──────────────────────────────
echo ""
echo "[Step 2/3] Computing activation patching scores for θ1 ..."
mkdir -p "${OUT_BASE}/theta1"
${PYTHON} run_patching.py \
    --model_path "${THETA1}" \
    --out_dir "${OUT_BASE}/theta1" \
    --dataset_path "${DATASET_PATH}" \
    --split "${SPLIT}" \
    --max_len ${MAX_LEN} \
    --batch_size ${BATCH_SIZE} \
    --num_batches ${NUM_BATCHES} \
    --method attribution \
    --seed ${SEED}
echo "[Step 2/3] Done. Results saved to ${OUT_BASE}/theta1"

# ── Step 3: Compute circuit shift (θ1 − θ0) ────────────────────────────────
echo ""
echo "[Step 3/3] Computing training-induced circuit shift ..."
mkdir -p "${OUT_BASE}/circuit_shift"
${PYTHON} run_circuit_shift.py \
    --theta0_scores "${OUT_BASE}/theta0/attribution_scores.json" \
    --theta1_scores "${OUT_BASE}/theta1/attribution_scores.json" \
    --out_dir "${OUT_BASE}/circuit_shift"
echo "[Step 3/3] Done."

# ── (Optional) Step 4: Visualize ───────────────────────────────────────────
echo ""
echo "[Visualize] Generating plots ..."
mkdir -p "${OUT_BASE}/figures"
${PYTHON} visualize.py \
    --theta0_dir "${OUT_BASE}/theta0" \
    --theta1_dir "${OUT_BASE}/theta1" \
    --shift_dir  "${OUT_BASE}/circuit_shift" \
    --out_dir    "${OUT_BASE}/figures" \
    --prefix attribution 2>/dev/null || echo "  [Visualize] skipped (matplotlib not available or score files missing)"

echo ""
echo "======================================================"
echo "  All results saved to: ${OUT_BASE}"
echo "  ├── theta0/           : θ0 importance scores"
echo "  ├── theta1/           : θ1 importance scores"
echo "  ├── circuit_shift/    : shift analysis + summary"
echo "  └── figures/          : plots"
echo "======================================================"
