#!/bin/bash
# scripts/make_plots.sh
# Usage: bash scripts/make_plots.sh [seed]

SEED=${1:-1}
TASK="MNLI"
OUT_DIR="outputs/${TASK}/seed${SEED}"

echo "======================================"
echo "Aggregating results and making plots (seed=${SEED})"
echo "======================================"

# Step 1: Aggregate jsonl files into CSV + stats + cases
echo ""
echo "[1/2] Aggregating JSONL → CSV + stats.json + cases.json..."
python -m src.analysis.aggregate \
    --exp_dir ${OUT_DIR} \
    --topk 20

# Step 2: Generate plots
echo ""
echo "[2/2] Generating plots..."
python -m src.analysis.plots \
    --exp_dir ${OUT_DIR}

echo ""
echo "======================================"
echo "Done! Output files:"
echo "  - ${OUT_DIR}/heads.csv"
echo "  - ${OUT_DIR}/stats.json"
echo "  - ${OUT_DIR}/cases.json"
echo "  - ${OUT_DIR}/fig_I_vs_U.png"
echo "  - ${OUT_DIR}/fig_I_vs_G.png"
echo "  - ${OUT_DIR}/fig_stats.png"
echo "======================================"
