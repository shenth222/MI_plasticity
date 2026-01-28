#!/bin/bash
# scripts/make_plots_lora.sh
# Usage: bash scripts/make_plots_lora.sh [seed] [task] [lora_r]

SEED=${1:-1}
TASK=${2:-"RTE"}
LORA_R=${3:-8}
OUT_DIR="outputs/LoRA/${TASK}/seed${SEED}_r${LORA_R}"

echo "======================================"
echo "Aggregating results and making plots (seed=${SEED}, task=${TASK})"
echo "FT Method: LoRA (rank=${LORA_R})"
echo "======================================"

# Check if measurement is done
if [ ! -f "${OUT_DIR}/update.jsonl" ]; then
    echo "ERROR: Measurement not complete. Run scripts/measure_lora.sh first."
    exit 1
fi

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
echo "  - ${OUT_DIR}/fig_Ipre_vs_Ipost.png"
echo "  - ${OUT_DIR}/fig_Ipost_corrs.png"
echo "======================================"
