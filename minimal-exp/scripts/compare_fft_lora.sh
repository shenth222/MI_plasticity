#!/bin/bash
# scripts/compare_fft_lora.sh
# 对比FFT和LoRA的实验结果
# Usage: bash scripts/compare_fft_lora.sh [seed] [task]

SEED=${1:-1}
TASK=${2:-"RTE"}
LORA_R=${3:-8}

FFT_DIR="outputs/FFT/${TASK}/seed${SEED}"
LORA_DIR="outputs/LoRA/${TASK}/seed${SEED}_r${LORA_R}"
COMPARE_DIR="outputs/COMPARE/${TASK}/seed${SEED}"

echo "======================================"
echo "Comparing FFT vs LoRA (seed=${SEED}, task=${TASK})"
echo "======================================"

# 检查两个实验目录是否存在
if [ ! -d "${FFT_DIR}" ]; then
    echo "ERROR: FFT experiment not found at ${FFT_DIR}"
    echo "Please run: bash scripts/run_mnli.sh ${SEED} ${TASK} FFT"
    echo "           bash scripts/measure_mnli.sh ${SEED} ${TASK} FFT"
    echo "           bash scripts/make_plots.sh ${SEED} ${TASK} FFT"
    exit 1
fi

if [ ! -d "${LORA_DIR}" ]; then
    echo "ERROR: LoRA experiment not found at ${LORA_DIR}"
    echo "Please run: bash scripts/run_lora.sh ${SEED} ${TASK} ${LORA_R}"
    echo "           bash scripts/measure_lora.sh ${SEED} ${TASK} ${LORA_R}"
    echo "           bash scripts/make_plots_lora.sh ${SEED} ${TASK} ${LORA_R}"
    exit 1
fi

# 创建对比目录
mkdir -p ${COMPARE_DIR}

# 生成对比可视化
echo ""
echo "Generating comparison plots..."
python -m src.analysis.compare_methods \
    --fft_dir ${FFT_DIR} \
    --lora_dir ${LORA_DIR} \
    --out_dir ${COMPARE_DIR} \
    --method_names "FFT,LoRA-r${LORA_R}"

echo ""
echo "======================================"
echo "Comparison complete! Output files:"
echo "  - ${COMPARE_DIR}/compare_I_vs_U.png"
echo "  - ${COMPARE_DIR}/compare_stats.png"
echo "  - ${COMPARE_DIR}/compare_metrics.json"
echo "  - ${COMPARE_DIR}/compare_summary.txt"
echo "======================================"
