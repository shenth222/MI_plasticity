#!/bin/bash
#
# 运行所有 signal types 的 ablation 实验
# 用法: bash scripts/run_ablation_all.sh <task> <seed>
#
# 示例:
#   bash scripts/run_ablation_all.sh mnli 42
#   bash scripts/run_ablation_all.sh rte 1
#

set -e

# 参数
TASK=${1:-"mnli"}
SEED=${2:-42}

# 所有 signal types
SIGNALS=(
    "baseline_adalora"
    "importance_only"
    "plasticity_only"
    "combo"
)

echo "========================================"
echo "Running Ablation 1: Signal Replacement"
echo "========================================"
echo "Task: ${TASK}"
echo "Seed: ${SEED}"
echo "Signals: ${SIGNALS[@]}"
echo "========================================"
echo ""

# 运行所有 signals
for SIGNAL in "${SIGNALS[@]}"; do
    echo ""
    echo "========================================"
    echo "[$(date)] Running signal: ${SIGNAL}"
    echo "========================================"
    
    if [ "${TASK}" == "mnli" ]; then
        bash scripts/run_mnli.sh ${SIGNAL} ${SEED}
    elif [ "${TASK}" == "rte" ]; then
        bash scripts/run_rte.sh ${SIGNAL} ${SEED}
    else
        echo "Unknown task: ${TASK}"
        exit 1
    fi
    
    echo ""
    echo "✓ Completed: ${SIGNAL}"
    echo ""
done

echo ""
echo "========================================"
echo "All experiments completed!"
echo "========================================"
echo "Task: ${TASK}"
echo "Seed: ${SEED}"
echo "Completed signals: ${SIGNALS[@]}"
echo ""
echo "Output directory: ./outputs/${TASK}/"
echo "========================================"

# 生成对比图
echo ""
echo "Generating comparison plots..."

python src/plots.py \
    --compare \
    --task ${TASK} \
    --signals ${SIGNALS[@]} \
    --seed ${SEED} \
    --output_dir ./outputs

echo ""
echo "✓ Comparison plots saved to: ./outputs/${TASK}/comparison/"
echo ""
echo "========================================"
echo "Ablation 1 completed!"
echo "========================================"
