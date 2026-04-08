#!/bin/bash
# fix-budget/scripts/run_mnli.sh
#
# 固定预算微调 MNLI 示例启动脚本。
# 运行前请确认：
#   1. GLUE_DATA_PATH 指向本地 GLUE 数据集目录
#   2. MODEL 指向本地 DeBERTa-v3-base 模型路径（或使用 Hub ID）
#   3. 在项目根目录（casual-exp/）执行本脚本

set -e

# ── 基本配置 ─────────────────────────────────────────────────────────────────
PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # casual-exp/
SCRIPT="$PROJ_ROOT/fix-budget/train/finetune_fixed_budget.py"
OUTBASE="$PROJ_ROOT/fix-budget/outputs"

MODEL=${MODEL:-"/data1/shenth/models/deberta/v3-base"}
GLUE_DATA_PATH=${GLUE_DATA_PATH:-"/data1/shenth/datasets/glue"}

TASK="MNLI"
BSZ=64
EPOCHS=10
SEED=42
LR=1e-5
MAX_LEN=256
WARMUP=0.06

BUDGET_RATIO=0.1     # 10% 注意力头
RESELECT=0           # 不重新选择（设为正整数开启，如 500）

# ─────────────────────────────────────────────────────────────────────────────
# 公共训练参数
# ─────────────────────────────────────────────────────────────────────────────
COMMON_ARGS=(
  --task           "$TASK"
  --model_name     "$MODEL"
  --dataset_path   "$GLUE_DATA_PATH"
  --bsz            "$BSZ"
  --epochs         "$EPOCHS"
  --seed           "$SEED"
  --lr             "$LR"
  --max_len        "$MAX_LEN"
  --warmup_ratio   "$WARMUP"
  --budget_ratio   "$BUDGET_RATIO"
  --reselect_every "$RESELECT"
)

# ─────────────────────────────────────────────────────────────────────────────
# 示例 1：随机选择
# ─────────────────────────────────────────────────────────────────────────────
run_random() {
  python "$SCRIPT" \
    "${COMMON_ARGS[@]}" \
    --selection_strategy random \
    --out_dir "$OUTBASE/random_r${BUDGET_RATIO/./}_seed${SEED}"
}

# ─────────────────────────────────────────────────────────────────────────────
# 示例 2：pre_importance / fisher
# ─────────────────────────────────────────────────────────────────────────────
run_fisher() {
  python "$SCRIPT" \
    "${COMMON_ARGS[@]}" \
    --selection_strategy         pre_importance \
    --pre_importance_metric      fisher \
    --pre_importance_num_batches 32 \
    --out_dir "$OUTBASE/fisher_r${BUDGET_RATIO/./}_seed${SEED}"
}

# ─────────────────────────────────────────────────────────────────────────────
# 示例 3：pre_importance / saliency（自动输出 grad_norm 和 taylor 两个变体）
# ─────────────────────────────────────────────────────────────────────────────
run_saliency() {
  python "$SCRIPT" \
    "${COMMON_ARGS[@]}" \
    --selection_strategy         pre_importance \
    --pre_importance_metric      saliency \
    --pre_importance_num_batches 32 \
    --out_dir "$OUTBASE/saliency_r${BUDGET_RATIO/./}_seed${SEED}"
}

# ─────────────────────────────────────────────────────────────────────────────
# 示例 4：pre_importance / perturbation
# ─────────────────────────────────────────────────────────────────────────────
run_perturbation() {
  python "$SCRIPT" \
    "${COMMON_ARGS[@]}" \
    --selection_strategy         pre_importance \
    --pre_importance_metric      perturbation \
    --pre_importance_num_batches 4 \
    --out_dir "$OUTBASE/perturbation_r${BUDGET_RATIO/./}_seed${SEED}"
}

# ─────────────────────────────────────────────────────────────────────────────
# 示例 5：pre_importance / singular_value（输出 nuclear_norm + top32_sum 变体）
# ─────────────────────────────────────────────────────────────────────────────
run_singular_value() {
  python "$SCRIPT" \
    "${COMMON_ARGS[@]}" \
    --selection_strategy    pre_importance \
    --pre_importance_metric singular_value \
    --pre_importance_top_k  32 \
    --out_dir "$OUTBASE/singular_value_r${BUDGET_RATIO/./}_seed${SEED}"
}

# ─────────────────────────────────────────────────────────────────────────────
# 示例 6：pre_importance / spectral_entropy
# ─────────────────────────────────────────────────────────────────────────────
run_spectral_entropy() {
  python "$SCRIPT" \
    "${COMMON_ARGS[@]}" \
    --selection_strategy    pre_importance \
    --pre_importance_metric spectral_entropy \
    --out_dir "$OUTBASE/spectral_entropy_r${BUDGET_RATIO/./}_seed${SEED}"
}

# ─────────────────────────────────────────────────────────────────────────────
# 示例 7：update_response / def1（短程试跑更新量）
# ─────────────────────────────────────────────────────────────────────────────
run_def1() {
  python "$SCRIPT" \
    "${COMMON_ARGS[@]}" \
    --selection_strategy update_response \
    --ur_metric          def1 \
    --ur_probe_steps     20 \
    --out_dir "$OUTBASE/def1_r${BUDGET_RATIO/./}_seed${SEED}"
}

# ─────────────────────────────────────────────────────────────────────────────
# 示例 8：update_response / def2（梯度-曲率归一化）
# ─────────────────────────────────────────────────────────────────────────────
run_def2() {
  python "$SCRIPT" \
    "${COMMON_ARGS[@]}" \
    --selection_strategy update_response \
    --ur_metric          def2 \
    --ur_num_batches     32 \
    --out_dir "$OUTBASE/def2_r${BUDGET_RATIO/./}_seed${SEED}"
}

# ─────────────────────────────────────────────────────────────────────────────
# 示例 9：update_response / def3（累积早期梯度范数，两阶段训练）
# ─────────────────────────────────────────────────────────────────────────────
run_def3() {
  python "$SCRIPT" \
    "${COMMON_ARGS[@]}" \
    --selection_strategy update_response \
    --ur_metric          def3 \
    --ur_T_early         200 \
    --out_dir "$OUTBASE/def3_r${BUDGET_RATIO/./}_seed${SEED}"
}

# ─────────────────────────────────────────────────────────────────────────────
# 示例 10：update_response / def4（梯度信噪比 Ppred）
# ─────────────────────────────────────────────────────────────────────────────
run_def4() {
  python "$SCRIPT" \
    "${COMMON_ARGS[@]}" \
    --selection_strategy update_response \
    --ur_metric          def4 \
    --ur_num_batches     32 \
    --out_dir "$OUTBASE/def4_r${BUDGET_RATIO/./}_seed${SEED}"
}

# ─────────────────────────────────────────────────────────────────────────────
# 示例 11：fisher + 每 500 步重新选择（开启周期性重选）
# ─────────────────────────────────────────────────────────────────────────────
run_fisher_reselect() {
  python "$SCRIPT" \
    "${COMMON_ARGS[@]}" \
    --selection_strategy         pre_importance \
    --pre_importance_metric      fisher \
    --pre_importance_num_batches 32 \
    --reselect_every             500 \
    --out_dir "$OUTBASE/fisher_resel500_r${BUDGET_RATIO/./}_seed${SEED}"
}

# ─────────────────────────────────────────────────────────────────────────────
# 入口：根据第一个参数选择运行哪种实验
# ─────────────────────────────────────────────────────────────────────────────
case "${1:-all}" in
  random)           run_random ;;
  fisher)           run_fisher ;;
  saliency)         run_saliency ;;
  perturbation)     run_perturbation ;;
  singular_value)   run_singular_value ;;
  spectral_entropy) run_spectral_entropy ;;
  def1)             run_def1 ;;
  def2)             run_def2 ;;
  def3)             run_def3 ;;
  def4)             run_def4 ;;
  fisher_reselect)  run_fisher_reselect ;;
  all)
    echo "运行全部实验..."
    run_random
    run_fisher
    run_saliency
    run_def2
    run_def3
    ;;
  *)
    echo "用法: $0 [random|fisher|saliency|perturbation|singular_value|spectral_entropy|def1|def2|def3|def4|fisher_reselect|all]"
    exit 1
    ;;
esac
