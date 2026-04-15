set -e
# ── 基本配置 ─────────────────────────────────────────────────────────────────
PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # casual-exp/
SCRIPT="$PROJ_ROOT/fix-budget/train/finetune_fixed_budget.py"
OUTBASE="$PROJ_ROOT/fix-budget/outputs"

MODEL=${MODEL:-"/data1/shenth/models/deberta/v3-base"}
GLUE_DATA_PATH=${GLUE_DATA_PATH:-"/data1/shenth/datasets/glue"}

TASK="MNLI"
BSZ=64
EPOCHS=4
SEED=42
LR=1e-6
MAX_LEN=256
WARMUP=0.06

BUDGET_RATIO=0.3     # 10% 注意力头
RESELECT=200           # 不重新选择（设为正整数开启，如 500）

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
  --save_steps 500 
  --save_total_limit 3
)

# ─────────────────────────────────────────────────────────────────────────────
# 示例 1：随机选择
# ─────────────────────────────────────────────────────────────────────────────
# python "$SCRIPT" \
#     "${COMMON_ARGS[@]}" \
#     --selection_strategy random \
#     --out_dir "$OUTBASE/random_r${BUDGET_RATIO/./}_seed${SEED}"

# python "$SCRIPT" \
#     "${COMMON_ARGS[@]}" \
#     --selection_strategy         pre_importance \
#     --pre_importance_metric      fisher \
#     --pre_importance_num_batches 32 \
#     --out_dir "$OUTBASE/fisher_r${BUDGET_RATIO/./}_seed${SEED}"

# python "$SCRIPT" \
#     "${COMMON_ARGS[@]}" \
#     --selection_strategy update_response \
#     --ur_metric          def4 \
#     --ur_num_batches     32 \
#     --out_dir "$OUTBASE/def4_r${BUDGET_RATIO/./}_seed${SEED}"

# python "$SCRIPT" \
#   "${COMMON_ARGS[@]}" \
#   --selection_strategy         pre_importance \
#   --pre_importance_metric      saliency \
#   --pre_importance_num_batches 32 \
#   --out_dir "$OUTBASE/saliency_r${BUDGET_RATIO/./}_seed${SEED}"

python "$SCRIPT" \
  "${COMMON_ARGS[@]}" \
  --selection_strategy         pre_importance \
  --pre_importance_metric      perturbation \
  --pre_importance_num_batches 4 \
  --out_dir "$OUTBASE/perturbation_r${BUDGET_RATIO/./}_seed${SEED}"

python "$SCRIPT" \
  "${COMMON_ARGS[@]}" \
  --selection_strategy    pre_importance \
  --pre_importance_metric singular_value \
  --pre_importance_top_k  32 \
  --out_dir "$OUTBASE/singular_value_r${BUDGET_RATIO/./}_seed${SEED}"

python "$SCRIPT" \
  "${COMMON_ARGS[@]}" \
  --selection_strategy    pre_importance \
  --pre_importance_metric spectral_entropy \
  --out_dir "$OUTBASE/spectral_entropy_r${BUDGET_RATIO/./}_seed${SEED}"

python "$SCRIPT" \
  "${COMMON_ARGS[@]}" \
  --selection_strategy update_response \
  --ur_metric          def1 \
  --ur_probe_steps     20 \
  --out_dir "$OUTBASE/def1_r${BUDGET_RATIO/./}_seed${SEED}"

python "$SCRIPT" \
  "${COMMON_ARGS[@]}" \
  --selection_strategy update_response \
  --ur_metric          def2 \
  --ur_num_batches     32 \
  --out_dir "$OUTBASE/def2_r${BUDGET_RATIO/./}_seed${SEED}"

python "$SCRIPT" \
  "${COMMON_ARGS[@]}" \
  --selection_strategy update_response \
  --ur_metric          def3 \
  --ur_T_early         200 \
  --out_dir "$OUTBASE/def3_r${BUDGET_RATIO/./}_seed${SEED}"