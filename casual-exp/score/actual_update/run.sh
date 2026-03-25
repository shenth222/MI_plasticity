#!/bin/bash
# score/actual_update/run.sh
# 运行实际更新量 U_m 实验（def1 绝对 / def2 相对 / def3 路径长度）
#
# 用法（从 casual-exp 根目录）：
#   bash score/actual_update/run.sh [seed] [task] [lr]
#
# 示例：
#   bash score/actual_update/run.sh 1 MNLI 2e-5
#   bash score/actual_update/run.sh 42 RTE 1e-5

set -e

# ── 参数（支持命令行覆盖）────────────────────────────────────────────────────
SEED=${1:-42}
TASK=${2:-"MNLI"}
LR=${3:-"1e-5"}

MODEL="/data1/shenth/models/deberta/v3-base"
OUT_DIR="score/actual_update/outputs/FFT/${TASK}/seed${SEED}/lr${LR}"

# ── 实际更新量参数 ────────────────────────────────────────────────────────────
#   --actual_update  : 选择计算的定义，逗号分隔（def1,def2,def3 全部计算）
#   --au_log_every   : def3 路径长度计算频率（1=每步精确；>1=近似，省计算）
#                      MNLI（~393k样本，~61k步/epoch）建议 10 或 50
#                      RTE  (~2.5k样本，~40步/epoch）建议 1（精确）
#   --au_epsilon     : def2 数值稳定项 ε
# ─────────────────────────────────────────────────────────────────────────────

AU_METRICS="def1,def2,def3"
AU_LOG_EVERY=1     # 按需调整：大任务（MNLI）建议改为 10
AU_EPSILON=1e-8

# ── 打印配置 ──────────────────────────────────────────────────────────────────
echo "============================================================"
echo "  U_m 实验：DeBERTa-v3-base FFT on ${TASK}"
echo "  seed=${SEED}  lr=${LR}  metrics=${AU_METRICS}"
echo "  def3 log_every=${AU_LOG_EVERY}"
echo "  out_dir=${OUT_DIR}"
echo "============================================================"

# ── 启动训练 ─────────────────────────────────────────────────────────────────
accelerate launch --num_processes=4 \
    -m score.actual_update.finetune_glue_U_m \
    --task          ${TASK}    \
    --model_name    ${MODEL}   \
    --out_dir       ${OUT_DIR} \
    --seed          ${SEED}    \
    --max_len       256        \
    --lr            ${LR}      \
    --lr_scheduler_type linear \
    --warmup_ratio  0.06       \
    --epochs        10         \
    --bsz           64         \
    \
    --actual_update  ${AU_METRICS}  \
    --au_log_every   ${AU_LOG_EVERY} \
    --au_epsilon     ${AU_EPSILON}

# ── 结果路径提示 ──────────────────────────────────────────────────────────────
echo ""
echo "Done. Results saved to:"
echo "  θ₀:      ${OUT_DIR}/ckpt_init"
echo "  θ₁:      ${OUT_DIR}/ckpt_final"
echo "  def1:    ${OUT_DIR}/actual_update/def1_absolute.json"
echo "  def2:    ${OUT_DIR}/actual_update/def2_relative.json"
echo "  def3:    ${OUT_DIR}/actual_update/def3_path_length.json"
echo "  config:  ${OUT_DIR}/run_config.json"
