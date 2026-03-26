#!/bin/bash
# score/training_gain/run.sh
# 训练时嵌入「路径积分训练收益 def3」实验脚本
#
# 用法（从 casual-exp 根目录）：
#   bash score/training_gain/run.sh [seed] [task] [lr]
#
# 示例：
#   bash score/training_gain/run.sh 1 MNLI 2e-5
#   bash score/training_gain/run.sh 42 RTE 1e-5
#
# ─────────────────────────────────────────────────────────────────────────────
# def1/def2（回滚收益）请在训练完成后使用独立脚本离线计算：
#   python -m score.training_gain.eval_rollback \
#       --theta0_path ${OUT_DIR}/ckpt_init \
#       --thetaT_path ${OUT_DIR}/ckpt_best \
#       --task ${TASK} --dataset_path ${GLUE_DATA} --output_dir ...
# ─────────────────────────────────────────────────────────────────────────────

set -e

# ── 参数（支持命令行覆盖）────────────────────────────────────────────────────
SEED=${1:-42}
TASK=${2:-"MNLI"}
LR=${3:-"1e-5"}

MODEL="/data1/shenth/models/deberta/v3-base"
OUT_DIR="score/training_gain/outputs/FFT/${TASK}/seed${SEED}/lr${LR}"

# ── def3 路径积分参数 ─────────────────────────────────────────────────────────
#   --gm_head_granularity : 是否计算注意力头级别（增加内存开销，但粒度更细）
#   --gm_log_every        : 每隔多少 optimizer step 计算一次梯度·Δθ
#                           RTE（~40 步/epoch）推荐 1（精确）
#                           MNLI（~61k 步/epoch）推荐 10 或 50（近似，省计算）
# ─────────────────────────────────────────────────────────────────────────────

GM_HEAD=true    # 先设为 false 调试；需要头粒度时改为 true
GM_LOG_EVERY=1   # RTE 步数少可精确；MNLI 建议改为 10

# ── 打印配置 ──────────────────────────────────────────────────────────────────
echo "============================================================"
echo "  G_m def3（路径积分）实验：DeBERTa-v3-base FFT on ${TASK}"
echo "  seed=${SEED}  lr=${LR}  log_every=${GM_LOG_EVERY}"
echo "  head_granularity=${GM_HEAD}"
echo "  out_dir=${OUT_DIR}"
echo "============================================================"
echo ""
echo "训练完成后，可运行 eval_rollback.py 离线计算 def1/def2 回滚收益："
echo "  python -m score.training_gain.eval_rollback \\"
echo "      --theta0_path ${OUT_DIR}/ckpt_init \\"
echo "      --thetaT_path ${OUT_DIR}/ckpt_best \\"
echo "      --task ${TASK} --dataset_path \${GLUE_DATA} --output_dir ..."
echo ""

# ── 启动训练 ─────────────────────────────────────────────────────────────────
EXTRA_ARGS=""
[ "${GM_HEAD}" = "true" ] && EXTRA_ARGS="${EXTRA_ARGS} --gm_head_granularity"

accelerate launch --num_processes=4 \
    -m score.training_gain.finetune_glue_G_m \
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
    --gm_log_every  ${GM_LOG_EVERY} \
    ${EXTRA_ARGS}

# ── 结果路径提示 ──────────────────────────────────────────────────────────────
echo ""
echo "Done. Results saved to:"
echo "  θ₀:   ${OUT_DIR}/ckpt_init         （可供 eval_rollback.py 使用）"
echo "  θ_best: ${OUT_DIR}/ckpt_best       （可供 eval_rollback.py 使用）"
echo "  θ₁:   ${OUT_DIR}/ckpt_final"
echo "  def3: ${OUT_DIR}/training_gain/def3_path_integral.json"
echo "        字段: module_scores, param_scores, steps_collected$([ "${GM_HEAD}" = "true" ] && echo ", head_scores")"
echo "  config: ${OUT_DIR}/run_config.json"
