#!/bin/bash
# score/training_gain/run_eval_rollback.sh
# 离线计算训练收益 def1（val-loss 变化）和 def2（val-accuracy 变化）
#
# 用法（从 casual-exp 根目录）：
#   bash score/training_gain/run_eval_rollback.sh [seed] [task] [lr]
#
# 示例：
#   bash score/training_gain/run_eval_rollback.sh 42 RTE 1e-5
#   bash score/training_gain/run_eval_rollback.sh 1 MNLI 2e-5
#
# 前提：已有训练产出的 ckpt_init 和 ckpt_best（由 run.sh 或 baseline 训练生成）。
#
# ─────────────────────────────────────────────────────────────────────────────
# 计算量警告（def1 / def2）：
#   回滚评估需对每个叶模块单独回滚并跑完整验证集：
#   DeBERTa-v3-base（~200 叶模块，48 attn 模块 × 12 头）：
#     · 仅模块级（HEAD=false）：~200 次完整验证 ≈ 几十分钟（RTE）/ 数小时（MNLI）
#     · 含头级别（HEAD=true）：~200 + 48×12 = ~776 次完整验证 ≈ 更长时间
#   建议先在 RTE 上调试，或用 --module_names 限定目标模块。
# ─────────────────────────────────────────────────────────────────────────────

set -e

# ── 参数（支持命令行覆盖）────────────────────────────────────────────────────
SEED=${1:-42}
TASK=${2:-"MNLI"}
LR=${3:-"1e-5"}

# 检查点来源（默认对应 run.sh / baseline 的输出路径，可按需修改）
# 可选 1：来自本脚本的 G_m 训练输出
# CKPT_BASE="score/training_gain/outputs/FFT/${TASK}/seed${SEED}/lr${LR}"
# 可选 2：来自 baseline 训练输出（注释掉上面一行，取消注释下面一行）
CKPT_BASE="/data1/shenth/work/MI_plasticity/casual-exp/baseline/outputs/FFT/MNLI/seed${SEED}/lr${LR}"

THETA0="${CKPT_BASE}/ckpt_init"
THETA_T="${CKPT_BASE}/ckpt_best"

DATASET_PATH="/data1/shenth/datasets/glue"
OUTPUT_DIR="score/training_gain/outputs/FFT/${TASK}/seed${SEED}/lr${LR}/training_gain"   # 与 def3 结果放同一目录

# ── 回滚评估参数 ──────────────────────────────────────────────────────────────
METRICS="def1,def2"          # def1 | def2 | def1,def2
BATCH_SIZE=64                # 越大越快，按显存调整
HEAD=true                   # 是否计算头级别（计算量 × num_heads，建议先 false）
MODULE_NAMES=""              # 仅计算指定模块（逗号分隔），空字符串=全部叶模块
                             # 示例：deberta.encoder.layer.0.attention.self.query_proj,...
MAX_LENGTH=256
DEVICE="cuda:0"

# ── 打印配置 ──────────────────────────────────────────────────────────────────
echo "============================================================"
echo "  G_m def1/def2（回滚收益）离线评估"
echo "  task=${TASK}  seed=${SEED}  lr=${LR}"
echo "  theta0=${THETA0}"
echo "  thetaT=${THETA_T}"
echo "  metrics=${METRICS}  head_granularity=${HEAD}"
echo "  batch_size=${BATCH_SIZE}  device=${DEVICE}"
echo "  output_dir=${OUTPUT_DIR}"
echo "============================================================"
echo ""

# ── 检查检查点是否存在 ─────────────────────────────────────────────────────────
if [ ! -d "${THETA0}" ]; then
    echo "❌ 未找到 θ^(0) 检查点：${THETA0}"
    echo "   请先运行 run.sh 或 baseline 训练，生成 ckpt_init。"
    exit 1
fi
if [ ! -d "${THETA_T}" ]; then
    echo "❌ 未找到 θ^(T) 检查点：${THETA_T}"
    echo "   请先运行 run.sh 或 baseline 训练，生成 ckpt_best。"
    exit 1
fi

# ── 构造额外参数 ──────────────────────────────────────────────────────────────
EXTRA_ARGS=""
[ "${HEAD}" = "true" ] && EXTRA_ARGS="${EXTRA_ARGS} --head_granularity"
[ -n "${MODULE_NAMES}" ] && EXTRA_ARGS="${EXTRA_ARGS} --module_names ${MODULE_NAMES}"

# ── 启动离线评估 ──────────────────────────────────────────────────────────────
python -m score.training_gain.eval_rollback \
    --theta0_path   "${THETA0}"        \
    --thetaT_path   "${THETA_T}"       \
    --task          "${TASK}"          \
    --dataset_path  "${DATASET_PATH}"  \
    --output_dir    "${OUTPUT_DIR}"    \
    --metrics       "${METRICS}"       \
    --batch_size    "${BATCH_SIZE}"    \
    --max_length    "${MAX_LENGTH}"    \
    --device        "${DEVICE}"        \
    ${EXTRA_ARGS}

# ── 结果路径提示 ──────────────────────────────────────────────────────────────
echo ""
echo "Done. Results saved to:"
if echo "${METRICS}" | grep -q "def1"; then
    echo "  def1: ${OUTPUT_DIR}/def1_rollback_loss.json"
    echo "        字段: baseline_loss, module_scores$([ "${HEAD}" = "true" ] && echo ", head_scores"), num_modules_computed"
fi
if echo "${METRICS}" | grep -q "def2"; then
    echo "  def2: ${OUTPUT_DIR}/def2_rollback_acc.json"
    echo "        字段: primary_metric, baseline_acc, module_scores$([ "${HEAD}" = "true" ] && echo ", head_scores")"
fi
