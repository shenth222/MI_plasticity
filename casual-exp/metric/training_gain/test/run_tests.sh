#!/bin/bash
# metric/training_gain/test/run_tests.sh
# 运行 training_gain 所有单元测试（含 def1/def2/def3/runner）
#
# 用法（从 casual-exp 根目录）：
#   bash metric/training_gain/test/run_tests.sh [test_name]
#
# 示例：
#   bash metric/training_gain/test/run_tests.sh          # 运行全部测试
#   bash metric/training_gain/test/run_tests.sh def12    # 仅运行 def1/def2
#   bash metric/training_gain/test/run_tests.sh def3     # 仅运行 def3
#   bash metric/training_gain/test/run_tests.sh runner   # 仅运行 runner
#
# ─────────────────────────────────────────────────────────────────────────────

set -e

TARGET=${1:-"all"}   # all | def12 | def3 | runner

cd "$(dirname "$0")/../../.."   # 切换到 casual-exp 根目录

echo "============================================================"
echo "  training_gain 单元测试"
echo "  target=${TARGET}"
echo "  cwd=$(pwd)"
echo "============================================================"
echo ""

run_test() {
    local module=$1
    local label=$2
    echo "──────────────────────────────────────────────────────────"
    echo "  运行：${label}"
    echo "──────────────────────────────────────────────────────────"
    conda run -n MI python -m "${module}"
    echo ""
}

case "${TARGET}" in
    def12)
        run_test "metric.training_gain.test.test_def12" \
                 "def1/def2 回滚收益（RollbackGainMetric）"
        ;;
    def3)
        run_test "metric.training_gain.test.test_def3" \
                 "def3 路径积分（PathIntegralGainMetric）"
        ;;
    runner)
        run_test "metric.training_gain.test.test_runner" \
                 "runner（TrainingGainRunner）"
        ;;
    all)
        run_test "metric.training_gain.test.test_def12" \
                 "def1/def2 回滚收益（RollbackGainMetric）"
        run_test "metric.training_gain.test.test_def3" \
                 "def3 路径积分（PathIntegralGainMetric）"
        run_test "metric.training_gain.test.test_runner" \
                 "runner（TrainingGainRunner）"
        ;;
    *)
        echo "未知 target='${TARGET}'，支持：all | def12 | def3 | runner"
        exit 1
        ;;
esac

echo "============================================================"
echo "  所有测试完成 ✓"
echo "============================================================"
