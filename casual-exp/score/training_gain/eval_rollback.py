"""
score/training_gain/eval_rollback.py

定义一 & 定义二（回滚训练收益）——独立离线评估脚本。

─────────────────────────────────────────────────────────────────────────────
def1/def2 不需要任何训练过程：
  · 直接接收两个检查点目录：
      --theta0_path   微调前初始参数 θ^(0)
      --thetaT_path   微调后参数    θ^(T)
  · 加载两个模型（结构相同，参数不同）
  · 在完整验证集上逐模块/头回滚并评估
  · 保存 def1_rollback_loss.json 和/或 def2_rollback_acc.json

注意与训练版本的区别：
  · finetune_glue_G_m.py  在训练时通过 RollbackCallback 捕获 θ^(0) 快照，
    适合需要同时训练并记录训练收益的场景。
  · 本脚本（eval_rollback.py）在训练完成后离线运行，
    不依赖训练过程，可用于分析已有检查点。

运行示例：
    cd /data1/shenth/work/MI_plasticity/casual-exp
    python -m score.training_gain.eval_rollback \\
        --theta0_path /data1/shenth/work/MI_plasticity/casual-exp/baseline/outputs/FFT/MNLI/seed42/lr2e-5/ckpt_init \\
        --thetaT_path /data1/shenth/work/MI_plasticity/casual-exp/baseline/outputs/FFT/MNLI/seed42/lr2e-5/ckpt_best \\
        --task mnli \\
        --dataset_path /data1/shenth/datasets/glue \\
        --output_dir /tmp/gm_rollback_test \\
        --metrics def1,def2 \\
        --batch_size 32

─────────────────────────────────────────────────────────────────────────────
"""

import os
import argparse
import torch
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

# ── 将 casual-exp 根目录加入 sys.path（支持直接运行或 -m 运行）
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from metric.training_gain.base import (
    group_params_by_module,
    snapshot_params,
)
from metric.training_gain.def12_rollback import (
    build_glue_eval_fn,
    compute_rollback,
    _save_def1,
    _save_def2,
    RollbackGainMetric,
)


# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="离线计算训练收益定义一（val loss 变化）和定义二（val accuracy 变化）。"
    )

    # ── 必需参数 ────────────────────────────────────────────────────────────
    parser.add_argument(
        "--theta0_path", type=str, required=True,
        help="初始模型检查点目录（θ^(0)，微调前）",
    )
    parser.add_argument(
        "--thetaT_path", type=str, required=True,
        help="微调后模型检查点目录（θ^(T)，ckpt_best 等）",
    )
    parser.add_argument(
        "--task", type=str, required=True,
        help="GLUE 任务名（如 mnli, rte, sst2, mrpc ...）",
    )
    parser.add_argument(
        "--dataset_path", type=str, required=True,
        help="本地 GLUE 数据集根目录",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="结果保存目录（JSON 文件写入此处）",
    )

    # ── 可选参数 ────────────────────────────────────────────────────────────
    parser.add_argument(
        "--metrics", type=str, default="def1,def2",
        help="计算哪些定义，逗号分隔（def1 / def2 / def1,def2）",
    )
    parser.add_argument(
        "--head_granularity", action="store_true",
        help="是否额外计算注意力头级别分数（计算量 × num_heads）",
    )
    parser.add_argument(
        "--module_names", type=str, default=None,
        help="仅计算这些模块（逗号分隔，默认全部叶模块）",
    )
    parser.add_argument(
        "--max_length", type=int, default=256,
        help="分词最大序列长度（默认 256）",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="评估时的 batch size（默认 32）",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="计算设备（如 cuda:0）；不指定时自动选择 GPU/CPU",
    )
    parser.add_argument(
        "--tokenizer_path", type=str, default=None,
        help="分词器路径（默认从 --thetaT_path 加载）",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ── 解析设备 ────────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print(f"[eval_rollback] 使用设备: {device}")

    # ── 解析计算哪些定义 ─────────────────────────────────────────────────────
    metrics_list = [m.strip().lower() for m in args.metrics.split(",") if m.strip()]
    compute_loss = "def1" in metrics_list
    compute_acc  = "def2" in metrics_list
    if not compute_loss and not compute_acc:
        raise ValueError("--metrics 至少需要包含 def1 或 def2")
    print(f"[eval_rollback] 计算指标: {', '.join(m for m in metrics_list if m in ('def1','def2'))}")

    # ── 解析模块过滤 ──────────────────────────────────────────────────────────
    module_names = None
    if args.module_names:
        module_names = [n.strip() for n in args.module_names.split(",") if n.strip()]
        print(f"[eval_rollback] 仅计算指定模块（{len(module_names)} 个）")

    # ── 加载分词器 ─────────────────────────────────────────────────────────
    tokenizer_path = args.tokenizer_path or args.thetaT_path
    print(f"[eval_rollback] 加载分词器: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # ── 加载 θ^(0) 模型，提取初始参数快照 ────────────────────────────────────
    print(f"[eval_rollback] 加载 θ^(0) 模型: {args.theta0_path}")
    model0 = AutoModelForSequenceClassification.from_pretrained(
        args.theta0_path, ignore_mismatched_sizes=True
    )
    theta0 = snapshot_params(model0)
    print(f"[eval_rollback] θ^(0) 已快照（{len(theta0)} 个参数）")
    del model0  # 释放显存

    # ── 加载 θ^(T) 模型 ───────────────────────────────────────────────────
    print(f"[eval_rollback] 加载 θ^(T) 模型: {args.thetaT_path}")
    modelT = AutoModelForSequenceClassification.from_pretrained(
        args.thetaT_path, ignore_mismatched_sizes=True
    )
    modelT = modelT.to(device)
    print(f"[eval_rollback] θ^(T) 已加载（设备: {device}）")

    # ── 构建评估函数 ──────────────────────────────────────────────────────
    print(f"[eval_rollback] 构建评估函数（任务={args.task}）...")
    eval_fn = build_glue_eval_fn(
        tokenizer=tokenizer,
        task_name=args.task,
        dataset_path=args.dataset_path,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    # ── 核心计算 ──────────────────────────────────────────────────────────
    module_groups = group_params_by_module(modelT)
    print(
        f"[eval_rollback] 开始回滚计算：\n"
        f"  叶模块数: {len(module_groups)}\n"
        f"  head_granularity: {args.head_granularity}\n"
        f"  output_dir: {args.output_dir}"
    )

    scores = compute_rollback(
        theta0=theta0,
        model=modelT,
        eval_fn=eval_fn,
        device=device,
        module_groups=module_groups,
        head_granularity=args.head_granularity,
        compute_loss=compute_loss,
        compute_acc=compute_acc,
        module_names=module_names,
    )

    # ── 保存结果 ──────────────────────────────────────────────────────────
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if compute_loss:
        _save_def1(scores, args.output_dir)
    if compute_acc:
        _save_def2(scores, args.output_dir)

    print(f"\n[eval_rollback] 完成！结果已保存至: {args.output_dir}")


if __name__ == "__main__":
    main()
