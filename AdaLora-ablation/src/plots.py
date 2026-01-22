"""
可视化模块
生成 rank 分配和 signal 曲线图
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional
import argparse

from logging_utils import read_jsonl

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def plot_rank_evolution(
    rank_file: str,
    output_file: str,
    title: Optional[str] = None,
):
    """
    绘制 rank 随训练步数的变化
    
    Args:
        rank_file: rank_pattern.jsonl 文件路径
        output_file: 输出图片路径
        title: 图表标题
    """
    records = read_jsonl(rank_file)
    
    if not records:
        print(f"No data in {rank_file}")
        return
    
    # 提取数据
    steps = [r["step"] for r in records]
    total_ranks = [r["total_rank"] for r in records]
    
    # 绘图
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(steps, total_ranks, marker='o', linewidth=2, markersize=4)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Total Rank (Budget)")
    ax.set_title(title or "Rank Evolution During Training")
    ax.grid(True, alpha=0.3)
    
    # 保存
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_file}")


def plot_signal_heatmap(
    signal_file: str,
    output_file: str,
    title: Optional[str] = None,
    top_k: int = 20,
):
    """
    绘制 module-level score 热力图
    
    Args:
        signal_file: signal_scores.jsonl 文件路径
        output_file: 输出图片路径
        title: 图表标题
        top_k: 只显示 top-k 个 modules
    """
    records = read_jsonl(signal_file)
    
    if not records:
        print(f"No data in {signal_file}")
        return
    
    # 提取数据
    steps = []
    all_scores = {}
    
    for r in records:
        step = r["step"]
        scores = r["scores"]
        
        steps.append(step)
        
        for module_name, score in scores.items():
            if module_name not in all_scores:
                all_scores[module_name] = []
            all_scores[module_name].append(score)
    
    # 选择 top-k modules（按最后一步的 score 排序）
    final_scores = {name: vals[-1] for name, vals in all_scores.items()}
    top_modules = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    top_module_names = [name for name, _ in top_modules]
    
    # 构建矩阵
    matrix = []
    for name in top_module_names:
        matrix.append(all_scores[name])
    
    matrix = np.array(matrix)
    
    # 绘制热力图
    fig, ax = plt.subplots(figsize=(14, 8))
    
    sns.heatmap(
        matrix,
        ax=ax,
        cmap="YlOrRd",
        xticklabels=[f"Step {s}" for s in steps],
        yticklabels=top_module_names,
        cbar_kws={"label": "Score"},
    )
    
    ax.set_title(title or f"Signal Scores (Top-{top_k} Modules)")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Module")
    
    # 保存
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_file}")


def plot_comparison(
    output_dirs: List[str],
    labels: List[str],
    output_file: str,
    metric: str = "total_rank",
):
    """
    对比不同实验的 metric
    
    Args:
        output_dirs: 输出目录列表
        labels: 实验标签列表
        output_file: 输出图片路径
        metric: 要对比的指标（total_rank / eval_accuracy）
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for output_dir, label in zip(output_dirs, labels):
        if metric == "total_rank":
            # 读取 rank_pattern.jsonl
            rank_file = os.path.join(output_dir, "rank_pattern.jsonl")
            if not os.path.exists(rank_file):
                print(f"File not found: {rank_file}")
                continue
            
            records = read_jsonl(rank_file)
            steps = [r["step"] for r in records]
            values = [r["total_rank"] for r in records]
            
            ax.plot(steps, values, marker='o', label=label, linewidth=2, markersize=3)
            ax.set_ylabel("Total Rank")
        
        elif metric == "eval_accuracy":
            # 读取 metrics.jsonl
            metrics_file = os.path.join(output_dir, "metrics.jsonl")
            if not os.path.exists(metrics_file):
                print(f"File not found: {metrics_file}")
                continue
            
            records = read_jsonl(metrics_file)
            eval_records = [r for r in records if "eval_accuracy" in r]
            
            steps = [r["step"] for r in eval_records]
            values = [r["eval_accuracy"] for r in eval_records]
            
            ax.plot(steps, values, marker='o', label=label, linewidth=2, markersize=3)
            ax.set_ylabel("Eval Accuracy")
    
    ax.set_xlabel("Training Step")
    ax.set_title(f"Comparison: {metric}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 保存
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_file}")


def main():
    """CLI 入口"""
    parser = argparse.ArgumentParser(description="Plot AdaLoRA ablation results")
    
    parser.add_argument("--task", type=str, default="mnli")
    parser.add_argument("--signal", type=str, default="importance_only")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    
    # 对比模式
    parser.add_argument("--compare", action="store_true", help="Compare multiple signals")
    parser.add_argument("--signals", nargs="+", help="Signals to compare")
    
    args = parser.parse_args()
    
    if args.compare and args.signals:
        # 对比模式
        output_dirs = [
            f"{args.output_dir}/{args.task}/{signal}/seed{args.seed}"
            for signal in args.signals
        ]
        labels = args.signals
        
        # 绘制对比图
        compare_output_dir = f"{args.output_dir}/{args.task}/comparison"
        os.makedirs(compare_output_dir, exist_ok=True)
        
        plot_comparison(
            output_dirs,
            labels,
            f"{compare_output_dir}/rank_comparison_seed{args.seed}.png",
            metric="total_rank",
        )
        
        plot_comparison(
            output_dirs,
            labels,
            f"{compare_output_dir}/accuracy_comparison_seed{args.seed}.png",
            metric="eval_accuracy",
        )
    
    else:
        # 单个实验模式
        exp_dir = f"{args.output_dir}/{args.task}/{args.signal}/seed{args.seed}"
        
        if not os.path.exists(exp_dir):
            print(f"Directory not found: {exp_dir}")
            return
        
        # 绘制 rank evolution
        rank_file = os.path.join(exp_dir, "rank_pattern.jsonl")
        if os.path.exists(rank_file):
            plot_rank_evolution(
                rank_file,
                os.path.join(exp_dir, "rank_evolution.png"),
                title=f"Rank Evolution - {args.task} - {args.signal} - seed{args.seed}",
            )
        
        # 绘制 signal heatmap
        signal_file = os.path.join(exp_dir, "signal_scores.jsonl")
        if os.path.exists(signal_file):
            plot_signal_heatmap(
                signal_file,
                os.path.join(exp_dir, "signal_heatmap.png"),
                title=f"Signal Scores - {args.task} - {args.signal} - seed{args.seed}",
                top_k=20,
            )


if __name__ == "__main__":
    main()
