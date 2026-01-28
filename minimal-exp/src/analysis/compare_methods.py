# src/analysis/compare_methods.py
import os, json, argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 配置matplotlib样式
rcParams['font.size'] = 10
rcParams['axes.labelsize'] = 11
rcParams['axes.titlesize'] = 12
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9
rcParams['legend.fontsize'] = 9

def load_csv(path: str):
    """加载heads.csv"""
    data = np.genfromtxt(path, delimiter=',', names=True)
    return data

def load_stats(path: str):
    """加载stats.json"""
    with open(path, "r") as f:
        return json.load(f)

def plot_compare_I_vs_U(data_fft, data_lora, out_path, method_names):
    """
    并排对比FFT和LoRA的I_pre vs Urel散点图
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    datasets = [data_fft, data_lora]
    titles = [f"{method_names[0]}: Importance vs Update", 
              f"{method_names[1]}: Importance vs Update"]
    
    for ax, data, title in zip(axes, datasets, titles):
        I_pre = data['I_pre']
        Urel = data['Urel']
        
        ax.scatter(I_pre, Urel, s=20, alpha=0.6, marker='o')
        ax.set_xlabel('Importance (I_pre)')
        ax.set_ylabel('Relative Update (Urel)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")

def plot_compare_stats_bars(stats_fft, stats_lora, out_path, method_names):
    """
    对比FFT和LoRA的统计指标柱状图
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Spearman相关系数对比
    ax = axes[0, 0]
    metrics = ['ρ(I_pre, U)', 'ρ(I_pre, Urel)']
    fft_vals = [stats_fft['spearman_rho_Ipre_U'], stats_fft['spearman_rho_Ipre_Urel']]
    lora_vals = [stats_lora['spearman_rho_Ipre_U'], stats_lora['spearman_rho_Ipre_Urel']]
    
    x = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width/2, fft_vals, width, label=method_names[0], alpha=0.7)
    ax.bar(x + width/2, lora_vals, width, label=method_names[1], alpha=0.7)
    ax.set_ylabel('Spearman ρ')
    ax.set_title('Correlation: Importance vs Update')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([-0.5, 1.0])
    
    # 2. Top-K重叠度对比
    ax = axes[0, 1]
    topk = stats_fft['topk']
    metrics = [f'Top-{topk}\n(I, U)', f'Top-{topk}\n(I, Urel)']
    fft_vals = [stats_fft['topk_overlap_Ipre_U'], stats_fft['topk_overlap_Ipre_Urel']]
    lora_vals = [stats_lora['topk_overlap_Ipre_U'], stats_lora['topk_overlap_Ipre_Urel']]
    
    x = np.arange(len(metrics))
    ax.bar(x - width/2, fft_vals, width, label=method_names[0], alpha=0.7)
    ax.bar(x + width/2, lora_vals, width, label=method_names[1], alpha=0.7)
    ax.set_ylabel('Overlap')
    ax.set_title('Top-K Overlap')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.0])
    
    # 3. 反例数量对比
    ax = axes[1, 0]
    metrics = ['Important-\nbut-static', 'Plastic-but-\nunimportant']
    fft_vals = [stats_fft['n_important_but_static'], stats_fft['n_plastic_but_unimportant']]
    lora_vals = [stats_lora['n_important_but_static'], stats_lora['n_plastic_but_unimportant']]
    
    x = np.arange(len(metrics))
    ax.bar(x - width/2, fft_vals, width, label=method_names[0], alpha=0.7)
    ax.bar(x + width/2, lora_vals, width, label=method_names[1], alpha=0.7)
    ax.set_ylabel('Number of Cases')
    ax.set_title('Counter-example Cases')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. I_post相关性对比（如果有）
    ax = axes[1, 1]
    if not np.isnan(stats_fft.get('spearman_rho_Ipost_U', np.nan)):
        metrics = ['ρ(I_post, U)', 'ρ(I_post, Urel)', 'ρ(I_post, G)']
        fft_vals = [
            stats_fft.get('spearman_rho_Ipost_U', 0),
            stats_fft.get('spearman_rho_Ipost_Urel', 0),
            stats_fft.get('spearman_rho_Ipost_G', 0)
        ]
        lora_vals = [
            stats_lora.get('spearman_rho_Ipost_U', 0),
            stats_lora.get('spearman_rho_Ipost_Urel', 0),
            stats_lora.get('spearman_rho_Ipost_G', 0)
        ]
        
        x = np.arange(len(metrics))
        ax.bar(x - width/2, fft_vals, width, label=method_names[0], alpha=0.7)
        ax.bar(x + width/2, lora_vals, width, label=method_names[1], alpha=0.7)
        ax.set_ylabel('Spearman ρ')
        ax.set_title('Post-training Importance Correlations')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=8)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([-0.5, 1.0])
    else:
        ax.text(0.5, 0.5, 'I_post data not available', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Post-training Importance Correlations')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")

def plot_update_magnitude_comparison(data_fft, data_lora, out_path, method_names):
    """
    对比FFT和LoRA的更新量分布
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Urel分布对比
    ax = axes[0]
    ax.hist(data_fft['Urel'], bins=30, alpha=0.6, label=method_names[0], edgecolor='black')
    ax.hist(data_lora['Urel'], bins=30, alpha=0.6, label=method_names[1], edgecolor='black')
    ax.set_xlabel('Relative Update (Urel)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Update Magnitude')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. U分布对比（对数尺度）
    ax = axes[1]
    U_fft = data_fft['U'][data_fft['U'] > 0]
    U_lora = data_lora['U'][data_lora['U'] > 0]
    ax.hist(np.log10(U_fft), bins=30, alpha=0.6, label=method_names[0], edgecolor='black')
    ax.hist(np.log10(U_lora), bins=30, alpha=0.6, label=method_names[1], edgecolor='black')
    ax.set_xlabel('log10(Update Magnitude U)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Absolute Update (log scale)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")

def plot_head_wise_comparison(data_fft, data_lora, out_path, method_names):
    """
    Head-wise对比：FFT vs LoRA的更新量和重要性
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. 更新量对比散点图
    ax = axes[0]
    ax.scatter(data_fft['Urel'], data_lora['Urel'], s=20, alpha=0.6)
    
    # 添加y=x参考线
    max_val = max(data_fft['Urel'].max(), data_lora['Urel'].max())
    min_val = min(data_fft['Urel'].min(), data_lora['Urel'].min())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, linewidth=1)
    
    ax.set_xlabel(f'{method_names[0]} Urel')
    ax.set_ylabel(f'{method_names[1]} Urel')
    ax.set_title('Head-wise Update Comparison')
    ax.grid(True, alpha=0.3)
    
    # 2. 重要性对比散点图
    ax = axes[1]
    ax.scatter(data_fft['I_pre'], data_lora['I_pre'], s=20, alpha=0.6)
    
    max_val = max(data_fft['I_pre'].max(), data_lora['I_pre'].max())
    min_val = min(data_fft['I_pre'].min(), data_lora['I_pre'].min())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, linewidth=1)
    
    ax.set_xlabel(f'{method_names[0]} I_pre')
    ax.set_ylabel(f'{method_names[1]} I_pre')
    ax.set_title('Head-wise Importance Comparison')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")

def generate_summary(stats_fft, stats_lora, out_path, method_names):
    """
    生成对比摘要文本文件
    """
    lines = []
    lines.append("=" * 60)
    lines.append(f"对比摘要: {method_names[0]} vs {method_names[1]}")
    lines.append("=" * 60)
    lines.append("")
    
    lines.append("## 1. Spearman相关系数（重要性 vs 更新量）")
    lines.append(f"  ρ(I_pre, U):")
    lines.append(f"    {method_names[0]}: {stats_fft['spearman_rho_Ipre_U']:.4f}")
    lines.append(f"    {method_names[1]}: {stats_lora['spearman_rho_Ipre_U']:.4f}")
    lines.append(f"    差异: {abs(stats_fft['spearman_rho_Ipre_U'] - stats_lora['spearman_rho_Ipre_U']):.4f}")
    lines.append("")
    
    lines.append(f"  ρ(I_pre, Urel):")
    lines.append(f"    {method_names[0]}: {stats_fft['spearman_rho_Ipre_Urel']:.4f}")
    lines.append(f"    {method_names[1]}: {stats_lora['spearman_rho_Ipre_Urel']:.4f}")
    lines.append(f"    差异: {abs(stats_fft['spearman_rho_Ipre_Urel'] - stats_lora['spearman_rho_Ipre_Urel']):.4f}")
    lines.append("")
    
    lines.append(f"## 2. Top-{stats_fft['topk']}重叠度")
    lines.append(f"  Top-K overlap (I_pre, U):")
    lines.append(f"    {method_names[0]}: {stats_fft['topk_overlap_Ipre_U']:.4f}")
    lines.append(f"    {method_names[1]}: {stats_lora['topk_overlap_Ipre_U']:.4f}")
    lines.append(f"    差异: {abs(stats_fft['topk_overlap_Ipre_U'] - stats_lora['topk_overlap_Ipre_U']):.4f}")
    lines.append("")
    
    lines.append("## 3. 反例数量")
    lines.append(f"  Important-but-static:")
    lines.append(f"    {method_names[0]}: {stats_fft['n_important_but_static']}")
    lines.append(f"    {method_names[1]}: {stats_lora['n_important_but_static']}")
    lines.append("")
    
    lines.append(f"  Plastic-but-unimportant:")
    lines.append(f"    {method_names[0]}: {stats_fft['n_plastic_but_unimportant']}")
    lines.append(f"    {method_names[1]}: {stats_lora['n_plastic_but_unimportant']}")
    lines.append("")
    
    lines.append("## 4. 主要发现")
    
    # 判断哪个方法的相关性更弱（更能说明重要性≠可塑性）
    rho_fft = stats_fft['spearman_rho_Ipre_Urel']
    rho_lora = stats_lora['spearman_rho_Ipre_Urel']
    
    if abs(rho_fft) < abs(rho_lora):
        lines.append(f"  - {method_names[0]}显示更弱的相关性，更能说明'重要性≠可塑性'")
    elif abs(rho_fft) > abs(rho_lora):
        lines.append(f"  - {method_names[1]}显示更弱的相关性，更能说明'重要性≠可塑性'")
    else:
        lines.append(f"  - 两种方法显示相似的相关性")
    
    # 反例数量比较
    total_fft = stats_fft['n_important_but_static'] + stats_fft['n_plastic_but_unimportant']
    total_lora = stats_lora['n_important_but_static'] + stats_lora['n_plastic_but_unimportant']
    
    if total_fft > total_lora:
        lines.append(f"  - {method_names[0]}产生更多反例 ({total_fft} vs {total_lora})")
    elif total_fft < total_lora:
        lines.append(f"  - {method_names[1]}产生更多反例 ({total_lora} vs {total_fft})")
    else:
        lines.append(f"  - 两种方法产生相同数量的反例 ({total_fft})")
    
    lines.append("")
    lines.append("=" * 60)
    
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    
    print(f"Saved {out_path}")
    print("\n".join(lines))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fft_dir", required=True, help="FFT实验目录")
    ap.add_argument("--lora_dir", required=True, help="LoRA实验目录")
    ap.add_argument("--out_dir", required=True, help="输出目录")
    ap.add_argument("--method_names", type=str, default="FFT,LoRA", 
                    help="方法名称，逗号分隔")
    args = ap.parse_args()
    
    method_names = args.method_names.split(",")
    if len(method_names) != 2:
        raise ValueError("method_names must contain exactly 2 names separated by comma")
    
    # 加载数据
    print(f"加载 {method_names[0]} 数据...")
    data_fft = load_csv(os.path.join(args.fft_dir, "heads.csv"))
    stats_fft = load_stats(os.path.join(args.fft_dir, "stats.json"))
    
    print(f"加载 {method_names[1]} 数据...")
    data_lora = load_csv(os.path.join(args.lora_dir, "heads.csv"))
    stats_lora = load_stats(os.path.join(args.lora_dir, "stats.json"))
    
    # 创建输出目录
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 生成对比图表
    print("\n生成对比图表...")
    plot_compare_I_vs_U(data_fft, data_lora, 
                       os.path.join(args.out_dir, "compare_I_vs_U.png"), 
                       method_names)
    
    plot_compare_stats_bars(stats_fft, stats_lora,
                           os.path.join(args.out_dir, "compare_stats.png"),
                           method_names)
    
    plot_update_magnitude_comparison(data_fft, data_lora,
                                    os.path.join(args.out_dir, "compare_update_dist.png"),
                                    method_names)
    
    plot_head_wise_comparison(data_fft, data_lora,
                             os.path.join(args.out_dir, "compare_headwise.png"),
                             method_names)
    
    # 保存对比指标
    compare_metrics = {
        method_names[0]: {
            "spearman_rho_Ipre_U": stats_fft['spearman_rho_Ipre_U'],
            "spearman_rho_Ipre_Urel": stats_fft['spearman_rho_Ipre_Urel'],
            "topk_overlap_Ipre_U": stats_fft['topk_overlap_Ipre_U'],
            "n_important_but_static": stats_fft['n_important_but_static'],
            "n_plastic_but_unimportant": stats_fft['n_plastic_but_unimportant'],
        },
        method_names[1]: {
            "spearman_rho_Ipre_U": stats_lora['spearman_rho_Ipre_U'],
            "spearman_rho_Ipre_Urel": stats_lora['spearman_rho_Ipre_Urel'],
            "topk_overlap_Ipre_U": stats_lora['topk_overlap_Ipre_U'],
            "n_important_but_static": stats_lora['n_important_but_static'],
            "n_plastic_but_unimportant": stats_lora['n_plastic_but_unimportant'],
        }
    }
    
    with open(os.path.join(args.out_dir, "compare_metrics.json"), "w") as f:
        json.dump(compare_metrics, f, indent=2)
    print(f"Saved {os.path.join(args.out_dir, 'compare_metrics.json')}")
    
    # 生成对比摘要
    generate_summary(stats_fft, stats_lora,
                    os.path.join(args.out_dir, "compare_summary.txt"),
                    method_names)

if __name__ == "__main__":
    main()
