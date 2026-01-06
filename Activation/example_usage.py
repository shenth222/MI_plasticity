#!/usr/bin/env python3
"""
使用示例：演示如何使用各个模块
"""

import sys
sys.path.insert(0, '.')

from src.data import PromptBuilder
from src.model import OnlineStats
import numpy as np

print("=" * 60)
print("LLaMA Attention Head Activation Collection - 使用示例")
print("=" * 60)

# 1. Prompt Builder 示例
print("\n1. Prompt Builder 示例")
print("-" * 60)

builder = PromptBuilder(template_name="arc_mcq_v1", few_shot=0)

question = "Which property of a mineral can be determined just by looking at it?"
option_labels = ["A", "B", "C", "D"]
option_texts = {
    "A": "color",
    "B": "hardness", 
    "C": "luster",
    "D": "streak"
}

prompt = builder.build(question, option_labels, option_texts)
print(prompt)

# 2. 测试 5 选项支持
print("\n2. 五选项支持示例")
print("-" * 60)

option_labels_5 = ["A", "B", "C", "D", "E"]
option_texts_5 = {
    "A": "选项 A",
    "B": "选项 B",
    "C": "选项 C",
    "D": "选项 D",
    "E": "选项 E"
}

prompt_5 = builder.build("这是一个五选项问题？", option_labels_5, option_texts_5)
print(prompt_5)

# 3. Online Statistics 示例
print("\n3. Online Statistics (Welford) 示例")
print("-" * 60)

# 假设有 4 层，每层 8 个 head
stats = OnlineStats(shape=(4, 8))

# 模拟添加 100 个样本
for i in range(100):
    # 随机生成激活值
    values = np.random.randn(4, 8) * 10 + 50
    stats.update(values)

mean = stats.get_mean()
std = stats.get_std()
count = stats.get_count()

print(f"统计了 {count[0, 0]} 个样本")
print(f"Layer 0, Head 0 的均值: {mean[0, 0]:.4f}")
print(f"Layer 0, Head 0 的标准差: {std[0, 0]:.4f}")
print(f"\n所有层所有 head 的均值范围: [{mean.min():.4f}, {mean.max():.4f}]")

# 4. 配置系统示例
print("\n4. 配置系统示例")
print("-" * 60)

from src.config import Config

# 从 YAML 加载
config = Config.from_yaml("configs/default.yaml")
print(f"模型路径: {config.model_path}")
print(f"批大小: {config.batch_size}")
print(f"Token 聚合策略: {config.token_agg}")
print(f"随机种子: {config.seed}")

print("\n" + "=" * 60)
print("所有示例运行成功！")
print("=" * 60)
print("\n准备运行完整采集流程：")
print("  1. 准备数据：将 ARC-Challenge 数据放到指定目录")
print("  2. 准备模型：下载 LLaMA 3.2-1B 模型")
print("  3. 修改配置：编辑 configs/default.yaml")
print("  4. 运行采集：bash scripts/run_arc_collect.sh")
print("\n详细文档请查看 README.md")

