# metric/update_response/attn_head.py
"""
注意力头级别工具函数——直接转发 metric.pre_importance.attn_head 的实现，
保持 update_response 包的接口自包含，避免每个子模块直接跨包导入。

可用函数与数据类（详见 metric.pre_importance.attn_head 的文档）：
  AttnHeadConfig           — num_heads / head_dim / hidden_size 配置数据类
  get_attn_head_config     — 从 model.config 提取注意力头配置（无 config → None）
  classify_attn_module     — 判断模块是否为 QKV/"out" 投影
  get_attn_modules         — 扫描模型，返回 {module_name: "qkv"/"out"}
  get_head_weight_view     — 按头切片权重矩阵（共享内存视图）
  get_head_bias_view       — 按头切片偏置向量（OUT 投影偏置返回 None）
  agg_head_scores_from_acc — 从元素级累积张量聚合头级别分数（梯度类指标通用）
"""

from metric.pre_importance.attn_head import (
    AttnHeadConfig,
    get_attn_head_config,
    classify_attn_module,
    get_attn_modules,
    get_head_weight_view,
    get_head_bias_view,
    agg_head_scores_from_acc,
)

__all__ = [
    "AttnHeadConfig",
    "get_attn_head_config",
    "classify_attn_module",
    "get_attn_modules",
    "get_head_weight_view",
    "get_head_bias_view",
    "agg_head_scores_from_acc",
]
