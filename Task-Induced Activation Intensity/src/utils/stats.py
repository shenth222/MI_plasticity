"""
统计归一化工具模块
"""
import numpy as np
from typing import Dict, List
from scipy import stats


def normalize_zscore(values: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """
    Z-score 归一化
    
    Args:
        values: 原始值数组
        eps: 防止除零的小常数
        
    Returns:
        归一化后的值
    """
    mean = np.mean(values)
    std = np.std(values)
    
    if std < eps:
        # 如果标准差太小，返回零数组
        return np.zeros_like(values)
    
    return (values - mean) / (std + eps)


def normalize_percentile(values: np.ndarray) -> np.ndarray:
    """
    Percentile 归一化（将值映射到 [0, 1] 的百分位数）
    
    Args:
        values: 原始值数组
        
    Returns:
        归一化后的值（范围 [0, 1]）
    """
    if len(values) == 0:
        return values
    
    # 计算每个值的百分位数
    percentiles = np.zeros_like(values, dtype=np.float32)
    
    for i, val in enumerate(values):
        percentiles[i] = stats.percentileofscore(values, val, kind='rank') / 100.0
    
    return percentiles


def layer_wise_normalize(
    scores_dict: Dict[str, List[float]],
    num_layers: int,
    num_heads: int,
    mode: str = "zscore"
) -> Dict[str, List[float]]:
    """
    对每层内的 head 分数进行归一化
    
    Args:
        scores_dict: 分数字典，key 为分数类型，value 为所有 (layer, head) 的分数列表
        num_layers: 层数
        num_heads: 每层的 head 数
        mode: 归一化模式（"zscore" 或 "percentile"）
        
    Returns:
        归一化后的分数字典
    """
    normalized_dict = {}
    
    for score_type, scores in scores_dict.items():
        # 转换为 numpy array
        scores_array = np.array(scores)
        
        # reshape 为 (num_layers, num_heads)
        if len(scores_array) != num_layers * num_heads:
            # 如果长度不匹配，跳过
            normalized_dict[score_type] = scores
            continue
        
        scores_matrix = scores_array.reshape(num_layers, num_heads)
        
        # 对每层进行归一化
        normalized_matrix = np.zeros_like(scores_matrix)
        
        for layer_idx in range(num_layers):
            layer_scores = scores_matrix[layer_idx]
            
            if mode == "zscore":
                normalized_matrix[layer_idx] = normalize_zscore(layer_scores)
            elif mode == "percentile":
                normalized_matrix[layer_idx] = normalize_percentile(layer_scores)
            else:
                raise ValueError(f"不支持的归一化模式: {mode}")
        
        # 展平回列表
        normalized_dict[score_type] = normalized_matrix.flatten().tolist()
    
    return normalized_dict


def compute_rank(values: np.ndarray) -> np.ndarray:
    """
    计算 rank（排名），值越大 rank 越大
    
    Args:
        values: 原始值数组
        
    Returns:
        rank 数组
    """
    # 使用 argsort 两次来获取 rank
    # rank 从 0 开始，值越大 rank 越大
    order = values.argsort()
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(values))
    
    return ranks.astype(np.float32)


def layer_wise_rank(
    scores: np.ndarray,
    num_layers: int,
    num_heads: int
) -> np.ndarray:
    """
    对每层内的 head 分数计算 rank
    
    Args:
        scores: 分数数组，形状 (num_layers * num_heads,)
        num_layers: 层数
        num_heads: 每层的 head 数
        
    Returns:
        rank 数组
    """
    scores_matrix = scores.reshape(num_layers, num_heads)
    rank_matrix = np.zeros_like(scores_matrix)
    
    for layer_idx in range(num_layers):
        layer_scores = scores_matrix[layer_idx]
        rank_matrix[layer_idx] = compute_rank(layer_scores)
    
    return rank_matrix.flatten()

