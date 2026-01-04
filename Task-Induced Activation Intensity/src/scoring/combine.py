"""
组合评分模块
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from ..utils.stats import layer_wise_rank, layer_wise_normalize


def combine_scores(
    out_scores: np.ndarray,
    ent_scores: np.ndarray,
    task_scores: Optional[np.ndarray],
    lambda_ent: float = 0.5,
    lambda_task: float = 1.0,
    num_layers: int = None,
    num_heads: int = None
) -> np.ndarray:
    """
    基于 rank 的分数融合
    
    Args:
        out_scores: shape (num_layers, num_heads)，head output 分数
        ent_scores: shape (num_layers, num_heads)，entropy 分数
        task_scores: shape (num_layers, num_heads) 或 None，task alignment 分数
        lambda_ent: entropy 分数权重
        lambda_task: task alignment 分数权重
        num_layers: 层数
        num_heads: 每层 head 数
        
    Returns:
        combined_scores: shape (num_layers, num_heads)
    """
    if num_layers is None:
        num_layers = out_scores.shape[0]
    if num_heads is None:
        num_heads = out_scores.shape[1]
    
    # 展平
    out_flat = out_scores.flatten()
    ent_flat = ent_scores.flatten()
    
    # 计算 layer-wise rank
    out_rank = layer_wise_rank(out_flat, num_layers, num_heads)
    ent_rank = layer_wise_rank(ent_flat, num_layers, num_heads)
    
    # 组合
    combined = out_rank + lambda_ent * ent_rank
    
    # 如果有 task scores，加入
    if task_scores is not None:
        task_flat = task_scores.flatten()
        task_rank = layer_wise_rank(task_flat, num_layers, num_heads)
        combined = combined + lambda_task * task_rank
    
    # 重塑
    combined_scores = combined.reshape(num_layers, num_heads)
    
    return combined_scores


def get_topk_heads(
    scores: np.ndarray,
    k: int,
    return_values: bool = True
) -> List[Dict]:
    """
    获取 Top-k heads（全局）
    
    Args:
        scores: shape (num_layers, num_heads)
        k: top k
        return_values: 是否返回分数值
        
    Returns:
        top-k heads 列表，每个元素为 {"layer": int, "head": int, "score": float}
    """
    num_layers, num_heads = scores.shape
    
    # 展平
    flat_scores = scores.flatten()
    
    # 获取 top-k 的索引
    topk_indices = np.argsort(flat_scores)[-k:][::-1]  # 降序
    
    # 转换为 (layer, head)
    topk_heads = []
    for idx in topk_indices:
        layer = idx // num_heads
        head = idx % num_heads
        
        head_info = {
            "layer": int(layer),
            "head": int(head)
        }
        
        if return_values:
            head_info["score"] = float(scores[layer, head])
        
        topk_heads.append(head_info)
    
    return topk_heads


def get_topk_heads_per_layer(
    scores: np.ndarray,
    k: int,
    return_values: bool = True
) -> Dict[int, List[Dict]]:
    """
    获取每层的 Top-k heads
    
    Args:
        scores: shape (num_layers, num_heads)
        k: top k
        return_values: 是否返回分数值
        
    Returns:
        字典，key 为层索引，value 为该层的 top-k heads 列表
    """
    num_layers, num_heads = scores.shape
    
    result = {}
    
    for layer_idx in range(num_layers):
        layer_scores = scores[layer_idx]
        
        # 获取 top-k 的 head 索引
        topk_head_indices = np.argsort(layer_scores)[-k:][::-1]  # 降序
        
        topk_heads = []
        for head_idx in topk_head_indices:
            head_info = {
                "head": int(head_idx)
            }
            
            if return_values:
                head_info["score"] = float(layer_scores[head_idx])
            
            topk_heads.append(head_info)
        
        result[int(layer_idx)] = topk_heads
    
    return result


def prepare_scores_for_saving(
    out_scores: np.ndarray,
    ent_scores: np.ndarray,
    task_scores: Optional[np.ndarray],
    out_scores_norm: np.ndarray,
    ent_scores_norm: np.ndarray,
    task_scores_norm: Optional[np.ndarray],
    combined_scores: np.ndarray
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    准备分数数据以便保存为 CSV
    
    Args:
        各种分数矩阵
        
    Returns:
        (raw_data, norm_data, combined_data)
    """
    num_layers, num_heads = out_scores.shape
    
    # Raw scores
    raw_data = []
    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            row = {
                "layer": layer_idx,
                "head": head_idx,
                "out_raw": float(out_scores[layer_idx, head_idx]),
                "ent_raw": float(ent_scores[layer_idx, head_idx]),
            }
            
            if task_scores is not None:
                row["task_raw"] = float(task_scores[layer_idx, head_idx])
            else:
                row["task_raw"] = None
            
            raw_data.append(row)
    
    # Normalized scores
    norm_data = []
    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            row = {
                "layer": layer_idx,
                "head": head_idx,
                "out_norm": float(out_scores_norm[layer_idx, head_idx]),
                "ent_norm": float(ent_scores_norm[layer_idx, head_idx]),
            }
            
            if task_scores_norm is not None:
                row["task_norm"] = float(task_scores_norm[layer_idx, head_idx])
            else:
                row["task_norm"] = None
            
            norm_data.append(row)
    
    # Combined scores
    combined_data = []
    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            row = {
                "layer": layer_idx,
                "head": head_idx,
                "combined": float(combined_scores[layer_idx, head_idx])
            }
            combined_data.append(row)
    
    return raw_data, norm_data, combined_data

