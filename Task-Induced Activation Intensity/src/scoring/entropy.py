"""
Attention Entropy 评分模块
"""
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional


def compute_attention_entropy(
    attention_probs: torch.Tensor,
    attention_mask: torch.Tensor,
    eps: float = 1e-9
) -> torch.Tensor:
    """
    计算注意力熵
    
    Args:
        attention_probs: shape (batch, num_heads, seq_len_q, seq_len_k)
        attention_mask: shape (batch, seq_len)
        eps: 防止 log(0) 的小常数
        
    Returns:
        entropy: shape (batch, num_heads, seq_len_q)
    """
    # 创建 key mask
    # attention_mask: (batch, seq_len) -> (batch, 1, 1, seq_len)
    key_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
    
    # 将 padding 位置的注意力设为 0
    masked_probs = attention_probs * key_mask
    
    # 重新归一化（确保在 key 维度上和为 1）
    sum_probs = masked_probs.sum(dim=-1, keepdim=True)  # (batch, num_heads, seq_len_q, 1)
    normalized_probs = masked_probs / (sum_probs + eps)
    
    # 计算熵: H = -sum(p * log(p))
    # 避免 log(0)
    log_probs = torch.log(normalized_probs + eps)
    entropy = -(normalized_probs * log_probs).sum(dim=-1)  # (batch, num_heads, seq_len_q)
    
    return entropy


def aggregate_entropy_scores(
    attention_probs: torch.Tensor,
    attention_mask: torch.Tensor,
    query_mode: str = "last_token",
    eps: float = 1e-9
) -> np.ndarray:
    """
    聚合每层每头的熵分数
    
    Args:
        attention_probs: shape (batch, num_heads, seq_len_q, seq_len_k)
        attention_mask: shape (batch, seq_len)
        query_mode: "last_token" 或 "all_tokens"
        eps: 小常数
        
    Returns:
        scores: shape (num_heads,)，分数为负熵（越低熵越高分）
    """
    # 计算熵
    entropy = compute_attention_entropy(
        attention_probs=attention_probs,
        attention_mask=attention_mask,
        eps=eps
    )  # (batch, num_heads, seq_len_q)
    
    batch_size, num_heads, seq_len_q = entropy.shape
    
    if query_mode == "last_token":
        # 只使用最后一个有效 token
        seq_lengths = attention_mask.sum(dim=1)  # (batch,)
        
        batch_head_entropy = []
        for batch_idx in range(batch_size):
            last_pos = int(seq_lengths[batch_idx].item()) - 1
            if last_pos >= 0 and last_pos < seq_len_q:
                batch_head_entropy.append(entropy[batch_idx, :, last_pos])  # (num_heads,)
        
        if batch_head_entropy:
            avg_entropy = torch.stack(batch_head_entropy).mean(dim=0)  # (num_heads,)
        else:
            avg_entropy = torch.zeros(num_heads, device=entropy.device)
    
    elif query_mode == "all_tokens":
        # 使用所有有效 tokens
        # 创建 query mask
        query_mask = attention_mask.unsqueeze(1)  # (batch, 1, seq_len)
        
        # 应用 mask
        masked_entropy = entropy * query_mask  # (batch, num_heads, seq_len_q)
        
        # 对 seq_len 求和，然后除以有效 token 数
        sum_entropy = masked_entropy.sum(dim=2)  # (batch, num_heads)
        num_valid_tokens = attention_mask.sum(dim=1, keepdim=True)  # (batch, 1)
        avg_entropy = sum_entropy / (num_valid_tokens + eps)  # (batch, num_heads)
        
        # 对 batch 求平均
        avg_entropy = avg_entropy.mean(dim=0)  # (num_heads,)
    
    else:
        raise ValueError(f"不支持的 query_mode: {query_mode}")
    
    # 返回负熵（越低熵越高分）
    scores = -avg_entropy.cpu().numpy()
    
    return scores


def compute_entropy_scores_all_layers(
    inference_results: List[Dict],
    query_mode: str = "last_token"
) -> Tuple[np.ndarray, Dict]:
    """
    计算所有层的熵分数
    
    Args:
        inference_results: 推理结果列表
        query_mode: query token 模式
        
    Returns:
        (scores, stats)
        - scores: shape (num_layers, num_heads)
        - stats: 统计信息字典
    """
    all_layer_scores = []
    
    for batch_result in inference_results:
        attentions = batch_result["attentions"]
        attention_mask = batch_result["attention_mask"]
        
        if attentions is None or len(attentions) == 0:
            continue
        
        batch_layer_scores = []
        
        for layer_attn in attentions:
            if layer_attn is None:
                continue
            
            # layer_attn: (batch, num_heads, seq_len, seq_len)
            layer_scores = aggregate_entropy_scores(
                attention_probs=layer_attn,
                attention_mask=attention_mask,
                query_mode=query_mode
            )
            
            batch_layer_scores.append(layer_scores)
        
        if batch_layer_scores:
            all_layer_scores.append(np.stack(batch_layer_scores))  # (num_layers, num_heads)
    
    if not all_layer_scores:
        raise ValueError("未能从任何批次中提取熵分数")
    
    # 聚合：对所有批次求平均
    avg_scores = np.mean(all_layer_scores, axis=0)  # (num_layers, num_heads)
    
    # 统计信息
    stats = {
        "mean": float(np.mean(avg_scores)),
        "std": float(np.std(avg_scores)),
        "min": float(np.min(avg_scores)),
        "max": float(np.max(avg_scores))
    }
    
    return avg_scores, stats

