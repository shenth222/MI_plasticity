"""
Attention to Task-Relevant Tokens 评分模块
"""
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional


def compute_task_alignment_score(
    attention_probs: torch.Tensor,
    attention_mask: torch.Tensor,
    question_spans: List[Optional[Tuple[int, int]]],
    query_mode: str = "last_token",
    eps: float = 1e-9
) -> Optional[np.ndarray]:
    """
    计算对任务相关 tokens（question span）的注意力强度
    
    Args:
        attention_probs: shape (batch, num_heads, seq_len_q, seq_len_k)
        attention_mask: shape (batch, seq_len)
        question_spans: 每个样本的 question span (start, end)，可能为 None
        query_mode: "last_token" 或 "all_tokens"
        eps: 小常数
        
    Returns:
        scores: shape (num_heads,) 或 None（如果所有 span 都为 None）
    """
    batch_size, num_heads, seq_len_q, seq_len_k = attention_probs.shape
    
    # 过滤出有效的 question spans
    valid_indices = [i for i, span in enumerate(question_spans) if span is not None]
    
    if not valid_indices:
        # 所有 spans 都为 None
        return None
    
    scores_list = []
    
    for batch_idx in valid_indices:
        span = question_spans[batch_idx]
        start, end = span
        
        # 确保 span 在合法范围内
        if start >= seq_len_k or end > seq_len_k or start >= end:
            continue
        
        # 获取该样本的 attention probs
        sample_attn = attention_probs[batch_idx]  # (num_heads, seq_len_q, seq_len_k)
        
        # 计算对 question span 的注意力质量
        # 方法：对 span 内的 keys 的注意力求和，然后除以 span 长度
        span_attn = sample_attn[:, :, start:end]  # (num_heads, seq_len_q, span_len)
        span_mass = span_attn.sum(dim=-1)  # (num_heads, seq_len_q)
        
        # 长度归一化
        span_len = end - start
        span_score = span_mass / span_len  # (num_heads, seq_len_q)
        
        if query_mode == "last_token":
            # 只使用最后一个有效 token
            seq_length = attention_mask[batch_idx].sum().item()
            last_pos = int(seq_length) - 1
            
            if last_pos >= 0 and last_pos < seq_len_q:
                head_scores = span_score[:, last_pos]  # (num_heads,)
                scores_list.append(head_scores)
        
        elif query_mode == "all_tokens":
            # 使用所有有效 tokens
            sample_mask = attention_mask[batch_idx]  # (seq_len,)
            
            # 应用 mask
            masked_scores = span_score * sample_mask.unsqueeze(0)  # (num_heads, seq_len_q)
            
            # 求平均
            num_valid = sample_mask.sum()
            avg_scores = masked_scores.sum(dim=1) / (num_valid + eps)  # (num_heads,)
            scores_list.append(avg_scores)
    
    if not scores_list:
        return None
    
    # 对 batch 求平均
    avg_scores = torch.stack(scores_list).mean(dim=0)  # (num_heads,)
    
    return avg_scores.cpu().numpy()


def compute_task_alignment_scores_all_layers(
    inference_results: List[Dict],
    query_mode: str = "last_token"
) -> Tuple[Optional[np.ndarray], Dict]:
    """
    计算所有层的 task alignment 分数
    
    Args:
        inference_results: 推理结果列表
        query_mode: query token 模式
        
    Returns:
        (scores, stats)
        - scores: shape (num_layers, num_heads) 或 None
        - stats: 统计信息字典
    """
    all_layer_scores = []
    total_valid_samples = 0
    total_samples = 0
    
    for batch_result in inference_results:
        attentions = batch_result["attentions"]
        attention_mask = batch_result["attention_mask"]
        question_spans = batch_result.get("question_spans", [])
        
        if attentions is None or len(attentions) == 0:
            continue
        
        total_samples += len(question_spans)
        
        batch_layer_scores = []
        
        for layer_attn in attentions:
            if layer_attn is None:
                continue
            
            # layer_attn: (batch, num_heads, seq_len, seq_len)
            layer_scores = compute_task_alignment_score(
                attention_probs=layer_attn,
                attention_mask=attention_mask,
                question_spans=question_spans,
                query_mode=query_mode
            )
            
            if layer_scores is not None:
                batch_layer_scores.append(layer_scores)
                if len(batch_layer_scores) == 1:  # 只在第一层计数
                    total_valid_samples += sum(1 for span in question_spans if span is not None)
            else:
                # 如果无法计算，用全零填充
                num_heads = layer_attn.shape[1]
                batch_layer_scores.append(np.zeros(num_heads, dtype=np.float32))
        
        if batch_layer_scores:
            all_layer_scores.append(np.stack(batch_layer_scores))  # (num_layers, num_heads)
    
    if not all_layer_scores:
        # 无法计算任何分数
        stats = {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "valid_samples": 0,
            "total_samples": total_samples,
            "success_rate": 0.0
        }
        return None, stats
    
    # 聚合：对所有批次求平均
    avg_scores = np.mean(all_layer_scores, axis=0)  # (num_layers, num_heads)
    
    # 统计信息
    stats = {
        "mean": float(np.mean(avg_scores)),
        "std": float(np.std(avg_scores)),
        "min": float(np.min(avg_scores)),
        "max": float(np.max(avg_scores)),
        "valid_samples": total_valid_samples,
        "total_samples": total_samples,
        "success_rate": total_valid_samples / total_samples if total_samples > 0 else 0.0
    }
    
    return avg_scores, stats

