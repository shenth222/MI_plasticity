"""
Head Output / Activation 强度评分模块
"""
import torch
import numpy as np
from typing import List, Dict, Tuple


def compute_head_output_norm(
    head_outputs: List[torch.Tensor],
    attention_mask: torch.Tensor,
    query_mode: str = "last_token"
) -> np.ndarray:
    """
    计算每层每头的输出强度（L2 norm）
    
    Args:
        head_outputs: list of tensors, 每个 shape (batch, seq_len, num_heads, head_dim)
        attention_mask: shape (batch, seq_len)
        query_mode: "last_token" 或 "all_tokens"
        
    Returns:
        scores: shape (num_layers, num_heads)
    """
    num_layers = len(head_outputs)
    num_heads = head_outputs[0].shape[2]
    
    scores = np.zeros((num_layers, num_heads), dtype=np.float32)
    
    for layer_idx, head_output in enumerate(head_outputs):
        # head_output: (batch, seq_len, num_heads, head_dim)
        batch_size, seq_len, num_heads_check, head_dim = head_output.shape
        assert num_heads_check == num_heads
        
        # 根据 attention_mask 选择有效 tokens
        # attention_mask: (batch, seq_len)
        mask_expanded = attention_mask.unsqueeze(2).unsqueeze(3)  # (batch, seq_len, 1, 1)
        
        # 计算 L2 norm: sqrt(sum(x^2))
        # (batch, seq_len, num_heads, head_dim) -> (batch, seq_len, num_heads)
        norms = torch.norm(head_output, p=2, dim=-1)  # (batch, seq_len, num_heads)
        
        # 应用 mask
        norms = norms * mask_expanded.squeeze(-1)  # (batch, seq_len, num_heads)
        
        if query_mode == "last_token":
            # 只使用最后一个有效 token
            # 找到每个样本的最后一个有效 token 位置
            seq_lengths = attention_mask.sum(dim=1)  # (batch,)
            
            batch_head_norms = []
            for batch_idx in range(batch_size):
                last_pos = int(seq_lengths[batch_idx].item()) - 1
                if last_pos >= 0:
                    batch_head_norms.append(norms[batch_idx, last_pos, :])  # (num_heads,)
            
            if batch_head_norms:
                # 对 batch 求平均
                avg_norms = torch.stack(batch_head_norms).mean(dim=0)  # (num_heads,)
                scores[layer_idx] = avg_norms.cpu().numpy()
        
        elif query_mode == "all_tokens":
            # 使用所有有效 tokens
            # 对 seq_len 求和，然后除以有效 token 数
            sum_norms = norms.sum(dim=1)  # (batch, num_heads)
            num_valid_tokens = attention_mask.sum(dim=1, keepdim=True)  # (batch, 1)
            avg_norms = sum_norms / (num_valid_tokens + 1e-9)  # (batch, num_heads)
            
            # 对 batch 求平均
            scores[layer_idx] = avg_norms.mean(dim=0).cpu().numpy()
        
        else:
            raise ValueError(f"不支持的 query_mode: {query_mode}")
    
    return scores


def aggregate_head_output_scores(
    inference_results: List[Dict],
    num_heads: int,
    query_mode: str = "last_token"
) -> Tuple[np.ndarray, Dict]:
    """
    聚合所有批次的 head output 分数
    
    Args:
        inference_results: 推理结果列表
        num_heads: head 数量
        query_mode: query token 模式
        
    Returns:
        (scores, stats)
        - scores: shape (num_layers, num_heads)，所有样本的平均分数
        - stats: 统计信息字典
    """
    all_scores = []
    
    for batch_result in inference_results:
        # 提取 head outputs
        hidden_states = batch_result["hidden_states"]
        attention_mask = batch_result["attention_mask"]
        
        # 计算 head outputs
        from ..model.forward import extract_head_outputs
        head_outputs = extract_head_outputs(
            hidden_states=hidden_states,
            num_heads=num_heads,
            attention_mask=attention_mask
        )
        
        # 计算分数
        batch_scores = compute_head_output_norm(
            head_outputs=head_outputs,
            attention_mask=attention_mask,
            query_mode=query_mode
        )
        
        all_scores.append(batch_scores)
    
    # 聚合：对所有批次求平均
    avg_scores = np.mean(all_scores, axis=0)  # (num_layers, num_heads)
    
    # 统计信息
    stats = {
        "mean": float(np.mean(avg_scores)),
        "std": float(np.std(avg_scores)),
        "min": float(np.min(avg_scores)),
        "max": float(np.max(avg_scores))
    }
    
    return avg_scores, stats

