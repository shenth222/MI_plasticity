"""
Prompt span 定位模块
"""
from typing import Tuple, Optional, List
import re


def extract_spans_from_prompt(
    prompt_str: str,
    tokenized_prompt: 'BatchEncoding',
    question_marker: str = "Question:",
    choices_marker: str = "Choices:",
    special_token_ids: Optional[List[int]] = None
) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
    """
    从 prompt 中提取 question span 和 choices span 的 token 索引范围
    
    Args:
        prompt_str: 原始 prompt 字符串
        tokenized_prompt: tokenizer 的输出（包含 input_ids）
        question_marker: question 开始标记
        choices_marker: choices 开始标记
        special_token_ids: 特殊 token ID 列表（用于排除）
        
    Returns:
        (question_span, choices_span)，每个 span 为 (start_idx, end_idx) 或 None
    """
    try:
        # 获取 token IDs
        input_ids = tokenized_prompt["input_ids"]
        if len(input_ids.shape) > 1:
            input_ids = input_ids[0]  # 取第一个样本
        
        # 将 input_ids 转换为 token 字符串列表
        # 注意：这里需要 tokenizer 对象，我们通过其他方式获取
        # 为了避免循环依赖，我们使用字符位置映射
        
        # 方法 1：基于字符位置
        # 找到 question 和 choices 在原始字符串中的位置
        question_start_char = prompt_str.find(question_marker)
        choices_start_char = prompt_str.find(choices_marker)
        
        if question_start_char == -1:
            return None, None
        
        # question span: 从 question_marker 之后到 choices_marker 之前
        if choices_start_char != -1:
            question_text = prompt_str[question_start_char + len(question_marker):choices_start_char].strip()
        else:
            # 如果没有 choices_marker，取到字符串末尾
            question_text = prompt_str[question_start_char + len(question_marker):].strip()
        
        # 使用 offset_mapping（如果可用）
        if "offset_mapping" in tokenized_prompt:
            offset_mapping = tokenized_prompt["offset_mapping"]
            if len(offset_mapping.shape) > 2:
                offset_mapping = offset_mapping[0]  # 取第一个样本
            
            question_start_token = None
            question_end_token = None
            
            question_char_start = question_start_char + len(question_marker)
            question_char_end = question_char_start + len(question_text)
            
            # 找到覆盖 question 文本的 token 范围
            for token_idx, (start, end) in enumerate(offset_mapping):
                if start == 0 and end == 0:
                    # 特殊 token，跳过
                    continue
                
                if start >= question_char_start and start < question_char_end:
                    if question_start_token is None:
                        question_start_token = token_idx
                    question_end_token = token_idx + 1
            
            if question_start_token is not None and question_end_token is not None:
                question_span = (question_start_token, question_end_token)
            else:
                question_span = None
            
            # choices span（可选）
            choices_span = None
            if choices_start_char != -1:
                choices_char_start = choices_start_char + len(choices_marker)
                choices_start_token = None
                choices_end_token = None
                
                for token_idx, (start, end) in enumerate(offset_mapping):
                    if start == 0 and end == 0:
                        continue
                    
                    if start >= choices_char_start:
                        if choices_start_token is None:
                            choices_start_token = token_idx
                        choices_end_token = token_idx + 1
                
                if choices_start_token is not None and choices_end_token is not None:
                    choices_span = (choices_start_token, choices_end_token)
            
            return question_span, choices_span
        
        # 如果没有 offset_mapping，使用简单的启发式方法
        # 假设 question 大约在前 1/3 到 2/3 的位置
        seq_len = len(input_ids)
        question_start_token = max(1, int(seq_len * 0.1))  # 跳过 BOS
        question_end_token = int(seq_len * 0.6)
        
        return (question_start_token, question_end_token), None
    
    except Exception as e:
        # 如果提取失败，返回 None
        return None, None


def get_valid_token_mask(
    input_ids: 'torch.Tensor',
    attention_mask: 'torch.Tensor',
    special_token_ids: List[int]
) -> 'torch.Tensor':
    """
    获取有效 token 的 mask（排除 padding 和特殊 token）
    
    Args:
        input_ids: shape (batch, seq_len)
        attention_mask: shape (batch, seq_len)
        special_token_ids: 特殊 token ID 列表
        
    Returns:
        valid_mask: shape (batch, seq_len)，1 表示有效 token
    """
    import torch
    
    # 基于 attention_mask
    valid_mask = attention_mask.clone()
    
    # 排除特殊 token
    for special_id in special_token_ids:
        valid_mask = valid_mask & (input_ids != special_id)
    
    return valid_mask.bool()


def extract_spans_simple(
    prompt_str: str,
    question_marker: str = "Question:",
    choices_marker: str = "Choices:"
) -> Tuple[Optional[str], Optional[str]]:
    """
    简单方法：直接从字符串中提取 question 和 choices 文本
    
    Args:
        prompt_str: 原始 prompt 字符串
        question_marker: question 开始标记
        choices_marker: choices 开始标记
        
    Returns:
        (question_text, choices_text)
    """
    try:
        question_start = prompt_str.find(question_marker)
        if question_start == -1:
            return None, None
        
        choices_start = prompt_str.find(choices_marker)
        
        if choices_start != -1:
            question_text = prompt_str[question_start + len(question_marker):choices_start].strip()
            choices_text = prompt_str[choices_start + len(choices_marker):].strip()
            return question_text, choices_text
        else:
            question_text = prompt_str[question_start + len(question_marker):].strip()
            return question_text, None
    
    except:
        return None, None

