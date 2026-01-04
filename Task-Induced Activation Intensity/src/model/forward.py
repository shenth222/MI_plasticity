"""
模型推理模块
"""
import torch
from typing import Dict, List, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from ..utils.span import extract_spans_from_prompt


def forward_with_cache(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    max_length: int = 512,
    device: str = "cuda:0"
) -> Dict:
    """
    执行 forward pass 并缓存中间结果
    
    Args:
        model: 模型
        tokenizer: tokenizer
        prompts: prompt 列表
        max_length: 最大序列长度
        device: 设备
        
    Returns:
        包含以下内容的字典：
        - input_ids: shape (batch, seq_len)
        - attention_mask: shape (batch, seq_len)
        - attentions: tuple of tensors, 每个 tensor shape (batch, num_heads, seq_len, seq_len)
        - hidden_states: tuple of tensors, 每个 tensor shape (batch, seq_len, hidden_size)
        - prompts: 原始 prompts
        - tokenized: tokenizer 的输出
    """
    # Tokenize
    tokenized = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_offsets_mapping=True  # 用于 span 提取
    )
    
    # 移动到设备
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True
        )
    
    # 提取结果
    result = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "attentions": outputs.attentions,  # tuple of (batch, num_heads, seq_len, seq_len)
        "hidden_states": outputs.hidden_states,  # tuple of (batch, seq_len, hidden_size)
        "prompts": prompts,
        "tokenized": tokenized
    }
    
    return result


def extract_attention_probs(
    attentions: Tuple[torch.Tensor, ...],
    attention_mask: torch.Tensor
) -> Tuple[torch.Tensor, ...]:
    """
    提取并处理 attention probabilities
    
    Args:
        attentions: tuple of attention tensors
        attention_mask: shape (batch, seq_len)
        
    Returns:
        处理后的 attention probabilities
    """
    # 检查是否成功获取 attentions
    if attentions is None or len(attentions) == 0:
        return None
    
    # 验证第一层
    if attentions[0] is None:
        return None
    
    return attentions


def extract_head_outputs(
    hidden_states: Tuple[torch.Tensor, ...],
    num_heads: int,
    attention_mask: torch.Tensor
) -> List[torch.Tensor]:
    """
    从 hidden states 中提取每层的 head outputs
    
    由于我们没有直接的 head outputs，我们使用 attention 层的输出
    并将其重塑为 (batch, seq_len, num_heads, head_dim)
    
    Args:
        hidden_states: tuple of hidden state tensors
        num_heads: head 数量
        attention_mask: shape (batch, seq_len)
        
    Returns:
        head_outputs 列表，每个元素 shape (batch, seq_len, num_heads, head_dim)
    """
    head_outputs = []
    
    # hidden_states[0] 是 embedding 层输出
    # hidden_states[1:] 是每层 transformer 的输出
    for layer_idx in range(1, len(hidden_states)):
        hidden = hidden_states[layer_idx]  # (batch, seq_len, hidden_size)
        
        batch_size, seq_len, hidden_size = hidden.shape
        head_dim = hidden_size // num_heads
        
        # 重塑为 (batch, seq_len, num_heads, head_dim)
        head_output = hidden.view(batch_size, seq_len, num_heads, head_dim)
        
        head_outputs.append(head_output)
    
    return head_outputs


def batch_inference(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset: 'Dataset',
    batch_size: int,
    max_length: int,
    device: str,
    prompt_template: str,
    logger=None
) -> List[Dict]:
    """
    批量推理
    
    Args:
        model: 模型
        tokenizer: tokenizer
        dataset: 数据集
        batch_size: 批次大小
        max_length: 最大序列长度
        device: 设备
        prompt_template: prompt 模板
        logger: logger
        
    Returns:
        推理结果列表
    """
    from ..data.prompt import create_prompt
    from tqdm import tqdm
    
    results = []
    num_batches = (len(dataset) + batch_size - 1) // batch_size
    
    iterator = tqdm(range(num_batches), desc="推理中")
    
    for batch_idx in iterator:
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(dataset))
        
        # 获取批次数据
        batch_data = [dataset[i] for i in range(start_idx, end_idx)]
        
        # 创建 prompts
        prompts = []
        for item in batch_data:
            prompt = create_prompt(
                question=item["question"],
                choices=item["choices"],
                choice_labels=item["choice_labels"],
                template=prompt_template
            )
            prompts.append(prompt)
        
        # Forward pass
        try:
            batch_result = forward_with_cache(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                max_length=max_length,
                device=device
            )
            
            # 提取 spans
            batch_result["question_spans"] = []
            for prompt, tokenized_sample in zip(prompts, batch_result["tokenized"]["input_ids"]):
                # 为单个样本创建 tokenized dict
                sample_tokenized = {
                    "input_ids": tokenized_sample.unsqueeze(0),
                    "offset_mapping": batch_result["tokenized"]["offset_mapping"][len(batch_result["question_spans"])].unsqueeze(0) if "offset_mapping" in batch_result["tokenized"] else None
                }
                
                question_span, _ = extract_spans_from_prompt(
                    prompt_str=prompt,
                    tokenized_prompt=sample_tokenized,
                    question_marker="Question:",
                    choices_marker="Choices:"
                )
                
                batch_result["question_spans"].append(question_span)
            
            results.append(batch_result)
            
        except Exception as e:
            if logger:
                logger.error(f"批次 {batch_idx} 推理失败: {e}")
            continue
    
    return results

