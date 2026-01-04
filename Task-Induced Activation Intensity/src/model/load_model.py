"""
模型加载模块
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Tuple, Optional


def load_tokenizer_and_model(
    model_path: str,
    dtype: str = "fp16",
    device: str = "cuda:0",
    attn_implementation: str = "eager"
) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """
    加载 tokenizer 和模型
    
    Args:
        model_path: 本地模型路径
        dtype: 数据类型（"fp16", "bf16", "fp32"）
        device: 设备
        attn_implementation: 注意力实现方式
            - "eager": 标准实现，会返回 attention weights
            - "sdpa": Scaled Dot Product Attention（PyTorch 2.0+）
            - "flash_attention_2": Flash Attention 2
            
    Returns:
        (tokenizer, model)
    """
    # 确定 torch dtype
    if dtype == "fp16":
        torch_dtype = torch.float16
    elif dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif dtype == "fp32":
        torch_dtype = torch.float32
    else:
        raise ValueError(f"不支持的 dtype: {dtype}")
    
    # 加载 tokenizer
    print(f"正在加载 tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=True
    )
    
    # 设置 pad_token（如果没有）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    print(f"正在加载模型: {model_path}")
    print(f"  - dtype: {dtype} ({torch_dtype})")
    print(f"  - device: {device}")
    print(f"  - attn_implementation: {attn_implementation}")
    
    # 模型加载配置
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": True,
        "output_attentions": True,  # 必须开启以获取 attention weights
    }
    
    # 设置注意力实现方式
    # 注意：只有 eager 模式才能返回 attention weights
    if attn_implementation == "eager":
        model_kwargs["attn_implementation"] = "eager"
    
    # 先加载到 CPU（避免多次移动）
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        **model_kwargs
    )
    
    # 移动到目标设备
    if device != "cpu":
        model = model.to(device)
    
    # 设置为评估模式
    model.eval()
    
    print(f"模型加载完成！")
    print(f"  - 参数量: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    
    # 打印模型结构信息
    if hasattr(model, "config"):
        config = model.config
        if hasattr(config, "num_hidden_layers"):
            print(f"  - 层数: {config.num_hidden_layers}")
        if hasattr(config, "num_attention_heads"):
            print(f"  - 注意力头数: {config.num_attention_heads}")
        if hasattr(config, "hidden_size"):
            print(f"  - 隐藏层大小: {config.hidden_size}")
    
    return tokenizer, model


def get_model_info(model: AutoModelForCausalLM) -> dict:
    """
    获取模型信息
    
    Args:
        model: 模型
        
    Returns:
        包含模型信息的字典
    """
    info = {}
    
    if hasattr(model, "config"):
        config = model.config
        info["num_layers"] = getattr(config, "num_hidden_layers", None)
        info["num_heads"] = getattr(config, "num_attention_heads", None)
        info["hidden_size"] = getattr(config, "hidden_size", None)
        info["num_key_value_heads"] = getattr(config, "num_key_value_heads", None)
        
        # 计算 head_dim
        if info["hidden_size"] and info["num_heads"]:
            info["head_dim"] = info["hidden_size"] // info["num_heads"]
    
    return info

