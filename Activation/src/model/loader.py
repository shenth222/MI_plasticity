import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple, Optional
from ..utils.logging import get_logger


logger = get_logger(__name__)


def load_model_tokenizer(
    model_path: str,
    dtype: str = "bf16",
    device_map: str = "auto",
    attn_implementation: Optional[str] = None
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load model and tokenizer for inference.
    
    Args:
        model_path: Path to model directory
        dtype: Model dtype (bf16/fp16/fp32)
        device_map: Device map strategy (auto/cuda/cpu)
        attn_implementation: Attention implementation (None/flash_attention_2/sdpa)
        
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model from {model_path}")
    logger.info(f"  dtype: {dtype}")
    logger.info(f"  device_map: {device_map}")
    logger.info(f"  attn_implementation: {attn_implementation}")
    
    # Map dtype string to torch dtype
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32
    }
    torch_dtype = dtype_map.get(dtype, torch.bfloat16)
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=True
    )
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Set pad_token to eos_token: {tokenizer.eos_token}")
    
    # Load model
    logger.info("Loading model...")
    model_kwargs = {
        "pretrained_model_name_or_path": model_path,
        "torch_dtype": torch_dtype,
        "device_map": device_map,
        "trust_remote_code": True,
    }
    
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation
    
    model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
    
    # Set to eval mode and disable gradients
    model.eval()
    torch.set_grad_enabled(False)
    
    logger.info("Model loaded successfully")
    logger.info(f"  Model type: {type(model).__name__}")
    logger.info(f"  Number of parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    
    # Log model architecture info
    if hasattr(model, "config"):
        config = model.config
        logger.info(f"  num_hidden_layers: {config.num_hidden_layers}")
        logger.info(f"  num_attention_heads: {config.num_attention_heads}")
        logger.info(f"  hidden_size: {config.hidden_size}")
    
    return model, tokenizer

