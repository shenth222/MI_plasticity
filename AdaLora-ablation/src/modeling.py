"""
模型构建模块
负责加载 base model 和应用 AdaLoRA
"""

import logging
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import get_peft_model, AdaLoraConfig, TaskType
from typing import Tuple
from config import ModelConfig, AdaLoRAConfig

logger = logging.getLogger(__name__)


def load_base_model_and_tokenizer(
    config: ModelConfig,
    num_labels: int,
) -> Tuple[torch.nn.Module, AutoTokenizer]:
    """
    加载 base model 和 tokenizer
    
    Args:
        config: 模型配置
        num_labels: 分类标签数
        
    Returns:
        (model, tokenizer)
    """
    logger.info(f"Loading model from {config.model_name_or_path}...")
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_name if config.tokenizer_name else config.model_name_or_path,
        cache_dir=config.cache_dir,
        use_fast=config.use_fast_tokenizer,
        revision=config.model_revision,
        trust_remote_code=config.trust_remote_code,
    )
    
    # 加载 model
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name_or_path,
        num_labels=num_labels,
        cache_dir=config.cache_dir,
        revision=config.model_revision,
        trust_remote_code=config.trust_remote_code,
    )
    
    logger.info(f"Model loaded: {model.__class__.__name__}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, tokenizer


def auto_detect_target_modules(model: torch.nn.Module) -> list:
    """
    自动探测 DeBERTa 的 target modules
    
    Args:
        model: base model
        
    Returns:
        target_modules 列表
    """
    target_modules = []
    
    # 遍历所有模块，找到 Linear layers
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # 检查是否是我们想要的模块
            keywords = ["query", "key", "value", "dense"]
            
            for kw in keywords:
                if kw in name.lower():
                    # 提取模块名（去除层级前缀）
                    parts = name.split(".")
                    module_name = parts[-1]
                    
                    if module_name not in target_modules:
                        target_modules.append(module_name)
                    break
    
    logger.info(f"Auto-detected target modules: {target_modules}")
    
    return target_modules


def create_adalora_model(
    base_model: torch.nn.Module,
    adalora_config: AdaLoRAConfig,
    auto_detect: bool = True,
) -> torch.nn.Module:
    """
    应用 AdaLoRA 到 base model
    
    Args:
        base_model: 基座模型
        adalora_config: AdaLoRA 配置
        auto_detect: 是否自动探测 target_modules
        
    Returns:
        PEFT model
    """
    # 自动探测 target_modules
    target_modules = adalora_config.target_modules
    
    if auto_detect:
        detected = auto_detect_target_modules(base_model)
        if detected:
            logger.info(f"Using auto-detected target modules: {detected}")
            target_modules = detected
        else:
            logger.warning("Auto-detection failed, using config target_modules")
    
    # 创建 PEFT config
    peft_config = AdaLoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=adalora_config.init_r,
        target_r=adalora_config.target_r,
        lora_alpha=adalora_config.lora_alpha,
        lora_dropout=adalora_config.lora_dropout,
        target_modules=target_modules,
        modules_to_save=adalora_config.modules_to_save,
        init_r=adalora_config.init_r,
        tinit=adalora_config.tinit,
        tfinal=adalora_config.tfinal,
        deltaT=adalora_config.deltaT,
        beta1=adalora_config.beta1,
        beta2=adalora_config.beta2,
        orth_reg_weight=adalora_config.orth_reg_weight,
        fan_in_fan_out=adalora_config.fan_in_fan_out,
        bias=adalora_config.bias,
    )
    
    logger.info("AdaLoRA Config:")
    logger.info(f"  init_r: {adalora_config.init_r}")
    logger.info(f"  target_r: {adalora_config.target_r}")
    logger.info(f"  lora_alpha: {adalora_config.lora_alpha}")
    logger.info(f"  tinit: {adalora_config.tinit}")
    logger.info(f"  tfinal: {adalora_config.tfinal}")
    logger.info(f"  deltaT: {adalora_config.deltaT}")
    logger.info(f"  target_modules: {target_modules}")
    
    # 应用 PEFT
    peft_model = get_peft_model(base_model, peft_config)
    
    # 打印模型信息
    peft_model.print_trainable_parameters()
    
    # 统计 LoRA modules
    num_lora_modules = 0
    total_lora_params = 0
    
    for name, module in peft_model.named_modules():
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            num_lora_modules += 1
            lora_A = module.lora_A['default'].weight
            lora_B = module.lora_B['default'].weight
            total_lora_params += lora_A.numel() + lora_B.numel()
    
    logger.info(f"Number of LoRA modules: {num_lora_modules}")
    logger.info(f"Total LoRA parameters: {total_lora_params:,}")
    
    return peft_model


def get_num_trainable_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """
    获取可训练参数数量
    
    Returns:
        (trainable_params, total_params)
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    return trainable_params, total_params


if __name__ == "__main__":
    # 测试模型加载
    from config import ModelConfig, AdaLoRAConfig
    
    model_config = ModelConfig(model_name_or_path="microsoft/deberta-v3-base")
    adalora_config = AdaLoRAConfig()
    
    # 加载 base model
    base_model, tokenizer = load_base_model_and_tokenizer(model_config, num_labels=3)
    
    # 应用 AdaLoRA
    peft_model = create_adalora_model(base_model, adalora_config)
    
    print("Model created successfully!")
    
    # 统计参数
    trainable, total = get_num_trainable_parameters(peft_model)
    print(f"Trainable: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")
