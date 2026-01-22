"""
配置管理模块
支持通过命令行参数和配置文件设置所有实验参数
"""

import argparse
from dataclasses import dataclass, field
from typing import List, Optional
import json


@dataclass
class ModelConfig:
    """模型相关配置"""
    model_name_or_path: str = "/data1/shenth/models/deberta/v3-base"  # 修改为实际路径
    tokenizer_name: Optional[str] = None
    cache_dir: Optional[str] = None
    use_fast_tokenizer: bool = True
    model_revision: str = "main"
    trust_remote_code: bool = False


@dataclass
class DataConfig:
    """数据相关配置"""
    task_name: str = "mnli"  # mnli / rte
    max_seq_length: int = 256
    train_file: Optional[str] = None
    validation_file: Optional[str] = None
    test_file: Optional[str] = None
    overwrite_cache: bool = False
    preprocessing_num_workers: int = 8
    pad_to_max_length: bool = True


@dataclass
class AdaLoRAConfig:
    """AdaLoRA 配置"""
    # 基础 LoRA 参数
    init_r: int = 12
    target_r: int = 4
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    
    # 动态 rank 调整参数
    tinit: int = 200          # 开始调整的步数
    tfinal: int = 200         # 停止调整的步数（相对于 tinit）
    deltaT: int = 10          # 调整间隔
    beta1: float = 0.85       # EMA decay for importance
    beta2: float = 0.85       # EMA decay for uncertainty
    orth_reg_weight: float = 0.5
    total_step: Optional[int] = None  # 总训练步数（训练时自动计算）
    
    # Target modules（自动探测 DeBERTa 模块名）
    target_modules: List[str] = field(default_factory=lambda: [
        "query_proj", "key_proj", "value_proj",  # Attention
        "output.dense", "intermediate.dense"      # FFN
    ])
    
    # 其他
    modules_to_save: Optional[List[str]] = None
    fan_in_fan_out: bool = False
    bias: str = "none"


@dataclass
class SignalConfig:
    """Scoring signal 配置"""
    signal_type: str = "baseline_adalora"  # baseline_adalora / importance_only / plasticity_only / combo
    
    # EMA 参数
    ema_decay: float = 0.9
    
    # Combo signal 参数
    combo_lambda: float = 1.0
    normalize_method: str = "zscore"  # zscore / minmax / none
    
    # 记录参数
    log_signal_every: int = 10
    log_rank_every: int = 10


@dataclass
class TrainingConfig:
    """训练配置"""
    output_dir: str = "./outputs"
    seed: int = 42
    
    # 训练超参数
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 64
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.0
    warmup_steps: int = 0
    
    # 优化器
    optim: str = "adamw_torch"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    
    # 混合精度
    fp16: bool = False
    bf16: bool = True
    
    # 评估与保存
    evaluation_strategy: str = "epoch"
    save_strategy: str = "epoch"
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "accuracy"
    greater_is_better: bool = True
    
    # 日志
    logging_dir: Optional[str] = None
    logging_steps: int = 10
    report_to: List[str] = field(default_factory=lambda: ["none"])
    
    # 其他
    remove_unused_columns: bool = True
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = True
    disable_tqdm: bool = False


@dataclass
class ExperimentConfig:
    """完整实验配置"""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    adalora: AdaLoRAConfig = field(default_factory=AdaLoRAConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # 实验元信息
    experiment_name: Optional[str] = None
    notes: Optional[str] = None
    
    def __post_init__(self):
        """后处理：设置默认值"""
        if self.experiment_name is None:
            self.experiment_name = f"{self.data.task_name}_{self.signal.signal_type}_seed{self.training.seed}"
        
        if self.training.logging_dir is None:
            self.training.logging_dir = f"{self.training.output_dir}/logs"
    
    def to_dict(self):
        """转换为字典"""
        return {
            "model": self.model.__dict__,
            "data": self.data.__dict__,
            "adalora": self.adalora.__dict__,
            "signal": self.signal.__dict__,
            "training": self.training.__dict__,
            "experiment_name": self.experiment_name,
            "notes": self.notes,
        }
    
    def save(self, path: str):
        """保存配置到 JSON"""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str):
        """从 JSON 加载配置"""
        with open(path) as f:
            data = json.load(f)
        
        return cls(
            model=ModelConfig(**data["model"]),
            data=DataConfig(**data["data"]),
            adalora=AdaLoRAConfig(**data["adalora"]),
            signal=SignalConfig(**data["signal"]),
            training=TrainingConfig(**data["training"]),
            experiment_name=data.get("experiment_name"),
            notes=data.get("notes"),
        )


def parse_args() -> ExperimentConfig:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="AdaLoRA Signal-Replacement Ablation")
    
    # 模式选择
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval", "export"])
    
    # 从配置文件加载
    parser.add_argument("--config", type=str, help="Path to config JSON file")
    
    # 快速设置
    parser.add_argument("--task", type=str, default="mnli", choices=["mnli", "rte"])
    parser.add_argument("--signal", type=str, default="baseline_adalora",
                       choices=["baseline_adalora", "importance_only", "plasticity_only", "combo"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    
    # 模型
    parser.add_argument("--model_path", type=str, help="Path to pretrained model")
    
    # AdaLoRA 参数
    parser.add_argument("--init_r", type=int, default=12)
    parser.add_argument("--target_r", type=int, default=4)
    parser.add_argument("--tinit", type=int, default=200)
    parser.add_argument("--tfinal", type=int, default=200)
    parser.add_argument("--deltaT", type=int, default=10)
    parser.add_argument("--total_step", type=int, default=None,
                       help="Override total training steps for AdaLoRA schedule")
    
    # Signal 参数
    parser.add_argument("--ema_decay", type=float, default=0.9)
    parser.add_argument("--combo_lambda", type=float, default=1.0)
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=False)
    
    # 其他
    parser.add_argument("--notes", type=str, help="Experiment notes")
    
    args = parser.parse_args()
    
    # 如果提供了配置文件，从文件加载
    if args.config:
        config = ExperimentConfig.load(args.config)
    else:
        config = ExperimentConfig()
    
    # 命令行参数覆盖配置文件
    if args.model_path:
        config.model.model_name_or_path = args.model_path
    
    config.data.task_name = args.task
    config.signal.signal_type = args.signal
    config.training.seed = args.seed
    
    # 设置输出目录
    config.training.output_dir = f"{args.output_dir}/{args.task}/{args.signal}/seed{args.seed}"
    config.training.logging_dir = f"{config.training.output_dir}/logs"
    
    # AdaLoRA 参数
    config.adalora.init_r = args.init_r
    config.adalora.target_r = args.target_r
    config.adalora.tinit = args.tinit
    config.adalora.tfinal = args.tfinal
    config.adalora.deltaT = args.deltaT
    config.adalora.total_step = args.total_step
    
    # Signal 参数
    config.signal.ema_decay = args.ema_decay
    config.signal.combo_lambda = args.combo_lambda
    
    # 训练参数
    config.training.num_train_epochs = args.epochs
    config.training.per_device_train_batch_size = args.batch_size
    config.training.learning_rate = args.learning_rate
    config.training.bf16 = args.bf16
    config.training.fp16 = args.fp16
    
    if args.notes:
        config.notes = args.notes
    
    return config, args.mode


# DeBERTa v3 模块名映射（自动探测后更新）
TARGET_MODULES_MAP = {
    "query": ["query_proj", "self_attn.q_proj"],
    "key": ["key_proj", "self_attn.k_proj"],
    "value": ["value_proj", "self_attn.v_proj"],
    "output": ["attention.output.dense"],
    "dense": ["intermediate.dense", "output.dense"],
}


def get_target_modules(model) -> List[str]:
    """自动探测模型的 target modules"""
    import torch
    
    target_modules = set()
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        
        lname = name.lower()
        if lname.endswith("query_proj"):
            target_modules.add("query_proj")
        if lname.endswith("key_proj"):
            target_modules.add("key_proj")
        if lname.endswith("value_proj"):
            target_modules.add("value_proj")
        if lname.endswith("output.dense"):
            target_modules.add("output.dense")
        if lname.endswith("intermediate.dense"):
            target_modules.add("intermediate.dense")
    
    return sorted(target_modules)


if __name__ == "__main__":
    # 测试配置
    config, mode = parse_args()
    print(json.dumps(config.to_dict(), indent=2))
