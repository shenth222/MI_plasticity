"""
命令行参数解析模块
"""
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Task-Induced Activation Intensity - Pre-Finetuning Head Scoring"
    )
    
    # 配置文件
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径（YAML 格式）"
    )
    
    # 模型配置
    parser.add_argument(
        "--model_path",
        type=str,
        help="本地模型路径"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["fp16", "bf16", "fp32"],
        help="数据类型"
    )
    
    # 数据配置
    parser.add_argument(
        "--data_path",
        type=str,
        help="数据集路径"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        help="最大样本数"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        help="最大序列长度"
    )
    
    # 推理配置
    parser.add_argument(
        "--batch_size",
        type=int,
        help="批次大小"
    )
    parser.add_argument(
        "--device",
        type=str,
        help="设备（如 cuda:0, cpu）"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="随机种子"
    )
    
    # 评分配置
    parser.add_argument(
        "--score_query_mode",
        type=str,
        choices=["last_token", "all_tokens"],
        help="评分时的 query token 模式"
    )
    parser.add_argument(
        "--norm_mode",
        type=str,
        choices=["zscore", "percentile"],
        help="归一化模式"
    )
    parser.add_argument(
        "--lambda_ent",
        type=float,
        help="Entropy 分数权重"
    )
    parser.add_argument(
        "--lambda_task",
        type=float,
        help="Task-align 分数权重"
    )
    
    # 输出配置
    parser.add_argument(
        "--output_dir",
        type=str,
        help="输出目录"
    )
    
    args = parser.parse_args()
    return args


def load_config(config_path: str = None) -> Dict[str, Any]:
    """加载配置文件"""
    # 默认配置
    default_config = {
        "model": {
            "path": None,
            "dtype": "fp16",
            "attn_implementation": "eager"
        },
        "data": {
            "path": None,
            "max_samples": 1024,
            "max_length": 512,
            "field_mapping": {
                "question": "question",
                "choices_text": "choices.text",
                "choices_label": "choices.label",
                "answer_key": "answerKey"
            }
        },
        "prompt": {
            "template": "Question: {question}\nChoices:\nA. {choice_A}\nB. {choice_B}\nC. {choice_C}\nD. {choice_D}\nAnswer:",
            "question_marker": "Question:",
            "choices_marker": "Choices:"
        },
        "inference": {
            "batch_size": 4,
            "device": "cuda:0",
            "seed": 42
        },
        "scoring": {
            "query_mode": "last_token",
            "norm_mode": "zscore",
            "lambda_ent": 0.5,
            "lambda_task": 1.0,
            "topk_global": 20,
            "topk_per_layer": 5
        },
        "output": {
            "dir": "outputs/run_001",
            "save_raw": True,
            "save_normalized": True,
            "save_combined": True,
            "save_topk": True
        }
    }
    
    # 如果提供了配置文件，则加载并覆盖默认配置
    if config_path and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            file_config = yaml.safe_load(f)
        
        # 递归更新配置
        def update_dict(base, override):
            for key, value in override.items():
                if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                    update_dict(base[key], value)
                else:
                    base[key] = value
        
        update_dict(default_config, file_config)
    
    return default_config


def merge_args_and_config(args, config: Dict[str, Any]) -> Dict[str, Any]:
    """合并命令行参数和配置文件，命令行参数优先级更高"""
    
    # 模型配置
    if args.model_path is not None:
        config["model"]["path"] = args.model_path
    if args.dtype is not None:
        config["model"]["dtype"] = args.dtype
    
    # 数据配置
    if args.data_path is not None:
        config["data"]["path"] = args.data_path
    if args.max_samples is not None:
        config["data"]["max_samples"] = args.max_samples
    if args.max_length is not None:
        config["data"]["max_length"] = args.max_length
    
    # 推理配置
    if args.batch_size is not None:
        config["inference"]["batch_size"] = args.batch_size
    if args.device is not None:
        config["inference"]["device"] = args.device
    if args.seed is not None:
        config["inference"]["seed"] = args.seed
    
    # 评分配置
    if args.score_query_mode is not None:
        config["scoring"]["query_mode"] = args.score_query_mode
    if args.norm_mode is not None:
        config["scoring"]["norm_mode"] = args.norm_mode
    if args.lambda_ent is not None:
        config["scoring"]["lambda_ent"] = args.lambda_ent
    if args.lambda_task is not None:
        config["scoring"]["lambda_task"] = args.lambda_task
    
    # 输出配置
    if args.output_dir is not None:
        config["output"]["dir"] = args.output_dir
    
    return config


def get_config() -> Dict[str, Any]:
    """获取最终配置（命令行参数 > 配置文件 > 默认值）"""
    args = parse_args()
    config = load_config(args.config)
    config = merge_args_and_config(args, config)
    
    # 验证必需参数
    if config["model"]["path"] is None:
        raise ValueError("必须提供 --model_path 或在配置文件中指定 model.path")
    if config["data"]["path"] is None:
        raise ValueError("必须提供 --data_path 或在配置文件中指定 data.path")
    if config["output"]["dir"] is None:
        raise ValueError("必须提供 --output_dir 或在配置文件中指定 output.dir")
    
    return config

