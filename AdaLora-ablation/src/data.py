"""
数据加载模块
支持 GLUE MNLI 和 RTE 任务
"""

import logging
from typing import Dict, Tuple
from datasets import load_dataset, DatasetDict
from transformers import PreTrainedTokenizer
from config import DataConfig

logger = logging.getLogger(__name__)


TASK_TO_KEYS = {
    "mnli": ("premise", "hypothesis"),
    "rte": ("sentence1", "sentence2"),
}

TASK_TO_NUM_LABELS = {
    "mnli": 3,  # entailment, neutral, contradiction
    "rte": 2,   # entailment, not_entailment
}


def load_glue_data(
    task_name: str,
    cache_dir: str = None,
) -> DatasetDict:
    """
    加载 GLUE 数据集
    
    Args:
        task_name: 任务名（mnli / rte）
        cache_dir: 缓存目录
        
    Returns:
        DatasetDict: 包含 train/validation/test 的数据集
    """
    task_name = task_name.lower()
    
    if task_name not in TASK_TO_KEYS:
        raise ValueError(f"Unknown task: {task_name}. Supported: {list(TASK_TO_KEYS.keys())}")
    
    logger.info(f"Loading GLUE {task_name.upper()} dataset...")
    
    # 加载数据集
    if task_name == "mnli":
        dataset = load_dataset("/data1/shenth/datasets/glue", "mnli", cache_dir=cache_dir)
        # MNLI 有 matched 和 mismatched 两个验证集
        # 使用 validation_matched 作为主验证集
        dataset["validation"] = dataset["validation_matched"]
    else:
        dataset = load_dataset("/data1/shenth/datasets/glue", task_name, cache_dir=cache_dir)
    
    logger.info(f"Dataset loaded: {dataset}")
    
    return dataset


def preprocess_function(
    examples: Dict,
    tokenizer: PreTrainedTokenizer,
    task_name: str,
    max_length: int = 256,
    padding: str = "max_length",
) -> Dict:
    """
    预处理函数：tokenize 输入文本
    
    Args:
        examples: 原始样本
        tokenizer: tokenizer
        task_name: 任务名
        max_length: 最大序列长度
        padding: padding 策略
        
    Returns:
        处理后的样本
    """
    sentence1_key, sentence2_key = TASK_TO_KEYS[task_name]
    
    # Tokenize
    result = tokenizer(
        examples[sentence1_key],
        examples[sentence2_key],
        padding=padding,
        max_length=max_length,
        truncation=True,
    )
    
    return result


def prepare_datasets(
    config: DataConfig,
    tokenizer: PreTrainedTokenizer,
    cache_dir: str = None,
) -> Tuple[DatasetDict, int]:
    """
    准备数据集：加载 + 预处理
    
    Args:
        config: 数据配置
        tokenizer: tokenizer
        cache_dir: 缓存目录
        
    Returns:
        (processed_dataset, num_labels)
    """
    # 加载数据
    raw_datasets = load_glue_data(
        task_name=config.task_name,
        cache_dir=cache_dir,
    )
    
    # 获取标签数
    num_labels = TASK_TO_NUM_LABELS[config.task_name]
    
    # 预处理
    logger.info("Tokenizing datasets...")
    
    def tokenize_fn(examples):
        return preprocess_function(
            examples,
            tokenizer,
            config.task_name,
            max_length=config.max_seq_length,
            padding="max_length" if config.pad_to_max_length else "do_not_pad",
        )
    
    tokenized_datasets = raw_datasets.map(
        tokenize_fn,
        batched=True,
        num_proc=config.preprocessing_num_workers,
        load_from_cache_file=not config.overwrite_cache,
        desc="Tokenizing",
    )
    
    # 移除不需要的列
    columns_to_remove = [col for col in tokenized_datasets["train"].column_names 
                         if col not in ["input_ids", "attention_mask", "label"]]
    
    tokenized_datasets = tokenized_datasets.remove_columns(columns_to_remove)
    
    # 重命名 label 为 labels（Trainer 需要）
    if "label" in tokenized_datasets["train"].column_names:
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    
    logger.info(f"Datasets prepared: {tokenized_datasets}")
    logger.info(f"Number of labels: {num_labels}")
    
    return tokenized_datasets, num_labels


def get_metric_name(task_name: str) -> str:
    """获取任务的主要评估指标名称"""
    return "accuracy"  # MNLI 和 RTE 都使用 accuracy


if __name__ == "__main__":
    # 测试数据加载
    from transformers import AutoTokenizer
    from config import DataConfig
    
    config = DataConfig(task_name="mnli", max_seq_length=128)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    
    datasets, num_labels = prepare_datasets(config, tokenizer)
    
    print(f"Train size: {len(datasets['train'])}")
    print(f"Validation size: {len(datasets['validation'])}")
    print(f"Sample: {datasets['train'][0]}")
