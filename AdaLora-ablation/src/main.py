"""
主入口模块
提供 train / eval / export 功能
"""

import os
import sys
import logging
import math
import torch
import numpy as np
from transformers import Trainer, TrainingArguments, set_seed, AutoTokenizer
import evaluate

# 添加当前目录到 Python path
sys.path.insert(0, os.path.dirname(__file__))

from config import parse_args, ExperimentConfig
from data import prepare_datasets
from modeling import load_base_model_and_tokenizer, create_adalora_model
from signal_tracker import SignalTracker
from callbacks import AdaLoRACallback, BudgetConsistencyCallback, MetricsWriterCallback
from patch_adalora import apply_patch
from logging_utils import (
    setup_experiment_logging,
    create_final_summary,
    summarize_metrics,
    summarize_rank_pattern,
)

logger = logging.getLogger(__name__)


def compute_metrics(eval_pred):
    """计算评估指标"""
    metric = evaluate.load("accuracy")
    
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    return metric.compute(predictions=predictions, references=labels)


def compute_total_steps(num_examples: int, config: ExperimentConfig) -> int:
    """计算总训练步数（与 Trainer 逻辑一致）"""
    if num_examples <= 0:
        raise ValueError("Training dataset is empty, cannot compute total steps.")
    
    per_device_bs = config.training.per_device_train_batch_size
    grad_accum = config.training.gradient_accumulation_steps
    
    if per_device_bs <= 0 or grad_accum <= 0:
        raise ValueError("Batch size and gradient_accumulation_steps must be positive.")
    
    # len_dataloader = ceil(num_examples / per_device_bs)
    len_dataloader = math.ceil(num_examples / per_device_bs)
    num_update_steps_per_epoch = max(
        len_dataloader // grad_accum + int(len_dataloader % grad_accum > 0),
        1,
    )
    
    total_steps = math.ceil(config.training.num_train_epochs * num_update_steps_per_epoch)
    return total_steps


def train(config: ExperimentConfig):
    """
    训练主函数
    
    Args:
        config: 实验配置
    """
    # 设置随机种子
    set_seed(config.training.seed)
    
    # 设置日志
    writers = setup_experiment_logging(config.training.output_dir, logging.INFO)
    
    logger.info("=" * 80)
    logger.info("Starting AdaLoRA Signal-Replacement Ablation Experiment")
    logger.info("=" * 80)
    logger.info(f"Task: {config.data.task_name}")
    logger.info(f"Signal type: {config.signal.signal_type}")
    logger.info(f"Seed: {config.training.seed}")
    logger.info(f"Output dir: {config.training.output_dir}")
    
    # 1. 加载模型和 tokenizer
    logger.info("\n[1/7] Loading model and tokenizer...")
    base_model, tokenizer = load_base_model_and_tokenizer(
        config.model,
        num_labels=3 if config.data.task_name == "mnli" else 2,
    )
    
    # 2. 准备数据
    logger.info("\n[2/7] Preparing datasets...")
    datasets, num_labels = prepare_datasets(
        config.data,
        tokenizer,
        cache_dir=config.model.cache_dir,
    )
    
    # 3. 计算总训练步数（AdaLoRA 必需）
    logger.info("\n[3/7] Computing total training steps...")
    computed_total_steps = compute_total_steps(len(datasets["train"]), config)
    if config.adalora.total_step is None:
        config.adalora.total_step = computed_total_steps
    elif config.adalora.total_step != computed_total_steps:
        logger.warning(
            f"Using overridden total_step={config.adalora.total_step}, "
            f"computed total_step={computed_total_steps} from current training setup."
        )
    logger.info(f"Total training steps: {config.adalora.total_step}")
    
    if config.adalora.tinit >= (config.adalora.total_step - config.adalora.tfinal):
        raise ValueError(
            "AdaLoRA schedule invalid: require tinit < total_step - tfinal. "
            f"Got tinit={config.adalora.tinit}, tfinal={config.adalora.tfinal}, "
            f"total_step={config.adalora.total_step}."
        )
    
    # 保存配置（含 total_step）
    config.save(os.path.join(config.training.output_dir, "config.json"))
    
    # 4. 应用 AdaLoRA
    logger.info("\n[4/7] Applying AdaLoRA...")
    model = create_adalora_model(base_model, config.adalora)
    
    # 5. 应用 patch（如果需要）
    logger.info("\n[5/7] Applying AdaLoRA patch...")
    apply_patch(config.signal.signal_type)
    
    # 6. 初始化 signal tracker（如果需要）
    signal_tracker = None
    use_external_scores = config.signal.signal_type != "baseline_adalora"
    
    if use_external_scores:
        logger.info("\n[6/7] Initializing signal tracker...")
        signal_tracker = SignalTracker(
            signal_type=config.signal.signal_type,
            ema_decay=config.signal.ema_decay,
            combo_lambda=config.signal.combo_lambda,
            normalize_method=config.signal.normalize_method,
        )
    else:
        logger.info("\n[6/7] Using baseline AdaLoRA (no signal tracker)")
    
    # 7. 设置 Trainer
    logger.info("\n[7/7] Setting up Trainer...")
    
    training_args = TrainingArguments(
        output_dir=config.training.output_dir,
        num_train_epochs=config.training.num_train_epochs,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        per_device_eval_batch_size=config.training.per_device_eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        max_grad_norm=config.training.max_grad_norm,
        warmup_ratio=config.training.warmup_ratio,
        warmup_steps=config.training.warmup_steps,
        optim=config.training.optim,
        adam_beta1=config.training.adam_beta1,
        adam_beta2=config.training.adam_beta2,
        adam_epsilon=config.training.adam_epsilon,
        fp16=config.training.fp16,
        bf16=config.training.bf16,
        eval_strategy=config.training.evaluation_strategy,
        save_strategy=config.training.save_strategy,
        save_total_limit=config.training.save_total_limit,
        load_best_model_at_end=config.training.load_best_model_at_end,
        metric_for_best_model=config.training.metric_for_best_model,
        greater_is_better=config.training.greater_is_better,
        logging_dir=config.training.logging_dir,
        logging_steps=config.training.logging_steps,
        report_to=config.training.report_to,
        remove_unused_columns=config.training.remove_unused_columns,
        dataloader_num_workers=config.training.dataloader_num_workers,
        dataloader_pin_memory=config.training.dataloader_pin_memory,
        disable_tqdm=config.training.disable_tqdm,
        seed=config.training.seed,
    )
    
    # Callbacks
    callbacks = [
        MetricsWriterCallback(metrics_logger=writers["metrics"]),
        AdaLoRACallback(
            signal_tracker=signal_tracker,
            rank_logger=writers["rank_pattern"],
            signal_logger=writers["signal_scores"],
            log_rank_every=config.signal.log_rank_every,
            log_signal_every=config.signal.log_signal_every,
            use_external_scores=use_external_scores,
        ),
        BudgetConsistencyCallback(
            tolerance=0.1,
        ),
    ]
    
    # 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        callbacks=callbacks,
    )
    
    # 7. 训练
    logger.info("\n" + "=" * 80)
    logger.info("Starting training...")
    logger.info("=" * 80)
    
    train_result = trainer.train()
    
    # 8. 保存模型
    logger.info("\nSaving model...")
    trainer.save_model()
    if tokenizer is not None:
        tokenizer.save_pretrained(config.training.output_dir)
    
    # 9. 最终评估
    logger.info("\nRunning final evaluation...")
    eval_results = trainer.evaluate()
    
    logger.info("\n" + "=" * 80)
    logger.info("Training completed!")
    logger.info(f"Final eval accuracy: {eval_results.get('eval_accuracy', 0):.4f}")
    logger.info("=" * 80)
    
    # 10. 关闭日志写入器
    for writer in writers.values():
        writer.close()
    
    # 11. 生成最终汇总
    logger.info("\nGenerating final summary...")
    
    metrics_summary = summarize_metrics(
        os.path.join(config.training.output_dir, "metrics.jsonl")
    )
    
    rank_summary = summarize_rank_pattern(
        os.path.join(config.training.output_dir, "rank_pattern.jsonl")
    )
    
    final_summary = create_final_summary(
        config.training.output_dir,
        config=config.to_dict(),
        metrics_summary=metrics_summary,
        rank_summary=rank_summary,
    )
    
    logger.info(f"\n✓ All outputs saved to: {config.training.output_dir}")
    
    return train_result, eval_results


def evaluate(config: ExperimentConfig):
    """
    评估模型
    
    Args:
        config: 实验配置
    """
    logger.info("Loading model for evaluation...")
    
    # 加载 checkpoint
    checkpoint_dir = config.training.output_dir
    
    if not os.path.exists(checkpoint_dir):
        raise ValueError(f"Checkpoint not found: {checkpoint_dir}")
    
    # 加载模型
    from transformers import AutoModelForSequenceClassification
    from peft import PeftModel
    
    base_model = AutoModelForSequenceClassification.from_pretrained(
        config.model.model_name_or_path,
        num_labels=3 if config.data.task_name == "mnli" else 2,
    )
    
    model = PeftModel.from_pretrained(base_model, checkpoint_dir)
    
    # 加载 tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    except Exception:
        logger.warning("Tokenizer not found in checkpoint, falling back to base model tokenizer.")
        tokenizer = AutoTokenizer.from_pretrained(config.model.model_name_or_path)
    
    # 准备数据
    datasets, _ = prepare_datasets(config.data, tokenizer)
    
    # 创建 Trainer
    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        per_device_eval_batch_size=config.training.per_device_eval_batch_size,
        dataloader_num_workers=config.training.dataloader_num_workers,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=datasets["validation"],
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )
    
    # 评估
    eval_results = trainer.evaluate()
    
    logger.info("Evaluation results:")
    for key, value in eval_results.items():
        logger.info(f"  {key}: {value}")
    
    return eval_results


def export_results(config: ExperimentConfig):
    """
    导出实验结果
    
    Args:
        config: 实验配置
    """
    logger.info("Exporting results...")
    
    output_dir = config.training.output_dir
    
    # 读取并打印汇总
    import json
    
    summary_file = os.path.join(output_dir, "final_summary.json")
    
    if os.path.exists(summary_file):
        with open(summary_file) as f:
            summary = json.load(f)
        
        print("\n" + "=" * 80)
        print("EXPERIMENT SUMMARY")
        print("=" * 80)
        print(json.dumps(summary, indent=2))
        print("=" * 80)
    else:
        logger.warning(f"Summary file not found: {summary_file}")


def main():
    """主函数"""
    # 解析参数
    config, mode = parse_args()
    
    # 根据模式执行
    if mode == "train":
        train(config)
    elif mode == "eval":
        evaluate(config)
    elif mode == "export":
        export_results(config)
    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    main()
