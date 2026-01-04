"""
主程序入口
"""
import sys
import os
from pathlib import Path
import numpy as np

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.args import get_config
from src.utils.seed import set_seed
from src.utils.logging import setup_logger, get_logger
from src.utils.io import save_json, save_yaml, save_csv, ensure_dir
from src.utils.stats import layer_wise_normalize
from src.data.cs170k_dataset import CS170kDataset
from src.model.load_model import load_tokenizer_and_model, get_model_info
from src.model.forward import batch_inference
from src.scoring.out_norm import aggregate_head_output_scores
from src.scoring.entropy import compute_entropy_scores_all_layers
from src.scoring.task_align import compute_task_alignment_scores_all_layers
from src.scoring.combine import (
    combine_scores,
    get_topk_heads,
    get_topk_heads_per_layer,
    prepare_scores_for_saving
)


def main():
    """主函数"""
    # 1. 加载配置
    print("=" * 80)
    print("Task-Induced Activation Intensity - Pre-Finetuning Head Scoring")
    print("=" * 80)
    
    config = get_config()
    
    # 2. 创建输出目录
    output_dir = config["output"]["dir"]
    ensure_dir(output_dir)
    
    # 3. 设置 logger
    logger = setup_logger(
        name="main",
        log_file=os.path.join(output_dir, "run.log")
    )
    
    logger.info("=" * 80)
    logger.info("开始运行")
    logger.info("=" * 80)
    
    # 4. 保存配置
    config_path = os.path.join(output_dir, "config.yaml")
    save_yaml(config, config_path)
    logger.info(f"配置已保存到: {config_path}")
    
    # 5. 设置随机种子
    seed = config["inference"]["seed"]
    set_seed(seed)
    logger.info(f"随机种子设置为: {seed}")
    
    # 6. 加载数据集
    logger.info("正在加载数据集...")
    dataset = CS170kDataset(
        data_path=config["data"]["path"],
        max_samples=config["data"]["max_samples"],
        field_mapping=config["data"].get("field_mapping")
    )
    
    dataset_stats = dataset.get_stats()
    logger.info(f"数据集统计:")
    logger.info(f"  - 总样本数: {dataset_stats['total_samples']}")
    logger.info(f"  - 平均问题长度: {dataset_stats['avg_question_length']:.1f} 字符")
    logger.info(f"  - 选项数范围: {dataset_stats['num_choices_range']}")
    
    # 7. 加载模型
    logger.info("正在加载模型...")
    tokenizer, model = load_tokenizer_and_model(
        model_path=config["model"]["path"],
        dtype=config["model"]["dtype"],
        device=config["inference"]["device"],
        attn_implementation=config["model"].get("attn_implementation", "eager")
    )
    
    model_info = get_model_info(model)
    logger.info(f"模型信息:")
    logger.info(f"  - 层数: {model_info.get('num_layers', 'N/A')}")
    logger.info(f"  - 注意力头数: {model_info.get('num_heads', 'N/A')}")
    logger.info(f"  - 隐藏层大小: {model_info.get('hidden_size', 'N/A')}")
    logger.info(f"  - Head 维度: {model_info.get('head_dim', 'N/A')}")
    
    num_layers = model_info["num_layers"]
    num_heads = model_info["num_heads"]
    
    # 8. 批量推理
    logger.info("=" * 80)
    logger.info("开始推理...")
    logger.info("=" * 80)
    
    inference_results = batch_inference(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        batch_size=config["inference"]["batch_size"],
        max_length=config["data"]["max_length"],
        device=config["inference"]["device"],
        prompt_template=config["prompt"]["template"],
        logger=logger
    )
    
    logger.info(f"推理完成，共 {len(inference_results)} 个批次")
    
    # 9. 计算评分
    logger.info("=" * 80)
    logger.info("开始计算评分...")
    logger.info("=" * 80)
    
    query_mode = config["scoring"]["query_mode"]
    norm_mode = config["scoring"]["norm_mode"]
    
    # 9.1 Head Output 强度
    logger.info("计算 Head Output 强度...")
    out_scores, out_stats = aggregate_head_output_scores(
        inference_results=inference_results,
        num_heads=num_heads,
        query_mode=query_mode
    )
    logger.info(f"Head Output 强度统计:")
    logger.info(f"  - Mean: {out_stats['mean']:.4f}")
    logger.info(f"  - Std: {out_stats['std']:.4f}")
    logger.info(f"  - Min: {out_stats['min']:.4f}")
    logger.info(f"  - Max: {out_stats['max']:.4f}")
    
    # 9.2 Attention Entropy
    logger.info("计算 Attention Entropy...")
    ent_scores, ent_stats = compute_entropy_scores_all_layers(
        inference_results=inference_results,
        query_mode=query_mode
    )
    logger.info(f"Attention Entropy 统计:")
    logger.info(f"  - Mean: {ent_stats['mean']:.4f}")
    logger.info(f"  - Std: {ent_stats['std']:.4f}")
    logger.info(f"  - Min: {ent_stats['min']:.4f}")
    logger.info(f"  - Max: {ent_stats['max']:.4f}")
    
    # 9.3 Task Alignment
    logger.info("计算 Task Alignment...")
    task_scores, task_stats = compute_task_alignment_scores_all_layers(
        inference_results=inference_results,
        query_mode=query_mode
    )
    
    if task_scores is not None:
        logger.info(f"Task Alignment 统计:")
        logger.info(f"  - Mean: {task_stats['mean']:.4f}")
        logger.info(f"  - Std: {task_stats['std']:.4f}")
        logger.info(f"  - Min: {task_stats['min']:.4f}")
        logger.info(f"  - Max: {task_stats['max']:.4f}")
        logger.info(f"  - 成功率: {task_stats['success_rate']:.2%} ({task_stats['valid_samples']}/{task_stats['total_samples']})")
    else:
        logger.warning("Task Alignment 计算失败，将使用零分数")
        task_scores = np.zeros_like(out_scores)
    
    # 10. Layer-wise Normalization
    logger.info("=" * 80)
    logger.info(f"进行 Layer-wise Normalization (模式: {norm_mode})...")
    logger.info("=" * 80)
    
    # 准备归一化
    scores_dict = {
        "out": out_scores.flatten().tolist(),
        "ent": ent_scores.flatten().tolist(),
    }
    
    if task_scores is not None:
        scores_dict["task"] = task_scores.flatten().tolist()
    
    # 归一化
    normalized_dict = layer_wise_normalize(
        scores_dict=scores_dict,
        num_layers=num_layers,
        num_heads=num_heads,
        mode=norm_mode
    )
    
    out_scores_norm = np.array(normalized_dict["out"]).reshape(num_layers, num_heads)
    ent_scores_norm = np.array(normalized_dict["ent"]).reshape(num_layers, num_heads)
    
    if "task" in normalized_dict:
        task_scores_norm = np.array(normalized_dict["task"]).reshape(num_layers, num_heads)
    else:
        task_scores_norm = None
    
    # 11. 组合评分
    logger.info("=" * 80)
    logger.info("计算组合评分...")
    logger.info("=" * 80)
    
    combined_scores = combine_scores(
        out_scores=out_scores_norm,
        ent_scores=ent_scores_norm,
        task_scores=task_scores_norm,
        lambda_ent=config["scoring"]["lambda_ent"],
        lambda_task=config["scoring"]["lambda_task"],
        num_layers=num_layers,
        num_heads=num_heads
    )
    
    logger.info(f"组合评分统计:")
    logger.info(f"  - Mean: {np.mean(combined_scores):.4f}")
    logger.info(f"  - Std: {np.std(combined_scores):.4f}")
    logger.info(f"  - Min: {np.min(combined_scores):.4f}")
    logger.info(f"  - Max: {np.max(combined_scores):.4f}")
    
    # 12. 获取 Top-k
    logger.info("=" * 80)
    logger.info("获取 Top-k heads...")
    logger.info("=" * 80)
    
    topk_global = get_topk_heads(
        scores=combined_scores,
        k=config["scoring"]["topk_global"]
    )
    
    topk_per_layer = get_topk_heads_per_layer(
        scores=combined_scores,
        k=config["scoring"]["topk_per_layer"]
    )
    
    logger.info(f"Top-{config['scoring']['topk_global']} heads (全局):")
    for i, head_info in enumerate(topk_global[:10]):  # 只打印前 10 个
        logger.info(f"  {i+1}. Layer {head_info['layer']}, Head {head_info['head']}: {head_info['score']:.4f}")
    
    # 13. 保存结果
    logger.info("=" * 80)
    logger.info("保存结果...")
    logger.info("=" * 80)
    
    # 准备数据
    raw_data, norm_data, combined_data = prepare_scores_for_saving(
        out_scores=out_scores,
        ent_scores=ent_scores,
        task_scores=task_scores,
        out_scores_norm=out_scores_norm,
        ent_scores_norm=ent_scores_norm,
        task_scores_norm=task_scores_norm,
        combined_scores=combined_scores
    )
    
    # 保存 CSV
    if config["output"]["save_raw"]:
        raw_path = os.path.join(output_dir, "scores_raw.csv")
        save_csv(raw_data, raw_path)
        logger.info(f"原始分数已保存到: {raw_path}")
    
    if config["output"]["save_normalized"]:
        norm_path = os.path.join(output_dir, "scores_norm.csv")
        save_csv(norm_data, norm_path)
        logger.info(f"归一化分数已保存到: {norm_path}")
    
    if config["output"]["save_combined"]:
        combined_path = os.path.join(output_dir, "scores_combined.csv")
        save_csv(combined_data, combined_path)
        logger.info(f"组合分数已保存到: {combined_path}")
    
    # 保存 Top-k JSON
    if config["output"]["save_topk"]:
        topk_global_path = os.path.join(output_dir, "topk_global.json")
        save_json(topk_global, topk_global_path)
        logger.info(f"Top-k 全局已保存到: {topk_global_path}")
        
        topk_per_layer_path = os.path.join(output_dir, "topk_per_layer.json")
        save_json(topk_per_layer, topk_per_layer_path)
        logger.info(f"Top-k 每层已保存到: {topk_per_layer_path}")
    
    # 14. 完成
    logger.info("=" * 80)
    logger.info("运行完成！")
    logger.info("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("=" * 80)
        print("运行失败！")
        print("=" * 80)
        print(f"错误: {e}")
        print("\n完整错误信息:")
        traceback.print_exc()
        sys.exit(1)

