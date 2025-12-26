"""
使用示例：展示如何使用激活提取工具
"""

import torch
from activation_extract import ActivationExtractor, ActivationComparator, create_simple_dataloader
from get_random_samples import get_random_samples
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    # 配置参数
    pre_tuned_model_path = "/data1/shenth/models/llama/3.2-1b"
    fine_tuned_model_path = "/data1/shenth/work/finetune/full/llama3.2-1b/results/3.2-1b-full-commonsense-test"
    data_path = "/data1/shenth/datasets/commonsense/merged_commonsense_train.json"
    sample_num = 1600
    batch_size = 8
    output_dir = "./activation_results"

    # 1. 加载数据
    print("加载数据...")
    samples = get_random_samples(data_path, sample_num, seed=42)
    print(f"加载了 {len(samples)} 个样本")

    # 2. 加载模型和 tokenizer
    print("加载微调前模型...")
    pre_tokenizer = AutoTokenizer.from_pretrained(pre_tuned_model_path)
    pre_model = AutoModelForCausalLM.from_pretrained(pre_tuned_model_path)

    print("加载微调后模型...")
    fine_tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_path)
    fine_model = AutoModelForCausalLM.from_pretrained(fine_tuned_model_path)

    # 设置 padding token
    if pre_tokenizer.pad_token is None:
        pre_tokenizer.pad_token = pre_tokenizer.eos_token
    if fine_tokenizer.pad_token is None:
        fine_tokenizer.pad_token = fine_tokenizer.eos_token

    # 3. 创建 tokenizer 函数（用于 extract_activations 的备用 tokenizer）
    def tokenize_fn(texts):
        if isinstance(texts, str):
            texts = [texts]
        return pre_tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

    # 4. 创建数据加载器（在数据加载器内部进行 tokenization）
    dataloader = create_simple_dataloader(samples, pre_tokenizer, batch_size)

    # 5. 创建激活提取器
    print("创建激活提取器...")
    pre_extractor = ActivationExtractor(pre_model, "pre_tuned", model_path=pre_tuned_model_path)
    fine_extractor = ActivationExtractor(fine_model, "fine_tuned", model_path=fine_tuned_model_path)

    # 如果自动检测失败，手动设置模型结构
    # pre_extractor.set_model_structure(num_layers=12, num_heads=12)
    # fine_extractor.set_model_structure(num_layers=12, num_heads=12)

    # 6. 注册 hooks
    print("注册 hooks...")
    # 提取所有层和头
    pre_extractor.register_hooks()
    fine_extractor.register_hooks()

    # 或者只提取特定层和头
    # pre_extractor.register_hooks(layers=[0, 1, 2], heads=[0, 1, 2, 3])
    # fine_extractor.register_hooks(layers=[0, 1, 2], heads=[0, 1, 2, 3])

    # 7. 提取激活值
    # 注意：数据加载器已经在内部进行了 tokenization，所以不需要传入 tokenizer
    print("提取微调前模型激活值...")
    pre_extractor.extract_activations(dataloader, tokenizer=None)

    print("提取微调后模型激活值...")
    fine_extractor.extract_activations(dataloader, tokenizer=None)

    # 8. 查看结果
    print("\n微调前模型激活强度 (R_h):")
    pre_R_h = pre_extractor.compute_R_h()
    for (layer, head), R_h in sorted(pre_R_h.items())[:10]:  # 只显示前10个
        print(f"  Layer {layer}, Head {head}: R_h = {R_h:.4f}")

    print("\n微调后模型激活强度 (R_h):")
    fine_R_h = fine_extractor.compute_R_h()
    for (layer, head), R_h in sorted(fine_R_h.items())[:10]:  # 只显示前10个
        print(f"  Layer {layer}, Head {head}: R_h = {R_h:.4f}")

    # 9. 保存结果
    print("\n保存结果...")
    import os
    os.makedirs(output_dir, exist_ok=True)

    pre_extractor.save_results(os.path.join(output_dir, "pre_tuned_activations.json"))
    fine_extractor.save_results(os.path.join(output_dir, "fine_tuned_activations.json"))

    # 10. 对比分析
    print("进行对比分析...")
    comparator = ActivationComparator(pre_extractor, fine_extractor)

    comparison = comparator.compare()
    print("\n激活强度变化最大的头（前10个）:")
    sorted_diffs = sorted(comparison['differences'].items(), 
                          key=lambda x: abs(x[1]), reverse=True)[:10]
    for (layer, head), diff in sorted_diffs:
        ratio = comparison['ratios'][(layer, head)]
        print(f"  Layer {layer}, Head {head}: 差异 = {diff:.4f}, 比率 = {ratio:.4f}")

    # 保存对比结果
    comparator.save_comparison(os.path.join(output_dir, "comparison.json"))
    comparator.visualize_comparison(os.path.join(output_dir, "comparison_heatmap.png"))

    # 11. 清理
    pre_extractor.remove_hooks()
    fine_extractor.remove_hooks()

    print(f"\n完成！结果保存在 {output_dir}")


if __name__ == "__main__":
    main()

