#!/usr/bin/env python
"""
最小测试脚本 - 验证项目结构和依赖
"""
import sys
from pathlib import Path

def test_imports():
    """测试所有模块导入"""
    print("测试模块导入...")
    
    try:
        from src.args import get_config, parse_args
        print("✓ args")
    except Exception as e:
        print(f"✗ args: {e}")
        return False
    
    try:
        from src.utils.seed import set_seed
        from src.utils.io import save_json, load_json
        from src.utils.logging import setup_logger
        from src.utils.span import extract_spans_from_prompt
        from src.utils.stats import normalize_zscore
        print("✓ utils")
    except Exception as e:
        print(f"✗ utils: {e}")
        return False
    
    try:
        from src.data.cs170k_dataset import CS170kDataset
        from src.data.prompt import create_prompt
        print("✓ data")
    except Exception as e:
        print(f"✗ data: {e}")
        return False
    
    try:
        from src.model.load_model import load_tokenizer_and_model
        from src.model.hooks import AttentionOutputHook
        from src.model.forward import forward_with_cache
        print("✓ model")
    except Exception as e:
        print(f"✗ model: {e}")
        return False
    
    try:
        from src.scoring.out_norm import compute_head_output_norm
        from src.scoring.entropy import compute_attention_entropy
        from src.scoring.task_align import compute_task_alignment_score
        from src.scoring.combine import combine_scores
        print("✓ scoring")
    except Exception as e:
        print(f"✗ scoring: {e}")
        return False
    
    return True


def test_dependencies():
    """测试依赖包"""
    print("\n测试依赖包...")
    
    packages = {
        "torch": "torch",
        "transformers": "transformers",
        "numpy": "numpy",
        "pandas": "pandas",
        "tqdm": "tqdm",
        "yaml": "pyyaml",
        "sklearn": "scikit-learn",
        "scipy": "scipy"
    }
    
    all_ok = True
    for pkg_import, pkg_name in packages.items():
        try:
            __import__(pkg_import)
            print(f"✓ {pkg_name}")
        except ImportError:
            print(f"✗ {pkg_name} - 未安装，请运行: pip install {pkg_name}")
            all_ok = False
    
    return all_ok


def test_directory_structure():
    """测试目录结构"""
    print("\n测试目录结构...")
    
    base_dir = Path(__file__).parent
    required_dirs = [
        "src",
        "src/utils",
        "src/data",
        "src/model",
        "src/scoring",
        "configs"
    ]
    
    required_files = [
        "README.md",
        "requirements.txt",
        "configs/default.yaml",
        "src/main.py",
        "src/args.py"
    ]
    
    all_ok = True
    
    for dir_path in required_dirs:
        full_path = base_dir / dir_path
        if full_path.exists():
            print(f"✓ {dir_path}/")
        else:
            print(f"✗ {dir_path}/ - 不存在")
            all_ok = False
    
    for file_path in required_files:
        full_path = base_dir / file_path
        if full_path.exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} - 不存在")
            all_ok = False
    
    return all_ok


def test_example_data():
    """测试示例数据加载"""
    print("\n测试示例数据...")
    
    try:
        from src.data.cs170k_dataset import CS170kDataset
        
        dataset = CS170kDataset(
            data_path="example_data.jsonl",
            max_samples=10
        )
        
        print(f"✓ 成功加载 {len(dataset)} 个样本")
        
        # 测试第一个样本
        sample = dataset[0]
        print(f"  - 问题: {sample['question'][:50]}...")
        print(f"  - 选项数: {len(sample['choices'])}")
        
        return True
    
    except Exception as e:
        print(f"✗ 加载失败: {e}")
        return False


def main():
    """主函数"""
    print("=" * 60)
    print("Task-Induced Activation Intensity - 项目结构测试")
    print("=" * 60)
    
    results = []
    
    # 测试目录结构
    results.append(("目录结构", test_directory_structure()))
    
    # 测试依赖
    results.append(("依赖包", test_dependencies()))
    
    # 测试导入
    results.append(("模块导入", test_imports()))
    
    # 测试示例数据
    results.append(("示例数据", test_example_data()))
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    
    if all_passed:
        print("✓ 所有测试通过！项目已准备就绪。")
        print("\n下一步：")
        print("1. 修改 configs/default.yaml 中的模型和数据路径")
        print("2. 运行: python src/main.py --config configs/default.yaml")
        return 0
    else:
        print("✗ 部分测试失败，请检查上述错误。")
        return 1


if __name__ == "__main__":
    sys.exit(main())

