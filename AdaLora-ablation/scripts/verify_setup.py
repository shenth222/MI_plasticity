#!/usr/bin/env python3
"""
验证环境设置
检查所有依赖是否正确安装
"""

import sys

def check_imports():
    """检查所有必要的包"""
    packages = {
        "torch": "PyTorch",
        "transformers": "Transformers",
        "peft": "PEFT",
        "datasets": "Datasets",
        "accelerate": "Accelerate",
        "evaluate": "Evaluate",
        "numpy": "NumPy",
        "matplotlib": "Matplotlib",
        "seaborn": "Seaborn",
        "pandas": "Pandas",
    }
    
    print("=" * 60)
    print("Checking package imports...")
    print("=" * 60)
    
    all_ok = True
    
    for package, name in packages.items():
        try:
            module = __import__(package)
            version = getattr(module, "__version__", "unknown")
            print(f"✓ {name:20s} ({package:15s}): {version}")
        except ImportError as e:
            print(f"✗ {name:20s} ({package:15s}): NOT FOUND")
            all_ok = False
    
    print("=" * 60)
    
    return all_ok


def check_peft_version():
    """检查 PEFT 版本"""
    import peft
    from packaging import version
    
    required = "0.18.1"
    current = peft.__version__
    
    print("\nChecking PEFT version...")
    print(f"  Required: {required}")
    print(f"  Current:  {current}")
    
    if version.parse(current) == version.parse(required):
        print("  ✓ Version matches!")
        return True
    else:
        print("  ⚠ Version mismatch! Patching may not work correctly.")
        return False


def check_model_path():
    """检查模型路径"""
    import os
    
    model_path = "/data1/shenth/models/deberta/v3-base"
    
    print("\nChecking model path...")
    print(f"  Path: {model_path}")
    
    if os.path.exists(model_path):
        print("  ✓ Model directory exists!")
        
        # 检查关键文件
        required_files = ["config.json", "pytorch_model.bin"]
        all_exist = True
        
        for f in required_files:
            path = os.path.join(model_path, f)
            if os.path.exists(path):
                print(f"    ✓ {f}")
            else:
                print(f"    ✗ {f} NOT FOUND")
                all_exist = False
        
        return all_exist
    else:
        print("  ✗ Model directory NOT FOUND!")
        print("  Please update MODEL_PATH in src/config.py")
        return False


def check_cuda():
    """检查 CUDA"""
    import torch
    
    print("\nChecking CUDA...")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  Device count: {torch.cuda.device_count()}")
        print(f"  Current device: {torch.cuda.current_device()}")
        print(f"  Device name: {torch.cuda.get_device_name(0)}")
        return True
    else:
        print("  ⚠ CUDA not available (CPU-only mode)")
        return False


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("AdaLoRA Ablation - Environment Verification")
    print("=" * 60 + "\n")
    
    results = {
        "Imports": check_imports(),
        "PEFT version": check_peft_version(),
        "Model path": check_model_path(),
        "CUDA": check_cuda(),
    }
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    
    for name, ok in results.items():
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {name:20s}: {status}")
    
    print("=" * 60)
    
    if all(results.values()):
        print("\n✓ All checks passed! You're ready to go.")
        return 0
    else:
        print("\n⚠ Some checks failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
