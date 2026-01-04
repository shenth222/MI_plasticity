"""
文件 I/O 工具模块
"""
import json
import csv
import yaml
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd


def ensure_dir(path: str):
    """确保目录存在，不存在则创建"""
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(data: Any, filepath: str):
    """保存 JSON 文件"""
    ensure_dir(str(Path(filepath).parent))
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(filepath: str) -> Any:
    """加载 JSON 文件"""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_yaml(data: Any, filepath: str):
    """保存 YAML 文件"""
    ensure_dir(str(Path(filepath).parent))
    with open(filepath, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def load_yaml(filepath: str) -> Any:
    """加载 YAML 文件"""
    with open(filepath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_csv(data: List[Dict[str, Any]], filepath: str):
    """保存 CSV 文件"""
    ensure_dir(str(Path(filepath).parent))
    if not data:
        return
    
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)


def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """
    加载 JSONL 文件（每行一个 JSON 对象）
    
    Args:
        filepath: 文件路径
        
    Returns:
        List of dictionaries
    """
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_data_file(filepath: str) -> List[Dict[str, Any]]:
    """
    自动识别并加载数据文件（支持 JSON 和 JSONL）
    
    Args:
        filepath: 文件路径
        
    Returns:
        List of dictionaries
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"数据文件不存在: {filepath}")
    
    # 尝试作为 JSONL 加载
    try:
        data = load_jsonl(str(filepath))
        if data:
            return data
    except:
        pass
    
    # 尝试作为 JSON 加载
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # 如果是列表，直接返回
        if isinstance(data, list):
            return data
        
        # 如果是字典，尝试找到数据列表
        if isinstance(data, dict):
            # 常见的键名
            for key in ["data", "examples", "instances", "samples"]:
                if key in data and isinstance(data[key], list):
                    return data[key]
            
            # 如果字典本身就是一个样本，包装成列表
            return [data]
    except Exception as e:
        raise ValueError(f"无法加载数据文件 {filepath}: {e}")
    
    raise ValueError(f"不支持的数据文件格式: {filepath}")

