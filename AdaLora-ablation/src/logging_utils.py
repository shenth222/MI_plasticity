"""
日志工具模块
提供 JSONL 格式的日志记录和结果汇总
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class JSONLWriter:
    """JSONL 格式日志写入器"""
    
    def __init__(self, filepath: str, mode: str = "w"):
        """
        Args:
            filepath: 文件路径
            mode: 写入模式（w / a）
        """
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        
        self.mode = mode
        self.file_handle = None
        self._open()
    
    def _open(self):
        """打开文件"""
        self.file_handle = open(self.filepath, self.mode, encoding="utf-8")
    
    def write(self, record: Dict[str, Any]):
        """写入一条记录"""
        if self.file_handle is None:
            self._open()
        
        # 添加时间戳
        if "timestamp" not in record:
            record["timestamp"] = datetime.now().isoformat()
        
        # 写入
        line = json.dumps(record, ensure_ascii=False)
        self.file_handle.write(line + "\n")
        self.file_handle.flush()
    
    def close(self):
        """关闭文件"""
        if self.file_handle is not None:
            self.file_handle.close()
            self.file_handle = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def __del__(self):
        self.close()


def read_jsonl(filepath: str) -> List[Dict]:
    """读取 JSONL 文件"""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def summarize_metrics(metrics_file: str) -> Dict[str, Any]:
    """
    汇总训练指标
    
    Args:
        metrics_file: metrics.jsonl 文件路径
        
    Returns:
        汇总统计
    """
    records = read_jsonl(metrics_file)
    
    if not records:
        return {}
    
    # 找到最后一个 eval 记录
    eval_records = [r for r in records if "eval_loss" in r]
    
    if not eval_records:
        return {}
    
    final_eval = eval_records[-1]
    
    summary = {
        "final_eval_loss": final_eval.get("eval_loss"),
        "final_eval_accuracy": final_eval.get("eval_accuracy"),
        "total_epochs": final_eval.get("epoch"),
        "num_eval_records": len(eval_records),
    }
    
    # 添加最佳指标
    best_accuracy = max(r.get("eval_accuracy", 0) for r in eval_records)
    summary["best_eval_accuracy"] = best_accuracy
    
    return summary


def summarize_rank_pattern(rank_file: str) -> Dict[str, Any]:
    """
    汇总 rank 分配模式
    
    Args:
        rank_file: rank_pattern.jsonl 文件路径
        
    Returns:
        汇总统计
    """
    records = read_jsonl(rank_file)
    
    if not records:
        return {}
    
    # 找到最后一个记录（final）
    final_record = None
    for r in reversed(records):
        if r.get("is_final", False):
            final_record = r
            break
    
    if final_record is None and records:
        final_record = records[-1]
    
    if final_record is None:
        return {}
    
    summary = {
        "final_total_rank": final_record.get("total_rank"),
        "final_num_modules": final_record.get("num_modules"),
        "final_avg_rank": final_record["total_rank"] / max(1, final_record["num_modules"]),
        "num_updates": len(records),
    }
    
    # 统计 rank 变化
    if len(records) > 1:
        initial_rank = records[0]["total_rank"]
        final_rank = final_record["total_rank"]
        summary["rank_reduction"] = initial_rank - final_rank
        summary["rank_reduction_ratio"] = (initial_rank - final_rank) / max(1, initial_rank)
    
    return summary


def create_final_summary(
    output_dir: str,
    config: Optional[Dict] = None,
    metrics_summary: Optional[Dict] = None,
    rank_summary: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    创建最终汇总报告
    
    Args:
        output_dir: 输出目录
        config: 实验配置
        metrics_summary: 指标汇总
        rank_summary: Rank 汇总
        
    Returns:
        完整汇总
    """
    summary = {
        "timestamp": datetime.now().isoformat(),
        "output_dir": output_dir,
    }
    
    if config:
        summary["config"] = config
    
    if metrics_summary:
        summary["metrics"] = metrics_summary
    
    if rank_summary:
        summary["rank_pattern"] = rank_summary
    
    # 保存到文件
    output_path = Path(output_dir) / "final_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Final summary saved to {output_path}")
    
    return summary


def setup_experiment_logging(
    output_dir: str,
    log_level: int = logging.INFO,
) -> Dict[str, JSONLWriter]:
    """
    设置实验日志
    
    Args:
        output_dir: 输出目录
        log_level: 日志级别
        
    Returns:
        日志写入器字典
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 设置 Python logging
    log_file = output_path / "training.log"
    
    # 配置根 logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # 文件 handler
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(log_level)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    
    # 控制台 handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter("%(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    
    # 添加 handlers（避免重复）
    if not root_logger.handlers:
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized: output_dir={output_dir}")
    
    # 创建 JSONL 写入器
    writers = {
        "metrics": JSONLWriter(output_path / "metrics.jsonl"),
        "rank_pattern": JSONLWriter(output_path / "rank_pattern.jsonl"),
        "signal_scores": JSONLWriter(output_path / "signal_scores.jsonl"),
    }
    
    return writers


class MetricsLogger:
    """训练指标记录器"""
    
    def __init__(self, writer: JSONLWriter):
        self.writer = writer
    
    def log_train_step(self, step: int, loss: float, lr: float):
        """记录训练步"""
        self.writer.write({
            "type": "train_step",
            "step": step,
            "loss": loss,
            "learning_rate": lr,
        })
    
    def log_eval(self, step: int, epoch: float, metrics: Dict[str, float]):
        """记录评估结果"""
        record = {
            "type": "eval",
            "step": step,
            "epoch": epoch,
        }
        record.update(metrics)
        self.writer.write(record)


if __name__ == "__main__":
    # 测试
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # 测试 JSONL 写入
        writer = JSONLWriter(f"{tmpdir}/test.jsonl")
        writer.write({"step": 1, "loss": 0.5})
        writer.write({"step": 2, "loss": 0.4})
        writer.close()
        
        # 读取
        records = read_jsonl(f"{tmpdir}/test.jsonl")
        print("Records:", records)
        
        # 测试实验日志设置
        writers = setup_experiment_logging(f"{tmpdir}/exp1")
        writers["metrics"].write({"epoch": 1, "eval_accuracy": 0.85})
        
        for w in writers.values():
            w.close()
        
        print("Test passed!")
