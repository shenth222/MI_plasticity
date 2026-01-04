"""
日志工具模块
"""
import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "main",
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    设置 logger
    
    Args:
        name: logger 名称
        log_file: 日志文件路径（可选）
        level: 日志级别
        
    Returns:
        配置好的 logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 清除已有的 handlers
    logger.handlers.clear()
    
    # 格式化器
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # 控制台 handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件 handler（如果提供了日志文件路径）
    if log_file:
        # 确保目录存在
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "main") -> logging.Logger:
    """获取已配置的 logger"""
    return logging.getLogger(name)

