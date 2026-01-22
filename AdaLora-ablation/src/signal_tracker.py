"""
Scoring Signal 计算模块
实现 importance, plasticity 和 combo signals 的在线计算
"""

import torch
import numpy as np
from typing import Dict, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class SignalTracker:
    """
    在线跟踪和计算 scoring signals
    
    支持的 signal types:
    - importance: EMA(|w · grad|)
    - plasticity: EMA(||grad||₂)
    - combo: zscore(importance) + λ * zscore(plasticity)
    """
    
    def __init__(
        self,
        signal_type: str = "importance_only",
        ema_decay: float = 0.9,
        combo_lambda: float = 1.0,
        normalize_method: str = "zscore",
    ):
        """
        Args:
            signal_type: 信号类型
            ema_decay: EMA 衰减系数
            combo_lambda: combo signal 的 plasticity 权重
            normalize_method: 归一化方法（zscore / minmax / none）
        """
        self.signal_type = signal_type
        self.ema_decay = ema_decay
        self.combo_lambda = combo_lambda
        self.normalize_method = normalize_method
        
        # 存储每个 module 的 EMA 统计量
        self.importance_ema = {}  # module_name -> scalar
        self.plasticity_ema = {}  # module_name -> scalar
        
        # 记录更新次数（用于 bias correction）
        self.update_count = defaultdict(int)
        
        # 缓存最新的 scores
        self.latest_scores = {}
        
        logger.info(f"Initialized SignalTracker: type={signal_type}, decay={ema_decay}, lambda={combo_lambda}")
    
    def update(self, model: torch.nn.Module):
        """
        更新所有 LoRA module 的 signal
        
        Args:
            model: 包含 LoRA adapter 的模型
        """
        # 获取所有 LoRA modules
        lora_modules = self._get_lora_modules(model)
        
        if not lora_modules:
            logger.warning("No LoRA modules found!")
            return
        
        # 对每个 module 计算 signal
        for module_name, module in lora_modules.items():
            # 计算 importance 和 plasticity
            importance = self._compute_importance(module)
            plasticity = self._compute_plasticity(module)
            
            # 更新 EMA
            self.update_count[module_name] += 1
            
            if module_name not in self.importance_ema:
                # 初始化
                self.importance_ema[module_name] = importance
                self.plasticity_ema[module_name] = plasticity
            else:
                # EMA 更新
                self.importance_ema[module_name] = (
                    self.ema_decay * self.importance_ema[module_name] 
                    + (1 - self.ema_decay) * importance
                )
                self.plasticity_ema[module_name] = (
                    self.ema_decay * self.plasticity_ema[module_name] 
                    + (1 - self.ema_decay) * plasticity
                )
        
        # 计算最终 scores
        self.latest_scores = self._compute_scores()
    
    def _get_lora_modules(self, model: torch.nn.Module) -> Dict[str, torch.nn.Module]:
        """获取所有 LoRA modules"""
        lora_modules = {}
        
        for name, module in model.named_modules():
            # 检查是否是 LoRA module（包含 lora_A 和 lora_B）
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                # 清理模块名（移除前缀）
                clean_name = name
                if "base_model.model." in clean_name:
                    clean_name = clean_name.replace("base_model.model.", "")
                
                lora_modules[clean_name] = module
        
        return lora_modules
    
    def _compute_importance(self, module: torch.nn.Module) -> float:
        """
        计算 importance: |w · grad|
        
        对于 LoRA: importance ≈ |A · B · grad|
        简化为: ||A||_F · ||B||_F · ||grad||
        """
        if not hasattr(module, 'lora_A') or not hasattr(module, 'lora_B'):
            return 0.0
        
        # 获取 LoRA 参数
        lora_A = module.lora_A['default'].weight  # [r, in_dim]
        lora_B = module.lora_B['default'].weight  # [out_dim, r]
        
        # 计算梯度范数
        grad_norm_A = 0.0
        grad_norm_B = 0.0
        
        if lora_A.grad is not None:
            grad_norm_A = torch.norm(lora_A.grad, p='fro').item()
        
        if lora_B.grad is not None:
            grad_norm_B = torch.norm(lora_B.grad, p='fro').item()
        
        # 计算参数范数
        param_norm_A = torch.norm(lora_A, p='fro').item()
        param_norm_B = torch.norm(lora_B, p='fro').item()
        
        # importance ≈ ||W||_F · ||grad||_F
        # 对于 LoRA: W = B @ A
        importance = (param_norm_A * param_norm_B) * (grad_norm_A + grad_norm_B)
        
        return importance
    
    def _compute_plasticity(self, module: torch.nn.Module) -> float:
        """
        计算 plasticity: ||grad||₂
        
        对于 LoRA: 计算 A 和 B 的梯度范数之和
        """
        if not hasattr(module, 'lora_A') or not hasattr(module, 'lora_B'):
            return 0.0
        
        lora_A = module.lora_A['default'].weight
        lora_B = module.lora_B['default'].weight
        
        grad_norm = 0.0
        
        if lora_A.grad is not None:
            grad_norm += torch.norm(lora_A.grad, p=2).item()
        
        if lora_B.grad is not None:
            grad_norm += torch.norm(lora_B.grad, p=2).item()
        
        return grad_norm
    
    def _compute_scores(self) -> Dict[str, float]:
        """
        根据 signal_type 计算最终 scores
        
        Returns:
            module_name -> score
        """
        if self.signal_type == "importance_only":
            return self.importance_ema.copy()
        
        elif self.signal_type == "plasticity_only":
            return self.plasticity_ema.copy()
        
        elif self.signal_type == "combo":
            # 归一化后组合
            imp_normalized = self._normalize(self.importance_ema)
            pla_normalized = self._normalize(self.plasticity_ema)
            
            combo_scores = {}
            for name in imp_normalized:
                combo_scores[name] = (
                    imp_normalized[name] + self.combo_lambda * pla_normalized[name]
                )
            
            return combo_scores
        
        else:
            raise ValueError(f"Unknown signal_type: {self.signal_type}")
    
    def _normalize(self, scores: Dict[str, float]) -> Dict[str, float]:
        """归一化 scores"""
        if not scores:
            return {}
        
        values = np.array(list(scores.values()))
        
        if self.normalize_method == "zscore":
            mean = values.mean()
            std = values.std()
            if std < 1e-8:
                std = 1.0
            normalized_values = (values - mean) / std
        
        elif self.normalize_method == "minmax":
            min_val = values.min()
            max_val = values.max()
            if max_val - min_val < 1e-8:
                normalized_values = np.zeros_like(values)
            else:
                normalized_values = (values - min_val) / (max_val - min_val)
        
        elif self.normalize_method == "none":
            normalized_values = values
        
        else:
            raise ValueError(f"Unknown normalize_method: {self.normalize_method}")
        
        return {name: val for name, val in zip(scores.keys(), normalized_values)}
    
    def get_scores(self) -> Dict[str, float]:
        """获取最新的 scores"""
        return self.latest_scores.copy()
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            "signal_type": self.signal_type,
            "num_modules": len(self.latest_scores),
            "importance_ema": self.importance_ema.copy(),
            "plasticity_ema": self.plasticity_ema.copy(),
            "latest_scores": self.latest_scores.copy(),
            "update_count": dict(self.update_count),
        }
    
    def reset(self):
        """重置所有统计量"""
        self.importance_ema.clear()
        self.plasticity_ema.clear()
        self.update_count.clear()
        self.latest_scores.clear()
        logger.info("SignalTracker reset")


def aggregate_scores_by_module(scores: Dict[str, float]) -> Dict[str, float]:
    """
    将详细的 module path 聚合到 module-level
    
    例如：
    'deberta.encoder.layer.0.attention.self.query_proj' -> 'layer.0.query_proj'
    
    Args:
        scores: 详细的 scores
        
    Returns:
        聚合后的 scores
    """
    aggregated = {}
    
    for full_name, score in scores.items():
        # 提取关键部分
        parts = full_name.split('.')
        
        # 找到 layer.X 和 最后的模块名
        layer_idx = None
        module_type = parts[-1] if parts else full_name
        
        for i, part in enumerate(parts):
            if part == "layer" and i + 1 < len(parts):
                layer_idx = parts[i + 1]
                break
        
        # 构建聚合名称
        if layer_idx is not None:
            agg_name = f"layer.{layer_idx}.{module_type}"
        else:
            agg_name = module_type
        
        # 如果有重复，取平均（通常不会有）
        if agg_name in aggregated:
            aggregated[agg_name] = (aggregated[agg_name] + score) / 2
        else:
            aggregated[agg_name] = score
    
    return aggregated


if __name__ == "__main__":
    # 测试
    tracker = SignalTracker(signal_type="combo", ema_decay=0.9, combo_lambda=1.0)
    
    # 模拟更新
    class DummyModule:
        def __init__(self):
            self.lora_A = {'default': type('obj', (object,), {'weight': torch.randn(8, 768, requires_grad=True)})}
            self.lora_B = {'default': type('obj', (object,), {'weight': torch.randn(768, 8, requires_grad=True)})}
            
            # 设置假梯度
            self.lora_A['default'].weight.grad = torch.randn_like(self.lora_A['default'].weight)
            self.lora_B['default'].weight.grad = torch.randn_like(self.lora_B['default'].weight)
    
    class DummyModel:
        def named_modules(self):
            return [
                ("layer.0.query_proj", DummyModule()),
                ("layer.0.key_proj", DummyModule()),
            ]
    
    model = DummyModel()
    tracker.update(model)
    
    print("Scores:", tracker.get_scores())
    print("Statistics:", tracker.get_statistics())
