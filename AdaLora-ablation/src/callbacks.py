"""
训练 Callbacks
负责调用 AdaLoRA 的 update_and_allocate 和记录 rank 分配
"""

import logging
from typing import Dict, Optional
from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments
import torch
from signal_tracker import SignalTracker
from patch_adalora import set_external_scores
from logging_utils import JSONLWriter

logger = logging.getLogger(__name__)


def _clean_module_name(name: str) -> str:
    """清理模块名（去除 PEFT 前缀等）"""
    if "base_model.model." in name:
        name = name.replace("base_model.model.", "")
    return name


def _clean_rank_pattern_key(key: str, adapter_name: Optional[str]) -> str:
    """将 rank_pattern 的参数名清理为模块名"""
    clean = key
    if adapter_name:
        clean = clean.replace(f".lora_E.{adapter_name}", "")
        clean = clean.replace(f".lora_A.{adapter_name}", "")
        clean = clean.replace(f".lora_B.{adapter_name}", "")
    if ".lora_E." in clean:
        clean = clean.split(".lora_E.")[0]
    if ".lora_A." in clean:
        clean = clean.split(".lora_A.")[0]
    if ".lora_B." in clean:
        clean = clean.split(".lora_B.")[0]
    return _clean_module_name(clean)


def _rank_pattern_to_module_ranks(rank_pattern: Dict, adapter_name: Optional[str]) -> Dict[str, int]:
    """将 rank_pattern 转成 module -> active rank"""
    ranks = {}
    for key, mask in rank_pattern.items():
        if mask is None:
            continue
        active = sum(1 for v in mask if bool(v))
        clean_name = _clean_rank_pattern_key(key, adapter_name)
        ranks[clean_name] = active
    return ranks


def _count_lora_modules(model: torch.nn.Module) -> int:
    """统计 LoRA 模块数量"""
    count = 0
    for _, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            count += 1
    return count


def _get_peft_config(model: torch.nn.Module):
    """获取当前 PEFT config 与 adapter 名称"""
    if hasattr(model, "peft_config") and model.peft_config:
        adapter_name = list(model.peft_config.keys())[0]
        return model.peft_config[adapter_name], adapter_name
    return None, None


def _get_rank_info(
    model: torch.nn.Module,
    rank_pattern: Optional[Dict] = None,
    adapter_name: Optional[str] = None,
    init_r: Optional[int] = None,
) -> Dict:
    """获取当前的 rank 分配信息"""
    if rank_pattern:
        ranks = _rank_pattern_to_module_ranks(rank_pattern, adapter_name)
        total_rank = sum(ranks.values())
        return {
            "ranks": ranks,
            "total_rank": total_rank,
            "num_modules": len(ranks),
        }
    
    # fallback: 使用当前 LoRA 维度作为近似
    ranks = {}
    for name, module in model.named_modules():
        if not (hasattr(module, "lora_A") and hasattr(module, "lora_B")):
            continue
        try:
            if adapter_name and adapter_name in module.lora_A:
                lora_A = module.lora_A[adapter_name]
            else:
                lora_A = next(iter(module.lora_A.values()))
            current_rank = lora_A.weight.shape[0]
        except Exception:
            current_rank = init_r or 0
        
        clean_name = _clean_module_name(name)
        ranks[clean_name] = current_rank
    
    total_rank = sum(ranks.values())
    return {
        "ranks": ranks,
        "total_rank": total_rank,
        "num_modules": len(ranks),
    }


class MetricsWriterCallback(TrainerCallback):
    """将 Trainer logs 写入 JSONL"""
    
    def __init__(self, metrics_logger: Optional[JSONLWriter] = None):
        super().__init__()
        self.metrics_logger = metrics_logger
    
    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict] = None,
        **kwargs,
    ):
        if self.metrics_logger is None or not logs:
            return control
        
        record = {"step": state.global_step}
        record.update(logs)
        self.metrics_logger.write(record)
        return control


class AdaLoRACallback(TrainerCallback):
    """
    AdaLoRA 专用 callback
    
    功能：
    1. 在每个 optimizer step 后调用 update_and_allocate
    2. 更新 signal tracker
    3. 记录 rank 分配和 signal scores
    """
    
    def __init__(
        self,
        signal_tracker: Optional[SignalTracker] = None,
        rank_logger: Optional[JSONLWriter] = None,
        signal_logger: Optional[JSONLWriter] = None,
        log_rank_every: int = 10,
        log_signal_every: int = 10,
        use_external_scores: bool = True,
    ):
        """
        Args:
            signal_tracker: Signal tracking 对象
            rank_logger: Rank 分配日志写入器
            signal_logger: Signal scores 日志写入器
            log_rank_every: 每隔多少步记录 rank
            log_signal_every: 每隔多少步记录 signal
            use_external_scores: 是否使用外部 scores（非 baseline）
        """
        super().__init__()
        self.signal_tracker = signal_tracker
        self.rank_logger = rank_logger
        self.signal_logger = signal_logger
        self.log_rank_every = log_rank_every
        self.log_signal_every = log_signal_every
        self.use_external_scores = use_external_scores
    
    def on_pre_optimizer_step(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: torch.nn.Module = None,
        **kwargs,
    ):
        """
        在 optimizer step 之前调用（梯度仍存在）
        """
        if model is None:
            return control
        
        global_step = state.global_step + 1
        
        # 1. 更新 signal tracker（如果使用外部 scores）
        if self.use_external_scores and self.signal_tracker is not None:
            self.signal_tracker.update(model)
            scores = self.signal_tracker.get_scores()
            
            # 设置外部 scores（供 patch 使用）
            if scores:
                set_external_scores(scores)
            
            # 记录 signal scores
            if self.signal_logger and global_step % self.log_signal_every == 0:
                self.signal_logger.write({
                    "step": global_step,
                    "scores": scores,
                    "statistics": self.signal_tracker.get_statistics(),
                })
        return control

    def on_optimizer_step(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: torch.nn.Module = None,
        **kwargs,
    ):
        """
        在 optimizer step 之后调用（权重已更新）
        """
        if model is None:
            return control
        
        global_step = state.global_step + 1
        
        # 2. 调用 AdaLoRA 的 update_and_allocate
        if hasattr(model, "base_model") and hasattr(model.base_model, "update_and_allocate"):
            try:
                model.base_model.update_and_allocate(global_step)
                
                peft_config, adapter_name = _get_peft_config(model)
                rank_pattern = getattr(peft_config, "rank_pattern", None) if peft_config else None
                init_r = getattr(peft_config, "init_r", None) if peft_config else None
                
                # 记录 rank 分配
                if self.rank_logger and global_step % self.log_rank_every == 0:
                    rank_info = _get_rank_info(
                        model,
                        rank_pattern=rank_pattern,
                        adapter_name=adapter_name,
                        init_r=init_r,
                    )
                    
                    self.rank_logger.write({
                        "step": global_step,
                        "ranks": rank_info["ranks"],
                        "total_rank": rank_info["total_rank"],
                        "num_modules": rank_info["num_modules"],
                    })
                    
                    # 打印摘要
                    logger.info(
                        f"[AdaLoRA Update] Step {global_step}: "
                        f"Total rank={rank_info['total_rank']}, "
                        f"Active modules={rank_info['num_modules']}"
                    )
            
            except Exception as e:
                logger.error(f"Error in update_and_allocate at step {global_step}: {e}")
        
        return control
    
    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: torch.nn.Module = None,
        **kwargs,
    ):
        """训练结束时记录最终状态"""
        if model is not None:
            peft_config, adapter_name = _get_peft_config(model)
            rank_pattern = getattr(peft_config, "rank_pattern", None) if peft_config else None
            init_r = getattr(peft_config, "init_r", None) if peft_config else None
            
            final_rank_info = _get_rank_info(
                model,
                rank_pattern=rank_pattern,
                adapter_name=adapter_name,
                init_r=init_r,
            )
            
            logger.info("=" * 50)
            logger.info("Final Rank Allocation:")
            logger.info(f"  Total rank: {final_rank_info['total_rank']}")
            logger.info(f"  Active modules: {final_rank_info['num_modules']}")
            logger.info(f"  Average rank: {final_rank_info['total_rank'] / max(1, final_rank_info['num_modules']):.2f}")
            logger.info("=" * 50)
            
            # 记录到日志
            if self.rank_logger:
                self.rank_logger.write({
                    "step": state.global_step,
                    "ranks": final_rank_info["ranks"],
                    "total_rank": final_rank_info["total_rank"],
                    "num_modules": final_rank_info["num_modules"],
                    "is_final": True,
                })
        
        return control


class BudgetConsistencyCallback(TrainerCallback):
    """
    Budget 一致性检查 callback
    
    确保不同 signal 下的总 budget（total rank）基本一致
    """
    
    def __init__(
        self,
        target_budget: Optional[int] = None,
        tolerance: float = 0.05,
    ):
        """
        Args:
            target_budget: 目标 budget（可选）
            tolerance: 允许的误差比例
        """
        super().__init__()
        self.target_budget = target_budget
        self.tolerance = tolerance
        self.budget_history = []
    
    def on_optimizer_step(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: torch.nn.Module = None,
        **kwargs,
    ):
        """检查 budget"""
        if model is None:
            return control
        
        global_step = state.global_step + 1
        
        peft_config, adapter_name = _get_peft_config(model)
        rank_pattern = getattr(peft_config, "rank_pattern", None) if peft_config else None
        init_r = getattr(peft_config, "init_r", None) if peft_config else None
        rank_info = _get_rank_info(
            model,
            rank_pattern=rank_pattern,
            adapter_name=adapter_name,
            init_r=init_r,
        )
        total_rank = rank_info["total_rank"]
        
        # 初始化目标 budget
        if self.target_budget is None and peft_config is not None:
            num_modules = rank_info["num_modules"] or _count_lora_modules(model)
            if num_modules > 0:
                self.target_budget = peft_config.target_r * num_modules
        
        self.budget_history.append((global_step, total_rank))
        
        # 检查是否超出目标
        if self.target_budget is not None:
            deviation = abs(total_rank - self.target_budget) / self.target_budget
            if deviation > self.tolerance:
                logger.warning(
                    f"⚠️  Budget deviation at step {global_step}: "
                    f"current={total_rank}, target={self.target_budget}, "
                    f"deviation={deviation:.2%}"
                )
        
        return control
    
    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """训练结束时报告 budget 统计"""
        if not self.budget_history:
            return
        
        budgets = [b for _, b in self.budget_history]
        min_budget = min(budgets)
        max_budget = max(budgets)
        final_budget = budgets[-1]
        
        logger.info("=" * 50)
        logger.info("Budget Consistency Report:")
        logger.info(f"  Min budget: {min_budget}")
        logger.info(f"  Max budget: {max_budget}")
        logger.info(f"  Final budget: {final_budget}")
        if self.target_budget:
            logger.info(f"  Target budget: {self.target_budget}")
            logger.info(f"  Final deviation: {abs(final_budget - self.target_budget) / self.target_budget:.2%}")
        logger.info("=" * 50)
        
        return control


if __name__ == "__main__":
    # 测试
    logging.basicConfig(level=logging.INFO)
    
    from signal_tracker import SignalTracker
    
    tracker = SignalTracker(signal_type="importance_only")
    callback = AdaLoRACallback(
        signal_tracker=tracker,
        log_rank_every=10,
        log_signal_every=10,
    )
    
    print("Callback initialized:", callback)
