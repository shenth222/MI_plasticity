"""
训练 Callbacks
负责调用 AdaLoRA 的 update_and_allocate 和记录 rank 分配
"""

import logging
from typing import Dict, Optional
from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments
import torch
from signal import SignalTracker
from patch_adalora import set_external_scores
from logging_utils import JSONLWriter

logger = logging.getLogger(__name__)


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
        
        self.last_step = -1
    
    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: torch.nn.Module = None,
        **kwargs,
    ):
        """
        在每个训练步后调用
        
        注意：需要区分 optimizer step 和 gradient accumulation step
        """
        # 检查是否真正进行了 optimizer step
        # state.global_step 在 optimizer step 后才会增加
        if state.global_step == self.last_step:
            # 还在 gradient accumulation，不调用 update_and_allocate
            return
        
        self.last_step = state.global_step
        global_step = state.global_step
        
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
        
        # 2. 调用 AdaLoRA 的 update_and_allocate
        if hasattr(model, 'base_model') and hasattr(model.base_model, 'peft_config'):
            # 这是一个 PEFT model
            peft_model = model
            
            # 获取 AdaLoRA config
            config_name = list(peft_model.peft_config.keys())[0]
            peft_config = peft_model.peft_config[config_name]
            
            if hasattr(peft_config, 'rank_allocator') and peft_config.rank_allocator is not None:
                # 调用 update_and_allocate
                try:
                    peft_config.rank_allocator.update_and_allocate(peft_model, global_step)
                    
                    # 记录 rank 分配
                    if self.rank_logger and global_step % self.log_rank_every == 0:
                        rank_info = self._get_rank_info(peft_model)
                        
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
    
    def _get_rank_info(self, model: torch.nn.Module) -> Dict:
        """获取当前的 rank 分配信息"""
        ranks = {}
        total_rank = 0
        num_modules = 0
        
        for name, module in model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                # 获取当前 rank
                lora_A = module.lora_A['default']
                current_rank = lora_A.weight.shape[0]
                
                # 清理名称
                clean_name = name
                if "base_model.model." in clean_name:
                    clean_name = clean_name.replace("base_model.model.", "")
                
                ranks[clean_name] = current_rank
                total_rank += current_rank
                num_modules += 1
        
        return {
            "ranks": ranks,
            "total_rank": total_rank,
            "num_modules": num_modules,
        }
    
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
            final_rank_info = self._get_rank_info(model)
            
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
    
    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: torch.nn.Module = None,
        **kwargs,
    ):
        """检查 budget"""
        if model is None:
            return
        
        # 计算当前 total rank
        total_rank = 0
        for name, module in model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                lora_A = module.lora_A['default']
                total_rank += lora_A.weight.shape[0]
        
        self.budget_history.append((state.global_step, total_rank))
        
        # 检查是否超出目标
        if self.target_budget is not None:
            deviation = abs(total_rank - self.target_budget) / self.target_budget
            if deviation > self.tolerance:
                logger.warning(
                    f"⚠️  Budget deviation at step {state.global_step}: "
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
    
    from signal import SignalTracker
    
    tracker = SignalTracker(signal_type="importance_only")
    callback = AdaLoRACallback(
        signal_tracker=tracker,
        log_rank_every=10,
        log_signal_every=10,
    )
    
    print("Callback initialized:", callback)
