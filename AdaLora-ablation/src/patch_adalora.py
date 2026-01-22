"""
AdaLoRA Patching 模块
通过 monkeypatch 替换 RankAllocator 的 scoring signal
"""

import logging
from typing import Dict, Optional
import peft
from packaging import version

logger = logging.getLogger(__name__)

# 全局变量：存储外部提供的 scores
_EXTERNAL_SCORES: Optional[Dict[str, float]] = None


def check_peft_version():
    """检查 PEFT 版本是否兼容"""
    required_version = "0.18.1"
    current_version = peft.__version__
    
    if version.parse(current_version) != version.parse(required_version):
        logger.warning(
            f"PEFT version mismatch: required {required_version}, got {current_version}. "
            f"Patching may fail!"
        )
        return False
    
    logger.info(f"PEFT version check passed: {current_version}")
    return True


def set_external_scores(scores: Dict[str, float]):
    """设置外部 scores（供 RankAllocator 使用）"""
    global _EXTERNAL_SCORES
    _EXTERNAL_SCORES = scores
    logger.debug(f"External scores updated: {len(scores)} modules")


def get_external_scores() -> Optional[Dict[str, float]]:
    """获取外部 scores"""
    return _EXTERNAL_SCORES


def patch_rank_allocator(use_external_scores: bool = True):
    """
    Monkeypatch PEFT 的 RankAllocator
    
    Args:
        use_external_scores: 是否使用外部提供的 scores
    """
    if not check_peft_version():
        logger.warning("Patching with incompatible PEFT version!")
    
    try:
        from peft.tuners.adalora import RankAllocator
    except ImportError:
        logger.error("Cannot import RankAllocator from peft.tuners.adalora")
        raise
    
    # 保存原始方法
    original_update_and_allocate = RankAllocator.update_and_allocate
    
    def patched_update_and_allocate(self, model, global_step):
        """
        替换后的 update_and_allocate 方法
        
        关键修改：在计算 importance 后，用外部 scores 替换
        """
        # 调用原始方法的前半部分：更新 EMA 统计量
        # 注意：这里我们完全调用原始方法，然后替换 ipt
        
        if not use_external_scores or _EXTERNAL_SCORES is None:
            # 不使用外部 scores，直接调用原始方法
            return original_update_and_allocate(self, model, global_step)
        
        # === 开始 patch 逻辑 ===
        
        # 1. 检查是否在调整窗口内
        if global_step < self.tinit or global_step > self.tfinal:
            return
        
        if (global_step - self.tinit) % self.deltaT != 0:
            return
        
        logger.info(f"[AdaLoRA Patched] Step {global_step}: Using external scores")
        
        # 2. 调用原始方法来更新 EMA（但会被我们的 scores 覆盖）
        # 为了避免重复调用，我们手动实现核心逻辑
        
        # 获取所有 AdaLoRA layers
        peft_model = model
        if hasattr(model, "base_model"):
            peft_model = model.base_model
        
        # 3. 收集当前的 ranks
        for name, module in peft_model.named_modules():
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                # 更新 rankallocator 的统计量（调用原始逻辑）
                # 但我们会在下一步替换 importance
                pass
        
        # 4. 替换 importance scores
        # RankAllocator 内部维护一个 ipt (importance) 字典
        if hasattr(self, "ipt"):
            # 将外部 scores 映射到 RankAllocator 的 ipt 格式
            for module_name, score in _EXTERNAL_SCORES.items():
                # 需要匹配 RankAllocator 内部的 key 格式
                # 通常是完整的 module path
                for key in self.ipt.keys():
                    if module_name in key or key.endswith(module_name):
                        self.ipt[key] = score
                        break
        
        # 5. 调用原始的 mask_to_budget 和 rank 调整逻辑
        # 这部分我们不修改，直接调用
        self.mask_to_budget(model, global_step)
        
        logger.debug(f"[AdaLoRA Patched] Rank allocation updated at step {global_step}")
    
    # 应用 patch
    RankAllocator.update_and_allocate = patched_update_and_allocate
    
    logger.info("RankAllocator patched successfully!")


def patch_rank_allocator_v2(use_external_scores: bool = True):
    """
    V2: 更干净的 patch 方法
    
    直接在 compute_importance 层面替换
    """
    if not check_peft_version():
        logger.warning("Patching with incompatible PEFT version!")
    
    try:
        from peft.tuners.adalora import RankAllocator
    except ImportError:
        logger.error("Cannot import RankAllocator from peft.tuners.adalora")
        raise
    
    # 保存原始方法
    if hasattr(RankAllocator, '_original_compute_importance'):
        logger.info("RankAllocator already patched, skipping...")
        return
    
    original_compute_importance = None
    
    # 检查是否有 _compute_importance 方法
    if hasattr(RankAllocator, '_compute_importance'):
        original_compute_importance = RankAllocator._compute_importance
    
    def patched_compute_importance(self, module_name, module):
        """替换 importance 计算"""
        if not use_external_scores or _EXTERNAL_SCORES is None:
            # 使用原始逻辑
            if original_compute_importance:
                return original_compute_importance(self, module_name, module)
            else:
                # Fallback: 返回默认值
                return 1.0
        
        # 使用外部 scores
        for ext_name, score in _EXTERNAL_SCORES.items():
            if ext_name in module_name or module_name.endswith(ext_name):
                return score
        
        # 如果找不到匹配，使用默认值
        logger.warning(f"No external score found for {module_name}, using default 1.0")
        return 1.0
    
    # 应用 patch
    if original_compute_importance:
        RankAllocator._original_compute_importance = original_compute_importance
        RankAllocator._compute_importance = patched_compute_importance
        logger.info("RankAllocator._compute_importance patched!")
    else:
        logger.warning("Cannot find _compute_importance method, trying alternative patch...")


def patch_rank_allocator_simple():
    """
    最简单的 patch：直接替换 update_and_allocate
    
    这个版本直接在 update_and_allocate 中注入外部 scores
    """
    try:
        from peft.tuners.adalora import RankAllocator
    except ImportError:
        logger.error("Cannot import RankAllocator")
        raise
    
    # 保存原始方法
    if not hasattr(RankAllocator, '_original_update_and_allocate'):
        RankAllocator._original_update_and_allocate = RankAllocator.update_and_allocate
    else:
        logger.info("Already patched, skipping...")
        return
    
    def patched_update_and_allocate(self, model, global_step):
        """注入外部 scores 的版本"""
        # 先调用原始方法（更新内部统计量）
        self._original_update_and_allocate(model, global_step)
        
        # 如果有外部 scores，替换 ipt
        if _EXTERNAL_SCORES is not None and hasattr(self, 'ipt'):
            logger.debug(f"[Patch] Injecting external scores at step {global_step}")
            
            # 遍历所有 ipt keys，尝试匹配外部 scores
            for ipt_key in list(self.ipt.keys()):
                matched = False
                for ext_name, score in _EXTERNAL_SCORES.items():
                    # 匹配逻辑：ext_name 在 ipt_key 中
                    if ext_name in ipt_key:
                        self.ipt[ipt_key] = score
                        matched = True
                        break
                
                if not matched:
                    # 尝试反向匹配
                    for ext_name, score in _EXTERNAL_SCORES.items():
                        if ipt_key.endswith(ext_name):
                            self.ipt[ipt_key] = score
                            break
    
    # 应用 patch
    RankAllocator.update_and_allocate = patched_update_and_allocate
    logger.info("✓ RankAllocator patched (simple version)")


def unpatch_rank_allocator():
    """恢复原始的 RankAllocator"""
    try:
        from peft.tuners.adalora import RankAllocator
    except ImportError:
        return
    
    if hasattr(RankAllocator, '_original_update_and_allocate'):
        RankAllocator.update_and_allocate = RankAllocator._original_update_and_allocate
        delattr(RankAllocator, '_original_update_and_allocate')
        logger.info("RankAllocator unpatched")
    
    if hasattr(RankAllocator, '_original_compute_importance'):
        RankAllocator._compute_importance = RankAllocator._original_compute_importance
        delattr(RankAllocator, '_original_compute_importance')


def apply_patch(signal_type: str):
    """
    根据 signal_type 应用 patch
    
    Args:
        signal_type: baseline_adalora 不 patch，其他类型需要 patch
    """
    if signal_type == "baseline_adalora":
        logger.info("Using baseline AdaLoRA (no patching)")
        return
    
    logger.info(f"Applying AdaLoRA patch for signal_type={signal_type}")
    check_peft_version()
    patch_rank_allocator_simple()


if __name__ == "__main__":
    # 测试 patch
    logging.basicConfig(level=logging.INFO)
    
    check_peft_version()
    
    # 设置测试 scores
    test_scores = {
        "layer.0.query_proj": 0.8,
        "layer.0.key_proj": 0.6,
        "layer.1.query_proj": 0.9,
    }
    
    set_external_scores(test_scores)
    print("External scores:", get_external_scores())
    
    # 应用 patch
    apply_patch("importance_only")
    print("Patch applied!")
