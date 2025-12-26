"""
激活提取工具：对比微调前后模型各注意力头的激活情况
计算每个头 h 的激活强度：R_h = E_batch[||y_h||_2]
其中 y_h 是该 head 的输出张量
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Callable
import numpy as np
from dataclasses import dataclass
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os


@dataclass
class ActivationStats:
    """存储每个头的激活统计信息"""
    layer_idx: int
    head_idx: int
    R_h: float  # 平均激活强度
    std: float  # 标准差
    min_val: float
    max_val: float
    num_samples: int


class ActivationHook:
    """Hook 类用于提取注意力头的输出"""

    def __init__(self, layer_idx: int, head_idx: int, d_head: Optional[int] = None):
        self.layer_idx = layer_idx
        self.head_idx = head_idx
        self.d_head = d_head  # 每个头的维度，用于拆分拼接的输出
        self.activations = []
        self.hook_handle = None

    def __call__(self, module, input, output):
        """Hook 函数：提取该头的输出"""
        # output 可能是元组（如 (attn_output, attn_weights)）或单个张量
        if isinstance(output, tuple):
            # 通常第一个元素是 attention 输出
            output = output[0]

        # output 可能的形状：
        # - (batch, seq_len, num_heads, d_head) - 分离的头
        # - (batch, seq_len, hidden_size) - 拼接的头，需要拆分
        # - (batch, seq_len, d_head) - 单个头

        if output.dim() == 4:
            # 如果是 (batch, seq_len, num_heads, d_head)，提取特定头
            head_output = output[:, :, self.head_idx, :]
        elif output.dim() == 3:
            # 可能是拼接的头，需要根据 hidden_size 和 d_head 拆分
            batch_size, seq_len, hidden_size = output.shape
            if self.d_head is not None:
                # 如果知道 d_head，可以拆分拼接的头
                num_heads = hidden_size // self.d_head
                if num_heads > self.head_idx and hidden_size == num_heads * self.d_head:
                    # 拆分: (batch, seq_len, hidden_size) -> (batch, seq_len, num_heads, d_head)
                    head_output = output.view(batch_size, seq_len, num_heads, self.d_head)
                    head_output = head_output[:, :, self.head_idx, :]
                else:
                    # 如果无法拆分，假设这是单个头的输出
                    head_output = output
            else:
                # 如果不知道 d_head，假设这是单个头的输出
                head_output = output
        else:
            raise ValueError(f"Unexpected output shape: {output.shape}, expected 3D or 4D tensor")

        # 计算每个样本的 L2 范数: ||y_h||_2
        # shape: (batch, seq_len)
        l2_norms = torch.norm(head_output, dim=-1)
        # 对序列维度求平均，得到每个样本的激活强度
        # shape: (batch,)
        sample_activations = l2_norms.mean(dim=1)

        self.activations.append(sample_activations.detach().cpu())

    def register(self, model, hook_point: str):
        """注册 hook，支持嵌套路径（如 'transformer.h.0.attn'）"""
        # 支持点号分隔的嵌套路径
        parts = hook_point.split('.')
        module = model
        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            else:
                raise ValueError(f"Hook point {hook_point} not found in model (failed at {part})")

        if isinstance(module, nn.Module):
            self.hook_handle = module.register_forward_hook(self)
        else:
            raise ValueError(f"Hook point {hook_point} is not a Module (got {type(module)})")

    def remove(self):
        """移除 hook"""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

    def get_stats(self) -> ActivationStats:
        """计算统计信息"""
        if not self.activations:
            raise ValueError("No activations collected")

        all_activations = torch.cat(self.activations, dim=0)
        R_h = all_activations.mean().item()
        std = all_activations.std().item()
        min_val = all_activations.min().item()
        max_val = all_activations.max().item()

        return ActivationStats(
            layer_idx=self.layer_idx,
            head_idx=self.head_idx,
            R_h=R_h,
            std=std,
            min_val=min_val,
            max_val=max_val,
            num_samples=len(all_activations)
        )

    def clear(self):
        """清空收集的激活值"""
        self.activations = []


class ActivationExtractor:
    """激活提取器主类"""

    def __init__(
        self,
        model: nn.Module,
        model_name: str = "model",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        model_path: Optional[str] = None
    ):
        self.model = model
        self.model_name = model_name
        self.device = device
        self.model_path = model_path
        self.model.to(device)
        self.model.eval()
        self.hooks: List[ActivationHook] = []
        self.num_layers = None
        self.num_heads = None
        self._detect_model_structure()

    def _load_config_from_file(self, config_path: str) -> Optional[Dict]:
        """从 config.json 文件加载配置"""
        try:
            if os.path.isfile(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"警告: 无法从 {config_path} 加载 config: {e}")
        return None

    def _get_config_from_model(self) -> Optional[Dict]:
        """从模型对象获取 config"""
        # 尝试从模型的 config 属性获取（transformers 模型通常有）
        if hasattr(self.model, 'config'):
            config = self.model.config
            # 如果是 transformers 的配置对象，转换为字典
            if hasattr(config, 'to_dict'):
                try:
                    return config.to_dict()
                except:
                    pass
            elif isinstance(config, dict):
                return config
            else:
                # 尝试直接访问属性并构建字典
                try:
                    result = {}
                    if hasattr(config, 'num_hidden_layers'):
                        result['num_hidden_layers'] = config.num_hidden_layers
                    if hasattr(config, 'num_layers'):
                        result['num_layers'] = config.num_layers
                    if hasattr(config, 'num_attention_heads'):
                        result['num_attention_heads'] = config.num_attention_heads
                    if hasattr(config, 'num_heads'):
                        result['num_heads'] = config.num_heads
                    if result:  # 至少有一个值才返回
                        return result
                except Exception as e:
                    pass
        return None

    def _extract_structure_from_config(self, config: Dict) -> Tuple[Optional[int], Optional[int]]:
        """从 config 字典中提取层数和头数"""
        num_layers = None
        num_heads = None
        
        # 尝试不同的键名（不同模型可能使用不同的键）
        num_layers = config.get('num_hidden_layers') or config.get('num_layers') or config.get('n_layer')
        num_heads = config.get('num_attention_heads') or config.get('num_heads') or config.get('n_head')
        
        return num_layers, num_heads

    def _detect_model_structure(self):
        """检测模型的层数和头数：优先从 config 读取，否则自动检测"""
        # 1. 首先尝试从模型的 config 属性获取
        config = self._get_config_from_model()
        if config:
            num_layers, num_heads = self._extract_structure_from_config(config)
            if num_layers is not None and num_heads is not None:
                self.num_layers = num_layers
                self.num_heads = num_heads
                print(f"从模型 config 获取结构: {self.num_layers} 层, {self.num_heads} 头")
                return
        
        # 2. 尝试从 config.json 文件读取
        if self.model_path:
            config_path = os.path.join(self.model_path, 'config.json')
            config = self._load_config_from_file(config_path)
            if config:
                num_layers, num_heads = self._extract_structure_from_config(config)
                if num_layers is not None and num_heads is not None:
                    self.num_layers = num_layers
                    self.num_heads = num_heads
                    print(f"从 config.json 获取结构: {self.num_layers} 层, {self.num_heads} 头")
                    return
        
        # 3. 如果都没有，尝试自动检测模型结构
        print("尝试自动检测模型结构...")
        # GPT / GPT-NeoX 等
        if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            self.num_layers = len(self.model.transformer.h)
            if self.num_layers > 0:
                attn = self.model.transformer.h[0].attn
                if hasattr(attn, 'num_heads'):
                    self.num_heads = attn.num_heads
                elif hasattr(attn, 'num_attention_heads'):
                    self.num_heads = attn.num_attention_heads
        # 一些自定义 encoder-only 结构
        elif hasattr(self.model, 'layers'):
            self.num_layers = len(self.model.layers)
            if self.num_layers > 0:
                attn = getattr(self.model.layers[0], 'attn', None)
                if attn is not None:
                    if hasattr(attn, 'num_heads'):
                        self.num_heads = attn.num_heads
                    elif hasattr(attn, 'num_attention_heads'):
                        self.num_heads = attn.num_attention_heads
        # BERT / encoder-only transformers
        elif hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'layer'):
            self.num_layers = len(self.model.encoder.layer)
            if self.num_layers > 0:
                attn = self.model.encoder.layer[0].attention
                if hasattr(attn, 'num_attention_heads'):
                    self.num_heads = attn.num_attention_heads
                elif hasattr(attn, 'num_heads'):
                    self.num_heads = attn.num_heads
        # Llama / 其它 transformers.CausalLM 包一层 model 的结构
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            # 例如：LlamaForCausalLM.model.layers[i].self_attn
            self.num_layers = len(self.model.model.layers)
            if self.num_layers > 0:
                attn = getattr(self.model.model.layers[0], 'self_attn', None)
                if attn is not None:
                    if hasattr(attn, 'num_heads'):
                        self.num_heads = attn.num_heads
                    elif hasattr(attn, 'num_attention_heads'):
                        self.num_heads = attn.num_attention_heads

        if self.num_layers is not None and self.num_heads is not None:
            print(f"自动检测到模型结构: {self.num_layers} 层, {self.num_heads} 头")
        else:
            print(f"警告: 无法自动检测模型结构，请手动设置 num_layers 和 num_heads")

    def set_model_structure(self, num_layers: int, num_heads: int):
        """手动设置模型结构"""
        self.num_layers = num_layers
        self.num_heads = num_heads

    def _get_hook_point(self, layer_idx: int) -> str:
        """根据模型结构获取 hook 点"""
        if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            return f'transformer.h.{layer_idx}.attn'
        elif hasattr(self.model, 'layers'):
            return f'layers.{layer_idx}.attn'
        elif hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'layer'):
            return f'encoder.layer.{layer_idx}.attention'
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return f'model.layers.{layer_idx}.self_attn'
        else:
            raise ValueError("无法确定 hook 点，请手动指定")

    def register_hooks(
        self,
        layers: Optional[List[int]] = None,
        heads: Optional[List[int]] = None,
        d_head: Optional[int] = None
    ):
        """
        注册所有需要的 hooks

        Args:
            layers: 要 hook 的层索引列表，None 表示所有层
            heads: 要 hook 的头索引列表，None 表示所有头
            d_head: 每个头的维度，用于拆分拼接的输出（可选）
        """
        if self.num_layers is None or self.num_heads is None:
            raise ValueError("请先设置模型结构 (num_layers, num_heads)")

        layers = layers if layers is not None else list(range(self.num_layers))
        heads = heads if heads is not None else list(range(self.num_heads))

        # 如果未指定 d_head，尝试从模型推断
        if d_head is None:
            d_head = self._infer_d_head()

        for layer_idx in layers:
            hook_point = self._get_hook_point(layer_idx)
            for head_idx in heads:
                hook = ActivationHook(layer_idx, head_idx, d_head=d_head)
                # 注意：这里需要根据实际模型结构调整 hook 点
                # 可能需要 hook 到 attention 的输出或者 value 的输出
                try:
                    hook.register(self.model, hook_point)
                    self.hooks.append(hook)
                except Exception as e:
                    print(f"警告: 无法在层 {layer_idx} 头 {head_idx} 注册 hook: {e}")

    def _infer_d_head(self) -> Optional[int]:
        """尝试从模型推断每个头的维度"""
        try:
            # 1) GPT / GPT-NeoX 等
            if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
                if len(self.model.transformer.h) > 0:
                    attn = self.model.transformer.h[0].attn
                    if hasattr(attn, 'head_dim'):
                        return attn.head_dim
                    # 一些实现使用 embed_dim / num_heads
                    if hasattr(attn, 'embed_dim') and hasattr(attn, 'num_heads'):
                        return attn.embed_dim // attn.num_heads
            # 2) 直接暴露 layers 列表的结构
            if hasattr(self.model, 'layers') and len(self.model.layers) > 0:
                attn = getattr(self.model.layers[0], 'attn', None)
                if attn is not None:
                    if hasattr(attn, 'head_dim'):
                        return attn.head_dim
                    if hasattr(attn, 'embed_dim') and hasattr(attn, 'num_heads'):
                        return attn.embed_dim // attn.num_heads
            # 3) Llama / 其它 CausalLM 包装结构：model.layers[i].self_attn
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers') and len(self.model.model.layers) > 0:
                attn = getattr(self.model.model.layers[0], 'self_attn', None)
                if attn is not None:
                    # LlamaAttention 通常有 head_dim 属性
                    if hasattr(attn, 'head_dim'):
                        return attn.head_dim
                    # 兜底：根据 hidden_size / num_heads 推断
                    # 优先从 config 里拿，避免访问内部私有属性
                    config = self._get_config_from_model()
                    if config is not None:
                        hidden_size = config.get('hidden_size')
                        num_heads = config.get('num_attention_heads') or config.get('num_heads')
                        if hidden_size is not None and num_heads:
                            return hidden_size // num_heads
                    # 再从模块本身推断
                    if hasattr(attn, 'hidden_size') and hasattr(attn, 'num_heads'):
                        return attn.hidden_size // attn.num_heads
        except Exception:
            pass
        return None

    def extract_activations(
        self,
        dataloader: torch.utils.data.DataLoader,
        tokenizer: Optional[Callable] = None,
        max_batches: Optional[int] = None
    ):
        """
        从数据加载器中提取激活值

        Args:
            dataloader: 数据加载器
            tokenizer: 可选的 tokenizer 函数，如果数据是文本需要先 tokenize
            max_batches: 最大批次数，None 表示处理所有批次
        """
        # 清空之前的激活值
        for hook in self.hooks:
            hook.clear()

        batch_count = 0
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"提取 {self.model_name} 激活值"):
                if max_batches is not None and batch_count >= max_batches:
                    break

                # 处理输入
                # 数据加载器应该已经通过 collate_fn 返回 tokenized 的字典或 BatchEncoding
                inputs = batch
                
                # 检查是否是 BatchEncoding 对象（transformers 的 tokenization 结果）
                try:
                    from transformers.tokenization_utils_base import BatchEncoding
                    if isinstance(batch, BatchEncoding):
                        # BatchEncoding 可以像字典一样使用，转换为普通字典
                        inputs = dict(batch)
                    elif isinstance(batch, dict):
                        inputs = batch
                    elif isinstance(batch, (list, tuple)) and len(batch) > 0:
                        if isinstance(batch[0], str):
                            if tokenizer is not None:
                                inputs = tokenizer(
                                    batch,
                                    return_tensors="pt",
                                    padding=True,
                                    truncation=True,
                                    max_length=512
                                )
                                # 如果是 BatchEncoding，转换为字典
                                if isinstance(inputs, BatchEncoding):
                                    inputs = dict(inputs)
                            else:
                                print(f"警告: 数据是字符串列表但未提供 tokenizer，跳过该批次")
                                continue
                        else:
                            # 尝试作为张量处理
                            inputs = batch
                    else:
                        # 尝试使用 tokenizer
                        if tokenizer is not None:
                            inputs = tokenizer(
                                batch,
                                return_tensors="pt",
                                padding=True,
                                truncation=True,
                                max_length=512
                            )
                            # 如果是 BatchEncoding，转换为字典
                            if isinstance(inputs, BatchEncoding):
                                inputs = dict(inputs)
                        else:
                            print(f"警告: 无法处理输入类型 {type(batch)}，跳过该批次")
                            continue
                except ImportError:
                    # 如果没有 transformers，使用标准字典检查
                    if isinstance(batch, dict):
                        inputs = batch
                    elif isinstance(batch, (list, tuple)) and len(batch) > 0:
                        if isinstance(batch[0], str) and tokenizer is not None:
                            inputs = tokenizer(
                                batch,
                                return_tensors="pt",
                                padding=True,
                                truncation=True,
                                max_length=512
                            )
                            # 尝试转换为字典
                            if hasattr(inputs, 'keys'):
                                inputs = dict(inputs)
                        else:
                            print(f"警告: 无法处理输入类型 {type(batch)}，跳过该批次")
                            continue
                    else:
                        print(f"警告: 无法处理输入类型 {type(batch)}，跳过该批次")
                        continue

                # 确保输入在正确的设备上
                if isinstance(inputs, dict):
                    inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                             for k, v in inputs.items()}
                elif isinstance(inputs, torch.Tensor):
                    inputs = inputs.to(self.device)
                else:
                    print(f"警告: 输入格式不正确（期望 dict 或 Tensor，得到 {type(inputs)}），跳过该批次")
                    continue

                # 前向传播
                try:
                    if isinstance(inputs, dict):
                        _ = self.model(**inputs)
                    elif isinstance(inputs, torch.Tensor):
                        _ = self.model(inputs)
                    else:
                        print(f"警告: 输入格式不正确，跳过该批次")
                        continue
                except Exception as e:
                    print(f"警告: 前向传播出错: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

                batch_count += 1

    def compute_R_h(self) -> Dict[Tuple[int, int], float]:
        """
        计算每个头的 R_h = E_batch[||y_h||_2]

        Returns:
            Dict[(layer_idx, head_idx), R_h]
        """
        R_h_dict = {}
        for hook in self.hooks:
            stats = hook.get_stats()
            R_h_dict[(stats.layer_idx, stats.head_idx)] = stats.R_h
        return R_h_dict

    def get_all_stats(self) -> List[ActivationStats]:
        """获取所有头的统计信息"""
        return [hook.get_stats() for hook in self.hooks]

    def remove_hooks(self):
        """移除所有 hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def save_results(self, output_path: str):
        """保存结果到文件"""
        results = {
            'model_name': self.model_name,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'activations': {}
        }

        for hook in self.hooks:
            stats = hook.get_stats()
            key = f"layer_{stats.layer_idx}_head_{stats.head_idx}"
            results['activations'][key] = {
                'layer_idx': stats.layer_idx,
                'head_idx': stats.head_idx,
                'R_h': stats.R_h,
                'std': stats.std,
                'min': stats.min_val,
                'max': stats.max_val,
                'num_samples': stats.num_samples
            }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"结果已保存到: {output_path}")


class ActivationComparator:
    """对比微调前后模型的激活情况"""

    def __init__(
        self,
        pre_tuned_extractor: ActivationExtractor,
        fine_tuned_extractor: ActivationExtractor
    ):
        self.pre_tuned = pre_tuned_extractor
        self.fine_tuned = fine_tuned_extractor

    def compare(self) -> Dict:
        """
        对比两个模型的激活情况

        Returns:
            包含对比结果的字典
        """
        pre_R_h = self.pre_tuned.compute_R_h()
        fine_R_h = self.fine_tuned.compute_R_h()

        comparison = {
            'pre_tuned': pre_R_h,
            'fine_tuned': fine_R_h,
            'differences': {},
            'ratios': {}
        }

        # 计算差异和比率
        for key in pre_R_h:
            if key in fine_R_h:
                diff = fine_R_h[key] - pre_R_h[key]
                ratio = fine_R_h[key] / pre_R_h[key] if pre_R_h[key] > 0 else 0
                comparison['differences'][key] = diff
                comparison['ratios'][key] = ratio

        return comparison

    def visualize_comparison(
        self,
        output_path: str,
        layers: Optional[List[int]] = None,
        heads: Optional[List[int]] = None
    ):
        """
        可视化对比结果

        Args:
            output_path: 输出图片路径
            layers: 要可视化的层，None 表示所有层
            heads: 要可视化的头，None 表示所有头
        """
        comparison = self.compare()

        pre_R_h = comparison['pre_tuned']
        fine_R_h = comparison['fine_tuned']

        # 准备数据
        data = []
        for (layer, head) in pre_R_h:
            if layers is not None and layer not in layers:
                continue
            if heads is not None and head not in heads:
                continue
            if (layer, head) in fine_R_h:
                data.append({
                    'layer': layer,
                    'head': head,
                    'pre_tuned': pre_R_h[(layer, head)],
                    'fine_tuned': fine_R_h[(layer, head)]
                })

        if not data:
            print("No data available for visualization")
            return

        # 创建热力图数据
        num_layers = max(d['layer'] for d in data) + 1
        num_heads = max(d['head'] for d in data) + 1

        pre_matrix = np.zeros((num_layers, num_heads))
        fine_matrix = np.zeros((num_layers, num_heads))
        diff_matrix = np.zeros((num_layers, num_heads))

        for d in data:
            layer, head = d['layer'], d['head']
            pre_matrix[layer, head] = d['pre_tuned']
            fine_matrix[layer, head] = d['fine_tuned']
            diff_matrix[layer, head] = d['fine_tuned'] - d['pre_tuned']

        # 绘制对比图
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Pre-tuned
        sns.heatmap(pre_matrix, ax=axes[0], cmap='viridis', cbar=True)
        axes[0].set_title('Pre-tuned: R_h = E[||y_h||_2]')
        axes[0].set_xlabel('Head Index')
        axes[0].set_ylabel('Layer Index')

        # Fine-tuned
        sns.heatmap(fine_matrix, ax=axes[1], cmap='viridis', cbar=True)
        axes[1].set_title('Fine-tuned: R_h = E[||y_h||_2]')
        axes[1].set_xlabel('Head Index')
        axes[1].set_ylabel('Layer Index')

        # Difference
        sns.heatmap(diff_matrix, ax=axes[2], cmap='RdBu_r', center=0, cbar=True)
        axes[2].set_title('Difference (Fine-tuned - Pre-tuned)')
        axes[2].set_xlabel('Head Index')
        axes[2].set_ylabel('Layer Index')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")
        plt.close()

    def save_comparison(self, output_path: str):
        """保存对比结果到 JSON"""
        comparison = self.compare()

        # 转换为可序列化的格式
        serializable = {
            'pre_tuned': {f"L{l}_H{h}": v for (l, h), v in comparison['pre_tuned'].items()},
            'fine_tuned': {f"L{l}_H{h}": v for (l, h), v in comparison['fine_tuned'].items()},
            'differences': {f"L{l}_H{h}": v for (l, h), v in comparison['differences'].items()},
            'ratios': {f"L{l}_H{h}": v for (l, h), v in comparison['ratios'].items()}
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)

        print(f"对比结果已保存到: {output_path}")


def create_simple_dataloader(samples: List[Dict], tokenizer, batch_size: int = 8):
    """
    创建简单的数据加载器

    Args:
        samples: 样本列表，每个样本是包含 'input' 或 'text' 的字典
        tokenizer: tokenizer 对象（如 AutoTokenizer），用于将文本转换为 tokenized 输入
        batch_size: 批次大小
    """
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, samples, tokenizer):
            self.samples = samples
            self.tokenizer = tokenizer

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            sample = self.samples[idx]
            # 尝试不同的键名
            text = sample.get('input', sample.get('text', sample.get('question', '')))
            if isinstance(text, list):
                text = ' '.join(text)
            
            return text  # 返回原始文本，在 collate_fn 中统一处理

    def collate_fn(batch_texts):
        """自定义 collate 函数，对批次文本进行 tokenization"""
        if tokenizer is not None:
            # 使用 tokenizer 对批次文本进行 tokenization
            tokenized = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            # 将 BatchEncoding 转换为普通字典
            if hasattr(tokenized, 'keys'):
                # 如果是 BatchEncoding 或类似对象，转换为字典
                return dict(tokenized)
            else:
                return tokenized
        else:
            # 如果没有 tokenizer，返回原始文本列表
            return batch_texts

    dataset = SimpleDataset(samples, tokenizer)
    return torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=collate_fn
    )


if __name__ == "__main__":
    import argparse
    from transformers import AutoTokenizer, AutoModelForCausalLM

    parser = argparse.ArgumentParser(description="提取和对比模型激活值")
    parser.add_argument("--pre_tuned_model", type=str, required=True, help="微调前模型路径")
    parser.add_argument("--fine_tuned_model", type=str, required=True, help="微调后模型路径")
    parser.add_argument("--data_path", type=str, required=True, help="数据文件路径 (JSON)")
    parser.add_argument("--sample_num", type=int, default=100, help="使用的样本数量")
    parser.add_argument("--batch_size", type=int, default=8, help="批次大小")
    parser.add_argument("--output_dir", type=str, default="./activation_results", help="输出目录")
    parser.add_argument("--layers", type=int, nargs='+', default=None, help="要分析的层索引，None表示所有层")
    parser.add_argument("--heads", type=int, nargs='+', default=None, help="要分析的头索引，None表示所有头")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据
    from get_random_samples import get_random_samples
    samples = get_random_samples(args.data_path, args.sample_num, args.seed)
    print(f"加载了 {len(samples)} 个样本")

    # 加载模型和 tokenizer
    print("加载微调前模型...")
    pre_tokenizer = AutoTokenizer.from_pretrained(args.pre_tuned_model)
    pre_model = AutoModelForCausalLM.from_pretrained(args.pre_tuned_model)

    print("加载微调后模型...")
    fine_tokenizer = AutoTokenizer.from_pretrained(args.fine_tuned_model)
    fine_model = AutoModelForCausalLM.from_pretrained(args.fine_tuned_model)

    # 设置 padding token
    if pre_tokenizer.pad_token is None:
        pre_tokenizer.pad_token = pre_tokenizer.eos_token
    if fine_tokenizer.pad_token is None:
        fine_tokenizer.pad_token = fine_tokenizer.eos_token

    # 创建 tokenizer 函数
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

    # 创建数据加载器
    dataloader = create_simple_dataloader(samples, tokenize_fn, args.batch_size)

    # 创建激活提取器
    print("创建激活提取器...")
    pre_extractor = ActivationExtractor(pre_model, "pre_tuned", model_path=args.pre_tuned_model)
    fine_extractor = ActivationExtractor(fine_model, "fine_tuned", model_path=args.fine_tuned_model)

    # 设置模型结构（如果自动检测失败）
    # pre_extractor.set_model_structure(num_layers=12, num_heads=12)  # 根据实际模型调整

    # 注册 hooks
    print("注册 hooks...")
    pre_extractor.register_hooks(layers=args.layers, heads=args.heads)
    fine_extractor.register_hooks(layers=args.layers, heads=args.heads)

    # 提取激活值
    print("提取微调前模型激活值...")
    pre_extractor.extract_activations(dataloader, tokenizer=None)

    print("提取微调后模型激活值...")
    fine_extractor.extract_activations(dataloader, tokenizer=None)

    # 保存单独的结果
    print("保存结果...")
    pre_extractor.save_results(str(output_dir / "pre_tuned_activations.json"))
    fine_extractor.save_results(str(output_dir / "fine_tuned_activations.json"))

    # 对比分析
    print("进行对比分析...")
    comparator = ActivationComparator(pre_extractor, fine_extractor)
    comparator.save_comparison(str(output_dir / "comparison.json"))
    comparator.visualize_comparison(str(output_dir / "comparison_heatmap.png"))

    # 清理
    pre_extractor.remove_hooks()
    fine_extractor.remove_hooks()

    print("完成！")

