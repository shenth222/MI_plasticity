"""
metric/pre_importance/test/conftest.py

测试公共 fixtures：小型模拟模型 + 假数据加载器。

包含两类模型：
  TinyClassifier   — 简单 2 层 Transformer，无 HF config（用于基本功能测试）
  TinyHFClassifier — 带 model.config 的 HF-style 模型，命名严格对齐 DeBERTa，
                     用于头级别粒度（head_granularity）测试
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers.modeling_outputs import SequenceClassifierOutput


# ============================================================================
# 基础模型（无 config，用于原有各指标测试）
# ============================================================================

class TinyAttention(nn.Module):
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.query = nn.Linear(hidden, hidden)
        self.key   = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, hidden)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attn = torch.softmax(q @ k.transpose(-1, -2) / q.size(-1) ** 0.5, dim=-1)
        return attn @ v


class TinyFFN(nn.Module):
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden * 2)
        self.fc2 = nn.Linear(hidden * 2, hidden)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class TinyLayer(nn.Module):
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.attention = TinyAttention(hidden)
        self.ffn       = TinyFFN(hidden)
        self.norm      = nn.LayerNorm(hidden)

    def forward(self, x):
        return self.norm(x + self.ffn(self.attention(x)))


class TinyClassifier(nn.Module):
    """
    2 层 Tiny Transformer 模拟分类器，接口与 HuggingFace 一致。
    无 model.config，不支持头级别测试。
    """

    def __init__(self, vocab_size: int = 200, hidden: int = 32, num_labels: int = 3):
        super().__init__()
        self.embedding  = nn.Embedding(vocab_size, hidden)
        self.layer0     = TinyLayer(hidden)
        self.layer1     = TinyLayer(hidden)
        self.classifier = nn.Linear(hidden, num_labels)
        self.num_labels = num_labels

    def forward(
        self,
        input_ids:      Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels:         Optional[torch.Tensor] = None,
        **kwargs,
    ) -> SequenceClassifierOutput:
        if input_ids is not None:
            x = self.embedding(input_ids)
        else:
            B = attention_mask.shape[0] if attention_mask is not None else 2
            x = torch.zeros(B, 16, self.embedding.embedding_dim,
                            device=self.classifier.weight.device)
        x = self.layer0(x)
        x = self.layer1(x)
        x = x.mean(dim=1)
        logits = self.classifier(x)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return SequenceClassifierOutput(loss=loss, logits=logits)


# ============================================================================
# HF-style 模型（带 config，命名对齐 DeBERTa，用于头级别测试）
# ============================================================================

@dataclass
class TinyConfig:
    """
    模拟 HuggingFace 模型配置。
    hidden_size 必须能整除 num_attention_heads。
    """
    hidden_size:           int = 8
    num_attention_heads:   int = 2    # head_dim = 8 // 2 = 4
    num_labels:            int = 3
    vocab_size:            int = 200


class TinyAttnSelf(nn.Module):
    """
    模拟 DeBERTa 的 attention.self 子模块：
    包含 query_proj / key_proj / value_proj（命名严格对齐）。
    """
    def __init__(self, cfg: TinyConfig):
        super().__init__()
        h = cfg.hidden_size
        self.query_proj = nn.Linear(h, h)
        self.key_proj   = nn.Linear(h, h)
        self.value_proj = nn.Linear(h, h)

    def forward(self, x):
        q = self.query_proj(x)
        k = self.key_proj(x)
        v = self.value_proj(x)
        scores = torch.softmax(q @ k.transpose(-1, -2) / (q.size(-1) ** 0.5), dim=-1)
        return scores @ v


class TinyAttnOutput(nn.Module):
    """
    模拟 DeBERTa 的 attention.output 子模块：
    包含 dense（命名严格对齐，为输出投影层）。
    """
    def __init__(self, cfg: TinyConfig):
        super().__init__()
        self.dense = nn.Linear(cfg.hidden_size, cfg.hidden_size)

    def forward(self, x):
        return self.dense(x)


class TinyAttn(nn.Module):
    """模拟 DeBERTa 完整 attention 模块：包含 self 和 output 子模块。"""
    def __init__(self, cfg: TinyConfig):
        super().__init__()
        self.self   = TinyAttnSelf(cfg)
        self.output = TinyAttnOutput(cfg)

    def forward(self, x):
        return self.output(self.self(x))


class TinyEncoderLayer(nn.Module):
    """模拟 DeBERTa encoder 的单层：attention + FFN。"""
    def __init__(self, cfg: TinyConfig):
        super().__init__()
        h = cfg.hidden_size
        self.attention    = TinyAttn(cfg)
        self.intermediate = nn.Linear(h, h * 2)
        self.output_ffn   = nn.Linear(h * 2, h)
        self.norm         = nn.LayerNorm(h)

    def forward(self, x):
        a = self.attention(x)
        f = torch.relu(self.intermediate(a))
        return self.norm(a + self.output_ffn(f))


class TinyEncoder(nn.Module):
    """模拟 DeBERTa encoder：包含 layer.0 和 layer.1 两层。"""
    def __init__(self, cfg: TinyConfig):
        super().__init__()
        self.layer = nn.ModuleList([TinyEncoderLayer(cfg) for _ in range(2)])

    def forward(self, x):
        for layer in self.layer:
            x = layer(x)
        return x


class TinyDeberta(nn.Module):
    """模拟 DeBERTa base 模型：embeddings + encoder。"""
    def __init__(self, cfg: TinyConfig):
        super().__init__()
        self.embeddings = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.encoder    = TinyEncoder(cfg)

    def forward(self, input_ids):
        return self.encoder(self.embeddings(input_ids))


class TinyHFClassifier(nn.Module):
    """
    模拟 HuggingFace AutoModelForSequenceClassification 的 DeBERTa-v3-base 版本。

    关键特性：
      1. 具有 model.config（包含 num_attention_heads、hidden_size）
      2. 命名严格对齐 DeBERTa：
           deberta.encoder.layer.{i}.attention.self.query_proj
           deberta.encoder.layer.{i}.attention.self.key_proj
           deberta.encoder.layer.{i}.attention.self.value_proj
           deberta.encoder.layer.{i}.attention.output.dense
      3. forward() 接口与 HF 一致，返回 SequenceClassifierOutput

    默认配置：hidden=8, num_heads=2, head_dim=4，2 层 encoder。
    """

    def __init__(self, cfg: Optional[TinyConfig] = None):
        super().__init__()
        self.config  = cfg or TinyConfig()
        self.deberta = TinyDeberta(self.config)
        self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels)
        self.num_labels = self.config.num_labels

    def forward(
        self,
        input_ids:      Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels:         Optional[torch.Tensor] = None,
        **kwargs,
    ) -> SequenceClassifierOutput:
        if input_ids is not None:
            x = self.deberta(input_ids)            # [B, L, H]
        else:
            B = attention_mask.shape[0] if attention_mask is not None else 2
            x = torch.zeros(B, 16, self.config.hidden_size,
                            device=self.classifier.weight.device)
        x = x.mean(dim=1)                          # [B, H]
        logits = self.classifier(x)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return SequenceClassifierOutput(loss=loss, logits=logits)


# ============================================================================
# 假数据加载器工厂
# ============================================================================

def make_fake_dataloader(
    batch_size:  int = 4,
    num_batches: int = 8,
    vocab_size:  int = 200,
    seq_len:     int = 16,
    num_labels:  int = 3,
) -> DataLoader:
    """
    生成完全随机的假 DataLoader，用于单元测试。
    batch 格式：{input_ids, attention_mask, labels}。
    """
    n = batch_size * num_batches
    input_ids      = torch.randint(0, vocab_size, (n, seq_len))
    attention_mask = torch.ones(n, seq_len, dtype=torch.long)
    labels         = torch.randint(0, num_labels, (n,))

    dataset = TensorDataset(input_ids, attention_mask, labels)

    def collate(batch):
        ids, masks, lbls = zip(*batch)
        return {
            "input_ids":      torch.stack(ids),
            "attention_mask": torch.stack(masks),
            "labels":         torch.stack(lbls),
        }

    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate)
