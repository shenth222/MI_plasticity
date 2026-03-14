"""
metric/pre_importance/test/conftest.py

测试公共 fixtures：小型模拟模型 + 假数据加载器。
所有 test_*.py 直接 from conftest import TinyClassifier, make_fake_dataloader 使用。

模型接口完全模拟 HuggingFace AutoModelForSequenceClassification：
  - forward(input_ids, attention_mask, labels) → SequenceClassifierOutput(loss, logits)
  - named_parameters() / named_modules() 符合标准 nn.Module 接口
"""

from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers.modeling_outputs import SequenceClassifierOutput


# ---------------------------------------------------------------------------
# 模拟模型（含嵌套子模块，模拟 Transformer 叶模块结构）
# ---------------------------------------------------------------------------

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

    参数：
        vocab_size  = 200
        hidden      = 32
        num_labels  = 3
        seq_len     用于生成假 DataLoader（非模型内置）
    """

    def __init__(
        self,
        vocab_size:  int = 200,
        hidden:      int = 32,
        num_labels:  int = 3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden)
        self.layer0    = TinyLayer(hidden)
        self.layer1    = TinyLayer(hidden)
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
            x = self.embedding(input_ids)          # [B, L, H]
        else:
            B = (attention_mask.shape[0]
                 if attention_mask is not None else 2)
            x = torch.zeros(
                B, 16, self.embedding.embedding_dim,
                device=self.classifier.weight.device,
            )

        x = self.layer0(x)
        x = self.layer1(x)
        x = x.mean(dim=1)                          # mean pooling → [B, H]
        logits = self.classifier(x)                # [B, num_labels]

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)


# ---------------------------------------------------------------------------
# 假数据加载器工厂
# ---------------------------------------------------------------------------

def make_fake_dataloader(
    batch_size:  int = 4,
    num_batches: int = 8,
    vocab_size:  int = 200,
    seq_len:     int = 16,
    num_labels:  int = 3,
) -> DataLoader:
    """
    生成一个完全随机的假 DataLoader，用于单元测试。
    返回 batch 格式与 data/glue.py 一致：{input_ids, attention_mask, labels}。
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
