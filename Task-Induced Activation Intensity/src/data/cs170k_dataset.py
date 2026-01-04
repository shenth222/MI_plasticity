"""
CS170k 数据集加载模块
"""
from typing import List, Dict, Any, Optional
from torch.utils.data import Dataset
from ..utils.io import load_data_file


def get_nested_field(data: Dict, field_path: str) -> Any:
    """
    获取嵌套字段的值
    
    Args:
        data: 数据字典
        field_path: 字段路径，如 "choices.text"
        
    Returns:
        字段值
    """
    fields = field_path.split(".")
    value = data
    
    for field in fields:
        if isinstance(value, dict) and field in value:
            value = value[field]
        else:
            return None
    
    return value


class CS170kDataset(Dataset):
    """
    CS170k Commonsense 数据集
    
    支持自定义字段映射以适配不同的数据格式
    """
    
    def __init__(
        self,
        data_path: str,
        max_samples: Optional[int] = None,
        field_mapping: Optional[Dict[str, str]] = None
    ):
        """
        Args:
            data_path: 数据文件路径（JSON 或 JSONL）
            max_samples: 最大样本数（用于调试）
            field_mapping: 字段映射字典，例如：
                {
                    "question": "question",
                    "choices_text": "choices.text",
                    "choices_label": "choices.label",
                    "answer_key": "answerKey"
                }
        """
        self.data_path = data_path
        self.field_mapping = field_mapping or {
            "question": "question",
            "choices_text": "choices.text",
            "choices_label": "choices.label",
            "answer_key": "answerKey"
        }
        
        # 加载数据
        self.raw_data = load_data_file(data_path)
        
        # 限制样本数
        if max_samples is not None and max_samples < len(self.raw_data):
            self.raw_data = self.raw_data[:max_samples]
        
        # 解析数据
        self.data = []
        for idx, item in enumerate(self.raw_data):
            parsed_item = self._parse_item(item, idx)
            if parsed_item is not None:
                self.data.append(parsed_item)
    
    def _parse_item(self, item: Dict, idx: int) -> Optional[Dict]:
        """
        解析单个数据项
        
        Args:
            item: 原始数据项
            idx: 索引
            
        Returns:
            解析后的数据项，格式：
            {
                "question": str,
                "choices": List[str],  # 选项文本列表
                "choice_labels": List[str],  # 选项标签列表（如 ["A", "B", "C", "D"]）
                "answer": str  # 正确答案标签（如 "A"）
            }
        """
        try:
            # 提取字段
            question = get_nested_field(item, self.field_mapping["question"])
            choices_text = get_nested_field(item, self.field_mapping["choices_text"])
            choices_label = get_nested_field(item, self.field_mapping.get("choices_label", "choices.label"))
            answer_key = get_nested_field(item, self.field_mapping["answer_key"])
            
            # 验证必需字段
            if question is None or choices_text is None:
                return None
            
            # 处理 choices（可能是列表或字典）
            if isinstance(choices_text, list):
                choices = choices_text
            elif isinstance(choices_text, dict):
                # 如果是字典，按字母顺序排序键
                choices = [choices_text[k] for k in sorted(choices_text.keys())]
            else:
                return None
            
            # 处理 choice_labels
            if choices_label is None:
                # 如果没有提供 labels，自动生成 A, B, C, D, ...
                choice_labels = [chr(65 + i) for i in range(len(choices))]
            elif isinstance(choices_label, list):
                choice_labels = choices_label
            elif isinstance(choices_label, dict):
                choice_labels = [choices_label[k] for k in sorted(choices_label.keys())]
            else:
                choice_labels = [chr(65 + i) for i in range(len(choices))]
            
            # 处理 answer_key
            if answer_key is None:
                answer = choice_labels[0]  # 默认第一个选项
            else:
                answer = str(answer_key).strip()
            
            return {
                "question": str(question).strip(),
                "choices": [str(c).strip() for c in choices],
                "choice_labels": choice_labels,
                "answer": answer
            }
        
        except Exception as e:
            # 如果解析失败，跳过该样本
            return None
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        return self.data[idx]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取数据集统计信息"""
        return {
            "total_samples": len(self.data),
            "avg_question_length": sum(len(item["question"]) for item in self.data) / len(self.data) if self.data else 0,
            "num_choices_range": (
                min(len(item["choices"]) for item in self.data) if self.data else 0,
                max(len(item["choices"]) for item in self.data) if self.data else 0
            )
        }

