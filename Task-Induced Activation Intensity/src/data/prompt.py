"""
Prompt 模板模块
"""
from typing import Dict, List


def create_prompt(
    question: str,
    choices: List[str],
    choice_labels: List[str],
    template: str = None
) -> str:
    """
    根据 question 和 choices 创建 prompt
    
    Args:
        question: 问题文本
        choices: 选项文本列表
        choice_labels: 选项标签列表（如 ["A", "B", "C", "D"]）
        template: 自定义模板（可选）
        
    Returns:
        格式化后的 prompt 字符串
    """
    # 默认模板
    if template is None:
        template = (
            "Question: {question}\n"
            "Choices:\n"
            "{choices_text}"
            "Answer:"
        )
    
    # 构建 choices 文本
    choices_lines = []
    for label, choice_text in zip(choice_labels, choices):
        choices_lines.append(f"{label}. {choice_text}")
    
    choices_text = "\n".join(choices_lines) + "\n"
    
    # 如果模板中有具体的 choice_A, choice_B 等占位符
    if "{choice_A}" in template or "{choice_B}" in template:
        # 构建 choices 字典
        choices_dict = {}
        for label, choice_text in zip(choice_labels, choices):
            choices_dict[f"choice_{label}"] = choice_text
        
        # 格式化
        try:
            prompt = template.format(
                question=question,
                **choices_dict
            )
        except KeyError:
            # 如果某些占位符缺失，回退到简单模板
            prompt = template.format(
                question=question,
                choices_text=choices_text
            )
    else:
        # 使用 choices_text
        prompt = template.format(
            question=question,
            choices_text=choices_text
        )
    
    return prompt


def batch_create_prompts(
    data_batch: List[Dict],
    template: str = None
) -> List[str]:
    """
    批量创建 prompts
    
    Args:
        data_batch: 数据批次，每个元素包含 question, choices, choice_labels
        template: 自定义模板（可选）
        
    Returns:
        prompt 列表
    """
    prompts = []
    
    for item in data_batch:
        prompt = create_prompt(
            question=item["question"],
            choices=item["choices"],
            choice_labels=item["choice_labels"],
            template=template
        )
        prompts.append(prompt)
    
    return prompts

