from typing import Dict, List, Callable


class PromptBuilder:
    """
    Unified prompt builder supporting multiple templates.
    Designed to be training-friendly (SFT/LoRA) and inference-friendly.
    """
    
    # Template registry
    _templates: Dict[str, Callable] = {}
    
    def __init__(self, template_name: str = "arc_mcq_v1", few_shot: int = 0):
        """
        Initialize prompt builder.
        
        Args:
            template_name: Name of the template to use
            few_shot: Number of few-shot examples (0/1/2)
        """
        self.template_name = template_name
        self.few_shot = few_shot
        
        if template_name not in self._templates:
            raise ValueError(f"Unknown template: {template_name}. "
                           f"Available: {list(self._templates.keys())}")
    
    def build(self, question: str, option_labels: List[str], 
              option_texts: Dict[str, str]) -> str:
        """
        Build prompt from question and options.
        
        Args:
            question: The question text
            option_labels: List of option labels (e.g., ["A", "B", "C", "D"])
            option_texts: Dictionary mapping labels to option texts
            
        Returns:
            Formatted prompt string
        """
        template_func = self._templates[self.template_name]
        return template_func(question, option_labels, option_texts, self.few_shot)
    
    @classmethod
    def register_template(cls, name: str):
        """
        Decorator to register a new template.
        
        Args:
            name: Template name
        """
        def decorator(func: Callable):
            cls._templates[name] = func
            return func
        return decorator


@PromptBuilder.register_template("arc_mcq_v1")
def arc_mcq_v1_template(question: str, option_labels: List[str], 
                        option_texts: Dict[str, str], few_shot: int = 0) -> str:
    """
    ARC multiple-choice template v1.
    
    Design principles:
    - Clear instruction emphasizing single letter output
    - Private reasoning (think step-by-step but don't output it)
    - Training-friendly: only train on the single letter answer
    - Supports 4-5 options (A-D or A-E)
    
    Args:
        question: The question text
        option_labels: List of option labels
        option_texts: Dictionary mapping labels to option texts
        few_shot: Number of few-shot examples (currently not implemented)
        
    Returns:
        Formatted prompt
    """
    # Build allowed letters string (e.g., "A, B, C, or D")
    if len(option_labels) == 2:
        allowed_str = f"{option_labels[0]} or {option_labels[1]}"
    else:
        allowed_str = ", ".join(option_labels[:-1]) + f", or {option_labels[-1]}"
    
    # Instruction
    instruction = f"""You are a careful reasoner. Read the question and choose the single best answer from the options.
Think step-by-step privately, but do not reveal your reasoning.
Return only the letter of the correct option ({allowed_str})."""
    
    # Input section
    input_section = f"Question: {question}\nOptions:\n"
    for label in option_labels:
        input_section += f"{label}. {option_texts[label]}\n"
    
    # Output section
    output_section = "Answer:"
    
    # Combine all sections
    prompt = f"{instruction}\n\n{input_section}\n{output_section} "
    
    return prompt


@PromptBuilder.register_template("arc_mcq_v2")
def arc_mcq_v2_template(question: str, option_labels: List[str], 
                        option_texts: Dict[str, str], few_shot: int = 0) -> str:
    """
    Alternative ARC template (more concise).
    
    Args:
        question: The question text
        option_labels: List of option labels
        option_texts: Dictionary mapping labels to option texts
        few_shot: Number of few-shot examples
        
    Returns:
        Formatted prompt
    """
    # Build options string
    options_str = "\n".join([f"{label}. {option_texts[label]}" 
                             for label in option_labels])
    
    prompt = f"""Read the question carefully and select the best answer.

Question: {question}

{options_str}

Answer with only a single letter: """
    
    return prompt

