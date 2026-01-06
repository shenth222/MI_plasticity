import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from .dataset_base import DatasetBase
from .prompt import PromptBuilder
from ..utils.logging import get_logger


logger = get_logger(__name__)


class ARCDataset(DatasetBase):
    """
    ARC-Challenge dataset loader.
    
    Supports:
    - Loading from local jsonl files
    - Loading from HuggingFace datasets (with local cache)
    - 4 or 5 options (A-D or A-E)
    - Robust answer key mapping (letter or number)
    """
    
    def __init__(self, 
                 data_dir: str,
                 template_name: str = "arc_mcq_v1",
                 few_shot: int = 0,
                 max_samples: int = -1,
                 split: str = "test"):
        """
        Initialize ARC dataset.
        
        Args:
            data_dir: Path to dataset directory (jsonl files or HF cache)
            template_name: Prompt template name
            few_shot: Number of few-shot examples
            max_samples: Maximum number of samples (-1 for all)
            split: Dataset split (train/validation/test)
        """
        self.data_dir = data_dir
        self.template_name = template_name
        self.few_shot = few_shot
        self.max_samples = max_samples
        self.split = split
        
        # Initialize prompt builder
        self.prompt_builder = PromptBuilder(template_name, few_shot)
        
        # Load data
        self.examples = self._load_data()
        
        # Statistics
        self.num_4opt = 0
        self.num_5opt = 0
        self.num_skipped = 0
        
        logger.info(f"Loaded {len(self.examples)} examples from ARC-Challenge ({split})")
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """
        Load data from directory.
        
        Returns:
            List of raw examples
        """
        # Try loading from jsonl first
        jsonl_path = Path(self.data_dir) / f"{self.split}.jsonl"
        if jsonl_path.exists():
            return self._load_from_jsonl(jsonl_path)
        
        # Try alternative naming
        jsonl_path = Path(self.data_dir) / f"ARC-Challenge-{self.split}.jsonl"
        if jsonl_path.exists():
            return self._load_from_jsonl(jsonl_path)
        
        # Try loading from HuggingFace datasets
        try:
            return self._load_from_hf()
        except Exception as e:
            logger.error(f"Failed to load from HuggingFace: {e}")
            raise FileNotFoundError(
                f"Could not find data in {self.data_dir}. "
                f"Expected jsonl file or HuggingFace dataset cache."
            )
    
    def _load_from_jsonl(self, filepath: Path) -> List[Dict[str, Any]]:
        """Load data from jsonl file."""
        logger.info(f"Loading from jsonl: {filepath}")
        examples = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))
                    if self.max_samples > 0 and len(examples) >= self.max_samples:
                        break
        return examples
    
    def _load_from_hf(self) -> List[Dict[str, Any]]:
        """Load data from HuggingFace datasets."""
        logger.info(f"Loading from HuggingFace datasets with cache_dir={self.data_dir}")
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets library is required. Install with: pip install datasets")
        
        dataset = load_dataset(
            "ai2_arc",
            "ARC-Challenge",
            split=self.split,
            cache_dir=self.data_dir
        )
        
        examples = []
        for i, item in enumerate(dataset):
            examples.append(item)
            if self.max_samples > 0 and len(examples) >= self.max_samples:
                break
        
        return examples
    
    def __len__(self) -> int:
        """Return number of valid examples."""
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        """Get a single formatted example."""
        raw_example = self.examples[idx]
        return self.format_example(raw_example)
    
    def format_example(self, example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Format a raw ARC example.
        
        Args:
            example: Raw example with fields:
                - id: question id
                - question: question text
                - choices: dict with 'text' (list) and 'label' (list)
                - answerKey: correct answer (letter or number)
                
        Returns:
            Formatted example or None if invalid
        """
        # Extract fields
        question_id = example.get("id", "unknown")
        question = example.get("question", "")
        choices = example.get("choices", {})
        answer_key = example.get("answerKey", "")
        
        # Parse choices
        if isinstance(choices, dict):
            choice_texts = choices.get("text", [])
            choice_labels = choices.get("label", [])
        else:
            logger.warning(f"Invalid choices format for {question_id}")
            self.num_skipped += 1
            return None
        
        # Validate number of options (4 or 5)
        num_options = len(choice_texts)
        if num_options < 4 or num_options > 5:
            logger.warning(f"Skipping {question_id}: {num_options} options (expected 4-5)")
            self.num_skipped += 1
            return None
        
        # Count option statistics
        if num_options == 4:
            self.num_4opt += 1
        elif num_options == 5:
            self.num_5opt += 1
        
        # Standardize to A/B/C/D/E labels
        standard_labels = ["A", "B", "C", "D", "E"][:num_options]
        
        # Map original labels to standard labels
        label_map = {}
        for i, orig_label in enumerate(choice_labels):
            label_map[orig_label] = standard_labels[i]
        
        # Create option_texts dictionary
        option_texts = {}
        for i, text in enumerate(choice_texts):
            option_texts[standard_labels[i]] = text
        
        # Map answer_key to standard label
        answer_letter = self._map_answer_key(answer_key, label_map, num_options, question_id)
        if answer_letter is None:
            self.num_skipped += 1
            return None
        
        # Build prompt
        prompt_text = self.prompt_builder.build(
            question=question,
            option_labels=standard_labels,
            option_texts=option_texts
        )
        
        return {
            "prompt_text": prompt_text,
            "answer_letter": answer_letter,
            "option_labels": standard_labels,
            "target_text": answer_letter,  # For future training
            "meta": {
                "id": question_id,
                "source": "arc_challenge",
                "question": question,
                "num_options": num_options,
                "original_answer_key": answer_key
            }
        }
    
    def _map_answer_key(self, answer_key: str, label_map: Dict[str, str], 
                       num_options: int, question_id: str) -> Optional[str]:
        """
        Map answer key to standard label (A/B/C/D/E).
        
        Handles:
        - Letter keys: "A", "B", etc. -> direct or via label_map
        - Number keys: "1", "2", etc. -> map to A/B/C/D/E
        
        Args:
            answer_key: Original answer key
            label_map: Mapping from original labels to standard labels
            num_options: Number of options
            question_id: Question ID (for logging)
            
        Returns:
            Standard answer letter or None if invalid
        """
        # If answer_key is in label_map (original labels)
        if answer_key in label_map:
            return label_map[answer_key]
        
        # If answer_key is already a standard letter
        standard_labels = ["A", "B", "C", "D", "E"][:num_options]
        if answer_key in standard_labels:
            return answer_key
        
        # If answer_key is a number string (1-based)
        if answer_key.isdigit():
            idx = int(answer_key) - 1  # Convert to 0-based
            if 0 <= idx < num_options:
                return standard_labels[idx]
            else:
                logger.warning(f"Invalid answer index {answer_key} for {question_id}")
                return None
        
        logger.warning(f"Could not map answer_key '{answer_key}' for {question_id}")
        return None
    
    def get_statistics(self) -> Dict[str, int]:
        """Get dataset statistics."""
        return {
            "total_examples": len(self.examples),
            "num_4opt": self.num_4opt,
            "num_5opt": self.num_5opt,
            "num_skipped": self.num_skipped
        }

