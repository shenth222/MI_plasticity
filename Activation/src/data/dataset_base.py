from abc import ABC, abstractmethod
from typing import Dict, Any
from torch.utils.data import Dataset


class DatasetBase(Dataset, ABC):
    """
    Base class for all datasets.
    Provides a unified interface for data loading and formatting.
    """
    
    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing:
                - prompt_text: str, the formatted prompt
                - answer_letter: str, the correct answer (A/B/C/D/E)
                - option_labels: List[str], list of option labels
                - target_text: str, the target text for training (same as answer_letter)
                - meta: Dict, metadata (id, source, etc.)
        """
        pass
    
    @abstractmethod
    def format_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format a raw example into the standard format.
        
        Args:
            example: Raw example from the dataset
            
        Returns:
            Formatted example dictionary
        """
        pass

