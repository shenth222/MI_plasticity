import argparse
import yaml
from pathlib import Path
from typing import Any, Dict


class Config:
    """Configuration manager for the project."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initialize config from dictionary.
        
        Args:
            config_dict: Configuration dictionary
        """
        self._config = config_dict
        
    def __getattr__(self, name: str) -> Any:
        """Allow attribute-style access."""
        if name.startswith('_'):
            return object.__getattribute__(self, name)
        return self._config.get(name)
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access."""
        return self._config[key]
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get config value with default."""
        return self._config.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self._config.copy()
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """
        Load config from YAML file.
        
        Args:
            yaml_path: Path to YAML config file
            
        Returns:
            Config instance
        """
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(config_dict)
    
    @classmethod
    def from_args_and_yaml(cls, yaml_path: str = None) -> 'Config':
        """
        Load config from YAML and override with command line arguments.
        
        Args:
            yaml_path: Path to YAML config file (optional)
            
        Returns:
            Config instance
        """
        parser = argparse.ArgumentParser(description="Collect attention head activations")
        
        # Config file
        parser.add_argument("--config", type=str, default=yaml_path,
                          help="Path to config YAML file")
        
        # Model
        parser.add_argument("--model_path", type=str, help="Path to model")
        parser.add_argument("--dtype", type=str, choices=["bf16", "fp16", "fp32"],
                          help="Model dtype")
        parser.add_argument("--device_map", type=str, help="Device map strategy")
        
        # Data
        parser.add_argument("--data_dir", type=str, help="Path to dataset directory")
        parser.add_argument("--max_samples", type=int, help="Maximum samples to process")
        parser.add_argument("--batch_size", type=int, help="Batch size")
        parser.add_argument("--max_length", type=int, help="Maximum sequence length")
        
        # Prompt
        parser.add_argument("--template_name", type=str, help="Prompt template name")
        parser.add_argument("--few_shot", type=int, choices=[0, 1, 2],
                          help="Number of few-shot examples")
        
        # Collection
        parser.add_argument("--token_agg", type=str, choices=["last", "all"],
                          help="Token aggregation strategy")
        
        # Output
        parser.add_argument("--output_dir", type=str, help="Output directory")
        parser.add_argument("--save_every", type=int, help="Save intermediate results every N steps")
        
        # Experiment
        parser.add_argument("--seed", type=int, help="Random seed")
        parser.add_argument("--experiment_name", type=str, help="Experiment name")
        
        args = parser.parse_args()
        
        # Load from YAML
        if args.config:
            config_dict = cls.from_yaml(args.config).to_dict()
        else:
            config_dict = {}
        
        # Override with command line arguments
        for key, value in vars(args).items():
            if value is not None and key != 'config':
                config_dict[key] = value
        
        return cls(config_dict)

