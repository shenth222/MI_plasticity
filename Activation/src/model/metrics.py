import numpy as np
import torch
from typing import Union, Tuple


class OnlineStats:
    """
    Online statistics calculator using Welford's algorithm.
    
    Efficiently computes mean and variance without storing all values.
    Supports multi-dimensional arrays (e.g., [num_layers, num_heads]).
    """
    
    def __init__(self, shape: Tuple[int, ...]):
        """
        Initialize online statistics.
        
        Args:
            shape: Shape of the statistics array (e.g., (num_layers, num_heads))
        """
        self.shape = shape
        self.count = np.zeros(shape, dtype=np.int64)
        self.mean = np.zeros(shape, dtype=np.float64)
        self.M2 = np.zeros(shape, dtype=np.float64)  # Sum of squared differences
    
    def update(self, values: Union[np.ndarray, torch.Tensor]) -> None:
        """
        Update statistics with new values using Welford's algorithm.
        
        Args:
            values: New values to add (same shape as self.shape)
        """
        # Convert to numpy if needed
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        
        values = np.asarray(values, dtype=np.float64)
        
        # Ensure shape matches
        if values.shape != self.shape:
            raise ValueError(f"Values shape {values.shape} doesn't match stats shape {self.shape}")
        
        # Welford's online algorithm
        self.count += 1
        delta = values - self.mean
        self.mean += delta / self.count
        delta2 = values - self.mean
        self.M2 += delta * delta2
    
    def update_batch(self, batch_values: Union[np.ndarray, torch.Tensor]) -> None:
        """
        Update statistics with a batch of values.
        
        Args:
            batch_values: Batch of values, shape (batch_size, *self.shape)
        """
        # Convert to numpy if needed
        if isinstance(batch_values, torch.Tensor):
            batch_values = batch_values.detach().cpu().numpy()
        
        batch_values = np.asarray(batch_values, dtype=np.float64)
        
        # Update for each item in batch
        for values in batch_values:
            self.update(values)
    
    def get_mean(self) -> np.ndarray:
        """Get current mean."""
        return self.mean.copy()
    
    def get_variance(self) -> np.ndarray:
        """Get current variance."""
        variance = np.zeros_like(self.M2)
        mask = self.count > 1
        variance[mask] = self.M2[mask] / (self.count[mask] - 1)
        return variance
    
    def get_std(self) -> np.ndarray:
        """Get current standard deviation."""
        return np.sqrt(self.get_variance())
    
    def get_count(self) -> np.ndarray:
        """Get current count."""
        return self.count.copy()
    
    def reset(self) -> None:
        """Reset all statistics."""
        self.count = np.zeros(self.shape, dtype=np.int64)
        self.mean = np.zeros(self.shape, dtype=np.float64)
        self.M2 = np.zeros(self.shape, dtype=np.float64)
    
    def get_summary(self) -> dict:
        """
        Get summary of all statistics.
        
        Returns:
            Dictionary with mean, variance, std, count
        """
        return {
            "mean": self.get_mean(),
            "variance": self.get_variance(),
            "std": self.get_std(),
            "count": self.get_count()
        }

