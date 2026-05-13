"""Core utilities for evolutionary algorithms project."""

import random
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    logger.info(f"Random seed set to {seed}")


def get_device() -> str:
    """Get available compute device with fallback.
    
    Returns:
        Device string ('cuda', 'mps', or 'cpu')
    """
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            logger.info("Using CUDA device")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
            logger.info("Using MPS device (Apple Silicon)")
        else:
            device = "cpu"
            logger.info("Using CPU device")
    except ImportError:
        device = "cpu"
        logger.info("PyTorch not available, using CPU")
    
    return device


class Config:
    """Configuration management for evolutionary algorithms."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize configuration.
        
        Args:
            config_dict: Configuration parameters
        """
        for key, value in config_dict.items():
            setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {key: value for key, value in self.__dict__.items() 
                if not key.startswith('_')}
    
    def update(self, **kwargs) -> None:
        """Update configuration parameters."""
        for key, value in kwargs.items():
            setattr(self, key, value)
