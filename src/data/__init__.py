"""Data loading and preprocessing utilities."""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Load and preprocess datasets for hyperparameter optimization."""
    
    AVAILABLE_DATASETS = {
        'iris': load_iris,
        'wine': load_wine,
        'breast_cancer': load_breast_cancer
    }
    
    def __init__(self, dataset_name: str = 'iris', test_size: float = 0.2, 
                 random_state: int = 42):
        """Initialize dataset loader.
        
        Args:
            dataset_name: Name of dataset to load
            test_size: Fraction of data for testing
            random_state: Random seed for reproducibility
        """
        self.dataset_name = dataset_name
        self.test_size = test_size
        self.random_state = random_state
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Load and split dataset.
        
        Returns:
            Tuple of (X_train, y_train, X_test, y_test)
        """
        if self.dataset_name not in self.AVAILABLE_DATASETS:
            raise ValueError(f"Dataset {self.dataset_name} not available. "
                           f"Choose from: {list(self.AVAILABLE_DATASETS.keys())}")
        
        # Load dataset
        dataset_func = self.AVAILABLE_DATASETS[self.dataset_name]
        dataset = dataset_func()
        
        # Convert to DataFrame
        X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
        y = pd.Series(dataset.target, name='target')
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state,
            stratify=y
        )
        
        logger.info(f"Loaded {self.dataset_name} dataset: "
                   f"{X_train.shape[0]} train, {X_test.shape[0]} test samples")
        
        return X_train, y_train, X_test, y_test
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get dataset information.
        
        Returns:
            Dictionary with dataset metadata
        """
        dataset_func = self.AVAILABLE_DATASETS[self.dataset_name]
        dataset = dataset_func()
        
        return {
            'name': self.dataset_name,
            'n_samples': dataset.data.shape[0],
            'n_features': dataset.data.shape[1],
            'n_classes': len(dataset.target_names),
            'feature_names': dataset.feature_names,
            'target_names': dataset.target_names
        }
