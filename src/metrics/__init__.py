"""Evaluation metrics and benchmarking for hyperparameter optimization."""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Callable
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
import logging
import time

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculate various metrics for model evaluation."""
    
    @staticmethod
    def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary of metrics
        """
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted')
        }
    
    @staticmethod
    def cross_validation_score(model, X: np.ndarray, y: np.ndarray, 
                             cv: int = 5) -> float:
        """Calculate cross-validation score.
        
        Args:
            model: Model to evaluate
            X: Features
            y: Labels
            cv: Number of CV folds
            
        Returns:
            Mean CV score
        """
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        return scores.mean()


class BenchmarkSuite:
    """Comprehensive benchmarking suite for hyperparameter optimization."""
    
    def __init__(self, datasets: List[str] = None):
        """Initialize benchmark suite.
        
        Args:
            datasets: List of dataset names to benchmark
        """
        self.datasets = datasets or ['iris', 'wine', 'breast_cancer']
        self.results: Dict[str, Dict[str, Any]] = {}
    
    def evaluate_algorithm(self, algorithm_name: str, 
                         algorithm_func: Callable,
                         param_ranges: Dict[str, Tuple],
                         **kwargs) -> Dict[str, Any]:
        """Evaluate an optimization algorithm.
        
        Args:
            algorithm_name: Name of the algorithm
            algorithm_func: Algorithm function to evaluate
            param_ranges: Parameter ranges for optimization
            **kwargs: Additional arguments for algorithm
            
        Returns:
            Evaluation results
        """
        logger.info(f"Evaluating {algorithm_name}")
        
        results = {
            'algorithm': algorithm_name,
            'datasets': {},
            'overall_stats': {}
        }
        
        all_scores = []
        all_times = []
        
        for dataset_name in self.datasets:
            logger.info(f"Testing on {dataset_name} dataset")
            
            # Load dataset
            from src.data import DatasetLoader
            loader = DatasetLoader(dataset_name)
            X_train, y_train, X_test, y_test = loader.load_data()
            
            # Define objective function
            def objective_func(params):
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(**params, random_state=42)
                model.fit(X_train, y_train)
                return accuracy_score(y_test, model.predict(X_test))
            
            # Run optimization
            start_time = time.time()
            best_params = algorithm_func(param_ranges, **kwargs)
            optimization_time = time.time() - start_time
            
            # Evaluate best model
            from sklearn.ensemble import RandomForestClassifier
            best_model = RandomForestClassifier(**best_params, random_state=42)
            best_model.fit(X_train, y_train)
            
            # Calculate metrics
            y_pred = best_model.predict(X_test)
            metrics = MetricsCalculator.classification_metrics(y_test, y_pred)
            cv_score = MetricsCalculator.cross_validation_score(best_model, X_train, y_train)
            
            dataset_results = {
                'best_params': best_params,
                'test_accuracy': metrics['accuracy'],
                'test_f1': metrics['f1'],
                'cv_score': cv_score,
                'optimization_time': optimization_time,
                'metrics': metrics
            }
            
            results['datasets'][dataset_name] = dataset_results
            all_scores.append(metrics['accuracy'])
            all_times.append(optimization_time)
        
        # Overall statistics
        results['overall_stats'] = {
            'mean_accuracy': np.mean(all_scores),
            'std_accuracy': np.std(all_scores),
            'mean_time': np.mean(all_times),
            'std_time': np.std(all_times),
            'best_accuracy': np.max(all_scores),
            'worst_accuracy': np.min(all_scores)
        }
        
        self.results[algorithm_name] = results
        return results
    
    def create_leaderboard(self) -> pd.DataFrame:
        """Create a leaderboard from all results.
        
        Returns:
            DataFrame with algorithm rankings
        """
        leaderboard_data = []
        
        for algo_name, results in self.results.items():
            stats = results['overall_stats']
            leaderboard_data.append({
                'Algorithm': algo_name,
                'Mean Accuracy': f"{stats['mean_accuracy']:.4f}",
                'Std Accuracy': f"{stats['std_accuracy']:.4f}",
                'Best Accuracy': f"{stats['best_accuracy']:.4f}",
                'Mean Time (s)': f"{stats['mean_time']:.2f}",
                'Std Time (s)': f"{stats['std_time']:.2f}"
            })
        
        df = pd.DataFrame(leaderboard_data)
        df = df.sort_values('Mean Accuracy', ascending=False)
        return df
    
    def get_detailed_results(self) -> Dict[str, Any]:
        """Get detailed results for all algorithms.
        
        Returns:
            Dictionary with detailed results
        """
        return self.results
