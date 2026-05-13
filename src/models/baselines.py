"""Baseline optimization methods for comparison."""

import numpy as np
import random
from typing import Dict, Any, Tuple, List, Callable
import itertools
from sklearn.model_selection import ParameterGrid
import optuna
from hyperopt import fmin, tpe, hp, Trials
from bayes_opt import BayesianOptimization
import logging

logger = logging.getLogger(__name__)


class RandomSearch:
    """Random search baseline."""
    
    def __init__(self, n_trials: int = 100, random_state: int = 42):
        """Initialize random search.
        
        Args:
            n_trials: Number of random trials
            random_state: Random seed
        """
        self.n_trials = n_trials
        self.random_state = random_state
        random.seed(random_state)
        np.random.seed(random_state)
    
    def optimize(self, param_ranges: Dict[str, Tuple], 
                objective_func: Callable) -> Dict[str, Any]:
        """Run random search optimization.
        
        Args:
            param_ranges: Parameter ranges
            objective_func: Function to optimize
            
        Returns:
            Best parameters found
        """
        best_params = None
        best_score = -np.inf
        
        for _ in range(self.n_trials):
            params = {}
            for param_name, (min_val, max_val) in param_ranges.items():
                if isinstance(min_val, int) and isinstance(max_val, int):
                    params[param_name] = random.randint(min_val, max_val)
                else:
                    params[param_name] = random.uniform(min_val, max_val)
            
            score = objective_func(params)
            if score > best_score:
                best_score = score
                best_params = params
        
        return best_params


class GridSearch:
    """Grid search baseline."""
    
    def __init__(self, n_points_per_param: int = 5):
        """Initialize grid search.
        
        Args:
            n_points_per_param: Number of points per parameter
        """
        self.n_points_per_param = n_points_per_param
    
    def optimize(self, param_ranges: Dict[str, Tuple], 
                objective_func: Callable) -> Dict[str, Any]:
        """Run grid search optimization.
        
        Args:
            param_ranges: Parameter ranges
            objective_func: Function to optimize
            
        Returns:
            Best parameters found
        """
        # Create parameter grid
        param_grid = {}
        for param_name, (min_val, max_val) in param_ranges.items():
            if isinstance(min_val, int) and isinstance(max_val, int):
                param_grid[param_name] = list(range(min_val, max_val + 1, 
                                                   max(1, (max_val - min_val) // self.n_points_per_param)))
            else:
                param_grid[param_name] = np.linspace(min_val, max_val, self.n_points_per_param).tolist()
        
        grid = ParameterGrid(param_grid)
        
        best_params = None
        best_score = -np.inf
        
        for params in grid:
            score = objective_func(params)
            if score > best_score:
                best_score = score
                best_params = params
        
        return best_params


class OptunaOptimizer:
    """Optuna-based optimization."""
    
    def __init__(self, n_trials: int = 100, random_state: int = 42):
        """Initialize Optuna optimizer.
        
        Args:
            n_trials: Number of trials
            random_state: Random seed
        """
        self.n_trials = n_trials
        self.random_state = random_state
    
    def optimize(self, param_ranges: Dict[str, Tuple], 
                objective_func: Callable) -> Dict[str, Any]:
        """Run Optuna optimization.
        
        Args:
            param_ranges: Parameter ranges
            objective_func: Function to optimize
            
        Returns:
            Best parameters found
        """
        def objective(trial):
            params = {}
            for param_name, (min_val, max_val) in param_ranges.items():
                if isinstance(min_val, int) and isinstance(max_val, int):
                    params[param_name] = trial.suggest_int(param_name, min_val, max_val)
                else:
                    params[param_name] = trial.suggest_float(param_name, min_val, max_val)
            
            return objective_func(params)
        
        study = optuna.create_study(direction='maximize', 
                                  sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=self.n_trials)
        
        return study.best_params


class HyperoptOptimizer:
    """Hyperopt-based optimization."""
    
    def __init__(self, max_evals: int = 100, random_state: int = 42):
        """Initialize Hyperopt optimizer.
        
        Args:
            max_evals: Maximum evaluations
            random_state: Random seed
        """
        self.max_evals = max_evals
        self.random_state = random_state
    
    def optimize(self, param_ranges: Dict[str, Tuple], 
                objective_func: Callable) -> Dict[str, Any]:
        """Run Hyperopt optimization.
        
        Args:
            param_ranges: Parameter ranges
            objective_func: Function to optimize
            
        Returns:
            Best parameters found
        """
        # Define search space
        space = {}
        for param_name, (min_val, max_val) in param_ranges.items():
            if isinstance(min_val, int) and isinstance(max_val, int):
                space[param_name] = hp.randint(param_name, min_val, max_val + 1)
            else:
                space[param_name] = hp.uniform(param_name, min_val, max_val)
        
        def objective(params):
            return -objective_func(params)  # Minimize negative
        
        trials = Trials()
        best = fmin(fn=objective, space=space, algo=tpe.suggest, 
                   max_evals=self.max_evals, trials=trials, 
                   rstate=np.random.RandomState(self.random_state))
        
        return best


class BayesianOptimizer:
    """Bayesian optimization."""
    
    def __init__(self, n_iter: int = 50, random_state: int = 42):
        """Initialize Bayesian optimizer.
        
        Args:
            n_iter: Number of iterations
            random_state: Random seed
        """
        self.n_iter = n_iter
        self.random_state = random_state
    
    def optimize(self, param_ranges: Dict[str, Tuple], 
                objective_func: Callable) -> Dict[str, Any]:
        """Run Bayesian optimization.
        
        Args:
            param_ranges: Parameter ranges
            objective_func: Function to optimize
            
        Returns:
            Best parameters found
        """
        # Define bounds
        bounds = {}
        for param_name, (min_val, max_val) in param_ranges.items():
            bounds[param_name] = (min_val, max_val)
        
        def objective(**params):
            return objective_func(params)
        
        optimizer = BayesianOptimization(
            f=objective,
            pbounds=bounds,
            random_state=self.random_state
        )
        
        optimizer.maximize(init_points=5, n_iter=self.n_iter)
        
        return optimizer.max['params']
