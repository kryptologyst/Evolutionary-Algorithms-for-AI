"""Main training script for evolutionary algorithms."""

import argparse
import yaml
import logging
from pathlib import Path
from typing import Dict, Any

from src.data import DatasetLoader
from src.models import GeneticAlgorithm, DifferentialEvolution
from src.models.baselines import (
    RandomSearch, GridSearch, OptunaOptimizer, 
    HyperoptOptimizer, BayesianOptimizer
)
from src.metrics import BenchmarkSuite
from src.utils import set_seed, Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Config:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration object
    """
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return Config(config_dict)


def run_benchmark(config: Config) -> None:
    """Run comprehensive benchmark of optimization algorithms.
    
    Args:
        config: Configuration object
    """
    logger.info("Starting evolutionary algorithms benchmark")
    
    # Set random seed
    set_seed(config.random_seed)
    
    # Define parameter ranges for RandomForest
    param_ranges = {
        'n_estimators': (50, 200),
        'max_depth': (3, 20),
        'min_samples_split': (2, 10),
        'min_samples_leaf': (1, 5),
        'max_features': (0.5, 1.0)
    }
    
    # Initialize benchmark suite
    benchmark = BenchmarkSuite(config.datasets)
    
    # Define algorithms to test
    algorithms = {
        'Random Search': lambda pr, **kwargs: RandomSearch(
            n_trials=config.n_trials, random_state=config.random_seed
        ).optimize(pr, lambda params: evaluate_model(params, config)),
        
        'Grid Search': lambda pr, **kwargs: GridSearch(
            n_points_per_param=config.grid_points
        ).optimize(pr, lambda params: evaluate_model(params, config)),
        
        'Optuna (TPE)': lambda pr, **kwargs: OptunaOptimizer(
            n_trials=config.n_trials, random_state=config.random_seed
        ).optimize(pr, lambda params: evaluate_model(params, config)),
        
        'Hyperopt (TPE)': lambda pr, **kwargs: HyperoptOptimizer(
            max_evals=config.n_trials, random_state=config.random_seed
        ).optimize(pr, lambda params: evaluate_model(params, config)),
        
        'Bayesian Optimization': lambda pr, **kwargs: BayesianOptimizer(
            n_iter=config.n_trials, random_state=config.random_seed
        ).optimize(pr, lambda params: evaluate_model(params, config)),
        
        'Genetic Algorithm': lambda pr, **kwargs: GeneticAlgorithm(
            param_ranges=pr,
            population_size=config.population_size,
            generations=config.generations,
            random_state=config.random_seed
        ).optimize(lambda params: evaluate_model(params, config)).params,
        
        'Differential Evolution': lambda pr, **kwargs: DifferentialEvolution(
            param_ranges=pr,
            population_size=config.population_size,
            generations=config.generations,
            random_state=config.random_seed
        ).optimize(lambda params: evaluate_model(params, config)).params
    }
    
    # Run benchmarks
    for algo_name, algo_func in algorithms.items():
        try:
            benchmark.evaluate_algorithm(
                algo_name, algo_func, param_ranges
            )
        except Exception as e:
            logger.error(f"Error evaluating {algo_name}: {e}")
    
    # Create leaderboard
    leaderboard = benchmark.create_leaderboard()
    print("\n" + "="*80)
    print("EVOLUTIONARY ALGORITHMS BENCHMARK RESULTS")
    print("="*80)
    print(leaderboard.to_string(index=False))
    
    # Save results
    results_path = Path(config.output_dir) / "benchmark_results.yaml"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    import yaml
    with open(results_path, 'w') as f:
        yaml.dump(benchmark.get_detailed_results(), f, default_flow_style=False)
    
    logger.info(f"Results saved to {results_path}")


def evaluate_model(params: Dict[str, Any], config: Config) -> float:
    """Evaluate model with given parameters.
    
    Args:
        params: Model parameters
        config: Configuration object
        
    Returns:
        Model accuracy score
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    
    # Load dataset (using first dataset for simplicity)
    loader = DatasetLoader(config.datasets[0])
    X_train, y_train, X_test, y_test = loader.load_data()
    
    # Create and train model
    model = RandomForestClassifier(**params, random_state=config.random_seed)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Evolutionary Algorithms Benchmark')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    config.update(output_dir=args.output_dir)
    
    # Run benchmark
    run_benchmark(config)


if __name__ == "__main__":
    main()
