# Evolutionary Algorithms for AI

A comprehensive research and educational framework for evolutionary algorithms applied to hyperparameter optimization in machine learning.

## Overview

This project implements and compares various evolutionary algorithms and optimization methods for finding optimal hyperparameters of machine learning models. It includes both classical baselines and advanced evolutionary approaches, providing a complete benchmarking suite for research and education.

## Features

- **Multiple Optimization Algorithms**: Random Search, Grid Search, Optuna (TPE), Hyperopt, Bayesian Optimization, Genetic Algorithm, Differential Evolution
- **Comprehensive Benchmarking**: Automated evaluation across multiple datasets with detailed metrics
- **Interactive Demo**: Streamlit-based web application for hands-on experimentation
- **Reproducible Research**: Deterministic seeding and comprehensive logging
- **Modern Architecture**: Clean, typed code with proper documentation

## Safety and Ethics

⚠️ **Important Disclaimers**:
- This tool is for **research and educational purposes only**
- **Not intended for production decisions** or critical applications
- Results may vary and should be validated independently
- No warranty or guarantee of performance

## Installation

### Prerequisites
- Python 3.10 or higher
- pip or conda package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Evolutionary-Algorithms-for-AI.git
cd Evolutionary-Algorithms-for-AI
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or install in development mode:
```bash
pip install -e ".[dev]"
```

3. Run pre-commit hooks (optional):
```bash
pre-commit install
```

## Quick Start

### Command Line Interface

Run the comprehensive benchmark:

```bash
python -m src.train --config configs/default.yaml --output-dir results
```

### Interactive Demo

Launch the Streamlit demo:

```bash
streamlit run demo/app.py
```

Then open your browser to `http://localhost:8501`

## Usage

### Basic Usage

```python
from src.data import DatasetLoader
from src.models import GeneticAlgorithm
from src.utils import set_seed

# Set random seed for reproducibility
set_seed(42)

# Load dataset
loader = DatasetLoader('iris')
X_train, y_train, X_test, y_test = loader.load_data()

# Define parameter ranges
param_ranges = {
    'n_estimators': (50, 200),
    'max_depth': (3, 20),
    'min_samples_split': (2, 10)
}

# Define objective function
def objective_func(params):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    
    model = RandomForestClassifier(**params, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

# Run genetic algorithm
ga = GeneticAlgorithm(param_ranges, population_size=20, generations=50)
best_individual = ga.optimize(objective_func)

print(f"Best parameters: {best_individual.params}")
print(f"Best accuracy: {best_individual.fitness:.4f}")
```

### Advanced Usage

```python
from src.metrics import BenchmarkSuite
from src.models.baselines import OptunaOptimizer, BayesianOptimizer

# Create benchmark suite
benchmark = BenchmarkSuite(['iris', 'wine', 'breast_cancer'])

# Evaluate multiple algorithms
algorithms = {
    'Optuna': OptunaOptimizer(n_trials=100),
    'Bayesian': BayesianOptimizer(n_iter=100),
    'Genetic Algorithm': GeneticAlgorithm(param_ranges, generations=50)
}

for name, algo in algorithms.items():
    benchmark.evaluate_algorithm(name, algo.optimize, param_ranges)

# View results
leaderboard = benchmark.create_leaderboard()
print(leaderboard)
```

## Configuration

The project uses YAML configuration files. See `configs/default.yaml` for available options:

```yaml
# Random seed for reproducibility
random_seed: 42

# Datasets to benchmark on
datasets:
  - iris
  - wine
  - breast_cancer

# Optimization parameters
n_trials: 100
population_size: 20
generations: 50

# Output directory
output_dir: results
```

## Project Structure

```
Evolutionary-Algorithms-for-AI/
├── src/                    # Source code
│   ├── data/              # Data loading utilities
│   ├── models/            # Evolutionary algorithms
│   ├── metrics/           # Evaluation metrics
│   ├── train/             # Training scripts
│   └── utils/             # Utility functions
├── configs/               # Configuration files
├── demo/                  # Streamlit demo
├── tests/                 # Unit tests
├── assets/                # Generated assets
├── data/                  # Data directory
├── results/               # Experiment results
├── requirements.txt       # Dependencies
├── pyproject.toml         # Project configuration
└── README.md              # This file
```

## Algorithms Implemented

### Baselines
- **Random Search**: Random sampling of parameter space
- **Grid Search**: Exhaustive search over parameter grid

### Bayesian Optimization
- **Optuna**: Tree-structured Parzen Estimator (TPE)
- **Hyperopt**: Bayesian optimization with TPE
- **Bayesian Optimization**: Gaussian Process-based optimization

### Evolutionary Algorithms
- **Genetic Algorithm**: Crossover, mutation, and selection
- **Differential Evolution**: Population-based differential mutation

## Evaluation Metrics

The framework provides comprehensive evaluation including:

- **Accuracy**: Classification accuracy on test set
- **Cross-validation Score**: Mean CV accuracy
- **Optimization Time**: Time to find best parameters
- **Convergence**: Fitness progression over generations
- **Statistical Significance**: Mean and standard deviation across datasets

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Development

### Code Quality

The project uses several tools for code quality:

- **Black**: Code formatting
- **Ruff**: Linting
- **MyPy**: Type checking
- **Pre-commit**: Git hooks

Run quality checks:

```bash
black src/
ruff check src/
mypy src/
```

### Testing

Run tests:

```bash
pytest tests/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{evolutionary_algorithms_ai,
  title={Evolutionary Algorithms for AI},
  author={kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Evolutionary-Algorithms-for-AI}
}
```

## Acknowledgments

- Author: [kryptologyst](https://github.com/kryptologyst)
- GitHub: https://github.com/kryptologyst
- Inspired by evolutionary computation research and hyperparameter optimization literature

## Disclaimer

This software is provided for educational and research purposes only. The authors make no warranties regarding the accuracy, reliability, or suitability for any purpose. Users are responsible for validating results independently and should not rely on this software for production decisions or critical applications.
# Evolutionary-Algorithms-for-AI
