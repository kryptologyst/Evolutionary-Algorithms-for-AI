"""Test suite for evolutionary algorithms."""

import pytest
import numpy as np
from src.data import DatasetLoader
from src.models import GeneticAlgorithm, DifferentialEvolution, Individual
from src.models.baselines import RandomSearch, GridSearch
from src.utils import set_seed


class TestIndividual:
    """Test Individual class."""
    
    def test_individual_creation(self):
        """Test individual creation."""
        params = {'param1': 10, 'param2': 0.5}
        individual = Individual(params, fitness=0.8)
        
        assert individual.params == params
        assert individual.fitness == 0.8
    
    def test_individual_string_representation(self):
        """Test string representation."""
        params = {'param1': 10}
        individual = Individual(params, fitness=0.8)
        str_repr = str(individual)
        
        assert 'param1' in str_repr
        assert '0.8' in str_repr


class TestDatasetLoader:
    """Test DatasetLoader class."""
    
    def test_load_iris(self):
        """Test loading iris dataset."""
        loader = DatasetLoader('iris')
        X_train, y_train, X_test, y_test = loader.load_data()
        
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)
    
    def test_dataset_info(self):
        """Test dataset info retrieval."""
        loader = DatasetLoader('iris')
        info = loader.get_dataset_info()
        
        assert 'name' in info
        assert 'n_samples' in info
        assert 'n_features' in info
        assert 'n_classes' in info


class TestGeneticAlgorithm:
    """Test GeneticAlgorithm class."""
    
    def test_initialization(self):
        """Test GA initialization."""
        param_ranges = {'param1': (1, 10), 'param2': (0.0, 1.0)}
        ga = GeneticAlgorithm(param_ranges, population_size=10, generations=5)
        
        assert ga.param_ranges == param_ranges
        assert ga.population_size == 10
        assert ga.generations == 5
    
    def test_generate_population(self):
        """Test population generation."""
        param_ranges = {'param1': (1, 10)}
        ga = GeneticAlgorithm(param_ranges, population_size=5)
        population = ga.generate_population()
        
        assert len(population) == 5
        for individual in population:
            assert 'param1' in individual.params
            assert 1 <= individual.params['param1'] <= 10
    
    def test_crossover(self):
        """Test crossover operation."""
        param_ranges = {'param1': (1, 10), 'param2': (0.0, 1.0)}
        ga = GeneticAlgorithm(param_ranges)
        
        parent1 = Individual({'param1': 5, 'param2': 0.3})
        parent2 = Individual({'param1': 7, 'param2': 0.7})
        
        offspring = ga.crossover([parent1, parent2])
        
        assert isinstance(offspring, Individual)
        assert 'param1' in offspring.params
        assert 'param2' in offspring.params
    
    def test_mutation(self):
        """Test mutation operation."""
        param_ranges = {'param1': (1, 10)}
        ga = GeneticAlgorithm(param_ranges, mutation_rate=1.0)  # Always mutate
        
        individual = Individual({'param1': 5})
        mutated = ga.mutate(individual)
        
        assert isinstance(mutated, Individual)
        assert 'param1' in mutated.params


class TestDifferentialEvolution:
    """Test DifferentialEvolution class."""
    
    def test_initialization(self):
        """Test DE initialization."""
        param_ranges = {'param1': (1, 10)}
        de = DifferentialEvolution(param_ranges, population_size=10, generations=5)
        
        assert de.param_ranges == param_ranges
        assert de.population_size == 10
        assert de.generations == 5
    
    def test_crossover(self):
        """Test DE crossover."""
        param_ranges = {'param1': (1, 10)}
        de = DifferentialEvolution(param_ranges, F=0.8, CR=1.0)
        
        target = Individual({'param1': 5})
        donor1 = Individual({'param1': 3})
        donor2 = Individual({'param1': 7})
        
        trial = de.crossover([target, donor1, donor2])
        
        assert isinstance(trial, Individual)
        assert 'param1' in trial.params


class TestBaselines:
    """Test baseline optimization methods."""
    
    def test_random_search(self):
        """Test random search."""
        param_ranges = {'param1': (1, 10)}
        
        def objective(params):
            return params['param1']  # Simple objective
        
        rs = RandomSearch(n_trials=10)
        best_params = rs.optimize(param_ranges, objective)
        
        assert 'param1' in best_params
        assert 1 <= best_params['param1'] <= 10
    
    def test_grid_search(self):
        """Test grid search."""
        param_ranges = {'param1': (1, 3)}
        
        def objective(params):
            return params['param1']
        
        gs = GridSearch(n_points_per_param=3)
        best_params = gs.optimize(param_ranges, objective)
        
        assert 'param1' in best_params
        assert best_params['param1'] == 3  # Should find maximum


if __name__ == "__main__":
    pytest.main([__file__])
