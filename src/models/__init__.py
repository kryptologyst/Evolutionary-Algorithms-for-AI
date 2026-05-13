"""Evolutionary algorithm implementations for hyperparameter optimization."""

import numpy as np
import random
from typing import Dict, Any, List, Tuple, Callable, Optional
from abc import ABC, abstractmethod
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Individual:
    """Represents an individual in the evolutionary algorithm."""
    
    def __init__(self, params: Dict[str, Any], fitness: Optional[float] = None):
        """Initialize individual.
        
        Args:
            params: Hyperparameter dictionary
            fitness: Fitness score
        """
        self.params = params
        self.fitness = fitness
    
    def __str__(self) -> str:
        return f"Individual(params={self.params}, fitness={self.fitness})"
    
    def __repr__(self) -> str:
        return self.__str__()


class BaseEvolutionaryAlgorithm(ABC):
    """Base class for evolutionary algorithms."""
    
    def __init__(self, param_ranges: Dict[str, Tuple], 
                 population_size: int = 20, 
                 generations: int = 50,
                 random_state: int = 42):
        """Initialize evolutionary algorithm.
        
        Args:
            param_ranges: Parameter ranges for optimization
            population_size: Size of population
            generations: Number of generations
            random_state: Random seed
        """
        self.param_ranges = param_ranges
        self.population_size = population_size
        self.generations = generations
        self.random_state = random_state
        
        # Set random seed
        random.seed(random_state)
        np.random.seed(random_state)
        
        self.population: List[Individual] = []
        self.fitness_history: List[float] = []
        
    def generate_population(self) -> List[Individual]:
        """Generate initial population."""
        population = []
        for _ in range(self.population_size):
            params = {}
            for param_name, (min_val, max_val) in self.param_ranges.items():
                if isinstance(min_val, int) and isinstance(max_val, int):
                    params[param_name] = random.randint(min_val, max_val)
                else:
                    params[param_name] = random.uniform(min_val, max_val)
            population.append(Individual(params))
        return population
    
    @abstractmethod
    def select_parents(self, population: List[Individual]) -> List[Individual]:
        """Select parents for reproduction."""
        pass
    
    @abstractmethod
    def crossover(self, parents: List[Individual]) -> Individual:
        """Create offspring from parents."""
        pass
    
    @abstractmethod
    def mutate(self, individual: Individual) -> Individual:
        """Mutate individual."""
        pass
    
    def evaluate_population(self, population: List[Individual], 
                          objective_func: Callable) -> List[Individual]:
        """Evaluate fitness of population."""
        for individual in population:
            if individual.fitness is None:
                individual.fitness = objective_func(individual.params)
        return population
    
    def optimize(self, objective_func: Callable) -> Individual:
        """Run evolutionary optimization.
        
        Args:
            objective_func: Function to optimize (higher is better)
            
        Returns:
            Best individual found
        """
        # Initialize population
        self.population = self.generate_population()
        
        # Main evolution loop
        for generation in tqdm(range(self.generations), desc="Evolution"):
            # Evaluate population
            self.population = self.evaluate_population(self.population, objective_func)
            
            # Track best fitness
            best_fitness = max(ind.fitness for ind in self.population)
            self.fitness_history.append(best_fitness)
            
            # Create new generation
            new_population = []
            
            # Elitism: keep best individual
            best_individual = max(self.population, key=lambda x: x.fitness)
            new_population.append(Individual(best_individual.params, best_individual.fitness))
            
            # Generate rest of population
            while len(new_population) < self.population_size:
                parents = self.select_parents(self.population)
                offspring = self.crossover(parents)
                offspring = self.mutate(offspring)
                new_population.append(offspring)
            
            self.population = new_population
            
            logger.info(f"Generation {generation + 1}: Best fitness = {best_fitness:.4f}")
        
        # Return best individual
        final_population = self.evaluate_population(self.population, objective_func)
        return max(final_population, key=lambda x: x.fitness)


class GeneticAlgorithm(BaseEvolutionaryAlgorithm):
    """Genetic Algorithm implementation."""
    
    def __init__(self, param_ranges: Dict[str, Tuple], 
                 population_size: int = 20, 
                 generations: int = 50,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8,
                 tournament_size: int = 3,
                 random_state: int = 42):
        """Initialize Genetic Algorithm.
        
        Args:
            param_ranges: Parameter ranges
            population_size: Population size
            generations: Number of generations
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            tournament_size: Size of tournament selection
            random_state: Random seed
        """
        super().__init__(param_ranges, population_size, generations, random_state)
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
    
    def select_parents(self, population: List[Individual]) -> List[Individual]:
        """Tournament selection."""
        parents = []
        for _ in range(2):
            tournament = random.sample(population, self.tournament_size)
            winner = max(tournament, key=lambda x: x.fitness)
            parents.append(winner)
        return parents
    
    def crossover(self, parents: List[Individual]) -> Individual:
        """Uniform crossover."""
        if random.random() > self.crossover_rate:
            return Individual(parents[0].params.copy())
        
        offspring_params = {}
        for param_name in self.param_ranges.keys():
            offspring_params[param_name] = random.choice([
                parents[0].params[param_name],
                parents[1].params[param_name]
            ])
        
        return Individual(offspring_params)
    
    def mutate(self, individual: Individual) -> Individual:
        """Uniform mutation."""
        mutated_params = individual.params.copy()
        
        for param_name, (min_val, max_val) in self.param_ranges.items():
            if random.random() < self.mutation_rate:
                if isinstance(min_val, int) and isinstance(max_val, int):
                    mutated_params[param_name] = random.randint(min_val, max_val)
                else:
                    mutated_params[param_name] = random.uniform(min_val, max_val)
        
        return Individual(mutated_params)


class DifferentialEvolution(BaseEvolutionaryAlgorithm):
    """Differential Evolution implementation."""
    
    def __init__(self, param_ranges: Dict[str, Tuple], 
                 population_size: int = 20, 
                 generations: int = 50,
                 F: float = 0.8,
                 CR: float = 0.9,
                 random_state: int = 42):
        """Initialize Differential Evolution.
        
        Args:
            param_ranges: Parameter ranges
            population_size: Population size
            generations: Number of generations
            F: Differential weight
            CR: Crossover probability
            random_state: Random seed
        """
        super().__init__(param_ranges, population_size, generations, random_state)
        self.F = F
        self.CR = CR
    
    def select_parents(self, population: List[Individual]) -> List[Individual]:
        """Select three parents for DE."""
        return random.sample(population, 3)
    
    def crossover(self, parents: List[Individual]) -> Individual:
        """DE crossover and mutation."""
        target, donor1, donor2 = parents
        
        # Create trial vector
        trial_params = {}
        param_names = list(self.param_ranges.keys())
        j_rand = random.randint(0, len(param_names) - 1)
        
        for i, param_name in enumerate(param_names):
            if random.random() < self.CR or i == j_rand:
                # Mutation
                trial_params[param_name] = (
                    donor1.params[param_name] + 
                    self.F * (donor2.params[param_name] - target.params[param_name])
                )
                
                # Ensure bounds
                min_val, max_val = self.param_ranges[param_name]
                trial_params[param_name] = max(min_val, min(max_val, trial_params[param_name]))
            else:
                trial_params[param_name] = target.params[param_name]
        
        return Individual(trial_params)
    
    def mutate(self, individual: Individual) -> Individual:
        """No additional mutation in DE."""
        return individual
