#!/usr/bin/env python3
"""Simple example demonstrating evolutionary algorithms for hyperparameter optimization."""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import random
from typing import Dict, Any, List, Tuple

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)


def generate_population(pop_size: int, param_ranges: Dict[str, Tuple]) -> List[Dict[str, Any]]:
    """Generate a population of hyperparameters for RandomForest.
    
    Args:
        pop_size: Size of population
        param_ranges: Parameter ranges
        
    Returns:
        List of parameter dictionaries
    """
    population = []
    for _ in range(pop_size):
        individual = {
            'n_estimators': random.randint(param_ranges['n_estimators'][0], param_ranges['n_estimators'][1]),
            'max_depth': random.randint(param_ranges['max_depth'][0], param_ranges['max_depth'][1]),
            'min_samples_split': random.randint(param_ranges['min_samples_split'][0], param_ranges['min_samples_split'][1])
        }
        population.append(individual)
    return population


def evaluate_individual(individual: Dict[str, Any], X_train: pd.DataFrame, 
                      y_train: pd.Series, X_test: pd.DataFrame, 
                      y_test: pd.Series) -> float:
    """Evaluate a set of hyperparameters (individual) using test set.
    
    Args:
        individual: Parameter dictionary
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Accuracy score
    """
    # Create RandomForest model with the individual's hyperparameters
    model = RandomForestClassifier(
        n_estimators=individual['n_estimators'],
        max_depth=individual['max_depth'],
        min_samples_split=individual['min_samples_split'],
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate model on the test set
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)


def select_parents(population: List[Dict[str, Any]], X_train: pd.DataFrame,
                  y_train: pd.Series, X_test: pd.DataFrame, 
                  y_test: pd.Series) -> List[Dict[str, Any]]:
    """Select two parents from the population based on their fitness (accuracy).
    
    Args:
        population: Current population
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        
    Returns:
        List of two parent individuals
    """
    fitness_scores = [
        evaluate_individual(individual, X_train, y_train, X_test, y_test) 
        for individual in population
    ]
    
    # Select two parents based on their fitness (higher fitness = better chance of selection)
    parents = np.random.choice(
        population, size=2, 
        p=np.array(fitness_scores) / np.sum(fitness_scores)
    )
    return parents.tolist()


def crossover(parents: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Perform crossover between two parents to create offspring.
    
    Args:
        parents: List of two parent individuals
        
    Returns:
        Offspring individual
    """
    parent1, parent2 = parents
    offspring = {}
    for param in parent1:
        # Choose randomly from the two parents
        offspring[param] = random.choice([parent1[param], parent2[param]])
    return offspring


def mutate(offspring: Dict[str, Any], param_ranges: Dict[str, Tuple]) -> Dict[str, Any]:
    """Perform mutation on an offspring (randomly change one hyperparameter).
    
    Args:
        offspring: Individual to mutate
        param_ranges: Parameter ranges
        
    Returns:
        Mutated individual
    """
    mutation_param = random.choice(list(offspring.keys()))
    # Mutate the selected parameter by randomly changing its value
    offspring[mutation_param] = random.randint(
        param_ranges[mutation_param][0], 
        param_ranges[mutation_param][1]
    )
    return offspring


def genetic_algorithm(X_train: pd.DataFrame, y_train: pd.Series,
                     X_test: pd.DataFrame, y_test: pd.Series,
                     generations: int = 10, pop_size: int = 10) -> Dict[str, Any]:
    """Genetic Algorithm for optimizing hyperparameters.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        generations: Number of generations
        pop_size: Population size
        
    Returns:
        Best individual found
    """
    param_ranges = {
        'n_estimators': (50, 200),
        'max_depth': (3, 20),
        'min_samples_split': (2, 10)
    }
    
    # Generate initial population
    population = generate_population(pop_size, param_ranges)
    
    for generation in range(generations):
        print(f"Generation {generation + 1}")
        
        # Select parents
        parents = select_parents(population, X_train, y_train, X_test, y_test)
        
        # Crossover to generate offspring
        offspring = crossover(parents)
        
        # Mutation
        offspring = mutate(offspring, param_ranges)
        
        # Evaluate the new offspring
        offspring_fitness = evaluate_individual(offspring, X_train, y_train, X_test, y_test)
        
        # Replace the worst-performing individual with the new offspring
        fitness_scores = [
            evaluate_individual(individual, X_train, y_train, X_test, y_test) 
            for individual in population
        ]
        worst_individual_idx = np.argmin(fitness_scores)
        population[worst_individual_idx] = offspring
        
        print(f"Best Accuracy in Generation {generation + 1}: {max(fitness_scores):.4f}")
    
    # Return the best individual after all generations
    fitness_scores = [
        evaluate_individual(individual, X_train, y_train, X_test, y_test) 
        for individual in population
    ]
    best_individual_idx = np.argmax(fitness_scores)
    best_individual = population[best_individual_idx]
    print(f"Best Hyperparameters: {best_individual}")
    return best_individual


def main():
    """Main function demonstrating evolutionary algorithm."""
    print("🧬 Evolutionary Algorithms for AI - Simple Example")
    print("=" * 60)
    
    # Load the Iris dataset
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Dataset loaded: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    
    # Run the Genetic Algorithm for hyperparameter optimization
    print("\n🚀 Running Genetic Algorithm...")
    best_params = genetic_algorithm(X_train, y_train, X_test, y_test, 
                                 generations=10, pop_size=10)
    
    # Train the final model with the best hyperparameters
    print("\n🎯 Training final model with best parameters...")
    final_model = RandomForestClassifier(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        random_state=42
    )
    final_model.fit(X_train, y_train)
    
    # Evaluate the final model
    y_pred_final = final_model.predict(X_test)
    final_accuracy = accuracy_score(y_test, y_pred_final)
    
    print(f"\n📊 Final Results:")
    print(f"Best Parameters: {best_params}")
    print(f"Final Model Accuracy: {final_accuracy:.4f}")
    
    # Compare with default parameters
    default_model = RandomForestClassifier(random_state=42)
    default_model.fit(X_train, y_train)
    default_pred = default_model.predict(X_test)
    default_accuracy = accuracy_score(y_test, default_pred)
    
    print(f"Default Model Accuracy: {default_accuracy:.4f}")
    print(f"Improvement: {final_accuracy - default_accuracy:.4f}")
    
    print("\n✅ Example completed successfully!")
    print("\nFor more advanced features, run:")
    print("  python -m src.train --config configs/default.yaml")
    print("  streamlit run demo/app.py")


if __name__ == "__main__":
    main()
