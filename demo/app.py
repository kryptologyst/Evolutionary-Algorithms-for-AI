"""Streamlit demo for evolutionary algorithms hyperparameter optimization."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from typing import Dict, Any, List

from src.data import DatasetLoader
from src.models import GeneticAlgorithm, DifferentialEvolution
from src.models.baselines import (
    RandomSearch, GridSearch, OptunaOptimizer, 
    HyperoptOptimizer, BayesianOptimizer
)
from src.utils import set_seed

# Page configuration
st.set_page_config(
    page_title="Evolutionary Algorithms for AI",
    page_icon="🧬",
    layout="wide"
)

# Title and description
st.title("🧬 Evolutionary Algorithms for Hyperparameter Optimization")
st.markdown("""
This demo showcases various evolutionary algorithms and optimization methods for finding 
optimal hyperparameters of machine learning models. Compare different approaches including 
Genetic Algorithms, Differential Evolution, and Bayesian Optimization.
""")

# Safety disclaimer
st.warning("""
⚠️ **Research Demo Only**: This tool is for educational and research purposes only. 
Not intended for production decisions or critical applications. Results may vary and 
should be validated independently.
""")

# Sidebar configuration
st.sidebar.header("Configuration")

# Dataset selection
dataset_name = st.sidebar.selectbox(
    "Select Dataset",
    ["iris", "wine", "breast_cancer"],
    help="Choose the dataset for hyperparameter optimization"
)

# Algorithm selection
algorithm = st.sidebar.selectbox(
    "Select Algorithm",
    [
        "Random Search",
        "Grid Search", 
        "Optuna (TPE)",
        "Hyperopt (TPE)",
        "Bayesian Optimization",
        "Genetic Algorithm",
        "Differential Evolution"
    ],
    help="Choose the optimization algorithm"
)

# Parameters
col1, col2 = st.sidebar.columns(2)

with col1:
    n_trials = st.number_input("Number of Trials", min_value=10, max_value=500, value=100)
    population_size = st.number_input("Population Size", min_value=5, max_value=100, value=20)

with col2:
    generations = st.number_input("Generations", min_value=5, max_value=200, value=50)
    random_seed = st.number_input("Random Seed", min_value=0, max_value=1000, value=42)

# Parameter ranges
st.sidebar.header("Parameter Ranges")

param_ranges = {
    'n_estimators': (st.sidebar.slider("N Estimators", 10, 300, (50, 200))),
    'max_depth': (st.sidebar.slider("Max Depth", 1, 30, (3, 20))),
    'min_samples_split': (st.sidebar.slider("Min Samples Split", 2, 20, (2, 10))),
    'min_samples_leaf': (st.sidebar.slider("Min Samples Leaf", 1, 10, (1, 5))),
    'max_features': (st.sidebar.slider("Max Features", 0.1, 1.0, (0.5, 1.0)))
}

# Main content
if st.button("🚀 Run Optimization", type="primary"):
    
    # Set random seed
    set_seed(random_seed)
    
    # Load dataset
    with st.spinner("Loading dataset..."):
        loader = DatasetLoader(dataset_name)
        X_train, y_train, X_test, y_test = loader.load_data()
        dataset_info = loader.get_dataset_info()
    
    # Display dataset info
    st.subheader("📊 Dataset Information")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Samples", dataset_info['n_samples'])
    with col2:
        st.metric("Features", dataset_info['n_features'])
    with col3:
        st.metric("Classes", dataset_info['n_classes'])
    with col4:
        st.metric("Train/Test Split", f"{len(X_train)}/{len(X_test)}")
    
    # Define objective function
    def objective_func(params):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        
        model = RandomForestClassifier(**params, random_state=random_seed)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return accuracy_score(y_test, y_pred)
    
    # Run optimization
    with st.spinner(f"Running {algorithm} optimization..."):
        start_time = time.time()
        
        if algorithm == "Random Search":
            optimizer = RandomSearch(n_trials=n_trials, random_state=random_seed)
            best_params = optimizer.optimize(param_ranges, objective_func)
            
        elif algorithm == "Grid Search":
            optimizer = GridSearch(n_points_per_param=5)
            best_params = optimizer.optimize(param_ranges, objective_func)
            
        elif algorithm == "Optuna (TPE)":
            optimizer = OptunaOptimizer(n_trials=n_trials, random_state=random_seed)
            best_params = optimizer.optimize(param_ranges, objective_func)
            
        elif algorithm == "Hyperopt (TPE)":
            optimizer = HyperoptOptimizer(max_evals=n_trials, random_state=random_seed)
            best_params = optimizer.optimize(param_ranges, objective_func)
            
        elif algorithm == "Bayesian Optimization":
            optimizer = BayesianOptimizer(n_iter=n_trials, random_state=random_seed)
            best_params = optimizer.optimize(param_ranges, objective_func)
            
        elif algorithm == "Genetic Algorithm":
            ga = GeneticAlgorithm(
                param_ranges=param_ranges,
                population_size=population_size,
                generations=generations,
                random_state=random_seed
            )
            best_individual = ga.optimize(objective_func)
            best_params = best_individual.params
            
        elif algorithm == "Differential Evolution":
            de = DifferentialEvolution(
                param_ranges=param_ranges,
                population_size=population_size,
                generations=generations,
                random_state=random_seed
            )
            best_individual = de.optimize(objective_func)
            best_params = best_individual.params
        
        optimization_time = time.time() - start_time
    
    # Display results
    st.subheader("🎯 Optimization Results")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("**Best Parameters Found:**")
        for param, value in best_params.items():
            st.write(f"- {param}: {value}")
    
    with col2:
        st.metric("Optimization Time", f"{optimization_time:.2f}s")
    
    # Evaluate best model
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report
    
    best_model = RandomForestClassifier(**best_params, random_state=random_seed)
    best_model.fit(X_train, y_train)
    
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    st.metric("Test Accuracy", f"{accuracy:.4f}")
    
    # Feature importance
    if hasattr(best_model, 'feature_importances_'):
        st.subheader("📈 Feature Importance")
        
        feature_importance = pd.DataFrame({
            'Feature': dataset_info['feature_names'],
            'Importance': best_model.feature_importances_
        }).sort_values('Importance', ascending=True)
        
        fig = px.bar(
            feature_importance, 
            x='Importance', 
            y='Feature',
            orientation='h',
            title="Random Forest Feature Importance"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Classification report
    st.subheader("📋 Classification Report")
    report = classification_report(y_test, y_pred, target_names=dataset_info['target_names'])
    st.text(report)

# Additional information
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown("""
This demo implements various optimization algorithms for hyperparameter tuning:

- **Random Search**: Random sampling of parameter space
- **Grid Search**: Exhaustive search over parameter grid
- **Optuna**: Tree-structured Parzen Estimator
- **Hyperopt**: Bayesian optimization with TPE
- **Bayesian Optimization**: Gaussian Process-based optimization
- **Genetic Algorithm**: Evolutionary algorithm with crossover and mutation
- **Differential Evolution**: Population-based optimization

**Author**: kryptologyst  
**GitHub**: https://github.com/kryptologyst
""")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
<p>🧬 Evolutionary Algorithms for AI | Author: <a href='https://github.com/kryptologyst'>kryptologyst</a></p>
<p><em>Research and educational purposes only. Not for production use.</em></p>
</div>
""", unsafe_allow_html=True)
