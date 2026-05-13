"""Visualization utilities for evolutionary algorithms."""

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class EvolutionVisualizer:
    """Visualization tools for evolutionary algorithms."""
    
    @staticmethod
    def plot_fitness_progression(fitness_history: List[float], 
                               title: str = "Fitness Progression",
                               save_path: Optional[str] = None) -> None:
        """Plot fitness progression over generations.
        
        Args:
            fitness_history: List of best fitness values per generation
            title: Plot title
            save_path: Optional path to save plot
        """
        plt.figure(figsize=(10, 6))
        plt.plot(fitness_history, linewidth=2, marker='o', markersize=4)
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_parameter_distribution(population: List[Dict[str, Any]], 
                                  param_name: str,
                                  title: str = None,
                                  save_path: Optional[str] = None) -> None:
        """Plot distribution of a parameter in the population.
        
        Args:
            population: List of individuals
            param_name: Name of parameter to plot
            title: Plot title
            save_path: Optional path to save plot
        """
        values = [ind.params[param_name] for ind in population]
        
        plt.figure(figsize=(8, 6))
        plt.hist(values, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel(param_name)
        plt.ylabel('Frequency')
        plt.title(title or f'Distribution of {param_name}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def create_interactive_fitness_plot(fitness_history: List[float],
                                      algorithm_name: str = "Algorithm") -> go.Figure:
        """Create interactive fitness progression plot.
        
        Args:
            fitness_history: List of best fitness values
            algorithm_name: Name of the algorithm
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=list(range(len(fitness_history))),
            y=fitness_history,
            mode='lines+markers',
            name=f'{algorithm_name} Best Fitness',
            line=dict(width=3),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            title=f'{algorithm_name} - Fitness Progression',
            xaxis_title='Generation',
            yaxis_title='Best Fitness',
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    @staticmethod
    def create_parameter_heatmap(results: Dict[str, Dict[str, Any]],
                               param1: str, param2: str,
                               metric: str = 'accuracy') -> go.Figure:
        """Create heatmap of parameter combinations.
        
        Args:
            results: Results dictionary
            param1: First parameter name
            param2: Second parameter name
            metric: Metric to visualize
            
        Returns:
            Plotly figure
        """
        # Extract data for heatmap
        data = []
        for algo_name, algo_results in results.items():
            for dataset_name, dataset_results in algo_results['datasets'].items():
                params = dataset_results['best_params']
                score = dataset_results['metrics'][metric]
                data.append({
                    'Algorithm': algo_name,
                    'Dataset': dataset_name,
                    param1: params[param1],
                    param2: params[param2],
                    'Score': score
                })
        
        df = pd.DataFrame(data)
        
        # Create pivot table for heatmap
        pivot_table = df.pivot_table(
            values='Score', 
            index=param1, 
            columns=param2, 
            aggfunc='mean'
        )
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_table.values,
            x=pivot_table.columns,
            y=pivot_table.index,
            colorscale='Viridis',
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=f'Parameter Heatmap: {param1} vs {param2}',
            xaxis_title=param2,
            yaxis_title=param1,
            template='plotly_white'
        )
        
        return fig


class BenchmarkVisualizer:
    """Visualization tools for benchmark results."""
    
    @staticmethod
    def create_leaderboard_plot(leaderboard_df: pd.DataFrame,
                               metric: str = 'Mean Accuracy',
                               save_path: Optional[str] = None) -> go.Figure:
        """Create interactive leaderboard plot.
        
        Args:
            leaderboard_df: Leaderboard DataFrame
            metric: Metric to plot
            save_path: Optional path to save plot
            
        Returns:
            Plotly figure
        """
        # Convert string metrics to float for plotting
        df_plot = leaderboard_df.copy()
        df_plot[metric] = df_plot[metric].astype(float)
        
        fig = px.bar(
            df_plot, 
            x=metric, 
            y='Algorithm',
            orientation='h',
            title=f'Algorithm Performance - {metric}',
            color=metric,
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            template='plotly_white',
            height=400,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    @staticmethod
    def create_comparison_radar(results: Dict[str, Dict[str, Any]]) -> go.Figure:
        """Create radar chart comparing algorithms.
        
        Args:
            results: Benchmark results
            
        Returns:
            Plotly figure
        """
        # Extract metrics for each algorithm
        algorithms = list(results.keys())
        metrics = ['mean_accuracy', 'mean_time']
        
        fig = go.Figure()
        
        for algo in algorithms:
            stats = results[algo]['overall_stats']
            values = [stats[metric] for metric in metrics]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics,
                fill='toself',
                name=algo
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Algorithm Comparison Radar Chart"
        )
        
        return fig
