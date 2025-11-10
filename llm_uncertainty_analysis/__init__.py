"""
LLM Uncertainty Analysis

A comprehensive framework for quantifying predictive uncertainty in autoregressive language models.

This package provides tools for:
- Calculating uncertainty metrics (entropy, surprisal, perplexity)
- Analyzing uncertainty across different context categories
- Running In-Context Learning experiments
- Statistical analysis and visualization

Authors: Benito Fuentes y Sebastian Vergara
Guide: Simón Vidal
Date: November 2025
Universidad de Chile - EL7024-1
"""

__version__ = "1.0.0"
__author__ = "Benito Fuentes, Sebastian Vergara"

from .config import setup_reproducibility, setup_visualization, device
from .models import ModelConfig, ContextCategory, CONTEXT_CATEGORIES, get_available_models
from .data_management import RealDatasetManager
from .metrics import calculate_entropy, calculate_surprisal, calculate_perplexity
from .analysis import UncertaintyAnalyzer
from .statistics import calculate_cohens_d, calculate_mutual_information
from .icl import generate_icl_prompt, measure_icl_entropy
from .visualization import plot_entropy_by_category, plot_icl_analysis, plot_cohens_d_comparison
from .experiments import run_category_comparison, run_icl_experiment

__all__ = [
    # Configuration
    'setup_reproducibility',
    'setup_visualization',
    'device',

    # Models
    'ModelConfig',
    'ContextCategory',
    'CONTEXT_CATEGORIES',
    'get_available_models',

    # Data Management
    'RealDatasetManager',

    # Metrics
    'calculate_entropy',
    'calculate_surprisal',
    'calculate_perplexity',

    # Analysis
    'UncertaintyAnalyzer',

    # Statistics
    'calculate_cohens_d',
    'calculate_mutual_information',

    # ICL
    'generate_icl_prompt',
    'measure_icl_entropy',

    # Visualization
    'plot_entropy_by_category',
    'plot_icl_analysis',
    'plot_cohens_d_comparison',

    # Experiments
    'run_category_comparison',
    'run_icl_experiment',
]
