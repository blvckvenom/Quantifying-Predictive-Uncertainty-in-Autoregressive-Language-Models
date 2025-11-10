"""
Multi-Model In-Context Learning Experiment

This module implements comprehensive ICL experiments across multiple models and categories.

Key features:
- Tests 3 models: distilgpt2, gpt2, gpt2-medium
- Tests 3 categories: factual, logical, creative
- Tests k-shot configurations: [0, 1, 2, 3, 5]
- Computes entropy trajectories, ΔH, and statistical metrics
- Sequential model loading with VRAM cleanup

Usage:
    from llm_uncertainty_analysis.experiments.multi_model_icl_experiment import (
        run_multi_model_icl_experiment
    )

    results = run_multi_model_icl_experiment(
        model_ids=['distilgpt2', 'gpt2', 'gpt2-medium'],
        n_examples_range=[0, 1, 2, 3, 5],
        n_queries_per_config=10,
        device='cuda'
    )
"""

import torch
import numpy as np
from typing import List, Dict, Tuple
from tqdm import tqdm

from ..analysis import UncertaintyAnalyzer
from ..icl import measure_icl_entropy, generate_icl_prompt
from .icl_category_configs import CATEGORY_CONFIGS, get_examples, get_queries


def run_single_model_icl_experiment(
    model_id: str,
    categories_config: Dict[str, Dict],
    n_examples_range: List[int],
    n_queries_per_config: int,
    device: str
) -> Dict:
    """
    Run ICL experiments for a single model across all categories.

    Args:
        model_id: HuggingFace model identifier (e.g., 'gpt2')
        categories_config: Dictionary of category configurations
        n_examples_range: List of k values for k-shot learning
        n_queries_per_config: Number of queries to test per configuration
        device: Computation device ('cuda' or 'cpu')

    Returns:
        Dictionary with structure:
        {
            'model_id': str,
            'categories': {
                'factual': {
                    'n_examples': [0,1,2,3,5],
                    'mean_entropy': [...],
                    'std_entropy': [...],
                    'delta_h': [...],  # Reduction from 0-shot
                    'percent_reduction': [...],
                    'per_query_results': [...]
                },
                'logical': {...},
                'creative': {...}
            }
        }
    """
    print(f"\n{'='*70}")
    print(f"MODEL: {model_id}")
    print(f"{'='*70}")

    # Load model
    print(f"Loading model '{model_id}'...")
    analyzer = UncertaintyAnalyzer(model_id, device=device)
    model = analyzer.model
    tokenizer = analyzer.tokenizer

    results = {
        'model_id': model_id,
        'categories': {}
    }

    # Iterate over categories
    for category_name, category_config in categories_config.items():
        print(f"\n--- Category: {category_name.upper()} ---")

        task_description = category_config['task_description']
        examples = category_config['examples']
        all_queries = category_config['queries'][:n_queries_per_config]

        # Storage for this category
        category_results = {
            'n_examples': [],
            'mean_entropy': [],
            'std_entropy': [],
            'delta_h': [],
            'percent_reduction': [],
            'per_query_results': []
        }

        # Iterate over k-shot configurations
        for k in n_examples_range:
            print(f"  Testing {k}-shot...", end=" ")

            # Measure entropy for each query
            entropies = []
            for query in all_queries:
                prompt = generate_icl_prompt(
                    task_description, examples, query, n_examples=k
                )
                entropy = measure_icl_entropy(model, tokenizer, prompt, device=device)
                entropies.append(entropy)

            # Calculate statistics
            mean_h = np.mean(entropies)
            std_h = np.std(entropies)

            category_results['n_examples'].append(k)
            category_results['mean_entropy'].append(mean_h)
            category_results['std_entropy'].append(std_h)
            category_results['per_query_results'].append(entropies.copy())

            print(f"H = {mean_h:.3f} +/- {std_h:.3f} bits")

        # Calculate ΔH from baseline (0-shot)
        baseline_entropy = category_results['mean_entropy'][0]
        for i, mean_h in enumerate(category_results['mean_entropy']):
            delta_h = baseline_entropy - mean_h
            percent_red = (delta_h / baseline_entropy * 100) if baseline_entropy > 0 else 0.0

            category_results['delta_h'].append(delta_h)
            category_results['percent_reduction'].append(percent_red)

        results['categories'][category_name] = category_results

    # Clean up model to free VRAM
    del analyzer, model, tokenizer
    if device == 'cuda':
        torch.cuda.empty_cache()

    return results


def run_multi_model_icl_experiment(
    model_ids: List[str] = None,
    n_examples_range: List[int] = None,
    n_queries_per_config: int = 10,
    device: str = 'cuda'
) -> Dict:
    """
    Run comprehensive ICL experiments across multiple models and categories.

    Args:
        model_ids: List of model IDs to test (default: COMPARISON_MODELS)
        n_examples_range: List of k values (default: [0,1,2,3,5])
        n_queries_per_config: Number of test queries per configuration
        device: Computation device

    Returns:
        Dictionary with structure:
        {
            'metadata': {...},
            'results_by_model': {
                'distilgpt2': {...},
                'gpt2': {...},
                'gpt2-medium': {...}
            },
            'comparison': {
                'scaling_summary': {...},
                'consistency_summary': {...}
            }
        }
    """
    # Default parameters
    if model_ids is None:
        from ..models.model_config import COMPARISON_MODELS
        model_ids = COMPARISON_MODELS

    if n_examples_range is None:
        n_examples_range = [0, 1, 2, 3, 5]

    print("\n" + "="*70)
    print("MULTI-MODEL IN-CONTEXT LEARNING EXPERIMENT")
    print("="*70)
    print(f"Models: {model_ids}")
    print(f"Categories: {list(CATEGORY_CONFIGS.keys())}")
    print(f"k-shot range: {n_examples_range}")
    print(f"Queries per config: {n_queries_per_config}")
    print(f"Device: {device}")
    print("="*70)

    # Storage for all results
    all_results = {
        'metadata': {
            'model_ids': model_ids,
            'n_examples_range': n_examples_range,
            'n_queries_per_config': n_queries_per_config,
            'device': device,
            'categories': list(CATEGORY_CONFIGS.keys()),
            # Dataset information (if using real datasets)
            'datasets': {
                category: {
                    'source': CATEGORY_CONFIGS[category].get('metadata', {}).get('source', 'synthetic'),
                    'n_examples': len(CATEGORY_CONFIGS[category]['examples']),
                    'n_queries': len(CATEGORY_CONFIGS[category]['queries']),
                    'total_prompts': len(CATEGORY_CONFIGS[category]['examples']) + len(CATEGORY_CONFIGS[category]['queries'])
                }
                for category in CATEGORY_CONFIGS.keys()
            }
        },
        'results_by_model': {}
    }

    # Run experiments sequentially for each model
    for model_id in model_ids:
        results = run_single_model_icl_experiment(
            model_id=model_id,
            categories_config=CATEGORY_CONFIGS,
            n_examples_range=n_examples_range,
            n_queries_per_config=n_queries_per_config,
            device=device
        )
        all_results['results_by_model'][model_id] = results

    # Generate comparison summaries
    all_results['comparison'] = generate_comparison_summary(all_results['results_by_model'])

    print("\n" + "="*70)
    print("MULTI-MODEL EXPERIMENT COMPLETED")
    print("="*70)

    return all_results


def generate_comparison_summary(results_by_model: Dict) -> Dict:
    """
    Generate summary comparing results across models.

    Args:
        results_by_model: Dictionary of results indexed by model_id

    Returns:
        Dictionary with scaling and consistency summaries
    """
    from ..models.model_config import get_model_params_numeric

    summary = {
        'scaling_summary': {},
        'consistency_summary': {}
    }

    # Extract model IDs and categories
    model_ids = list(results_by_model.keys())
    categories = list(results_by_model[model_ids[0]]['categories'].keys())

    # -------------------------------------------------------------------------
    # SCALING SUMMARY: ΔH vs model size
    # -------------------------------------------------------------------------
    scaling_data = {}

    for category in categories:
        scaling_data[category] = {
            'model_ids': [],
            'model_params': [],
            'delta_h_5shot': [],  # ΔH at 5-shot
            'baseline_entropy': []
        }

        for model_id in model_ids:
            cat_results = results_by_model[model_id]['categories'][category]

            # Get 5-shot ΔH (last value in delta_h list)
            delta_h_5shot = cat_results['delta_h'][-1]
            baseline_h = cat_results['mean_entropy'][0]

            scaling_data[category]['model_ids'].append(model_id)
            scaling_data[category]['model_params'].append(get_model_params_numeric(model_id))
            scaling_data[category]['delta_h_5shot'].append(delta_h_5shot)
            scaling_data[category]['baseline_entropy'].append(baseline_h)

    summary['scaling_summary'] = scaling_data

    # -------------------------------------------------------------------------
    # CONSISTENCY SUMMARY: Ranking of categories by ΔH for each model
    # -------------------------------------------------------------------------
    consistency_data = {}

    for model_id in model_ids:
        # Get ΔH at 5-shot for each category
        category_delta_h = {}
        for category in categories:
            delta_h_5shot = results_by_model[model_id]['categories'][category]['delta_h'][-1]
            category_delta_h[category] = delta_h_5shot

        # Rank categories by ΔH (descending)
        ranked_categories = sorted(category_delta_h.items(), key=lambda x: x[1], reverse=True)

        consistency_data[model_id] = {
            'category_delta_h': category_delta_h,
            'ranking': [cat for cat, _ in ranked_categories],
            'ranking_values': [val for _, val in ranked_categories]
        }

    summary['consistency_summary'] = consistency_data

    return summary


def print_results_summary(results: Dict):
    """
    Print human-readable summary of experiment results.

    Args:
        results: Output from run_multi_model_icl_experiment()
    """
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    # Per-model summary
    for model_id, model_results in results['results_by_model'].items():
        print(f"\n--- {model_id.upper()} ---")

        for category, cat_results in model_results['categories'].items():
            n_examples = cat_results['n_examples']
            delta_h = cat_results['delta_h']
            percent_red = cat_results['percent_reduction']

            print(f"\n  {category.capitalize()}:")
            for i, k in enumerate(n_examples):
                print(f"    {k}-shot: DeltaH = {delta_h[i]:+.3f} bits ({percent_red[i]:+.1f}%)")

    # Scaling summary
    print("\n" + "="*70)
    print("SCALING ANALYSIS (DeltaH at 5-shot)")
    print("="*70)

    scaling = results['comparison']['scaling_summary']
    for category, data in scaling.items():
        print(f"\n{category.capitalize()}:")
        for i, model_id in enumerate(data['model_ids']):
            params_m = data['model_params'][i] / 1e6
            delta_h = data['delta_h_5shot'][i]
            print(f"  {model_id} ({params_m:.0f}M): DeltaH = {delta_h:.3f} bits")

    # Consistency summary
    print("\n" + "="*70)
    print("CONSISTENCY ANALYSIS (Ranking by DeltaH)")
    print("="*70)

    consistency = results['comparison']['consistency_summary']
    for model_id, data in consistency.items():
        print(f"\n{model_id}: {' > '.join(data['ranking'])}")

    print("\n" + "="*70)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_category_baseline_entropy(results: Dict, model_id: str, category: str) -> float:
    """Extract baseline (0-shot) entropy for a model-category pair."""
    return results['results_by_model'][model_id]['categories'][category]['mean_entropy'][0]


def get_category_delta_h(results: Dict, model_id: str, category: str, k_shot: int) -> float:
    """Extract ΔH for a model-category pair at k-shot."""
    cat_results = results['results_by_model'][model_id]['categories'][category]
    n_examples = cat_results['n_examples']

    if k_shot in n_examples:
        idx = n_examples.index(k_shot)
        return cat_results['delta_h'][idx]
    else:
        raise ValueError(f"k_shot={k_shot} not in n_examples={n_examples}")


if __name__ == "__main__":
    # Example usage
    from ..config import device, setup_reproducibility

    setup_reproducibility()

    # Run comprehensive experiment
    results = run_multi_model_icl_experiment(
        model_ids=['distilgpt2', 'gpt2'],  # Test with 2 models first
        n_examples_range=[0, 1, 3],  # Smaller range for testing
        n_queries_per_config=3,  # Few queries for testing
        device=str(device)
    )

    # Print summary
    print_results_summary(results)
