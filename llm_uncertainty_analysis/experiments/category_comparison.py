"""
Category Comparison Experiment

This module runs the main experiment comparing uncertainty across categories.
"""

import pandas as pd
from scipy import stats
from typing import List, Dict


def run_category_comparison(analyzer, all_samples: List[Dict]) -> pd.DataFrame:
    """
    Run the main category comparison experiment.

    Args:
        analyzer: UncertaintyAnalyzer instance
        all_samples: List of all samples to analyze

    Returns:
        DataFrame with results
    """
    print("\n" + "="*70)
    print("RUNNING CATEGORY COMPARISON EXPERIMENT")
    print("="*70)

    # Analyze all samples
    results_df = analyzer.analyze_dataset(all_samples)

    # Calculate statistics by category
    category_stats = results_df.groupby('category').agg({
        'mean_entropy': ['mean', 'std', 'count'],
        'mean_surprisal': ['mean', 'std'],
        'mean_perplexity': ['mean', 'std']
    }).round(3)

    print("\nResults by context category:")
    print(category_stats)

    # Statistical tests
    print("\n" + "="*70)
    print("STATISTICAL TESTS")
    print("="*70)

    # Get entropy values by category
    categories = results_df['category'].unique()
    entropy_by_category = {cat: results_df[results_df['category'] == cat]['mean_entropy'].values
                           for cat in categories}

    # ANOVA
    f_stat, p_value = stats.f_oneway(*entropy_by_category.values())
    print(f"\nOne-way ANOVA:")
    print(f"  F-statistic: {f_stat:.3f}")
    print(f"  p-value: {p_value:.3e}")
    print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}")

    # Pairwise t-tests
    print(f"\nPairwise t-tests:")
    for i, cat1 in enumerate(categories):
        for cat2 in categories[i+1:]:
            t_stat, p_val = stats.ttest_ind(entropy_by_category[cat1],
                                           entropy_by_category[cat2])
            print(f"  {cat1} vs {cat2}: t={t_stat:.3f}, p={p_val:.3e}")

    return results_df


def calculate_descriptive_stats(results_df: pd.DataFrame) -> Dict:
    """
    Calculate descriptive statistics for the results.

    Args:
        results_df: DataFrame with experimental results

    Returns:
        Dictionary with descriptive statistics
    """
    stats_dict = {}

    for category in results_df['category'].unique():
        cat_data = results_df[results_df['category'] == category]

        stats_dict[category] = {
            'n_samples': len(cat_data),
            'mean_entropy': cat_data['mean_entropy'].mean(),
            'std_entropy': cat_data['mean_entropy'].std(),
            'median_entropy': cat_data['mean_entropy'].median(),
            'min_entropy': cat_data['mean_entropy'].min(),
            'max_entropy': cat_data['mean_entropy'].max()
        }

    return stats_dict
