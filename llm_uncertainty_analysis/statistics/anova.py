"""
ANOVA Analysis

This module contains functions for ANOVA and post-hoc tests.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from itertools import combinations
from typing import Dict, List, Tuple


def run_anova(data_by_category: Dict[str, np.ndarray]) -> Tuple[float, float]:
    """
    Run one-way ANOVA to test for differences between groups.

    Args:
        data_by_category: Dictionary mapping category names to arrays of values

    Returns:
        Tuple of (F-statistic, p-value)
    """
    groups = list(data_by_category.values())
    f_stat, p_value = f_oneway(*groups)
    return f_stat, p_value


def calculate_eta_squared(data_by_category: Dict[str, np.ndarray]) -> float:
    """
    Calculate eta-squared (η²) effect size for ANOVA.

    Args:
        data_by_category: Dictionary mapping category names to arrays of values

    Returns:
        Eta-squared value (proportion of variance explained)
    """
    # Grand mean
    all_data = np.concatenate(list(data_by_category.values()))
    grand_mean = np.mean(all_data)

    # SS between
    ss_between = sum([
        len(data) * (np.mean(data) - grand_mean)**2
        for data in data_by_category.values()
    ])

    # SS total
    ss_total = np.sum((all_data - grand_mean)**2)

    # Eta squared
    eta_squared = ss_between / ss_total
    return eta_squared


def run_tukey_hsd(all_values: List[float],
                  all_labels: List[str],
                  alpha: float = 0.05) -> pd.DataFrame:
    """
    Run Tukey HSD post-hoc test.

    Args:
        all_values: List of all data values
        all_labels: List of group labels corresponding to values
        alpha: Significance level

    Returns:
        DataFrame with Tukey HSD results
    """
    tukey_result = pairwise_tukeyhsd(
        endog=all_values,
        groups=all_labels,
        alpha=alpha
    )

    # Convert to DataFrame
    tukey_df = pd.DataFrame(
        data=tukey_result.summary().data[1:],
        columns=tukey_result.summary().data[0]
    )

    return tukey_df


def run_bonferroni_correction(data_by_category: Dict[str, np.ndarray],
                               alpha: float = 0.05) -> List[Dict]:
    """
    Run pairwise t-tests with Bonferroni correction.

    Args:
        data_by_category: Dictionary mapping category names to arrays
        alpha: Original significance level

    Returns:
        List of dictionaries with test results
    """
    categories = list(data_by_category.keys())
    pairwise_comparisons = list(combinations(categories, 2))
    n_comparisons = len(pairwise_comparisons)
    alpha_bonferroni = alpha / n_comparisons

    results = []

    for cat1, cat2 in pairwise_comparisons:
        t_stat, p_value = stats.ttest_ind(
            data_by_category[cat1],
            data_by_category[cat2]
        )

        results.append({
            'comparison': f"{cat1} vs {cat2}",
            'cat1': cat1,
            'cat2': cat2,
            'mean1': np.mean(data_by_category[cat1]),
            'mean2': np.mean(data_by_category[cat2]),
            'mean_diff': np.mean(data_by_category[cat1]) - np.mean(data_by_category[cat2]),
            't_statistic': t_stat,
            'p_value': p_value,
            'alpha_bonferroni': alpha_bonferroni,
            'significant_bonferroni': p_value < alpha_bonferroni
        })

    return results


def print_anova_summary(f_stat: float,
                        p_value: float,
                        eta_squared: float,
                        data_by_category: Dict[str, np.ndarray]):
    """
    Print formatted ANOVA summary.

    Args:
        f_stat: F-statistic
        p_value: P-value
        eta_squared: Effect size
        data_by_category: Dictionary with data
    """
    print("="*70)
    print("ANOVA SUMMARY")
    print("="*70)

    # Descriptive statistics
    print("\nDescriptive Statistics:")
    print("-"*70)
    for cat, data in data_by_category.items():
        print(f"\n{cat.upper()}:")
        print(f"  n = {len(data)}")
        print(f"  Mean = {np.mean(data):.4f} bits")
        print(f"  Std = {np.std(data, ddof=1):.4f} bits")
        print(f"  Min = {np.min(data):.4f}, Max = {np.max(data):.4f}")

    # ANOVA results
    print("\n" + "="*70)
    print("ONE-WAY ANOVA RESULTS:")
    print("-"*70)
    print(f"\nF-statistic: {f_stat:.4f}")
    print(f"P-value: {p_value:.6f}")
    print(f"Eta-squared (η²): {eta_squared:.4f}")

    # Interpretation
    print("\nInterpretation:")
    if p_value < 0.001:
        print("  *** HIGHLY SIGNIFICANT (p < 0.001)")
    elif p_value < 0.01:
        print("  ** VERY SIGNIFICANT (p < 0.01)")
    elif p_value < 0.05:
        print("  * SIGNIFICANT (p < 0.05)")
    else:
        print("  NOT SIGNIFICANT (p >= 0.05)")

    # Effect size interpretation
    if eta_squared < 0.01:
        effect = "negligible"
    elif eta_squared < 0.06:
        effect = "small"
    elif eta_squared < 0.14:
        effect = "medium"
    else:
        effect = "large"

    print(f"\nEffect size: {effect}")
    print(f"  ({eta_squared*100:.2f}% of variance explained by category)")
