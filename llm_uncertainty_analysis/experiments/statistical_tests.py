"""
Statistical Tests Runner

This module runs complete statistical analysis including ANOVA and post-hoc tests.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from ..statistics.anova import (
    run_anova,
    calculate_eta_squared,
    run_tukey_hsd,
    run_bonferroni_correction,
    print_anova_summary
)
from ..statistics.effect_size import calculate_cohens_d, interpret_cohens_d


def run_complete_statistical_analysis(results_df: pd.DataFrame) -> Dict:
    """
    Run complete statistical analysis on results.

    Args:
        results_df: DataFrame with analysis results

    Returns:
        Dictionary with all statistical test results
    """
    print("\n" + "="*70)
    print("COMPLETE STATISTICAL ANALYSIS")
    print("="*70)

    # Prepare data by category
    categories = results_df['category'].unique()
    data_by_category = {
        cat: results_df[results_df['category'] == cat]['mean_entropy'].values
        for cat in categories
    }

    # 1. ANOVA
    print("\n1. ONE-WAY ANOVA")
    print("-"*70)
    f_stat, p_value = run_anova(data_by_category)
    eta_squared = calculate_eta_squared(data_by_category)

    print_anova_summary(f_stat, p_value, eta_squared, data_by_category)

    # 2. Tukey HSD
    print("\n" + "="*70)
    print("2. TUKEY HSD POST-HOC TEST")
    print("-"*70)

    # Prepare data for Tukey
    all_values = []
    all_labels = []
    for cat, data in data_by_category.items():
        all_values.extend(data)
        all_labels.extend([cat] * len(data))

    tukey_df = run_tukey_hsd(all_values, all_labels)
    print(tukey_df)

    # 3. Bonferroni correction
    print("\n" + "="*70)
    print("3. BONFERRONI CORRECTION")
    print("-"*70)

    bonferroni_results = run_bonferroni_correction(data_by_category)

    print(f"\nAlpha level: 0.05")
    print(f"Bonferroni-corrected alpha: {bonferroni_results[0]['alpha_bonferroni']:.4f}")
    print(f"\nResults:")

    for result in bonferroni_results:
        sig_marker = "✓" if result['significant_bonferroni'] else "✗"
        print(f"\n{sig_marker} {result['comparison']}:")
        print(f"    t = {result['t_statistic']:.4f}, p = {result['p_value']:.6f}")
        print(f"    Mean diff = {result['mean_diff']:.4f} bits")

    # 4. Cohen's d for all pairs
    print("\n" + "="*70)
    print("4. EFFECT SIZES (COHEN'S d)")
    print("-"*70)

    cohens_d_results = []
    from itertools import combinations

    for cat1, cat2 in combinations(categories, 2):
        d = calculate_cohens_d(data_by_category[cat1], data_by_category[cat2])
        magnitude, description = interpret_cohens_d(d)

        cohens_d_results.append({
            'comparison': f"{cat1} vs {cat2}",
            'cohens_d': d,
            'magnitude': magnitude,
            'description': description
        })

        print(f"\n{cat1.upper()} vs {cat2.upper()}:")
        print(f"  Cohen's d = {d:.4f}")
        print(f"  Magnitude: {magnitude}")
        print(f"  Interpretation: {description}")

    # Return all results
    return {
        'anova': {
            'f_statistic': f_stat,
            'p_value': p_value,
            'eta_squared': eta_squared
        },
        'tukey_hsd': tukey_df,
        'bonferroni': bonferroni_results,
        'cohens_d': cohens_d_results,
        'data_by_category': data_by_category
    }
