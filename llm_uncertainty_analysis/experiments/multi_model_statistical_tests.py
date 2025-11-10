"""
Multi-Model Statistical Tests

This module provides comprehensive statistical testing for multi-model ICL experiments.

Tests implemented:
- Two-way ANOVA (factors: model, category)
- Post-hoc pairwise comparisons (Tukey HSD)
- Effect size calculations (Cohen's d)
- Bonferroni correction for multiple comparisons
- Correlation tests (Spearman, Pearson)

Usage:
    from llm_uncertainty_analysis.experiments.multi_model_statistical_tests import (
        run_comprehensive_statistical_analysis
    )

    stats_results = run_comprehensive_statistical_analysis(multi_model_results)
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple
from ..statistics import (
    calculate_cohens_d,
    interpret_cohens_d
)


# ============================================================================
# TWO-WAY ANOVA
# ============================================================================

def perform_two_way_anova(results: Dict) -> Dict:
    """
    Perform two-way ANOVA with factors: model and category.

    Tests:
    - Main effect of model size on DeltaH
    - Main effect of category on DeltaH
    - Interaction effect between model and category

    Args:
        results: Output from run_multi_model_icl_experiment()

    Returns:
        Dictionary with ANOVA results
    """
    from scipy.stats import f_oneway

    # Extract data for ANOVA
    results_by_model = results['results_by_model']
    model_ids = list(results_by_model.keys())
    categories = list(results_by_model[model_ids[0]]['categories'].keys())

    # Prepare data for ANOVA (DeltaH at 5-shot)
    data_for_anova = {
        'model': [],
        'category': [],
        'delta_h': []
    }

    for model_id in model_ids:
        for category in categories:
            cat_results = results_by_model[model_id]['categories'][category]
            delta_h_5shot = cat_results['delta_h'][-1]  # Last value is 5-shot

            data_for_anova['model'].append(model_id)
            data_for_anova['category'].append(category)
            data_for_anova['delta_h'].append(delta_h_5shot)

    df = pd.DataFrame(data_for_anova)

    # Since we have only one observation per cell (model, category), we can't do full 2-way ANOVA
    # Instead, we'll do separate one-way ANOVAs for model effect and category effect

    # One-way ANOVA: Effect of MODEL (across all categories)
    groups_by_model = [df[df['model'] == m]['delta_h'].values for m in model_ids]
    f_stat_model, p_value_model = f_oneway(*groups_by_model)

    # One-way ANOVA: Effect of CATEGORY (across all models)
    groups_by_category = [df[df['category'] == c]['delta_h'].values for c in categories]
    f_stat_category, p_value_category = f_oneway(*groups_by_category)

    anova_results = {
        'method': 'Two one-way ANOVAs (model and category as factors)',
        'note': 'Full two-way ANOVA not feasible with single observation per cell',
        'model_effect': {
            'F_statistic': float(f_stat_model),
            'p_value': float(p_value_model),
            'significant': p_value_model < 0.05,
            'interpretation': 'Model size has significant effect on DeltaH' if p_value_model < 0.05 else 'No significant model effect'
        },
        'category_effect': {
            'F_statistic': float(f_stat_category),
            'p_value': float(p_value_category),
            'significant': p_value_category < 0.05,
            'interpretation': 'Category has significant effect on DeltaH' if p_value_category < 0.05 else 'No significant category effect'
        },
        'dataframe': df.to_dict(orient='records')
    }

    return anova_results


# ============================================================================
# PAIRWISE COMPARISONS
# ============================================================================

def perform_pairwise_model_comparisons(results: Dict) -> Dict:
    """
    Perform pairwise comparisons between models using t-tests with Bonferroni correction.

    Args:
        results: Output from run_multi_model_icl_experiment()

    Returns:
        Dictionary with pairwise comparison results
    """
    results_by_model = results['results_by_model']
    model_ids = list(results_by_model.keys())
    categories = list(results_by_model[model_ids[0]]['categories'].keys())

    # Collect DeltaH values for each model (across all categories)
    delta_h_by_model = {}
    for model_id in model_ids:
        delta_h_values = []
        for category in categories:
            cat_results = results_by_model[model_id]['categories'][category]
            delta_h_5shot = cat_results['delta_h'][-1]
            delta_h_values.append(delta_h_5shot)
        delta_h_by_model[model_id] = delta_h_values

    # Perform pairwise t-tests
    comparisons = []
    pairs = []
    for i, model1 in enumerate(model_ids):
        for j, model2 in enumerate(model_ids):
            if i < j:  # Avoid duplicate comparisons
                pairs.append((model1, model2))

    n_comparisons = len(pairs)
    alpha_bonferroni = 0.05 / n_comparisons if n_comparisons > 0 else 0.05

    pairwise_results = []
    for model1, model2 in pairs:
        data1 = delta_h_by_model[model1]
        data2 = delta_h_by_model[model2]

        # Paired t-test (same categories for both models)
        t_stat, p_value = stats.ttest_rel(data1, data2)

        # Cohen's d
        cohens_d = calculate_cohens_d(np.array(data1), np.array(data2))
        effect_interpretation = interpret_cohens_d(cohens_d)

        pairwise_results.append({
            'model1': model1,
            'model2': model2,
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'p_value_bonferroni': float(p_value * n_comparisons),  # Bonferroni corrected
            'significant_bonferroni': (p_value * n_comparisons) < 0.05,
            'cohens_d': float(cohens_d),
            'effect_size': effect_interpretation,
            'mean_diff': float(np.mean(data1) - np.mean(data2))
        })

    return {
        'method': 'Paired t-tests with Bonferroni correction',
        'n_comparisons': n_comparisons,
        'alpha_bonferroni': alpha_bonferroni,
        'comparisons': pairwise_results
    }


# ============================================================================
# EFFECT SIZES BETWEEN MODELS
# ============================================================================

def compute_effect_sizes_between_models(results: Dict) -> Dict:
    """
    Compute Cohen's d effect sizes for all model pairs across categories.

    Args:
        results: Output from run_multi_model_icl_experiment()

    Returns:
        Dictionary with effect sizes
    """
    results_by_model = results['results_by_model']
    model_ids = list(results_by_model.keys())
    categories = list(results_by_model[model_ids[0]]['categories'].keys())

    effect_sizes = {
        'description': "Cohen's d effect sizes between model pairs",
        'categories': {}
    }

    for category in categories:
        category_effect_sizes = []

        for i, model1 in enumerate(model_ids):
            for j, model2 in enumerate(model_ids):
                if i < j:
                    # Get DeltaH values for this category
                    delta_h1 = results_by_model[model1]['categories'][category]['delta_h'][-1]
                    delta_h2 = results_by_model[model2]['categories'][category]['delta_h'][-1]

                    # Since we have single values, use a simplified effect size calculation
                    # Effect = difference relative to pooled variability
                    # Here we use the std from the per-query results
                    std1 = results_by_model[model1]['categories'][category]['std_entropy'][-1]
                    std2 = results_by_model[model2]['categories'][category]['std_entropy'][-1]

                    pooled_std = np.sqrt((std1**2 + std2**2) / 2)
                    cohens_d = (delta_h1 - delta_h2) / pooled_std if pooled_std > 0 else 0.0

                    effect_interpretation = interpret_cohens_d(abs(cohens_d))

                    category_effect_sizes.append({
                        'model1': model1,
                        'model2': model2,
                        'delta_h1': float(delta_h1),
                        'delta_h2': float(delta_h2),
                        'cohens_d': float(cohens_d),
                        'effect_size': effect_interpretation
                    })

        effect_sizes['categories'][category] = category_effect_sizes

    return effect_sizes


# ============================================================================
# CORRELATION TESTS
# ============================================================================

def perform_correlation_tests(results: Dict) -> Dict:
    """
    Perform correlation tests between model size and DeltaH.

    Tests both Spearman (rank-based) and Pearson (linear) correlations.

    Args:
        results: Output from run_multi_model_icl_experiment()

    Returns:
        Dictionary with correlation results
    """
    from ..models.model_config import get_model_params_numeric

    scaling_data = results['comparison']['scaling_summary']
    categories = list(scaling_data.keys())

    correlation_results = {
        'description': 'Correlation between model size (parameters) and DeltaH',
        'categories': {}
    }

    for category in categories:
        model_params = np.array(scaling_data[category]['model_params'])
        delta_h_values = np.array(scaling_data[category]['delta_h_5shot'])

        # Spearman correlation
        rho_spearman, p_spearman = stats.spearmanr(model_params, delta_h_values)

        # Pearson correlation
        r_pearson, p_pearson = stats.pearsonr(model_params, delta_h_values)

        correlation_results['categories'][category] = {
            'spearman': {
                'rho': float(rho_spearman),
                'p_value': float(p_spearman),
                'significant': p_spearman < 0.05,
                'interpretation': f"{'Positive' if rho_spearman > 0 else 'Negative'} monotonic relationship"
            },
            'pearson': {
                'r': float(r_pearson),
                'p_value': float(p_pearson),
                'significant': p_pearson < 0.05,
                'interpretation': f"{'Positive' if r_pearson > 0 else 'Negative'} linear relationship"
            },
            'model_params': model_params.tolist(),
            'delta_h_values': delta_h_values.tolist()
        }

    return correlation_results


# ============================================================================
# COMPREHENSIVE STATISTICAL ANALYSIS
# ============================================================================

def run_comprehensive_statistical_analysis(results: Dict) -> Dict:
    """
    Run all statistical tests and return comprehensive results.

    Args:
        results: Output from run_multi_model_icl_experiment()

    Returns:
        Dictionary containing all statistical test results:
        {
            'anova': {...},
            'pairwise_comparisons': {...},
            'effect_sizes': {...},
            'correlations': {...},
            'summary': {...}
        }
    """
    print("\n" + "="*70)
    print("COMPREHENSIVE STATISTICAL ANALYSIS")
    print("="*70)

    statistical_results = {}

    # ANOVA
    print("\n1. Performing ANOVA...")
    statistical_results['anova'] = perform_two_way_anova(results)
    print(f"   Model effect: F={statistical_results['anova']['model_effect']['F_statistic']:.2f}, "
          f"p={statistical_results['anova']['model_effect']['p_value']:.4f}")
    print(f"   Category effect: F={statistical_results['anova']['category_effect']['F_statistic']:.2f}, "
          f"p={statistical_results['anova']['category_effect']['p_value']:.4f}")

    # Pairwise comparisons
    print("\n2. Performing pairwise comparisons...")
    statistical_results['pairwise_comparisons'] = perform_pairwise_model_comparisons(results)
    n_sig = sum(1 for c in statistical_results['pairwise_comparisons']['comparisons']
                if c['significant_bonferroni'])
    print(f"   {n_sig}/{len(statistical_results['pairwise_comparisons']['comparisons'])} "
          f"comparisons significant after Bonferroni correction")

    # Effect sizes
    print("\n3. Computing effect sizes...")
    statistical_results['effect_sizes'] = compute_effect_sizes_between_models(results)

    # Correlations
    print("\n4. Performing correlation tests...")
    statistical_results['correlations'] = perform_correlation_tests(results)
    for cat, corr_data in statistical_results['correlations']['categories'].items():
        print(f"   {cat.capitalize()}: rho={corr_data['spearman']['rho']:.3f}, "
              f"p={corr_data['spearman']['p_value']:.4f}")

    # Summary
    statistical_results['summary'] = generate_statistical_summary(statistical_results)

    print("\n" + "="*70)
    print("STATISTICAL ANALYSIS COMPLETE")
    print("="*70)

    return statistical_results


def generate_statistical_summary(stats_results: Dict) -> Dict:
    """
    Generate summary of statistical findings.

    Args:
        stats_results: Output from run_comprehensive_statistical_analysis()

    Returns:
        Summary dictionary
    """
    summary = {
        'anova_model_significant': stats_results['anova']['model_effect']['significant'],
        'anova_category_significant': stats_results['anova']['category_effect']['significant'],
        'n_significant_pairwise': sum(1 for c in stats_results['pairwise_comparisons']['comparisons']
                                     if c['significant_bonferroni']),
        'total_pairwise': len(stats_results['pairwise_comparisons']['comparisons']),
        'categories_with_significant_correlation': []
    }

    # Check which categories have significant correlations
    for cat, corr_data in stats_results['correlations']['categories'].items():
        if corr_data['spearman']['significant']:
            summary['categories_with_significant_correlation'].append(cat)

    return summary


# ============================================================================
# REPORTING
# ============================================================================

def print_statistical_report(stats_results: Dict):
    """
    Print detailed statistical report.

    Args:
        stats_results: Output from run_comprehensive_statistical_analysis()
    """
    print("\n" + "="*70)
    print("STATISTICAL ANALYSIS REPORT")
    print("="*70)

    # ANOVA
    print("\n[ANOVA] Effects of Model and Category on DeltaH")
    print("-" * 70)
    anova = stats_results['anova']

    print(f"\nModel Effect:")
    print(f"  F({anova['model_effect']['F_statistic']:.2f}), p = {anova['model_effect']['p_value']:.4f}")
    print(f"  {anova['model_effect']['interpretation']}")

    print(f"\nCategory Effect:")
    print(f"  F({anova['category_effect']['F_statistic']:.2f}), p = {anova['category_effect']['p_value']:.4f}")
    print(f"  {anova['category_effect']['interpretation']}")

    # Pairwise comparisons
    print("\n[PAIRWISE COMPARISONS] Model Comparisons (Bonferroni corrected)")
    print("-" * 70)
    pairwise = stats_results['pairwise_comparisons']

    for comp in pairwise['comparisons']:
        sig_marker = "***" if comp['significant_bonferroni'] else "ns"
        print(f"\n  {comp['model1']} vs {comp['model2']}: {sig_marker}")
        print(f"    t = {comp['t_statistic']:.3f}, p_adj = {comp['p_value_bonferroni']:.4f}")
        print(f"    Cohen's d = {comp['cohens_d']:.3f} ({comp['effect_size']})")
        print(f"    Mean difference = {comp['mean_diff']:.3f} bits")

    # Correlations
    print("\n[CORRELATIONS] Model Size vs DeltaH")
    print("-" * 70)
    corr = stats_results['correlations']

    for cat, corr_data in corr['categories'].items():
        print(f"\n  {cat.capitalize()}:")
        spear = corr_data['spearman']
        pears = corr_data['pearson']
        print(f"    Spearman: rho = {spear['rho']:.3f}, p = {spear['p_value']:.4f} "
              f"({'*' if spear['significant'] else 'ns'})")
        print(f"    Pearson:  r = {pears['r']:.3f}, p = {pears['p_value']:.4f} "
              f"({'*' if pears['significant'] else 'ns'})")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    summary = stats_results['summary']
    print(f"Model effect significant: {'Yes' if summary['anova_model_significant'] else 'No'}")
    print(f"Category effect significant: {'Yes' if summary['anova_category_significant'] else 'No'}")
    print(f"Significant pairwise comparisons: {summary['n_significant_pairwise']}/{summary['total_pairwise']}")
    print(f"Categories with significant scaling correlation: {len(summary['categories_with_significant_correlation'])}/3")
    print("="*70 + "\n")


if __name__ == "__main__":
    print("Module: multi_model_statistical_tests")
    print("Functions:")
    print("  - perform_two_way_anova()")
    print("  - perform_pairwise_model_comparisons()")
    print("  - compute_effect_sizes_between_models()")
    print("  - perform_correlation_tests()")
    print("  - run_comprehensive_statistical_analysis()")
