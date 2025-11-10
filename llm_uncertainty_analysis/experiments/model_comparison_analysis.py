"""
Model Comparison and Hypothesis Validation

This module analyzes multi-model ICL results to validate scaling and consistency hypotheses.

Hypotheses tested:
- H1 (Scaling): DeltaH increases with model size - Larger models benefit more from context
- H2 (Consistency): Category ranking (factual > logical > creative) consistent across models

Usage:
    from llm_uncertainty_analysis.experiments.model_comparison_analysis import (
        validate_hypotheses
    )

    validation_results = validate_hypotheses(multi_model_results)
    print(validation_results['H1_scaling']['conclusion'])
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
from ..models.model_config import get_model_params_numeric


# ============================================================================
# HYPOTHESIS 1: SCALING ANALYSIS
# ============================================================================

def analyze_scaling_hypothesis(results: Dict) -> Dict:
    """
    Test H1: DeltaH increases with model size.

    Performs Spearman correlation between log(model_params) and DeltaH for each category.

    Args:
        results: Output from run_multi_model_icl_experiment()

    Returns:
        Dictionary with structure:
        {
            'hypothesis': 'DeltaH increases with model size',
            'method': 'Spearman correlation',
            'categories': {
                'factual': {
                    'correlation': float,  # Spearman rho
                    'p_value': float,
                    'supported': bool,
                    'model_params': [...],
                    'delta_h_values': [...]
                },
                ...
            },
            'overall_conclusion': str
        }
    """
    scaling_data = results['comparison']['scaling_summary']
    categories = list(scaling_data.keys())

    analysis = {
        'hypothesis': 'DeltaH increases with model size',
        'method': 'Spearman rank correlation',
        'alpha': 0.05,
        'categories': {}
    }

    supported_count = 0

    for category in categories:
        model_params = np.array(scaling_data[category]['model_params'])
        delta_h_values = np.array(scaling_data[category]['delta_h_5shot'])

        # Spearman correlation (handles monotonic relationships)
        rho, p_value = stats.spearmanr(model_params, delta_h_values)

        supported = (rho > 0) and (p_value < 0.05)
        if supported:
            supported_count += 1

        analysis['categories'][category] = {
            'correlation': float(rho),
            'p_value': float(p_value),
            'supported': supported,
            'model_params': model_params.tolist(),
            'delta_h_values': delta_h_values.tolist(),
            'interpretation': f"{'Positive' if rho > 0 else 'Negative'} correlation "
                            f"({'significant' if p_value < 0.05 else 'not significant'})"
        }

    # Overall conclusion
    if supported_count == len(categories):
        conclusion = "STRONGLY SUPPORTED: All categories show positive scaling"
    elif supported_count >= len(categories) / 2:
        conclusion = "PARTIALLY SUPPORTED: Majority of categories show positive scaling"
    else:
        conclusion = "NOT SUPPORTED: Scaling effect not consistent"

    analysis['overall_conclusion'] = conclusion
    analysis['supported_categories'] = supported_count
    analysis['total_categories'] = len(categories)

    return analysis


# ============================================================================
# HYPOTHESIS 2: CONSISTENCY ANALYSIS
# ============================================================================

def analyze_consistency_hypothesis(results: Dict, expected_ranking: List[str] = None) -> Dict:
    """
    Test H2: Category ranking is consistent across models.

    Uses Kendall's W (coefficient of concordance) to measure agreement.

    Args:
        results: Output from run_multi_model_icl_experiment()
        expected_ranking: Expected order (default: ['factual', 'logical', 'creative'])

    Returns:
        Dictionary with structure:
        {
            'hypothesis': str,
            'method': str,
            'expected_ranking': [...],
            'model_rankings': {...},
            'kendall_w': float,  # 0-1, 1 = perfect agreement
            'chi_square': float,
            'p_value': float,
            'supported': bool,
            'conclusion': str
        }
    """
    if expected_ranking is None:
        expected_ranking = ['factual', 'logical', 'creative']

    consistency_data = results['comparison']['consistency_summary']
    model_ids = list(consistency_data.keys())

    analysis = {
        'hypothesis': 'Category ranking is consistent across models',
        'method': "Kendall's W (coefficient of concordance)",
        'expected_ranking': expected_ranking,
        'alpha': 0.05,
        'model_rankings': {}
    }

    # Extract rankings for each model
    rankings_matrix = []
    for model_id in model_ids:
        ranking = consistency_data[model_id]['ranking']
        analysis['model_rankings'][model_id] = ranking

        # Convert to rank numbers (1 = highest DeltaH)
        rank_numbers = [ranking.index(cat) + 1 for cat in expected_ranking]
        rankings_matrix.append(rank_numbers)

    rankings_matrix = np.array(rankings_matrix)  # shape: (n_models, n_categories)

    # Calculate Kendall's W
    n_models = rankings_matrix.shape[0]
    n_categories = rankings_matrix.shape[1]

    # Sum of ranks for each category
    rank_sums = rankings_matrix.sum(axis=0)

    # Mean of rank sums
    mean_rank_sum = rank_sums.mean()

    # Sum of squared deviations
    S = ((rank_sums - mean_rank_sum) ** 2).sum()

    # Kendall's W
    W = (12 * S) / (n_models ** 2 * (n_categories ** 3 - n_categories))

    # Chi-square test statistic
    chi_square = n_models * (n_categories - 1) * W
    df = n_categories - 1
    p_value = 1 - stats.chi2.cdf(chi_square, df)

    # Determine if supported
    # W > 0.7 indicates strong agreement, p < 0.05 indicates significance
    supported = (W > 0.5) and (p_value < 0.05)

    analysis['kendall_w'] = float(W)
    analysis['chi_square'] = float(chi_square)
    analysis['degrees_of_freedom'] = int(df)
    analysis['p_value'] = float(p_value)
    analysis['supported'] = supported

    # Interpretation
    if W >= 0.9:
        agreement_level = "very strong"
    elif W >= 0.7:
        agreement_level = "strong"
    elif W >= 0.5:
        agreement_level = "moderate"
    elif W >= 0.3:
        agreement_level = "weak"
    else:
        agreement_level = "very weak"

    analysis['agreement_level'] = agreement_level

    # Conclusion
    if supported:
        conclusion = f"SUPPORTED: {agreement_level.capitalize()} agreement (W={W:.3f}, p={p_value:.4f})"
    else:
        conclusion = f"NOT SUPPORTED: {agreement_level.capitalize()} agreement (W={W:.3f}, p={p_value:.4f})"

    analysis['conclusion'] = conclusion

    # Check if actual rankings match expected
    matches = []
    for model_id in model_ids:
        ranking = analysis['model_rankings'][model_id]
        match = (ranking == expected_ranking)
        matches.append(match)
        analysis['model_rankings'][model_id + '_matches_expected'] = match

    analysis['exact_matches'] = sum(matches)
    analysis['total_models'] = len(model_ids)

    return analysis


# ============================================================================
# EFFICIENCY ANALYSIS
# ============================================================================

def compute_model_efficiency(results: Dict) -> Dict:
    """
    Compute ICL efficiency: DeltaH per parameter.

    Identifies which model is most "efficient" at leveraging context.

    Args:
        results: Output from run_multi_model_icl_experiment()

    Returns:
        Dictionary with efficiency metrics
    """
    scaling_data = results['comparison']['scaling_summary']
    categories = list(scaling_data.keys())

    efficiency = {
        'description': 'DeltaH per million parameters',
        'categories': {}
    }

    for category in categories:
        model_ids = scaling_data[category]['model_ids']
        model_params = np.array(scaling_data[category]['model_params'])
        delta_h_values = np.array(scaling_data[category]['delta_h_5shot'])

        # Efficiency = DeltaH / (params / 1e6)
        params_millions = model_params / 1e6
        efficiency_values = delta_h_values / params_millions

        # Find most efficient model
        most_efficient_idx = np.argmax(efficiency_values)
        most_efficient_model = model_ids[most_efficient_idx]

        efficiency['categories'][category] = {
            'model_ids': model_ids,
            'efficiency_values': efficiency_values.tolist(),
            'most_efficient_model': most_efficient_model,
            'most_efficient_value': float(efficiency_values[most_efficient_idx])
        }

    return efficiency


# ============================================================================
# CONVERGENCE ANALYSIS
# ============================================================================

def analyze_convergence_patterns(results: Dict) -> Dict:
    """
    Analyze how DeltaH changes with increasing k.

    Checks for saturation effects: do larger models saturate faster?

    Args:
        results: Output from run_multi_model_icl_experiment()

    Returns:
        Dictionary with convergence analysis
    """
    results_by_model = results['results_by_model']
    model_ids = list(results_by_model.keys())
    categories = list(results_by_model[model_ids[0]]['categories'].keys())

    convergence = {
        'description': 'Analysis of DeltaH saturation with increasing k',
        'categories': {}
    }

    for category in categories:
        category_convergence = {}

        for model_id in model_ids:
            cat_results = results_by_model[model_id]['categories'][category]
            n_examples = cat_results['n_examples']
            delta_h = cat_results['delta_h']

            # Calculate marginal gains: Delta(DeltaH) between consecutive k values
            marginal_gains = []
            for i in range(1, len(delta_h)):
                gain = delta_h[i] - delta_h[i-1]
                marginal_gains.append(gain)

            # Check if diminishing returns (marginal gains decreasing)
            diminishing = all(marginal_gains[i] <= marginal_gains[i-1]
                            for i in range(1, len(marginal_gains)))

            category_convergence[model_id] = {
                'n_examples': n_examples,
                'delta_h': delta_h,
                'marginal_gains': marginal_gains,
                'shows_diminishing_returns': diminishing
            }

        convergence['categories'][category] = category_convergence

    return convergence


# ============================================================================
# MASTER VALIDATION FUNCTION
# ============================================================================

def validate_hypotheses(results: Dict) -> Dict:
    """
    Validate all hypotheses and generate comprehensive analysis.

    Args:
        results: Output from run_multi_model_icl_experiment()

    Returns:
        Dictionary with all validation results:
        {
            'H1_scaling': {...},
            'H2_consistency': {...},
            'efficiency': {...},
            'convergence': {...},
            'summary': {...}
        }
    """
    print("\n" + "="*70)
    print("HYPOTHESIS VALIDATION")
    print("="*70)

    validation = {}

    # H1: Scaling
    print("\nTesting H1 (Scaling)...")
    validation['H1_scaling'] = analyze_scaling_hypothesis(results)
    print(f"  Result: {validation['H1_scaling']['overall_conclusion']}")

    # H2: Consistency
    print("\nTesting H2 (Consistency)...")
    validation['H2_consistency'] = analyze_consistency_hypothesis(results)
    print(f"  Result: {validation['H2_consistency']['conclusion']}")

    # Efficiency analysis
    print("\nComputing efficiency...")
    validation['efficiency'] = compute_model_efficiency(results)

    # Convergence analysis
    print("\nAnalyzing convergence patterns...")
    validation['convergence'] = analyze_convergence_patterns(results)

    # Summary
    h1_supported = validation['H1_scaling']['supported_categories'] >= 2
    h2_supported = validation['H2_consistency']['supported']

    validation['summary'] = {
        'H1_scaling_supported': h1_supported,
        'H2_consistency_supported': h2_supported,
        'both_supported': h1_supported and h2_supported,
        'overall_assessment': get_overall_assessment(h1_supported, h2_supported)
    }

    print("\n" + "="*70)
    print("VALIDATION COMPLETE")
    print("="*70)
    print(f"H1 (Scaling): {'[SUPPORTED] SUPPORTED' if h1_supported else '[NOT SUPPORTED] NOT SUPPORTED'}")
    print(f"H2 (Consistency): {'[SUPPORTED] SUPPORTED' if h2_supported else '[NOT SUPPORTED] NOT SUPPORTED'}")
    print(f"\nOverall: {validation['summary']['overall_assessment']}")
    print("="*70)

    return validation


def get_overall_assessment(h1_supported: bool, h2_supported: bool) -> str:
    """Generate overall assessment based on hypothesis results."""
    if h1_supported and h2_supported:
        return "STRONG EVIDENCE: Both scaling and consistency hypotheses supported"
    elif h1_supported:
        return "PARTIAL EVIDENCE: Scaling supported, consistency weak"
    elif h2_supported:
        return "PARTIAL EVIDENCE: Consistency supported, scaling weak"
    else:
        return "WEAK EVIDENCE: Neither hypothesis strongly supported"


# ============================================================================
# PRETTY PRINTING
# ============================================================================

def print_validation_report(validation: Dict):
    """
    Print detailed validation report.

    Args:
        validation: Output from validate_hypotheses()
    """
    print("\n" + "="*70)
    print("HYPOTHESIS VALIDATION REPORT")
    print("="*70)

    # H1: Scaling
    h1 = validation['H1_scaling']
    print(f"\n[H1] {h1['hypothesis']}")
    print(f"Method: {h1['method']}")
    print(f"Conclusion: {h1['overall_conclusion']}\n")

    for category, data in h1['categories'].items():
        print(f"  {category.capitalize()}:")
        print(f"    rho = {data['correlation']:.3f}, p = {data['p_value']:.4f}")
        print(f"    {data['interpretation']}")

    # H2: Consistency
    h2 = validation['H2_consistency']
    print(f"\n[H2] {h2['hypothesis']}")
    print(f"Method: {h2['method']}")
    print(f"Conclusion: {h2['conclusion']}\n")

    print(f"  Expected ranking: {' > '.join(h2['expected_ranking'])}")
    print(f"  Agreement level: {h2['agreement_level']} (W = {h2['kendall_w']:.3f})")
    print(f"  Exact matches: {h2['exact_matches']}/{h2['total_models']}\n")

    for model_id, ranking in h2['model_rankings'].items():
        if not model_id.endswith('_matches_expected'):
            match = "[SUPPORTED]" if h2['model_rankings'].get(model_id + '_matches_expected') else "[NOT SUPPORTED]"
            print(f"    {model_id}: {' > '.join(ranking)} {match}")

    # Efficiency
    print(f"\n[EFFICIENCY] DeltaH per million parameters")
    eff = validation['efficiency']
    for category, data in eff['categories'].items():
        print(f"\n  {category.capitalize()}:")
        for i, model_id in enumerate(data['model_ids']):
            eff_val = data['efficiency_values'][i]
            marker = " <- Most efficient" if model_id == data['most_efficient_model'] else ""
            print(f"    {model_id}: {eff_val:.6f}{marker}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    summary = validation['summary']
    print(f"H1 (Scaling): {'[SUPPORTED] SUPPORTED' if summary['H1_scaling_supported'] else '[NOT SUPPORTED] NOT SUPPORTED'}")
    print(f"H2 (Consistency): {'[SUPPORTED] SUPPORTED' if summary['H2_consistency_supported'] else '[NOT SUPPORTED] NOT SUPPORTED'}")
    print(f"\n{summary['overall_assessment']}")
    print("="*70 + "\n")


if __name__ == "__main__":
    print("Module: model_comparison_analysis")
    print("Functions:")
    print("  - analyze_scaling_hypothesis()")
    print("  - analyze_consistency_hypothesis()")
    print("  - compute_model_efficiency()")
    print("  - analyze_convergence_patterns()")
    print("  - validate_hypotheses()")
