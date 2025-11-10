"""
Advanced Visualization Functions

This module contains advanced plotting functions for statistical analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def plot_anova_boxplot(data_by_category: Dict[str, np.ndarray],
                        f_stat: float,
                        p_value: float,
                        save_path: Optional[str] = None,
                        figsize: tuple = (12, 7)):
    """
    Create boxplot for ANOVA with statistical annotations.

    Args:
        data_by_category: Dictionary with data by category
        f_stat: F-statistic from ANOVA
        p_value: P-value from ANOVA
        save_path: Optional path to save figure
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Prepare data
    plot_data = []
    for cat, data in data_by_category.items():
        for value in data:
            plot_data.append({'Category': cat.capitalize(), 'Entropy (bits)': value})

    plot_df = pd.DataFrame(plot_data)

    # Create boxplot
    sns.boxplot(x='Category', y='Entropy (bits)', data=plot_df, palette='Set2', ax=ax)
    sns.stripplot(x='Category', y='Entropy (bits)', data=plot_df,
                  color='black', alpha=0.3, size=3, ax=ax)

    # Add grand mean line
    grand_mean = np.mean(np.concatenate(list(data_by_category.values())))
    ax.axhline(y=grand_mean, color='red', linestyle='--', linewidth=1.5,
               label=f'Grand mean = {grand_mean:.2f}', alpha=0.7)

    # Title with ANOVA results
    n_groups = len(data_by_category)
    n_total = sum(len(d) for d in data_by_category.values())
    df_between = n_groups - 1
    df_within = n_total - n_groups

    ax.set_title(f'Entropy Distribution by Context Category\n' +
                f'ANOVA: F({df_between}, {df_within}) = {f_stat:.2f}, p = {p_value:.4f}',
                fontsize=13, fontweight='bold', pad=20)

    ax.set_ylabel('Mean Entropy (bits)', fontsize=12)
    ax.set_xlabel('Context Category', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    # Annotate means
    for i, (cat, data) in enumerate(data_by_category.items()):
        mean_val = np.mean(data)
        ax.text(i, mean_val, f'μ={mean_val:.2f}',
                ha='center', va='bottom', fontweight='bold', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig


def plot_tukey_hsd_intervals(tukey_df: pd.DataFrame,
                             save_path: Optional[str] = None,
                             figsize: tuple = (12, 6)):
    """
    Plot Tukey HSD confidence intervals.

    Args:
        tukey_df: DataFrame with Tukey HSD results
        save_path: Optional path to save
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Extract data
    comparisons = []
    means_diff = []
    lower_bounds = []
    upper_bounds = []
    colors = []

    for idx, row in tukey_df.iterrows():
        comparison = f"{row['group1']} vs {row['group2']}"
        comparisons.append(comparison)
        means_diff.append(float(row['meandiff']))
        lower_bounds.append(float(row['lower']))
        upper_bounds.append(float(row['upper']))

        # Color by significance
        colors.append('red' if row['reject'] else 'gray')

    # Plot intervals
    y_pos = np.arange(len(comparisons))

    for i, (comp, mean, lower, upper, color) in enumerate(
            zip(comparisons, means_diff, lower_bounds, upper_bounds, colors)):
        ax.plot([lower, upper], [i, i], color=color, linewidth=2, alpha=0.7)
        ax.plot(mean, i, 'o', color=color, markersize=10)

    # Zero line
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5,
               label='No difference')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(comparisons)
    ax.set_xlabel('Mean Difference (bits)', fontsize=12)
    ax.set_ylabel('Pairwise Comparison', fontsize=12)
    ax.set_title('Tukey HSD: 95% Confidence Intervals for Mean Differences\n' +
                '(Red = Significant, Gray = Not Significant)',
                fontsize=13, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend(loc='best')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_confidence_intervals(data_by_category: Dict[str, np.ndarray],
                              save_path: Optional[str] = None,
                              figsize: tuple = (10, 6)):
    """
    Plot means with 95% confidence intervals.

    Args:
        data_by_category: Dictionary with data
        save_path: Optional save path
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    categories = list(data_by_category.keys())
    means = []
    cis = []

    # Calculate stats
    for cat in categories:
        data = data_by_category[cat]
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        se = std / np.sqrt(len(data))
        ci_95 = 1.96 * se

        means.append(mean)
        cis.append(ci_95)

    # Plot
    x_pos = np.arange(len(categories))
    colors = {'factual': '#3498db', 'logical': '#2ecc71', 'creative': '#e74c3c'}
    bar_colors = [colors.get(cat, 'gray') for cat in categories]

    bars = ax.bar(x_pos, means, yerr=cis, capsize=8, alpha=0.7,
                  color=bar_colors, edgecolor='black', linewidth=1.5,
                  error_kw={'linewidth': 2, 'ecolor': 'black'})

    ax.set_xticks(x_pos)
    ax.set_xticklabels([cat.capitalize() for cat in categories])
    ax.set_ylabel('Mean Entropy (bits)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Context Category', fontsize=12, fontweight='bold')
    ax.set_title('Mean Entropy by Category (95% Confidence Intervals)',
                fontsize=13, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y')

    # Annotate values
    for i, (bar, mean_val, ci) in enumerate(zip(bars, means, cis)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + ci + 0.05,
                f'{mean_val:.3f}',
                ha='center', va='bottom', fontweight='bold', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_mutual_information_heatmap(icl_results: Dict,
                                    save_path: Optional[str] = None,
                                    figsize: tuple = (10, 6)):
    """
    Create heatmap of mutual information across ICL configurations.

    Args:
        icl_results: Dictionary with ICL results by category
        save_path: Optional save path
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Create matrix
    categories = list(icl_results.keys())
    n_examples_list = icl_results[categories[0]]['n_examples']

    mi_matrix = np.array([
        [icl_results[cat]['mutual_info'][i] for i in range(len(n_examples_list))]
        for cat in categories
    ])

    # Create heatmap
    im = ax.imshow(mi_matrix, cmap='YlOrRd', aspect='auto', vmin=0)

    # Configure ticks
    ax.set_xticks(np.arange(len(n_examples_list)))
    ax.set_yticks(np.arange(len(categories)))
    ax.set_xticklabels([f'{n}-shot' for n in n_examples_list])
    ax.set_yticklabels([cat.capitalize() for cat in categories])

    # Add values in cells
    for i in range(len(categories)):
        for j in range(len(n_examples_list)):
            value = mi_matrix[i, j]
            text = ax.text(j, i, f'{value:.3f}',
                          ha="center", va="center",
                          color="black" if value < 0.5 else "white",
                          fontsize=11, fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Mutual Information I(E; Y) (bits)',
                   rotation=270, labelpad=20,
                   fontsize=11, fontweight='bold')

    ax.set_xlabel('ICL Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Context Category', fontsize=12, fontweight='bold')
    ax.set_title('Mutual Information: Uncertainty Reduction with ICL',
                fontsize=12, fontweight='bold', pad=15)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig
