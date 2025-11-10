"""
Visualization Functions

This module contains functions for creating plots and visualizations.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional


def plot_entropy_by_category(results_df: pd.DataFrame,
                              save_path: Optional[str] = None,
                              figsize: tuple = (10, 6)):
    """
    Create boxplot of entropy distributions by category.

    Args:
        results_df: DataFrame with 'category' and 'mean_entropy' columns
        save_path: Optional path to save the figure
        figsize: Figure size tuple
    """
    plt.figure(figsize=figsize)

    # Create boxplot
    sns.boxplot(data=results_df, x='category', y='mean_entropy',
                palette={'factual': '#2ecc71', 'logical': '#f39c12', 'creative': '#e74c3c'})

    plt.xlabel('Category', fontsize=12, fontweight='bold')
    plt.ylabel('Mean Entropy (bits)', fontsize=12, fontweight='bold')
    plt.title('Entropy Distribution by Context Category', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.tight_layout()
    return plt.gcf()


def plot_icl_analysis(icl_results: Dict,
                      save_path: Optional[str] = None,
                      figsize: tuple = (16, 12)):
    """
    Create comprehensive ICL analysis plot with 4 subplots.

    Args:
        icl_results: Dictionary with ICL experimental results
        save_path: Optional path to save the figure
        figsize: Figure size tuple
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Subplot 1: Entropy vs n-examples
    ax1 = axes[0, 0]
    for category in icl_results.keys():
        data = icl_results[category]
        ax1.plot(data['n_examples'], data['mean_entropy'],
                marker='o', linewidth=2, label=category)

    ax1.set_xlabel('Number of Examples', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Mean Entropy (bits)', fontsize=12, fontweight='bold')
    ax1.set_title('ICL Effect on Entropy', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Entropy reduction by category
    ax2 = axes[0, 1]
    reductions = []
    categories = []
    for category, data in icl_results.items():
        reduction = data['mean_entropy'][0] - data['mean_entropy'][-1]
        reductions.append(reduction)
        categories.append(category)

    ax2.bar(categories, reductions, color=['#2ecc71', '#f39c12', '#e74c3c'])
    ax2.set_ylabel('Entropy Reduction (bits)', fontsize=12, fontweight='bold')
    ax2.set_title('Total Entropy Reduction', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Subplot 3: Percentage reduction
    ax3 = axes[1, 0]
    pct_reductions = [(r / icl_results[cat]['mean_entropy'][0]) * 100
                      for cat, r in zip(categories, reductions)]

    ax3.bar(categories, pct_reductions, color=['#2ecc71', '#f39c12', '#e74c3c'])
    ax3.set_ylabel('Percentage Reduction (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Relative Entropy Reduction', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # Subplot 4: Mutual information heatmap
    ax4 = axes[1, 1]
    mi_matrix = np.array([[icl_results[cat]['mutual_info'][i]
                          for i in range(len(icl_results[cat]['mutual_info']))]
                         for cat in icl_results.keys()])

    im = ax4.imshow(mi_matrix, cmap='YlOrRd', aspect='auto')
    ax4.set_xticks(range(len(icl_results[categories[0]]['n_examples'])))
    ax4.set_xticklabels(icl_results[categories[0]]['n_examples'])
    ax4.set_yticks(range(len(categories)))
    ax4.set_yticklabels(categories)
    ax4.set_xlabel('Number of Examples', fontsize=12, fontweight='bold')
    ax4.set_title('Mutual Information Heatmap', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax4, label='MI (bits)')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.tight_layout()
    return fig


def plot_cohens_d_comparison(cohens_d_results: List[Dict],
                             save_path: Optional[str] = None,
                             figsize: tuple = (10, 6)):
    """
    Create bar plot of Cohen's d effect sizes.

    Args:
        cohens_d_results: List of dictionaries with Cohen's d results
        save_path: Optional path to save the figure
        figsize: Figure size tuple
    """
    plt.figure(figsize=figsize)

    comparisons = [r['comparison'] for r in cohens_d_results]
    d_values = [r['cohens_d'] for r in cohens_d_results]

    # Color bars based on magnitude
    colors = []
    for d in d_values:
        abs_d = abs(d)
        if abs_d < 0.2:
            colors.append('#d3d3d3')  # Gray - negligible
        elif abs_d < 0.5:
            colors.append('#90EE90')  # Light green - small
        elif abs_d < 0.8:
            colors.append('#FFA500')  # Orange - medium
        else:
            colors.append('#FF6347')  # Red - large

    bars = plt.bar(range(len(comparisons)), d_values, color=colors,
                   edgecolor='black', alpha=0.8, linewidth=1.5)

    # Reference lines
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
    plt.axhline(y=0.2, color='green', linestyle='--', alpha=0.7, label='Small (d=0.2)')
    plt.axhline(y=-0.2, color='green', linestyle='--', alpha=0.7)
    plt.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Medium (d=0.5)')
    plt.axhline(y=-0.5, color='orange', linestyle='--', alpha=0.7)
    plt.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Large (d=0.8)')
    plt.axhline(y=-0.8, color='red', linestyle='--', alpha=0.7)

    plt.xticks(range(len(comparisons)), comparisons, rotation=0)
    plt.ylabel("Cohen's d", fontsize=12, fontweight='bold')
    plt.xlabel('Comparison', fontsize=12, fontweight='bold')
    plt.title("Effect Size Between Categories (Cohen's d)", fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')

    # Add values on bars
    for bar, d_val in zip(bars, d_values):
        height = bar.get_height()
        y_pos = height + 0.05 if height > 0 else height - 0.15
        plt.text(bar.get_x() + bar.get_width()/2., y_pos, f'{d_val:.2f}',
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=11, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.tight_layout()
    return plt.gcf()
