"""
Complete Analysis Runner

This module orchestrates the complete analysis pipeline including all experiments.
"""

import pandas as pd
from pathlib import Path
from typing import Dict
from ..analysis import UncertaintyAnalyzer
from ..data_management import RealDatasetManager
from .category_comparison import run_category_comparison
from .statistical_tests import run_complete_statistical_analysis
from .icl_experiment import run_icl_experiment
from ..visualization.advanced_plots import (
    plot_anova_boxplot,
    plot_tukey_hsd_intervals,
    plot_confidence_intervals,
    plot_mutual_information_heatmap
)
from ..visualization.plots import plot_entropy_by_category
from ..utils import ensure_directory


def run_complete_analysis(model_name: str = "gpt2",
                          data_dir: str = "data",
                          output_dir: str = "results",
                          n_samples_per_category: int = 50,
                          device: str = "cuda") -> Dict:
    """
    Run complete analysis pipeline.

    Args:
        model_name: Model to use
        data_dir: Data directory path
        output_dir: Output directory path
        n_samples_per_category: Samples per category
        device: Device to use

    Returns:
        Dictionary with all results
    """
    print("="*80)
    print("COMPLETE LLM UNCERTAINTY ANALYSIS PIPELINE")
    print("="*80)
    print(f"Model: {model_name}")
    print(f"Samples per category: {n_samples_per_category}")
    print("="*80)

    # Create output directories
    output_path = Path(output_dir)
    fig_path = ensure_directory(output_path / "figures")

    # 1. Load data
    print("\n[1/6] Loading datasets...")
    dataset_manager = RealDatasetManager(data_dir=data_dir)
    datasets = dataset_manager.load_all_datasets(n_per_category=n_samples_per_category)

    # Combine samples
    all_samples = []
    for category, samples in datasets.items():
        all_samples.extend(samples)

    print(f"Total samples loaded: {len(all_samples)}")

    # 2. Initialize analyzer
    print("\n[2/6] Initializing analyzer...")
    analyzer = UncertaintyAnalyzer(model_name, device=device)

    # 3. Run category comparison
    print("\n[3/6] Running category comparison experiment...")
    results_df = run_category_comparison(analyzer, all_samples)

    # Save results
    results_df.to_csv(output_path / "category_results.csv", index=False)
    print(f"Results saved to: {output_path / 'category_results.csv'}")

    # 4. Statistical analysis
    print("\n[4/6] Running statistical tests...")
    statistical_results = run_complete_statistical_analysis(results_df)

    # 5. Create visualizations
    print("\n[5/6] Creating visualizations...")

    # Basic entropy plot
    plot_entropy_by_category(results_df, save_path=str(fig_path / "entropy_by_category.png"))

    # ANOVA boxplot
    plot_anova_boxplot(
        statistical_results['data_by_category'],
        statistical_results['anova']['f_statistic'],
        statistical_results['anova']['p_value'],
        save_path=str(fig_path / "anova_boxplot.png")
    )

    # Tukey HSD intervals
    plot_tukey_hsd_intervals(
        statistical_results['tukey_hsd'],
        save_path=str(fig_path / "tukey_hsd_intervals.png")
    )

    # Confidence intervals
    plot_confidence_intervals(
        statistical_results['data_by_category'],
        save_path=str(fig_path / "confidence_intervals.png")
    )

    # 6. ICL experiment (optional, small scale)
    print("\n[6/6] Running ICL experiment (sample)...")

    # Define ICL task
    task_description = "Answer factual questions."
    examples = [
        ("What is the capital of France?", "Paris"),
        ("What is 2 + 2?", "4"),
        ("What color is the sky?", "Blue"),
    ]
    queries = ["What is the capital of Italy?", "What is 3 + 3?"]

    icl_results_factual = run_icl_experiment(
        analyzer.model,
        analyzer.tokenizer,
        task_description,
        examples,
        queries,
        n_examples_range=[0, 1, 2, 3],
        device=str(device)
    )

    # Combine results
    complete_results = {
        'category_analysis': results_df,
        'statistical_tests': statistical_results,
        'icl_sample': icl_results_factual
    }

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"Results saved to: {output_path}")
    print(f"Figures saved to: {fig_path}")

    return complete_results
