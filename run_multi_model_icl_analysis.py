#!/usr/bin/env python3
"""
Multi-Model In-Context Learning Analysis - Main Execution Script

This script orchestrates the complete multi-model ICL analysis pipeline:
1. Run ICL experiments across 3 models and 3 categories
2. Perform statistical analysis
3. Validate hypotheses
4. Generate results JSON files
5. Print comprehensive summary

Usage:
    python run_multi_model_icl_analysis.py [--quick]

Options:
    --quick: Run with reduced configurations (2 models, 3 k-shot values, 3 queries)
             for testing purposes

Output:
    - outputs/multi_model_icl/results.json
    - outputs/multi_model_icl/statistical_analysis.json
    - outputs/multi_model_icl/hypothesis_validation.json
    - Console summary

Expected runtime:
    - Full: 2-4 hours (GPU) / 8-12 hours (CPU)
    - Quick: 15-30 minutes (GPU) / 1-2 hours (CPU)

Author: Benito Fuentes, Sebastian Vergara
Course: EL7024-1 - Universidad de Chile
Date: November 2025
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from llm_uncertainty_analysis.config import (
    setup_reproducibility,
    device,
    print_device_info
)
from llm_uncertainty_analysis.models.model_config import (
    COMPARISON_MODELS,
    print_comparison_models_info
)
from llm_uncertainty_analysis.experiments.multi_model_icl_experiment import (
    run_multi_model_icl_experiment,
    print_results_summary
)
from llm_uncertainty_analysis.experiments.model_comparison_analysis import (
    validate_hypotheses,
    print_validation_report
)
from llm_uncertainty_analysis.experiments.multi_model_statistical_tests import (
    run_comprehensive_statistical_analysis,
    print_statistical_report
)


def ensure_output_directory(dir_path: str) -> Path:
    """Create output directory if it doesn't exist."""
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: dict, filepath: str):
    """Save dictionary as JSON with pretty printing."""
    # Convert numpy types to native Python types for JSON serialization
    import numpy as np

    def convert(obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert(item) for item in obj]
        else:
            return obj

    converted_data = convert(data)

    with open(filepath, 'w') as f:
        json.dump(converted_data, f, indent=2)

    print(f"[OK] Saved: {filepath}")


def main():
    """Main execution function."""

    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Run multi-model ICL analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--quick', action='store_true',
                       help='Run quick test with reduced configurations')

    args = parser.parse_args()

    # ========================================================================
    # SETUP
    # ========================================================================

    print("\n" + "="*70)
    print("MULTI-MODEL IN-CONTEXT LEARNING ANALYSIS")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {'QUICK TEST' if args.quick else 'FULL ANALYSIS'}")
    print("="*70)

    # Setup reproducibility
    setup_reproducibility()

    # Print device info
    print_device_info()

    # Print model info
    print_comparison_models_info()

    # Configure experiment parameters
    if args.quick:
        print("\n[WARNING] QUICK MODE: Using reduced configurations for testing")
        model_ids = COMPARISON_MODELS[:2]  # Only 2 models
        n_examples_range = [0, 1, 3]  # Only 3 k-shot values
        n_queries = 3  # Only 3 queries
    else:
        model_ids = COMPARISON_MODELS
        n_examples_range = [0, 1, 2, 3, 5]
        n_queries = 10

    print(f"\nExperiment Configuration:")
    print(f"  Models: {model_ids}")
    print(f"  k-shot range: {n_examples_range}")
    print(f"  Queries per config: {n_queries}")
    print(f"  Device: {device}")

    # Create output directory
    output_dir = ensure_output_directory("outputs/multi_model_icl")
    print(f"\nOutput directory: {output_dir.absolute()}")

    # ========================================================================
    # PHASE 1: RUN ICL EXPERIMENTS
    # ========================================================================

    print("\n" + "="*70)
    print("PHASE 1: RUNNING ICL EXPERIMENTS")
    print("="*70)

    results = run_multi_model_icl_experiment(
        model_ids=model_ids,
        n_examples_range=n_examples_range,
        n_queries_per_config=n_queries,
        device=str(device)
    )

    # Add metadata
    results['metadata']['timestamp'] = datetime.now().isoformat()
    results['metadata']['quick_mode'] = args.quick

    # Save results
    save_json(results, str(output_dir / 'results.json'))

    # Print summary
    print_results_summary(results)

    # ========================================================================
    # PHASE 2: STATISTICAL ANALYSIS
    # ========================================================================

    print("\n" + "="*70)
    print("PHASE 2: STATISTICAL ANALYSIS")
    print("="*70)

    stats_results = run_comprehensive_statistical_analysis(results)

    # Save statistical results
    save_json(stats_results, str(output_dir / 'statistical_analysis.json'))

    # Print report
    print_statistical_report(stats_results)

    # ========================================================================
    # PHASE 3: HYPOTHESIS VALIDATION
    # ========================================================================

    print("\n" + "="*70)
    print("PHASE 3: HYPOTHESIS VALIDATION")
    print("="*70)

    validation_results = validate_hypotheses(results)

    # Save validation results
    save_json(validation_results, str(output_dir / 'hypothesis_validation.json'))

    # Print report
    print_validation_report(validation_results)

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print(f"\n[OUTPUT FILES]:")
    print(f"   - {output_dir / 'results.json'}")
    print(f"   - {output_dir / 'statistical_analysis.json'}")
    print(f"   - {output_dir / 'hypothesis_validation.json'}")

    print(f"\n[KEY FINDINGS]:")
    h1_supported = validation_results['summary']['H1_scaling_supported']
    h2_supported = validation_results['summary']['H2_consistency_supported']

    print(f"   H1 (Scaling): {'[SUPPORTED]' if h1_supported else '[NOT SUPPORTED]'}")
    print(f"   H2 (Consistency): {'[SUPPORTED]' if h2_supported else '[NOT SUPPORTED]'}")

    # Get category with highest ΔH
    scaling_summary = results['comparison']['scaling_summary']
    max_category = None
    max_delta_h = -float('inf')

    for category, data in scaling_summary.items():
        # Get ΔH for largest model (last in list)
        category_max = data['delta_h_5shot'][-1]
        if category_max > max_delta_h:
            max_delta_h = category_max
            max_category = category

    print(f"\n   Category with largest DeltaH: {max_category.capitalize()} "
          f"({max_delta_h:.3f} bits at 5-shot)")

    print("\n[NEXT STEPS]:")
    print("   1. Generate figures:")
    print("      python paper/generate_figures.py --results outputs/multi_model_icl/results.json")
    print("   2. Generate tables:")
    print("      python paper/generate_tables.py --results outputs/multi_model_icl/results.json "
          "--validation outputs/multi_model_icl/hypothesis_validation.json")
    print("   3. Compile paper:")
    print("      python paper/compile_paper.py")

    print("\n" + "="*70 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
