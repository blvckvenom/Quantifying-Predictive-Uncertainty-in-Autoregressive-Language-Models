#!/usr/bin/env python3
"""
Main Analysis Script

This script runs the complete LLM uncertainty analysis pipeline.

Usage:
    python run_analysis.py [--model MODEL_NAME] [--data-dir DATA_DIR] [--output-dir OUTPUT_DIR]

Example:
    python run_analysis.py --model gpt2 --data-dir ../data --output-dir ../results
"""

import argparse
from pathlib import Path
import torch

# Import modules
from config import setup_reproducibility, setup_visualization, device, print_device_info
from models import get_available_models, print_model_info, print_category_info
from data_management import RealDatasetManager
from analysis import UncertaintyAnalyzer
from experiments import run_category_comparison, calculate_descriptive_stats
from visualization import plot_entropy_by_category
from utils import ensure_directory


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run LLM Uncertainty Analysis')

    parser.add_argument('--model', type=str, default='gpt2',
                        help='Model to use (default: gpt2)')
    parser.add_argument('--data-dir', type=str, default='../data',
                        help='Data directory (default: ../data)')
    parser.add_argument('--output-dir', type=str, default='../results',
                        help='Output directory (default: ../results)')
    parser.add_argument('--n-samples', type=int, default=50,
                        help='Number of samples per category (default: 50)')

    return parser.parse_args()


def main():
    """Main analysis pipeline."""
    try:
        # Parse arguments
        args = parse_args()

        print("="*80)
        print("LLM UNCERTAINTY ANALYSIS")
        print("="*80)
        print(f"Model: {args.model}")
        print(f"Data directory: {args.data_dir}")
        print(f"Output directory: {args.output_dir}")
        print("="*80 + "\n")

        # 1. Setup
        print("STEP 1: Setup and Configuration")
        print("-"*80)
        setup_reproducibility()
        setup_visualization()
        print_device_info()
        print()

        # 2. Show available models
        print("STEP 2: Model Configuration")
        print("-"*80)
        models = get_available_models(device)
        print_model_info(models)
        print()

        # 3. Show context categories
        print("STEP 3: Context Categories")
        print("-"*80)
        print_category_info()
        print()

        # 4. Load datasets
        print("STEP 4: Load Datasets")
        print("-"*80)
        dataset_manager = RealDatasetManager(data_dir=args.data_dir)
        datasets = dataset_manager.load_all_datasets(n_per_category=args.n_samples)

        # Combine all samples
        all_samples = []
        for category, samples in datasets.items():
            all_samples.extend(samples)

        print(f"Total samples loaded: {len(all_samples)}")
        print()

        # 5. Initialize analyzer
        print("STEP 5: Initialize Uncertainty Analyzer")
        print("-"*80)
        analyzer = UncertaintyAnalyzer(args.model, device=str(device))
        print()

        # 6. Run category comparison experiment
        print("STEP 6: Run Category Comparison Experiment")
        print("-"*80)
        results_df = run_category_comparison(analyzer, all_samples)
        print()

        # 7. Calculate descriptive statistics
        print("STEP 7: Descriptive Statistics")
        print("-"*80)
        stats_dict = calculate_descriptive_stats(results_df)
        for category, stats in stats_dict.items():
            print(f"\n{category.upper()}:")
            for key, value in stats.items():
                print(f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}")
        print()

        # 8. Create visualizations
        print("STEP 8: Create Visualizations")
        print("-"*80)
        output_dir = ensure_directory(args.output_dir)
        fig_dir = ensure_directory(output_dir / "figures")

        # Plot entropy by category
        plot_path = fig_dir / "entropy_by_category.png"
        plot_entropy_by_category(results_df, save_path=str(plot_path))
        print()

        # 9. Save results
        print("STEP 9: Save Results")
        print("-"*80)
        results_df.to_csv(output_dir / "results.csv", index=False)
        print(f"Results saved to: {output_dir / 'results.csv'}")
        print()

        print("="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)
        
        return 0

    except FileNotFoundError as e:
        print(f"\n❌ ERROR: File not found - {e}")
        print("Please check that the data directory exists and contains the required files.")
        print(f"Expected data directory: {args.data_dir}")
        return 1

    except torch.cuda.OutOfMemoryError:
        print(f"\n❌ ERROR: GPU out of memory")
        print("Try one of these solutions:")
        print("  1. Run with CPU: python run_analysis.py --device cpu")
        print("  2. Reduce samples: python run_analysis.py --n-samples 10")
        return 1

    except ImportError as e:
        print(f"\n❌ ERROR: Missing dependency - {e}")
        print("Please install all requirements:")
        print("  pip install -r requirements.txt")
        return 1

    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {e}")
        print("\nFull traceback:")
        import traceback
        traceback.print_exc()
        print("\nIf the problem persists, please report this error.")
        return 1


if __name__ == "__main__":
    exit(main())