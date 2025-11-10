#!/usr/bin/env python3
"""
Complete Analysis Runner

This script runs the FULL analysis pipeline including all statistical tests and visualizations.

Usage:
    python run_complete_analysis.py [--model MODEL] [--data-dir DIR] [--output-dir DIR]
"""

import argparse
from pathlib import Path
import torch

from config import setup_reproducibility, setup_visualization, device, print_device_info
from experiments.complete_analysis import run_complete_analysis


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run Complete LLM Uncertainty Analysis')

    parser.add_argument('--model', type=str, default='gpt2',
                        help='Model to use (default: gpt2)')
    parser.add_argument('--data-dir', type=str, default='../data',
                        help='Data directory (default: ../data)')
    parser.add_argument('--output-dir', type=str, default='../results_complete',
                        help='Output directory (default: ../results_complete)')
    parser.add_argument('--n-samples', type=int, default=50,
                        help='Samples per category (default: 50)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu, default: auto-detect)')

    return parser.parse_args()


def main():
    """Run complete analysis pipeline."""
    try:
        args = parse_args()

        # Setup
        setup_reproducibility()
        setup_visualization()
        print_device_info()

        # Determine device
        device_str = args.device if args.device else str(device)

        print("\n" + "="*80)
        print("CONFIGURATION")
        print("="*80)
        print(f"Model: {args.model}")
        print(f"Data directory: {args.data_dir}")
        print(f"Output directory: {args.output_dir}")
        print(f"Samples per category: {args.n_samples}")
        print(f"Device: {device_str}")
        print("="*80)

        # Run complete analysis
        results = run_complete_analysis(
            model_name=args.model,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            n_samples_per_category=args.n_samples,
            device=device_str
        )

        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)

        # Print key results
        anova = results['statistical_tests']['anova']
        print(f"\nANOVA Results:")
        print(f"  F-statistic: {anova['f_statistic']:.4f}")
        print(f"  P-value: {anova['p_value']:.6f}")
        print(f"  Eta-squared: {anova['eta_squared']:.4f}")

        print(f"\nEffect Sizes (Cohen's d):")
        for result in results['statistical_tests']['cohens_d']:
            print(f"  {result['comparison']}: d = {result['cohens_d']:.4f} ({result['magnitude']})")

        print("\n" + "="*80)
        print("All results saved successfully!")
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
        print("  1. Run with CPU: python run_complete_analysis.py --device cpu")
        print("  2. Reduce samples: python run_complete_analysis.py --n-samples 10")
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