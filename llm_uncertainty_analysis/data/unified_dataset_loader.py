"""
Unified Dataset Loader for ICL Multi-Model Analysis

This module provides a unified API to load all three datasets:
1. LAMA TREx (factual knowledge)
2. Stanford SNLI (logical reasoning)
3. Project Gutenberg Poetry (creative generation)

It consolidates the individual loaders and provides a single entry point
for the ICL experiments.

Usage:
    from llm_uncertainty_analysis.data.unified_dataset_loader import load_all_datasets

    datasets = load_all_datasets(factual_n=1000, logical_n=300, creative_n=500)

    # Access individual categories
    factual_examples = datasets['factual']
    logical_examples = datasets['logical']
    creative_examples = datasets['creative']
"""

from typing import Dict, List, Optional
import sys
from pathlib import Path

# Import individual loaders
try:
    from .lama_loader import load_lama_factual, validate_lama_data
    from .snli_loader import load_snli_balanced, validate_snli_data
    from .gutenberg_loader import load_gutenberg_poetry, validate_gutenberg_data
except ImportError:
    # Fallback for direct execution
    from lama_loader import load_lama_factual, validate_lama_data
    from snli_loader import load_snli_balanced, validate_snli_data
    from gutenberg_loader import load_gutenberg_poetry, validate_gutenberg_data


def load_all_datasets(factual_n: int = 1000,
                     logical_n: int = 300,
                     creative_n: int = 500,
                     seed: int = 42,
                     lama_dir: str = 'data/lama_data/data/TREx',
                     gutenberg_file: str = 'data/gutenberg-poetry-v001.ndjson.gz',
                     snli_split: str = 'validation') -> Dict[str, List[Dict]]:
    """
    Load all three datasets for ICL multi-model analysis.

    This is the main entry point for loading datasets. It coordinates
    the three individual loaders and returns a unified dictionary.

    Args:
        factual_n: Number of factual prompts from LAMA (default: 1000)
        logical_n: Number of logical prompts from SNLI (default: 300)
        creative_n: Number of creative prompts from Gutenberg (default: 500)
        seed: Random seed for reproducibility
        lama_dir: Directory containing LAMA TREx data
        gutenberg_file: Path to Gutenberg poetry compressed file
        snli_split: SNLI split to use ('train', 'validation', or 'test')

    Returns:
        Dictionary with three keys:
            'factual': List of LAMA examples
            'logical': List of SNLI examples
            'creative': List of Gutenberg examples

    Example:
        >>> datasets = load_all_datasets(factual_n=1000, logical_n=300, creative_n=500)
        >>> len(datasets['factual'])
        1000
        >>> datasets['factual'][0].keys()
        dict_keys(['prompt', 'answer', 'category', 'source', 'metadata'])
    """
    print("\n" + "="*70)
    print("UNIFIED DATASET LOADER")
    print("="*70)
    print(f"Target sizes: Factual={factual_n}, Logical={logical_n}, Creative={creative_n}")
    print(f"Random seed: {seed}")
    print("="*70)

    datasets = {}

    # 1. Load LAMA (Factual)
    print("\n[1/3] Loading LAMA (Factual Knowledge)...")
    try:
        datasets['factual'] = load_lama_factual(
            n_samples=factual_n,
            data_dir=lama_dir,
            seed=seed
        )
        if not datasets['factual']:
            print("[WARNING] No factual examples loaded from LAMA")
    except Exception as e:
        print(f"[ERROR] Failed to load LAMA: {e}")
        datasets['factual'] = []

    # 2. Load SNLI (Logical)
    print("\n[2/3] Loading SNLI (Logical Reasoning)...")
    try:
        datasets['logical'] = load_snli_balanced(
            n_samples=logical_n,
            split=snli_split,
            seed=seed
        )
        if not datasets['logical']:
            print("[WARNING] No logical examples loaded from SNLI")
    except Exception as e:
        print(f"[ERROR] Failed to load SNLI: {e}")
        datasets['logical'] = []

    # 3. Load Gutenberg (Creative)
    print("\n[3/3] Loading Gutenberg (Creative Text)...")
    try:
        n_books = min(50, creative_n // 10)  # Aim for ~10 lines per book
        datasets['creative'] = load_gutenberg_poetry(
            n_samples=creative_n,
            n_books=n_books,
            data_file=gutenberg_file,
            seed=seed
        )
        if not datasets['creative']:
            print("[WARNING] No creative examples loaded from Gutenberg")
    except Exception as e:
        print(f"[ERROR] Failed to load Gutenberg: {e}")
        datasets['creative'] = []

    # Summary
    print("\n" + "="*70)
    print("DATASET LOADING COMPLETE")
    print("="*70)
    print(f"Factual (LAMA):     {len(datasets.get('factual', []))} examples")
    print(f"Logical (SNLI):     {len(datasets.get('logical', []))} examples")
    print(f"Creative (Gutenberg): {len(datasets.get('creative', []))} examples")
    print(f"TOTAL:              {sum(len(v) for v in datasets.values())} examples")
    print("="*70 + "\n")

    return datasets


def get_dataset_statistics(datasets: Dict[str, List[Dict]]) -> Dict:
    """
    Generate comprehensive statistics for loaded datasets.

    Args:
        datasets: Dictionary returned by load_all_datasets()

    Returns:
        Dictionary with detailed statistics for each category
    """
    stats = {}

    if 'factual' in datasets and datasets['factual']:
        stats['factual'] = validate_lama_data(datasets['factual'])
        stats['factual']['source'] = 'LAMA TREx'

    if 'logical' in datasets and datasets['logical']:
        stats['logical'] = validate_snli_data(datasets['logical'])
        stats['logical']['source'] = 'Stanford SNLI'

    if 'creative' in datasets and datasets['creative']:
        stats['creative'] = validate_gutenberg_data(datasets['creative'])
        stats['creative']['source'] = 'Project Gutenberg Poetry'

    return stats


def export_datasets_to_json(datasets: Dict[str, List[Dict]], output_file: str = 'datasets_export.json'):
    """
    Export loaded datasets to JSON file for inspection or reuse.

    Args:
        datasets: Dictionary returned by load_all_datasets()
        output_file: Path to output JSON file
    """
    import json

    print(f"\n[Export] Saving datasets to {output_file}...")

    output_data = {
        'metadata': {
            'total_examples': sum(len(v) for v in datasets.values()),
            'categories': {
                'factual': len(datasets.get('factual', [])),
                'logical': len(datasets.get('logical', [])),
                'creative': len(datasets.get('creative', []))
            }
        },
        'datasets': datasets
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"[Export] Saved {output_data['metadata']['total_examples']} examples to {output_file}")


if __name__ == '__main__':
    # Test the unified loader
    print("="*70)
    print("UNIFIED DATASET LOADER - TEST")
    print("="*70)

    # Load all datasets
    datasets = load_all_datasets(
        factual_n=1000,
        logical_n=300,
        creative_n=500,
        seed=42
    )

    # Generate statistics
    print("\n" + "="*70)
    print("DETAILED STATISTICS")
    print("="*70)

    stats = get_dataset_statistics(datasets)

    for category, cat_stats in stats.items():
        print(f"\n{category.upper()}:")
        print(f"  Source: {cat_stats.get('source', 'Unknown')}")
        print(f"  Total: {cat_stats.get('total', 0)}")

        if category == 'factual':
            print(f"  By relation:")
            for rel, count in cat_stats.get('by_relation', {}).items():
                from lama_loader import RELATION_INFO
                print(f"    - {rel} ({RELATION_INFO[rel]['label']}): {count}")

        elif category == 'logical':
            print(f"  By label:")
            for label, count in cat_stats.get('by_label', {}).items():
                print(f"    - {label}: {count}")

        elif category == 'creative':
            print(f"  Unique books: {cat_stats.get('unique_books', 0)}")

        print(f"  Avg prompt length: {cat_stats.get('avg_prompt_length', 0):.1f} chars")
        print(f"  Avg answer length: {cat_stats.get('avg_answer_length', 0):.1f} chars")

    # Show sample from each category
    print("\n" + "="*70)
    print("SAMPLE EXAMPLES (1 per category)")
    print("="*70)

    for category in ['factual', 'logical', 'creative']:
        if category in datasets and datasets[category]:
            ex = datasets[category][0]
            print(f"\n{category.upper()}:")
            print(f"  Source: {ex['source']}")
            print(f"  Prompt: {ex['prompt'][:100]}...")
            print(f"  Answer: {ex['answer']}")

    # Optional: Export to JSON
    # export_datasets_to_json(datasets, 'loaded_datasets.json')
