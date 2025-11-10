"""
SNLI Dataset Loader for Logical Reasoning Prompts

This module downloads and loads examples from the Stanford Natural Language Inference (SNLI)
corpus for logical reasoning tasks in In-Context Learning experiments.

SNLI Paper: Bowman et al., "A large annotated corpus for learning natural language inference" (EMNLP 2015)

Dataset: 570,152 human-written English sentence pairs
- Train: 550,152 examples
- Validation: 10,000 examples
- Test: 10,000 examples

Labels:
- 0 (entailment): hypothesis logically follows from premise
- 1 (neutral): neither entailment nor contradiction
- 2 (contradiction): hypothesis contradicts premise

This loader downloads from Hugging Face and balances labels equally.
"""

import random
from typing import List, Dict, Optional
import sys


def load_snli_balanced(n_samples: int = 300,
                      split: str = 'validation',
                      seed: int = 42) -> List[Dict]:
    """
    Load balanced SNLI dataset with equal examples per label.

    Downloads SNLI from Hugging Face and extracts n_samples examples
    with balanced label distribution.

    Args:
        n_samples: Total number of samples (must be divisible by 3 for perfect balance)
        split: Dataset split to use ('train', 'validation', or 'test')
        seed: Random seed for sampling

    Returns:
        List of dicts in ICL format: {'prompt', 'answer', 'category', 'source', 'metadata'}
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("[ERROR] 'datasets' library not found. Install with: pip install datasets")
        sys.exit(1)

    print(f"\n[SNLI Loader] Downloading SNLI {split} split from Hugging Face...")

    # Download SNLI
    try:
        snli = load_dataset("stanfordnlp/snli", split=split, trust_remote_code=True)
        print(f"[SNLI Loader] Downloaded {len(snli)} examples")
    except Exception as e:
        print(f"[ERROR] Failed to download SNLI: {e}")
        print("[INFO] Make sure you have internet connection and 'datasets' installed")
        return []

    # Label mapping
    label_map = {
        0: 'entailment',
        1: 'neutral',
        2: 'contradiction'
    }

    # Target: n_samples // 3 examples per label for perfect balance
    n_per_label = n_samples // 3

    # Collect examples by label
    samples_by_label = {0: [], 1: [], 2: []}

    print(f"[SNLI Loader] Extracting {n_per_label} examples per label...")

    random.seed(seed)
    indices = list(range(len(snli)))
    random.shuffle(indices)

    for idx in indices:
        item = snli[idx]
        label = item['label']

        # Skip invalid labels (-1)
        if label == -1:
            continue

        # Check if we need more of this label
        if len(samples_by_label[label]) < n_per_label:
            premise = item['premise']
            hypothesis = item['hypothesis']

            # Skip if missing data
            if not premise or not hypothesis:
                continue

            # Format prompt in ICL style
            prompt = f"Premise: {premise}\nHypothesis: {hypothesis}\nRelation:"

            samples_by_label[label].append({
                'prompt': prompt,
                'answer': label_map[label],
                'category': 'logical',
                'source': 'snli',
                'metadata': {
                    'premise': premise,
                    'hypothesis': hypothesis,
                    'label_id': label
                }
            })

        # Check if we have enough examples for all labels
        if all(len(samples) >= n_per_label for samples in samples_by_label.values()):
            break

    # Combine all samples
    all_samples = []
    for label, samples in samples_by_label.items():
        all_samples.extend(samples)
        print(f"  - {label_map[label]}: {len(samples)} examples")

    # Shuffle combined samples
    random.shuffle(all_samples)

    print(f"[SNLI Loader] Total loaded: {len(all_samples)} logical prompts\n")

    return all_samples


def validate_snli_data(examples: List[Dict]) -> Dict:
    """
    Validate loaded SNLI data and return statistics.

    Args:
        examples: List of loaded examples

    Returns:
        Dict with validation statistics
    """
    stats = {
        'total': len(examples),
        'by_label': {'entailment': 0, 'neutral': 0, 'contradiction': 0},
        'avg_prompt_length': 0,
        'avg_premise_length': 0,
        'avg_hypothesis_length': 0
    }

    prompt_lengths = []
    premise_lengths = []
    hypothesis_lengths = []

    for ex in examples:
        # Count by label
        label = ex['answer']
        stats['by_label'][label] += 1

        # Track lengths
        prompt_lengths.append(len(ex['prompt']))
        premise_lengths.append(len(ex['metadata']['premise']))
        hypothesis_lengths.append(len(ex['metadata']['hypothesis']))

    stats['avg_prompt_length'] = sum(prompt_lengths) / len(prompt_lengths) if prompt_lengths else 0
    stats['avg_premise_length'] = sum(premise_lengths) / len(premise_lengths) if premise_lengths else 0
    stats['avg_hypothesis_length'] = sum(hypothesis_lengths) / len(hypothesis_lengths) if hypothesis_lengths else 0

    return stats


if __name__ == '__main__':
    # Test the loader
    print("="*70)
    print("SNLI LOGICAL LOADER - TEST")
    print("="*70)

    examples = load_snli_balanced(n_samples=300)

    print("\n" + "="*70)
    print("VALIDATION STATISTICS")
    print("="*70)

    stats = validate_snli_data(examples)

    print(f"Total examples: {stats['total']}")
    print(f"Label distribution:")
    for label, count in stats['by_label'].items():
        percentage = (count / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f"  - {label}: {count} ({percentage:.1f}%)")
    print(f"Avg prompt length: {stats['avg_prompt_length']:.1f} chars")
    print(f"Avg premise length: {stats['avg_premise_length']:.1f} chars")
    print(f"Avg hypothesis length: {stats['avg_hypothesis_length']:.1f} chars")

    print("\n" + "="*70)
    print("SAMPLE EXAMPLES (first 3)")
    print("="*70)

    for i, ex in enumerate(examples[:3], 1):
        print(f"\n{i}. Label: {ex['answer']}")
        print(f"   Premise: {ex['metadata']['premise']}")
        print(f"   Hypothesis: {ex['metadata']['hypothesis']}")
        print(f"   Full prompt: {ex['prompt'][:100]}...")
