"""
LAMA Dataset Loader for Factual Knowledge Prompts

This module extracts factual knowledge prompts from the LAMA (Language Model Analysis)
benchmark, specifically from TREx (T-REx) relations.

LAMA Paper: Petroni et al., "Language Models as Knowledge Bases?" (EMNLP 2019)

Supported Relations:
- P19: place of birth
- P37: official language
- P106: occupation
- P36: capital

Each relation provides ~250 examples for a total of 1,000 factual prompts.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Optional
import sys


# Relation metadata
RELATION_INFO = {
    'P19': {
        'label': 'place of birth',
        'template': '[X] was born in [Y]',
        'file': 'P19.jsonl'
    },
    'P37': {
        'label': 'official language',
        'template': 'The official language of [X] is [Y]',
        'file': 'P37.jsonl'
    },
    'P106': {
        'label': 'occupation',
        'template': '[X] is a [Y] by profession',
        'file': 'P106.jsonl'
    },
    'P36': {
        'label': 'capital',
        'template': 'The capital of [X] is [Y]',
        'file': 'P36.jsonl'
    }
}


def load_trex_relation(relation_id: str, data_dir: str = 'data/lama_data/data/TREx',
                       max_samples: int = 250, seed: int = 42) -> List[Dict]:
    """
    Load examples from a single TREx relation file.

    Args:
        relation_id: Relation ID (e.g., 'P19', 'P37')
        data_dir: Directory containing TREx JSONL files
        max_samples: Maximum number of samples to extract
        seed: Random seed for sampling

    Returns:
        List of dicts in ICL format: {'prompt', 'answer', 'category', 'source', 'metadata'}
    """
    if relation_id not in RELATION_INFO:
        raise ValueError(f"Unknown relation: {relation_id}. Supported: {list(RELATION_INFO.keys())}")

    relation = RELATION_INFO[relation_id]
    file_path = Path(data_dir) / relation['file']

    if not file_path.exists():
        raise FileNotFoundError(f"TREx file not found: {file_path}")

    examples = []

    # Read JSONL file
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())

                    # Extract components
                    obj_label = entry.get('obj_label', '')
                    sub_label = entry.get('sub_label', '')
                    evidences = entry.get('evidences', [])

                    # Quality filters
                    if not obj_label or not sub_label:
                        continue

                    # Answer length filter (avoid very long answers)
                    if len(obj_label) > 50:
                        continue

                    # Skip if no evidences
                    if not evidences:
                        continue

                    # Get masked sentence from first evidence
                    masked_sent = evidences[0].get('masked_sentence', '')
                    if not masked_sent or '[MASK]' not in masked_sent:
                        continue

                    # Convert [MASK] format to ICL prompt
                    # Strategy: Remove [MASK] and everything after it, clean up
                    prompt = masked_sent.replace('[MASK]', '').strip()

                    # Remove trailing punctuation if present
                    if prompt.endswith(',') or prompt.endswith('.'):
                        prompt = prompt[:-1].strip()

                    # Additional cleanup: remove trailing "is", "was", etc.
                    # to make prompt more natural
                    trailing_words = [' is', ' was', ' in', ' at', ' of', ' the']
                    for word in trailing_words:
                        if prompt.endswith(word):
                            prompt = prompt[:-len(word)].strip()

                    # Ensure prompt ends cleanly
                    if not prompt:
                        continue

                    examples.append({
                        'prompt': prompt,
                        'answer': obj_label,
                        'category': 'factual',
                        'source': f'lama-trex-{relation_id}',
                        'metadata': {
                            'relation': relation_id,
                            'relation_label': relation['label'],
                            'subject': sub_label,
                            'uuid': entry.get('uuid', '')
                        }
                    })

                except (json.JSONDecodeError, KeyError, UnicodeDecodeError) as e:
                    # Skip malformed entries
                    continue

    except Exception as e:
        print(f"[ERROR] Failed to load {relation_id}: {e}")
        return []

    # Sample if we have too many
    if len(examples) > max_samples:
        random.seed(seed)
        examples = random.sample(examples, max_samples)

    return examples


def load_lama_factual(n_samples: int = 1000,
                     data_dir: str = 'data/lama_data/data/TREx',
                     seed: int = 42) -> List[Dict]:
    """
    Load factual knowledge prompts from LAMA TREx.

    Extracts examples from 4 high-quality relations:
    - P19 (place of birth): 250 examples
    - P37 (official language): 250 examples
    - P106 (occupation): 250 examples
    - P36 (capital): 250 examples

    Total: 1,000 factual prompts

    Args:
        n_samples: Total number of samples to load (must be divisible by 4)
        data_dir: Directory containing TREx data
        seed: Random seed for sampling

    Returns:
        List of dicts in ICL format
    """
    # Distribute samples evenly across relations
    samples_per_relation = n_samples // 4

    all_examples = []

    print(f"\n[LAMA Loader] Loading {n_samples} factual prompts from TREx...")

    for relation_id in ['P19', 'P37', 'P106', 'P36']:
        print(f"  - Loading {relation_id} ({RELATION_INFO[relation_id]['label']}): ", end='')

        try:
            examples = load_trex_relation(
                relation_id,
                data_dir=data_dir,
                max_samples=samples_per_relation,
                seed=seed
            )
            all_examples.extend(examples)
            print(f"{len(examples)} examples")

        except Exception as e:
            print(f"[FAILED] {e}")
            continue

    # Shuffle all examples
    random.seed(seed)
    random.shuffle(all_examples)

    print(f"[LAMA Loader] Total loaded: {len(all_examples)} factual prompts\n")

    return all_examples


def validate_lama_data(examples: List[Dict]) -> Dict:
    """
    Validate loaded LAMA data and return statistics.

    Args:
        examples: List of loaded examples

    Returns:
        Dict with validation statistics
    """
    stats = {
        'total': len(examples),
        'by_relation': {},
        'avg_prompt_length': 0,
        'avg_answer_length': 0,
        'unique_subjects': set()
    }

    prompt_lengths = []
    answer_lengths = []

    for ex in examples:
        # Count by relation
        relation = ex['metadata']['relation']
        stats['by_relation'][relation] = stats['by_relation'].get(relation, 0) + 1

        # Track lengths
        prompt_lengths.append(len(ex['prompt']))
        answer_lengths.append(len(ex['answer']))

        # Track unique subjects
        stats['unique_subjects'].add(ex['metadata']['subject'])

    stats['avg_prompt_length'] = sum(prompt_lengths) / len(prompt_lengths) if prompt_lengths else 0
    stats['avg_answer_length'] = sum(answer_lengths) / len(answer_lengths) if answer_lengths else 0
    stats['unique_subjects'] = len(stats['unique_subjects'])

    return stats


if __name__ == '__main__':
    # Test the loader
    print("="*70)
    print("LAMA FACTUAL LOADER - TEST")
    print("="*70)

    examples = load_lama_factual(n_samples=1000)

    print("\n" + "="*70)
    print("VALIDATION STATISTICS")
    print("="*70)

    stats = validate_lama_data(examples)

    print(f"Total examples: {stats['total']}")
    print(f"By relation:")
    for rel, count in stats['by_relation'].items():
        print(f"  - {rel} ({RELATION_INFO[rel]['label']}): {count}")
    print(f"Unique subjects: {stats['unique_subjects']}")
    print(f"Avg prompt length: {stats['avg_prompt_length']:.1f} chars")
    print(f"Avg answer length: {stats['avg_answer_length']:.1f} chars")

    print("\n" + "="*70)
    print("SAMPLE EXAMPLES (first 5)")
    print("="*70)

    for i, ex in enumerate(examples[:5], 1):
        print(f"\n{i}. [{ex['metadata']['relation']}] {ex['metadata']['relation_label']}")
        print(f"   Prompt: {ex['prompt']}")
        print(f"   Answer: {ex['answer']}")
        print(f"   Subject: {ex['metadata']['subject']}")
