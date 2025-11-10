"""
Project Gutenberg Poetry Loader for Creative Text Generation

This module samples poetry lines from the Gutenberg Poetry corpus for creative
text generation tasks in In-Context Learning experiments.

Dataset: gutenberg-poetry-v001.ndjson.gz
- Total lines: 3,085,117
- Books: 1,191 classical literature works
- Format: NDJSON (newline-delimited JSON)

Each entry contains:
- s: Single line of poetry
- gid: Gutenberg book ID

This loader uses stratified sampling to ensure diversity across books,
extracting line pairs for next-line prediction tasks.
"""

import gzip
import json
import random
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict


def load_gutenberg_poetry(n_samples: int = 500,
                          n_books: int = 50,
                          data_file: str = 'data/gutenberg-poetry-v001.ndjson.gz',
                          seed: int = 42) -> List[Dict]:
    """
    Load poetry lines from Project Gutenberg with stratified sampling.

    Uses stratified sampling to ensure diversity:
    - Sample n_books different books
    - Extract (n_samples // n_books) line pairs per book
    - Total: n_samples line pairs for next-line prediction

    Args:
        n_samples: Total number of line pairs to extract
        n_books: Number of different books to sample from
        data_file: Path to compressed NDJSON file
        seed: Random seed for sampling

    Returns:
        List of dicts in ICL format: {'prompt', 'answer', 'category', 'source', 'metadata'}
    """
    data_path = Path(data_file)

    if not data_path.exists():
        print(f"[ERROR] Gutenberg poetry file not found: {data_path}")
        return []

    print(f"\n[Gutenberg Loader] Loading {n_samples} poetry lines from {n_books} books...")
    print(f"[Gutenberg Loader] Reading from: {data_path}")

    random.seed(seed)

    # Calculate lines per book
    lines_per_book = n_samples // n_books

    # Step 1: Group lines by book (streaming to avoid loading all 3M lines)
    print("[Gutenberg Loader] Grouping lines by book (streaming)...")

    books = defaultdict(list)
    total_lines_read = 0

    try:
        with gzip.open(data_path, 'rt', encoding='utf-8') as f:
            for line in f:
                total_lines_read += 1

                try:
                    entry = json.loads(line.strip())
                    text = entry.get('s', '').strip()
                    gid = entry.get('gid', '')

                    # Filter quality
                    if not text or not gid:
                        continue

                    # Skip very short lines (likely metadata/errors)
                    if len(text) < 5:
                        continue

                    # Skip lines that are just numbers or punctuation
                    if text.replace(' ', '').replace('.', '').replace(',', '').isdigit():
                        continue

                    books[gid].append(text)

                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue

                # Progress indicator
                if total_lines_read % 100000 == 0:
                    print(f"  - Read {total_lines_read:,} lines, found {len(books)} books...")

    except Exception as e:
        print(f"[ERROR] Failed to read Gutenberg file: {e}")
        return []

    print(f"[Gutenberg Loader] Finished reading {total_lines_read:,} lines")
    print(f"[Gutenberg Loader] Found {len(books)} books with valid lines")

    # Step 2: Select n_books randomly
    if len(books) < n_books:
        print(f"[WARNING] Only {len(books)} books available, using all")
        selected_book_ids = list(books.keys())
    else:
        selected_book_ids = random.sample(list(books.keys()), n_books)

    print(f"[Gutenberg Loader] Selected {len(selected_book_ids)} books for sampling")

    # Step 3: Extract line pairs from each selected book
    all_examples = []

    for gid in selected_book_ids:
        book_lines = books[gid]

        # Need at least 2 consecutive lines for pair
        if len(book_lines) < 2:
            continue

        # Create pairs: (line[i], line[i+1])
        pairs = []
        for i in range(len(book_lines) - 1):
            line1 = book_lines[i]
            line2 = book_lines[i + 1]

            # Both lines should be non-empty
            if line1.strip() and line2.strip():
                pairs.append((line1, line2))

        # Sample lines_per_book pairs from this book
        if len(pairs) > lines_per_book:
            sampled_pairs = random.sample(pairs, lines_per_book)
        else:
            sampled_pairs = pairs

        # Convert to ICL format
        for line1, line2 in sampled_pairs:
            all_examples.append({
                'prompt': line1,
                'answer': line2,
                'category': 'creative',
                'source': 'gutenberg-poetry',
                'metadata': {
                    'gutenberg_id': gid,
                    'pair_type': 'consecutive_lines'
                }
            })

    # Shuffle final examples
    random.shuffle(all_examples)

    # Trim to exact n_samples if needed
    if len(all_examples) > n_samples:
        all_examples = all_examples[:n_samples]

    print(f"[Gutenberg Loader] Extracted {len(all_examples)} line pairs")

    # Show distribution
    book_counts = defaultdict(int)
    for ex in all_examples:
        book_counts[ex['metadata']['gutenberg_id']] += 1

    print(f"[Gutenberg Loader] Average lines per book: {len(all_examples) / len(book_counts):.1f}")
    print(f"[Gutenberg Loader] Book ID range: {min(book_counts.keys())} - {max(book_counts.keys())}\n")

    return all_examples


def validate_gutenberg_data(examples: List[Dict]) -> Dict:
    """
    Validate loaded Gutenberg data and return statistics.

    Args:
        examples: List of loaded examples

    Returns:
        Dict with validation statistics
    """
    stats = {
        'total': len(examples),
        'unique_books': set(),
        'by_book': defaultdict(int),
        'avg_prompt_length': 0,
        'avg_answer_length': 0,
        'min_prompt_length': float('inf'),
        'max_prompt_length': 0
    }

    prompt_lengths = []
    answer_lengths = []

    for ex in examples:
        gid = ex['metadata']['gutenberg_id']
        stats['unique_books'].add(gid)
        stats['by_book'][gid] += 1

        prompt_len = len(ex['prompt'])
        answer_len = len(ex['answer'])

        prompt_lengths.append(prompt_len)
        answer_lengths.append(answer_len)

        stats['min_prompt_length'] = min(stats['min_prompt_length'], prompt_len)
        stats['max_prompt_length'] = max(stats['max_prompt_length'], prompt_len)

    stats['avg_prompt_length'] = sum(prompt_lengths) / len(prompt_lengths) if prompt_lengths else 0
    stats['avg_answer_length'] = sum(answer_lengths) / len(answer_lengths) if answer_lengths else 0
    stats['unique_books'] = len(stats['unique_books'])

    return stats


if __name__ == '__main__':
    # Test the loader
    print("="*70)
    print("GUTENBERG POETRY LOADER - TEST")
    print("="*70)

    examples = load_gutenberg_poetry(n_samples=500, n_books=50)

    print("\n" + "="*70)
    print("VALIDATION STATISTICS")
    print("="*70)

    stats = validate_gutenberg_data(examples)

    print(f"Total examples: {stats['total']}")
    print(f"Unique books: {stats['unique_books']}")
    print(f"Avg lines per book: {stats['total'] / stats['unique_books']:.1f}")
    print(f"Avg prompt length: {stats['avg_prompt_length']:.1f} chars")
    print(f"Avg answer length: {stats['avg_answer_length']:.1f} chars")
    print(f"Prompt length range: {stats['min_prompt_length']} - {stats['max_prompt_length']}")

    print("\n" + "="*70)
    print("SAMPLE EXAMPLES (first 5)")
    print("="*70)

    for i, ex in enumerate(examples[:5], 1):
        print(f"\n{i}. Book ID: {ex['metadata']['gutenberg_id']}")
        print(f"   Prompt: {ex['prompt']}")
        print(f"   Answer: {ex['answer']}")
