"""
Dataset Manager

This module manages real datasets for uncertainty analysis.

Data Sources (in priority order):
- data/lama_data/data/TREx/*.jsonl: Thousands of factual knowledge triples (primary for factual)
- data/gutenberg-poetry-v001.ndjson.gz: 3+ million poetry verses (primary for creative)
- data/consolidated_datasets.json: Consolidated samples (fallback, SNLI source)

Loading Strategy:
1. Try to load from original files (TREx, Gutenberg)
2. Fall back to consolidated_datasets.json if originals not available
3. Return up to n_samples per category
"""

import json
import gzip
import random
from pathlib import Path
from typing import List, Dict, Optional


class RealDatasetManager:
    """
    Manager for real datasets already downloaded in the repository.

    Files used:
    - data/lama_data/data/TREx/*.jsonl: Thousands of factual knowledge triples
    - data/gutenberg-poetry-v001.ndjson.gz: 3+ million poetry verses
    - data/consolidated_datasets.json: Consolidated samples (SNLI source, fallback)

    The manager loads data directly from original files when available
    to obtain up to 50 balanced samples per category.

    Example:
        >>> manager = RealDatasetManager(data_dir="data")
        >>> factual = manager.load_factual_data(n_samples=50)
        >>> logical = manager.load_logical_data(n_samples=50)
        >>> creative = manager.load_creative_data(n_samples=50)
        >>> all_data = manager.load_all_datasets(n_per_category=50)
    """

    def __init__(self, data_dir: str = "data"):
        """
        Initialize the dataset manager.

        Args:
            data_dir: Directory where data files are located

        Raises:
            FileNotFoundError: If consolidated file is not found
        """
        self.data_dir = Path(data_dir)
        self.consolidated_file = self.data_dir / "consolidated_datasets.json"
        self.poetry_file = self.data_dir / "gutenberg-poetry-v001.ndjson.gz"
        self.lama_dir = self.data_dir / "lama_data" / "data" / "TREx"

        # Validate that consolidated file exists (required)
        if not self.consolidated_file.exists():
            raise FileNotFoundError(
                f"❌ Consolidated file not found: {self.consolidated_file}\n"
                f"   Make sure the file exists in the 'data/' directory"
            )

        print("=" * 80)
        print("INITIALIZING RealDatasetManager")
        print("=" * 80)
        print(f"✓ Consolidated file: {self.consolidated_file}")
        print(f"  {'✓' if self.poetry_file.exists() else '⚠️'} Poetry file: {self.poetry_file}")
        print(f"  {'✓' if self.lama_dir.exists() else '⚠️'} LAMA TREx dir: {self.lama_dir}")

        # Load and validate consolidated file structure
        self._validate_consolidated_file()

    def _validate_consolidated_file(self):
        """Validate consolidated file structure and show statistics."""
        try:
            with open(self.consolidated_file, 'r', encoding='utf-8') as f:
                all_data = json.load(f)

            # Count by source
            sources = {}
            for item in all_data:
                source = item.get('source', 'unknown')
                sources[source] = sources.get(source, 0) + 1

            print(f"\n📊 Consolidated file statistics:")
            print(f"   Total samples: {len(all_data)}")
            print(f"   Distribution by source:")
            for source, count in sorted(sources.items()):
                print(f"     - {source}: {count} samples")
            print("=" * 80 + "\n")

        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing {self.consolidated_file}: {e}")

    def load_factual_data(self, n_samples: int = 50) -> List[Dict]:
        """
        Load factual data from LAMA TREx files.

        Primary source: data/lama_data/data/TREx/*.jsonl (thousands available)
        Fallback: consolidated_datasets.json

        Args:
            n_samples: Maximum number of samples to return

        Returns:
            List of dictionaries with factual data
        """
        # First try to load from direct LAMA files
        if self.lama_dir.exists():
            # Load from TREx files
            all_lama_samples = []
            trex_files = sorted(self.lama_dir.glob("*.jsonl"))[:10]  # Use first 10 files

            for trex_file in trex_files:
                with open(trex_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    for line in lines[:10]:  # 10 samples per file
                        try:
                            item = json.loads(line)
                            # Create prompt without [MASK]
                            if 'masked_sentences' in item and item['masked_sentences']:
                                prompt = item['masked_sentences'][0].replace('[MASK]', '').strip()
                                prompt = ' '.join(prompt.split())
                            elif 'sub_label' in item and 'predicate_id' in item:
                                prompt = f"{item['sub_label']} {item['predicate_id']}"
                            else:
                                continue

                            all_lama_samples.append({
                                'prompt': prompt,
                                'answer': item.get('obj_label', ''),
                                'category': 'factual',
                                'source': 'lama',
                                'metadata': {
                                    'predicate': item.get('predicate_id', ''),
                                    'subject': item.get('sub_label', '')
                                }
                            })

                            if len(all_lama_samples) >= n_samples:
                                break
                        except json.JSONDecodeError:
                            continue

                if len(all_lama_samples) >= n_samples:
                    break

            print(f"📚 Factual data loaded from TREx: {len(all_lama_samples)} LAMA samples")
            print(f"   Returning: {min(n_samples, len(all_lama_samples))} samples")
            return all_lama_samples[:n_samples]

        else:
            # Fallback: use consolidated file
            with open(self.consolidated_file, 'r', encoding='utf-8') as f:
                all_data = json.load(f)

            factual = [item for item in all_data if item.get('source') == 'lama-synthetic']

            print(f"📚 Factual data loaded from consolidated: {len(factual)} samples")
            print(f"   ⚠️  WARNING: Only {len(factual)} samples available (requested: {n_samples})")
            print(f"   Returning: {min(n_samples, len(factual))} samples")

            return factual[:n_samples]

    def load_logical_data(self, n_samples: int = 50) -> List[Dict]:
        """
        Load logical data from consolidated_datasets.json (source='snli').

        Note: SNLI data is only available in consolidated file (limited to ~50 samples).

        Args:
            n_samples: Maximum number of samples to return

        Returns:
            List of dictionaries with logical data
        """
        with open(self.consolidated_file, 'r', encoding='utf-8') as f:
            all_data = json.load(f)

        logical = [item for item in all_data if item.get('source') == 'snli']

        print(f"🧠 Logical data loaded: {len(logical)} SNLI samples")

        # Warning if insufficient samples
        if len(logical) < n_samples:
            print(f"   ⚠️  LIMITATION: Only {len(logical)} SNLI samples available in consolidated file")
            print(f"   Requested: {n_samples}, Returning: {len(logical)}")
            print(f"   Note: To increase sample size, download full SNLI dataset")
        else:
            print(f"   Returning: {min(n_samples, len(logical))} samples")

        if len(logical) == 0:
            print("   ⚠️  WARNING: No data found with source='snli'")

        return logical[:n_samples]

    def load_creative_data(self, n_samples: int = 50) -> List[Dict]:
        """
        Load creative data from gutenberg-poetry-v001.ndjson.gz.

        Primary source: gutenberg-poetry-v001.ndjson.gz (3M+ verses)
        Fallback: consolidated_datasets.json

        Args:
            n_samples: Maximum number of samples to return

        Returns:
            List of dictionaries with creative data
        """
        # Try to load from complete Gutenberg file
        if self.poetry_file.exists():
            random.seed(42)  # For reproducibility

            print(f"🎨 Loading from complete Gutenberg Poetry file...")
            creative_samples = []

            with gzip.open(self.poetry_file, 'rt', encoding='utf-8') as f:
                # Read first n_samples*10 lines and select randomly
                lines = []
                for i, line in enumerate(f):
                    lines.append(line)
                    if i >= n_samples * 10:  # Read 10x more for variety
                        break

                # Randomly select n_samples
                selected_lines = random.sample(lines, min(n_samples, len(lines)))

                for line in selected_lines:
                    try:
                        poem = json.loads(line)
                        # Use 's' field (poetry line) as prompt
                        if 's' in poem and poem['s'].strip():
                            creative_samples.append({
                                'prompt': poem['s'].strip(),
                                'answer': None,  # Poetry doesn't have unique answer
                                'category': 'creative',
                                'source': 'gutenberg-poetry',
                                'metadata': {
                                    'author': poem.get('a', 'Unknown'),
                                    'title': poem.get('t', 'Untitled')
                                }
                            })
                    except json.JSONDecodeError:
                        continue

            print(f"🎨 Creative data loaded from complete file: {len(creative_samples)} samples")
            print(f"   Returning: {min(n_samples, len(creative_samples))} samples")
            return creative_samples[:n_samples]

        else:
            # Fallback: use consolidated file
            with open(self.consolidated_file, 'r', encoding='utf-8') as f:
                all_data = json.load(f)

            creative = [item for item in all_data if item.get('source') == 'gutenberg-poetry']

            print(f"🎨 Creative data loaded from consolidated: {len(creative)} samples")
            print(f"   ⚠️  WARNING: Only {len(creative)} samples available (requested: {n_samples})")
            print(f"   Returning: {min(n_samples, len(creative))} samples")

            return creative[:n_samples]

    def load_all_datasets(self, n_per_category: int = 50) -> Dict[str, List[Dict]]:
        """
        Load all datasets at once.

        Args:
            n_per_category: Number of samples per category

        Returns:
            Dictionary with keys 'factual', 'logical', 'creative'
        """
        print("\n" + "=" * 80)
        print("LOADING ALL DATASETS")
        print("=" * 80 + "\n")

        datasets = {
            'factual': self.load_factual_data(n_per_category),
            'logical': self.load_logical_data(n_per_category),
            'creative': self.load_creative_data(n_per_category)
        }

        # Calculate total dynamically (NO hardcoded numbers)
        total = sum(len(data) for data in datasets.values())

        print(f"\n✅ Total samples loaded: {total}")
        print(f"   - Factual: {len(datasets['factual'])} samples")
        print(f"   - Logical: {len(datasets['logical'])} samples")
        print(f"   - Creative: {len(datasets['creative'])} samples")
        print("=" * 80 + "\n")

        return datasets

    def get_sample_info(self, category: str, index: int = 0):
        """
        Show detailed information about a specific sample.

        Args:
            category: 'factual', 'logical', or 'creative'
            index: Sample index
        """
        if category == 'factual':
            data = self.load_factual_data(index + 1)
        elif category == 'logical':
            data = self.load_logical_data(index + 1)
        elif category == 'creative':
            data = self.load_creative_data(index + 1)
        else:
            raise ValueError(f"Invalid category: {category}")

        if index >= len(data):
            print(f"⚠️  Index {index} out of range (only {len(data)} samples)")
            return

        sample = data[index]
        print(f"\n📋 SAMPLE {index} - CATEGORY: {category.upper()}")
        print("=" * 80)
        for key, value in sample.items():
            if isinstance(value, str) and len(value) > 100:
                print(f"{key}: {value[:100]}...")
            else:
                print(f"{key}: {value}")
        print("=" * 80 + "\n")