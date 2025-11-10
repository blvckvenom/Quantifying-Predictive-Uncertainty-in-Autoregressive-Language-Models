"""
In-Context Learning Category Configurations

This module defines configurations for ICL experiments across three categories:
1. Factual Knowledge: Completion of factual statements (LAMA TREx)
2. Logical Reasoning: Natural language inference (Stanford SNLI)
3. Creative Generation: Poetry continuation (Project Gutenberg)

Each configuration includes:
- task_description: Instructions for the task
- examples: Training examples (query, answer) pairs for k-shot learning
- queries: Test queries to evaluate ICL effectiveness
- metadata: Dataset provenance information

NEW: Now supports loading real datasets from LAMA, SNLI, and Gutenberg.
Set USE_REAL_DATASETS=True to load from established benchmarks.

Usage:
    from llm_uncertainty_analysis.experiments.icl_category_configs import CATEGORY_CONFIGS

    factual_config = CATEGORY_CONFIGS['factual']
    examples = factual_config['examples'][:3]  # Use 3-shot
    queries = factual_config['queries'][:5]     # Test on 5 queries
"""

from typing import Dict, List, Tuple
import os

# Configuration: Set to True to load real datasets (LAMA, SNLI, Gutenberg)
USE_REAL_DATASETS = os.environ.get('USE_REAL_DATASETS', 'True').lower() == 'true'

# Dataset sizes (only used when USE_REAL_DATASETS=True)
DATASET_SIZES = {
    'factual': 1000,   # LAMA TREx examples
    'logical': 300,    # SNLI examples (100 per label)
    'creative': 500    # Gutenberg poetry lines
}

# Train/Test split ratio for real datasets
# 20% for few-shot examples, 80% for evaluation queries
TRAIN_TEST_SPLIT_RATIO = 0.2


# ============================================================================
# FACTUAL KNOWLEDGE CONFIGURATION
# ============================================================================

FACTUAL_CONFIG = {
    'task_description': 'Complete the following factual statements with accurate information.',

    'examples': [
        # Geography
        ('The capital of France is', 'Paris'),
        ('The largest ocean on Earth is the', 'Pacific Ocean'),
        ('Mount Everest is located in the', 'Himalayas'),

        # Science
        ('The speed of light is approximately', '300,000 kilometers per second'),
        ('Water boils at', '100 degrees Celsius'),
        ('The chemical symbol for gold is', 'Au'),
        ('The smallest unit of life is a', 'cell'),
        ('DNA stands for', 'deoxyribonucleic acid'),

        # History & General Knowledge
        ('The first president of the United States was', 'George Washington'),
        ('The inventor of the telephone was', 'Alexander Graham Bell'),
        ('The human body has', '206 bones'),
        ('The largest planet in our solar system is', 'Jupiter'),
    ],

    'queries': [
        # Geography queries
        'The capital of Germany is',
        'The capital of Japan is',
        'The longest river in the world is the',

        # Science queries
        'The chemical symbol for water is',
        'The process by which plants make food is called',
        'Gravity was discovered by',
        'The Earth orbits the',

        # General knowledge queries
        'The author of "Romeo and Juliet" was',
        'The Great Wall is located in',
        'The currency of Japan is the',
    ]
}


# ============================================================================
# LOGICAL REASONING CONFIGURATION
# ============================================================================

LOGICAL_CONFIG = {
    'task_description': 'Complete the following logical reasoning problems using step-by-step inference.',

    'examples': [
        # Syllogistic reasoning
        ('If all cats are animals, and Fluffy is a cat, then Fluffy is', 'an animal'),
        ('If all birds have wings, and a penguin is a bird, then a penguin has', 'wings'),
        ('If all humans need oxygen, and Maria is a human, then Maria needs', 'oxygen'),

        # Transitive reasoning
        ('If 5 is greater than 3, and 3 is greater than 1, then 5 is', 'greater than 1'),
        ('If A equals B, and B equals C, then A', 'equals C'),

        # Temporal reasoning
        ('If today is Monday, then yesterday was', 'Sunday'),
        ('If tomorrow is Friday, then today is', 'Thursday'),

        # Arithmetic reasoning
        ('If 2 plus 2 equals 4, then 4 minus 2 equals', '2'),
    ],

    'queries': [
        # Syllogistic reasoning queries
        'If all dogs are mammals, and Rex is a dog, then Rex is a',
        'If all flowers are plants, and a rose is a flower, then a rose is a',
        'If all triangles have three sides, and ABC is a triangle, then ABC has',

        # Transitive reasoning queries
        'If 10 is greater than 5, and 5 is greater than 2, then 10 is',
        'If X equals Y, and Y equals Z, then X',

        # Temporal reasoning queries
        'If yesterday was Wednesday, then today is',
        'If last month was January, then this month is',

        # Arithmetic reasoning queries
        'If 3 plus 3 equals 6, then 6 minus 3 equals',
    ]
}


# ============================================================================
# CREATIVE GENERATION CONFIGURATION
# ============================================================================

CREATIVE_CONFIG = {
    'task_description': 'Continue the following creative writing prompts in a consistent narrative style.',

    'examples': [
        # Fantasy/Adventure
        ('Once upon a time, in a land far away, there lived', 'a brave knight who protected the kingdom'),
        ('The mysterious door slowly opened, revealing', 'a hidden staircase descending into darkness'),
        ('In the depths of the ancient forest, they discovered', 'a magical spring with glowing waters'),

        # Atmospheric/Poetic
        ('As the sun set over the horizon, she whispered', 'a secret wish to the evening star'),
        ('The old wizard raised his staff and', 'summoned a brilliant light that pierced the shadows'),

        # Suspense/Mystery
        ('Under the light of the full moon, the creature', 'emerged from the shadows with glowing eyes'),
        ('The abandoned castle held secrets that', 'had been buried for centuries'),
    ],

    'queries': [
        # Fantasy continuation queries
        'With a flash of lightning, everything suddenly',
        'The prophecy spoke of a hero who would',
        'Beyond the mountains lay a kingdom where',
        'As the clock struck midnight, the spell',

        # Discovery/Adventure queries
        'In the treasure chest, they found',
        'The ancient map led them to',
        'Deep beneath the ocean waves, there existed',

        # Magical/Mystical queries
        'When she touched the crystal, it',
    ]
}


# ============================================================================
# REAL DATASET LOADING
# ============================================================================

def _load_real_datasets() -> Dict[str, Dict]:
    """
    Load real datasets from LAMA, SNLI, and Gutenberg.

    Returns:
        Dictionary with configurations for each category using real data
    """
    print("\n[Config] Loading REAL datasets (LAMA + SNLI + Gutenberg)...")

    try:
        # Import the unified loader
        import sys
        from pathlib import Path

        # Add parent directory to path if needed
        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from llm_uncertainty_analysis.data.unified_dataset_loader import load_all_datasets

        # Load all datasets
        datasets = load_all_datasets(
            factual_n=DATASET_SIZES['factual'],
            logical_n=DATASET_SIZES['logical'],
            creative_n=DATASET_SIZES['creative'],
            seed=42
        )

        configs = {}

        # Process each category
        for category, data in datasets.items():
            if not data:
                print(f"[WARNING] No data loaded for {category}, using fallback")
                continue

            # Split into examples (train) and queries (test)
            split_idx = int(len(data) * TRAIN_TEST_SPLIT_RATIO)

            # Extract examples (prompt, answer) tuples
            examples = [(item['prompt'], item['answer']) for item in data[:split_idx]]

            # Extract queries (just prompts)
            queries = [item['prompt'] for item in data[split_idx:]]

            # Determine task description
            if category == 'factual':
                task_desc = 'Complete factual knowledge queries using information from Wikipedia/Wikidata.'
            elif category == 'logical':
                task_desc = 'Determine the logical relation (entailment/neutral/contradiction) between premise and hypothesis.'
            elif category == 'creative':
                task_desc = 'Continue poetic text by predicting the next line.'
            else:
                task_desc = 'Complete the following task.'

            # Create configuration
            configs[category] = {
                'task_description': task_desc,
                'examples': examples,
                'queries': queries,
                'metadata': {
                    'source': data[0]['source'] if data else 'unknown',
                    'total_samples': len(data),
                    'n_examples': len(examples),
                    'n_queries': len(queries),
                    'split_ratio': TRAIN_TEST_SPLIT_RATIO
                }
            }

        print(f"[Config] Loaded real datasets: {', '.join(configs.keys())}")

        return configs

    except Exception as e:
        print(f"[ERROR] Failed to load real datasets: {e}")
        print("[Config] Falling back to hardcoded configurations")
        return None


# ============================================================================
# COMBINED CONFIGURATION DICTIONARY
# ============================================================================

# Load configurations (real or hardcoded)
if USE_REAL_DATASETS:
    _real_configs = _load_real_datasets()

    if _real_configs:
        # Use real datasets
        CATEGORY_CONFIGS: Dict[str, Dict[str, any]] = _real_configs
        print("[Config] Using REAL datasets for ICL experiments")
    else:
        # Fallback to hardcoded
        CATEGORY_CONFIGS: Dict[str, Dict[str, any]] = {
            'factual': FACTUAL_CONFIG,
            'logical': LOGICAL_CONFIG,
            'creative': CREATIVE_CONFIG
        }
        print("[Config] Using HARDCODED (synthetic) datasets")
else:
    # Use hardcoded configurations
    CATEGORY_CONFIGS: Dict[str, Dict[str, any]] = {
        'factual': FACTUAL_CONFIG,
        'logical': LOGICAL_CONFIG,
        'creative': CREATIVE_CONFIG
    }
    print("[Config] Using HARDCODED (synthetic) datasets (USE_REAL_DATASETS=False)")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_category_config(category: str) -> Dict:
    """
    Get configuration for a specific category.

    Args:
        category: One of 'factual', 'logical', 'creative'

    Returns:
        Configuration dictionary for the category

    Raises:
        ValueError: If category is not valid

    Example:
        >>> config = get_category_config('factual')
        >>> print(config['task_description'])
        'Complete the following factual statements with accurate information.'
    """
    if category not in CATEGORY_CONFIGS:
        raise ValueError(f"Invalid category '{category}'. Must be one of: {list(CATEGORY_CONFIGS.keys())}")

    return CATEGORY_CONFIGS[category]


def get_examples(category: str, n_examples: int = None) -> List[Tuple[str, str]]:
    """
    Get examples for a category.

    Args:
        category: One of 'factual', 'logical', 'creative'
        n_examples: Number of examples to return (None = all)

    Returns:
        List of (query, answer) tuples

    Example:
        >>> examples = get_examples('factual', n_examples=3)
        >>> print(len(examples))
        3
    """
    config = get_category_config(category)
    examples = config['examples']

    if n_examples is not None:
        return examples[:n_examples]
    return examples


def get_queries(category: str, n_queries: int = None) -> List[str]:
    """
    Get test queries for a category.

    Args:
        category: One of 'factual', 'logical', 'creative'
        n_queries: Number of queries to return (None = all)

    Returns:
        List of query strings

    Example:
        >>> queries = get_queries('factual', n_queries=5)
        >>> print(len(queries))
        5
    """
    config = get_category_config(category)
    queries = config['queries']

    if n_queries is not None:
        return queries[:n_queries]
    return queries


def print_category_info(category: str):
    """
    Print information about a category configuration.

    Args:
        category: One of 'factual', 'logical', 'creative'

    Example:
        >>> print_category_info('factual')
    """
    config = get_category_config(category)

    print(f"\n{'='*70}")
    print(f"CATEGORY: {category.upper()}")
    print(f"{'='*70}")
    print(f"\nTask Description:")
    print(f"  {config['task_description']}")
    print(f"\nAvailable Examples: {len(config['examples'])}")
    print(f"Test Queries: {len(config['queries'])}")

    print(f"\nSample Examples:")
    for i, (query, answer) in enumerate(config['examples'][:3], 1):
        print(f"  {i}. Q: {query}")
        print(f"     A: {answer}")

    print(f"\nSample Queries:")
    for i, query in enumerate(config['queries'][:3], 1):
        print(f"  {i}. {query}")

    print(f"{'='*70}\n")


def print_all_categories_summary():
    """Print summary of all category configurations."""
    print("\n" + "="*70)
    print("ICL CATEGORY CONFIGURATIONS SUMMARY")
    print("="*70)

    for category in CATEGORY_CONFIGS.keys():
        config = CATEGORY_CONFIGS[category]
        print(f"\n{category.upper()}:")
        print(f"  Examples: {len(config['examples'])}")
        print(f"  Queries: {len(config['queries'])}")
        print(f"  Description: {config['task_description'][:60]}...")

    print("\n" + "="*70)
    print("Total configurations: 3 categories")
    print("Usage: CATEGORY_CONFIGS['factual'], CATEGORY_CONFIGS['logical'], etc.")
    print("="*70 + "\n")


# ============================================================================
# VALIDATION
# ============================================================================

def validate_configurations():
    """
    Validate that all configurations have the required structure.

    Returns:
        bool: True if all validations pass

    Raises:
        AssertionError: If any validation fails
    """
    required_keys = ['task_description', 'examples', 'queries']

    for category, config in CATEGORY_CONFIGS.items():
        # Check required keys
        for key in required_keys:
            assert key in config, f"Category '{category}' missing key '{key}'"

        # Check examples structure
        assert len(config['examples']) > 0, f"Category '{category}' has no examples"
        for example in config['examples']:
            assert isinstance(example, tuple), f"Examples must be tuples in '{category}'"
            assert len(example) == 2, f"Examples must be (query, answer) pairs in '{category}'"

        # Check queries structure
        assert len(config['queries']) > 0, f"Category '{category}' has no queries"
        for query in config['queries']:
            assert isinstance(query, str), f"Queries must be strings in '{category}'"

    return True


# Run validation on module import
if __name__ == "__main__":
    # Validate configurations
    try:
        validate_configurations()
        print("✅ All configurations validated successfully!")
    except AssertionError as e:
        print(f"❌ Validation failed: {e}")
        raise

    # Print summary
    print_all_categories_summary()

    # Print detailed info for each category
    for category in CATEGORY_CONFIGS.keys():
        print_category_info(category)
