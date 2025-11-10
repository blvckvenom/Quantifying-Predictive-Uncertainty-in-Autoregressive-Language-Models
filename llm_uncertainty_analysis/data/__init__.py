"""
Data Loading Module for ICL Multi-Model Analysis

This module provides loaders for three established datasets:
- LAMA TREx: Factual knowledge from Wikipedia/Wikidata
- Stanford SNLI: Logical reasoning (natural language inference)
- Project Gutenberg: Creative text (poetry)
"""

from .lama_loader import load_lama_factual
from .snli_loader import load_snli_balanced
from .gutenberg_loader import load_gutenberg_poetry
from .unified_dataset_loader import load_all_datasets, get_dataset_statistics

__all__ = [
    'load_lama_factual',
    'load_snli_balanced',
    'load_gutenberg_poetry',
    'load_all_datasets',
    'get_dataset_statistics'
]
