"""
Metrics Module

This module contains functions for calculating uncertainty metrics:
- Entropy: Shannon entropy of probability distributions
- Surprisal: Self-information of tokens
- Perplexity: Exponential of surprisal
"""

from .entropy import calculate_entropy, calculate_entropy_from_logits
from .surprisal import calculate_surprisal
from .perplexity import calculate_perplexity

__all__ = [
    'calculate_entropy',
    'calculate_entropy_from_logits',
    'calculate_surprisal',
    'calculate_perplexity'
]
