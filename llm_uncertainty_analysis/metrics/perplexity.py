"""
Perplexity Calculation

This module contains functions for calculating perplexity from surprisal.
"""


def calculate_perplexity(surprisal: float) -> float:
    """
    Calculate perplexity from surprisal.

    Perplexity is 2 raised to the surprisal:
    PPL = 2^S = 2^(-log₂(p)) = 1/p

    Args:
        surprisal: Surprisal in bits

    Returns:
        Perplexity (always >= 1)

    Examples:
        >>> calculate_perplexity(1.0)  # p = 0.5
        2.0

        >>> calculate_perplexity(0.0)  # p = 1.0
        1.0

        >>> calculate_perplexity(float('inf'))  # p = 0
        inf

    Notes:
        - Interpretation: "equivalent to guessing from PPL equiprobable choices"
        - PPL = 100 means as much uncertainty as choosing from 100 random words
    """
    return 2 ** surprisal
