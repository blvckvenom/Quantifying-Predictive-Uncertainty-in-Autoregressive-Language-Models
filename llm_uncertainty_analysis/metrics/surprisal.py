"""
Surprisal Calculation

This module contains functions for calculating surprisal (self-information) of tokens.
"""

import numpy as np


def calculate_surprisal(prob_true: float, validate: bool = True) -> float:
    """
    Calculate the surprisal (self-information) of the true token.

    Surprisal measures how much "surprise" or "information" observing an event provides:
    S(x) = -log₂(p(x))

    Args:
        prob_true: Probability assigned to the true token/event. Must be
                   in the range [0, 1].
        validate: If True, validates that probability is in [0, 1].
                  Default: True. Set to False for better performance.

    Returns:
        Surprisal in bits (using log base 2). Returns float('inf') if p = 0.

    Raises:
        ValueError: If validate=True and prob_true is not in [0, 1]

    Examples:
        >>> # Very probable token (low surprise)
        >>> calculate_surprisal(0.5)
        1.0

        >>> # Very improbable token (high surprise)
        >>> calculate_surprisal(0.01)
        6.643856189774724

        >>> # Impossible token (infinite surprise)
        >>> calculate_surprisal(0.0)
        inf

    Notes:
        - Relation to entropy: H = E[S(x)] = Σ p(x) * S(x)
        - Lower probability → Higher surprisal
        - p = 1 → S = 0 (no surprise, certain event)
        - p → 0 → S → ∞ (maximum surprise)
        - Correlates with human reading times (Levy, 2008)
        - Also known as "self-information" or "own information"
    """
    # Optional validations
    if validate:
        if not isinstance(prob_true, (int, float, np.number)):
            raise TypeError(f"prob_true must be a number, received: {type(prob_true)}")

        if prob_true < 0 or prob_true > 1:
            raise ValueError(f"prob_true must be in [0, 1], received: {prob_true:.6f}")

    # Edge case: zero probability → infinite surprisal
    if prob_true <= 0:
        return float('inf')

    # Edge case: probability 1 → zero surprisal (no surprise)
    if prob_true >= 1.0:
        return 0.0

    # Calculate surprisal: S = -log₂(p)
    surprisal = -np.log2(prob_true)

    return float(surprisal)
