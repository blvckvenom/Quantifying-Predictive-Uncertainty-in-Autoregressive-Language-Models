"""
Entropy Calculation

This module contains functions for calculating Shannon entropy from probability distributions.
"""

import numpy as np


def calculate_entropy(probs: np.ndarray, validate: bool = True) -> float:
    """
    Calculate Shannon entropy of a probability distribution.

    Entropy measures the average uncertainty in a distribution:
    H = -Σ p(x) * log₂(p(x))

    Args:
        probs: Probability array. Must be a 1D array where each element
               represents the probability of an event.
        validate: If True, validates that probabilities sum to 1 and are >= 0.
                  Default: True. Set to False for better performance if
                  already validated previously.

    Returns:
        Entropy in bits (using log base 2)

    Raises:
        ValueError: If validate=True and probabilities don't meet constraints

    Examples:
        >>> # Uniform distribution (maximum entropy)
        >>> uniform_probs = np.array([0.25, 0.25, 0.25, 0.25])
        >>> calculate_entropy(uniform_probs)
        2.0

        >>> # Deterministic distribution (minimum entropy)
        >>> certain_probs = np.array([1.0, 0.0, 0.0, 0.0])
        >>> calculate_entropy(certain_probs)
        0.0

    Notes:
        - For uniform distribution with n elements: H = log₂(n)
        - For deterministic distribution: H = 0
        - Maximum entropy occurs when all probabilities are equal
    """
    # Optional validations
    if validate:
        # Verify it's a numpy array
        if not isinstance(probs, np.ndarray):
            probs = np.array(probs)

        # Verify it's 1D
        if probs.ndim != 1:
            raise ValueError(f"probs must be a 1D array, received shape: {probs.shape}")

        # Verify all probabilities are >= 0
        if np.any(probs < 0):
            raise ValueError(f"All probabilities must be >= 0, min found: {probs.min():.6f}")

        # Verify they sum approximately to 1 (with tolerance for numerical errors)
        prob_sum = np.sum(probs)
        if not np.isclose(prob_sum, 1.0, atol=1e-6):
            raise ValueError(f"Probabilities must sum to 1, current sum: {prob_sum:.6f}")

    # Filter zero probabilities to avoid log(0)
    # Only consider p(x) > 0 since lim_{p->0} p*log(p) = 0
    probs_nonzero = probs[probs > 0]

    # Edge case: empty distribution or all probabilities are 0
    if len(probs_nonzero) == 0:
        return 0.0

    # Calculate entropy: H = -Σ p(x) * log₂(p(x))
    entropy = -np.sum(probs_nonzero * np.log2(probs_nonzero))

    return entropy


def calculate_entropy_from_logits(logits: np.ndarray, validate: bool = True) -> float:
    """
    Calculate entropy directly from logits (more efficient and numerically stable).

    Args:
        logits: Array of logits (unnormalized scores)
        validate: If True, validates inputs

    Returns:
        Entropy in bits
    """
    # Apply softmax to obtain probabilities
    # Using log-sum-exp trick for numerical stability
    logits_shifted = logits - np.max(logits)  # Prevent overflow
    exp_logits = np.exp(logits_shifted)
    probs = exp_logits / np.sum(exp_logits)

    return calculate_entropy(probs, validate=validate)
