"""
Mutual Information Calculation

This module contains functions for calculating mutual information between evidence and predictions.
"""

import numpy as np
from ..metrics import calculate_entropy


def calculate_mutual_information(probs_without_evidence: np.ndarray,
                                  probs_with_evidence: np.ndarray) -> float:
    """
    Calculate mutual information I(E; Y_t | Y_<t).

    Mutual information quantifies how much evidence E reduces uncertainty
    about the next token Y_t, given the previous context Y_<t.

    Formula:
        I(E; Y_t | Y_<t) = H(Y_t | Y_<t) - H(Y_t | Y_<t, E)

    Where:
        - H(Y_t | Y_<t): Entropy without additional evidence (baseline)
        - H(Y_t | Y_<t, E): Entropy with additional evidence
        - I > 0: Evidence reduces uncertainty (useful)
        - I = 0: Evidence provides no information (redundant)
        - I < 0: Theoretically impossible (would indicate errors)

    Args:
        probs_without_evidence: Probability distribution over tokens WITHOUT
            additional evidence. Shape: (vocab_size,)
        probs_with_evidence: Probability distribution over tokens WITH
            additional evidence. Shape: (vocab_size,)

    Returns:
        Mutual information in bits. Typical values:
            - I > 1.0 bits: Very informative evidence
            - 0.5 < I < 1.0: Moderately informative evidence
            - 0.0 < I < 0.5: Slightly informative evidence
            - I ≈ 0.0: Non-informative evidence

    Raises:
        ValueError: If probabilities are invalid or have different shapes

    Example:
        >>> # Uniform distribution (high uncertainty)
        >>> probs_baseline = np.ones(100) / 100  # H = log2(100) ≈ 6.64 bits
        >>> # Concentrated distribution (low uncertainty)
        >>> probs_informed = np.zeros(100)
        >>> probs_informed[0] = 0.9
        >>> probs_informed[1:] = 0.1 / 99
        >>> I = calculate_mutual_information(probs_baseline, probs_informed)
        >>> print(f"I = {I:.3f} bits")  # I ≈ 3.2 bits (significant reduction)
    """
    # Validate inputs
    if probs_without_evidence.shape != probs_with_evidence.shape:
        raise ValueError(
            f"Distributions must have the same shape. "
            f"Got: {probs_without_evidence.shape} vs {probs_with_evidence.shape}"
        )

    # Calculate entropies using existing function
    H_without = calculate_entropy(probs_without_evidence)
    H_with = calculate_entropy(probs_with_evidence)

    # Mutual information = entropy reduction
    mutual_info = H_without - H_with

    # Validate result (I should be non-negative)
    if mutual_info < -1e-10:  # Small tolerance for numerical errors
        print(f"⚠️  WARNING: Negative mutual information ({mutual_info:.6f} bits)")
        print(f"    H_without = {H_without:.6f}, H_with = {H_with:.6f}")
        print(f"    This suggests numerical error or invalid distributions.")

    return mutual_info


def interpret_mutual_information(mi_value: float) -> str:
    """
    Interpret the value of mutual information.

    Args:
        mi_value: Mutual information in bits

    Returns:
        Qualitative interpretation
    """
    if mi_value < 0:
        return "❌ INVALID (negative - calculation error)"
    elif mi_value < 0.1:
        return "📊 VERY LOW - evidence almost non-informative"
    elif mi_value < 0.5:
        return "📈 LOW - evidence slightly informative"
    elif mi_value < 1.0:
        return "📊 MODERATE - evidence moderately informative"
    elif mi_value < 2.0:
        return "📈 HIGH - evidence very informative"
    else:
        return "🔥 VERY HIGH - evidence extremely informative"
