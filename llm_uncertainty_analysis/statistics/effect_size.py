"""
Effect Size Calculations

This module contains functions for calculating effect sizes (Cohen's d).
"""

import numpy as np
from typing import Tuple


def calculate_cohens_d(group1, group2) -> float:
    """
    Calculate Cohen's d to measure effect size between two groups.

    Cohen's d standardizes the mean difference using the pooled standard deviation.

    Formula:
        d = (M1 - M2) / pooled_std

        where pooled_std = sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))

    Args:
        group1: Array or list of values from the first group
        group2: Array or list of values from the second group

    Returns:
        Cohen's d (float): Standardized effect size

    Interpretation according to Cohen (1988):
        - |d| < 0.2:  NEGLIGIBLE effect
        - |d| = 0.2:  SMALL effect
        - |d| = 0.5:  MEDIUM effect
        - |d| = 0.8:  LARGE effect
        - |d| > 1.0:  VERY LARGE effect

    References:
        Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences.
        Lawrence Erlbaum Associates.
    """
    # Convert to numpy arrays if needed
    group1 = np.array(group1)
    group2 = np.array(group2)

    # Sample sizes
    n1, n2 = len(group1), len(group2)

    # Means
    mean1 = np.mean(group1)
    mean2 = np.mean(group2)

    # Variances (with Bessel's correction: ddof=1)
    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)

    # Pooled standard deviation
    # Combines variability from both groups in a weighted manner
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    # Cohen's d: standardized mean difference
    cohens_d = (mean1 - mean2) / pooled_std

    return cohens_d


def interpret_cohens_d(d: float) -> Tuple[str, str]:
    """
    Interpret the magnitude of Cohen's d according to standard criteria.

    Args:
        d: Value of Cohen's d

    Returns:
        Tuple of (magnitude, description)
    """
    abs_d = abs(d)

    if abs_d < 0.2:
        return ("negligible", "very small or no effect")
    elif abs_d < 0.5:
        return ("small", "detectable but modest effect")
    elif abs_d < 0.8:
        return ("medium", "clearly visible effect")
    elif abs_d < 1.2:
        return ("large", "substantial effect")
    else:
        return ("very large", "extremely strong effect")
