"""
Visualization Configuration

This module contains visualization settings for matplotlib and seaborn plots.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import warnings


def setup_visualization():
    """Configure global visualization settings for matplotlib and seaborn."""
    # Seaborn style
    sns.set_style("whitegrid")

    # Matplotlib settings
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.size'] = 12

    # Suppress warnings
    warnings.filterwarnings('ignore')

    print("[OK] Visualization settings configured")
    print("  - Style: whitegrid")
    print("  - Figure size: (10, 6)")
    print("  - Font size: 12")


# Color palette for categories
CATEGORY_COLORS = {
    'factual': '#2ecc71',    # Green - high certainty
    'logical': '#f39c12',    # Orange - medium certainty
    'creative': '#e74c3c'    # Red - high uncertainty
}


# Color palette for certainty levels
CERTAINTY_COLORS = {
    'high-certainty': '#27ae60',
    'medium-certainty': '#f39c12',
    'low-certainty': '#c0392b'
}
