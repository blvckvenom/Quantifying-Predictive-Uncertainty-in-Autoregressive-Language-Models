"""
Reproducibility Utilities

This module contains functions to ensure reproducible results.
"""

import random
import numpy as np
import torch


def setup_reproducibility(seed: int = 42):
    """
    Configure all random seeds for reproducibility.

    Sets seeds for:
    - Python's random module
    - NumPy
    - PyTorch (CPU and CUDA)
    - CuDNN (if available)

    Args:
        seed: Random seed value (default: 42)
    """
    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)

    # PyTorch CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # CuDNN deterministic mode
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"✅ Reproducibility configured with seed={seed}")
    print(f"   - Python random: {seed}")
    print(f"   - NumPy: {seed}")
    print(f"   - PyTorch: {seed}")
    if torch.cuda.is_available():
        print(f"   - CUDA: {seed}")
        print(f"   - CuDNN deterministic: True")