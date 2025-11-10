"""
Global Settings and Constants

This module contains all global configuration settings for the LLM uncertainty analysis project,
including reproducibility settings, device configuration, and constant values.
"""

import random
import numpy as np
import torch


# Reproducibility seed
SEED = 42

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_reproducibility():
    """
    Configure global seeds for reproducibility across all random number generators.

    This ensures that experiments are reproducible across multiple runs and
    between different machines (CPU/GPU).
    """
    # Set seeds for all random number generation libraries
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # If GPU is available, also configure CUDA seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        # Additional configurations for GPU reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"[OK] Global seed configured: SEED = {SEED}")
    print(f"[OK] Reproducibility guaranteed in: random, numpy, torch")
    if torch.cuda.is_available():
        print(f"[OK] CUDA reproducibility configured")
        print(f"  - cudnn.deterministic = True")
        print(f"  - cudnn.benchmark = False")


def print_device_info():
    """Print information about the device being used (CPU/GPU)."""
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("Using CPU")
