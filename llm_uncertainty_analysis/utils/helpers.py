"""
Helper Utilities

This module contains general utility functions.
"""

from pathlib import Path
from typing import Union


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path

    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_results(data, filepath: Union[str, Path], format: str = 'csv'):
    """
    Save results to file.

    Args:
        data: Data to save (DataFrame, dict, etc.)
        filepath: Output file path
        format: File format ('csv', 'json', 'pkl')
    """
    filepath = Path(filepath)
    ensure_directory(filepath.parent)

    if format == 'csv':
        data.to_csv(filepath, index=False)
    elif format == 'json':
        import json
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    elif format == 'pkl':
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    else:
        raise ValueError(f"Unsupported format: {format}")

    print(f"Results saved to: {filepath}")
