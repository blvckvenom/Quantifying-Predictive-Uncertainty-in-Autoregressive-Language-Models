"""
ICL Entropy Measurement

This module contains functions for measuring entropy in ICL scenarios.
"""

import torch
from typing import List, Tuple, Dict
from ..metrics import calculate_entropy


def measure_icl_entropy(model, tokenizer, prompt: str, device: str = 'cpu') -> float:
    """
    Measure the entropy of the first response token.

    This function specifically analyzes the token that follows "A:" to
    measure the model's uncertainty in its initial prediction.

    Args:
        model: Language model (GPT-2)
        tokenizer: Corresponding tokenizer
        prompt: Complete prompt (with or without examples)
        device: Device for computation

    Returns:
        Entropy in bits of the first response token
    """
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors='pt').to(device)

    # Get model logits
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Logits of last token (before generating response)
    last_token_logits = logits[0, -1, :]

    # Convert to probabilities
    probs = torch.softmax(last_token_logits, dim=0)

    # Calculate entropy
    entropy = calculate_entropy(probs.cpu().numpy())

    return entropy


def measure_entropy_reduction(model, tokenizer,
                              task_description: str,
                              examples: List[Tuple[str, str]],
                              query: str,
                              n_shot_baseline: int = 0,
                              n_shot_comparison: int = 5,
                              device: str = 'cpu') -> Dict[str, float]:
    """
    Measure entropy reduction from ICL examples.

    Compares entropy between two ICL conditions (typically 0-shot vs k-shot).

    Args:
        model: Language model
        tokenizer: Tokenizer
        task_description: Task description
        examples: Available examples (list of (question, answer) tuples)
        query: Query to analyze
        n_shot_baseline: Number of examples in baseline (default: 0)
        n_shot_comparison: Number of examples in comparison (default: 5)
        device: Computation device

    Returns:
        Dictionary with:
            - 'H_baseline': Entropy with baseline examples (bits)
            - 'H_comparison': Entropy with comparison examples (bits)
            - 'delta_H': Entropy reduction (bits), positive = uncertainty reduced
            - 'reduction_percent': Percentage reduction (%)
            - 'n_shot_baseline': Number of examples in baseline
            - 'n_shot_comparison': Number of examples in comparison

    Example:
        >>> # Compare 0-shot vs 5-shot
        >>> results = measure_entropy_reduction(
        ...     model, tokenizer, task, examples, query,
        ...     n_shot_baseline=0, n_shot_comparison=5
        ... )
        >>> print(f"Delta H = {results['delta_H']:.3f} bits")
        >>> print(f"Reduction: {results['reduction_percent']:.1f}%")
    """
    from .prompt_generation import generate_icl_prompt

    # Generate prompts
    prompt_baseline = generate_icl_prompt(
        task_description, examples, query, n_examples=n_shot_baseline
    )
    prompt_comparison = generate_icl_prompt(
        task_description, examples, query, n_examples=n_shot_comparison
    )

    # Measure entropies
    H_baseline = measure_icl_entropy(model, tokenizer, prompt_baseline, device)
    H_comparison = measure_icl_entropy(model, tokenizer, prompt_comparison, device)

    # Calculate reduction
    delta_H = H_baseline - H_comparison
    reduction_percent = (delta_H / H_baseline * 100) if H_baseline > 0 else 0.0

    return {
        'H_baseline': H_baseline,
        'H_comparison': H_comparison,
        'delta_H': delta_H,
        'reduction_percent': reduction_percent,
        'n_shot_baseline': n_shot_baseline,
        'n_shot_comparison': n_shot_comparison
    }