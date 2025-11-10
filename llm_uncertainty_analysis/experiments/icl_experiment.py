"""
In-Context Learning Experiment

This module runs ICL experiments to measure how examples affect uncertainty.
"""

import numpy as np
from typing import Dict, List, Tuple
from ..icl import generate_icl_prompt, measure_icl_entropy


def run_icl_experiment(model,
                       tokenizer,
                       task_description: str,
                       examples: List[Tuple[str, str]],
                       queries: List[str],
                       n_examples_range: List[int],
                       device: str = 'cpu') -> Dict:
    """
    Run ICL experiment with varying numbers of examples.

    Args:
        model: Language model
        tokenizer: Tokenizer
        task_description: Task description for prompts
        examples: List of (question, answer) example tuples
        queries: List of query strings to test
        n_examples_range: List of example counts to test (e.g., [0, 1, 2, 3, 5])
        device: Device for computation

    Returns:
        Dictionary with experimental results
    """
    print(f"\nRunning ICL experiment with {len(queries)} queries...")
    print(f"Testing with n_examples: {n_examples_range}")

    results = {
        'n_examples': n_examples_range,
        'mean_entropy': [],
        'std_entropy': [],
        'mutual_info': []
    }

    baseline_entropy = None

    for n_ex in n_examples_range:
        entropies = []

        for query in queries:
            # Generate prompt
            prompt = generate_icl_prompt(task_description, examples, query, n_ex)

            # Measure entropy
            entropy = measure_icl_entropy(model, tokenizer, prompt, device)
            entropies.append(entropy)

        # Calculate statistics
        mean_entropy = np.mean(entropies)
        std_entropy = np.std(entropies)

        results['mean_entropy'].append(mean_entropy)
        results['std_entropy'].append(std_entropy)

        # Calculate mutual information (compared to 0-shot baseline)
        if baseline_entropy is None:
            baseline_entropy = mean_entropy
            results['mutual_info'].append(0.0)
        else:
            mi = baseline_entropy - mean_entropy
            results['mutual_info'].append(mi)

        print(f"  n={n_ex}: H={mean_entropy:.3f}±{std_entropy:.3f} bits, "
              f"MI={results['mutual_info'][-1]:.3f} bits")

    return results
