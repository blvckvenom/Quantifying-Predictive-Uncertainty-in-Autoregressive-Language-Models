"""
In-Context Learning Prompt Generation

This module contains functions for generating ICL prompts with varying numbers of examples.
"""

from typing import List, Tuple


def generate_icl_prompt(task_description: str,
                        examples: List[Tuple[str, str]],
                        query: str,
                        n_examples: int = 0) -> str:
    """
    Generate an In-Context Learning prompt with n examples.

    Args:
        task_description: Description of the task
        examples: List of tuples (input, output) of examples
        query: Question/input for which we want a prediction
        n_examples: Number of examples to include (0 = zero-shot)

    Returns:
        Formatted prompt for ICL
    """
    if n_examples == 0:
        # Zero-shot: only instruction + query
        prompt = f"{task_description}\n\nQ: {query}\nA:"
    else:
        # Few-shot: instruction + examples + query
        prompt_parts = [task_description, ""]

        # Add examples
        for i in range(min(n_examples, len(examples))):
            q, a = examples[i]
            prompt_parts.append(f"Q: {q}\nA: {a}")

        # Add final query
        prompt_parts.append(f"Q: {query}\nA:")
        prompt = "\n\n".join(prompt_parts)

    return prompt
