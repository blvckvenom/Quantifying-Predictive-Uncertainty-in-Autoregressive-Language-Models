"""
Uncertainty Analyzer

This module contains the main analyzer class for quantifying uncertainty in language models.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import pandas as pd
from typing import List, Dict
from tqdm import tqdm

from ..metrics import calculate_entropy, calculate_surprisal, calculate_perplexity


class UncertaintyAnalyzer:
    """Complete pipeline for uncertainty analysis in LLMs."""

    def __init__(self, model_name: str, device: str = "cuda"):
        """
        Initialize the uncertainty analyzer.

        Args:
            model_name: HuggingFace model identifier
            device: Device to use ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.load_model()

    def load_model(self):
        """Load model and tokenizer."""
        print(f"Loading {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        # Configure padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def compute_token_metrics(self, text: str) -> pd.DataFrame:
        """
        Calculate entropy and surprisal for each token in the text.

        Args:
            text: Input text to analyze

        Returns:
            DataFrame with per-token metrics
        """
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(self.device)
        input_ids = inputs["input_ids"]

        # Get logits
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # [batch_size, seq_len, vocab_size]

        results = []
        seq_len = input_ids.shape[1]

        for i in range(1, seq_len):  # Start from 1 because we predict the next token
            # Logits for predicting token i
            current_logits = logits[0, i-1, :]  # [vocab_size]
            probs = F.softmax(current_logits, dim=-1)

            # Convert to numpy to use our improved function
            probs_np = probs.cpu().numpy()

            # Entropy using validated function (in bits)
            entropy = calculate_entropy(probs_np, validate=False)  # Already validated by softmax

            # Surprisal of the true token using validated function (in bits)
            true_token_id = input_ids[0, i]
            true_token_prob = probs[true_token_id].item()

            # Use calculate_surprisal for consistency and robustness
            surprisal = calculate_surprisal(true_token_prob, validate=False)

            # Perplexity calculated from surprisal
            perplexity = calculate_perplexity(surprisal)

            # Decoded token
            token_str = self.tokenizer.decode(true_token_id)

            results.append({
                "position": i,
                "token": token_str,
                "token_id": true_token_id.item(),
                "entropy": entropy,
                "surprisal": surprisal,
                "perplexity": perplexity,
                "probability": true_token_prob
            })

        return pd.DataFrame(results)

    def compute_sequence_metrics(self, text: str) -> Dict:
        """
        Calculate aggregated metrics for the entire sequence.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary with sequence-level metrics
        """
        token_metrics = self.compute_token_metrics(text)

        return {
            "text": text,
            "num_tokens": len(token_metrics),
            "mean_entropy": token_metrics["entropy"].mean(),
            "std_entropy": token_metrics["entropy"].std(),
            "max_entropy": token_metrics["entropy"].max(),
            "min_entropy": token_metrics["entropy"].min(),
            "mean_surprisal": token_metrics["surprisal"].mean(),
            "std_surprisal": token_metrics["surprisal"].std(),
            "mean_perplexity": token_metrics["perplexity"].mean(),
            "cross_entropy": token_metrics["surprisal"].mean()  # CE = mean surprisal
        }

    def analyze_dataset(self, samples: List[Dict]) -> pd.DataFrame:
        """
        Analyze a set of samples.

        Args:
            samples: List of sample dictionaries with 'prompt', 'answer', and 'category' keys

        Returns:
            DataFrame with results for all samples
        """
        results = []

        for sample in tqdm(samples, desc=f"Analyzing with {self.model_name}"):
            metrics = self.compute_sequence_metrics(sample["prompt"])
            metrics["category"] = sample["category"]
            metrics["has_answer"] = sample["answer"] is not None
            results.append(metrics)

        return pd.DataFrame(results)
