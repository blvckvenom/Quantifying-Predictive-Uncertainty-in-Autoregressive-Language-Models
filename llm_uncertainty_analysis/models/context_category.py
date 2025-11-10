"""
Context Category

This module defines context categories for uncertainty analysis.
"""

from dataclasses import dataclass
from typing import List, Tuple  # ← AÑADIR Tuple


@dataclass
class ContextCategory:
    """
    Define a context category for uncertainty analysis.

    Includes numerical thresholds to classify entropy into three levels:
    - high-certainty: H < 5.0 bits (high certainty/predictability)
    - medium-certainty: 5.0 <= H < 10.0 bits (medium certainty)
    - low-certainty: H >= 10.0 bits (low certainty/high uncertainty)
    
    Attributes:
        name: Category name (factual, logical, creative)
        description: Human-readable description
        expected_entropy: Qualitative expectation ('low', 'medium', 'high')
        expected_range: Quantitative range (min, max) in bits
        datasets: List of datasets for this category
        example: Example prompt
        low_threshold: Entropy threshold for high certainty (default: 5.0)
        high_threshold: Entropy threshold for low certainty (default: 10.0)
    """
    name: str
    description: str
    expected_entropy: str  # 'low', 'medium', 'high'
    expected_range: Tuple[float, float]  # ← AÑADIR ESTE CAMPO (CRÍTICO)
    datasets: List[str]
    example: str
    low_threshold: float = 5.0   # Threshold for high certainty
    high_threshold: float = 10.0  # Threshold for low certainty

    def classify_entropy(self, entropy: float) -> str:
        """
        Classify an entropy value into one of three certainty categories.

        Args:
            entropy: Entropy value in bits

        Returns:
            Certainty category: 'high-certainty', 'medium-certainty', or 'low-certainty'
        """
        if entropy < self.low_threshold:
            return "high-certainty"
        elif entropy < self.high_threshold:
            return "medium-certainty"
        else:
            return "low-certainty"

    def is_in_expected_range(self, entropy: float) -> bool:
        """
        Check if entropy value is within expected range for this category.
        
        This is useful for validating hypotheses about entropy by category.

        Args:
            entropy: Entropy value in bits

        Returns:
            True if entropy is within expected range, False otherwise
            
        Example:
            >>> factual = CONTEXT_CATEGORIES[0]  # expected_range=(0.0, 5.0)
            >>> factual.is_in_expected_range(3.2)
            True
            >>> factual.is_in_expected_range(8.5)
            False
        """
        min_entropy, max_entropy = self.expected_range
        return min_entropy <= entropy <= max_entropy


# ============================================================================
# CATEGORÍAS DE CONTEXTO DEFINIDAS
# ============================================================================

CONTEXT_CATEGORIES = [
    ContextCategory(
        name="factual",
        description="Completación de hechos conocidos con respuesta única",
        expected_entropy="low",
        expected_range=(0.0, 5.0),  # ← AÑADIR
        datasets=["lama", "squad"],
        example="The capital of France is [MASK]"
    ),
    ContextCategory(
        name="logical",
        description="Problemas de razonamiento con estructura lógica clara",
        expected_entropy="medium",
        expected_range=(5.0, 10.0),  # ← AÑADIR
        datasets=["gsm8k", "arithmetic", "snli"],
        example="If 2 + 2 = 4, then 3 + 3 = [MASK]"
    ),
    ContextCategory(
        name="creative",
        description="Generación abierta con múltiples continuaciones válidas",
        expected_entropy="high",
        expected_range=(10.0, float('inf')),  # ← AÑADIR
        datasets=["gutenberg_poetry", "writingprompts"],
        example="Once upon a time, there was a [MASK]"
    )
]


# ============================================================================
# FUNCIONES HELPER
# ============================================================================

def get_category_by_name(name: str) -> ContextCategory:
    """
    Get category by name.
    
    Args:
        name: Category name ('factual', 'logical', 'creative')
    
    Returns:
        ContextCategory object
    
    Raises:
        ValueError: If category name not found
    """
    for cat in CONTEXT_CATEGORIES:
        if cat.name == name:
            return cat
    raise ValueError(f"Category '{name}' not found. Valid: {[c.name for c in CONTEXT_CATEGORIES]}")


def print_category_info(categories: List[ContextCategory] = CONTEXT_CATEGORIES):
    """Print information about context categories."""
    print("\n" + "="*70)
    print("CATEGORÍAS DE CONTEXTO DEFINIDAS:")
    print("="*70)
    
    for cat in categories:
        print(f"\n{cat.name.upper()}:")
        print(f"  Descripción: {cat.description}")
        print(f"  Entropía esperada: {cat.expected_entropy}")
        print(f"  Rango esperado: [{cat.expected_range[0]:.1f}, {cat.expected_range[1]:.1f}] bits")
        print(f"  Datasets: {', '.join(cat.datasets)}")
        print(f"  Ejemplo: {cat.example}")
        print(f"  Umbrales clasificación: low={cat.low_threshold}, high={cat.high_threshold}")
    
    print("\n" + "="*70)


def demo_entropy_classification():
    """Demonstrate entropy classification with examples."""
    print("\n" + "="*60)
    print("EJEMPLOS DE CLASIFICACIÓN DE ENTROPÍA:")
    print("="*60)

    test_entropies = [3.5, 7.2, 12.8]
    test_category = CONTEXT_CATEGORIES[0]  # Factual category

    for H in test_entropies:
        certainty_level = test_category.classify_entropy(H)
        in_range = test_category.is_in_expected_range(H)
        range_indicator = "✓" if in_range else "✗"
        
        print(f"Entropía = {H:.1f} bits → Categoría: {certainty_level} "
              f"[{range_indicator} dentro de rango esperado para {test_category.name}]")
    
    print("="*60)


def validate_hypothesis(results_df, verbose: bool = True):
    """
    Validate hypothesis that entropy varies by category as expected.
    
    Args:
        results_df: DataFrame with columns ['category', 'mean_entropy']
        verbose: If True, print detailed validation
    
    Returns:
        dict with validation results
    """
    validation = {}
    
    for cat in CONTEXT_CATEGORIES:
        # Get mean entropy for this category
        cat_data = results_df[results_df['category'] == cat.name]
        if len(cat_data) == 0:
            continue
            
        mean_entropy = cat_data['mean_entropy'].mean()
        in_range = cat.is_in_expected_range(mean_entropy)
        
        validation[cat.name] = {
            'mean_entropy': mean_entropy,
            'expected_range': cat.expected_range,
            'in_expected_range': in_range
        }
        
        if verbose:
            status = "✓ CORRECTO" if in_range else "✗ FUERA DE RANGO"
            print(f"{cat.name.upper()}: {mean_entropy:.2f} bits [{status}]")
            print(f"  Rango esperado: [{cat.expected_range[0]:.1f}, {cat.expected_range[1]:.1f}]")
    
    return validation