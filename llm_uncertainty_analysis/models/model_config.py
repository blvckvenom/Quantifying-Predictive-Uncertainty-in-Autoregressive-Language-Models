"""
Model Configuration

This module defines model configurations for evaluation.

CONFIGURACIÓN DE MODELOS:

Para el HITO INTERMEDIO:
- Se usa DEFAULT_MODEL (GPT-2 Small) únicamente
- Razón: Enfoque en metodología completa, no comparación de modelos
- Uso: analyzer = UncertaintyAnalyzer(DEFAULT_MODEL.model_id)

Para la ENTREGA FINAL (Análisis Multi-Modelo):
- Se comparan 3 modelos de COMPARISON_MODELS para análisis de escalamiento
- Hipótesis: (H1) ΔH aumenta con tamaño, (H2) Ranking consistente entre modelos
- Uso: for model_id in COMPARISON_MODELS: analyzer = UncertaintyAnalyzer(model_id)

Esta arquitectura permite:
1. Simplicidad para desarrollo inicial (hito)
2. Escalabilidad para análisis exhaustivo (entrega final)
3. Comparación multi-modelo con validación de hipótesis
"""

from dataclasses import dataclass
from typing import List, Optional
import torch


@dataclass
class ModelConfig:
    """Configuration for each model to evaluate."""
    name: str
    model_id: str
    params: str  # Number of parameters
    memory_gb: float  # Required memory
    justification: str


# ============================================================================
# MODELOS DISPONIBLES
# ============================================================================

MODELS = [
    ModelConfig(
        name="GPT-2 Small",
        model_id="gpt2",
        params="124M",
        memory_gb=0.5,
        justification="Baseline ligero, permite iteración rápida y debugging"
    ),
    ModelConfig(
        name="GPT-2 Medium",
        model_id="gpt2-medium",
        params="355M",
        memory_gb=1.5,
        justification="Balance entre capacidad y eficiencia computacional"
    ),
    ModelConfig(
        name="DistilGPT-2",
        model_id="distilgpt2",
        params="82M",
        memory_gb=0.3,
        justification="Versión destilada, útil para comparar efecto de compresión"
    )
]


# ============================================================================
# MODELOS PARA COMPARACIÓN MULTI-MODELO (entrega final)
# ============================================================================

# Lista ordenada por tamaño (pequeño -> mediano -> grande) para análisis de escalamiento
COMPARISON_MODELS = ['distilgpt2', 'gpt2', 'gpt2-medium']

# Mapeo de model_id a número de parámetros (para análisis de correlación)
MODEL_PARAMS_NUMERIC = {
    'distilgpt2': 82_000_000,      # 82M
    'gpt2': 124_000_000,            # 124M
    'gpt2-medium': 355_000_000,     # 355M
}


# ============================================================================
# MODELO POR DEFECTO (para hito intermedio y uso simple)
# ============================================================================

DEFAULT_MODEL = MODELS[0]  # GPT-2 Small


# ============================================================================
# FUNCIONES HELPER
# ============================================================================

def get_model_by_id(model_id: str) -> Optional[ModelConfig]:
    """
    Get model configuration by model_id.
    
    Args:
        model_id: HuggingFace model identifier (e.g., 'gpt2', 'gpt2-medium')
    
    Returns:
        ModelConfig if found, None otherwise
    
    Example:
        >>> model = get_model_by_id('gpt2')
        >>> print(model.name)
        'GPT-2 Small'
    """
    for model in MODELS:
        if model.model_id == model_id:
            return model
    return None


def get_models_by_ids(model_ids: List[str]) -> List[ModelConfig]:
    """
    Get multiple model configurations by their IDs.
    
    Args:
        model_ids: List of HuggingFace model identifiers
    
    Returns:
        List of ModelConfig objects (skips invalid IDs with warning)
    
    Example:
        >>> models = get_models_by_ids(['gpt2', 'gpt2-medium'])
        >>> print(len(models))
        2
    """
    models = []
    for model_id in model_ids:
        model = get_model_by_id(model_id)
        if model:
            models.append(model)
        else:
            print(f"⚠️  Warning: Model '{model_id}' not found in MODELS list")
    return models


def get_available_models(device: torch.device) -> List[ModelConfig]:
    """
    Get list of available models based on device capabilities.
    
    Note: Returns models that COULD run on the device, but DEFAULT_MODEL
    is used for the hito intermedio regardless.

    Args:
        device: torch device (CPU or CUDA)

    Returns:
        List of ModelConfig objects that can run on the device
    """
    available = MODELS.copy()

    # If GPU with sufficient memory, could add larger models
    if device.type == "cuda" and torch.cuda.get_device_properties(0).total_memory > 8e9:
        available.append(
            ModelConfig(
                name="GPT-Neo 1.3B",
                model_id="EleutherAI/gpt-neo-1.3B",
                params="1.3B",
                memory_gb=5.0,
                justification="Modelo más grande para comparar efecto de escala"
            )
        )

    return available


def validate_models_for_device(
    model_configs: List[ModelConfig], 
    device: torch.device
) -> List[ModelConfig]:
    """
    Validate that models can run on the specified device.
    
    Args:
        model_configs: List of models to validate
        device: Target device (CPU or CUDA)
    
    Returns:
        List of validated models (filtered if device has insufficient memory)
    """
    valid_models = []
    
    if device.type == "cpu":
        print("⚠️  Running on CPU - all models supported but may be slow")
        return model_configs
    
    # Check GPU memory
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    for model in model_configs:
        # Use 80% of available memory as safety margin
        if model.memory_gb <= gpu_memory_gb * 0.8:
            valid_models.append(model)
        else:
            print(f"⚠️  Skipping {model.name}: requires {model.memory_gb}GB, "
                  f"only {gpu_memory_gb:.1f}GB available")
    
    return valid_models


def print_model_info(models: List[ModelConfig] = None, highlight_default: bool = True):
    """
    Print information about models.
    
    Args:
        models: List of models to print. If None, prints all MODELS.
        highlight_default: If True, marks DEFAULT_MODEL with [DEFAULT]
    """
    if models is None:
        models = MODELS
    
    print("\n" + "="*70)
    print("MODELOS DISPONIBLES:")
    print("="*70)
    
    for i, model in enumerate(models, 1):
        is_default = highlight_default and (model.model_id == DEFAULT_MODEL.model_id)
        marker = " [DEFAULT]" if is_default else ""
        
        print(f"\n{i}. {model.name} ({model.params} params, ~{model.memory_gb}GB){marker}")
        print(f"   ID: {model.model_id}")
        print(f"   Justificación: {model.justification}")
    
    print("\n" + "="*70)
    
    if highlight_default:
        print(f"\n💡 Modelo por defecto para hito intermedio: {DEFAULT_MODEL.name}")
        print(f"   Uso: analyzer = UncertaintyAnalyzer('{DEFAULT_MODEL.model_id}')")


def print_default_model_info():
    """Print information about the DEFAULT_MODEL being used."""
    print("\n" + "="*70)
    print("MODELO SELECCIONADO PARA EXPERIMENTOS:")
    print("="*70)
    print(f"  Nombre: {DEFAULT_MODEL.name}")
    print(f"  ID: {DEFAULT_MODEL.model_id}")
    print(f"  Parámetros: {DEFAULT_MODEL.params}")
    print(f"  Memoria requerida: {DEFAULT_MODEL.memory_gb} GB")
    print(f"  Justificación: {DEFAULT_MODEL.justification}")
    print("="*70 + "\n")


# ============================================================================
# FUNCIONES PARA COMPARACIÓN MULTI-MODELO
# ============================================================================

def get_model_params_numeric(model_id: str) -> int:
    """
    Get numeric parameter count for a model.

    Args:
        model_id: HuggingFace model identifier

    Returns:
        Number of parameters as integer

    Example:
        >>> get_model_params_numeric('gpt2')
        124000000
    """
    return MODEL_PARAMS_NUMERIC.get(model_id, 0)


def get_model_comparison_config() -> dict:
    """
    Get configuration for multi-model comparison experiments.

    Returns:
        Dictionary with comparison configuration including:
        - model_ids: List of model IDs for comparison
        - model_configs: List of ModelConfig objects
        - params_map: Mapping of model_id to parameter count
        - size_order: Order of models by size (for plotting)

    Example:
        >>> config = get_model_comparison_config()
        >>> print(config['model_ids'])
        ['distilgpt2', 'gpt2', 'gpt2-medium']
    """
    model_configs = get_models_by_ids(COMPARISON_MODELS)

    return {
        'model_ids': COMPARISON_MODELS,
        'model_configs': model_configs,
        'params_map': MODEL_PARAMS_NUMERIC,
        'size_order': COMPARISON_MODELS,  # Already ordered small->large
        'n_models': len(COMPARISON_MODELS),
        'model_names': [m.name for m in model_configs],
        'model_params_str': [m.params for m in model_configs],
    }


def print_comparison_models_info():
    """Print information about models used in multi-model comparison."""
    print("\n" + "="*70)
    print("MODELOS PARA COMPARACIÓN MULTI-MODELO:")
    print("="*70)

    models = get_models_by_ids(COMPARISON_MODELS)

    for i, model in enumerate(models, 1):
        params_numeric = get_model_params_numeric(model.model_id)
        print(f"\n{i}. {model.name}")
        print(f"   ID: {model.model_id}")
        print(f"   Parámetros: {model.params} ({params_numeric:,})")
        print(f"   Memoria: ~{model.memory_gb}GB VRAM")
        print(f"   Justificación: {model.justification}")

    print("\n" + "="*70)
    print("Analisis de escalamiento: DistilGPT-2 -> GPT-2 -> GPT-2 Medium")
    print("   Hipotesis H1: DeltaH aumenta con tamano del modelo")
    print("   Hipotesis H2: Ranking de categorias consistente entre modelos")
    print("="*70 + "\n")