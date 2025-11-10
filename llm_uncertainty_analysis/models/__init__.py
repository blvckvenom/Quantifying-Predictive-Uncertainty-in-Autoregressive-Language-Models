"""
Models Module

This module contains model and context category configurations.
"""

from .model_config import (
    ModelConfig,
    MODELS,
    DEFAULT_MODEL,
    get_model_by_id,
    get_models_by_ids,
    get_available_models,
    validate_models_for_device,
    print_model_info,
    print_default_model_info
)

from .context_category import (
    ContextCategory,
    CONTEXT_CATEGORIES,
    get_category_by_name,
    print_category_info,
    demo_entropy_classification,
    validate_hypothesis
)

__all__ = [
    # Model configuration
    'ModelConfig',
    'MODELS',
    'DEFAULT_MODEL',
    'get_model_by_id',
    'get_models_by_ids',
    'get_available_models',
    'validate_models_for_device',
    'print_model_info',
    'print_default_model_info',

    # Context categories
    'ContextCategory',
    'CONTEXT_CATEGORIES',
    'get_category_by_name',
    'print_category_info',
    'demo_entropy_classification',
    'validate_hypothesis'
]