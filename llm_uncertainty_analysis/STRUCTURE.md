# Project Structure

This document provides a detailed overview of the project structure and organization.

## Complete Directory Tree

```
llm_uncertainty_analysis/
│
├── __init__.py                          # Main package initialization
├── run_analysis.py                      # Main execution script
├── requirements.txt                     # Project dependencies
├── README.md                            # Project documentation
├── STRUCTURE.md                         # This file
│
├── config/                              # Configuration module
│   ├── __init__.py                      # Config module exports
│   ├── settings.py                      # Global settings (SEED, device, reproducibility)
│   └── visualization.py                 # Visualization configuration (colors, styles)
│
├── data/                                # Dataset loaders for real benchmarks
│   ├── __init__.py                      # Data module exports
│   ├── lama_loader.py                   # LAMA TREx factual knowledge loader
│   ├── snli_loader.py                   # Stanford SNLI logical reasoning loader
│   ├── gutenberg_loader.py              # Project Gutenberg poetry loader (3M+ lines)
│   └── unified_dataset_loader.py        # Unified API for all three datasets
│
├── models/                              # Model and category definitions
│   ├── __init__.py                      # Models module exports
│   ├── model_config.py                  # ModelConfig dataclass and helpers
│   └── context_category.py              # ContextCategory dataclass and categories
│
├── data_management/                     # Dataset loading and management
│   ├── __init__.py                      # Data management module exports
│   └── dataset_manager.py               # RealDatasetManager class
│
├── metrics/                             # Uncertainty metrics calculation
│   ├── __init__.py                      # Metrics module exports
│   ├── entropy.py                       # Shannon entropy calculation
│   ├── surprisal.py                     # Surprisal (self-information) calculation
│   └── perplexity.py                    # Perplexity calculation
│
├── analysis/                            # Main analysis pipeline
│   ├── __init__.py                      # Analysis module exports
│   └── uncertainty_analyzer.py          # UncertaintyAnalyzer class (complete pipeline)
│
├── statistics/                          # Statistical analysis
│   ├── __init__.py                      # Statistics module exports
│   ├── effect_size.py                   # Cohen's d calculation and interpretation
│   └── mutual_information.py            # Mutual information calculation
│
├── icl/                                 # In-Context Learning analysis
│   ├── __init__.py                      # ICL module exports
│   ├── prompt_generation.py             # ICL prompt generation functions
│   └── entropy_measurement.py           # ICL entropy measurement functions
│
├── visualization/                       # Plotting and visualization
│   ├── __init__.py                      # Visualization module exports
│   └── plots.py                         # All plotting functions
│
├── experiments/                         # Experiment runners
│   ├── __init__.py                      # Experiments module exports
│   ├── category_comparison.py           # Category comparison experiment
│   ├── icl_experiment.py                # In-Context Learning experiment
│   ├── statistical_tests.py             # Complete statistical analysis
│   ├── complete_analysis.py             # Full pipeline orchestrator
│   ├── multi_model_icl_experiment.py    # MAIN: Multi-model ICL analysis
│   ├── model_comparison_analysis.py     # Hypothesis validation (H1, H2)
│   ├── multi_model_statistical_tests.py # ANOVA, pairwise comparisons, effect sizes
│   └── icl_category_configs.py          # ICL task configurations and examples
│
└── utils/                               # Utility functions
    ├── __init__.py                      # Utils module exports
    └── helpers.py                       # Helper utilities (directory management, etc.)
```

---

## Module Breakdown

### 1. Configuration Module (`config/`)

**Purpose**: Centralize all configuration settings and reproducibility setup.

- **settings.py**:
  - Global constants (SEED, device)
  - Reproducibility setup functions
  - Device information printing

- **visualization.py**:
  - Matplotlib and seaborn configuration
  - Color palettes for categories
  - Style settings

### 2. Data Module (`data/`)

**Purpose**: Load real-world benchmark datasets for ICL experiments.

- **lama_loader.py**:
  - `load_lama_factual()` - Load factual knowledge prompts from LAMA TREx
  - `load_trex_relation()` - Load specific Wikidata relation
  - Supported relations: P19 (birth place), P37 (language), P106 (occupation), P36 (capital)
  - 1,000 prompts total (250 per relation)

- **snli_loader.py**:
  - `load_snli_balanced()` - Load logical reasoning prompts from Stanford SNLI
  - Downloads automatically from Hugging Face
  - Balanced label distribution: entailment, neutral, contradiction
  - 300 prompts (100 per label)

- **gutenberg_loader.py**:
  - `load_gutenberg_poetry()` - Load creative text from Project Gutenberg
  - Stratified sampling across 50 books
  - 3M+ poetry lines available
  - 500 line pairs for next-line prediction

- **unified_dataset_loader.py**:
  - `load_all_datasets()` - Load all three datasets at once
  - `get_dataset_statistics()` - Compute comprehensive statistics
  - `export_datasets_to_json()` - Save datasets for inspection
  - Main entry point for experiments

### 3. Models Module (`models/`)

**Purpose**: Define data models and configurations for the analysis.

- **model_config.py**:
  - `ModelConfig` dataclass
  - `get_available_models()` function
  - Model information printing

- **context_category.py**:
  - `ContextCategory` dataclass
  - `CONTEXT_CATEGORIES` list (factual, logical, creative)
  - Entropy classification methods
  - Category information printing

### 3. Data Management Module (`data_management/`)

**Purpose**: Handle loading and management of datasets.

- **dataset_manager.py**:
  - `RealDatasetManager` class
  - Methods: `load_factual_data()`, `load_logical_data()`, `load_creative_data()`
  - Dataset validation and statistics

### 4. Metrics Module (`metrics/`)

**Purpose**: Calculate uncertainty metrics from probability distributions.

- **entropy.py**:
  - `calculate_entropy()` - Shannon entropy calculation
  - `calculate_entropy_from_logits()` - Direct calculation from logits

- **surprisal.py**:
  - `calculate_surprisal()` - Self-information calculation

- **perplexity.py**:
  - `calculate_perplexity()` - Perplexity from surprisal

### 5. Analysis Module (`analysis/`)

**Purpose**: Main uncertainty analysis pipeline.

- **uncertainty_analyzer.py**:
  - `UncertaintyAnalyzer` class
  - Methods:
    - `load_model()` - Load HuggingFace model
    - `compute_token_metrics()` - Per-token metrics
    - `compute_sequence_metrics()` - Sequence-level metrics
    - `analyze_dataset()` - Batch analysis

### 6. Statistics Module (`statistics/`)

**Purpose**: Statistical analysis and hypothesis testing.

- **effect_size.py**:
  - `calculate_cohens_d()` - Effect size calculation
  - `interpret_cohens_d()` - Interpretation of Cohen's d values

- **mutual_information.py**:
  - `calculate_mutual_information()` - MI calculation
  - `interpret_mutual_information()` - Interpretation

### 7. ICL Module (`icl/`)

**Purpose**: In-Context Learning analysis.

- **prompt_generation.py**:
  - `generate_icl_prompt()` - Generate k-shot prompts

- **entropy_measurement.py**:
  - `measure_icl_entropy()` - Measure entropy in ICL scenarios

### 8. Visualization Module (`visualization/`)

**Purpose**: Create plots and visualizations.

- **plots.py**:
  - `plot_entropy_by_category()` - Boxplots by category
  - `plot_icl_analysis()` - Comprehensive ICL plots (4-panel)
  - `plot_cohens_d_comparison()` - Effect size bar plots

### 9. Experiments Module (`experiments/`)

**Purpose**: Orchestrate complete experiments.

- **category_comparison.py**:
  - `run_category_comparison()` - Main category comparison
  - `calculate_descriptive_stats()` - Descriptive statistics

- **icl_experiment.py**:
  - `run_icl_experiment()` - ICL experiment with varying examples

- **statistical_tests.py**:
  - Complete statistical analysis pipeline
  - ANOVA, t-tests, effect sizes

- **complete_analysis.py**:
  - `run_complete_analysis()` - Full pipeline orchestrator
  - Combines all experiments and analysis

- **multi_model_icl_experiment.py** (MAIN SCRIPT):
  - `run_multi_model_icl_experiment()` - Multi-model ICL analysis
  - `run_single_model_icl_experiment()` - Single model analysis
  - Tests 3 models across 3 categories with k-shot configurations
  - Main entry point for multi-model experiments

- **model_comparison_analysis.py**:
  - `analyze_scaling_hypothesis()` - Test H1 (model size effect)
  - `analyze_consistency_hypothesis()` - Test H2 (category ranking)
  - `validate_hypotheses()` - Complete hypothesis validation
  - Uses Spearman correlation and Kendall's W

- **multi_model_statistical_tests.py**:
  - `perform_two_way_anova()` - ANOVA with model and category factors
  - `perform_pairwise_model_comparisons()` - Pairwise t-tests
  - `calculate_effect_sizes()` - Cohen's d for all comparisons
  - Bonferroni correction for multiple comparisons

- **icl_category_configs.py**:
  - `CATEGORY_CONFIGS` - Task descriptions and examples
  - `get_examples()` - Retrieve ICL examples by category
  - `get_queries()` - Retrieve test queries
  - Configuration for factual, logical, creative tasks

### 10. Utils Module (`utils/`)

**Purpose**: General utility functions.

- **helpers.py**:
  - `ensure_directory()` - Create directories if needed
  - `save_results()` - Save results in multiple formats

---

## Main Execution Script

**run_analysis.py**: Complete pipeline orchestration

```
Command line interface:
├── Parse arguments (model, data-dir, output-dir, n-samples)
├── Setup (reproducibility, visualization, device)
├── Load datasets
├── Initialize analyzer
├── Run experiments
├── Create visualizations
└── Save results
```

---

## File Count Summary

```
Total Python files: 41
├── Config:         2 files + 1 __init__.py
├── Data:           4 files + 1 __init__.py  (NEW: lama, snli, gutenberg, unified loaders)
├── Models:         2 files + 1 __init__.py
├── Data Mgmt:      1 file  + 1 __init__.py
├── Metrics:        3 files + 1 __init__.py
├── Analysis:       1 file  + 1 __init__.py
├── Statistics:     2 files + 1 __init__.py
├── ICL:            2 files + 1 __init__.py
├── Visualization:  2 files + 1 __init__.py  (plots.py, advanced_plots.py)
├── Experiments:    9 files + 1 __init__.py  (UPDATED: added 5 multi-model files)
├── Utils:          1 file  + 1 __init__.py
└── Main:           2 files (run_analysis.py, __init__.py)
```

---

## Key Design Principles

1. **Modular**: Each module has a single, clear responsibility
2. **Reusable**: Functions can be imported and used independently
3. **Documented**: Every module, class, and function has docstrings
4. **Testable**: Clear separation makes unit testing straightforward
5. **Extensible**: Easy to add new experiments, metrics, or visualizations

---

## How the Modules Interact

```
run_analysis.py
    │
    ├─> config (setup)
    │
    ├─> models (get configurations)
    │
    ├─> data_management (load datasets)
    │       │
    │       └─> Returns: samples list
    │
    ├─> analysis (create analyzer)
    │       │
    │       ├─> metrics (calculate uncertainty)
    │       │
    │       └─> Returns: results DataFrame
    │
    ├─> experiments (run analyses)
    │       │
    │       ├─> statistics (calculate stats)
    │       ├─> icl (ICL experiments)
    │       │
    │       └─> Returns: processed results
    │
    ├─> visualization (create plots)
    │       │
    │       └─> Saves: PNG figures
    │
    └─> utils (save results)
            │
            └─> Saves: CSV, JSON files
```

---

## Benefits of This Structure

### For Development
- **Easy to navigate**: Clear folder structure
- **Easy to extend**: Add new files in appropriate modules
- **Easy to debug**: Isolate issues to specific modules
- **Easy to test**: Test individual components

### For Users
- **Easy to use**: Simple imports from top-level package
- **Easy to customize**: Override specific components
- **Easy to understand**: Clear separation of concerns
- **Easy to integrate**: Use as library or standalone

### For Maintenance
- **Easy to update**: Modify isolated components
- **Easy to document**: Each module is self-contained
- **Easy to refactor**: Clear dependencies between modules
- **Easy to version**: Track changes by module

---

## Comparison with Original File

**Original**: Single file with 4,605 lines
**New structure**: 29 organized files across 10 modules

### Advantages:
- ✅ Much easier to navigate and find specific functionality
- ✅ Better code organization and separation of concerns
- ✅ Reusable components that can be imported individually
- ✅ Easier to test individual components
- ✅ Easier to extend with new features
- ✅ Better documentation structure
- ✅ Follows Python best practices
- ✅ Can be installed as a package
- ✅ Supports both CLI and library usage
- ✅ Professional project structure

---

## Future Extensions

This structure makes it easy to add:

1. **New experiments**: Add files to `experiments/`
2. **New metrics**: Add files to `metrics/`
3. **New visualizations**: Add functions to `visualization/plots.py`
4. **New datasets**: Extend `data_management/dataset_manager.py`
5. **New models**: Add configurations to `models/model_config.py`
6. **Tests**: Create `tests/` directory mirroring structure
7. **Documentation**: Add `docs/` directory
8. **Examples**: Add `examples/` directory with notebooks

---

## Recommended Next Steps

1. **Add tests**: Create `tests/` directory with unit tests
2. **Add examples**: Create `examples/` with Jupyter notebooks
3. **Add docs**: Create detailed API documentation
4. **Add setup.py**: Make installable with pip
5. **Add CI/CD**: GitHub Actions for testing
6. **Add type hints**: Complete type annotations
7. **Add logging**: Structured logging throughout
8. **Add config files**: YAML/JSON configuration files
