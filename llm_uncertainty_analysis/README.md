# LLM Uncertainty Analysis

A comprehensive framework for quantifying predictive uncertainty in autoregressive language models.

## Overview

This project investigates how **entropy** and **surprisal** can serve as quantitative metrics to analyze uncertainty in Large Language Models (LLMs). We treat these metrics as a "thermometer" to measure the model's confidence in different linguistic contexts.

### Objectives

1. Quantify the change in predictive entropy when the model processes different types of context
2. Compare uncertainty between high and low certainty contexts
3. Analyze the effect of In-Context Learning (ICL) on entropy reduction

### Authors

- **Benito Fuentes**
- **Sebastian Vergara**

**Guide:** Simón Vidal
**Institution:** Universidad de Chile - EL7024-1
**Date:** November 2025

---

## Project Structure

```
llm_uncertainty_analysis/
├── config/                  # Configuration and settings
│   ├── settings.py         # Global settings (SEED, device, etc.)
│   ├── visualization.py    # Visualization configuration
│   └── __init__.py
├── data/                    # Dataset loaders for real benchmarks
│   ├── lama_loader.py      # LAMA TREx factual knowledge loader
│   ├── snli_loader.py      # Stanford SNLI logical reasoning loader
│   ├── gutenberg_loader.py # Project Gutenberg poetry loader
│   ├── unified_dataset_loader.py  # Unified API for all datasets
│   └── __init__.py
├── models/                  # Model and category definitions
│   ├── model_config.py     # ModelConfig dataclass
│   ├── context_category.py # ContextCategory dataclass
│   └── __init__.py
├── data_management/         # Dataset loading and management
│   ├── dataset_manager.py  # RealDatasetManager class
│   └── __init__.py
├── metrics/                 # Uncertainty metrics
│   ├── entropy.py          # Shannon entropy calculation
│   ├── surprisal.py        # Surprisal calculation
│   ├── perplexity.py       # Perplexity calculation
│   └── __init__.py
├── analysis/                # Main analysis pipeline
│   ├── uncertainty_analyzer.py  # UncertaintyAnalyzer class
│   └── __init__.py
├── statistics/              # Statistical analysis
│   ├── anova.py            # ANOVA, Tukey HSD, Bonferroni
│   ├── effect_size.py      # Cohen's d calculation
│   ├── mutual_information.py  # Mutual information
│   └── __init__.py
├── icl/                     # In-Context Learning
│   ├── prompt_generation.py   # ICL prompt generation
│   ├── entropy_measurement.py # ICL entropy measurement
│   └── __init__.py
├── visualization/           # Plotting and visualization
│   ├── plots.py            # Basic visualization functions
│   ├── advanced_plots.py   # Advanced statistical plots
│   └── __init__.py
├── experiments/             # Experiment runners
│   ├── category_comparison.py  # Category comparison experiment
│   ├── icl_experiment.py       # ICL experiment
│   ├── statistical_tests.py    # Complete statistical analysis
│   ├── complete_analysis.py    # Full pipeline orchestrator
│   ├── multi_model_icl_experiment.py    # MAIN: Multi-model ICL analysis
│   ├── model_comparison_analysis.py     # Hypothesis validation (H1, H2)
│   ├── multi_model_statistical_tests.py # ANOVA, pairwise comparisons
│   ├── icl_category_configs.py          # ICL task configurations
│   └── __init__.py
├── utils/                   # Utility functions
│   ├── helpers.py          # Helper utilities
│   └── __init__.py
├── run_analysis.py          # Basic analysis script
├── run_complete_analysis.py # Complete analysis with all tests
├── requirements.txt         # Dependencies
├── setup.py                # Package installation
├── __init__.py             # Package initialization
└── README.md               # This file
```

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip

### Setup

1. Clone or download the repository:
```bash
cd llm_uncertainty_analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. For GPU support (optional but recommended):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## Usage

### Quick Start

Run the complete analysis pipeline:

```bash
python run_analysis.py
```

### **NEW: Multi-Model ICL Analysis** (Hitos Inicial + Intermedio)

Run the comprehensive multi-model in-context learning analysis:

```bash
python run_multi_model_icl_analysis.py
```

This will:
1. Test 3 models: DistilGPT-2, GPT-2, GPT-2 Medium
2. Analyze 3 categories: factual, logical, creative
3. Test k-shot configurations: 0, 1, 2, 3, 5 examples
4. Validate scaling and consistency hypotheses
5. Generate results JSON files in `outputs/multi_model_icl/`

For quick testing (2 models, 3 k-shot values, 3 queries):
```bash
python run_multi_model_icl_analysis.py --quick
```

### Generate Publication Figures and Tables

After running the multi-model analysis, generate figures and tables:

```bash
# Generate all 5 figures
python paper/generate_figures.py --results outputs/multi_model_icl/results.json

# Generate all 4 tables
python paper/generate_tables.py --results outputs/multi_model_icl/results.json --validation outputs/multi_model_icl/hypothesis_validation.json

# Compile paper to PDF (requires LaTeX)
python paper/compile_paper.py
```

### Custom Configuration

```bash
python run_analysis.py --model gpt2-medium --data-dir ../data --output-dir ../results --n-samples 100
```

#### Arguments:
- `--model`: Model to use (default: `gpt2`)
- `--data-dir`: Data directory (default: `../data`)
- `--output-dir`: Output directory (default: `../results`)
- `--n-samples`: Number of samples per category (default: `50`)

### Using as a Library

```python
from llm_uncertainty_analysis import (
    setup_reproducibility,
    UncertaintyAnalyzer,
    RealDatasetManager,
    plot_entropy_by_category
)

# Setup
setup_reproducibility()

# Load data
dataset_manager = RealDatasetManager(data_dir="data")
datasets = dataset_manager.load_all_datasets(n_per_category=50)

# Analyze
analyzer = UncertaintyAnalyzer("gpt2", device="cuda")
results = analyzer.analyze_dataset(datasets['factual'])

# Visualize
plot_entropy_by_category(results, save_path="entropy_plot.png")
```

---

## Key Features

### Metrics

- **Entropy (H)**: Measures uncertainty in the probability distribution
  - Formula: `H = -Σ p(x) * log₂(p(x))`
  - Units: bits

- **Surprisal (S)**: Measures how "surprising" a specific token is
  - Formula: `S(x) = -log₂(p(x))`
  - Correlates with human reading times

- **Perplexity (PPL)**: Exponential of surprisal
  - Formula: `PPL = 2^S`
  - Interpretation: "equivalent to guessing from PPL equiprobable choices"

### Context Categories

1. **Factual**: Known facts with unique answers (low expected entropy)
   - Example: "The capital of France is"
   - Datasets: LAMA, SQuAD

2. **Logical**: Reasoning problems with clear logical structure (medium expected entropy)
   - Example: "If 2 + 2 = 4, then 3 + 3 ="
   - Datasets: SNLI, arithmetic

3. **Creative**: Open-ended generation with multiple valid continuations (high expected entropy)
   - Example: "Once upon a time, there was a"
   - Datasets: Gutenberg Poetry, WritingPrompts

### Statistical Analysis

- **ANOVA**: Test for significant differences between categories
- **Cohen's d**: Effect size measurement
- **Tukey HSD**: Post-hoc pairwise comparisons
- **Mutual Information**: Quantify information gain from evidence

#### ANOVA Example

```python
from llm_uncertainty_analysis.statistics import run_anova, calculate_eta_squared, run_tukey_hsd
from llm_uncertainty_analysis.visualization import plot_anova_boxplot
import numpy as np

# Prepare data by category
data_by_category = {
    'factual': np.array([3.2, 3.5, 3.1, 3.4, 3.3]),
    'logical': np.array([5.1, 5.3, 5.0, 5.2, 5.4]),
    'creative': np.array([7.5, 7.8, 7.6, 7.9, 7.7])
}

# Run one-way ANOVA
f_stat, p_value = run_anova(data_by_category)
print(f"ANOVA: F = {f_stat:.2f}, p = {p_value:.4f}")

# Calculate effect size (eta-squared)
eta_sq = calculate_eta_squared(data_by_category)
print(f"Effect size (η²) = {eta_sq:.3f}")

# Post-hoc Tukey HSD
all_values = np.concatenate(list(data_by_category.values()))
all_labels = []
for cat, data in data_by_category.items():
    all_labels.extend([cat] * len(data))

tukey_df = run_tukey_hsd(all_values, all_labels, alpha=0.05)
print(tukey_df)

# Visualize ANOVA results
plot_anova_boxplot(data_by_category, f_stat, p_value, save_path='anova_plot.png')
```

#### Advanced Plots Example

```python
from llm_uncertainty_analysis.visualization.advanced_plots import (
    plot_anova_boxplot,
    plot_tukey_hsd_intervals,
    plot_correlation_heatmap,
    plot_multi_model_comparison
)

# 1. ANOVA visualization with statistical annotations
plot_anova_boxplot(
    data_by_category={'factual': [...], 'logical': [...], 'creative': [...]},
    f_stat=45.2,
    p_value=0.0001,
    save_path='anova_boxplot.png'
)

# 2. Tukey HSD confidence intervals
tukey_df = run_tukey_hsd(all_values, all_labels)
plot_tukey_hsd_intervals(tukey_df, save_path='tukey_intervals.png')

# 3. Multi-model comparison (for ICL experiments)
plot_multi_model_comparison(
    results={'distilgpt2': {...}, 'gpt2': {...}},
    save_path='multi_model_comparison.png'
)
```

### In-Context Learning (ICL)

- Analyze how examples affect model uncertainty
- Test 0-shot through k-shot configurations
- Measure entropy reduction with increasing examples

### Multi-Model Comparative Analysis

**NEW**: Comprehensive framework for comparing ICL effectiveness across model scales:

**Hypotheses Tested:**
- **H1 (Scaling)**: ΔH increases with model size
  - Test: Spearman correlation between model parameters and entropy reduction
  - Expected: Larger models leverage context more effectively

- **H2 (Consistency)**: Category rankings consistent across models
  - Test: Kendall's W coefficient of concordance
  - Expected: Task characteristics dominate regardless of model size

**Models Compared:**
1. DistilGPT-2 (82M parameters)
2. GPT-2 (124M parameters)
3. GPT-2 Medium (355M parameters)

**Analysis Includes:**
- Entropy trajectories H(k) for each (model, category) pair
- Statistical significance testing (ANOVA, pairwise comparisons)
- Effect size calculations (Cohen's d)
- Efficiency metrics (ΔH per parameter)
- Convergence pattern analysis

**Outputs:**
- 3 JSON result files (results, statistics, validation)
- 5 publication-quality figures (PDF, 300 DPI)
- 4 LaTeX tables ready for IEEE format
- Complete academic paper in paper/icl_multi_model_analysis.tex

---

## Data

The project uses three real datasets:

1. **LAMA** (Language Model Analysis) - Factual knowledge
2. **SNLI** (Stanford Natural Language Inference) - Logical reasoning
3. **Gutenberg Poetry Corpus** - Creative text

Data should be placed in the following structure:
```
data/
├── consolidated_datasets.json
├── lama_data/
│   └── data/
│       └── TREx/
│           └── *.jsonl
└── gutenberg-poetry-v001.ndjson.gz
```

### Dataset Loaders

The `data/` module provides specialized loaders for each dataset:

#### 1. LAMA TREx Factual Loader

```python
from llm_uncertainty_analysis.data import load_lama_factual

# Load 1,000 factual prompts from 4 relations
factual_examples = load_lama_factual(
    n_samples=1000,
    data_dir='data/lama_data/data/TREx',
    seed=42
)

# Relations: P19 (birth place), P37 (language), P106 (occupation), P36 (capital)
# Format: {'prompt': str, 'answer': str, 'category': 'factual', 'source': str, 'metadata': dict}
```

#### 2. Stanford SNLI Logical Loader

```python
from llm_uncertainty_analysis.data import load_snli_balanced

# Load 300 balanced logical prompts (100 per label)
logical_examples = load_snli_balanced(
    n_samples=300,
    split='validation',
    seed=42
)

# Labels: entailment, neutral, contradiction
# Downloads automatically from Hugging Face
```

#### 3. Project Gutenberg Poetry Loader

```python
from llm_uncertainty_analysis.data import load_gutenberg_poetry

# Load 500 poetry line pairs from 50 books
creative_examples = load_gutenberg_poetry(
    n_samples=500,
    n_books=50,
    data_file='data/gutenberg-poetry-v001.ndjson.gz',
    seed=42
)

# Stratified sampling: ~10 line pairs per book for diversity
```

#### 4. Unified Loader (Recommended)

```python
from llm_uncertainty_analysis.data import load_all_datasets

# Load all three datasets at once
datasets = load_all_datasets(
    factual_n=1000,
    logical_n=300,
    creative_n=500,
    seed=42
)

# Access by category
factual = datasets['factual']
logical = datasets['logical']
creative = datasets['creative']
```

---

## Output

The analysis produces:

1. **Results CSV**: Detailed metrics for each sample
2. **Figures**: Visualizations saved as PNG files
   - Entropy distribution by category
   - ICL analysis plots
   - Effect size comparisons

Output directory structure:
```
results/
├── results.csv
└── figures/
    ├── entropy_by_category.png
    ├── icl_analysis.png
    └── cohens_d_comparison.png
```

---

## Theoretical Background

### Shannon Entropy

Entropy measures the average uncertainty in a distribution:

```
H(Y_t | Y_<t) = -Σ p(v | Y_<t) * log₂(p(v | Y_<t))
```

- **Interpretation**: Higher entropy = higher uncertainty
- **Units**: bits (using log base 2)

### Surprisal

Surprisal quantifies how "surprising" a specific token is:

```
S(y_t | Y_<t) = -log₂(p(y_t | Y_<t))
```

- **Interpretation**: Higher surprisal = less expected token
- **Cognitive relation**: Correlates with human reading times (Levy, 2008)

### References

1. **Levy (2008)** - "Expectation-based syntactic comprehension"
2. **Goodkind & Bicknell (2018)** - "Predictive power of word surprisal"
3. **Petroni et al. (2019)** - "Language models as knowledge bases?"
4. **Gonen et al. (2022)** - "Detecting and calibrating uncertainty"

---

## Development

### Adding New Experiments

Create a new file in `experiments/` and define your experiment function:

```python
def run_my_experiment(analyzer, samples):
    """My custom experiment."""
    results = analyzer.analyze_dataset(samples)
    # Custom processing
    return results
```

### Adding New Visualizations

Add plotting functions to `visualization/plots.py`:

```python
def plot_my_visualization(data, save_path=None):
    """Create my custom plot."""
    plt.figure(figsize=(10, 6))
    # Plotting code
    if save_path:
        plt.savefig(save_path, dpi=300)
    return plt.gcf()
```

---

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use a smaller model
2. **Missing data files**: Ensure data directory structure matches expected format
3. **Import errors**: Verify all dependencies are installed

### Performance Tips

- Use GPU for faster computation (set `device="cuda"`)
- Reduce `n_samples` for faster prototyping
- Cache results to avoid recomputation

---

## License

This project is part of academic research at Universidad de Chile.

---

## Citation

If you use this code in your research, please cite:

```
@misc{fuentes2024llmuncertainty,
  title={Quantifying Predictive Uncertainty in Autoregressive Language Models},
  author={Fuentes, Benito and Vergara, Sebastian},
  year={2024},
  institution={Universidad de Chile},
  note={EL7024-1 Project}
}
```

---

## Contact

For questions or issues, please contact:
- Benito Fuentes
- Sebastian Vergara

**Guide:** Simón Vidal
**Universidad de Chile** - EL7024-1
