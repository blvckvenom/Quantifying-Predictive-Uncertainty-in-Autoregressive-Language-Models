# Migration Guide

This guide helps you migrate from the original single-file script to the new modular structure.

## Overview

**Before**: Single file (`Proyecto_llm_transformado a python.py`) with 4,605 lines
**After**: Modular package with 29 files organized in 10 modules

---

## Quick Reference: Where Did Everything Go?

### Imports and Setup

**Before**:
```python
# Lines 71-134 in original file
import random
import numpy as np
import torch
SEED = 42
random.seed(SEED)
# ... etc
```

**After**:
```python
from llm_uncertainty_analysis.config import setup_reproducibility, device
setup_reproducibility()
```

---

### Model Configuration

**Before**:
```python
# Lines 144-194 in original file
@dataclass
class ModelConfig:
    name: str
    model_id: str
    # ...

MODELS = [ModelConfig(...), ...]
```

**After**:
```python
from llm_uncertainty_analysis.models import ModelConfig, get_available_models
models = get_available_models(device)
```

---

### Context Categories

**Before**:
```python
# Lines 203-270 in original file
@dataclass
class ContextCategory:
    name: str
    description: str
    # ...

CONTEXT_CATEGORIES = [...]
```

**After**:
```python
from llm_uncertainty_analysis.models import ContextCategory, CONTEXT_CATEGORIES
```

---

### Dataset Management

**Before**:
```python
# Lines 297-584 in original file
class RealDatasetManager:
    def __init__(self, data_dir="data"):
        # ...
    def load_factual_data(self, n_samples=50):
        # ...
```

**After**:
```python
from llm_uncertainty_analysis.data_management import RealDatasetManager
dataset_manager = RealDatasetManager(data_dir="data")
datasets = dataset_manager.load_all_datasets(n_per_category=50)
```

---

### Entropy Calculation

**Before**:
```python
# Lines 927-993 in original file
def calculate_entropy(probs: np.ndarray, validate: bool = True) -> float:
    # ...
```

**After**:
```python
from llm_uncertainty_analysis.metrics import calculate_entropy
entropy = calculate_entropy(probs)
```

---

### Surprisal Calculation

**Before**:
```python
# Lines 1016-1076 in original file
def calculate_surprisal(prob_true: float, validate: bool = True) -> float:
    # ...
```

**After**:
```python
from llm_uncertainty_analysis.metrics import calculate_surprisal
surprisal = calculate_surprisal(prob_true)
```

---

### Perplexity Calculation

**Before**:
```python
# Lines 1078-1105 in original file
def calculate_perplexity(surprisal: float) -> float:
    # ...
```

**After**:
```python
from llm_uncertainty_analysis.metrics import calculate_perplexity
perplexity = calculate_perplexity(surprisal)
```

---

### Uncertainty Analyzer

**Before**:
```python
# Lines 1237-1334 in original file
class UncertaintyAnalyzer:
    def __init__(self, model_name: str, device: str = "cuda"):
        # ...
    def compute_token_metrics(self, text: str) -> pd.DataFrame:
        # ...
```

**After**:
```python
from llm_uncertainty_analysis.analysis import UncertaintyAnalyzer
analyzer = UncertaintyAnalyzer("gpt2", device="cuda")
results = analyzer.analyze_dataset(samples)
```

---

### Cohen's d

**Before**:
```python
# Lines 2157-2209 in original file
def calculate_cohens_d(group1, group2):
    # ...

def interpret_cohens_d(d):
    # ...
```

**After**:
```python
from llm_uncertainty_analysis.statistics import calculate_cohens_d, interpret_cohens_d
d = calculate_cohens_d(group1, group2)
magnitude, description = interpret_cohens_d(d)
```

---

### Mutual Information

**Before**:
```python
# Lines 2841-2904 in original file
def calculate_mutual_information(probs_without_evidence, probs_with_evidence):
    # ...

def interpret_mutual_information(mi_value):
    # ...
```

**After**:
```python
from llm_uncertainty_analysis.statistics import (
    calculate_mutual_information,
    interpret_mutual_information
)
mi = calculate_mutual_information(probs_baseline, probs_with_examples)
interpretation = interpret_mutual_information(mi)
```

---

### ICL Functions

**Before**:
```python
# Lines 3265-3330 in original file
def generate_icl_prompt(task_description, examples, query, n_examples=0):
    # ...

def measure_icl_entropy(model, tokenizer, prompt, device='cpu'):
    # ...
```

**After**:
```python
from llm_uncertainty_analysis.icl import generate_icl_prompt, measure_icl_entropy

prompt = generate_icl_prompt(task_desc, examples, query, n_examples=3)
entropy = measure_icl_entropy(model, tokenizer, prompt, device)
```

---

### Dataset Loading

**Before**:
```python
# Lines 297-584 in original file
# Manual dataset loading from various sources
# Scattered across the monolithic file
```

**After (NEW - Real Benchmarks)**:
```python
from llm_uncertainty_analysis.data import (
    load_lama_factual,      # LAMA TREx factual knowledge
    load_snli_balanced,     # Stanford SNLI logical reasoning
    load_gutenberg_poetry,  # Project Gutenberg creative text
    load_all_datasets       # Load all at once (recommended)
)

# Load all three datasets
datasets = load_all_datasets(
    factual_n=1000,   # LAMA: 1,000 factual prompts
    logical_n=300,    # SNLI: 300 logical prompts (balanced labels)
    creative_n=500,   # Gutenberg: 500 poetry line pairs
    seed=42
)

# Access by category
factual_examples = datasets['factual']
logical_examples = datasets['logical']
creative_examples = datasets['creative']

# Each example has the format:
# {
#   'prompt': str,      # The input prompt
#   'answer': str,      # The expected answer/continuation
#   'category': str,    # 'factual', 'logical', or 'creative'
#   'source': str,      # Dataset source identifier
#   'metadata': dict    # Additional source-specific metadata
# }
```

**Individual Loaders**:
```python
# LAMA TREx (Factual)
from llm_uncertainty_analysis.data import load_lama_factual
factual = load_lama_factual(
    n_samples=1000,
    data_dir='data/lama_data/data/TREx',
    seed=42
)
# Relations: P19 (birth), P37 (language), P106 (occupation), P36 (capital)

# Stanford SNLI (Logical)
from llm_uncertainty_analysis.data import load_snli_balanced
logical = load_snli_balanced(
    n_samples=300,
    split='validation',  # or 'train', 'test'
    seed=42
)
# Downloads automatically from Hugging Face
# Labels: entailment, neutral, contradiction (balanced)

# Project Gutenberg (Creative)
from llm_uncertainty_analysis.data import load_gutenberg_poetry
creative = load_gutenberg_poetry(
    n_samples=500,
    n_books=50,  # Stratified sampling across books
    data_file='data/gutenberg-poetry-v001.ndjson.gz',
    seed=42
)
# 3M+ poetry lines from 1,191 classical books
```

---

### Multi-Model Experiments

**Before**:
```python
# No multi-model comparison functionality in original file
```

**After (NEW)**:
```python
from llm_uncertainty_analysis.experiments import run_multi_model_icl_experiment
from llm_uncertainty_analysis.experiments import (
    validate_hypotheses,
    run_comprehensive_statistical_analysis
)

# Run multi-model ICL experiment
results = run_multi_model_icl_experiment(
    model_ids=['distilgpt2', 'gpt2', 'gpt2-medium'],
    n_examples_range=[0, 1, 2, 3, 5],  # k-shot configurations
    n_queries_per_config=10,
    device='cuda'
)

# Validate hypotheses
# H1: ΔH increases with model size (Spearman correlation)
# H2: Category ranking consistent across models (Kendall's W)
validation = validate_hypotheses(results)

print(f"H1 (Scaling): {validation['H1_scaling']['overall_conclusion']}")
print(f"H2 (Consistency): {validation['H2_consistency']['conclusion']}")

# Statistical analysis
stats = run_comprehensive_statistical_analysis(results)
print(f"ANOVA - Model effect: p = {stats['anova']['model_effect']['p_value']:.4f}")
print(f"ANOVA - Category effect: p = {stats['anova']['category_effect']['p_value']:.4f}")
```

**Multi-Model Components**:
```python
# 1. Main experiment runner
from llm_uncertainty_analysis.experiments.multi_model_icl_experiment import (
    run_multi_model_icl_experiment,
    run_single_model_icl_experiment
)

# 2. Hypothesis validation
from llm_uncertainty_analysis.experiments.model_comparison_analysis import (
    analyze_scaling_hypothesis,      # Test H1: ΔH vs model size
    analyze_consistency_hypothesis,   # Test H2: category ranking
    validate_hypotheses               # Complete validation
)

# 3. Statistical tests
from llm_uncertainty_analysis.experiments.multi_model_statistical_tests import (
    perform_two_way_anova,           # ANOVA with model and category
    perform_pairwise_model_comparisons,  # Pairwise t-tests
    calculate_effect_sizes           # Cohen's d for all pairs
)

# 4. Task configurations
from llm_uncertainty_analysis.experiments.icl_category_configs import (
    CATEGORY_CONFIGS,  # Predefined task descriptions
    get_examples,      # Get ICL examples by category
    get_queries        # Get test queries
)
```

---

## Complete Migration Example

### Original Approach (Single File)

```python
# Everything in one file
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# Setup
SEED = 42
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define classes
@dataclass
class ModelConfig:
    # ...

class RealDatasetManager:
    # ...

# Define functions
def calculate_entropy(probs):
    # ...

def calculate_surprisal(prob):
    # ...

class UncertaintyAnalyzer:
    # ...

# Load data
dataset_manager = RealDatasetManager()
data = dataset_manager.load_all_datasets()

# Run analysis
analyzer = UncertaintyAnalyzer("gpt2")
results = analyzer.analyze_dataset(data['factual'])

# Visualize
# ... plotting code ...
```

### New Approach (Modular)

```python
# Using the new package structure
from llm_uncertainty_analysis import (
    setup_reproducibility,
    setup_visualization,
    device,
    RealDatasetManager,
    UncertaintyAnalyzer,
    run_category_comparison,
    plot_entropy_by_category
)

# Setup
setup_reproducibility()
setup_visualization()

# Load data
dataset_manager = RealDatasetManager(data_dir="data")
datasets = dataset_manager.load_all_datasets(n_per_category=50)
all_samples = []
for samples in datasets.values():
    all_samples.extend(samples)

# Run analysis
analyzer = UncertaintyAnalyzer("gpt2", device=str(device))
results = run_category_comparison(analyzer, all_samples)

# Visualize
plot_entropy_by_category(results, save_path="entropy_plot.png")
```

---

## Script Migration

### Before: Running the Original Script

```bash
python "Proyecto_llm_transformado a python.py"
```

### After: Running the New Structure

**Option 1: Command Line**
```bash
cd llm_uncertainty_analysis
python run_analysis.py --model gpt2 --data-dir ../data --output-dir ../results
```

**Option 2: As a Library**
```python
from llm_uncertainty_analysis import *

# Your analysis code here
```

**Option 3: Interactive (Jupyter)**
```python
import sys
sys.path.append('path/to/llm_uncertainty_analysis')

from llm_uncertainty_analysis import *

# Interactive exploration
```

---

## Advantages of New Structure

### 1. Modularity
- **Before**: Everything in one file - hard to find specific functionality
- **After**: Organized modules - easy to locate and import what you need

### 2. Reusability
- **Before**: Copy-paste code between projects
- **After**: Import as library: `from llm_uncertainty_analysis import ...`

### 3. Testing
- **Before**: Hard to test individual components
- **After**: Easy to write unit tests for each module

### 4. Maintenance
- **Before**: Changes affect entire file
- **After**: Modify isolated components without side effects

### 5. Collaboration
- **Before**: Merge conflicts on single large file
- **After**: Work on different modules simultaneously

### 6. Documentation
- **Before**: Long docstrings in one file
- **After**: Dedicated README, module docs, and API reference

### 7. Version Control
- **Before**: Hard to track what changed
- **After**: Clear git history by module

---

## Common Migration Patterns

### Pattern 1: Import Consolidation

**Before**:
```python
# Scattered throughout the file
import torch
# ... 100 lines later ...
import numpy as np
# ... 200 lines later ...
from transformers import AutoModelForCausalLM
```

**After**:
```python
# All imports at top of config module
from llm_uncertainty_analysis import *
# or import only what you need
```

### Pattern 2: Function Organization

**Before**:
```python
# Functions scattered throughout
def calculate_entropy(...):
    pass

# ... 500 lines later ...

def calculate_surprisal(...):
    pass
```

**After**:
```python
# Organized by purpose
from llm_uncertainty_analysis.metrics import (
    calculate_entropy,
    calculate_surprisal,
    calculate_perplexity
)
```

### Pattern 3: Class Organization

**Before**:
```python
# Classes mixed with functions
class ModelConfig:
    pass

def some_function():
    pass

class ContextCategory:
    pass
```

**After**:
```python
# Classes grouped by module
from llm_uncertainty_analysis.models import ModelConfig, ContextCategory
```

---

## Breaking Changes

### None!

The new structure maintains **100% backward compatibility** with function signatures and behavior. The only difference is the import path.

**Old way (still works)**:
```python
# If you have the original file
from Proyecto_llm_transformado_a_python import calculate_entropy
```

**New way (recommended)**:
```python
from llm_uncertainty_analysis.metrics import calculate_entropy
```

---

## Step-by-Step Migration Process

### Step 1: Install Dependencies
```bash
cd llm_uncertainty_analysis
pip install -r requirements.txt
```

### Step 2: Update Imports

Replace:
```python
# Old imports (from original file)
from Proyecto_llm_transformado_a_python import calculate_entropy
```

With:
```python
# New imports (from package)
from llm_uncertainty_analysis.metrics import calculate_entropy
```

### Step 3: Update Data Paths

Ensure your data directory structure matches:
```
data/
├── consolidated_datasets.json
├── lama_data/
└── gutenberg-poetry-v001.ndjson.gz
```

### Step 4: Test
```bash
python run_analysis.py --n-samples 10  # Quick test with small dataset
```

### Step 5: Full Run
```bash
python run_analysis.py  # Full analysis
```

---

## Troubleshooting

### Issue: Import Error

**Error**: `ModuleNotFoundError: No module named 'llm_uncertainty_analysis'`

**Solution**:
```bash
# Option 1: Add to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/path/to/llm_uncertainty_analysis

# Option 2: Install as package
cd llm_uncertainty_analysis
pip install -e .

# Option 3: Add to sys.path in script
import sys
sys.path.append('/path/to/llm_uncertainty_analysis')
```

### Issue: Data Not Found

**Error**: `FileNotFoundError: data/consolidated_datasets.json`

**Solution**: Update data path in script:
```python
dataset_manager = RealDatasetManager(data_dir="../data")  # Adjust path
```

### Issue: CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solution**: Reduce batch size or use smaller model:
```python
analyzer = UncertaintyAnalyzer("distilgpt2", device="cuda")  # Smaller model
# OR
analyzer = UncertaintyAnalyzer("gpt2", device="cpu")  # Use CPU
```

---

## Support

If you encounter issues during migration:

1. Check this guide for common patterns
2. Review the STRUCTURE.md file for module organization
3. Check the README.md for usage examples
4. Review individual module docstrings

---

## Summary

**Key Takeaway**: The new structure provides the same functionality with better organization, easier usage, and professional structure suitable for production use, collaboration, and extension.

Migration is straightforward: update imports, adjust paths, and enjoy the benefits of modular code!
