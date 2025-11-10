## Quick Start Guide

Get started with LLM Uncertainty Analysis in 5 minutes!

---

## Installation

### 1. Navigate to Project Directory
```bash
cd llm_uncertainty_analysis
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. (Optional) Install as Package
```bash
pip install -e .
```

---

## Basic Usage

### Option 1: Command Line (Easiest)

Run the complete analysis pipeline:

```bash
python run_analysis.py
```

With custom options:

```bash
python run_analysis.py --model gpt2 --data-dir ../data --output-dir ../results --n-samples 50
```

**Arguments:**
- `--model`: Model name (default: `gpt2`)
- `--data-dir`: Data directory path (default: `../data`)
- `--output-dir`: Output directory path (default: `../results`)
- `--n-samples`: Samples per category (default: `50`)

---

### Option 2: Python Script

Create a script `my_analysis.py`:

```python
from llm_uncertainty_analysis import (
    setup_reproducibility,
    setup_visualization,
    device,
    RealDatasetManager,
    UncertaintyAnalyzer,
    plot_entropy_by_category
)

# Setup
setup_reproducibility()
setup_visualization()

# Load data
dataset_manager = RealDatasetManager(data_dir="data")
datasets = dataset_manager.load_all_datasets(n_per_category=50)

# Combine samples
all_samples = []
for samples in datasets.values():
    all_samples.extend(samples)

# Analyze
analyzer = UncertaintyAnalyzer("gpt2", device=str(device))
results = analyzer.analyze_dataset(all_samples)

# Visualize
plot_entropy_by_category(results, save_path="entropy_plot.png")

# Print statistics
print(results.groupby('category')['mean_entropy'].describe())
```

Run it:
```bash
python my_analysis.py
```

---

### Option 3: Interactive (Jupyter Notebook)

```python
# Cell 1: Setup
from llm_uncertainty_analysis import *
setup_reproducibility()
setup_visualization()

# Cell 2: Load data
dataset_manager = RealDatasetManager(data_dir="../data")
datasets = dataset_manager.load_all_datasets(n_per_category=10)  # Small for quick test

# Cell 3: Analyze
analyzer = UncertaintyAnalyzer("gpt2", device=str(device))
results = analyzer.analyze_dataset(datasets['factual'][:5])  # Test with 5 samples

# Cell 4: View results
results[['text', 'category', 'mean_entropy', 'mean_surprisal']]

# Cell 5: Visualize
plot_entropy_by_category(results)
```

---

## Example Workflows

### Workflow 1: Compare Categories

```python
from llm_uncertainty_analysis import *

# Setup
setup_reproducibility()
dataset_manager = RealDatasetManager(data_dir="data")

# Load all categories
factual = dataset_manager.load_factual_data(n_samples=20)
logical = dataset_manager.load_logical_data(n_samples=20)
creative = dataset_manager.load_creative_data(n_samples=20)

# Analyze each
analyzer = UncertaintyAnalyzer("gpt2")
results_factual = analyzer.analyze_dataset(factual)
results_logical = analyzer.analyze_dataset(logical)
results_creative = analyzer.analyze_dataset(creative)

# Compare
import pandas as pd
all_results = pd.concat([results_factual, results_logical, results_creative])
print(all_results.groupby('category')['mean_entropy'].mean())
```

### Workflow 2: ICL Experiment

```python
from llm_uncertainty_analysis import *
from llm_uncertainty_analysis.icl import generate_icl_prompt, measure_icl_entropy

# Setup
setup_reproducibility()
analyzer = UncertaintyAnalyzer("gpt2")

# Define task
task = "Answer factual questions."
examples = [
    ("What is the capital of France?", "Paris"),
    ("What is 2 + 2?", "4"),
    ("What color is the sky?", "Blue")
]
query = "What is the capital of Italy?"

# Test different numbers of examples
for n in [0, 1, 2, 3]:
    prompt = generate_icl_prompt(task, examples, query, n_examples=n)
    entropy = measure_icl_entropy(analyzer.model, analyzer.tokenizer, prompt, str(device))
    print(f"n={n}: H={entropy:.3f} bits")
```

### Workflow 3: Single Sample Analysis

```python
from llm_uncertainty_analysis import *

# Setup
analyzer = UncertaintyAnalyzer("gpt2")

# Analyze single text
text = "The capital of France is"
token_metrics = analyzer.compute_token_metrics(text)
sequence_metrics = analyzer.compute_sequence_metrics(text)

# View per-token details
print("\nPer-token metrics:")
print(token_metrics[['token', 'entropy', 'surprisal', 'probability']])

# View sequence summary
print("\nSequence metrics:")
for key, value in sequence_metrics.items():
    if isinstance(value, (int, float)):
        print(f"  {key}: {value:.3f}")
```

### Workflow 4: Statistical Analysis with ANOVA

```python
from llm_uncertainty_analysis import *
from llm_uncertainty_analysis.statistics import (
    run_anova, calculate_eta_squared, run_tukey_hsd
)
from llm_uncertainty_analysis.visualization.advanced_plots import plot_anova_boxplot
import numpy as np

# Setup
setup_reproducibility()
analyzer = UncertaintyAnalyzer("gpt2")

# Load data from real benchmarks
from llm_uncertainty_analysis.data import load_all_datasets
datasets = load_all_datasets(factual_n=50, logical_n=50, creative_n=50)

# Analyze each category
results = {}
for category, examples in datasets.items():
    entropies = []
    for ex in examples[:10]:  # Sample 10 per category
        metrics = analyzer.compute_sequence_metrics(ex['prompt'])
        entropies.append(metrics['mean_entropy'])
    results[category] = np.array(entropies)

# Run ANOVA
f_stat, p_value = run_anova(results)
print(f"\nANOVA Results:")
print(f"  F-statistic: {f_stat:.2f}")
print(f"  p-value: {p_value:.4f}")
print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}")

# Effect size
eta_sq = calculate_eta_squared(results)
print(f"  Effect size (η²): {eta_sq:.3f}")

# Post-hoc Tukey HSD
all_values = np.concatenate(list(results.values()))
all_labels = []
for cat, data in results.items():
    all_labels.extend([cat] * len(data))

tukey_df = run_tukey_hsd(all_values, all_labels)
print(f"\nTukey HSD Post-hoc Test:")
print(tukey_df)

# Visualize
plot_anova_boxplot(results, f_stat, p_value, save_path='anova_analysis.png')
print(f"\nPlot saved to: anova_analysis.png")
```

---

## Common Tasks

### Task 1: Change Model

```python
# Use a different model
analyzer = UncertaintyAnalyzer("gpt2-medium", device="cuda")
# or
analyzer = UncertaintyAnalyzer("distilgpt2", device="cpu")
```

### Task 2: Save Results

```python
# Save to CSV
results.to_csv("results.csv", index=False)

# Save to JSON
results.to_json("results.json", orient="records")

# Save to pickle
results.to_pickle("results.pkl")
```

### Task 3: Load Specific Category

```python
dataset_manager = RealDatasetManager(data_dir="data")

# Load only factual data
factual_samples = dataset_manager.load_factual_data(n_samples=50)

# Load only creative data
creative_samples = dataset_manager.load_creative_data(n_samples=50)
```

### Task 4: Custom Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Custom plot
plt.figure(figsize=(12, 6))
sns.violinplot(data=results, x='category', y='mean_entropy')
plt.title('Entropy Distribution by Category')
plt.ylabel('Mean Entropy (bits)')
plt.savefig('custom_plot.png', dpi=300)
plt.show()
```

---

## Troubleshooting

### Issue: Module not found

```bash
# Solution 1: Install as package
pip install -e .

# Solution 2: Add to Python path
export PYTHONPATH=$PYTHONPATH:/path/to/llm_uncertainty_analysis

# Solution 3: In Python
import sys
sys.path.append('/path/to/llm_uncertainty_analysis')
```

### Issue: CUDA out of memory

```python
# Solution 1: Use CPU
analyzer = UncertaintyAnalyzer("gpt2", device="cpu")

# Solution 2: Use smaller model
analyzer = UncertaintyAnalyzer("distilgpt2", device="cuda")

# Solution 3: Reduce batch size (analyze fewer samples at once)
results = analyzer.analyze_dataset(samples[:10])  # Process in smaller batches
```

### Issue: Data not found

```python
# Check current directory
import os
print(os.getcwd())

# Adjust path
dataset_manager = RealDatasetManager(data_dir="../data")  # Go up one level
# or
dataset_manager = RealDatasetManager(data_dir="/absolute/path/to/data")
```

---

## Quick Test

Test if everything is working:

```python
# quick_test.py
from llm_uncertainty_analysis import *

print("Testing setup...")
setup_reproducibility()
print("✓ Setup OK")

print("\nTesting model loading...")
analyzer = UncertaintyAnalyzer("gpt2", device="cpu")
print("✓ Model loaded")

print("\nTesting analysis...")
sample = {"prompt": "The capital of France is", "answer": "Paris", "category": "factual"}
result = analyzer.compute_sequence_metrics(sample["prompt"])
print(f"✓ Analysis OK - Entropy: {result['mean_entropy']:.3f} bits")

print("\n✅ All tests passed!")
```

Run it:
```bash
python quick_test.py
```

Expected output:
```
Testing setup...
✓ Setup OK

Testing model loading...
Loading gpt2...
✓ Model loaded

Testing analysis...
✓ Analysis OK - Entropy: 6.234 bits

✅ All tests passed!
```

---

## Next Steps

1. **Read the README**: Detailed documentation → `README.md`
2. **See structure**: Project organization → `STRUCTURE.md`
3. **Migration guide**: From old code → `MIGRATION_GUIDE.md`
4. **Run full analysis**: `python run_analysis.py`
5. **Explore modules**: Check individual module docstrings

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────────┐
│                   IMPORT CHEAT SHEET                        │
├─────────────────────────────────────────────────────────────┤
│ Setup:                                                      │
│   from llm_uncertainty_analysis import setup_reproducibility│
│   from llm_uncertainty_analysis import device              │
│                                                             │
│ Data (OLD - legacy):                                        │
│   from llm_uncertainty_analysis import RealDatasetManager  │
│                                                             │
│ Data (NEW - real benchmarks):                               │
│   from llm_uncertainty_analysis.data import (              │
│       load_lama_factual,        # LAMA TREx (1000 prompts) │
│       load_snli_balanced,       # SNLI (300 prompts)       │
│       load_gutenberg_poetry,    # Gutenberg (500 prompts)  │
│       load_all_datasets         # Load all at once         │
│   )                                                         │
│                                                             │
│ Analysis:                                                   │
│   from llm_uncertainty_analysis import UncertaintyAnalyzer │
│                                                             │
│ Metrics:                                                    │
│   from llm_uncertainty_analysis.metrics import (           │
│       calculate_entropy,                                    │
│       calculate_surprisal,                                  │
│       calculate_perplexity                                  │
│   )                                                         │
│                                                             │
│ Stats:                                                      │
│   from llm_uncertainty_analysis.statistics import (        │
│       calculate_cohens_d,                                   │
│       calculate_mutual_information,                         │
│       run_anova,                # One-way ANOVA            │
│       calculate_eta_squared,    # Effect size (η²)         │
│       run_tukey_hsd,            # Post-hoc test            │
│       run_bonferroni_correction # Multiple comparisons     │
│   )                                                         │
│                                                             │
│ Visualization:                                              │
│   from llm_uncertainty_analysis.visualization import (     │
│       plot_entropy_by_category,                             │
│       plot_icl_analysis                                     │
│   )                                                         │
│                                                             │
│ Experiments:                                                │
│   from llm_uncertainty_analysis.experiments import (       │
│       run_category_comparison,                              │
│       run_icl_experiment                                    │
│   )                                                         │
└─────────────────────────────────────────────────────────────┘
```

---

## Support

Need help? Check:
1. This quickstart guide
2. README.md - Full documentation
3. STRUCTURE.md - Project organization
4. Module docstrings - `help(function_name)`

---

**Happy analyzing! 🚀**
