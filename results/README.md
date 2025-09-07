# Results Packaging

This directory contains the results tracking and visualization system for TOP experiments, following best practices for prototype result packaging.

## Structure

```
results/
├── track_experiments.py     # Experiment tracking system
├── plot_results.py          # Generate plots and tables
├── metrics.csv              # Experiment results (auto-generated)
├── fig_learning_curves.png  # Learning curves plot
├── fig_method_comparison.png # Method comparison plot
├── summary_table.csv        # Summary statistics
├── results_table.md         # Markdown table for README
├── qualitative_examples.md  # Qualitative examples
└── report.md                # Comprehensive report
```

## Usage

### Track Experiments

```python
from results.track_experiments import ExperimentTracker

tracker = ExperimentTracker()

# Log an experiment
config = {...}  # Your experiment config
results = {...}  # Your results
metadata = {...}  # Additional metadata

exp_id = tracker.log_experiment(config, results, metadata)
```

### Generate Plots and Tables

```bash
# Generate all plots and tables
python results/plot_results.py

# Or from the main directory
python run_experiments.py --all
```

### View Results

```python
# Load and analyze results
import pandas as pd
df = pd.read_csv('results/metrics.csv')
print(df.groupby('method')['final_ppl'].mean())
```

## Metrics Tracked

- **Model Performance**: Perplexity, MRR, Hit@K
- **Training Efficiency**: Steps to convergence, GPU hours
- **Resource Usage**: Peak VRAM, total compute time
- **Configuration**: All hyperparameters and settings

## Best Practices

1. **Version Control**: Keep `metrics.csv` and plotting scripts in version control
2. **Large Files**: Use `.gitignore` for large checkpoints and logs
3. **Reproducibility**: Include random seeds and exact configurations
4. **Documentation**: Update README with new results
5. **Visualization**: Generate publication-ready plots

## Integration with Streamlit

The results are automatically integrated with the Streamlit dashboard:

- View experiment history
- Compare different methods
- Generate new plots
- Export results

## Example Workflow

1. Run experiments: `python run_experiments.py --all`
2. Generate plots: `python results/plot_results.py`
3. Update README: Copy markdown table from `results/results_table.md`
4. View dashboard: `python run_streamlit.py`
