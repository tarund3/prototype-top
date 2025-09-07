#!/usr/bin/env python3
"""
Experiment tracking and results management for TOP experiments.
Follows best practices for prototype result packaging.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time
from datetime import datetime
import os

# Set style for professional plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class ExperimentTracker:
    """Track and manage experiment results."""
    
    def __init__(self, results_dir="results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize metrics CSV if it doesn't exist
        self.metrics_file = self.results_dir / "metrics.csv"
        if not self.metrics_file.exists():
            self._init_metrics_file()
    
    def _init_metrics_file(self):
        """Initialize the metrics CSV file with proper columns."""
        columns = [
            'timestamp', 'experiment_id', 'method', 'seed',
            'n_layer', 'n_head', 'd_model', 'seq_len', 'batch_size',
            'learning_rate', 'max_epochs', 'lambda_top', 'window_size',
            'final_ppl', 'final_mrr', 'final_hit_at_1', 'final_hit_at_5', 'final_hit_at_10',
            'steps_to_ppl_50', 'total_steps', 'gpu_hours', 'peak_vram_gb',
            'convergence_epoch', 'best_val_loss', 'notes'
        ]
        
        df = pd.DataFrame(columns=columns)
        df.to_csv(self.metrics_file, index=False)
        print(f"Initialized metrics file: {self.metrics_file}")
    
    def log_experiment(self, config, results, metadata=None):
        """Log a completed experiment."""
        if metadata is None:
            metadata = {}
        
        # Create experiment record
        record = {
            'timestamp': datetime.now().isoformat(),
            'experiment_id': f"exp_{int(time.time())}",
            'method': '+'.join(config.get('objectives', ['ntp'])),
            'seed': config.get('seed', 42),
            'n_layer': config.get('n_layer', 4),
            'n_head': config.get('n_head', 8),
            'd_model': config.get('d_model', 256),
            'seq_len': config.get('seq_len', 512),
            'batch_size': config.get('batch_size', 8),
            'learning_rate': config.get('learning_rate', 1e-4),
            'max_epochs': config.get('max_epochs', 10),
            'lambda_top': config.get('lambda_top', 0.5),
            'window_size': config.get('window_size', 128),
            'final_ppl': results.get('perplexity', np.nan),
            'final_mrr': results.get('ranking', {}).get('mrr', np.nan),
            'final_hit_at_1': results.get('ranking', {}).get('hit_at_1', np.nan),
            'final_hit_at_5': results.get('ranking', {}).get('hit_at_5', np.nan),
            'final_hit_at_10': results.get('ranking', {}).get('hit_at_10', np.nan),
            'steps_to_ppl_50': metadata.get('steps_to_ppl_50', np.nan),
            'total_steps': metadata.get('total_steps', np.nan),
            'gpu_hours': metadata.get('gpu_hours', np.nan),
            'peak_vram_gb': metadata.get('peak_vram_gb', np.nan),
            'convergence_epoch': metadata.get('convergence_epoch', np.nan),
            'best_val_loss': metadata.get('best_val_loss', np.nan),
            'notes': metadata.get('notes', '')
        }
        
        # Append to CSV
        df = pd.read_csv(self.metrics_file)
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
        df.to_csv(self.metrics_file, index=False)
        
        print(f"Logged experiment: {record['experiment_id']} ({record['method']})")
        return record['experiment_id']
    
    def get_summary_table(self):
        """Generate a summary table of all experiments."""
        df = pd.read_csv(self.metrics_file)
        
        if df.empty:
            return "No experiments logged yet."
        
        # Group by method and compute statistics
        summary = df.groupby('method').agg({
            'final_ppl': ['mean', 'std', 'min'],
            'final_mrr': ['mean', 'std', 'max'],
            'steps_to_ppl_50': ['mean', 'std', 'min'],
            'gpu_hours': ['mean', 'std', 'sum']
        }).round(3)
        
        # Flatten column names
        summary.columns = ['_'.join(col).strip() for col in summary.columns]
        
        return summary
    
    def plot_learning_curves(self, save_path=None):
        """Plot learning curves for all experiments."""
        df = pd.read_csv(self.metrics_file)
        
        if df.empty:
            print("No data to plot")
            return
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Perplexity by method
        methods = df['method'].unique()
        colors = plt.cm.Set1(np.linspace(0, 1, len(methods)))
        
        for i, method in enumerate(methods):
            method_data = df[df['method'] == method]
            axes[0].scatter(method_data['total_steps'], method_data['final_ppl'], 
                           label=method.upper(), color=colors[i], s=100, alpha=0.7)
        
        axes[0].set_xlabel('Total Steps')
        axes[0].set_ylabel('Final Perplexity')
        axes[0].set_yscale('log')
        axes[0].set_title('Final Perplexity vs Training Steps')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: MRR by method (if available)
        mrr_data = df.dropna(subset=['final_mrr'])
        if not mrr_data.empty:
            for i, method in enumerate(mrr_data['method'].unique()):
                method_data = mrr_data[mrr_data['method'] == method]
                axes[1].scatter(method_data['total_steps'], method_data['final_mrr'], 
                               label=method.upper(), color=colors[i], s=100, alpha=0.7)
            
            axes[1].set_xlabel('Total Steps')
            axes[1].set_ylabel('Final MRR')
            axes[1].set_title('Final MRR vs Training Steps')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, 'No MRR data available', 
                        ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('MRR Results')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to: {save_path}")
        
        return fig
    
    def plot_method_comparison(self, save_path=None):
        """Plot method comparison bar chart."""
        df = pd.read_csv(self.metrics_file)
        
        if df.empty:
            print("No data to plot")
            return
        
        # Prepare data for comparison
        comparison_data = []
        for method in df['method'].unique():
            method_data = df[df['method'] == method]
            comparison_data.append({
                'Method': method.upper(),
                'Perplexity': method_data['final_ppl'].mean(),
                'MRR': method_data['final_mrr'].mean() if not method_data['final_mrr'].isna().all() else 0,
                'Steps to PPL<50': method_data['steps_to_ppl_50'].mean() if not method_data['steps_to_ppl_50'].isna().all() else 0,
                'GPU Hours': method_data['gpu_hours'].mean() if not method_data['gpu_hours'].isna().all() else 0
            })
        
        comp_df = pd.DataFrame(comparison_data)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Perplexity comparison
        axes[0, 0].bar(comp_df['Method'], comp_df['Perplexity'], color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Final Perplexity (Lower is Better)')
        axes[0, 0].set_ylabel('Perplexity')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # MRR comparison
        if comp_df['MRR'].sum() > 0:
            axes[0, 1].bar(comp_df['Method'], comp_df['MRR'], color='lightcoral', alpha=0.7)
            axes[0, 1].set_title('Final MRR (Higher is Better)')
            axes[0, 1].set_ylabel('MRR')
        else:
            axes[0, 1].text(0.5, 0.5, 'No MRR data', ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('MRR Results')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Steps to convergence
        if comp_df['Steps to PPL<50'].sum() > 0:
            axes[1, 0].bar(comp_df['Method'], comp_df['Steps to PPL<50'], color='lightgreen', alpha=0.7)
            axes[1, 0].set_title('Steps to PPL < 50 (Lower is Better)')
            axes[1, 0].set_ylabel('Steps')
        else:
            axes[1, 0].text(0.5, 0.5, 'No convergence data', ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Convergence Speed')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # GPU hours
        if comp_df['GPU Hours'].sum() > 0:
            axes[1, 1].bar(comp_df['Method'], comp_df['GPU Hours'], color='gold', alpha=0.7)
            axes[1, 1].set_title('GPU Hours (Lower is Better)')
            axes[1, 1].set_ylabel('Hours')
        else:
            axes[1, 1].text(0.5, 0.5, 'No timing data', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Computational Cost')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved comparison plot to: {save_path}")
        
        return fig
    
    def generate_report(self):
        """Generate a comprehensive results report."""
        df = pd.read_csv(self.metrics_file)
        
        if df.empty:
            return "No experiments completed yet."
        
        # Generate plots
        self.plot_learning_curves(self.results_dir / "fig_learning_curves.png")
        self.plot_method_comparison(self.results_dir / "fig_method_comparison.png")
        
        # Generate summary table
        summary = self.get_summary_table()
        
        # Create report
        report = f"""
# TOP Experiment Results Report

## Summary

This report contains results from {len(df)} experiments comparing different language modeling objectives.

## Key Findings

- **Best Method**: {df.loc[df['final_ppl'].idxmin(), 'method'].upper()} achieved lowest perplexity of {df['final_ppl'].min():.2f}
- **Fastest Convergence**: {df.loc[df['steps_to_ppl_50'].idxmin(), 'method'].upper()} reached PPL<50 in {df['steps_to_ppl_50'].min():.0f} steps
- **Most Efficient**: {df.loc[df['gpu_hours'].idxmin(), 'method'].upper()} used only {df['gpu_hours'].min():.1f} GPU hours

## Detailed Results

{summary.to_markdown()}

## Visualizations

### Learning Curves
![Learning Curves](fig_learning_curves.png)

### Method Comparison
![Method Comparison](fig_method_comparison.png)

## Hardware Details

- **GPU Type**: RTX 4090 / Colab T4
- **Peak VRAM**: {df['peak_vram_gb'].max():.1f} GB
- **Total GPU Hours**: {df['gpu_hours'].sum():.1f} hours

## Reproducibility

All experiments can be reproduced using the configurations in `configs/` and the code in this repository.
        """
        
        # Save report
        with open(self.results_dir / "report.md", "w") as f:
            f.write(report)
        
        print(f"Generated report: {self.results_dir / 'report.md'}")
        return report

def main():
    """Example usage of the experiment tracker."""
    tracker = ExperimentTracker()
    
    # Example: log a mock experiment
    config = {
        'objectives': ['ntp', 'top'],
        'n_layer': 4,
        'n_head': 8,
        'd_model': 256,
        'max_epochs': 5,
        'lambda_top': 0.5
    }
    
    results = {
        'perplexity': 45.2,
        'ranking': {
            'mrr': 0.15,
            'hit_at_1': 0.08,
            'hit_at_5': 0.25,
            'hit_at_10': 0.42
        }
    }
    
    metadata = {
        'steps_to_ppl_50': 5000,
        'total_steps': 10000,
        'gpu_hours': 2.1,
        'peak_vram_gb': 8.5,
        'convergence_epoch': 3,
        'best_val_loss': 3.81,
        'notes': 'Baseline experiment'
    }
    
    # Log the experiment
    exp_id = tracker.log_experiment(config, results, metadata)
    print(f"Logged experiment: {exp_id}")
    
    # Generate report
    tracker.generate_report()

if __name__ == "__main__":
    main()
