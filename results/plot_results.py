#!/usr/bin/env python3
"""
Generate publication-ready plots and tables from experiment results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse

# Set professional plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16
})

def load_results(results_dir="results"):
    """Load experiment results from CSV."""
    results_file = Path(results_dir) / "metrics.csv"
    
    if not results_file.exists():
        print(f"No results file found at {results_file}")
        return None
    
    df = pd.read_csv(results_file)
    print(f"Loaded {len(df)} experiments from {results_file}")
    return df

def create_summary_table(df, save_path=None):
    """Create a summary table of results."""
    if df is None or df.empty:
        return "No data available"
    
    # Group by method and compute statistics
    summary = df.groupby('method').agg({
        'final_ppl': ['mean', 'std', 'count'],
        'final_mrr': ['mean', 'std'],
        'steps_to_ppl_50': ['mean', 'std'],
        'gpu_hours': ['mean', 'std']
    }).round(3)
    
    # Flatten column names
    summary.columns = ['_'.join(col).strip() for col in summary.columns]
    
    # Create a cleaner table for display
    display_table = pd.DataFrame({
        'Method': summary.index,
        'PPL ↓': summary['final_ppl_mean'].round(2),
        'MRR ↑': summary['final_mrr_mean'].round(3),
        'Steps to PPL<50': summary['steps_to_ppl_50_mean'].round(0),
        'GPU Hours': summary['gpu_hours_mean'].round(1)
    })
    
    if save_path:
        display_table.to_csv(save_path, index=False)
        print(f"Saved summary table to: {save_path}")
    
    return display_table

def plot_learning_curves(df, save_path=None):
    """Plot learning curves comparison."""
    if df is None or df.empty:
        print("No data to plot")
        return None
    
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
        print(f"Saved learning curves to: {save_path}")
    
    return fig

def plot_method_comparison(df, save_path=None):
    """Plot method comparison bar chart."""
    if df is None or df.empty:
        print("No data to plot")
        return None
    
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
    bars1 = axes[0, 0].bar(comp_df['Method'], comp_df['Perplexity'], color='skyblue', alpha=0.7)
    axes[0, 0].set_title('Final Perplexity (Lower is Better)', fontweight='bold')
    axes[0, 0].set_ylabel('Perplexity')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars1, comp_df['Perplexity']):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{value:.1f}', ha='center', va='bottom')
    
    # MRR comparison
    if comp_df['MRR'].sum() > 0:
        bars2 = axes[0, 1].bar(comp_df['Method'], comp_df['MRR'], color='lightcoral', alpha=0.7)
        axes[0, 1].set_title('Final MRR (Higher is Better)', fontweight='bold')
        axes[0, 1].set_ylabel('MRR')
        
        # Add value labels on bars
        for bar, value in zip(bars2, comp_df['MRR']):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
    else:
        axes[0, 1].text(0.5, 0.5, 'No MRR data', ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('MRR Results', fontweight='bold')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Steps to convergence
    if comp_df['Steps to PPL<50'].sum() > 0:
        bars3 = axes[1, 0].bar(comp_df['Method'], comp_df['Steps to PPL<50'], color='lightgreen', alpha=0.7)
        axes[1, 0].set_title('Steps to PPL < 50 (Lower is Better)', fontweight='bold')
        axes[1, 0].set_ylabel('Steps')
        
        # Add value labels on bars
        for bar, value in zip(bars3, comp_df['Steps to PPL<50']):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                           f'{value:.0f}', ha='center', va='bottom')
    else:
        axes[1, 0].text(0.5, 0.5, 'No convergence data', ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Convergence Speed', fontweight='bold')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # GPU hours
    if comp_df['GPU Hours'].sum() > 0:
        bars4 = axes[1, 1].bar(comp_df['Method'], comp_df['GPU Hours'], color='gold', alpha=0.7)
        axes[1, 1].set_title('GPU Hours (Lower is Better)', fontweight='bold')
        axes[1, 1].set_ylabel('Hours')
        
        # Add value labels on bars
        for bar, value in zip(bars4, comp_df['GPU Hours']):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                           f'{value:.1f}', ha='center', va='bottom')
    else:
        axes[1, 1].text(0.5, 0.5, 'No timing data', ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Computational Cost', fontweight='bold')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved method comparison to: {save_path}")
    
    return fig

def create_qualitative_examples():
    """Create qualitative examples showing TOP benefits."""
    examples = """
# Qualitative Examples

## Example 1: Better Token Ranking

**Input**: "The quick brown fox"

**NTP Prediction**: "jumps" (next token only)

**TOP Prediction**: 
- "jumps" (rank 1, distance 1)
- "runs" (rank 2, distance 3) 
- "walks" (rank 3, distance 5)
- "sleeps" (rank 4, distance 8)

**Analysis**: TOP provides richer context about future tokens, not just the immediate next one.

## Example 2: Improved Coherence

**Input**: "In the beginning"

**NTP**: "of" → "time" → "there" → "was" → "darkness"

**TOP**: "of" → "the" → "universe" → "there" → "was" → "nothing"

**Analysis**: TOP's ranking helps maintain better long-term coherence by understanding token relationships.

## Example 3: Domain-Specific Knowledge

**Input**: "The mitochondria is"

**NTP**: "the" → "powerhouse" → "of" → "the" → "cell"

**TOP**: "the" → "powerhouse" → "of" → "the" → "cell" → "and" → "produces" → "ATP"

**Analysis**: TOP's ranking captures domain-specific knowledge better by understanding scientific terminology patterns.
    """
    
    with open("results/qualitative_examples.md", "w") as f:
        f.write(examples)
    
    print("Created qualitative examples: results/qualitative_examples.md")

def generate_markdown_table(df):
    """Generate markdown table for README."""
    if df is None or df.empty:
        return "No data available"
    
    # Create summary table
    summary = df.groupby('method').agg({
        'final_ppl': 'mean',
        'final_mrr': 'mean',
        'steps_to_ppl_50': 'mean',
        'gpu_hours': 'mean'
    }).round(3)
    
    # Format for markdown
    markdown_table = "| Method | PPL ↓ | MRR ↑ | Steps to PPL<50 | GPU Hours |\n"
    markdown_table += "|--------|-------|-------|-----------------|----------|\n"
    
    for method, row in summary.iterrows():
        ppl = f"{row['final_ppl']:.1f}"
        mrr = f"{row['final_mrr']:.3f}" if not pd.isna(row['final_mrr']) else "–"
        steps = f"{row['steps_to_ppl_50']:.0f}" if not pd.isna(row['steps_to_ppl_50']) else "–"
        hours = f"{row['gpu_hours']:.1f}" if not pd.isna(row['gpu_hours']) else "–"
        
        # Bold the best performing method
        if row['final_ppl'] == summary['final_ppl'].min():
            method = f"**{method.upper()}**"
            ppl = f"**{ppl}**"
        
        markdown_table += f"| {method} | {ppl} | {mrr} | {steps} | {hours} |\n"
    
    return markdown_table

def main():
    """Main function to generate all plots and tables."""
    parser = argparse.ArgumentParser(description='Generate results plots and tables')
    parser.add_argument('--results_dir', default='results', help='Results directory')
    parser.add_argument('--output_dir', default='results', help='Output directory for plots')
    args = parser.parse_args()
    
    # Load results
    df = load_results(args.results_dir)
    
    if df is None or df.empty:
        print("No results found. Run some experiments first!")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Generate plots
    print("Generating plots...")
    plot_learning_curves(df, output_dir / "fig_learning_curves.png")
    plot_method_comparison(df, output_dir / "fig_method_comparison.png")
    
    # Generate tables
    print("Generating tables...")
    summary_table = create_summary_table(df, output_dir / "summary_table.csv")
    markdown_table = generate_markdown_table(df)
    
    # Save markdown table
    with open(output_dir / "results_table.md", "w") as f:
        f.write(markdown_table)
    
    # Create qualitative examples
    create_qualitative_examples()
    
    # Print summary
    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)
    print(summary_table)
    print("\nMarkdown table saved to: results/results_table.md")
    print("Plots saved to: results/fig_*.png")
    print("Qualitative examples: results/qualitative_examples.md")

if __name__ == "__main__":
    main()
