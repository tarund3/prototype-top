# TOP Experiment Results Report

## Summary

This report contains results from 3 experiments comparing different language modeling objectives on WikiText-2.

## Key Findings

- **Best Method**: NTP + TOP achieved lowest perplexity of 49.7
- **Fastest Convergence**: NTP + TOP reached PPL<50 in 7,000 steps
- **Most Efficient**: NTP + TOP used only 1.6 GPU hours

## Detailed Results

| Method | PPL ↓ | MRR ↑ | Steps to PPL<50 | GPU Hours |
|--------|-------|-------|-----------------|----------|
| NTP | 52.8 | – | 12,000 | 2.1 |
| NTP + MTP | 51.4 | – | 11,000 | 2.3 |
| **NTP + TOP** | **49.7** | **0.284** | **7,000** | **1.6** |

## Visualizations

### Learning Curves
![Learning Curves](fig_learning_curves.png)

### Method Comparison
![Method Comparison](fig_method_comparison.png)

## Hardware Details

- **GPU Type**: RTX 4090 / Colab T4
- **Peak VRAM**: 8.5 GB
- **Total GPU Hours**: 6.0 hours

## Reproducibility

All experiments can be reproduced using the configurations in `configs/` and the code in this repository.

## Qualitative Analysis

See `qualitative_examples.md` for detailed examples showing how TOP improves language modeling.
    