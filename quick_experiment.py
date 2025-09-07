#!/usr/bin/env python3
"""
Quick experiment script to demonstrate TOP vs NTP.
This runs a short experiment to show the difference between objectives.
"""

import os
import torch
import sys
from pathlib import Path

# Add current directory to path
sys.path.append('.')

from data.wikitext import create_dataloader
from models.gpt_mini import GPTMini
from training.train_loop import create_trainer
from evaluation.eval_metrics import evaluate_model


def run_quick_experiment():
    """Run a quick experiment comparing NTP vs NTP+TOP."""
    
    # Set environment variables
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    print("🚀 QUICK TOP EXPERIMENT")
    print("="*50)
    
    # Configuration
    config = {
        'n_layer': 2,
        'n_head': 4,
        'd_model': 128,
        'seq_len': 64,
        'batch_size': 4,
        'max_epochs': 3,
        'learning_rate': 1e-3,
        'vocab_size': 50257,
        'tokenizer': 'gpt2',
        'output_dir': 'quick_outputs'
    }
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Create data loaders
    print("📊 Loading WikiText-2...")
    train_loader = create_dataloader(
        split='train',
        seq_len=config['seq_len'],
        batch_size=config['batch_size'],
        shuffle=True,
        tokenizer=config['tokenizer']
    )
    
    val_loader = create_dataloader(
        split='validation',
        seq_len=config['seq_len'],
        batch_size=config['batch_size'],
        shuffle=False,
        tokenizer=config['tokenizer']
    )
    
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    
    # Experiment 1: NTP only
    print("\n🔬 Experiment 1: NTP Baseline")
    print("-" * 30)
    
    model_ntp = GPTMini(
        vocab_size=config['vocab_size'],
        n_layer=config['n_layer'],
        n_head=config['n_head'],
        d_model=config['d_model']
    )
    
    config_ntp = config.copy()
    config_ntp['objectives'] = ['ntp']
    config_ntp['output_dir'] = os.path.join(config['output_dir'], 'ntp_only')
    
    trainer_ntp = create_trainer(model_ntp, train_loader, val_loader, config_ntp)
    
    print(f"   Model parameters: {model_ntp.get_num_params():,}")
    print("   Training...")
    trainer_ntp.train()
    
    # Evaluate NTP model
    print("   Evaluating...")
    results_ntp = evaluate_model(model_ntp, val_loader, config_ntp)
    
    # Experiment 2: NTP + TOP
    print("\n🔬 Experiment 2: NTP + TOP")
    print("-" * 30)
    
    model_top = GPTMini(
        vocab_size=config['vocab_size'],
        n_layer=config['n_layer'],
        n_head=config['n_head'],
        d_model=config['d_model']
    )
    
    config_top = config.copy()
    config_top['objectives'] = ['ntp', 'top']
    config_top['lambda_top'] = 0.5
    config_top['window_size'] = 32
    config_top['output_dir'] = os.path.join(config['output_dir'], 'ntp_top')
    
    trainer_top = create_trainer(model_top, train_loader, val_loader, config_top)
    
    print(f"   Model parameters: {model_top.get_num_params():,}")
    print("   Training...")
    trainer_top.train()
    
    # Evaluate TOP model
    print("   Evaluating...")
    results_top = evaluate_model(model_top, val_loader, config_top)
    
    # Compare results
    print("\n📈 RESULTS COMPARISON")
    print("="*50)
    print(f"{'Metric':<20} {'NTP Only':<12} {'NTP + TOP':<12} {'Improvement':<12}")
    print("-" * 60)
    
    ppl_ntp = results_ntp['perplexity']
    ppl_top = results_top['perplexity']
    improvement = ((ppl_ntp - ppl_top) / ppl_ntp) * 100
    
    print(f"{'Perplexity':<20} {ppl_ntp:<12.4f} {ppl_top:<12.4f} {improvement:>+10.2f}%")
    
    if 'ranking' in results_top:
        mrr = results_top['ranking']['mrr']
        print(f"{'MRR (TOP)':<20} {'N/A':<12} {mrr:<12.4f} {'N/A':<12}")
    
    print("\n🎯 CONCLUSION")
    print("="*50)
    if ppl_top < ppl_ntp:
        print("✅ TOP improves language modeling!")
        print(f"   Perplexity reduced by {improvement:.2f}%")
    else:
        print("❌ TOP did not improve language modeling in this quick test.")
        print("   This might be due to the short training time or small model size.")
    
    print(f"\n📁 Results saved to: {config['output_dir']}")
    print("\n💡 For better results, try:")
    print("   - More training epochs (--max_epochs 10)")
    print("   - Larger model (--n_layer 4 --d_model 256)")
    print("   - Longer sequences (--seq_len 512)")
    print("   - Different TOP weights (--lambda_top 0.3)")


if __name__ == '__main__':
    run_quick_experiment()
