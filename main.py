#!/usr/bin/env python3
"""
Token Order Prediction (TOP) Experiment
Main entry point for training and evaluating language models with different objectives.
"""

import argparse
import json
import os
import torch
import random
import numpy as np
from pathlib import Path

from data.wikitext import create_dataloader
from models.gpt_mini import GPTMini
from training.train_loop import create_trainer
from evaluation.eval_metrics import evaluate_model, compare_models, plot_training_curves


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def create_config(args):
    """Create configuration dictionary from arguments."""
    config = {
        # Model configuration
        'vocab_size': 50257,  # GPT-2 vocab size
        'n_layer': args.n_layer,
        'n_head': args.n_head,
        'd_model': args.d_model,
        'max_seq_len': args.max_seq_len,
        
        # Training configuration
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'max_epochs': args.max_epochs,
        'grad_clip': args.grad_clip,
        
        # Data configuration
        'seq_len': args.seq_len,
        'tokenizer': args.tokenizer,
        
        # Objective configuration
        'objectives': args.objectives,
        'lambda_ntp': args.lambda_ntp,
        'lambda_mtp': args.lambda_mtp,
        'lambda_top': args.lambda_top,
        
        # TOP-specific configuration
        'window_size': args.window_size,
        'k_future': args.k_future,
        'top_temperature': args.top_temperature,
        
        # Output configuration
        'output_dir': args.output_dir,
        'save_every': args.save_every,
        'seed': args.seed,
    }
    
    return config


def train_model(args):
    """Train a model with the specified configuration."""
    print("="*80)
    print("TOKEN ORDER PREDICTION (TOP) EXPERIMENT")
    print("="*80)
    
    # Set random seed
    set_seed(args.seed)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create configuration
    config = create_config(args)
    
    # Save configuration
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create data loaders
    print("Loading WikiText-2 dataset...")
    train_loader = create_dataloader(
        split='train',
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        shuffle=True,
        tokenizer=args.tokenizer
    )
    
    val_loader = create_dataloader(
        split='validation',
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        shuffle=False,
        tokenizer=args.tokenizer
    )
    
    # Get vocabulary size from tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    config['vocab_size'] = len(tokenizer)
    
    # Create model
    print(f"Creating model with {args.n_layer} layers, {args.n_head} heads, d_model={args.d_model}")
    model = GPTMini(
        vocab_size=config['vocab_size'],
        n_layer=args.n_layer,
        n_head=args.n_head,
        d_model=args.d_model,
        max_seq_len=args.max_seq_len
    ).to(device)
    
    print(f"Model parameters: {model.get_num_params():,}")
    
    # Create trainer
    trainer = create_trainer(model, train_loader, val_loader, config)
    
    # Train model
    print(f"\nTraining with objectives: {args.objectives}")
    trainer.train()
    
    # Final evaluation
    print("\nFinal evaluation...")
    results = evaluate_model(model, val_loader, config, device)
    
    # Save results
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTraining completed! Results saved to {args.output_dir}")
    return model, results


def evaluate_model_from_checkpoint(args):
    """Evaluate a model from checkpoint."""
    print("Evaluating model from checkpoint...")
    
    # Set random seed
    set_seed(args.seed)
    
    # Get device
    device = get_device()
    
    # Load configuration
    config_path = os.path.join(args.checkpoint_dir, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Create data loader
    val_loader = create_dataloader(
        split='validation',
        seq_len=config['seq_len'],
        batch_size=args.batch_size,
        shuffle=False,
        tokenizer=config['tokenizer']
    )
    
    # Create model
    model = GPTMini(
        vocab_size=config['vocab_size'],
        n_layer=config['n_layer'],
        n_head=config['n_head'],
        d_model=config['d_model'],
        max_seq_len=config['max_seq_len']
    ).to(device)
    
    # Load checkpoint
    checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate
    results = evaluate_model(model, val_loader, config, device)
    
    print("Evaluation Results:")
    print(f"Perplexity: {results['perplexity']:.4f}")
    if 'ranking' in results:
        print(f"MRR: {results['ranking']['mrr']:.4f}")
    
    return results


def compare_experiments(args):
    """Compare multiple experiments."""
    print("Comparing experiments...")
    
    # Load all experiment results
    results = {}
    for exp_dir in args.experiment_dirs:
        exp_name = os.path.basename(exp_dir)
        
        # Load config
        config_path = os.path.join(exp_dir, 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Load results
        results_path = os.path.join(exp_dir, 'results.json')
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                results[exp_name] = json.load(f)
        else:
            print(f"Warning: No results found for {exp_name}")
    
    # Print comparison
    print("\n" + "="*80)
    print("EXPERIMENT COMPARISON")
    print("="*80)
    
    print("\nPerplexity:")
    for name, result in results.items():
        print(f"  {name:20s}: {result['perplexity']:.4f}")
    
    if any('ranking' in result for result in results.values()):
        print("\nRanking Metrics:")
        print(f"{'Experiment':<20s} {'MRR':<8s} {'Hit@1':<8s} {'Hit@5':<8s} {'Hit@10':<8s}")
        print("-" * 60)
        
        for name, result in results.items():
            if 'ranking' in result:
                r = result['ranking']
                print(f"{name:<20s} {r['mrr']:<8.4f} {r['hit_at_1']:<8.4f} "
                      f"{r['hit_at_5']:<8.4f} {r['hit_at_10']:<8.4f}")


def main():
    parser = argparse.ArgumentParser(description='Token Order Prediction Experiment')
    
    # Model arguments
    parser.add_argument('--n_layer', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--n_head', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--d_model', type=int, default=256, help='Model dimension')
    parser.add_argument('--max_seq_len', type=int, default=1024, help='Maximum sequence length')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--max_epochs', type=int, default=10, help='Maximum epochs')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping')
    
    # Data arguments
    parser.add_argument('--seq_len', type=int, default=512, help='Sequence length')
    parser.add_argument('--tokenizer', type=str, default='gpt2', help='Tokenizer name')
    
    # Objective arguments
    parser.add_argument('--objectives', nargs='+', default=['ntp'], 
                       choices=['ntp', 'mtp', 'top'], help='Training objectives')
    parser.add_argument('--lambda_ntp', type=float, default=1.0, help='NTP loss weight')
    parser.add_argument('--lambda_mtp', type=float, default=0.5, help='MTP loss weight')
    parser.add_argument('--lambda_top', type=float, default=0.5, help='TOP loss weight')
    
    # TOP-specific arguments
    parser.add_argument('--window_size', type=int, default=128, help='TOP ranking window size')
    parser.add_argument('--k_future', type=int, default=3, help='MTP future tokens')
    parser.add_argument('--top_temperature', type=float, default=1.0, help='TOP temperature')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--save_every', type=int, default=5, help='Save every N epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Mode arguments
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'eval', 'compare'], help='Mode')
    parser.add_argument('--checkpoint_dir', type=str, help='Checkpoint directory for evaluation')
    parser.add_argument('--experiment_dirs', nargs='+', help='Experiment directories for comparison')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_model(args)
    elif args.mode == 'eval':
        if not args.checkpoint_dir:
            print("Error: --checkpoint_dir required for evaluation mode")
            return
        evaluate_model_from_checkpoint(args)
    elif args.mode == 'compare':
        if not args.experiment_dirs:
            print("Error: --experiment_dirs required for comparison mode")
            return
        compare_experiments(args)


if __name__ == '__main__':
    main()
