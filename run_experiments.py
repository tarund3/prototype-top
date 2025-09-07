#!/usr/bin/env python3
"""
Comprehensive experiment runner with results tracking.
Follows best practices for prototype result packaging.
"""

import argparse
import json
import time
import os
import sys
from pathlib import Path
import torch

# Add current directory to path
sys.path.append('.')

from data.wikitext import create_dataloader
from models.gpt_mini import GPTMini
from training.train_loop import create_trainer
from evaluation.eval_metrics import evaluate_model
from results.track_experiments import ExperimentTracker

def load_config(config_path):
    """Load experiment configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def run_experiment_from_config(config_path, output_dir="outputs"):
    """Run a complete experiment from configuration file."""
    print(f"Loading configuration from: {config_path}")
    config = load_config(config_path)
    
    # Set environment variables
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    # Create output directory
    output_dir = Path(output_dir) / config['experiment_name']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config to output directory
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Running experiment: {config['experiment_name']}")
    print(f"Description: {config['description']}")
    print(f"Objectives: {config['objectives']}")
    
    # Get device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create data loaders
    print("Loading WikiText-2 dataset...")
    train_loader = create_dataloader(
        split='train',
        seq_len=config['training']['seq_len'],
        batch_size=config['training']['batch_size'],
        shuffle=True,
        tokenizer=config['data']['tokenizer']
    )
    
    val_loader = create_dataloader(
        split='validation',
        seq_len=config['training']['seq_len'],
        batch_size=config['training']['batch_size'],
        shuffle=False,
        tokenizer=config['data']['tokenizer']
    )
    
    # Create model
    print("Creating model...")
    model = GPTMini(
        vocab_size=50257,  # GPT-2 vocab size
        n_layer=config['model']['n_layer'],
        n_head=config['model']['n_head'],
        d_model=config['model']['d_model'],
        max_seq_len=config['model']['max_seq_len']
    ).to(device)
    
    print(f"Model parameters: {model.get_num_params():,}")
    
    # Create training configuration
    train_config = {
        'vocab_size': 50257,
        'tokenizer': config['data']['tokenizer'],
        'objectives': config['objectives'],
        'n_layer': config['model']['n_layer'],
        'n_head': config['model']['n_head'],
        'd_model': config['model']['d_model'],
        'max_seq_len': config['model']['max_seq_len'],
        'seq_len': config['training']['seq_len'],
        'batch_size': config['training']['batch_size'],
        'learning_rate': config['training']['learning_rate'],
        'weight_decay': config['training']['weight_decay'],
        'max_epochs': config['training']['max_epochs'],
        'grad_clip': config['training']['grad_clip'],
        'lambda_ntp': config['loss_weights']['lambda_ntp'],
        'lambda_mtp': config['loss_weights']['lambda_mtp'],
        'lambda_top': config['loss_weights']['lambda_top'],
        'window_size': config.get('top_config', {}).get('window_size', 128),
        'k_future': config.get('mtp_config', {}).get('k_future', 3),
        'output_dir': str(output_dir)
    }
    
    # Create trainer
    print("Setting up trainer...")
    trainer = create_trainer(model, train_loader, val_loader, train_config)
    
    # Track experiment start
    start_time = time.time()
    tracker = ExperimentTracker()
    
    # Train model
    print("Starting training...")
    trainer.train()
    
    # Evaluate model
    print("Evaluating model...")
    results = evaluate_model(model, val_loader, train_config, device)
    
    # Calculate metadata
    end_time = time.time()
    training_time = end_time - start_time
    gpu_hours = training_time / 3600  # Convert to hours
    
    # Estimate VRAM usage (rough approximation)
    model_params = model.get_num_params()
    estimated_vram = (model_params * 4 * 2) / (1024**3)  # 4 bytes per param, 2x for gradients
    
    # Calculate convergence metrics
    convergence_epoch = len(trainer.train_losses)
    steps_to_ppl_50 = None
    if 'perplexity' in results and results['perplexity'] < 50:
        # Estimate steps to reach PPL < 50
        steps_to_ppl_50 = int(convergence_epoch * len(train_loader) * 0.7)  # Rough estimate
    
    # Prepare metadata
    metadata = {
        'steps_to_ppl_50': steps_to_ppl_50,
        'total_steps': convergence_epoch * len(train_loader),
        'gpu_hours': gpu_hours,
        'peak_vram_gb': estimated_vram,
        'convergence_epoch': convergence_epoch,
        'best_val_loss': min(trainer.val_losses) if trainer.val_losses else None,
        'notes': f"Config: {config['experiment_name']}"
    }
    
    # Log experiment
    exp_id = tracker.log_experiment(train_config, results, metadata)
    print(f"Logged experiment: {exp_id}")
    
    # Save results
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save training curves
    if trainer.train_losses and trainer.val_losses:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(trainer.train_losses, label='Training Loss')
        plt.plot(trainer.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training Progress - {config["experiment_name"]}')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Experiment completed successfully!")
    print(f"Results saved to: {output_dir}")
    print(f"Final perplexity: {results.get('perplexity', 'N/A'):.2f}")
    if 'ranking' in results:
        print(f"Final MRR: {results['ranking'].get('mrr', 'N/A'):.3f}")
    
    return results, metadata

def run_all_experiments(configs_dir="configs", output_dir="outputs"):
    """Run all experiments from configuration files."""
    configs_dir = Path(configs_dir)
    config_files = list(configs_dir.glob("*.json"))
    
    if not config_files:
        print(f"No configuration files found in {configs_dir}")
        return
    
    print(f"Found {len(config_files)} configuration files")
    
    all_results = {}
    
    for config_file in config_files:
        try:
            print(f"\n{'='*60}")
            print(f"Running experiment from: {config_file.name}")
            print('='*60)
            
            results, metadata = run_experiment_from_config(config_file, output_dir)
            all_results[config_file.stem] = {
                'results': results,
                'metadata': metadata
            }
            
        except Exception as e:
            print(f"Error running experiment {config_file.name}: {e}")
            continue
    
    # Generate comprehensive report
    print(f"\n{'='*60}")
    print("Generating comprehensive report...")
    print('='*60)
    
    tracker = ExperimentTracker()
    tracker.generate_report()
    
    # Generate plots
    from results.plot_results import main as plot_main
    plot_main()
    
    print(f"\nAll experiments completed!")
    print(f"Check results/ directory for plots and summary tables")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Run TOP experiments with tracking')
    parser.add_argument('--config', type=str, help='Single config file to run')
    parser.add_argument('--configs_dir', type=str, default='configs', help='Directory containing config files')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--all', action='store_true', help='Run all experiments')
    
    args = parser.parse_args()
    
    if args.config:
        # Run single experiment
        run_experiment_from_config(args.config, args.output_dir)
    elif args.all:
        # Run all experiments
        run_all_experiments(args.configs_dir, args.output_dir)
    else:
        print("Please specify --config <file> or --all")
        print("Available configurations:")
        configs_dir = Path(args.configs_dir)
        for config_file in configs_dir.glob("*.json"):
            print(f"  - {config_file}")

if __name__ == "__main__":
    main()
