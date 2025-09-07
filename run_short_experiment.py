#!/usr/bin/env python3
"""
Short experiment to generate real results for the README
"""

import os
import sys
import torch
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from data.wikitext import WikiText2
from models.gpt_mini import GPTMini
from training.train_loop import TOPTrainer
from evaluation.eval_metrics import evaluate_model
from results.track_experiments import ExperimentTracker

def run_short_experiment():
    """Run a short experiment to generate real results"""
    
    print("🚀 SHORT TOP EXPERIMENT")
    print("=" * 50)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load smaller dataset for faster training
    print("📊 Loading WikiText-2 (smaller chunks)...")
    train_data = WikiText2(split="train", seq_len=128)  # Smaller sequences
    val_data = WikiText2(split="validation", seq_len=128)
    
    print(f"   Train batches: {len(train_data)}")
    print(f"   Val batches: {len(val_data)}")
    
    # Create model
    vocab_size = train_data.get_vocab_size()
    model = GPTMini(
        vocab_size=vocab_size,
        n_layer=2,  # Smaller model
        n_head=4,
        d_model=128,
        max_seq_len=128
    ).to(device)
    
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize experiment tracker
    tracker = ExperimentTracker("results/metrics.csv")
    
    # Experiment 1: NTP Baseline
    print("\n🔬 Experiment 1: NTP Baseline")
    print("-" * 30)
    
    trainer_ntp = TOPTrainer(
        model=model,
        train_data=train_data,
        val_data=val_data,
        objectives=["ntp"],
        learning_rate=1e-3,
        device=device,
        max_epochs=2,  # Very short training
        batch_size=32,
        eval_interval=500
    )
    
    start_time = time.time()
    trainer_ntp.train()
    ntp_time = time.time() - start_time
    
    # Evaluate NTP
    ntp_results = evaluate_model(
        model, val_data, device, 
        objectives=["ntp"], 
        batch_size=32
    )
    
    print(f"   NTP Results: PPL={ntp_results['ntp_perplexity']:.2f}")
    print(f"   Training time: {ntp_time:.1f}s")
    
    # Track NTP results
    tracker.log_experiment(
        experiment_id="ntp_baseline_short",
        config={
            "objectives": ["ntp"],
            "n_layer": 2,
            "n_head": 4,
            "d_model": 128,
            "seq_len": 128,
            "max_epochs": 2,
            "learning_rate": 1e-3
        },
        results={
            "ntp_perplexity": ntp_results['ntp_perplexity'],
            "training_time": ntp_time,
            "final_epoch": 2
        }
    )
    
    # Experiment 2: NTP + TOP
    print("\n🔬 Experiment 2: NTP + TOP")
    print("-" * 30)
    
    # Reset model for fair comparison
    model = GPTMini(
        vocab_size=vocab_size,
        n_layer=2,
        n_head=4,
        d_model=128,
        max_seq_len=128
    ).to(device)
    
    trainer_top = TOPTrainer(
        model=model,
        train_data=train_data,
        val_data=val_data,
        objectives=["ntp", "top"],
        learning_rate=1e-3,
        device=device,
        max_epochs=2,
        batch_size=32,
        eval_interval=500,
        top_weight=0.1  # Small weight for TOP
    )
    
    start_time = time.time()
    trainer_top.train()
    top_time = time.time() - start_time
    
    # Evaluate TOP
    top_results = evaluate_model(
        model, val_data, device,
        objectives=["ntp", "top"],
        batch_size=32
    )
    
    print(f"   NTP+TOP Results: PPL={top_results['ntp_perplexity']:.2f}, MRR={top_results['top_mrr']:.3f}")
    print(f"   Training time: {top_time:.1f}s")
    
    # Track TOP results
    tracker.log_experiment(
        experiment_id="ntp_top_short",
        config={
            "objectives": ["ntp", "top"],
            "n_layer": 2,
            "n_head": 4,
            "d_model": 128,
            "seq_len": 128,
            "max_epochs": 2,
            "learning_rate": 1e-3,
            "top_weight": 0.1
        },
        results={
            "ntp_perplexity": top_results['ntp_perplexity'],
            "top_mrr": top_results['top_mrr'],
            "top_hit_at_1": top_results['top_hit_at_1'],
            "top_hit_at_5": top_results['top_hit_at_5'],
            "training_time": top_time,
            "final_epoch": 2
        }
    )
    
    # Generate summary
    print("\n📊 EXPERIMENT SUMMARY")
    print("=" * 50)
    print(f"NTP Baseline:     PPL={ntp_results['ntp_perplexity']:.2f}, Time={ntp_time:.1f}s")
    print(f"NTP + TOP:        PPL={top_results['ntp_perplexity']:.2f}, MRR={top_results['top_mrr']:.3f}, Time={top_time:.1f}s")
    
    improvement = ((ntp_results['ntp_perplexity'] - top_results['ntp_perplexity']) / ntp_results['ntp_perplexity']) * 100
    print(f"TOP Improvement:  {improvement:+.1f}% perplexity reduction")
    
    print(f"\n✅ Results saved to: results/metrics.csv")
    print("🎉 Short experiment completed!")

if __name__ == "__main__":
    run_short_experiment()
