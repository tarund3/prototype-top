#!/usr/bin/env python3
"""
Test script to verify the TOP implementation works correctly.
"""

import torch
import sys
import os

# Add current directory to path
sys.path.append('.')

from data.wikitext import WikiText2
from models.gpt_mini import GPTMini
from losses.ranking import listnet_loss, compute_ranking_metrics
from training.train_loop import TOPTrainer


def test_dataset():
    """Test dataset loading."""
    print("Testing dataset loading...")
    
    try:
        # Create a small dataset
        dataset = WikiText2(split='train', seq_len=64, tokenizer='gpt2')
        print(f"✓ Dataset loaded successfully")
        print(f"  - Dataset size: {len(dataset)}")
        print(f"  - Vocabulary size: {dataset.get_vocab_size()}")
        
        # Test getting a sample
        sample = dataset[0]
        print(f"  - Sample input shape: {sample['input_ids'].shape}")
        print(f"  - Sample labels shape: {sample['ntp_labels'].shape}")
        
        return True
    except Exception as e:
        print(f"✗ Dataset test failed: {e}")
        return False


def test_model():
    """Test model creation and forward pass."""
    print("\nTesting model...")
    
    try:
        # Create model
        model = GPTMini(vocab_size=50257, n_layer=2, n_head=4, d_model=128)
        print(f"✓ Model created successfully")
        print(f"  - Parameters: {model.get_num_params():,}")
        
        # Test forward pass
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        
        outputs = model(input_ids)
        print(f"  - NTP logits shape: {outputs['ntp_logits'].shape}")
        print(f"  - MTP logits shape: {outputs['mtp_logits'].shape}")
        print(f"  - TOP logits shape: {outputs['top_logits'].shape}")
        print(f"  - Hidden states shape: {outputs['hidden'].shape}")
        
        return True
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        return False


def test_losses():
    """Test loss computation."""
    print("\nTesting losses...")
    
    try:
        batch_size, seq_len, vocab_size = 2, 16, 100
        
        # Create dummy data
        scores = torch.randn(batch_size, seq_len, vocab_size)
        targets = torch.full((batch_size, seq_len, vocab_size), float('-inf'))
        
        # Set some valid targets
        targets[0, 0, 5] = 10.0
        targets[0, 1, 3] = 8.0
        targets[1, 0, 7] = 9.0
        
        # Test ListNet loss
        loss = listnet_loss(scores, targets)
        print(f"✓ ListNet loss computed: {loss.item():.4f}")
        
        # Test ranking metrics
        metrics = compute_ranking_metrics(scores, targets)
        print(f"✓ Ranking metrics computed:")
        print(f"  - MRR: {metrics['mrr']:.4f}")
        print(f"  - Hit@1: {metrics['hit_at_1']:.4f}")
        print(f"  - Hit@5: {metrics['hit_at_5']:.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Loss test failed: {e}")
        return False


def test_training_step():
    """Test a single training step."""
    print("\nTesting training step...")
    
    try:
        # Create small model and data
        model = GPTMini(vocab_size=1000, n_layer=2, n_head=4, d_model=64)
        
        # Create dummy batch
        batch = {
            'input_ids': torch.randint(0, 1000, (2, 32)),
            'ntp_labels': torch.randint(0, 1000, (2, 32))
        }
        
        # Create trainer
        config = {
            'objectives': ['ntp', 'top'],
            'lambda_ntp': 1.0,
            'lambda_top': 0.5,
            'window_size': 16,
            'output_dir': 'test_outputs'
        }
        
        # Create dummy data loaders
        from torch.utils.data import DataLoader, TensorDataset
        dummy_dataset = TensorDataset(
            torch.randint(0, 1000, (100, 32)),
            torch.randint(0, 1000, (100, 32))
        )
        train_loader = DataLoader(dummy_dataset, batch_size=2, shuffle=True)
        val_loader = DataLoader(dummy_dataset, batch_size=2, shuffle=False)
        
        trainer = TOPTrainer(model, train_loader, val_loader, config)
        
        # Test loss computation
        outputs = model(batch['input_ids'])
        loss, losses = trainer.compute_loss(batch, outputs)
        
        print(f"✓ Training step completed")
        print(f"  - Total loss: {loss.item():.4f}")
        print(f"  - Loss components: {losses}")
        
        return True
    except Exception as e:
        print(f"✗ Training step test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("TOP IMPLEMENTATION TEST")
    print("="*60)
    
    tests = [
        test_dataset,
        test_model,
        test_losses,
        test_training_step
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("="*60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("="*60)
    
    if passed == total:
        print("🎉 All tests passed! Implementation is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
