import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import math


def compute_perplexity(model, dataloader, device='cuda'):
    """Compute perplexity on a dataset."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # Move to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            # Forward pass
            outputs = model(batch['input_ids'])
            
            # Compute NTP loss
            ntp_loss = F.cross_entropy(
                outputs['ntp_logits'].view(-1, outputs['ntp_logits'].size(-1)),
                batch['ntp_labels'].view(-1),
                ignore_index=-100,
                reduction='sum'
            )
            
            # Count valid tokens
            valid_tokens = (batch['ntp_labels'] != -100).sum().item()
            
            total_loss += ntp_loss.item()
            total_tokens += valid_tokens
    
    # Perplexity = exp(average loss)
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    return perplexity, avg_loss


def compute_ranking_metrics(model, dataloader, window_size=128, device='cuda'):
    """Compute ranking metrics for TOP evaluation."""
    model.eval()
    
    mrr_scores = []
    hit_at_k = defaultdict(list)
    k_values = [1, 5, 10, 20]
    
    with torch.no_grad():
        for batch in dataloader:
            # Move to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            # Forward pass
            outputs = model(batch['input_ids'])
            
            # Create proximity targets
            proximity_targets = create_proximity_targets(
                batch['input_ids'], 
                batch['ntp_labels'], 
                window_size,
                model.vocab_size
            )
            
            # Compute metrics for each position
            batch_size, seq_len = batch['input_ids'].shape
            
            for b in range(batch_size):
                for s in range(seq_len):
                    true_scores = proximity_targets[b, s]
                    pred_scores = outputs['top_logits'][b, s]
                    
                    # Get valid items (non-negative proximity scores)
                    valid_items = (true_scores != float('-inf')).nonzero(as_tuple=True)[0]
                    
                    if len(valid_items) == 0:
                        continue
                    
                    # Find the highest-scoring true item
                    best_item = valid_items[true_scores[valid_items].argmax()]
                    
                    # Get prediction ranking
                    pred_ranking = pred_scores.argsort(descending=True)
                    
                    # Compute MRR
                    rank = (pred_ranking == best_item).nonzero(as_tuple=True)[0]
                    if len(rank) > 0:
                        mrr_scores.append(1.0 / (rank[0].item() + 1))
                    else:
                        mrr_scores.append(0.0)
                    
                    # Compute Hit@K
                    for k in k_values:
                        top_k_preds = pred_ranking[:k]
                        hit = any(item in top_k_preds for item in valid_items)
                        hit_at_k[k].append(float(hit))
    
    # Compute averages
    metrics = {
        'mrr': np.mean(mrr_scores) if mrr_scores else 0.0,
        'num_samples': len(mrr_scores)
    }
    
    for k in k_values:
        metrics[f'hit_at_{k}'] = np.mean(hit_at_k[k]) if hit_at_k[k] else 0.0
    
    return metrics


def create_proximity_targets(input_ids, targets, window_size, vocab_size):
    """Create proximity targets for TOP ranking."""
    batch_size, seq_len = input_ids.shape
    
    # Initialize with -inf
    proximity_scores = torch.full(
        (batch_size, seq_len, vocab_size),
        float('-inf'),
        device=input_ids.device
    )
    
    # For each position, look ahead in the window
    for dist in range(1, min(window_size + 1, seq_len)):
        # Get tokens that appear 'dist' steps ahead
        future_tokens = torch.roll(targets, -dist, dims=1)
        
        # Create mask for valid positions
        valid_mask = torch.arange(seq_len, device=input_ids.device) < (seq_len - dist)
        valid_mask = valid_mask.unsqueeze(0).expand(batch_size, -1)
        
        # Set proximity scores (higher score = closer in time)
        score = window_size - dist
        # Use advanced indexing to set scores
        for b in range(batch_size):
            valid_positions = valid_mask[b]
            if valid_positions.any():
                pos_indices = torch.arange(seq_len, device=input_ids.device)[valid_positions]
                token_indices = future_tokens[b, valid_positions]
                proximity_scores[b, pos_indices, token_indices] = score
        
    return proximity_scores


def evaluate_model(model, dataloader, config, device='cuda'):
    """Comprehensive model evaluation."""
    print("Evaluating model...")
    
    results = {}
    
    # Compute perplexity
    print("Computing perplexity...")
    perplexity, avg_loss = compute_perplexity(model, dataloader, device)
    results['perplexity'] = perplexity
    results['avg_loss'] = avg_loss
    
    print(f"Perplexity: {perplexity:.4f}")
    print(f"Average Loss: {avg_loss:.4f}")
    
    # Compute ranking metrics if TOP is used
    if 'top' in config.get('objectives', []):
        print("Computing ranking metrics...")
        window_size = config.get('window_size', 128)
        ranking_metrics = compute_ranking_metrics(
            model, dataloader, window_size, device
        )
        results['ranking'] = ranking_metrics
        
        print(f"MRR: {ranking_metrics['mrr']:.4f}")
        for k in [1, 5, 10, 20]:
            if f'hit_at_{k}' in ranking_metrics:
                print(f"Hit@{k}: {ranking_metrics[f'hit_at_{k}']:.4f}")
    
    return results


def generate_text(model, tokenizer, prompt="The quick brown fox", max_length=50, device='cuda'):
    """Generate text using the model."""
    model.eval()
    
    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    with torch.no_grad():
        generated = model.generate(
            input_ids,
            max_length=max_length,
            temperature=0.8,
            top_k=50,
            top_p=0.9
        )
    
    # Decode generated text
    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    
    return generated_text


def compare_models(models, dataloader, config, device='cuda'):
    """Compare multiple models on the same dataset."""
    results = {}
    
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        results[name] = evaluate_model(model, dataloader, config, device)
    
    # Print comparison table
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    
    # Perplexity comparison
    print("\nPerplexity:")
    for name, result in results.items():
        print(f"  {name:20s}: {result['perplexity']:.4f}")
    
    # Ranking metrics comparison
    if any('ranking' in result for result in results.values()):
        print("\nRanking Metrics:")
        print(f"{'Model':<20s} {'MRR':<8s} {'Hit@1':<8s} {'Hit@5':<8s} {'Hit@10':<8s}")
        print("-" * 60)
        
        for name, result in results.items():
            if 'ranking' in result:
                r = result['ranking']
                print(f"{name:<20s} {r['mrr']:<8.4f} {r['hit_at_1']:<8.4f} "
                      f"{r['hit_at_5']:<8.4f} {r['hit_at_10']:<8.4f}")
    
    return results


def plot_training_curves(train_losses, val_losses, save_path=None):
    """Plot training curves."""
    import matplotlib.pyplot as plt
    
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    
    plt.show()
