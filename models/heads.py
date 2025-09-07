import torch
import torch.nn as nn
import torch.nn.functional as F


class PredictionHead(nn.Module):
    """Base class for prediction heads."""
    
    def __init__(self, d_model, vocab_size, bias=False):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size, bias=bias)
        
    def forward(self, hidden_states):
        """Forward pass through the head."""
        return self.linear(hidden_states)


class NTPHead(PredictionHead):
    """Next Token Prediction head."""
    
    def __init__(self, d_model, vocab_size, bias=False):
        super().__init__(d_model, vocab_size, bias)
        
    def compute_loss(self, logits, targets, ignore_index=-100):
        """Compute cross-entropy loss for next token prediction."""
        # Flatten for cross-entropy
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)
        
        return F.cross_entropy(logits_flat, targets_flat, ignore_index=ignore_index)


class MTPHead(PredictionHead):
    """Multi-Token Prediction head."""
    
    def __init__(self, d_model, vocab_size, k_future=3, bias=False):
        super().__init__(d_model, vocab_size, bias)
        self.k_future = k_future
        
    def compute_loss(self, logits, targets, k_future=None):
        """Compute average cross-entropy loss for next k tokens."""
        if k_future is None:
            k_future = self.k_future
            
        losses = []
        for i in range(k_future):
            # Shift targets by i positions
            shifted_targets = torch.roll(targets, -i, dims=1)
            
            # Compute loss for this future position
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = shifted_targets.view(-1)
            
            loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=-100)
            losses.append(loss)
            
        return sum(losses) / len(losses)


class TOPHead(PredictionHead):
    """Token Order Prediction head."""
    
    def __init__(self, d_model, vocab_size, window_size=128, bias=False):
        super().__init__(d_model, vocab_size, bias)
        self.window_size = window_size
        
    def create_proximity_targets(self, input_ids, targets, window_size=None):
        """
        Create proximity targets for TOP ranking.
        
        Args:
            input_ids: [batch_size, seq_len] input token indices
            targets: [batch_size, seq_len] target token indices (usually input_ids shifted by 1)
            window_size: size of the ranking window
            
        Returns:
            proximity_scores: [batch_size, seq_len, vocab_size] proximity scores
        """
        if window_size is None:
            window_size = self.window_size
            
        batch_size, seq_len = input_ids.shape
        vocab_size = self.linear.out_features
        
        # Initialize with -inf (will be masked out)
        proximity_scores = torch.full(
            (batch_size, seq_len, vocab_size), 
            float('-inf'), 
            device=input_ids.device
        )
        
        # For each position, look ahead in the window
        for dist in range(1, min(window_size + 1, seq_len)):
            # Get tokens that appear 'dist' steps ahead
            future_tokens = torch.roll(targets, -dist, dims=1)
            
            # Create mask for valid positions (not rolled around)
            valid_mask = torch.arange(seq_len, device=input_ids.device) < (seq_len - dist)
            valid_mask = valid_mask.unsqueeze(0).expand(batch_size, -1)
            
            # Set proximity scores (higher score = closer in time)
            score = window_size - dist
            proximity_scores[valid_mask, torch.arange(seq_len, device=input_ids.device), future_tokens] = score
            
        return proximity_scores
    
    def compute_loss(self, logits, proximity_targets):
        """Compute ListNet ranking loss."""
        # Convert to probabilities
        p_true = F.softmax(proximity_targets, dim=-1)
        p_pred = F.log_softmax(logits, dim=-1)
        
        # ListNet loss: KL divergence between true and predicted distributions
        loss = -(p_true * p_pred).sum(dim=-1)
        
        # Mask out invalid positions (where all targets are -inf)
        valid_mask = (proximity_targets != float('-inf')).any(dim=-1)
        loss = loss * valid_mask.float()
        
        return loss.sum() / valid_mask.sum().clamp(min=1)
    
    def compute_mrr(self, logits, proximity_targets, k=10):
        """
        Compute Mean Reciprocal Rank for TOP evaluation.
        
        Args:
            logits: [batch_size, seq_len, vocab_size] prediction logits
            proximity_targets: [batch_size, seq_len, vocab_size] true proximity scores
            k: number of top predictions to consider
            
        Returns:
            mrr: mean reciprocal rank
        """
        # Get top-k predictions
        _, top_k_indices = torch.topk(logits, k, dim=-1)
        
        # Find ranks of true tokens
        ranks = []
        for batch_idx in range(logits.size(0)):
            for seq_idx in range(logits.size(1)):
                # Get true tokens for this position (non-negative proximity scores)
                true_tokens = (proximity_targets[batch_idx, seq_idx] > 0).nonzero(as_tuple=True)[0]
                
                if len(true_tokens) > 0:
                    # Find the rank of the highest-scoring true token
                    true_scores = proximity_targets[batch_idx, seq_idx, true_tokens]
                    best_true_token = true_tokens[true_scores.argmax()]
                    
                    # Find rank in top-k predictions
                    rank = (top_k_indices[batch_idx, seq_idx] == best_true_token).nonzero(as_tuple=True)[0]
                    if len(rank) > 0:
                        ranks.append(1.0 / (rank[0].item() + 1))
                    else:
                        ranks.append(0.0)
        
        return torch.tensor(ranks).mean() if ranks else torch.tensor(0.0)
