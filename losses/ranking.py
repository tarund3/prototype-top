import torch
import torch.nn.functional as F


def listnet_loss(scores, targets, temperature=1.0):
    """
    ListNet ranking loss implementation.
    
    Args:
        scores: [batch_size, seq_len, vocab_size] predicted scores
        targets: [batch_size, seq_len, vocab_size] true proximity scores
        temperature: temperature for softmax scaling
        
    Returns:
        loss: scalar tensor
    """
    # Apply temperature scaling
    scores = scores / temperature
    targets = targets / temperature
    
    # Mask out invalid positions (where all targets are -inf)
    valid_mask = (targets != float('-inf')).any(dim=-1)
    
    if not valid_mask.any():
        return torch.tensor(0.0, device=scores.device, requires_grad=True)
    
    # Convert to probabilities
    p_true = F.softmax(targets, dim=-1)
    p_pred = F.log_softmax(scores, dim=-1)
    
    # ListNet loss: KL divergence between true and predicted distributions
    loss = -(p_true * p_pred).sum(dim=-1)
    
    # Apply mask
    loss = loss * valid_mask.float()
    
    # Average over valid positions
    return loss.sum() / valid_mask.sum().clamp(min=1)


def listmle_loss(scores, targets, k=10):
    """
    ListMLE (Listwise Maximum Likelihood Estimation) loss.
    
    Args:
        scores: [batch_size, seq_len, vocab_size] predicted scores
        targets: [batch_size, seq_len, vocab_size] true proximity scores
        k: number of top items to consider
        
    Returns:
        loss: scalar tensor
    """
    batch_size, seq_len, vocab_size = scores.shape
    
    losses = []
    for b in range(batch_size):
        for s in range(seq_len):
            # Get true ranking (sorted by proximity scores)
            true_scores = targets[b, s]
            valid_items = (true_scores != float('-inf')).nonzero(as_tuple=True)[0]
            
            if len(valid_items) == 0:
                continue
                
            # Sort by true scores (descending)
            sorted_indices = true_scores[valid_items].argsort(descending=True)
            true_ranking = valid_items[sorted_indices]
            
            # Get top-k items
            top_k = min(k, len(true_ranking))
            true_ranking = true_ranking[:top_k]
            
            if top_k == 0:
                continue
                
            # Compute ListMLE loss
            pred_scores = scores[b, s, true_ranking]
            
            # Compute log-likelihood
            log_likelihood = 0
            for i in range(top_k):
                # Probability of item i being ranked first among remaining items
                remaining_scores = pred_scores[i:]
                log_prob = F.log_softmax(remaining_scores, dim=0)[0]
                log_likelihood += log_prob
                
            losses.append(-log_likelihood)
    
    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=scores.device)


def pairwise_ranking_loss(scores, targets, margin=1.0):
    """
    Pairwise ranking loss (RankNet style).
    
    Args:
        scores: [batch_size, seq_len, vocab_size] predicted scores
        targets: [batch_size, seq_len, vocab_size] true proximity scores
        margin: margin for ranking loss
        
    Returns:
        loss: scalar tensor
    """
    batch_size, seq_len, vocab_size = scores.shape
    
    losses = []
    for b in range(batch_size):
        for s in range(seq_len):
            true_scores = targets[b, s]
            valid_items = (true_scores != float('-inf')).nonzero(as_tuple=True)[0]
            
            if len(valid_items) < 2:
                continue
                
            # Get all pairs of valid items
            for i in range(len(valid_items)):
                for j in range(i + 1, len(valid_items)):
                    item_i, item_j = valid_items[i], valid_items[j]
                    score_i, score_j = scores[b, s, item_i], scores[b, s, item_j]
                    true_i, true_j = true_scores[item_i], true_scores[item_j]
                    
                    # True ranking: higher score should rank higher
                    if true_i > true_j:
                        # Item i should rank higher than item j
                        loss = F.relu(margin - (score_i - score_j))
                    else:
                        # Item j should rank higher than item i
                        loss = F.relu(margin - (score_j - score_i))
                    
                    losses.append(loss)
    
    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=scores.device)


def compute_ranking_metrics(scores, targets, k_values=[1, 5, 10]):
    """
    Compute ranking metrics (MRR, NDCG, etc.).
    
    Args:
        scores: [batch_size, seq_len, vocab_size] predicted scores
        targets: [batch_size, seq_len, vocab_size] true proximity scores
        k_values: list of k values for top-k metrics
        
    Returns:
        metrics: dict with various ranking metrics
    """
    batch_size, seq_len, vocab_size = scores.shape
    
    metrics = {}
    
    # Mean Reciprocal Rank (MRR)
    mrr_scores = []
    for b in range(batch_size):
        for s in range(seq_len):
            true_scores = targets[b, s]
            valid_items = (true_scores != float('-inf')).nonzero(as_tuple=True)[0]
            
            if len(valid_items) == 0:
                continue
                
            # Find the highest-scoring true item
            best_item = valid_items[true_scores[valid_items].argmax()]
            
            # Find its rank in predictions
            pred_ranking = scores[b, s].argsort(descending=True)
            rank = (pred_ranking == best_item).nonzero(as_tuple=True)[0]
            
            if len(rank) > 0:
                mrr_scores.append(1.0 / (rank[0].item() + 1))
            else:
                mrr_scores.append(0.0)
    
    metrics['mrr'] = torch.tensor(mrr_scores).mean() if mrr_scores else torch.tensor(0.0)
    
    # Top-k metrics
    for k in k_values:
        hit_at_k = []
        for b in range(batch_size):
            for s in range(seq_len):
                true_scores = targets[b, s]
                valid_items = (true_scores != float('-inf')).nonzero(as_tuple=True)[0]
                
                if len(valid_items) == 0:
                    continue
                    
                # Get top-k predictions
                top_k_preds = scores[b, s].argsort(descending=True)[:k]
                
                # Check if any valid item is in top-k
                hit = any(item in top_k_preds for item in valid_items)
                hit_at_k.append(float(hit))
        
        metrics[f'hit_at_{k}'] = torch.tensor(hit_at_k).mean() if hit_at_k else torch.tensor(0.0)
    
    return metrics
