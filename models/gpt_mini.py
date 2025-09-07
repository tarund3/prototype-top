import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GPTMini(nn.Module):
    """Mini GPT model with NTP, MTP, and TOP prediction heads."""
    
    def __init__(self, vocab_size, n_layer=4, n_head=8, d_model=256, max_seq_len=1024):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layer = n_layer
        self.n_head = n_head
        self.max_seq_len = max_seq_len
        
        # Token and position embeddings
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(max_seq_len, d_model))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_head,
                dim_feedforward=4 * d_model,
                batch_first=True,
                dropout=0.1
            )
            for _ in range(n_layer)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(d_model)
        
        # Prediction heads
        self.head_ntp = nn.Linear(d_model, vocab_size, bias=False)
        self.head_mtp = nn.Linear(d_model, vocab_size, bias=False)
        self.head_top = nn.Linear(d_model, vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights following GPT-2 style."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
            
    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass through the model.
        
        Args:
            input_ids: [batch_size, seq_len] token indices
            attention_mask: [batch_size, seq_len] attention mask (optional)
            
        Returns:
            dict with logits for each head and hidden states
        """
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        x = self.tok_emb(input_ids)  # [batch_size, seq_len, d_model]
        
        # Position embeddings
        x = x + self.pos_emb[:seq_len].unsqueeze(0)  # [batch_size, seq_len, d_model]
        
        # Create causal attention mask for autoregressive generation
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, device=input_ids.device)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, src_key_padding_mask=~attention_mask.bool())
        
        # Final layer norm
        h = self.ln_f(x)  # [batch_size, seq_len, d_model]
        
        # Generate predictions from all heads
        return {
            "ntp_logits": self.head_ntp(h),      # [batch_size, seq_len, vocab_size]
            "mtp_logits": self.head_mtp(h),      # [batch_size, seq_len, vocab_size]
            "top_logits": self.head_top(h),      # [batch_size, seq_len, vocab_size]
            "hidden": h                          # [batch_size, seq_len, d_model]
        }
    
    def get_num_params(self):
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def generate(self, input_ids, max_length=100, temperature=1.0, top_k=None, top_p=None):
        """
        Generate text using the NTP head.
        
        Args:
            input_ids: [batch_size, seq_len] starting tokens
            max_length: maximum length to generate
            temperature: sampling temperature
            top_k: top-k sampling
            top_p: nucleus sampling
            
        Returns:
            generated token sequences
        """
        self.eval()
        with torch.no_grad():
            for _ in range(max_length):
                # Get predictions
                outputs = self.forward(input_ids)
                logits = outputs["ntp_logits"][:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                
                # Apply top-p filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = -float('Inf')
                
                # Sample next token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids
