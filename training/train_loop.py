import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
from tqdm import tqdm
import json

from models.gpt_mini import GPTMini
from models.heads import NTPHead, MTPHead, TOPHead
from losses.ranking import listnet_loss, compute_ranking_metrics


class TOPTrainer:
    """Trainer for Token Order Prediction experiments."""
    
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        # Setup scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('max_epochs', 10)
        )
        
        # Loss weights
        self.lambda_ntp = config.get('lambda_ntp', 1.0)
        self.lambda_mtp = config.get('lambda_mtp', 0.5)
        self.lambda_top = config.get('lambda_top', 0.5)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        # Create output directory
        os.makedirs(config.get('output_dir', 'outputs'), exist_ok=True)
        
    def compute_loss(self, batch, outputs):
        """Compute combined loss from all objectives."""
        input_ids = batch['input_ids']
        ntp_labels = batch['ntp_labels']
        
        losses = {}
        total_loss = 0
        
        # NTP Loss
        if 'ntp' in self.config.get('objectives', ['ntp']):
            ntp_loss = F.cross_entropy(
                outputs['ntp_logits'].view(-1, outputs['ntp_logits'].size(-1)),
                ntp_labels.view(-1),
                ignore_index=-100
            )
            losses['ntp'] = ntp_loss.item()
            total_loss += self.lambda_ntp * ntp_loss
        
        # MTP Loss
        if 'mtp' in self.config.get('objectives', []):
            k_future = self.config.get('k_future', 3)
            mtp_losses = []
            
            for i in range(k_future):
                shifted_labels = torch.roll(ntp_labels, -i, dims=1)
                mtp_loss = F.cross_entropy(
                    outputs['mtp_logits'].view(-1, outputs['mtp_logits'].size(-1)),
                    shifted_labels.view(-1),
                    ignore_index=-100
                )
                mtp_losses.append(mtp_loss)
            
            mtp_loss = sum(mtp_losses) / len(mtp_losses)
            losses['mtp'] = mtp_loss.item()
            total_loss += self.lambda_mtp * mtp_loss
        
        # TOP Loss
        if 'top' in self.config.get('objectives', []):
            window_size = self.config.get('window_size', 128)
            proximity_targets = self.create_proximity_targets(
                input_ids, ntp_labels, window_size
            )
            
            top_loss = listnet_loss(
                outputs['top_logits'],
                proximity_targets,
                temperature=self.config.get('top_temperature', 1.0)
            )
            losses['top'] = top_loss.item()
            total_loss += self.lambda_top * top_loss
        
        losses['total'] = total_loss.item()
        return total_loss, losses
    
    def create_proximity_targets(self, input_ids, targets, window_size):
        """Create proximity targets for TOP ranking."""
        batch_size, seq_len = input_ids.shape
        vocab_size = self.model.vocab_size
        
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
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        loss_components = {'ntp': 0, 'mtp': 0, 'top': 0}
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.model.tok_emb.weight.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch['input_ids'])
            
            # Compute loss
            loss, losses = self.compute_loss(batch, outputs)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['grad_clip']
                )
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            for key in losses:
                if key in loss_components:
                    loss_components[key] += losses[key]
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'ntp': f'{losses.get("ntp", 0):.4f}',
                'mtp': f'{losses.get("mtp", 0):.4f}',
                'top': f'{losses.get("top", 0):.4f}'
            })
        
        # Average losses
        avg_loss = total_loss / len(self.train_loader)
        for key in loss_components:
            loss_components[key] /= len(self.train_loader)
        
        return avg_loss, loss_components
    
    def validate(self):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        loss_components = {'ntp': 0, 'mtp': 0, 'top': 0}
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # Move to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.model.tok_emb.weight.device)
                
                # Forward pass
                outputs = self.model(batch['input_ids'])
                
                # Compute loss
                loss, losses = self.compute_loss(batch, outputs)
                
                # Update metrics
                total_loss += loss.item()
                for key in losses:
                    if key in loss_components:
                        loss_components[key] += losses[key]
        
        # Average losses
        avg_loss = total_loss / len(self.val_loader)
        for key in loss_components:
            loss_components[key] /= len(self.val_loader)
        
        return avg_loss, loss_components
    
    def train(self):
        """Main training loop."""
        print(f"Starting training with objectives: {self.config.get('objectives', ['ntp'])}")
        print(f"Model parameters: {self.model.get_num_params():,}")
        
        for epoch in range(self.config.get('max_epochs', 10)):
            self.current_epoch = epoch
            
            # Train
            train_loss, train_components = self.train_epoch()
            
            # Validate
            val_loss, val_components = self.validate()
            
            # Update learning rate
            self.scheduler.step()
            
            # Log metrics
            print(f"\nEpoch {epoch + 1}/{self.config.get('max_epochs', 10)}")
            print(f"Train Loss: {train_loss:.4f} (NTP: {train_components['ntp']:.4f}, "
                  f"MTP: {train_components['mtp']:.4f}, TOP: {train_components['top']:.4f})")
            print(f"Val Loss: {val_loss:.4f} (NTP: {val_components['ntp']:.4f}, "
                  f"MTP: {val_components['mtp']:.4f}, TOP: {val_components['top']:.4f})")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('best_model.pt')
                print("New best model saved!")
            
            # Save regular checkpoint
            if (epoch + 1) % self.config.get('save_every', 5) == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pt')
            
            # Store losses for plotting
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
        
        print(f"\nTraining completed! Best validation loss: {self.best_val_loss:.4f}")
    
    def save_checkpoint(self, filename):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        filepath = os.path.join(self.config.get('output_dir', 'outputs'), filename)
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location='cpu')
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        
        print(f"Checkpoint loaded from {filepath}")


def create_trainer(model, train_loader, val_loader, config):
    """Create a trainer instance."""
    return TOPTrainer(model, train_loader, val_loader, config)
