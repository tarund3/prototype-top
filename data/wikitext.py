import torch
import itertools
import random
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import Dataset


class WikiText2(Dataset):
    """WikiText-2 dataset loader with tokenization and sequence chunking."""
    
    def __init__(self, split="train", seq_len=512, tokenizer="gpt2"):
        self.tok = AutoTokenizer.from_pretrained(tokenizer)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
            
        # Load dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        
        # Tokenize all text and concatenate
        ids = list(itertools.chain.from_iterable(
            self.tok(t).input_ids + [self.tok.eos_token_id] 
            for t in ds["text"] if t.strip()  # Skip empty texts
        ))
        
        # Drop last partial chunk to ensure all sequences are complete
        self.ids = torch.tensor(ids[:len(ids) // seq_len * seq_len])
        self.seq_len = seq_len
        self.vocab_size = len(self.tok)
        
    def __len__(self):
        return len(self.ids) // self.seq_len
    
    def __getitem__(self, idx):
        s = idx * self.seq_len
        x = self.ids[s : s + self.seq_len + 1]  # +1 for NTP label
        
        return {
            "input_ids": x[:-1],
            "ntp_labels": x[1:],
            "sequence_length": self.seq_len
        }
    
    def get_vocab_size(self):
        return self.vocab_size
    
    def get_tokenizer(self):
        return self.tok


def create_dataloader(split="train", seq_len=512, batch_size=8, shuffle=True, tokenizer="gpt2"):
    """Create a DataLoader for WikiText-2."""
    from torch.utils.data import DataLoader
    
    dataset = WikiText2(split=split, seq_len=seq_len, tokenizer=tokenizer)
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=0,  # Avoid multiprocessing issues
        pin_memory=True
    )
