"""Transformer model implementation for character-level language modeling."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Causal self-attention with masking for autoregressive generation."""
    
    def __init__(self, n_embd, n_head, block_size, attn_drop=0.0, resid_drop=0.1):
        super().__init__()
        assert n_embd % n_head == 0, "Embedding dimension must be divisible by number of heads"
        
        self.n_head = n_head
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.resid_drop = nn.Dropout(resid_drop)
        
        # Causal mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        )
    
    def forward(self, x):
        B, T, C = x.size()
        nh = self.n_head
        hs = C // nh
        
        # Compute Q, K, V
        k = self.key(x).view(B, T, nh, hs).transpose(1, 2)
        q = self.query(x).view(B, T, nh, hs).transpose(1, 2)
        v = self.value(x).view(B, T, nh, hs).transpose(1, 2)
        
        # Scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hs)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        
        # Apply attention to values
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        
        return self.resid_drop(y)


class Block(nn.Module):
    """Transformer block with attention and MLP."""
    
    def __init__(self, n_embd, n_head, block_size, attn_drop=0.0, resid_drop=0.1, mlp_drop=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, attn_drop, resid_drop)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Dropout(mlp_drop),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(resid_drop)
        )
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyCharGPT(nn.Module):
    """Tiny GPT model for character-level language modeling."""
    
    def __init__(self, vocab_size, block_size, n_layer=6, n_head=8, n_embd=256, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        
        # Token and position embeddings
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, n_embd))
        self.drop = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(n_embd, n_head, block_size, attn_drop=dropout, resid_drop=dropout, mlp_drop=dropout)
            for _ in range(n_layer)
        ])
        
        # Output layers
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights following GPT-2 initialization."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, idx, targets=None):
        """
        Forward pass through the model.
        
        Args:
            idx: Input token indices [B, T]
            targets: Target token indices [B, T] (optional)
        
        Returns:
            logits: Output logits [B, T, vocab_size]
            loss: Cross-entropy loss if targets provided
        """
        B, T = idx.shape
        assert T <= self.block_size, f"Sequence length {T} exceeds block size {self.block_size}"
        
        # Forward pass
        x = self.tok_emb(idx) + self.pos_emb[:, :T, :]
        x = self.drop(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                label_smoothing=0.05  # Small smoothing helps with micro batches
            )
        
        return logits, loss
    
    def count_parameters(self, trainable_only=True):
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
