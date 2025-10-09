"""General utility functions."""

import random
import numpy as np
import torch


def select_device(force_device=None):
    """
    Select appropriate compute device.
    
    Args:
        force_device: Force specific device ("cpu", "cuda", or None for auto)
    
    Returns:
        torch.device object
    """
    if force_device == "cpu":
        return torch.device("cpu")
    
    if force_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    
    # Auto-select
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_model_state(model):
    """
    Extract model state as a dictionary of cloned tensors.
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary mapping parameter names to cloned tensors
    """
    with torch.no_grad():
        return {k: v.detach().clone() for k, v in model.named_parameters()}


def subtract_state(state_a, state_b):
    """
    Compute difference between two model states.
    
    Args:
        state_a: First model state dictionary
        state_b: Second model state dictionary
    
    Returns:
        Dictionary with element-wise differences
    """
    result = {}
    for (ka, va), (kb, vb) in zip(state_a.items(), state_b.items()):
        assert ka == kb, f"State keys don't match: {ka} vs {kb}"
        result[ka] = va - vb
    return result


def apply_aggregated_update(model, updates):
    """
    Apply averaged parameter updates to model.
    
    Args:
        model: PyTorch model to update
        updates: List of state dictionaries with parameter deltas
    """
    # Aggregate updates by averaging
    aggregated = {}
    for state_dict in updates:
        for key, value in state_dict.items():
            aggregated.setdefault(key, []).append(value.float())
    
    # Apply averaged updates to model
    with torch.no_grad():
        name_to_param = dict(model.named_parameters())
        for key, value_list in aggregated.items():
            mean_delta = torch.stack(value_list, dim=0).mean(dim=0)
            if key in name_to_param:
                param = name_to_param[key]
                param.data.add_(mean_delta.to(param.device))


def count_trainable_params(model):
    """
    Count number of trainable parameters in model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class CosineWithWarmup:
    """Cosine learning rate schedule with linear warmup."""
    
    def __init__(self, optimizer, warmup_steps, total_steps):
        """
        Initialize scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            warmup_steps: Number of warmup steps
            total_steps: Total number of training steps
        """
        self.optimizer = optimizer
        self.warmup = max(1, warmup_steps)
        self.total = max(1, total_steps)
        self.step_count = 0
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
    
    def step(self):
        """Update learning rate for current step."""
        self.step_count += 1
        
        if self.step_count <= self.warmup:
            # Linear warmup
            scale = self.step_count / self.warmup
        else:
            # Cosine decay
            progress = (self.step_count - self.warmup) / max(1, self.total - self.warmup)
            scale = 0.5 * (1 + np.cos(np.pi * progress))
        
        # Apply scaling to all parameter groups
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * scale
