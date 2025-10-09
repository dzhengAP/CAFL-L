"""CAFL policy and constraint management."""

import math


def apply_layer_freezing(model, k_unfreeze):
    """
    Apply layer freezing by unfreezing only the last k layers.
    
    Args:
        model: PyTorch model with 'blocks' attribute
        k_unfreeze: Number of layers to keep unfrozen
    """
    n_layers = len(model.blocks)
    
    for i, block in enumerate(model.blocks):
        # Freeze/unfreeze based on depth
        requires_grad = (i >= n_layers - k_unfreeze)
        for param in block.parameters():
            param.requires_grad = requires_grad
    
    # Always keep output layers trainable
    for param in model.ln_f.parameters():
        param.requires_grad = True
    for param in model.lm_head.parameters():
        param.requires_grad = True


def compute_cafl_policy(lam_E, lam_C, lam_M, lam_T, 
                       base_k, base_steps, base_bs):
    """
    Compute CAFL training policy from dual variables.
    
    Args:
        lam_E: Dual variable for energy constraint
        lam_C: Dual variable for communication constraint
        lam_M: Dual variable for memory constraint
        lam_T: Dual variable for temperature constraint
        base_k: Base number of unfrozen layers
        base_steps: Base number of training steps
        base_bs: Base batch size
    
    Returns:
        k_effective: Number of layers to unfreeze
        steps_effective: Number of training steps
        batch_size_effective: Effective batch size
        compression_level: Communication compression level (0, 1, or 2)
    """
    # Compute pressure indicators
    depth_pressure = 0.8 * lam_C + 0.8 * lam_M + 0.4 * lam_T
    steps_pressure = 0.9 * lam_E + 1.1 * lam_T
    batch_divisor = 1.0 + 8.0 * lam_T + 6.0 * lam_M
    
    # Adjust layer depth
    max_drop = max(0, base_k - 1)
    layer_drop = int(min(max_drop, math.ceil(base_k * min(1.0, 1.8 * depth_pressure))))
    k_effective = max(1, base_k - layer_drop)
    
    # Adjust training steps (up to 90% reduction)
    cut_fraction = min(0.9, 1.6 * steps_pressure)
    steps_effective = max(10, int(round(base_steps * (1.0 - cut_fraction))))
    
    # Adjust batch size
    batch_size_effective = max(8, int(base_bs / batch_divisor))
    
    # Determine compression level
    # 0 = 32-bit (4 bytes), 1 = 8/4-bit (1 byte), 2 = 2-bit (0.25 bytes)
    compression_level = 0
    if lam_C > 0.03 or lam_M > 0.04:
        compression_level = 1
    if lam_C > 0.08 or lam_M > 0.07:
        compression_level = 2
    
    return k_effective, steps_effective, batch_size_effective, compression_level


def get_compression_bytes_per_float(compression_level):
    """
    Get bytes per float for given compression level.
    
    Args:
        compression_level: 0 (32-bit), 1 (8-bit), or 2 (2-bit)
    
    Returns:
        Bytes per floating point value
    """
    compression_map = {
        0: 4.0,   # 32-bit float
        1: 1.0,   # 8-bit quantized
        2: 0.25   # 2-bit quantized
    }
    return compression_map.get(compression_level, 4.0)


class DualAscentOptimizer:
    """Optimizer for dual variables in CAFL."""
    
    def __init__(self, dual_lr=0.01, dead_zone=0.05):
        """
        Initialize dual ascent optimizer.
        
        Args:
            dual_lr: Learning rate for dual variables
            dead_zone: Dead zone around budget (±5% no update)
        """
        self.dual_lr = dual_lr
        self.dead_zone = dead_zone
        
        # Dual variables
        self.lam_E = 0.0
        self.lam_C = 0.0
        self.lam_M = 0.0
        self.lam_T = 0.0
    
    def _apply_dead_zone(self, usage_ratio):
        """Apply dead zone to usage ratio."""
        upper = 1.0 + self.dead_zone
        lower = 1.0 - self.dead_zone
        
        if usage_ratio > upper:
            return usage_ratio - upper
        elif usage_ratio < lower:
            return usage_ratio - lower
        else:
            return 0.0
    
    def update(self, E_usage, C_usage, M_usage, T_usage,
               E_budget, C_budget, M_budget, T_budget):
        """
        Update dual variables based on resource usage.
        
        Args:
            E_usage, C_usage, M_usage, T_usage: Current resource usage
            E_budget, C_budget, M_budget, T_budget: Resource budgets
        """
        eps = 1e-9
        
        # Compute usage ratios
        E_ratio = E_usage / (E_budget + eps)
        C_ratio = C_usage / (C_budget + eps)
        M_ratio = M_usage / (M_budget + eps)
        T_ratio = T_usage / (T_budget + eps)
        
        # Update dual variables with dead zone
        self.lam_E = max(0.0, self.lam_E + self.dual_lr * self._apply_dead_zone(E_ratio))
        self.lam_C = max(0.0, self.lam_C + self.dual_lr * self._apply_dead_zone(C_ratio))
        self.lam_M = max(0.0, self.lam_M + self.dual_lr * self._apply_dead_zone(M_ratio))
        self.lam_T = max(0.0, self.lam_T + self.dual_lr * self._apply_dead_zone(T_ratio))
    
    def get_duals(self):
        """Get current dual variable values."""
        return self.lam_E, self.lam_C, self.lam_M, self.lam_T
    
    def scale_duals(self, scale_factor):
        """Scale all dual variables by a factor."""
        self.lam_E *= scale_factor
        self.lam_C *= scale_factor
        self.lam_M *= scale_factor
        self.lam_T *= scale_factor
