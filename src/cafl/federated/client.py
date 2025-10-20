"""Client-side federated learning logic."""

import math
import torch
import torch.nn as nn
from copy import deepcopy

from ..data.dataset import batchify
from ..utils.helpers import get_model_state, subtract_state, CosineWithWarmup
from ..utils.metrics import estimate_energy, estimate_memory, estimate_temperature


def client_local_update(model, base_state, client_data, steps, physical_batch_size,
                       learning_rate, device, block_size, energy_scale=1.0,
                       weight_decay=0.05, warmup_fraction=0.1, use_amp=True,
                       seed=0, grad_accumulation=1, base_steps_ref=60, base_bs_ref=16):
    """
    Perform local training update for a federated client.
    
    Args:
        model: PyTorch model
        base_state: Starting model state (from server)
        client_data: Client's token data tensor
        steps: Number of training steps
        physical_batch_size: Physical batch size (for memory/temp)
        learning_rate: Base learning rate
        device: Device to train on
        block_size: Sequence length
        energy_scale: Energy estimation scaling factor
        weight_decay: Weight decay coefficient
        warmup_fraction: Fraction of steps for warmup
        use_amp: Use automatic mixed precision
        seed: Random seed for this client
        grad_accumulation: Gradient accumulation steps
        base_steps_ref: Reference steps for LR scaling
        base_bs_ref: Reference batch size for LR scaling
    
    Returns
        stats: Dictionary with training statistics
        delta: Parameter update (difference from base_state)
    """
    # Reset model to server state
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(base_state[name].to(param.device))
    
    model.train()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    # Adjust learning rate based on token budget
    token_ratio = (steps * physical_batch_size) / max(1, base_steps_ref * base_bs_ref)
    if token_ratio >= 0.9:
        effective_lr = learning_rate
    else:
        # Scale up LR when using fewer tokens (but cap at 5x)
        effective_lr = min(learning_rate * (1.0 / max(0.3, token_ratio)), 5 * learning_rate)
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=effective_lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.98),
        eps=1e-8
    )
    
    warmup_steps = int(warmup_fraction * max(1, steps))
    scheduler = CosineWithWarmup(optimizer, warmup_steps, max(1, steps))
    
    # Setup random generator for this client
    rng = torch.Generator(device=device).manual_seed(seed)
    
    # Mixed precision setup
    use_amp = use_amp and (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    # Training loop
    total_loss = 0.0
    
    for step_idx in range(steps):
        # Skip if data too short
        if client_data.numel() <= block_size + 1:
            break
        
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        
        # Gradient accumulation loop
        for _ in range(max(1, grad_accumulation)):
            x, y = batchify(client_data, block_size, physical_batch_size, rng)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            
            # Forward pass with optional AMP
            if use_amp:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    logits, loss = model(x, y)
            else:
                logits, loss = model(x, y)
            
            # Scale loss by accumulation steps
            loss = loss / max(1, grad_accumulation)
            
            # Backward pass
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            step_loss += float(loss.item()) * max(1, grad_accumulation)
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        
        # Optimizer step
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        
        scheduler.step()
        total_loss += step_loss
    
    # Compute parameter update
    new_state = get_model_state(model)
    delta = subtract_state(new_state, base_state)
    
    # Compute resource usage statistics
    stats = {
        "train_loss": total_loss / max(1, steps),
        "E_used": estimate_energy(model, steps, physical_batch_size, energy_scale),
        "M_used": estimate_memory(model, physical_batch_size),
        "T_used": estimate_temperature(steps, physical_batch_size)
    }
    
    return stats, delta
