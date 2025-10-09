“”“Resource estimation and metrics utilities.”””

import math
import numpy as np
import torch
from .helpers import count_trainable_params

def estimate_energy(model, steps, batch_size, energy_scale=1.0):
“””
Estimate energy consumption for training.

```
Args:
    model: PyTorch model
    steps: Number of training steps
    batch_size: Batch size
    energy_scale: Scaling factor for energy estimation

Returns:
    Estimated energy consumption (arbitrary units)
"""
n_params = count_trainable_params(model)
return energy_scale * 1e-3 * n_params * steps * max(1, batch_size)
```

def estimate_memory(model, batch_size):
“””
Estimate memory pressure for training.

```
Args:
    model: PyTorch model
    batch_size: Batch size

Returns:
    Memory pressure estimate [0, 1]
"""
n_params = count_trainable_params(model)
memory = 0.2 + 2e-8 * n_params * batch_size
return float(max(0.2, min(0.9, memory)))
```

def estimate_temperature(steps, batch_size, base_temp=0.35):
“””
Estimate temperature increase from computation.

```
Args:
    steps: Number of training steps
    batch_size: Batch size
    base_temp: Base temperature value

Returns:
    Temperature estimate [0, 1]
"""
temp = base_temp + 0.0015 * steps + 0.0005 * batch_size
return float(max(0.0, min(1.0, temp)))
```

def estimate_communication_bytes(delta_state, sparsity=0.8, bytes_per_float=4.0):
“””
Estimate communication cost in bytes.

```
Args:
    delta_state: Dictionary of parameter updates
    sparsity: Fraction of parameters transmitted
    bytes_per_float: Bytes per floating point value

Returns:
    Estimated communication cost in bytes
"""
total_bytes = 0.0
for value in delta_state.values():
    if torch.is_floating_point(value):
        total_bytes += value.numel() * sparsity * bytes_per_float
return total_bytes
```

@torch.no_grad()
def evaluate_model(model, valid_ids, device, block_size=None):
“””
Evaluate model on validation data.

```
Args:
    model: PyTorch model
    valid_ids: Validation token tensor
    device: Device to run evaluation on
    block_size: Sequence length (uses model.block_size if None)

Returns:
    accuracy: Token-level accuracy
    loss: Average loss
    num_batches: Number of evaluation batches
"""
model.eval()

if block_size is None:
    block_size = model.block_size

losses = []
accuracies = []
num_batches = 0

seq_len = len(valid_ids)

# Handle edge case
if seq_len <= block_size + 1:
    return 0.0, math.log(max(2, seq_len)), 0

# Evaluate on non-overlapping chunks
for i in range(0, seq_len - block_size - 1, block_size):
    x = valid_ids[i:i + block_size].unsqueeze(0).to(device, non_blocking=True)
    y = valid_ids[i + 1:i + 1 + block_size].unsqueeze(0).to(device, non_blocking=True)
    
    logits, loss = model(x, y)
    
    losses.append(float(loss.item()))
    
    # Compute accuracy
    predictions = torch.argmax(logits, dim=-1)
    accuracy = (predictions == y).float().mean().item()
    accuracies.append(accuracy)
    
    num_batches += 1

return float(np.mean(accuracies)), float(np.mean(losses)), num_batches
```