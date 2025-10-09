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