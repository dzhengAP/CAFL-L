"""Federated learning module."""

from .client import client_local_update
from .server import run_federated_learning
from .policy import (
    apply_layer_freezing,
    compute_cafl_policy,
    get_compression_bytes_per_float,
    DualAscentOptimizer
)

__all__ = [
    "client_local_update",
    "run_federated_learning",
    "apply_layer_freezing",
    "compute_cafl_policy",
    "get_compression_bytes_per_float",
    "DualAscentOptimizer"
]
