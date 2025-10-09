"""Utilities module."""

from .helpers import (
    select_device,
    set_seed,
    get_model_state,
    subtract_state,
    apply_aggregated_update,
    count_trainable_params,
    CosineWithWarmup
)

from .metrics import (
    estimate_energy,
    estimate_memory,
    estimate_temperature,
    estimate_communication_bytes,
    evaluate_model
)

from .visualization import (
    configure_plot_style,
    plot_comparison,
    plot_training_metrics
)

__all__ = [
    # helpers
    "select_device",
    "set_seed",
    "get_model_state",
    "subtract_state",
    "apply_aggregated_update",
    "count_trainable_params",
    "CosineWithWarmup",
    # metrics
    "estimate_energy",
    "estimate_memory",
    "estimate_temperature",
    "estimate_communication_bytes",
    "evaluate_model",
    # visualization
    "configure_plot_style",
    "plot_comparison",
    "plot_training_metrics"
]
