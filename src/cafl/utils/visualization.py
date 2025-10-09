"""Visualization utilities for plotting training metrics."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def configure_plot_style():
    """Configure matplotlib style for publication-quality plots."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "font.size": 12,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05
    })


def plot_comparison(rounds, fedavg_values, cafl_values, ylabel, 
                   budget=None, output_path="plot.png", title=None):
    """
    Plot comparison between FedAvg and CAFL methods.
    
    Args:
        rounds: List of round numbers
        fedavg_values: Metric values for FedAvg
        cafl_values: Metric values for CAFL
        ylabel: Y-axis label
        budget: Optional budget line to plot
        output_path: Path to save figure
        title: Optional plot title
    """
    configure_plot_style()
    
    plt.figure(figsize=(5, 3.4))
    plt.plot(rounds, fedavg_values, marker="o", ms=3, label="FedAvg", linewidth=2)
    plt.plot(rounds, cafl_values, marker="s", ms=3, label="CAFL", linewidth=2)
    
    if budget is not None:
        plt.axhline(budget, ls="--", lw=1.2, color="red", alpha=0.7, label="Budget")
    
    plt.xlabel("Round")
    plt.ylabel(ylabel)
    
    if title:
        plt.title(title)
    
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_training_metrics(logs_dict, save_dir="."):
    """
    Generate all training metric plots.
    
    Args:
        logs_dict: Dictionary with 'fedavg' and 'cafl' subdicts containing metrics
        save_dir: Directory to save plots
    """
    import os
    
    fedavg_logs = logs_dict["fedavg"]
    cafl_logs = logs_dict["cafl"]
    rounds = fedavg_logs["round"]
    
    # Accuracy plot
    plot_comparison(
        rounds,
        fedavg_logs["val_acc"],
        cafl_logs["val_acc"],
        "Validation Accuracy",
        output_path=os.path.join(save_dir, "acc.png")
    )
    
    # Loss plot
    plot_comparison(
        rounds,
        fedavg_logs["val_loss"],
        cafl_logs["val_loss"],
        "Validation Loss",
        output_path=os.path.join(save_dir, "loss.png")
    )
    
    # Energy plot
    plot_comparison(
        rounds,
        fedavg_logs["E_used"],
        cafl_logs["E_used"],
        "Energy (proxy)",
        budget=logs_dict.get("budget_energy"),
        output_path=os.path.join(save_dir, "energy.png")
    )
    
    # Communication plot (convert to MB)
    plot_comparison(
        rounds,
        [x / 1e6 for x in fedavg_logs["C_used"]],
        [x / 1e6 for x in cafl_logs["C_used"]],
        "Communication (MB)",
        budget=logs_dict.get("budget_comm", 0) / 1e6 if logs_dict.get("budget_comm") else None,
        output_path=os.path.join(save_dir, "comm.png")
    )
    
    # Memory plot
    plot_comparison(
        rounds,
        fedavg_logs["M_used"],
        cafl_logs["M_used"],
        "Memory Pressure",
        budget=logs_dict.get("budget_memory"),
        output_path=os.path.join(save_dir, "memory.png")
    )
    
    # Temperature plot
    plot_comparison(
        rounds,
        fedavg_logs["T_used"],
        cafl_logs["T_used"],
        "Temperature Proxy",
        budget=logs_dict.get("budget_temp"),
        output_path=os.path.join(save_dir, "temp.png")
    )
    
    print(f"Plots saved to {save_dir}/")
