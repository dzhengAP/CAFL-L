"""Example: Basic CAFL training script."""

import torch
from cafl.models import TinyCharGPT
from cafl.data import prepare_shakespeare_data
from cafl.federated import run_federated_learning
from cafl.utils import set_seed, select_device
from cafl.utils.visualization import plot_training_metrics


def main():
    """Run basic CAFL training example."""
    
    # Configuration
    seed = 42
    device = select_device()  # Auto-detect CUDA/CPU
    
    print(f"Using device: {device}")
    set_seed(seed)
    
    # Prepare data
    print("\nPreparing Shakespeare dataset...")
    clients_data, valid_data, vocab_info = prepare_shakespeare_data(
        num_clients=12,
        overlap=32,
        device=device
    )
    
    # Create model
    print("\nCreating model...")
    model = TinyCharGPT(
        vocab_size=vocab_info["vocab_size"],
        block_size=96,
        n_layer=2,
        n_head=4,
        n_embd=96,
        dropout=0.1
    )
    
    print(f"Model has {model.count_parameters():,} parameters")
    
    # Define resource budgets
    budgets = {
        "energy": 5.0e7,
        "comm": 8.0e6,
        "memory": 0.50,
        "temp": 0.40
    }
    
    # Run CAFL training
    print("\nRunning CAFL training...")
    cafl_results = run_federated_learning(
        method="cafl",
        base_model=model,
        clients_data=clients_data,
        valid_data=valid_data,
        rounds=8,
        clients_per_round=4,
        local_steps=50,
        batch_size=12,
        learning_rate=0.002,
        block_size=96,
        budgets=tuple(budgets.values()),
        dual_lr=0.01,
        seed=seed,
        device=device,
        k_unfreeze=2,
        save_prefix="example_cafl",
        energy_scale=4.0,
        use_amp=(device.type == "cuda")
    )
    
    # Run baseline FedAvg for comparison
    print("\nRunning FedAvg baseline...")
    fedavg_results = run_federated_learning(
        method="fedavg",
        base_model=model,
        clients_data=clients_data,
        valid_data=valid_data,
        rounds=8,
        clients_per_round=4,
        local_steps=50,
        batch_size=12,
        learning_rate=0.002,
        block_size=96,
        budgets=tuple(budgets.values()),
        dual_lr=0.01,
        seed=seed,
        device=device,
        k_unfreeze=2,
        save_prefix="example_fedavg",
        energy_scale=4.0,
        use_amp=(device.type == "cuda")
    )
    
    # Generate comparison plots
    print("\nGenerating plots...")
    plot_data = {
        "fedavg": fedavg_results,
        "cafl": cafl_results,
        **{f"budget_{k}": v for k, v in budgets.items()}
    }
    plot_training_metrics(plot_data, save_dir=".")
    
    # Print summary
    print("\n" + "="*60)
    print("Training Summary")
    print("="*60)
    
    final_cafl_acc = cafl_results["val_acc"][-1]
    final_fedavg_acc = fedavg_results["val_acc"][-1]
    
    print(f"\nFinal Validation Accuracy:")
    print(f"  FedAvg: {final_fedavg_acc:.4f}")
    print(f"  CAFL:   {final_cafl_acc:.4f}")
    print(f"  Improvement: {final_cafl_acc - final_fedavg_acc:+.4f}")
    
    print(f"\nFinal Resource Usage (CAFL):")
    print(f"  Energy: {cafl_results['E_used'][-1]:.2e} / {budgets['energy']:.2e}")
    print(f"  Comm:   {cafl_results['C_used'][-1]:.2e} / {budgets['comm']:.2e}")
    print(f"  Memory: {cafl_results['M_used'][-1]:.3f} / {budgets['memory']:.3f}")
    print(f"  Temp:   {cafl_results['T_used'][-1]:.3f} / {budgets['temp']:.3f}")
    
    print("\nOutputs saved:")
    print("  - example_cafl_metrics.csv")
    print("  - example_fedavg_metrics.csv")
    print("  - acc.png, loss.png, energy.png, comm.png, memory.png, temp.png")


if __name__ == "__main__":
    main()
