"""Main training script for CAFL."""

import os
import argparse
import torch

from cafl.models.transformer import TinyCharGPT
from cafl.data.dataset import prepare_shakespeare_data
from cafl.federated.server import run_federated_learning
from cafl.utils.helpers import select_device, set_seed
from cafl.utils.visualization import plot_training_metrics


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Constraint-Aware Federated Learning (CAFL)"
    )
    
    # Training schedule
    parser.add_argument("--rounds", type=int, default=8,
                       help="Number of federated rounds")
    parser.add_argument("--local_steps", type=int, default=60,
                       help="Local training steps per round")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.0025,
                       help="Learning rate")
    parser.add_argument("--block_size", type=int, default=128,
                       help="Sequence length")
    parser.add_argument("--overlap", type=int, default=48,
                       help="Overlap between client data shards")
    parser.add_argument("--num_clients", type=int, default=16,
                       help="Number of federated clients")
    parser.add_argument("--clients_per_round", type=int, default=6,
                       help="Clients to sample per round")
    
    # Model architecture
    parser.add_argument("--n_layer", type=int, default=2,
                       help="Number of transformer layers")
    parser.add_argument("--n_head", type=int, default=4,
                       help="Number of attention heads")
    parser.add_argument("--n_embd", type=int, default=128,
                       help="Embedding dimension")
    parser.add_argument("--dropout", type=float, default=0.1,
                       help="Dropout rate")
    
    # Resource budgets
    parser.add_argument("--budget_energy", type=float, default=6.0e7,
                       help="Energy budget")
    parser.add_argument("--budget_comm", type=float, default=9.0e6,
                       help="Communication budget (bytes)")
    parser.add_argument("--budget_memory", type=float, default=0.55,
                       help="Memory pressure budget")
    parser.add_argument("--budget_temp", type=float, default=0.45,
                       help="Temperature budget")
    
    # CAFL parameters
    parser.add_argument("--dual_lr", type=float, default=0.01,
                       help="Learning rate for dual variables")
    parser.add_argument("--k_unfreeze", type=int, default=2,
                       help="Initial number of unfrozen layers")
    parser.add_argument("--energy_scale", type=float, default=5.0,
                       help="Energy estimation scaling factor")
    
    # System configuration
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--save_dir", type=str, default=".",
                       help="Directory to save outputs")
    parser.add_argument("--device", type=str, default=None,
                       choices=[None, "cpu", "cuda"],
                       help="Device to use (auto-detect if None)")
    parser.add_argument("--amp", action="store_true",
                       help="Enable mixed precision training")
    parser.add_argument("--fast", action="store_true",
                       help="Use tiny preset for quick testing")
    
    return parser.parse_args()


def apply_fast_preset(args):
    """Apply fast testing preset."""
    args.rounds = 6
    args.local_steps = 50
    args.batch_size = 12
    args.block_size = 96
    args.clients_per_round = 4
    args.n_layer = 2
    args.n_head = 4
    args.n_embd = 96
    args.k_unfreeze = 2


def setup_device(args):
    """Setup and configure compute device."""
    device = select_device(args.device)
    
    # Enable AMP by default on CUDA
    if device.type == "cuda" and not args.amp:
        args.amp = True
    
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(0)
        compute_cap = torch.cuda.get_device_capability(0)
        print(f"[Device] CUDA: {device_name} (CC {compute_cap[0]}.{compute_cap[1]}) | AMP={args.amp}")
        
        # Optimize CUDA settings
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    else:
        print("[Device] CPU | AMP=False")
    
    return device


def create_model(vocab_size, args):
    """Create TinyCharGPT model."""
    model = TinyCharGPT(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout
    )
    
    num_params = model.count_parameters(trainable_only=False)
    print(f"[Model] Total parameters: {num_params:,}")
    
    return model


def main():
    """Main training function."""
    args = parse_arguments()
    
    # Apply fast preset if requested
    if args.fast:
        apply_fast_preset(args)
        print("[Config] Using fast preset for quick testing")
    
    # Set seed and device
    set_seed(args.seed)
    device = setup_device(args)
    
    # Create output directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Prepare data
    print("\n[Data] Preparing Shakespeare dataset...")
    clients_data, valid_data, vocab_info = prepare_shakespeare_data(
        num_clients=args.num_clients,
        overlap=args.overlap,
        device=device
    )
    
    # Create model
    print("\n[Model] Creating TinyCharGPT...")
    model = create_model(vocab_info["vocab_size"], args)
    
    # Resource budgets
    budgets = (
        args.budget_energy,
        args.budget_comm,
        args.budget_memory,
        args.budget_temp
    )
    
    # Run FedAvg
    print("\n" + "="*60)
    print("Training with FedAvg (baseline)")
    print("="*60)
    
    fedavg_logs = run_federated_learning(
        method="fedavg",
        base_model=model,
        clients_data=clients_data,
        valid_data=valid_data,
        rounds=args.rounds,
        clients_per_round=args.clients_per_round,
        local_steps=args.local_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        block_size=args.block_size,
        budgets=budgets,
        dual_lr=args.dual_lr,
        seed=args.seed,
        device=device,
        k_unfreeze=args.k_unfreeze,
        save_prefix=os.path.join(args.save_dir, "fedavg"),
        energy_scale=args.energy_scale,
        use_amp=args.amp
    )
    
    # Run CAFL
    print("\n" + "="*60)
    print("Training with CAFL (constraint-aware)")
    print("="*60)
    
    cafl_logs = run_federated_learning(
        method="cafl",
        base_model=model,
        clients_data=clients_data,
        valid_data=valid_data,
        rounds=args.rounds,
        clients_per_round=args.clients_per_round,
        local_steps=args.local_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        block_size=args.block_size,
        budgets=budgets,
        dual_lr=args.dual_lr,
        seed=args.seed,
        device=device,
        k_unfreeze=args.k_unfreeze,
        save_prefix=os.path.join(args.save_dir, "cafl"),
        energy_scale=args.energy_scale,
        use_amp=args.amp
    )
    
    # Generate plots
    print("\n[Visualization] Generating comparison plots...")
    plot_data = {
        "fedavg": fedavg_logs,
        "cafl": cafl_logs,
        "budget_energy": args.budget_energy,
        "budget_comm": args.budget_comm,
        "budget_memory": args.budget_memory,
        "budget_temp": args.budget_temp
    }
    plot_training_metrics(plot_data, save_dir=args.save_dir)
    
    # Summary
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Results saved to: {args.save_dir}")
    print("\nFiles generated:")
    print("  - fedavg_metrics.csv")
    print("  - cafl_metrics.csv")
    print("  - acc.png (validation accuracy)")
    print("  - loss.png (validation loss)")
    print("  - energy.png (energy usage)")
    print("  - comm.png (communication overhead)")
    print("  - memory.png (memory pressure)")
    print("  - temp.png (temperature proxy)")
    
    # Final metrics
    final_fedavg_acc = fedavg_logs["val_acc"][-1]
    final_cafl_acc = cafl_logs["val_acc"][-1]
    print(f"\nFinal Validation Accuracy:")
    print(f"  FedAvg: {final_fedavg_acc:.4f}")
    print(f"  CAFL:   {final_cafl_acc:.4f}")
    print(f"  Improvement: {(final_cafl_acc - final_fedavg_acc):.4f}")


if __name__ == "__main__":
    main()
