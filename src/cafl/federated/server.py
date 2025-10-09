"""Server-side federated learning orchestration."""

import os
import csv
import random
import math
from copy import deepcopy

from .client import client_local_update
from .policy import (
    apply_layer_freezing, 
    compute_cafl_policy, 
    get_compression_bytes_per_float,
    DualAscentOptimizer
)
from ..utils.helpers import get_model_state, apply_aggregated_update
from ..utils.metrics import evaluate_model, estimate_communication_bytes


def run_federated_learning(method, base_model, clients_data, valid_data, rounds,
                          clients_per_round, local_steps, batch_size, learning_rate,
                          block_size, budgets, dual_lr, seed, device, k_unfreeze,
                          save_prefix, energy_scale=1.0, use_amp=True):
    """
    Run federated learning with FedAvg or CAFL.
    
    Args:
        method: 'fedavg' or 'cafl'
        base_model: Initial model
        clients_data: List of client data tensors
        valid_data: Validation data tensor
        rounds: Number of federated rounds
        clients_per_round: Clients to sample per round
        local_steps: Local training steps
        batch_size: Batch size
        learning_rate: Learning rate
        block_size: Sequence length
        budgets: Tuple of (energy, comm, memory, temp) budgets
        dual_lr: Dual variable learning rate
        seed: Random seed
        device: Training device
        k_unfreeze: Initial number of unfrozen layers
        save_prefix: Prefix for saving files
        energy_scale: Energy estimation scale
        use_amp: Use automatic mixed precision
    
    Returns:
        Dictionary with training logs
    """
    random.seed(seed)
    
    # Setup
    model = deepcopy(base_model).to(device)
    worker_model = deepcopy(model).to(device)
    
    E_budget, C_budget, M_budget, T_budget = budgets
    
    # Dual optimizer for CAFL
    dual_optimizer = DualAscentOptimizer(dual_lr=dual_lr, dead_zone=0.05)
    
    # Tracking variables
    E_smooth = C_smooth = M_smooth = T_smooth = 0.0
    ema_beta = 0.5
    
    # Baseline references for token preservation
    BASE_STEPS = local_steps
    BASE_BS = batch_size
    
    # Adaptive parameters (for CAFL)
    adaptive_k = k_unfreeze
    adaptive_steps = local_steps
    adaptive_bs = batch_size
    
    # Logging
    log_fields = [
        "round", "val_acc", "val_loss", "E_used", "C_used", "M_used", "T_used",
        "lam_E", "lam_C", "lam_M", "lam_T", "k_eff", "steps_eff", "bs_eff", "comp_level"
    ]
    log_rows = []
    
    # Training loop
    for round_idx in range(rounds + 1):
        # Evaluate
        val_acc, val_loss, _ = evaluate_model(model, valid_data, device)
        
        # Default values for logging
        k_used = k_unfreeze
        steps_used = local_steps
        bs_used = batch_size
        comp_level = 0
        
        if round_idx >= 1:
            # Sample clients
            num_clients = min(clients_per_round, len(clients_data))
            selected_clients = random.sample(range(len(clients_data)), k=num_clients)
            
            # Apply CAFL policy
            if method == "cafl":
                lam_E, lam_C, lam_M, lam_T = dual_optimizer.get_duals()
                k_used, steps_used, bs_used, comp_level = compute_cafl_policy(
                    lam_E, lam_C, lam_M, lam_T,
                    adaptive_k, adaptive_steps, adaptive_bs
                )
                
                # Temperature-first constraints
                if T_smooth > 1.00 * T_budget:
                    bs_used = max(8, min(bs_used, 8))
                if T_smooth > 1.05 * T_budget:
                    steps_used = max(int(0.85 * BASE_STEPS), int(0.9 * steps_used))
            
            # Apply layer freezing
            apply_layer_freezing(model, k_used)
            apply_layer_freezing(worker_model, k_used)
            
            # Get current server state
            server_state = get_model_state(model)
            
            # Token preservation through gradient accumulation
            target_tokens = BASE_STEPS * BASE_BS
            phys_bs = max(8, min(bs_used, 8))
            steps_used = max(int(0.8 * BASE_STEPS), steps_used)
            grad_accum = max(1, math.ceil(target_tokens / max(1, steps_used * phys_bs)))
            
            # Communication setup
            bytes_per_float = get_compression_bytes_per_float(comp_level)
            sparsity = 0.8 if k_used > 2 else 0.6
            
            # Client updates
            updates = []
            round_energy = round_comm = round_memory = round_temp = 0.0
            
            for client_idx, client_id in enumerate(selected_clients):
                stats, delta = client_local_update(
                    worker_model,
                    server_state,
                    clients_data[client_id],
                    steps_used,
                    phys_bs,
                    learning_rate,
                    device,
                    block_size,
                    energy_scale=energy_scale,
                    use_amp=use_amp,
                    seed=seed + client_idx,
                    grad_accumulation=grad_accum,
                    base_steps_ref=BASE_STEPS,
                    base_bs_ref=BASE_BS
                )
                
                # Estimate communication cost
                comm_bytes = estimate_communication_bytes(delta, sparsity, bytes_per_float)
                
                # Move delta to CPU to save memory
                updates.append({k: v.detach().cpu() for k, v in delta.items()})
                
                # Accumulate resource usage
                round_energy += stats["E_used"]
                round_comm += comm_bytes
                round_memory += stats["M_used"]
                round_temp += stats["T_used"]
            
            # Aggregate updates
            apply_aggregated_update(model, updates)
            
            # Average resource usage
            num_selected = max(1, len(selected_clients))
            E_avg = round_energy / num_selected
            C_avg = round_comm / num_selected
            M_avg = round_memory / num_selected
            T_avg = round_temp / num_selected
            
            # Smooth with EMA
            E_smooth = ema_beta * E_smooth + (1 - ema_beta) * E_avg
            C_smooth = ema_beta * C_smooth + (1 - ema_beta) * C_avg
            M_smooth = ema_beta * M_smooth + (1 - ema_beta) * M_avg
            T_smooth = ema_beta * T_smooth + (1 - ema_beta) * T_avg
            
            # Update dual variables for CAFL
            if method == "cafl":
                dual_optimizer.update(
                    E_smooth, C_smooth, M_smooth, T_smooth,
                    E_budget, C_budget, M_budget, T_budget
                )
                
                # Recovery mechanism when safely under budgets
                margin = 0.90
                if (E_smooth <= margin * E_budget and 
                    C_smooth <= margin * C_budget and
                    M_smooth <= margin * M_budget and
                    T_smooth <= margin * T_budget):
                    # Gradually increase capacity
                    adaptive_k = min(adaptive_k + 1, len(model.blocks))
                    adaptive_steps = min(adaptive_steps + 10, int(1.1 * BASE_STEPS))
                    dual_optimizer.scale_duals(0.85)
                
                # Hard clamp if still exceeding budgets
                tolerance = 0.02
                over_budget = (
                    E_smooth > (1 + tolerance) * E_budget or
                    C_smooth > (1 + tolerance) * C_budget or
                    M_smooth > (1 + tolerance) * M_budget or
                    T_smooth > (1 + tolerance) * T_budget
                )
                
                if over_budget:
                    adaptive_k = max(1, min(adaptive_k, 2))
                    adaptive_steps = max(10, min(adaptive_steps, 30))
                    adaptive_bs = max(8, min(adaptive_bs, 8))
                    dual_optimizer.scale_duals(1.2)
        
        # Log metrics
        lam_E, lam_C, lam_M, lam_T = dual_optimizer.get_duals()
        log_rows.append({
            "round": round_idx,
            "val_acc": val_acc,
            "val_loss": val_loss,
            "E_used": E_smooth,
            "C_used": C_smooth,
            "M_used": M_smooth,
            "T_used": T_smooth,
            "lam_E": lam_E,
            "lam_C": lam_C,
            "lam_M": lam_M,
            "lam_T": lam_T,
            "k_eff": k_used,
            "steps_eff": steps_used,
            "bs_eff": bs_used,
            "comp_level": comp_level
        })
    
    # Save metrics to CSV
    csv_path = f"{save_prefix}_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_fields)
        writer.writeheader()
        writer.writerows(log_rows)
    
    # Convert to column-wise dictionary
    logs = {field: [row[field] for row in log_rows] for field in log_fields}
    
    return logs
