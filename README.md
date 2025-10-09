# CAFL: Constraint-Aware Federated Learning

Constraint-Aware Federated Learning with token-budget preservation for improved accuracy under resource constraints.

## Features

- **Adaptive Resource Management**: Dynamically adjusts training parameters based on energy, communication, memory, and temperature budgets
- **Token-Budget Preservation**: Maintains effective training throughput through gradient accumulation
- **Dual Ascent Optimization**: Uses Lagrangian dual ascent to optimize constraint satisfaction
- **Modular Architecture**: Clean, maintainable codebase organized into logical modules

## Installation

```bash
# Clone the repository
git clone https://github.com/dzhengAP/cafl.git
cd cafl

# Install in development mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

## Project Structure

```
cafl/
├── src/cafl/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── transformer.py    # TinyCharGPT model
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py        # Data loading and preprocessing
│   ├── federated/
│   │   ├── __init__.py
│   │   ├── client.py         # Client-side training logic
│   │   ├── server.py         # Server aggregation
│   │   └── policy.py         # CAFL policy and constraints
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── helpers.py        # General utilities
│   │   ├── metrics.py        # Resource estimation
│   │   └── visualization.py  # Plotting functions
│   └── train.py              # Main training script
├── examples/
│   └── basic_training.py     # Example usage
├── tests/
│   └── test_*.py             # Unit tests
├── setup.py
├── requirements.txt
├── README.md
└── LICENSE
```

## Quick Start

```bash
# Run with default settings
cafl-train

# Run with custom parameters
cafl-train --rounds 10 --num_clients 20 --budget_energy 8.0e7

# Fast test run
cafl-train --fast

# Use CPU only
cafl-train --device cpu
```

## Usage Example

```python
from cafl.models import TinyCharGPT
from cafl.federated import run_federated_learning
from cafl.data import prepare_shakespeare_data

# Prepare data
clients_data, valid_data = prepare_shakespeare_data(num_clients=16)

# Create model
model = TinyCharGPT(vocab_size=95, block_size=128)

# Run federated learning
results = run_federated_learning(
    model=model,
    clients_data=clients_data,
    valid_data=valid_data,
    method="cafl",
    rounds=10,
    budgets={
        "energy": 6.0e7,
        "comm": 9.0e6,
        "memory": 0.55,
        "temp": 0.45
    }
)
```

## Command-Line Arguments

### Training Schedule

- `--rounds`: Number of federated rounds (default: 8)
- `--local_steps`: Local training steps per round (default: 60)
- `--batch_size`: Batch size for training (default: 16)
- `--lr`: Learning rate (default: 0.0025)
- `--clients_per_round`: Clients participating per round (default: 6)

### Model Configuration

- `--n_layer`: Number of transformer layers (default: 2)
- `--n_head`: Number of attention heads (default: 4)
- `--n_embd`: Embedding dimension (default: 128)

### Resource Budgets

- `--budget_energy`: Energy budget (default: 6.0e7)
- `--budget_comm`: Communication budget in bytes (default: 9.0e6)
- `--budget_memory`: Memory pressure budget (default: 0.55)
- `--budget_temp`: Temperature budget (default: 0.45)

### Other Options

- `--device`: Device to use (choices: cpu, cuda, auto)
- `--amp`: Enable mixed precision training
- `--fast`: Use tiny preset for quick testing
- `--seed`: Random seed (default: 42)

## Outputs

Training produces the following outputs in the specified directory:

- `fedavg_metrics.csv`: Metrics for baseline FedAvg
- `cafl_metrics.csv`: Metrics for CAFL method
- `acc.png`: Validation accuracy comparison
- `loss.png`: Validation loss comparison
- `energy.png`: Energy usage vs budget
- `comm.png`: Communication overhead vs budget
- `memory.png`: Memory pressure vs budget
- `temp.png`: Temperature proxy vs budget

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{zheng2025cafllconstraintawarefederatedlearning,
      title={CAFL-L: Constraint-Aware Federated Learning with Lagrangian Dual Optimization for On-Device Language Models}, 
      author={Dongqi Zheng and Wenjin Fu},
      year={2025},
      eprint={2510.03298},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.03298}, 
}
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or issues, please open an issue on GitHub or contact: your.email@example.com
