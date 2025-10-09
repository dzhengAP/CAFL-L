“”“CAFL: Constraint-Aware Federated Learning.”””

**version** = “0.1.0”

from cafl.models import TinyCharGPT
from cafl.data import prepare_shakespeare_data
from cafl.federated import run_federated_learning

**all** = [
“TinyCharGPT”,
“prepare_shakespeare_data”,
“run_federated_learning”,
]