"""wiring_tuning: predict neuronal visual tuning from EM synaptic wiring.

Paper 1 of the MICrONS function-from-wiring series. Provides a synthetic
benchmark (planted tuning-from-wiring signal), graph representations,
GNN and baseline models, a training harness, and a benchmark runner.
"""

from .graphs import ConnectomeGraph, sample_subgraph
from .simulate import make_synthetic_connectome, degree_preserving_rewire
from .models import GNNRegressor, BaselineMLP, DegreeBaseline
from .train import train_model, evaluate, split_masks, r2_score, pearson_corr
from .benchmark import run_benchmark

__all__ = [
    "ConnectomeGraph",
    "sample_subgraph",
    "make_synthetic_connectome",
    "degree_preserving_rewire",
    "GNNRegressor",
    "BaselineMLP",
    "DegreeBaseline",
    "train_model",
    "evaluate",
    "split_masks",
    "r2_score",
    "pearson_corr",
    "run_benchmark",
]

__version__ = "0.1.0"
