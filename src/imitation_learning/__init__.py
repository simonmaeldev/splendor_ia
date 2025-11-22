"""
Splendor Imitation Learning Module

This package implements supervised learning to train a multi-head neural network
that imitates MCTS (Monte Carlo Tree Search) decision-making behavior in Splendor.

The system processes game state representations and learns to predict the action
that MCTS would select, enabling faster inference than full tree search.
"""

__version__ = "0.1.0"

# Try to import torch-dependent modules, but don't fail if torch isn't installed
try:
    from .model import MultiHeadSplendorNet, SharedTrunk, PredictionHead
    from .dataset import SplendorDataset, create_dataloaders
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    # These will be None if torch is not installed
    MultiHeadSplendorNet = None
    SharedTrunk = None
    PredictionHead = None
    SplendorDataset = None
    create_dataloaders = None

from .utils import set_seed

__all__ = [
    "__version__",
    "set_seed",
]

if _HAS_TORCH:
    __all__.extend([
        "MultiHeadSplendorNet",
        "SharedTrunk",
        "PredictionHead",
        "SplendorDataset",
        "create_dataloaders",
    ])
