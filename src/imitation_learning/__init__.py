"""
Splendor Imitation Learning Module

This package implements supervised learning to train a multi-head neural network
that imitates MCTS (Monte Carlo Tree Search) decision-making behavior in Splendor.

The system processes game state representations and learns to predict the action
that MCTS would select, enabling faster inference than full tree search.
"""

__version__ = "0.1.0"

from .model import MultiHeadSplendorNet, SharedTrunk, PredictionHead
from .dataset import SplendorDataset, create_dataloaders
from .utils import set_seed

__all__ = [
    "__version__",
    "MultiHeadSplendorNet",
    "SharedTrunk",
    "PredictionHead",
    "SplendorDataset",
    "create_dataloaders",
    "set_seed",
]
