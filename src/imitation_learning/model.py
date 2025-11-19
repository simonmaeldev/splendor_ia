"""
Multi-head neural network architecture for Splendor imitation learning.

This module implements a neural network with:
- Shared trunk that processes game state into a shared representation
- 7 specialized prediction heads for different action components
"""

from typing import Dict, List

import torch
import torch.nn as nn


class SharedTrunk(nn.Module):
    """
    Shared trunk network that processes game state into a shared representation.

    This trunk is shared across all prediction heads, allowing them to learn
    from a common feature representation.
    """

    def __init__(self, input_dim: int, trunk_dims: List[int], dropout: float = 0.3):
        """
        Initialize shared trunk.

        Args:
            input_dim: Dimension of input features
            trunk_dims: List of hidden layer dimensions (e.g., [512, 256])
            dropout: Dropout probability for regularization
        """
        super().__init__()

        layers = []
        prev_dim = input_dim

        for dim in trunk_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = dim

        self.trunk = nn.Sequential(*layers)
        self.output_dim = trunk_dims[-1] if trunk_dims else input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through trunk.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Shared representation of shape (batch_size, output_dim)
        """
        return self.trunk(x)


class PredictionHead(nn.Module):
    """
    Prediction head for a specific action component.

    Each head takes the shared trunk output and predicts among its set of classes.
    """

    def __init__(self, input_dim: int, head_dims: List[int], num_classes: int, dropout: float = 0.3):
        """
        Initialize prediction head.

        Args:
            input_dim: Dimension of trunk output
            head_dims: List of hidden layer dimensions (e.g., [128, 64])
            num_classes: Number of output classes for this head
            dropout: Dropout probability for regularization
        """
        super().__init__()

        layers = []
        prev_dim = input_dim

        for dim in head_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = dim

        # Final layer outputs raw logits (no activation)
        layers.append(nn.Linear(prev_dim, num_classes))

        self.head = nn.Sequential(*layers)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through head.

        Args:
            x: Trunk output of shape (batch_size, input_dim)

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        return self.head(x)


class MultiHeadSplendorNet(nn.Module):
    """
    Multi-head neural network for Splendor action prediction.

    Architecture:
    - Shared trunk processes game state
    - 7 independent heads predict different action components:
      1. action_type: Which type of action (BUILD/RESERVE/TAKE2/TAKE3)
      2. card_selection: Which card to build (if action_type==BUILD)
      3. card_reservation: Which card to reserve (if action_type==RESERVE)
      4. gem_take3: Which gems to take (if action_type==TAKE3)
      5. gem_take2: Which gem color to take 2 of (if action_type==TAKE2)
      6. noble: Which noble to select (if BUILD + noble available)
      7. gems_removed: Which gems to discard (if overflow)

    During training, heads are only queried conditionally based on action_type.
    During inference, you query heads based on the predicted action_type.
    """

    def __init__(self, config: Dict):
        """
        Initialize multi-head model.

        Args:
            config: Configuration dict with keys:
                - input_dim: Input feature dimension
                - trunk_dims: List of trunk hidden dimensions
                - head_dims: List of head hidden dimensions
                - dropout: Dropout probability
                - num_classes: Dict mapping head name to number of classes
        """
        super().__init__()

        self.config = config

        # Shared trunk
        self.trunk = SharedTrunk(
            input_dim=config['input_dim'],
            trunk_dims=config['trunk_dims'],
            dropout=config['dropout']
        )

        trunk_output_dim = self.trunk.output_dim

        # Prediction heads
        self.heads = nn.ModuleDict({
            'action_type': PredictionHead(
                trunk_output_dim,
                config['head_dims'],
                config['num_classes']['action_type'],
                config['dropout']
            ),
            'card_selection': PredictionHead(
                trunk_output_dim,
                config['head_dims'],
                config['num_classes']['card_selection'],
                config['dropout']
            ),
            'card_reservation': PredictionHead(
                trunk_output_dim,
                config['head_dims'],
                config['num_classes']['card_reservation'],
                config['dropout']
            ),
            'gem_take3': PredictionHead(
                trunk_output_dim,
                config['head_dims'],
                config['num_classes']['gem_take3'],
                config['dropout']
            ),
            'gem_take2': PredictionHead(
                trunk_output_dim,
                config['head_dims'],
                config['num_classes']['gem_take2'],
                config['dropout']
            ),
            'noble': PredictionHead(
                trunk_output_dim,
                config['head_dims'],
                config['num_classes']['noble'],
                config['dropout']
            ),
            'gems_removed': PredictionHead(
                trunk_output_dim,
                config['head_dims'],
                config['num_classes']['gems_removed'],
                config['dropout']
            ),
        })

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights using He initialization for ReLU networks."""
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through model.

        Args:
            x: Input state tensor of shape (batch_size, input_dim)

        Returns:
            Dict mapping head name to logits tensor of shape (batch_size, num_classes)

        Example:
            >>> model = MultiHeadSplendorNet(config)
            >>> x = torch.randn(32, 400)  # Batch of 32 samples
            >>> outputs = model(x)
            >>> outputs['action_type'].shape
            torch.Size([32, 4])
            >>> outputs['card_selection'].shape
            torch.Size([32, 15])
        """
        # Pass through shared trunk
        trunk_output = self.trunk(x)

        # Pass trunk output through each head independently
        outputs = {
            name: head(trunk_output)
            for name, head in self.heads.items()
        }

        return outputs

    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def print_summary(self):
        """Print model architecture summary."""
        print("\n" + "="*60)
        print("Multi-Head Splendor Network")
        print("="*60)

        print(f"\nShared Trunk:")
        print(f"  Input dim: {self.config['input_dim']}")
        print(f"  Hidden dims: {self.config['trunk_dims']}")
        print(f"  Output dim: {self.trunk.output_dim}")
        print(f"  Dropout: {self.config['dropout']}")

        print(f"\nPrediction Heads:")
        for name, head in self.heads.items():
            print(f"  {name}:")
            print(f"    Hidden dims: {self.config['head_dims']}")
            print(f"    Output classes: {head.num_classes}")

        total_params = self.get_num_parameters()
        print(f"\nTotal parameters: {total_params:,}")
        print("="*60 + "\n")


if __name__ == "__main__":
    # Test model architecture
    print("Testing MultiHeadSplendorNet...")

    # Create dummy config
    config = {
        'input_dim': 420,  # Example dimension after one-hot encoding
        'trunk_dims': [512, 256],
        'head_dims': [128, 64],
        'dropout': 0.3,
        'num_classes': {
            'action_type': 4,
            'card_selection': 15,
            'card_reservation': 15,
            'gem_take3': 26,
            'gem_take2': 5,
            'noble': 5,
            'gems_removed': 84
        }
    }

    # Create model
    model = MultiHeadSplendorNet(config)
    model.print_summary()

    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, config['input_dim'])

    print(f"Testing forward pass with batch_size={batch_size}...")
    outputs = model(x)

    print(f"\nOutput shapes:")
    for name, logits in outputs.items():
        print(f"  {name}: {logits.shape}")

    # Verify shapes
    assert outputs['action_type'].shape == (batch_size, 4)
    assert outputs['card_selection'].shape == (batch_size, 15)
    assert outputs['card_reservation'].shape == (batch_size, 15)
    assert outputs['gem_take3'].shape == (batch_size, 26)
    assert outputs['gem_take2'].shape == (batch_size, 5)
    assert outputs['noble'].shape == (batch_size, 5)
    assert outputs['gems_removed'].shape == (batch_size, 84)

    # Test CUDA if available
    if torch.cuda.is_available():
        print("\nTesting CUDA...")
        model_cuda = model.cuda()
        x_cuda = x.cuda()
        outputs_cuda = model_cuda(x_cuda)
        print(f"  CUDA forward pass successful!")
        print(f"  Device: {next(model_cuda.parameters()).device}")

    print("\n✓ Model architecture test passed!")
