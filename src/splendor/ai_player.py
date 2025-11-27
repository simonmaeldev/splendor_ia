"""AI player using trained neural network models for Splendor.

This module provides the ModelPlayer class that loads a trained model and uses it
to select actions during gameplay. It integrates feature engineering, legal move
masking, and action decoding to translate game state into neural network predictions
and back into executable moves.

Architecture:
1. Load model from .pth checkpoint file
2. Convert Board object to CSV row format
3. Extract engineered features (reusing feature_engineering.py)
4. Generate legal action masks (reusing utils.py)
5. Apply masks to model logits before prediction
6. Decode multi-head predictions to Move object (using action_decoder.py)

Example usage:
    >>> model_player = ModelPlayer("data/models/baseline/best_model.pth")
    >>> move = model_player.get_action(board)
    >>> board.doMove(move)
"""

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

# Ensure src is in path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from imitation_learning.model import MultiHeadSplendorNet
from imitation_learning.feature_engineering import extract_all_features, get_all_feature_names
from imitation_learning.utils import generate_all_masks_from_row
from imitation_learning.parallel_processor import fill_nan_values_for_row, compact_cards_and_add_position_for_row
from .action_decoder import decode_predictions_to_move
from .board import Board
from .move import Move
import json


def load_model(model_path: str, device: str = 'cpu') -> tuple[MultiHeadSplendorNet, Dict]:
    """Load trained model from checkpoint file.

    Args:
        model_path: Path to .pth checkpoint file
        device: Device to load model on ('cpu' or 'cuda')

    Returns:
        Tuple of (model, config_dict)

    Raises:
        FileNotFoundError: If model_path doesn't exist
        KeyError: If checkpoint doesn't contain required keys

    Example:
        >>> model, config = load_model("data/models/baseline/best_model.pth")
        >>> model.eval()
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    # Load checkpoint (weights_only=False for compatibility with numpy objects)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Extract or infer config
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        # Try to infer config from model state dict
        if 'model_state_dict' not in checkpoint:
            raise KeyError(f"Checkpoint missing both 'config' and 'model_state_dict'. Available keys: {list(checkpoint.keys())}")

        state_dict = checkpoint['model_state_dict']

        # Infer dimensions from state dict keys
        # trunk.0.weight has shape (trunk_dim[0], input_dim)
        input_dim = state_dict['trunk.trunk.0.weight'].shape[1]

        # Infer trunk dimensions from sequential layers
        trunk_dims = []
        i = 0
        while f'trunk.trunk.{i}.weight' in state_dict:
            trunk_dims.append(state_dict[f'trunk.trunk.{i}.weight'].shape[0])
            i += 3  # Skip Linear, ReLU, Dropout

        # Infer head dimensions from action_type head
        head_dims = []
        i = 0
        while f'heads.action_type.head.{i}.weight' in state_dict:
            if i == 0:
                # First layer connects from trunk output
                pass
            else:
                head_dims.append(state_dict[f'heads.action_type.head.{i}.weight'].shape[1])
            i += 3  # Skip Linear, ReLU, Dropout

        # Infer num_classes from final layers (find the last linear layer for each head)
        head_names = ['action_type', 'card_selection', 'card_reservation', 'gem_take3', 'gem_take2', 'noble', 'gems_removed']
        num_classes = {}

        for head_name in head_names:
            # Find all weight keys for this head
            head_keys = [k for k in state_dict.keys() if k.startswith(f'heads.{head_name}.head.') and k.endswith('.weight')]
            # Find the largest index (last layer)
            max_idx = max([int(k.split('.')[3]) for k in head_keys])
            # Get output dimension from last layer
            last_key = f'heads.{head_name}.head.{max_idx}.weight'
            num_classes[head_name] = state_dict[last_key].shape[0]

        config = {
            'input_dim': input_dim,
            'trunk_dims': trunk_dims,
            'head_dims': head_dims if head_dims else [128, 64],  # Default if can't infer
            'dropout': 0.3,  # Default
            'num_classes': num_classes
        }

    # Create model
    model = MultiHeadSplendorNet(config)

    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise KeyError(f"Checkpoint missing 'model_state_dict'. Available keys: {list(checkpoint.keys())}")

    # Set to evaluation mode
    model.eval()

    return model, config


def board_to_csv_row(board: Board) -> Dict[str, float]:
    """Convert Board object to CSV row dictionary.

    This is the reverse of state_reconstruction.reconstruct_board_from_csv_row().
    It extracts all raw game state features that were originally exported by csv_exporter.py.

    Args:
        board: Board object from live gameplay

    Returns:
        Dictionary mapping CSV column names to values

    Example:
        >>> row = board_to_csv_row(board)
        >>> row['num_players']
        4.0
        >>> row['current_player']
        0.0

    Notes:
        - Current player is always at index 0 (rotated perspective)
        - Features are padded with NaN for missing players/cards/nobles
        - Deck counts come from len(board.decks[i])
    """
    current_player_idx = board.currentPlayer
    num_players = len(board.players)

    row: Dict[str, float] = {}

    # Game metadata
    row['num_players'] = float(num_players)
    row['current_player'] = 0.0  # Always 0 in rotated view
    row['turn_number'] = float(board.nbTurn)

    # Board tokens (gems_board_*)
    colors = ['white', 'blue', 'green', 'red', 'black', 'gold']
    for i, color in enumerate(colors):
        row[f'gems_board_{color}'] = float(board.tokens[i])

    # Deck remaining counts
    row['deck_level1_remaining'] = float(len(board.decks[0]))
    row['deck_level2_remaining'] = float(len(board.decks[1]))
    row['deck_level3_remaining'] = float(len(board.decks[2]))

    # Nobles (up to 5, padded with NaN)
    for noble_idx in range(5):
        prefix = f'noble{noble_idx}_'

        if noble_idx < len(board.characters):
            noble = board.characters[noble_idx]
            row[f'{prefix}vp'] = float(noble.vp)
            for i, color in enumerate(colors[:5]):  # No gold for nobles
                row[f'{prefix}req_{color}'] = float(noble.cost[i])
        else:
            # Pad with NaN
            row[f'{prefix}vp'] = float('nan')
            for color in colors[:5]:
                row[f'{prefix}req_{color}'] = float('nan')

    # Displayed cards (12 total: 3 levels × 4 cards)
    card_idx = 0
    for level in range(3):
        for position in range(4):
            prefix = f'card{card_idx}_'

            if position < len(board.displayedCards[level]):
                card = board.displayedCards[level][position]
                row[f'{prefix}vp'] = float(card.vp)
                row[f'{prefix}level'] = float(card.lvl)

                # Card costs
                for i, color in enumerate(colors[:5]):
                    row[f'{prefix}cost_{color}'] = float(card.cost[i])

                # Card bonus (one-hot encoding)
                for i, color in enumerate(colors[:5]):
                    row[f'{prefix}bonus_{color}'] = 1.0 if card.bonus == i else 0.0
            else:
                # Pad with 0 (fillna behavior)
                row[f'{prefix}vp'] = 0.0
                row[f'{prefix}level'] = 0.0
                for color in colors[:5]:
                    row[f'{prefix}cost_{color}'] = 0.0
                    row[f'{prefix}bonus_{color}'] = 0.0

            card_idx += 1

    # Players (rotated so current player is at index 0)
    for player_idx in range(4):
        prefix = f'player{player_idx}_'

        # Calculate actual player index (with rotation)
        if player_idx < num_players:
            actual_player_idx = (current_player_idx + player_idx) % num_players
            player = board.players[actual_player_idx]

            # Player position (absolute player number before rotation)
            row[f'player{player_idx}_position'] = float(actual_player_idx)

            # Player tokens
            for i, color in enumerate(colors):
                row[f'{prefix}gems_{color}'] = float(player.tokens[i])

            # Player VP and bonuses
            row[f'{prefix}vp'] = float(player.vp)
            for i, color in enumerate(colors[:5]):
                row[f'{prefix}bonus_{color}'] = float(player.reductions[i])

            # Reserved cards (up to 3)
            for reserved_idx in range(3):
                card_prefix = f'{prefix}reserved{reserved_idx}_'

                if reserved_idx < len(player.reserved):
                    card = player.reserved[reserved_idx]
                    row[f'{card_prefix}vp'] = float(card.vp)
                    row[f'{card_prefix}level'] = float(card.lvl)

                    for i, color in enumerate(colors[:5]):
                        row[f'{card_prefix}cost_{color}'] = float(card.cost[i])

                    for i, color in enumerate(colors[:5]):
                        row[f'{card_prefix}bonus_{color}'] = 1.0 if card.bonus == i else 0.0
                else:
                    # Pad with 0
                    row[f'{card_prefix}vp'] = 0.0
                    row[f'{card_prefix}level'] = 0.0
                    for color in colors[:5]:
                        row[f'{card_prefix}cost_{color}'] = 0.0
                        row[f'{card_prefix}bonus_{color}'] = 0.0

            # Built cards count
            row[f'{prefix}num_cards_built'] = float(len(player.built))

            # Nobles acquired count
            row[f'{prefix}num_nobles'] = float(len(player.characters))

        else:
            # Pad missing players with NaN
            row[f'player{player_idx}_position'] = float('nan')

            for color in colors:
                row[f'{prefix}gems_{color}'] = float('nan')

            row[f'{prefix}vp'] = float('nan')
            for color in colors[:5]:
                row[f'{prefix}bonus_{color}'] = float('nan')

            for reserved_idx in range(3):
                card_prefix = f'{prefix}reserved{reserved_idx}_'
                row[f'{card_prefix}vp'] = float('nan')
                row[f'{card_prefix}level'] = float('nan')
                for color in colors[:5]:
                    row[f'{card_prefix}cost_{color}'] = float('nan')
                    row[f'{card_prefix}bonus_{color}'] = float('nan')

            row[f'{prefix}num_cards_built'] = float('nan')
            row[f'{prefix}num_nobles'] = float('nan')

    return row


def preprocess_csv_row(row_dict: Dict) -> Dict:
    """Preprocess CSV row with same transformations as training pipeline.

    Applies:
    1. NaN filling (0 for features, keep NaN for labels)
    2. Card compaction and position indices

    Args:
        row_dict: Raw CSV row dictionary from board_to_csv_row

    Returns:
        Preprocessed row dictionary
    """
    # Fill NaN values
    filled_row = fill_nan_values_for_row(row_dict)

    # Compact cards and add position indices
    compacted_row = compact_cards_and_add_position_for_row(filled_row)

    return compacted_row


def create_onehot_features(row_dict: Dict) -> Dict[str, float]:
    """Create one-hot encoded features for categorical variables.

    This replicates the one-hot encoding done during training in merge_batches.py
    lines 387-406.

    Args:
        row_dict: Preprocessed row dictionary

    Returns:
        Dictionary of one-hot feature name -> value (0 or 1)
    """
    onehot_features = {}

    # One-hot encode current_player (4 values: 0, 1, 2, 3)
    current_player = int(row_dict.get('current_player', 0))
    for i in range(4):
        onehot_features[f'current_player_{i}'] = 1.0 if current_player == i else 0.0

    # One-hot encode num_players (3 values: 2, 3, 4)
    num_players = int(row_dict.get('num_players', 2))
    for n in [2, 3, 4]:
        onehot_features[f'num_players_{n}'] = 1.0 if num_players == n else 0.0

    # One-hot encode player positions (4 players × 4 positions)
    for player_idx in range(4):
        position_col = f'player{player_idx}_position'
        if position_col in row_dict:
            position = int(row_dict[position_col])
            for pos in range(4):
                onehot_features[f'{position_col}_{pos}'] = 1.0 if position == pos else 0.0

    return onehot_features


def get_csv_feature_columns(row_dict: Dict) -> List[str]:
    """Get ordered list of CSV feature column names.

    This replicates the logic from identify_column_groups in data_preprocessing.py,
    excluding metadata and labels but including position columns added by compaction.

    Args:
        row_dict: Preprocessed CSV row dictionary

    Returns:
        Ordered list of CSV feature column names
    """
    # Metadata columns to exclude (matches identify_column_groups)
    metadata_cols = {'game_id', 'turn_number', 'current_player', 'num_players'}
    metadata_cols.update([f'player{i}_position' for i in range(4)])

    # Label columns to exclude (matches identify_column_groups)
    label_cols = {
        'action_type',
        'card_selection', 'card_reservation', 'noble_selection',
    }
    # Add gem label columns
    for color in ['white', 'blue', 'green', 'red', 'black']:
        label_cols.add(f'gem_take3_{color}')
        label_cols.add(f'gem_take2_{color}')
    for color in ['white', 'blue', 'green', 'red', 'black', 'gold']:
        label_cols.add(f'gems_removed_{color}')

    # Feature columns are everything else (in sorted order for consistency)
    feature_cols = []
    for col in sorted(row_dict.keys()):
        if col not in metadata_cols and col not in label_cols:
            feature_cols.append(col)

    return feature_cols


def apply_legal_move_masking(
    logits: Dict[str, torch.Tensor],
    masks: Dict[str, np.ndarray]
) -> Dict[str, torch.Tensor]:
    """Apply legal action masks to model logits.

    This ensures the model can only select legal actions by applying a strong
    negative penalty (-1e10) to illegal actions before computing argmax.

    Args:
        logits: Dict mapping head name to logits tensor (shape: [num_classes])
        masks: Dict mapping head name to binary mask array (shape: [num_classes])
                1 = legal action, 0 = illegal action

    Returns:
        Dict mapping head name to masked logits tensor

    Example:
        >>> logits = {'action_type': torch.tensor([0.5, 1.2, 0.8, 0.3])}
        >>> masks = {'action_type': np.array([1, 0, 1, 1])}
        >>> masked = apply_legal_move_masking(logits, masks)
        >>> masked['action_type'].argmax()
        tensor(2)  # Selects legal action with highest logit
    """
    masked_logits = {}

    for head_name, logit_tensor in logits.items():
        if head_name in masks:
            # Convert mask to tensor
            mask_tensor = torch.from_numpy(masks[head_name]).float()

            # Apply strong negative penalty to illegal actions
            # Legal actions (mask=1) → no change
            # Illegal actions (mask=0) → subtract 1e10
            masked_logit = logit_tensor + (1 - mask_tensor) * -1e10

            masked_logits[head_name] = masked_logit
        else:
            # No mask for this head, use original logits
            masked_logits[head_name] = logit_tensor

    return masked_logits


class ModelPlayer:
    """AI player that uses a trained neural network for action selection.

    This class handles:
    - Model loading and caching
    - Board state to feature conversion
    - Legal move masking
    - Model inference
    - Prediction decoding to Move objects

    Example:
        >>> player = ModelPlayer("data/models/baseline/best_model.pth")
        >>> move = player.get_action(board)
        >>> print(f"Predicted action: {move}")
    """

    def __init__(self, model_path: str, device: str = 'cpu'):
        """Initialize ModelPlayer with a trained model.

        Args:
            model_path: Path to .pth checkpoint file
            device: Device to run inference on ('cpu' or 'cuda')

        Raises:
            FileNotFoundError: If model_path doesn't exist
            KeyError: If checkpoint is malformed
        """
        self.model_path = model_path
        self.device = device

        # Load model
        self.model, self.config = load_model(model_path, device)

        # Cache feature names for consistent ordering
        self.feature_names = get_all_feature_names()

    def get_model_input_features(self, board: Board) -> torch.Tensor:
        """Convert board state to model input tensor.

        This method replicates the exact feature engineering pipeline used during training
        in merge_batches.py process_merged_data(). The feature order is:
        1. CSV features (after preprocessing and excluding metadata/labels/positions)
        2. One-hot encodings (current_player, num_players, player positions)
        3. turn_number
        4. Strategic/engineered features (893 features from extract_all_features)

        Args:
            board: Board object from live gameplay

        Returns:
            Feature tensor of shape (1, input_dim) ready for model inference

        Example:
            >>> features = player.get_model_input_features(board)
            >>> features.shape
            torch.Size([1, 1308])
        """
        # Convert board to CSV row
        csv_row = board_to_csv_row(board)

        model_input_dim = self.config['input_dim']

        if model_input_dim < 500:
            # Model expects only CSV features (old format - backward compatibility)
            # Convert CSV row to ordered list
            feature_values = []
            df_row = pd.DataFrame([csv_row])
            feature_values = df_row.values.flatten().tolist()

            # Fill NaN with 0
            feature_values = [0.0 if isinstance(v, float) and pd.isna(v) else float(v) for v in feature_values]

        else:
            # Model expects CSV + strategic + one-hot features (new format)
            # This replicates the training pipeline in merge_batches.py
            # Order: CSV features -> Strategic features -> One-hot features

            # Step 1: Preprocess CSV row (NaN filling, card compaction, position indices)
            preprocessed_row = preprocess_csv_row(csv_row)

            # Step 2: Extract strategic features
            strategic_features = extract_all_features(preprocessed_row, board)

            # Step 3: Create one-hot encoded features
            onehot_features = create_onehot_features(preprocessed_row)

            # Step 4: Load feature column order from training (if available)
            # This ensures exact matching with training feature order
            feature_cols_path = Path('data/processed/feature_cols.json')
            if feature_cols_path.exists():
                # Use exact training feature order
                with open(feature_cols_path, 'r') as f:
                    training_feature_cols = json.load(f)

                # Extract features in exact training order
                feature_values = []
                for col in training_feature_cols:
                    if col in preprocessed_row:
                        # CSV feature
                        feature_values.append(float(preprocessed_row[col]))
                    elif col in strategic_features:
                        # Strategic feature
                        feature_values.append(float(strategic_features[col]))
                    elif col in onehot_features:
                        # One-hot feature
                        feature_values.append(float(onehot_features[col]))
                    else:
                        # Missing feature - use 0
                        feature_values.append(0.0)
            else:
                # Fallback: construct features in expected order
                # Order: CSV -> Strategic -> One-hot
                print(f"WARNING: feature_cols.json not found, using fallback feature ordering")

                csv_feature_cols = get_csv_feature_columns(preprocessed_row)
                csv_values = [float(preprocessed_row.get(col, 0.0)) for col in csv_feature_cols]

                strategic_values = []
                for name in self.feature_names:
                    value = strategic_features.get(name, 0.0)
                    if name not in csv_feature_cols:
                        strategic_values.append(value)

                onehot_values = [onehot_features[col] for col in sorted(onehot_features.keys())]

                feature_values = csv_values + strategic_values + onehot_values

            # Debug information
            total_features = len(feature_values)

            if total_features != model_input_dim:
                print(f"[ModelPlayer] Feature dimension mismatch:")
                print(f"  - Total features: {total_features}")
                print(f"  - Expected model input_dim: {model_input_dim}")
                print(f"  - Mismatch: {total_features - model_input_dim}")

                # Trim or pad if needed (should not happen after fix)
                if total_features > model_input_dim:
                    print(f"WARNING: Trimming features from {total_features} to {model_input_dim}")
                    feature_values = feature_values[:model_input_dim]
                elif total_features < model_input_dim:
                    print(f"WARNING: Padding features from {total_features} to {model_input_dim}")
                    feature_values.extend([0.0] * (model_input_dim - total_features))

        # Convert to tensor and add batch dimension
        feature_tensor = torch.tensor(feature_values, dtype=torch.float32).unsqueeze(0)

        return feature_tensor

    def get_action(self, board: Board) -> Move:
        """Get AI player's action for the current board state.

        This is the main entry point for the ModelPlayer. It:
        1. Extracts features from the board
        2. Generates legal action masks
        3. Runs model inference
        4. Applies masks to logits
        5. Decodes predictions to Move object

        Args:
            board: Current game state

        Returns:
            Move object representing the selected action

        Example:
            >>> move = player.get_action(board)
            >>> move.actionType in [BUILD, RESERVE, 2, 3]
            True
        """
        # Extract features
        features = self.get_model_input_features(board)

        # Generate legal action masks
        csv_row = board_to_csv_row(board)
        masks = generate_all_masks_from_row(csv_row, board)

        # Run model inference (no gradient computation needed)
        with torch.no_grad():
            logits = self.model(features)

            # Remove batch dimension from logits
            logits = {name: tensor.squeeze(0) for name, tensor in logits.items()}

            # Apply legal action masks
            masked_logits = apply_legal_move_masking(logits, masks)

            # Get predictions by taking argmax
            predictions = {
                name: tensor.argmax().item()
                for name, tensor in masked_logits.items()
            }

        # Decode predictions to Move object
        move = decode_predictions_to_move(predictions, board)

        return move


# Global cache for model players (keyed by model path)
_model_player_cache: Dict[str, ModelPlayer] = {}


def get_model_action(board: Board, model_path: str) -> Move:
    """Get AI action from trained model (ISMCTS-style interface).

    This function provides a similar interface to ISMCTS() for use in game loops.
    It caches ModelPlayer instances to avoid reloading the same model multiple times.

    Args:
        board: Current game state
        model_path: Path to .pth model checkpoint

    Returns:
        Move object representing the selected action

    Example:
        >>> # In game loop (similar to ISMCTS usage)
        >>> if player_type == "MODEL":
        ...     move = get_model_action(board, "data/models/baseline/best_model.pth")
        >>> else:
        ...     move = ISMCTS(board, itermax=1000)
        >>> board.doMove(move)
    """
    # Check cache
    if model_path not in _model_player_cache:
        _model_player_cache[model_path] = ModelPlayer(model_path)

    player = _model_player_cache[model_path]
    return player.get_action(board)


if __name__ == "__main__":
    # Test model loading and inference
    print("Testing ModelPlayer...")

    # Check if a test model exists
    test_model_path = Path("data/models/baseline/best_model.pth")

    if test_model_path.exists():
        print(f"\n✓ Found test model at {test_model_path}")

        try:
            player = ModelPlayer(str(test_model_path))
            print(f"✓ ModelPlayer initialized successfully")
            print(f"  - Model config: {player.config}")
            print(f"  - Feature count: {len(player.feature_names)}")
        except Exception as e:
            print(f"✗ Error initializing ModelPlayer: {e}")
    else:
        print(f"\n✗ Test model not found at {test_model_path}")
        print("  Run training first to create a model checkpoint")

    print("\n✓ ModelPlayer module test complete")
