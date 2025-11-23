"""Utility functions for imitation learning pipeline.

This module contains helper functions for:
- Reproducibility (seed setting)
- Gem combination encoding (take3, take2, removal)
- Mask generation for conditional loss computation
- Metric computation for evaluation
- Visualization of training results
"""

import random
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix

from src.splendor.board import Board
from src.splendor.constants import BUILD, RESERVE, TOKENS
from src.splendor.move import Move
from src.splendor.cards import Card

# Add parent directory to path to access utils package
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.state_reconstruction import reconstruct_board_from_csv_row


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across all libraries.

    This function ensures deterministic behavior by setting seeds for:
    - Python's random module
    - NumPy's random generator
    - PyTorch's CPU and GPU random generators
    - PyTorch's cuDNN backend (disabling non-deterministic algorithms)

    Args:
        seed: Random seed value (typically 42 for reproducibility)

    Example:
        >>> set_seed(42)
        >>> # All subsequent random operations will be deterministic

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior (may impact performance slightly)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def generate_gem_take3_classes() -> Tuple[
    Dict[int, Tuple[str, ...]],
    Dict[Tuple[str, ...], int],
]:
    """Generate bidirectional mappings for gem_take3 action encoding.

    This handles choosing 0, 1, 2, or 3 gems from 5 colors (white, blue, green, red, black).
    Total of 26 classes:
    - Class 0: no gems (empty tuple)
    - Classes 1-5: single gems (5 combinations)
    - Classes 6-15: two gems (10 combinations)
    - Classes 16-25: three gems (10 combinations)

    Returns:
        Tuple containing:
        - class_to_combo: Dict mapping class index to tuple of gem color strings
        - combo_to_class: Dict mapping tuple of gem color strings to class index

    Example:
        >>> class_to_combo, combo_to_class = generate_gem_take3_classes()
        >>> class_to_combo[0]
        ()
        >>> class_to_combo[1]
        ('white',)
        >>> combo_to_class[('white', 'blue')]
        6

    """
    colors = ["white", "blue", "green", "red", "black"]
    class_to_combo: Dict[int, Tuple[str, ...]] = {}
    combo_to_class: Dict[Tuple[str, ...], int] = {}

    class_idx = 0

    # Class 0: no gems (empty tuple)
    class_to_combo[class_idx] = tuple()
    combo_to_class[tuple()] = class_idx
    class_idx += 1

    # Classes 1-5: single gems
    for color in colors:
        combo = (color,)
        class_to_combo[class_idx] = combo
        combo_to_class[combo] = class_idx
        class_idx += 1

    # Classes 6-15: two gems (10 combinations)
    for combo in combinations(colors, 2):
        class_to_combo[class_idx] = combo
        combo_to_class[combo] = class_idx
        class_idx += 1

    # Classes 16-25: three gems (10 combinations)
    for combo in combinations(colors, 3):
        class_to_combo[class_idx] = combo
        combo_to_class[combo] = class_idx
        class_idx += 1

    return class_to_combo, combo_to_class


def generate_gem_removal_classes() -> Tuple[
    Dict[int, Tuple[int, ...]],
    Dict[Tuple[int, ...], int],
]:
    """Generate bidirectional mappings for gem_removed action encoding.

    This handles removing 0-3 gems total from 6 types (white, blue, green, red, black, gold).
    Each gem type can have 0-3 removed, and total removal is ≤ 3.

    Returns:
        Tuple containing:
        - class_to_removal: Dict mapping class index to tuple of 6 counts (w,b,g,r,bl,go)
        - removal_to_class: Dict mapping tuple of 6 counts to class index

    Example:
        >>> class_to_removal, removal_to_class = generate_gem_removal_classes()
        >>> class_to_removal[0]
        (0, 0, 0, 0, 0, 0)  # No removal
        >>> removal_to_class[(1, 0, 0, 0, 0, 0)]
        1  # Remove 1 white

    """
    class_to_removal: Dict[int, Tuple[int, ...]] = {}
    removal_to_class: Dict[Tuple[int, ...], int] = {}

    class_idx = 0

    # Enumerate all valid combinations: sum ≤ 3, each count ≤ 3
    for white in range(4):  # 0-3
        for blue in range(4):
            for green in range(4):
                for red in range(4):
                    for black in range(4):
                        for gold in range(4):
                            total = white + blue + green + red + black + gold
                            if total <= 3:
                                removal = (white, blue, green, red, black, gold)
                                class_to_removal[class_idx] = removal
                                removal_to_class[removal] = class_idx
                                class_idx += 1

    return class_to_removal, removal_to_class


def encode_gem_take3(row: pd.Series, combo_to_class: Dict[Tuple[str, ...], int]) -> int:
    """Encode gem_take3 binary columns to single class index.

    Args:
        row: DataFrame row containing gem_take3_white through gem_take3_black columns
        combo_to_class: Mapping from gem tuple to class index

    Returns:
        Class index (0-25) if gems are selected, -1 if not applicable (all 0/NaN)

    Example:
        >>> combo_to_class = generate_gem_take3_classes()[1]
        >>> row = pd.Series({'gem_take3_white': 1, 'gem_take3_blue': 1,
        ...                  'gem_take3_green': 0, 'gem_take3_red': 0,
        ...                  'gem_take3_black': 0})
        >>> encode_gem_take3(row, combo_to_class)
        6  # Class for (white, blue)

    """
    colors = ["white", "blue", "green", "red", "black"]
    selected_gems = []

    for color in colors:
        col_name = f"gem_take3_{color}"
        if row[col_name] == 1:
            selected_gems.append(color)

    # If no gems selected, return -1 (not applicable)
    if not selected_gems:
        return -1

    # Convert list to sorted tuple and look up class
    gems_tuple = tuple(selected_gems)
    return combo_to_class.get(gems_tuple, -1)


def encode_gem_take2(row: pd.Series) -> int:
    """Encode gem_take2 binary columns to single class index.

    Args:
        row: DataFrame row containing gem_take2_white through gem_take2_black columns

    Returns:
        Class index (0-4) for the color selected, -1 if not applicable
        Mapping: white=0, blue=1, green=2, red=3, black=4

    Example:
        >>> row = pd.Series({'gem_take2_white': 0, 'gem_take2_blue': 0,
        ...                  'gem_take2_green': 1, 'gem_take2_red': 0,
        ...                  'gem_take2_black': 0})
        >>> encode_gem_take2(row)
        2  # Green

    """
    colors = ["white", "blue", "green", "red", "black"]

    for idx, color in enumerate(colors):
        col_name = f"gem_take2_{color}"
        if row[col_name] == 2:
            return idx

    # If no gem selected, return -1
    return -1


def encode_gems_removed(
    row: pd.Series,
    removal_to_class: Dict[Tuple[int, ...], int],
) -> int:
    """Encode gems_removed count columns to single class index.

    Args:
        row: DataFrame row containing gems_removed_white through gems_removed_gold columns
        removal_to_class: Mapping from 6-tuple of counts to class index

    Returns:
        Class index for the removal pattern (0 for no removal)

    Example:
        >>> removal_to_class = generate_gem_removal_classes()[1]
        >>> row = pd.Series({'gems_removed_white': 1, 'gems_removed_blue': 0,
        ...                  'gems_removed_green': 0, 'gems_removed_red': 0,
        ...                  'gems_removed_black': 0, 'gems_removed_gold': 0})
        >>> encode_gems_removed(row, removal_to_class)
        1  # Class for removing 1 white

    """
    colors = ["white", "blue", "green", "red", "black", "gold"]
    counts = tuple(int(row[f"gems_removed_{color}"]) for color in colors)

    return removal_to_class.get(counts, 0)


def get_action_type_mask(action_types: np.ndarray, target_type: int) -> np.ndarray:
    """Create boolean mask for samples matching target action type.

    Args:
        action_types: Array of action type labels (0=build, 1=reserve, 2=take2, 3=take3)
        target_type: Action type to filter for

    Returns:
        Boolean array where True indicates sample matches target_type

    Example:
        >>> action_types = np.array([0, 1, 0, 2, 3])
        >>> get_action_type_mask(action_types, 0)
        array([ True, False,  True, False, False])

    """
    return action_types == target_type


def get_valid_label_mask(labels: np.ndarray) -> np.ndarray:
    """Create boolean mask excluding -1 values (not applicable labels).

    Args:
        labels: Array of labels where -1 indicates "not applicable"

    Returns:
        Boolean array where True indicates valid label (!= -1)

    Example:
        >>> labels = np.array([5, -1, 3, -1, 0])
        >>> get_valid_label_mask(labels)
        array([ True, False,  True, False,  True])

    """
    return labels != -1


def combine_masks(*masks: np.ndarray) -> np.ndarray:
    """Combine multiple boolean masks using logical AND.

    Args:
        *masks: Variable number of boolean arrays to combine

    Returns:
        Boolean array that is True only where all input masks are True

    Example:
        >>> mask1 = np.array([True, True, False, True])
        >>> mask2 = np.array([True, False, False, True])
        >>> combine_masks(mask1, mask2)
        array([ True, False, False,  True])

    """
    if not masks:
        raise ValueError("At least one mask required")

    result = masks[0].copy()
    for mask in masks[1:]:
        result = np.logical_and(result, mask)

    return result


def compute_accuracy(
    predictions: np.ndarray,
    labels: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """Compute classification accuracy, optionally with masking.

    Args:
        predictions: Array of predicted class indices
        labels: Array of true class indices
        mask: Optional boolean mask to filter samples (e.g., exclude -1 labels)

    Returns:
        Accuracy as float between 0 and 1

    Example:
        >>> predictions = np.array([0, 1, 2, 0, 1])
        >>> labels = np.array([0, 1, 0, 0, -1])
        >>> mask = labels != -1
        >>> compute_accuracy(predictions, labels, mask)
        0.75  # 3 out of 4 valid predictions correct

    """
    if mask is not None:
        predictions = predictions[mask]
        labels = labels[mask]

    if len(labels) == 0:
        return 0.0

    return np.mean(predictions == labels)


def compute_confusion_matrix(
    predictions: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Compute confusion matrix for multi-class classification.

    Args:
        predictions: Array of predicted class indices
        labels: Array of true class indices
        num_classes: Total number of classes

    Returns:
        Confusion matrix of shape (num_classes, num_classes)
        Entry [i, j] = count of samples with true class i predicted as j

    Example:
        >>> predictions = np.array([0, 1, 2, 0])
        >>> labels = np.array([0, 1, 0, 0])
        >>> compute_confusion_matrix(predictions, labels, 3)
        array([[2, 0, 1],
               [0, 1, 0],
               [0, 0, 0]])

    """
    return sklearn_confusion_matrix(
        labels,
        predictions,
        labels=list(range(num_classes)),
    )


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    val_accuracies: Dict[str, List[float]],
    save_path: str,
) -> None:
    """Plot training and validation curves.

    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        val_accuracies: Dict mapping head name to list of accuracies per epoch
        save_path: Path to save the figure

    Example:
        >>> plot_training_curves([0.5, 0.3, 0.2], [0.6, 0.4, 0.3],
        ...                      {'action_type': [0.6, 0.7, 0.8]},
        ...                      'logs/training_curves.png')

    """
    epochs = list(range(1, len(train_losses) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Loss plot
    ax1.plot(epochs, train_losses, label="Train Loss", marker="o")
    ax1.plot(epochs, val_losses, label="Val Loss", marker="s")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy plot
    for head_name, accuracies in val_accuracies.items():
        ax2.plot(epochs, accuracies, label=head_name, marker="o")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Validation Accuracy by Head")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: str,
) -> None:
    """Plot confusion matrix as heatmap.

    Args:
        cm: Confusion matrix of shape (num_classes, num_classes)
        class_names: List of class names for axis labels
        save_path: Path to save the figure

    Example:
        >>> cm = np.array([[100, 10], [20, 80]])
        >>> plot_confusion_matrix(cm, ['BUILD', 'RESERVE'], 'logs/confusion.png')

    """
    plt.figure(figsize=(10, 8))

    # Normalize to percentages
    cm_normalized = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-10) * 100

    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Percentage (%)"},
    )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix (Normalized by True Class)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_per_class_accuracy(accuracies: Dict[str, float], save_path: str) -> None:
    """Plot per-head accuracy as bar chart.

    Args:
        accuracies: Dict mapping head name to accuracy value
        save_path: Path to save the figure

    Example:
        >>> accuracies = {'action_type': 0.82, 'card_selection': 0.65}
        >>> plot_per_class_accuracy(accuracies, 'logs/accuracies.png')

    """
    plt.figure(figsize=(12, 6))

    heads = list(accuracies.keys())
    values = [accuracies[head] for head in heads]

    bars = plt.bar(range(len(heads)), values, color="steelblue", alpha=0.8)

    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, values)):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.xticks(range(len(heads)), heads, rotation=45, ha="right")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy by Prediction Head")
    plt.ylim(0, 1.0)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def get_num_gem_removal_classes() -> int:
    """Calculate the total number of gem removal classes.

    Returns:
        Number of valid removal combinations

    Example:
        >>> get_num_gem_removal_classes()
        84  # Or whatever the actual count is

    """
    _, removal_to_class = generate_gem_removal_classes()
    return len(removal_to_class)


# ============================================================================
# MASK GENERATION FUNCTIONS FOR LEGAL MOVE MASKING
# ============================================================================


def get_mask_from_move_action_type(move: Move, mask: np.ndarray) -> None:
    """Update action_type mask for a single move.

    Args:
        move: Single Move object
        mask: Binary mask array to update in-place (shape: 4)
    """
    # Early return if mask is already all ones (all actions legal)
    if np.all(mask == 1):
        return

    if move.actionType == BUILD:
        mask[0] = 1
    elif move.actionType == RESERVE:
        mask[1] = 1
    elif move.actionType == TOKENS:
        # Differentiate TAKE2 vs TAKE3 based on token pattern
        tokens = move.action  # List of 6 integers
        non_zero_indices = [i for i in range(5) if tokens[i] > 0]  # Exclude gold

        # TAKE2 pattern: exactly one color with value 2
        if len(non_zero_indices) == 1 and tokens[non_zero_indices[0]] == 2:
            mask[2] = 1  # TAKE2
        else:
            mask[3] = 1  # TAKE3


def get_mask_from_move_card_selection(
    move: Move, board: Board, mask: np.ndarray
) -> None:
    """Update card_selection mask for a single move.

    Args:
        move: Single Move object
        board: Board object (needed to map Card objects to indices)
        mask: Binary mask array to update in-place (shape: 15)
    """
    if move.actionType == BUILD:
        card = move.action  # Card object
        current_player = board.players[board.currentPlayer]

        # Search visible cards (0-11)
        # Note: card.lvl is 1, 2, or 3, but displayedCards is indexed [0, 1, 2]
        card_idx = 0
        for level in range(3):
            for displayed_card in board.displayedCards[level]:
                if displayed_card is card:  # Identity comparison
                    mask[card_idx] = 1
                    return
                else:
                    card_idx += 1

        # Search reserved cards (12-14)
        for reserved_idx, reserved_card in enumerate(current_player.reserved):
            if reserved_card is card:
                mask[12 + reserved_idx] = 1
                return

        raise ValueError("we did not find the card that the player should build in the cards available to him")

def get_mask_from_move_card_reservation(
    move: Move, board: Board, mask: np.ndarray
) -> None:
    """Update card_reservation mask for a single move.

    Args:
        move: Single Move object
        board: Board object (needed to map Card objects to indices)
        mask: Binary mask array to update in-place (shape: 15)
    """
    if move.actionType == RESERVE:
        action = move.action

        if isinstance(action, int):
            # Top deck reservation: action is deck level (1, 2, or 3)
            # Map to indices 12, 13, 14
            card_idx = 11 + action  # level 1->12, 2->13, 3->14
        else:
            # Reserve from visible card: action is Card object
            card: Card = action
            card_idx = 0
            for level in range(3):
                for displayed_card in board.displayedCards[level]:
                    if displayed_card is card:  # Identity comparison
                        mask[card_idx] = 1
                        return
                    else:
                        card_idx += 1


        if card_idx is not None:
            mask[card_idx] = 1


def is_take2_action(tokens: List[int]) -> bool:
    """Check if token pattern represents a TAKE2 action.
    
    Args:
        tokens: List of 6 integers representing token counts
        
    Returns:
        True if this is a TAKE2 action (exactly one color with value 2)
    """
    non_zero_indices = [i for i in range(5) if tokens[i] > 0]  # Exclude gold
    return len(non_zero_indices) == 1 and tokens[non_zero_indices[0]] == 2


def get_mask_from_move_gem_take3(
        move: Move, combo_to_class: Dict[Tuple[str, ...], int], mask: np.ndarray
) -> None:
    """Update gem_take3 mask for a single move.

    Args:
        move: Single Move object
        combo_to_class: Mapping from tuple of color names to class index
        mask: Binary mask array to update in-place (shape: 26)
    """
    if np.all(mask == 1):
        return
    colors = ["white", "blue", "green", "red", "black"]

    if move.actionType == TOKENS:
        tokens = move.action  # List of 6 integers

        # Check if it's TAKE3 (NOT TAKE2)
        if not is_take2_action(tokens):
            # TAKE3: Multiple colors with value 1
            selected_colors = [colors[i] for i in range(5) if tokens[i] > 0]
            gems_tuple = tuple(selected_colors)

            class_idx = combo_to_class.get(gems_tuple, -1)
            if class_idx != -1:
                mask[class_idx] = 1


def get_mask_from_move_gem_take2(move: Move, mask: np.ndarray) -> None:
    """Update gem_take2 mask for a single move.

    Args:
        move: Single Move object
        mask: Binary mask array to update in-place (shape: 5)
    """
    if np.all(mask == 1):
        return
    if move.actionType == TOKENS:
        tokens = move.action  # List of 6 integers

        # Check if it's TAKE2: exactly one color with value 2
        if is_take2_action(tokens):
            non_zero_indices = [i for i in range(5) if tokens[i] > 0]
            color_idx = non_zero_indices[0]
            mask[color_idx] = 1


def get_mask_from_move_noble(move: Move, board: Board, mask: np.ndarray) -> None:
    """Update noble mask for a single move.

    Args:
        move: Single Move object
        board: Board object (needed to map Character objects to indices)
        mask: Binary mask array to update in-place (shape: 5)
    """
    if move.character is not None:
        # Find character index in board.characters
        for idx, character in enumerate(board.characters):
            if character is move.character:
                mask[idx] = 1
                break


def get_mask_from_move_gems_removed(
        move: Move, removal_to_class: Dict[Tuple[int, ...], int], mask: np.ndarray
) -> None:
    """Update gems_removed mask for a single move.

    Args:
        move: Single Move object
        removal_to_class: Mapping from tuple of 6 counts to class index
        mask: Binary mask array to update in-place (shape: 84)
    """
    removal_tuple = tuple(move.tokensToRemove)
    class_idx = removal_to_class.get(removal_tuple, -1)

    if class_idx != -1:
        mask[class_idx] = 1


def generate_all_masks_from_row(row: Dict) -> Dict[str, np.ndarray]:
    """Generate all 7 legal action masks from a CSV row.

    This is the main entry point for mask generation. It:
    1. Reconstructs the board state from CSV features
    2. Calls board.getMoves() to get legal moves
    3. Converts moves to masks for each prediction head

    Args:
        row: CSV row as dictionary (from pandas DataFrame.to_dict('records'))

    Returns:
        Dict with keys: {action_type, card_selection, card_reservation,
                        gem_take3, gem_take2, noble, gems_removed}
        Each value is a binary mask array

    Example:
        >>> row = df.iloc[0].to_dict()
        >>> masks = generate_all_masks_from_row(row)
        >>> masks['action_type']  # [1, 0, 1, 1] for BUILD, TAKE2, TAKE3 legal
        >>> masks['card_selection'].shape  # (15,)

    Note:
        If reconstruction fails, returns all-ones masks (allow all actions)
        and logs the error. This ensures preprocessing continues even if
        a few samples have issues.

    """
    try:
        # Import here to avoid circular dependency

        # Reconstruct board state
        board = reconstruct_board_from_csv_row(row)

        # Get legal moves
        moves = board.getMoves()

        # Generate encoding lookups
        _, combo_to_class_take3 = generate_gem_take3_classes()
        _, removal_to_class = generate_gem_removal_classes()

        # Initialize masks
        masks = {
            "action_type": np.zeros(4, dtype=np.int8),
            "card_selection": np.zeros(15, dtype=np.int8),
            "card_reservation": np.zeros(15, dtype=np.int8),
            "gem_take3": np.zeros(26, dtype=np.int8),
            "gem_take2": np.zeros(5, dtype=np.int8),
            "noble": np.zeros(5, dtype=np.int8),
            "gems_removed": np.zeros(get_num_gem_removal_classes(), dtype=np.int8),
        }

        # Process each move once and update all masks
        for move in moves:
            get_mask_from_move_action_type(move, masks["action_type"])
            get_mask_from_move_card_selection(move, board, masks["card_selection"])
            get_mask_from_move_card_reservation(move, board, masks["card_reservation"])
            get_mask_from_move_gem_take3(move, combo_to_class_take3, masks["gem_take3"])
            get_mask_from_move_gem_take2(move, masks["gem_take2"])
            get_mask_from_move_noble(move, board, masks["noble"])
            get_mask_from_move_gems_removed(
                move, removal_to_class, masks["gems_removed"]
            )

        # Class 0 (no removal) should typically be legal
        if removal_to_class.get((0, 0, 0, 0, 0, 0)) is not None:
            masks["gems_removed"][0] = 1

        return masks

    except Exception as e:
        # Log error and return all-ones masks as fallback
        game_id = row.get("game_id", "unknown")
        turn_num = row.get("turn_number", "unknown")
        print(
            f"WARNING: Mask generation failed for game_id={game_id}, turn={turn_num}: {e}",
        )

        # Return all-ones masks (allow all actions) as fallback
        return {
            "action_type": np.ones(4, dtype=np.int8),
            "card_selection": np.ones(15, dtype=np.int8),
            "card_reservation": np.ones(15, dtype=np.int8),
            "gem_take3": np.ones(26, dtype=np.int8),
            "gem_take2": np.ones(5, dtype=np.int8),
            "noble": np.ones(5, dtype=np.int8),
            "gems_removed": np.ones(get_num_gem_removal_classes(), dtype=np.int8),
        }


if __name__ == "__main__":
    # Test gem combination encoding
    print("Testing gem combination encoding...")

    # Test gem_take3
    class_to_combo, combo_to_class = generate_gem_take3_classes()
    print(f"\nGem Take 3 Classes: {len(class_to_combo)}")
    print(f"Class 0: {class_to_combo[0]}")
    print(f"Class 1: {class_to_combo[1]}")
    print(f"Class 6: {class_to_combo[6]}")
    print(f"Class 16: {class_to_combo[16]}")
    assert len(class_to_combo) == 26, "Should have exactly 26 gem_take3 classes"

    # Test gem_removal
    class_to_removal, removal_to_class = generate_gem_removal_classes()
    print(f"\nGem Removal Classes: {len(class_to_removal)}")
    print(f"Class 0: {class_to_removal[0]}")
    print(f"Class 1: {class_to_removal[1]}")
    assert class_to_removal[0] == (0, 0, 0, 0, 0, 0), "Class 0 should be no removal"

    # Test mask functions
    action_types = np.array([0, 1, 0, 2, 3])
    mask = get_action_type_mask(action_types, 0)
    assert np.sum(mask) == 2, "Should have 2 BUILD actions"

    labels = np.array([5, -1, 3, -1, 0])
    valid_mask = get_valid_label_mask(labels)
    assert np.sum(valid_mask) == 3, "Should have 3 valid labels"

    combined = combine_masks(mask, valid_mask)
    assert np.sum(combined) == 2, "Should have 2 samples matching both masks"

    # Test accuracy computation
    predictions = np.array([0, 1, 2, 0, 1])
    labels = np.array([0, 1, 0, 0, -1])
    mask = labels != -1
    acc = compute_accuracy(predictions, labels, mask)
    assert 0.74 < acc < 0.76, f"Accuracy should be 0.75, got {acc}"

    print("\n✓ All utility function tests passed!")
