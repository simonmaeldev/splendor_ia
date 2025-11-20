"""
Utility functions for imitation learning pipeline.

This module contains helper functions for:
- Reproducibility (seed setting)
- Gem combination encoding (take3, take2, removal)
- Mask generation for conditional loss computation
- Metric computation for evaluation
- Visualization of training results
"""

import random
from typing import Dict, List, Tuple, Optional
from itertools import combinations
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility across all libraries.

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


def generate_gem_take3_classes() -> Tuple[Dict[int, Tuple[str, ...]], Dict[Tuple[str, ...], int]]:
    """
    Generate bidirectional mappings for gem_take3 action encoding.

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
    colors = ['white', 'blue', 'green', 'red', 'black']
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


def generate_gem_removal_classes() -> Tuple[Dict[int, Tuple[int, ...]], Dict[Tuple[int, ...], int]]:
    """
    Generate bidirectional mappings for gem_removed action encoding.

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
    """
    Encode gem_take3 binary columns to single class index.

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
    colors = ['white', 'blue', 'green', 'red', 'black']
    selected_gems = []

    for color in colors:
        col_name = f'gem_take3_{color}'
        if row[col_name] == 1:
            selected_gems.append(color)

    # If no gems selected, return -1 (not applicable)
    if not selected_gems:
        return -1

    # Convert list to sorted tuple and look up class
    gems_tuple = tuple(selected_gems)
    return combo_to_class.get(gems_tuple, -1)


def encode_gem_take2(row: pd.Series) -> int:
    """
    Encode gem_take2 binary columns to single class index.

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
    colors = ['white', 'blue', 'green', 'red', 'black']

    for idx, color in enumerate(colors):
        col_name = f'gem_take2_{color}'
        if row[col_name] == 2:
            return idx

    # If no gem selected, return -1
    return -1


def encode_gems_removed(row: pd.Series, removal_to_class: Dict[Tuple[int, ...], int]) -> int:
    """
    Encode gems_removed count columns to single class index.

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
    colors = ['white', 'blue', 'green', 'red', 'black', 'gold']
    counts = tuple(int(row[f'gems_removed_{color}']) for color in colors)

    return removal_to_class.get(counts, 0)


def get_action_type_mask(action_types: np.ndarray, target_type: int) -> np.ndarray:
    """
    Create boolean mask for samples matching target action type.

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
    """
    Create boolean mask excluding -1 values (not applicable labels).

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
    """
    Combine multiple boolean masks using logical AND.

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


def compute_accuracy(predictions: np.ndarray, labels: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """
    Compute classification accuracy, optionally with masking.

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


def compute_confusion_matrix(predictions: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Compute confusion matrix for multi-class classification.

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
    return sklearn_confusion_matrix(labels, predictions, labels=list(range(num_classes)))


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    val_accuracies: Dict[str, List[float]],
    save_path: str
) -> None:
    """
    Plot training and validation curves.

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
    ax1.plot(epochs, train_losses, label='Train Loss', marker='o')
    ax1.plot(epochs, val_losses, label='Val Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy plot
    for head_name, accuracies in val_accuracies.items():
        ax2.plot(epochs, accuracies, label=head_name, marker='o')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy by Head')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: str) -> None:
    """
    Plot confusion matrix as heatmap.

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
    cm_normalized = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10) * 100

    sns.heatmap(cm_normalized, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Percentage (%)'})

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix (Normalized by True Class)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_per_class_accuracy(accuracies: Dict[str, float], save_path: str) -> None:
    """
    Plot per-head accuracy as bar chart.

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

    bars = plt.bar(range(len(heads)), values, color='steelblue', alpha=0.8)

    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, values)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.xticks(range(len(heads)), heads, rotation=45, ha='right')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy by Prediction Head')
    plt.ylim(0, 1.0)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def get_num_gem_removal_classes() -> int:
    """
    Calculate the total number of gem removal classes.

    Returns:
        Number of valid removal combinations

    Example:
        >>> get_num_gem_removal_classes()
        56  # Or whatever the actual count is
    """
    _, removal_to_class = generate_gem_removal_classes()
    return len(removal_to_class)


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
