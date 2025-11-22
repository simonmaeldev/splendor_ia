"""
PyTorch Dataset for loading preprocessed Splendor game data.

This module provides a Dataset class that loads preprocessed features and labels
from disk and returns them in PyTorch tensor format for training.
"""

import os
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class SplendorDataset(Dataset):
    """
    PyTorch Dataset for Splendor imitation learning.

    Loads preprocessed features, labels, and legal action masks, returns them as tensors.
    All labels are preserved including -1 values which will be masked during training.
    Masks indicate which actions are legal for each state.
    """

    def __init__(self, X: np.ndarray, labels: Dict[str, np.ndarray], masks: Dict[str, np.ndarray]):
        """
        Initialize dataset with features, labels, and masks.

        Args:
            X: Feature array of shape (n_samples, input_dim)
            labels: Dict mapping head name to label array of shape (n_samples,)
                   Keys: action_type, card_selection, card_reservation,
                         gem_take3, gem_take2, noble, gems_removed
            masks: Dict mapping head name to mask array of shape (n_samples, n_classes_for_head)
                   Binary masks where 1 = legal action, 0 = illegal action
        """
        self.X = torch.from_numpy(X).float()

        self.labels = {
            name: torch.from_numpy(arr).long()
            for name, arr in labels.items()
        }

        self.masks = {
            name: torch.from_numpy(arr).float()
            for name, arr in masks.items()
        }

        # Verify all labels and masks have same length
        assert all(len(arr) == len(self.X) for arr in self.labels.values()), \
            "All labels must have same length as features"
        assert all(len(arr) == len(self.X) for arr in self.masks.values()), \
            "All masks must have same length as features"

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Tuple of (state_tensor, labels_dict, masks_dict)
            - state_tensor: float tensor of shape (input_dim,)
            - labels_dict: dict of 7 long tensors (scalar for each head)
            - masks_dict: dict of 7 float tensors (n_classes_for_head,) with 0/1 values
        """
        state = self.X[idx]
        labels_dict = {name: labels[idx] for name, labels in self.labels.items()}
        masks_dict = {name: masks[idx] for name, masks in self.masks.items()}

        return state, labels_dict, masks_dict


def load_preprocessed_data(processed_dir: str) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray,
    Dict, Dict, Dict,
    Dict, Dict, Dict
]:
    """
    Load all preprocessed data from disk.

    Args:
        processed_dir: Directory containing preprocessed files

    Returns:
        Tuple of (X_train, X_val, X_test, labels_train, labels_val, labels_test,
                 masks_train, masks_val, masks_test)
    """
    print(f"Loading preprocessed data from {processed_dir}...")

    # Load features
    X_train = np.load(os.path.join(processed_dir, 'X_train.npy'))
    X_val = np.load(os.path.join(processed_dir, 'X_val.npy'))
    X_test = np.load(os.path.join(processed_dir, 'X_test.npy'))

    print(f"  Features: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

    # Load labels
    labels_train = dict(np.load(os.path.join(processed_dir, 'labels_train.npz')))
    labels_val = dict(np.load(os.path.join(processed_dir, 'labels_val.npz')))
    labels_test = dict(np.load(os.path.join(processed_dir, 'labels_test.npz')))

    print(f"  Labels: {list(labels_train.keys())}")

    # Load masks
    masks_train = dict(np.load(os.path.join(processed_dir, 'masks_train.npz')))
    masks_val = dict(np.load(os.path.join(processed_dir, 'masks_val.npz')))
    masks_test = dict(np.load(os.path.join(processed_dir, 'masks_test.npz')))

    print(f"  Masks: {list(masks_train.keys())}")

    return X_train, X_val, X_test, labels_train, labels_val, labels_test, masks_train, masks_val, masks_test


def create_dataloaders(
    processed_dir: str,
    batch_size: int,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for train, validation, and test sets.

    Args:
        processed_dir: Directory containing preprocessed data files
        batch_size: Batch size for DataLoader
        num_workers: Number of workers for parallel data loading

    Returns:
        Tuple of (train_loader, val_loader, test_loader)

    Example:
        >>> train_loader, val_loader, test_loader = create_dataloaders(
        ...     "data/processed", batch_size=256, num_workers=4
        ... )
        >>> for batch_states, batch_labels, batch_masks in train_loader:
        ...     # batch_states: (batch_size, input_dim)
        ...     # batch_labels: dict of 7 tensors, each (batch_size,)
        ...     # batch_masks: dict of 7 tensors, each (batch_size, n_classes_for_head)
        ...     pass
    """
    # Load preprocessed data
    (X_train, X_val, X_test,
     labels_train, labels_val, labels_test,
     masks_train, masks_val, masks_test) = load_preprocessed_data(processed_dir)

    # Create datasets
    train_dataset = SplendorDataset(X_train, labels_train, masks_train)
    val_dataset = SplendorDataset(X_val, labels_val, masks_val)
    test_dataset = SplendorDataset(X_test, labels_test, masks_test)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data
        num_workers=num_workers,
        pin_memory=True  # Faster GPU transfer
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle validation
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle test
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"\nDataLoaders created:")
    print(f"  Train: {len(train_loader)} batches ({len(train_dataset)} samples)")
    print(f"  Val: {len(val_loader)} batches ({len(val_dataset)} samples)")
    print(f"  Test: {len(test_loader)} batches ({len(test_dataset)} samples)")

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test dataset loading (requires preprocessed data)
    import sys

    if len(sys.argv) < 2:
        print("Usage: python dataset.py <processed_dir>")
        print("Example: python dataset.py ../../data/processed")
        sys.exit(1)

    processed_dir = sys.argv[1]

    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            processed_dir, batch_size=32, num_workers=0
        )

        # Test one batch
        print("\nTesting one batch...")
        states, labels, masks = next(iter(train_loader))

        print(f"  States shape: {states.shape}, dtype: {states.dtype}")
        print(f"  Labels:")
        for name, tensor in labels.items():
            print(f"    {name}: shape={tensor.shape}, dtype={tensor.dtype}, "
                  f"min={tensor.min()}, max={tensor.max()}")
        print(f"  Masks:")
        for name, tensor in masks.items():
            print(f"    {name}: shape={tensor.shape}, dtype={tensor.dtype}, "
                  f"legal_actions={tensor.sum(dim=1).mean():.1f}")

        print("\n✓ Dataset test passed!")

    except FileNotFoundError as e:
        print(f"Error: Preprocessed data not found at {processed_dir}")
        print("Run data_preprocessing.py first to generate preprocessed data.")
        sys.exit(1)
