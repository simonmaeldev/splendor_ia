"""
Training loop for Splendor imitation learning.

This module implements the complete training pipeline including:
- Conditional loss computation for each prediction head
- Training and validation loops
- Checkpointing and early stopping
- Weights & Biases logging
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import wandb
import yaml

from .dataset import create_dataloaders
from .model import MultiHeadSplendorNet
from .utils import set_seed, compute_accuracy, plot_training_curves


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def compute_class_weights(labels: np.ndarray, num_classes: int, device: torch.device) -> torch.Tensor:
    """
    Compute inverse frequency class weights for imbalanced datasets.

    Args:
        labels: Array of class labels
        num_classes: Total number of classes
        device: Device to place tensor on

    Returns:
        Tensor of weights for each class
    """
    # Filter out -1 (not applicable) labels
    valid_labels = labels[labels != -1]

    if len(valid_labels) == 0:
        return torch.ones(num_classes, device=device)

    # Count frequency of each class
    counts = np.bincount(valid_labels, minlength=num_classes)

    # Compute inverse frequency
    # Add 1 to avoid division by zero for rare classes
    weights = 1.0 / (counts + 1)

    # Normalize so sum = num_classes
    weights = weights / weights.sum() * num_classes

    return torch.tensor(weights, dtype=torch.float32, device=device)


def compute_conditional_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    weights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute cross-entropy loss only for samples matching the mask.

    Args:
        logits: Predicted logits of shape (batch_size, num_classes)
        labels: True labels of shape (batch_size,)
        mask: Boolean mask of shape (batch_size,) indicating which samples to include
        weights: Optional class weights

    Returns:
        Scalar loss (0.0 if no valid samples)
    """
    if not mask.any():
        # No valid samples, return zero loss
        return torch.tensor(0.0, device=logits.device)

    # Apply mask
    logits_masked = logits[mask]
    labels_masked = labels[mask]

    # Compute loss
    loss = F.cross_entropy(logits_masked, labels_masked, weight=weights)

    return loss


def compute_total_loss(
    outputs: Dict[str, torch.Tensor],
    labels: Dict[str, torch.Tensor],
    class_weights: Optional[Dict[str, torch.Tensor]] = None
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute total loss across all heads with conditional masking.

    Args:
        outputs: Dict mapping head name to logits tensor
        labels: Dict mapping head name to label tensor
        class_weights: Optional dict of class weights per head

    Returns:
        Tuple of (total_loss, per_head_losses_dict)
    """
    device = outputs['action_type'].device
    action_types = labels['action_type']
    losses = {}

    # Action type: always compute (every sample has action type)
    action_type_weight = class_weights.get('action_type') if class_weights else None
    losses['action_type'] = F.cross_entropy(
        outputs['action_type'],
        action_types,
        weight=action_type_weight
    )

    # Card selection: only when action_type == 0 (BUILD) AND label != -1
    build_mask = action_types == 0
    card_sel_labels = labels['card_selection']
    card_sel_valid = card_sel_labels != -1
    card_sel_mask = build_mask & card_sel_valid
    losses['card_selection'] = compute_conditional_loss(
        outputs['card_selection'],
        card_sel_labels,
        card_sel_mask
    )

    # Card reservation: only when action_type == 1 (RESERVE) AND label != -1
    reserve_mask = action_types == 1
    card_res_labels = labels['card_reservation']
    card_res_valid = card_res_labels != -1
    card_res_mask = reserve_mask & card_res_valid
    losses['card_reservation'] = compute_conditional_loss(
        outputs['card_reservation'],
        card_res_labels,
        card_res_mask
    )

    # Gem take3: only when action_type == 3 (TAKE3) AND label != -1
    take3_mask = action_types == 3
    gem3_labels = labels['gem_take3']
    gem3_valid = gem3_labels != -1
    gem3_mask = take3_mask & gem3_valid
    losses['gem_take3'] = compute_conditional_loss(
        outputs['gem_take3'],
        gem3_labels,
        gem3_mask
    )

    # Gem take2: only when action_type == 2 (TAKE2) AND label != -1
    take2_mask = action_types == 2
    gem2_labels = labels['gem_take2']
    gem2_valid = gem2_labels != -1
    gem2_mask = take2_mask & gem2_valid
    losses['gem_take2'] = compute_conditional_loss(
        outputs['gem_take2'],
        gem2_labels,
        gem2_mask
    )

    # Noble: only when action_type == 0 (BUILD) AND label != -1
    noble_labels = labels['noble']
    noble_valid = noble_labels != -1
    noble_mask = build_mask & noble_valid
    losses['noble'] = compute_conditional_loss(
        outputs['noble'],
        noble_labels,
        noble_mask
    )

    # Gems removed: only when label != 0 (class 0 = no removal)
    gems_rem_labels = labels['gems_removed']
    gems_rem_mask = gems_rem_labels != 0
    losses['gems_removed'] = compute_conditional_loss(
        outputs['gems_removed'],
        gems_rem_labels,
        gems_rem_mask
    )

    # Total loss is sum of all losses
    total_loss = sum(losses.values())

    # Convert to float for logging
    losses_float = {k: v.item() for k, v in losses.items()}

    return total_loss, losses_float


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    class_weights: Optional[Dict[str, torch.Tensor]] = None,
    gradient_clip_norm: Optional[float] = None
) -> Tuple[float, Dict[str, float]]:
    """
    Train model for one epoch.

    Args:
        model: Neural network model
        dataloader: Training data loader
        optimizer: Optimizer
        device: Device to run on
        class_weights: Optional class weights for each head
        gradient_clip_norm: Optional gradient clipping value

    Returns:
        Tuple of (average_total_loss, dict_of_average_per_head_losses)
    """
    model.train()

    total_loss_accum = 0.0
    losses_accum = {name: 0.0 for name in ['action_type', 'card_selection', 'card_reservation',
                                            'gem_take3', 'gem_take2', 'noble', 'gems_removed']}
    num_batches = 0

    pbar = tqdm(dataloader, desc='Training', leave=False)

    for states, labels in pbar:
        # Move to device
        states = states.to(device)
        labels = {k: v.to(device) for k, v in labels.items()}

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(states)

        # Compute loss
        total_loss, per_head_losses = compute_total_loss(outputs, labels, class_weights)

        # Backward pass
        total_loss.backward()

        # Gradient clipping if specified
        if gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)

        # Optimizer step
        optimizer.step()

        # Accumulate losses
        total_loss_accum += total_loss.item()
        for name, loss_val in per_head_losses.items():
            losses_accum[name] += loss_val
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({'loss': f'{total_loss.item():.4f}'})

    # Average losses
    avg_total_loss = total_loss_accum / num_batches
    avg_per_head_losses = {k: v / num_batches for k, v in losses_accum.items()}

    return avg_total_loss, avg_per_head_losses


def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device
) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    """
    Validate model on validation set.

    Args:
        model: Neural network model
        dataloader: Validation data loader
        device: Device to run on

    Returns:
        Tuple of (avg_total_loss, dict_of_avg_per_head_losses, dict_of_per_head_accuracies)
    """
    model.eval()

    total_loss_accum = 0.0
    losses_accum = {name: 0.0 for name in ['action_type', 'card_selection', 'card_reservation',
                                            'gem_take3', 'gem_take2', 'noble', 'gems_removed']}
    accuracies_accum = {name: [] for name in losses_accum.keys()}
    num_batches = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation', leave=False)

        for states, labels in pbar:
            # Move to device
            states = states.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}

            # Forward pass
            outputs = model(states)

            # Compute loss
            total_loss, per_head_losses = compute_total_loss(outputs, labels)

            # Accumulate losses
            total_loss_accum += total_loss.item()
            for name, loss_val in per_head_losses.items():
                losses_accum[name] += loss_val
            num_batches += 1

            # Compute accuracies
            action_types = labels['action_type'].cpu().numpy()

            # Action type accuracy (always compute)
            action_type_preds = outputs['action_type'].argmax(dim=1).cpu().numpy()
            accuracies_accum['action_type'].append(compute_accuracy(action_type_preds, action_types))

            # Card selection accuracy (only for BUILD actions)
            build_mask = action_types == 0
            card_sel_labels = labels['card_selection'].cpu().numpy()
            card_sel_valid = card_sel_labels != -1
            card_sel_mask = build_mask & card_sel_valid
            if card_sel_mask.any():
                card_sel_preds = outputs['card_selection'].argmax(dim=1).cpu().numpy()
                accuracies_accum['card_selection'].append(
                    compute_accuracy(card_sel_preds, card_sel_labels, card_sel_mask)
                )

            # Card reservation accuracy (only for RESERVE actions)
            reserve_mask = action_types == 1
            card_res_labels = labels['card_reservation'].cpu().numpy()
            card_res_valid = card_res_labels != -1
            card_res_mask = reserve_mask & card_res_valid
            if card_res_mask.any():
                card_res_preds = outputs['card_reservation'].argmax(dim=1).cpu().numpy()
                accuracies_accum['card_reservation'].append(
                    compute_accuracy(card_res_preds, card_res_labels, card_res_mask)
                )

            # Gem take3 accuracy
            take3_mask = action_types == 3
            gem3_labels = labels['gem_take3'].cpu().numpy()
            gem3_valid = gem3_labels != -1
            gem3_mask = take3_mask & gem3_valid
            if gem3_mask.any():
                gem3_preds = outputs['gem_take3'].argmax(dim=1).cpu().numpy()
                accuracies_accum['gem_take3'].append(
                    compute_accuracy(gem3_preds, gem3_labels, gem3_mask)
                )

            # Gem take2 accuracy
            take2_mask = action_types == 2
            gem2_labels = labels['gem_take2'].cpu().numpy()
            gem2_valid = gem2_labels != -1
            gem2_mask = take2_mask & gem2_valid
            if gem2_mask.any():
                gem2_preds = outputs['gem_take2'].argmax(dim=1).cpu().numpy()
                accuracies_accum['gem_take2'].append(
                    compute_accuracy(gem2_preds, gem2_labels, gem2_mask)
                )

            # Noble accuracy
            noble_labels = labels['noble'].cpu().numpy()
            noble_valid = noble_labels != -1
            noble_mask = build_mask & noble_valid
            if noble_mask.any():
                noble_preds = outputs['noble'].argmax(dim=1).cpu().numpy()
                accuracies_accum['noble'].append(
                    compute_accuracy(noble_preds, noble_labels, noble_mask)
                )

            # Gems removed accuracy
            gems_rem_labels = labels['gems_removed'].cpu().numpy()
            gems_rem_mask = gems_rem_labels != 0
            if gems_rem_mask.any():
                gems_rem_preds = outputs['gems_removed'].argmax(dim=1).cpu().numpy()
                accuracies_accum['gems_removed'].append(
                    compute_accuracy(gems_rem_preds, gems_rem_labels, gems_rem_mask)
                )

    # Average losses and accuracies
    avg_total_loss = total_loss_accum / num_batches
    avg_per_head_losses = {k: v / num_batches for k, v in losses_accum.items()}
    avg_per_head_accuracies = {
        k: np.mean(v) if v else 0.0
        for k, v in accuracies_accum.items()
    }

    return avg_total_loss, avg_per_head_losses, avg_per_head_accuracies


class EarlyStopping:
    """Early stopping to stop training when validation loss stops improving."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should stop.

        Args:
            val_loss: Current validation loss

        Returns:
            True if training should stop
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

        return self.should_stop


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    val_acc: Dict[str, float],
    save_path: str
):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_acc': val_acc
    }
    torch.save(checkpoint, save_path)


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str
) -> Tuple[int, float, Dict[str, float]]:
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['val_loss'], checkpoint['val_acc']


def main():
    """Main training loop."""
    parser = argparse.ArgumentParser(description='Train Splendor imitation learning model')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from {args.config}...")
    config = load_config(args.config)

    # Set seed
    set_seed(config['seed'])
    print(f"Random seed set to {config['seed']}")

    # Load preprocessing stats to get input_dim
    stats_path = os.path.join(config['data']['processed_dir'], 'preprocessing_stats.json')
    with open(stats_path, 'r') as f:
        stats = json.load(f)

    config['model']['input_dim'] = stats['input_dim']
    print(f"Input dimension: {config['model']['input_dim']}")

    # Set device
    device = torch.device(config['compute']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        config['data']['processed_dir'],
        config['training']['batch_size'],
        config['compute']['num_workers']
    )

    # Initialize model
    model = MultiHeadSplendorNet(config['model'])
    model.to(device)
    model.print_summary()

    # Initialize optimizer
    optimizer = Adam(model.parameters(), lr=config['training']['learning_rate'])

    # Initialize scheduler if enabled
    scheduler = None
    if config['training']['scheduler']['enabled']:
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=config['training']['scheduler']['mode'],
            factor=config['training']['scheduler']['factor'],
            patience=config['training']['scheduler']['patience']
        )

    # Initialize wandb
    wandb.init(
        project=config['logging']['wandb_project'],
        entity=config['logging'].get('wandb_entity'),
        config=config,
        name=f"splendor_il_{Path(args.config).stem}"
    )
    wandb.watch(model, log='all', log_freq=100)

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=config['training']['patience'])

    # Load checkpoint if resuming
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        start_epoch, best_val_loss, _ = load_checkpoint(model, optimizer, args.resume)
        start_epoch += 1

    # Training history
    train_losses = []
    val_losses = []
    val_accuracies = {name: [] for name in ['action_type', 'card_selection', 'card_reservation',
                                              'gem_take3', 'gem_take2', 'noble', 'gems_removed']}

    # Training loop
    print(f"\nStarting training for {config['training']['epochs']} epochs...")
    for epoch in range(start_epoch, config['training']['epochs']):
        print(f"\nEpoch {epoch+1}/{config['training']['epochs']}")

        # Train
        train_loss, train_per_head_losses = train_one_epoch(
            model, train_loader, optimizer, device,
            gradient_clip_norm=config['training'].get('gradient_clip_norm')
        )
        train_losses.append(train_loss)

        # Validate
        val_loss, val_per_head_losses, val_per_head_accs = validate(
            model, val_loader, device
        )
        val_losses.append(val_loss)
        for name, acc in val_per_head_accs.items():
            val_accuracies[name].append(acc)

        # Log to wandb
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            **{f'train_loss_{k}': v for k, v in train_per_head_losses.items()},
            **{f'val_loss_{k}': v for k, v in val_per_head_losses.items()},
            **{f'val_acc_{k}': v for k, v in val_per_head_accs.items()},
            'learning_rate': optimizer.param_groups[0]['lr']
        })

        # Print progress
        print(f"  Train loss: {train_loss:.4f}")
        print(f"  Val loss: {val_loss:.4f}")
        print(f"  Val acc (action_type): {val_per_head_accs['action_type']:.4f}")

        # Step scheduler
        if scheduler is not None:
            scheduler.step(val_loss)

        # Save checkpoint if improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(config['checkpointing']['save_dir'], 'best_model.pth')
            save_checkpoint(model, optimizer, epoch, val_loss, val_per_head_accs, save_path)
            print(f"  ✓ Saved best model (val_loss={val_loss:.4f})")

        # Save periodic checkpoint
        if (epoch + 1) % config['checkpointing']['save_every'] == 0:
            save_path = os.path.join(config['checkpointing']['save_dir'], f'checkpoint_epoch_{epoch+1}.pth')
            save_checkpoint(model, optimizer, epoch, val_loss, val_per_head_accs, save_path)

        # Check early stopping
        if early_stopping(val_loss):
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break

    # Save final model
    final_save_path = os.path.join(config['checkpointing']['save_dir'], 'final_model.pth')
    save_checkpoint(model, optimizer, epoch, val_loss, val_per_head_accs, final_save_path)
    print(f"\nSaved final model to {final_save_path}")

    # Plot training curves
    if config['logging']['save_plots']:
        plot_path = os.path.join('logs', f'training_curves_{wandb.run.id}.png')
        plot_training_curves(train_losses, val_losses, val_accuracies, plot_path)
        print(f"Saved training curves to {plot_path}")

    # Print best results
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {os.path.join(config['checkpointing']['save_dir'], 'best_model.pth')}")
    print(f"{'='*60}\n")

    wandb.finish()


if __name__ == "__main__":
    main()
