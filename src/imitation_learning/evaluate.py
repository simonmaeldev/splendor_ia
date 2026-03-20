"""
Evaluation metrics and analysis for Splendor imitation learning.

This module provides comprehensive evaluation including:
- Per-head metrics (accuracy, confusion matrices)
- Overall action accuracy
- Baseline comparisons
- Visualization of results
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import yaml

from .dataset import create_dataloaders
from .model import MultiHeadSplendorNet
from .train import apply_legal_action_mask
from .utils import (
    compute_accuracy,
    compute_classification_metrics,
    compute_confusion_matrix,
    plot_confusion_matrix,
    plot_per_class_accuracy,
    plot_classification_metrics,
    set_seed
)


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def evaluate_action_type(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    enable_masking: bool = True
) -> Dict:
    """
    Comprehensive evaluation of action_type head.

    Args:
        model: Trained model
        dataloader: Data loader (typically test set)
        device: Device to run on
        enable_masking: If False, skip masking during evaluation (default: True)

    Returns:
        Dict containing overall accuracy, per-class accuracy, confusion matrix
    """
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for states, labels, masks in tqdm(dataloader, desc='Evaluating action_type'):
            states = states.to(device)
            masks = {k: v.to(device) for k, v in masks.items()}
            outputs = model(states)

            # Apply legal action mask to enforce only legal actions
            action_type_logits = apply_legal_action_mask(outputs['action_type'], masks['action_type'], enable_masking)
            preds = action_type_logits.argmax(dim=1).cpu().numpy()
            labs = labels['action_type'].cpu().numpy()

            all_preds.append(preds)
            all_labels.append(labs)

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Compute comprehensive metrics
    metrics = compute_classification_metrics(all_preds, all_labels, num_classes=4)

    # Per-class accuracy
    per_class_acc = {}
    class_names = ['build', 'reserve', 'take2', 'take3']
    for class_idx, class_name in enumerate(class_names):
        mask = all_labels == class_idx
        if mask.any():
            per_class_acc[class_name] = compute_accuracy(all_preds, all_labels, mask)
        else:
            per_class_acc[class_name] = 0.0

    # Confusion matrix
    cm = compute_confusion_matrix(all_preds, all_labels, num_classes=4)

    return {
        'overall_accuracy': float(metrics['accuracy']),
        'precision': float(metrics['precision']),
        'recall': float(metrics['recall']),
        'specificity': float(metrics['specificity']),
        'f_score': float(metrics['f_score']),
        'per_class_accuracy': per_class_acc,
        'confusion_matrix': cm.tolist(),
        'class_distribution': np.bincount(all_labels, minlength=4).tolist()
    }


def evaluate_conditional_head(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    head_name: str,
    action_type_filter: int,
    num_classes: int,
    enable_masking: bool = True
) -> Dict:
    """
    Evaluate a conditional prediction head (only applicable for specific action types).

    Args:
        model: Trained model
        dataloader: Data loader
        device: Device
        head_name: Name of the head to evaluate
        action_type_filter: Which action type this head applies to
        num_classes: Number of classes for this head
        enable_masking: If False, skip masking during evaluation (default: True)

    Returns:
        Dict containing comprehensive metrics and sample frequency
    """
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for states, labels, legal_masks in tqdm(dataloader, desc=f'Evaluating {head_name}'):
            states = states.to(device)
            legal_masks = {k: v.to(device) for k, v in legal_masks.items()}
            outputs = model(states)

            # Filter by action type
            action_types = labels['action_type'].cpu().numpy()
            mask = action_types == action_type_filter

            # Additional filter for valid labels
            head_labels = labels[head_name].cpu().numpy()
            valid_mask = head_labels != -1
            combined_mask = mask & valid_mask

            if combined_mask.any():
                # Apply legal action mask to enforce only legal actions
                head_logits = apply_legal_action_mask(outputs[head_name], legal_masks[head_name], enable_masking)
                preds = head_logits.argmax(dim=1).cpu().numpy()
                all_preds.append(preds[combined_mask])
                all_labels.append(head_labels[combined_mask])

    if not all_preds:
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'specificity': 0.0,
            'f_score': 0.0,
            'num_samples': 0,
            'frequency': 0.0
        }

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Compute comprehensive metrics
    metrics = compute_classification_metrics(all_preds, all_labels, num_classes=num_classes)

    return {
        'accuracy': float(metrics['accuracy']),
        'precision': float(metrics['precision']),
        'recall': float(metrics['recall']),
        'specificity': float(metrics['specificity']),
        'f_score': float(metrics['f_score']),
        'num_samples': len(all_labels),
        'frequency': len(all_labels) / len(dataloader.dataset)
    }


def evaluate_gems_removed_head(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int,
    enable_masking: bool = True
) -> Dict:
    """
    Evaluate gems_removed head across ALL action types.

    Unlike other conditional heads, gems_removed can apply to any action type
    when the player has > 10 tokens after taking an action. This function
    evaluates only samples where gem removal was actually required (label != 0).

    Args:
        model: Trained model
        dataloader: Data loader
        device: Device
        num_classes: Number of gem removal classes
        enable_masking: If False, skip masking during evaluation (default: True)

    Returns:
        Dict containing comprehensive metrics, sample count, and frequency
    """
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for states, labels, legal_masks in tqdm(dataloader, desc='Evaluating gems_removed'):
            states = states.to(device)
            legal_masks = {k: v.to(device) for k, v in legal_masks.items()}
            outputs = model(states)

            # Filter for samples where gems_removed is needed (label != 0)
            # Note: 0 means "no removal needed", not "invalid" like -1 in other heads
            gems_removed_labels = labels['gems_removed'].cpu().numpy()
            mask = gems_removed_labels != 0

            if mask.any():
                # Apply legal action mask to enforce only legal token removals
                gems_removed_logits = apply_legal_action_mask(
                    outputs['gems_removed'],
                    legal_masks['gems_removed'],
                    enable_masking
                )
                preds = gems_removed_logits.argmax(dim=1).cpu().numpy()
                all_preds.append(preds[mask])
                all_labels.append(gems_removed_labels[mask])

    if not all_preds:
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'specificity': 0.0,
            'f_score': 0.0,
            'num_samples': 0,
            'frequency': 0.0
        }

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Compute comprehensive metrics
    metrics = compute_classification_metrics(all_preds, all_labels, num_classes=num_classes)

    return {
        'accuracy': float(metrics['accuracy']),
        'precision': float(metrics['precision']),
        'recall': float(metrics['recall']),
        'specificity': float(metrics['specificity']),
        'f_score': float(metrics['f_score']),
        'num_samples': len(all_labels),
        'frequency': len(all_labels) / len(dataloader.dataset)
    }


def evaluate_overall_action_accuracy(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    enable_masking: bool = True
) -> float:
    """
    Compute overall action accuracy: action_type correct AND target correct.

    This is the true end-to-end performance metric.

    Args:
        model: Trained model
        dataloader: Data loader
        device: Device
        enable_masking: If False, skip masking during evaluation (default: True)

    Returns:
        Overall action accuracy
    """
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for states, labels, legal_masks in tqdm(dataloader, desc='Computing overall accuracy'):
            states = states.to(device)
            legal_masks = {k: v.to(device) for k, v in legal_masks.items()}
            outputs = model(states)

            # Apply legal action masks to all heads
            masked_outputs = {}
            for head_name in outputs.keys():
                if head_name in legal_masks:
                    masked_outputs[head_name] = apply_legal_action_mask(outputs[head_name], legal_masks[head_name], enable_masking)
                else:
                    masked_outputs[head_name] = outputs[head_name]

            batch_size = len(states)
            action_types = labels['action_type'].cpu().numpy()
            action_type_preds = masked_outputs['action_type'].argmax(dim=1).cpu().numpy()

            for i in range(batch_size):
                true_action = action_types[i]
                pred_action = action_type_preds[i]

                # Action type must be correct
                if true_action != pred_action:
                    total += 1
                    continue

                # Check if target is also correct based on action type
                target_correct = False

                if true_action == 0:  # BUILD
                    card_sel_label = labels['card_selection'][i].item()
                    if card_sel_label != -1:
                        card_sel_pred = masked_outputs['card_selection'][i].argmax().item()
                        target_correct = (card_sel_pred == card_sel_label)
                    else:
                        target_correct = True  # No target to predict

                elif true_action == 1:  # RESERVE
                    card_res_label = labels['card_reservation'][i].item()
                    if card_res_label != -1:
                        card_res_pred = masked_outputs['card_reservation'][i].argmax().item()
                        target_correct = (card_res_pred == card_res_label)
                    else:
                        target_correct = True

                elif true_action == 2:  # TAKE2
                    gem2_label = labels['gem_take2'][i].item()
                    if gem2_label != -1:
                        gem2_pred = masked_outputs['gem_take2'][i].argmax().item()
                        target_correct = (gem2_pred == gem2_label)
                    else:
                        target_correct = True

                elif true_action == 3:  # TAKE3
                    gem3_label = labels['gem_take3'][i].item()
                    if gem3_label != -1:
                        gem3_pred = masked_outputs['gem_take3'][i].argmax().item()
                        target_correct = (gem3_pred == gem3_label)
                    else:
                        target_correct = True

                if target_correct:
                    correct += 1
                total += 1

    return correct / total if total > 0 else 0.0


class RandomBaseline:
    """Random baseline that predicts uniformly among classes."""

    def predict_action_type(self, batch_size: int) -> np.ndarray:
        """Predict random action types."""
        return np.random.randint(0, 4, size=batch_size)

    def predict_card_selection(self, batch_size: int) -> np.ndarray:
        """Predict random card selections."""
        return np.random.randint(0, 15, size=batch_size)


def evaluate_random_baseline(dataloader: torch.utils.data.DataLoader) -> Dict:
    """
    Evaluate random baseline performance.

    Args:
        dataloader: Data loader

    Returns:
        Dict of metrics
    """
    baseline = RandomBaseline()

    action_type_correct = 0
    overall_correct = 0
    total = 0

    for states, labels, masks in tqdm(dataloader, desc='Evaluating random baseline'):
        batch_size = len(states)
        action_types = labels['action_type'].cpu().numpy()

        # Random action type predictions
        action_type_preds = baseline.predict_action_type(batch_size)
        action_type_correct += np.sum(action_type_preds == action_types)

        # For overall accuracy, assume random is correct 1/num_classes of the time
        # This is an approximation
        overall_correct += batch_size * 0.25 * 0.067  # ~0.25 action type * ~1/15 target

        total += batch_size

    return {
        'action_type_accuracy': action_type_correct / total,
        'overall_accuracy_estimate': overall_correct / total
    }


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(description='Evaluate Splendor imitation learning model')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='Which split to evaluate on')
    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from {args.config}...")
    config = load_config(args.config)

    # Set seed
    set_seed(config['seed'])

    # Load preprocessing stats to get input_dim
    stats_path = os.path.join(config['data']['processed_dir'], 'preprocessing_stats.json')
    with open(stats_path, 'r') as f:
        stats = json.load(f)

    config['model']['input_dim'] = stats['input_dim']

    # Set device
    device = torch.device(config['compute']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        config['data']['processed_dir'],
        config['training']['batch_size'],
        config['compute']['num_workers']
    )

    # Select dataloader
    dataloader = {'train': train_loader, 'val': val_loader, 'test': test_loader}[args.split]
    print(f"Evaluating on {args.split} split ({len(dataloader.dataset)} samples)")

    # Load model
    model = MultiHeadSplendorNet(config['model'])
    model.to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from {args.checkpoint}")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Checkpoint val_loss: {checkpoint['val_loss']:.4f}")

    # Evaluate action type
    print("\nEvaluating action_type head...")
    action_type_results = evaluate_action_type(model, dataloader, device)
    print(f"  Overall accuracy: {action_type_results['overall_accuracy']:.4f}")
    print(f"  Per-class accuracy:")
    for class_name, acc in action_type_results['per_class_accuracy'].items():
        print(f"    {class_name}: {acc:.4f}")

    # Evaluate conditional heads
    print("\nEvaluating conditional heads...")
    results = {'action_type': action_type_results}

    # Card selection (BUILD) - 15 classes
    card_sel_results = evaluate_conditional_head(model, dataloader, device, 'card_selection', 0, num_classes=15)
    results['card_selection'] = card_sel_results
    print(f"  card_selection: acc={card_sel_results['accuracy']:.4f}, f-score={card_sel_results['f_score']:.4f}, "
          f"samples={card_sel_results['num_samples']}")

    # Card reservation (RESERVE) - 15 classes
    card_res_results = evaluate_conditional_head(model, dataloader, device, 'card_reservation', 1, num_classes=15)
    results['card_reservation'] = card_res_results
    print(f"  card_reservation: acc={card_res_results['accuracy']:.4f}, f-score={card_res_results['f_score']:.4f}, "
          f"samples={card_res_results['num_samples']}")

    # Gem take3 (TAKE3) - 26 classes
    gem3_results = evaluate_conditional_head(model, dataloader, device, 'gem_take3', 3, num_classes=26)
    results['gem_take3'] = gem3_results
    print(f"  gem_take3: acc={gem3_results['accuracy']:.4f}, f-score={gem3_results['f_score']:.4f}, "
          f"samples={gem3_results['num_samples']}")

    # Gem take2 (TAKE2) - 5 classes
    gem2_results = evaluate_conditional_head(model, dataloader, device, 'gem_take2', 2, num_classes=5)
    results['gem_take2'] = gem2_results
    print(f"  gem_take2: acc={gem2_results['accuracy']:.4f}, f-score={gem2_results['f_score']:.4f}, "
          f"samples={gem2_results['num_samples']}")

    # Noble (BUILD with noble available) - 5 classes
    noble_results = evaluate_conditional_head(model, dataloader, device, 'noble', 0, num_classes=5)
    results['noble'] = noble_results
    print(f"  noble: acc={noble_results['accuracy']:.4f}, f-score={noble_results['f_score']:.4f}, "
          f"samples={noble_results['num_samples']}")

    # Gems removed (overflow) - 84 classes (from constants)
    from .constants import NUM_GEM_REMOVAL_CLASSES
    gems_rem_results = evaluate_gems_removed_head(model, dataloader, device, num_classes=NUM_GEM_REMOVAL_CLASSES)
    results['gems_removed'] = gems_rem_results
    print(f"  gems_removed: acc={gems_rem_results['accuracy']:.4f}, f-score={gems_rem_results['f_score']:.4f}, "
          f"samples={gems_rem_results['num_samples']}")

    # Overall action accuracy
    print("\nComputing overall action accuracy...")
    overall_acc = evaluate_overall_action_accuracy(model, dataloader, device)
    results['overall_accuracy'] = float(overall_acc)
    print(f"  Overall action accuracy: {overall_acc:.4f}")

    # Baseline comparison
    print("\nEvaluating random baseline...")
    baseline_results = evaluate_random_baseline(dataloader)
    results['baseline'] = baseline_results
    print(f"  Random baseline action_type acc: {baseline_results['action_type_accuracy']:.4f}")

    # Save results
    results_dir = 'logs'
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f'evaluation_results_{args.split}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_path}")

    # Generate visualizations
    print("\nGenerating visualizations...")

    # Confusion matrix
    cm_path = os.path.join(results_dir, f'confusion_matrix_{args.split}.png')
    cm = np.array(action_type_results['confusion_matrix'])
    plot_confusion_matrix(cm, ['BUILD', 'RESERVE', 'TAKE2', 'TAKE3'], cm_path)
    print(f"  Saved confusion matrix to {cm_path}")

    # Per-head accuracies (legacy plot)
    acc_dict = {
        'action_type': action_type_results['overall_accuracy'],
        'card_selection': card_sel_results['accuracy'],
        'card_reservation': card_res_results['accuracy'],
        'gem_take3': gem3_results['accuracy'],
        'gem_take2': gem2_results['accuracy'],
        'noble': noble_results['accuracy'],
        'gems_removed': gems_rem_results['accuracy']
    }
    acc_path = os.path.join(results_dir, f'per_head_accuracies_{args.split}.png')
    plot_per_class_accuracy(acc_dict, acc_path)
    print(f"  Saved per-head accuracies to {acc_path}")

    # Comprehensive metrics plot (NEW)
    metrics_dict = {
        'action_type': {
            'accuracy': action_type_results['overall_accuracy'],
            'precision': action_type_results['precision'],
            'recall': action_type_results['recall'],
            'specificity': action_type_results['specificity'],
            'f_score': action_type_results['f_score']
        },
        'card_selection': {
            'accuracy': card_sel_results['accuracy'],
            'precision': card_sel_results['precision'],
            'recall': card_sel_results['recall'],
            'specificity': card_sel_results['specificity'],
            'f_score': card_sel_results['f_score']
        },
        'card_reservation': {
            'accuracy': card_res_results['accuracy'],
            'precision': card_res_results['precision'],
            'recall': card_res_results['recall'],
            'specificity': card_res_results['specificity'],
            'f_score': card_res_results['f_score']
        },
        'gem_take3': {
            'accuracy': gem3_results['accuracy'],
            'precision': gem3_results['precision'],
            'recall': gem3_results['recall'],
            'specificity': gem3_results['specificity'],
            'f_score': gem3_results['f_score']
        },
        'gem_take2': {
            'accuracy': gem2_results['accuracy'],
            'precision': gem2_results['precision'],
            'recall': gem2_results['recall'],
            'specificity': gem2_results['specificity'],
            'f_score': gem2_results['f_score']
        },
        'noble': {
            'accuracy': noble_results['accuracy'],
            'precision': noble_results['precision'],
            'recall': noble_results['recall'],
            'specificity': noble_results['specificity'],
            'f_score': noble_results['f_score']
        },
        'gems_removed': {
            'accuracy': gems_rem_results['accuracy'],
            'precision': gems_rem_results['precision'],
            'recall': gems_rem_results['recall'],
            'specificity': gems_rem_results['specificity'],
            'f_score': gems_rem_results['f_score']
        }
    }
    metrics_path = os.path.join(results_dir, f'classification_metrics_{args.split}.png')
    plot_classification_metrics(metrics_dict, metrics_path)
    print(f"  Saved classification metrics to {metrics_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"Evaluation Summary ({args.split} split)")
    print(f"{'='*60}")
    print(f"Action type accuracy: {action_type_results['overall_accuracy']:.4f}")
    print(f"Overall action accuracy: {overall_acc:.4f}")
    print(f"Random baseline: {baseline_results['action_type_accuracy']:.4f}")
    print(f"Improvement over random: {(action_type_results['overall_accuracy'] - baseline_results['action_type_accuracy'])*100:.1f}%")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
