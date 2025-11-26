"""
Evaluate multiple trained models on the validation set.

This script loads best_model.pth from multiple checkpoint directories,
evaluates them on the validation set, and reports accuracy metrics for each head
in both detailed CLI output and LaTeX table format.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from imitation_learning.model import MultiHeadSplendorNet
from imitation_learning.dataset import create_dataloaders
from imitation_learning.utils import compute_accuracy, compute_masked_predictions
from tqdm import tqdm


# ============================================================================
# CONFIGURATION - EDIT THESE PATHS
# ============================================================================
# List of model checkpoint directories (each contains config.yaml and best_model.pth)
MODEL_DIRS = [
    # "data/models/202511261002_config_small",     # <- baseline
    # "data/models/202511261024_config_medium",
    # "data/models/202511261041_config_large",
    # "data/models/202511230802_config_small",     # <- masking of illegal actions
    # "data/models/202511230816_config_medium",
    # "data/models/202511230830_config_large",
    "data/models/202511240928_config_small",     # <- feature engineering
    "data/models/202511241025_config_tuning",    # <- 512,512,256  128,64   100,10
    "data/models/202511241159_config_tuning",    # <- 768,512,256  256,128,64   150,20
    "data/models/202511250809_config_tuning",    # <- deactivate card_reservation, nobe, gems_removed
]
# ============================================================================


def load_model_checkpoint(checkpoint_path: str, config: Dict, device: torch.device) -> MultiHeadSplendorNet:
    """Load model from checkpoint."""
    model = MultiHeadSplendorNet(config['model'])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def evaluate_model(
    model: MultiHeadSplendorNet,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluate model on validation set.

    Returns:
        Dict mapping head name to accuracy
    """
    model.eval()

    # Track all heads
    accuracies_accum = {
        'action_type': [],
        'card_selection': [],
        'card_reservation': [],
        'gem_take3': [],
        'gem_take2': [],
        'noble': [],
        'gems_removed': []
    }

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Evaluating', leave=False)

        for states, labels, masks in pbar:
            # Move to device
            states = states.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}
            masks = {k: v.to(device) for k, v in masks.items()}

            # Forward pass
            outputs = model(states)

            # Compute accuracies with masked predictions
            action_types = labels['action_type'].cpu().numpy()

            # Action type accuracy (always compute) - apply legal action mask
            action_type_preds = compute_masked_predictions(
                outputs['action_type'], masks['action_type']
            ).cpu().numpy()
            accuracies_accum['action_type'].append(
                compute_accuracy(action_type_preds, action_types)
            )

            # Card selection accuracy (only for BUILD actions) - apply legal action mask
            build_mask = action_types == 0
            card_sel_labels = labels['card_selection'].cpu().numpy()
            card_sel_valid = card_sel_labels != -1
            card_sel_mask = build_mask & card_sel_valid
            if card_sel_mask.any():
                card_sel_preds = compute_masked_predictions(
                    outputs['card_selection'], masks['card_selection']
                ).cpu().numpy()
                accuracies_accum['card_selection'].append(
                    compute_accuracy(card_sel_preds, card_sel_labels, card_sel_mask)
                )

            # Gem take3 accuracy - apply legal action mask
            take3_mask = action_types == 3
            gem3_labels = labels['gem_take3'].cpu().numpy()
            gem3_valid = gem3_labels != -1
            gem3_mask = take3_mask & gem3_valid
            if gem3_mask.any():
                gem3_preds = compute_masked_predictions(
                    outputs['gem_take3'], masks['gem_take3']
                ).cpu().numpy()
                accuracies_accum['gem_take3'].append(
                    compute_accuracy(gem3_preds, gem3_labels, gem3_mask)
                )

            # Gem take2 accuracy - apply legal action mask
            take2_mask = action_types == 2
            gem2_labels = labels['gem_take2'].cpu().numpy()
            gem2_valid = gem2_labels != -1
            gem2_mask = take2_mask & gem2_valid
            if gem2_mask.any():
                gem2_preds = compute_masked_predictions(
                    outputs['gem_take2'], masks['gem_take2']
                ).cpu().numpy()
                accuracies_accum['gem_take2'].append(
                    compute_accuracy(gem2_preds, gem2_labels, gem2_mask)
                )

            # Card reservation accuracy (only for RESERVE actions) - apply legal action mask
            reserve_mask = action_types == 1
            card_res_labels = labels['card_reservation'].cpu().numpy()
            card_res_valid = card_res_labels != -1
            card_res_mask = reserve_mask & card_res_valid
            if card_res_mask.any():
                card_res_preds = compute_masked_predictions(
                    outputs['card_reservation'], masks['card_reservation']
                ).cpu().numpy()
                accuracies_accum['card_reservation'].append(
                    compute_accuracy(card_res_preds, card_res_labels, card_res_mask)
                )

            # Noble accuracy - apply legal action mask
            noble_labels = labels['noble'].cpu().numpy()
            noble_valid = noble_labels != -1
            noble_mask = build_mask & noble_valid
            if noble_mask.any():
                noble_preds = compute_masked_predictions(
                    outputs['noble'], masks['noble']
                ).cpu().numpy()
                accuracies_accum['noble'].append(
                    compute_accuracy(noble_preds, noble_labels, noble_mask)
                )

            # Gems removed accuracy - apply legal action mask
            gems_rem_labels = labels['gems_removed'].cpu().numpy()
            gems_rem_mask = gems_rem_labels != 0
            if gems_rem_mask.any():
                gems_rem_preds = compute_masked_predictions(
                    outputs['gems_removed'], masks['gems_removed']
                ).cpu().numpy()
                accuracies_accum['gems_removed'].append(
                    compute_accuracy(gems_rem_preds, gems_rem_labels, gems_rem_mask)
                )

    # Average accuracies
    avg_accuracies = {
        k: np.mean(v) if v else 0.0
        for k, v in accuracies_accum.items()
    }

    return avg_accuracies


def print_results_cli(results: Dict[str, Dict[str, float]]):
    """Print results in detailed CLI format."""
    print("\n" + "="*80)
    print("VALIDATION SET EVALUATION RESULTS")
    print("="*80)

    for model_name, accuracies in results.items():
        print(f"\nModel: {model_name}")
        print("-" * 80)
        for head_name, accuracy in accuracies.items():
            print(f"  {head_name:20s}: {accuracy:7.4f} ({accuracy*100:6.2f}%)")

        # Calculate average accuracy across heads
        avg_acc = np.mean(list(accuracies.values()))
        print(f"  {'Average':20s}: {avg_acc:7.4f} ({avg_acc*100:6.2f}%)")

    print("\n" + "="*80 + "\n")


def print_results_latex(results: Dict[str, Dict[str, float]]):
    """Print results as LaTeX table."""
    print("\n" + "="*80)
    print("LATEX TABLE")
    print("="*80 + "\n")

    # Get all head names from first model
    head_names = list(next(iter(results.values())).keys())

    # Print table header
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Validation Accuracy by Model and Prediction Head}")
    print("\\label{tab:model_comparison}")
    print("\\begin{tabular}{l" + "c" * len(head_names) + "c}")
    print("\\toprule")

    # Column headers
    headers = ["Model"] + [h.replace("_", " ").title() for h in head_names] + ["Average"]
    print(" & ".join(headers) + " \\\\")
    print("\\midrule")

    # Data rows
    for model_name, accuracies in results.items():
        # Shorten model name for table
        short_name = model_name.replace("_config_", " ").replace("checkpoints/", "")

        row = [short_name]
        for head_name in head_names:
            acc = accuracies[head_name]
            row.append(f"{acc*100:.2f}\\%")

        # Add average
        avg_acc = np.mean(list(accuracies.values()))
        row.append(f"\\textbf{{{avg_acc*100:.2f}\\%}}")

        print(" & ".join(row) + " \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

    print("\n" + "="*80 + "\n")


def main():
    """Main evaluation loop."""
    import json

    # Evaluate each model
    results = {}

    for model_dir in MODEL_DIRS:
        model_path = os.path.join(model_dir, 'best_model.pth')
        config_path = os.path.join(model_dir, 'config.yaml')

        if not os.path.exists(model_path):
            print(f"WARNING: Model not found at {model_path}, skipping...")
            continue

        if not os.path.exists(config_path):
            print(f"WARNING: Config not found at {config_path}, skipping...")
            continue

        print(f"\n{'='*80}")
        print(f"Evaluating: {model_dir}")
        print(f"Config: {config_path}")
        print(f"{'='*80}")

        try:
            # Load config from the model directory
            print(f"Loading configuration from {config_path}...")
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            # Set device
            device = torch.device(config['compute']['device'] if torch.cuda.is_available() else 'cpu')
            print(f"Using device: {device}")

            # Create validation dataloader
            print("Loading validation dataset...")
            _, val_loader, _ = create_dataloaders(
                config['data']['processed_dir'],
                config['training']['batch_size'],
                config['compute']['num_workers']
            )
            print(f"Validation set size: {len(val_loader.dataset)} samples")

            # Load preprocessing stats for input_dim
            stats_path = os.path.join(config['data']['processed_dir'], 'preprocessing_stats.json')
            with open(stats_path, 'r') as f:
                stats = json.load(f)
            config['model']['input_dim'] = stats['input_dim']

            # Load model
            model = load_model_checkpoint(model_path, config, device)
            print(f"Model loaded successfully from {model_path}")

            # Evaluate
            accuracies = evaluate_model(model, val_loader, device)

            results[model_dir] = accuracies

            print(f"✓ Evaluation complete")

        except Exception as e:
            print(f"ERROR: Failed to evaluate {model_dir}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not results:
        print("\nNo models were successfully evaluated!")
        return

    # Print results
    print_results_cli(results)
    print_results_latex(results)

    print("Evaluation complete!")


if __name__ == "__main__":
    main()
