"""Standalone script for merging preprocessed batch files.

This script loads batch files created by the parallel preprocessing pipeline,
combines them into final arrays, performs splitting, normalization, validation,
and saves the final preprocessed data.

This allows:
- Re-running merge with different strategies without reprocessing files
- Debugging merge issues independently
- Inspecting intermediate batch files
- Iterating on merge logic quickly

Usage:
    python -m src.imitation_learning.merge_batches --config config.yaml
    python -m src.imitation_learning.merge_batches --config config.yaml --cleanup
    python -m src.imitation_learning.merge_batches --intermediate-dir data/custom
"""

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .constants import CLASS_TO_COMBO_TAKE3, CLASS_TO_REMOVAL
from .data_preprocessing import (
    create_normalization_mask,
    get_all_feature_names,
    identify_column_groups,
    load_config,
    normalize_features,
    save_preprocessed_data,
    set_seed,
    split_by_game_id,
    validate_masks,
)
from .parallel_processor import delete_batch_files, load_batch_from_file


def log_memory_usage(label: str = ""):
    """Log current memory usage."""
    try:
        import psutil

        process = psutil.Process()
        mem_info = process.memory_info()
        mem_gb = mem_info.rss / 1024 / 1024 / 1024
        print(f"[Memory] {label}: {mem_gb:.2f} GB")
    except ImportError:
        pass


def discover_batch_files(intermediate_dir: str) -> List[str]:
    """Discover and sort batch files in intermediate directory.

    Args:
        intermediate_dir: Directory containing batch_*.npz files

    Returns:
        List of batch file paths, sorted by batch number

    """
    intermediate_path = Path(intermediate_dir)
    if not intermediate_path.exists():
        raise ValueError(f"Intermediate directory does not exist: {intermediate_dir}")

    # Find all batch files
    batch_files = list(intermediate_path.glob("batch_*.npz"))

    if not batch_files:
        raise ValueError(f"No batch files found in {intermediate_dir}")

    # Sort by batch number (extract number from batch_XXXX.npz)
    def get_batch_number(path: Path) -> int:
        stem = path.stem  # "batch_0000"
        number_part = stem.split("_")[1]  # "0000"
        return int(number_part)

    batch_files_sorted = sorted(batch_files, key=get_batch_number)

    print(f"Found {len(batch_files_sorted)} batch files in {intermediate_dir}")

    return [str(f) for f in batch_files_sorted]


def merge_batches(
    batch_file_paths: List[str], config: Dict, monitor_memory: bool = False,
) -> Tuple:
    """Merge batch files into combined arrays.

    Args:
        batch_file_paths: List of paths to batch files
        config: Configuration dictionary
        monitor_memory: Whether to log memory usage

    Returns:
        Tuple of (df_compacted, strategic_features_list, labels_list, masks_list)

    """
    print(f"\n{'=' * 60}")
    print(f"Merging {len(batch_file_paths)} batches...")
    print(f"{'=' * 60}")

    if monitor_memory:
        log_memory_usage("Before batch merging")

    df_compacted_list = []
    strategic_features_list = []
    labels_list = []
    masks_list = []

    for i, batch_file in enumerate(batch_file_paths):
        print(f"\nLoading batch {i + 1}/{len(batch_file_paths)}...")

        # Load batch
        batch_df, batch_features, batch_labels, batch_masks = load_batch_from_file(
            batch_file,
        )

        if monitor_memory:
            log_memory_usage(f"After loading batch {i + 1}")

        # Append to final lists
        df_compacted_list.append(batch_df)
        strategic_features_list.extend(batch_features)
        labels_list.extend(batch_labels)
        masks_list.extend(batch_masks)

        # Clear batch data from memory
        del batch_df, batch_features, batch_labels, batch_masks
        gc.collect()

        if monitor_memory:
            log_memory_usage(f"After merging batch {i + 1}")

    # Combine all batch DataFrames
    print(f"\nCombining {len(df_compacted_list)} DataFrames...")
    if df_compacted_list:
        df_compacted = pd.concat(df_compacted_list, axis=0, ignore_index=True)
    else:
        df_compacted = pd.DataFrame()

    # Clear DataFrame list
    del df_compacted_list
    gc.collect()

    if len(df_compacted) == 0:
        raise ValueError("No samples were successfully processed!")

    if monitor_memory:
        log_memory_usage("After batch merging complete")

    print(f"\nTotal samples processed: {len(df_compacted):,}")

    return df_compacted, strategic_features_list, labels_list, masks_list


def process_merged_data(
    df_compacted: pd.DataFrame,
    strategic_features_list: List[Dict],
    labels_list: List[Dict],
    masks_list: List[Dict],
    config: Dict,
    skip_validation: bool = False,
) -> None:
    """Process merged data: feature engineering, normalization, validation, and saving.

    Args:
        df_compacted: Combined DataFrame from all batches
        strategic_features_list: Combined strategic features from all batches
        labels_list: Combined labels from all batches
        masks_list: Combined masks from all batches
        config: Configuration dictionary
        skip_validation: Whether to skip mask validation

    """
    # Convert strategic features list to DataFrame
    strategic_df = pd.DataFrame(strategic_features_list)

    # Get expected feature names and fill missing columns with zeros
    expected_feature_names = get_all_feature_names()
    for feat_name in expected_feature_names:
        if feat_name not in strategic_df.columns:
            strategic_df[feat_name] = 0.0

    # Reorder columns to match expected order
    strategic_df = strategic_df[expected_feature_names]

    # Add strategic features to dataframe
    df_eng = pd.concat(
        [df_compacted.reset_index(drop=True), strategic_df.reset_index(drop=True)],
        axis=1,
    )

    # Identify column groups (use df_compacted which has raw features + position features)
    metadata_cols, label_cols, feature_cols = identify_column_groups(df_compacted)

    # Engineer one-hot features (current_player, num_players, positions)
    print("\nEngineering one-hot features...")
    onehot_cols = []

    # One-hot encode current_player
    for i in range(4):
        col_name = f"current_player_{i}"
        df_eng[col_name] = (df_eng["current_player"] == i).astype(int)
        onehot_cols.append(col_name)

    # One-hot encode num_players
    for n in [2, 3, 4]:
        col_name = f"num_players_{n}"
        df_eng[col_name] = (df_eng["num_players"] == n).astype(int)
        onehot_cols.append(col_name)

    # One-hot encode player positions
    for player_idx in range(4):
        position_col = f"player{player_idx}_position"
        if position_col in df_eng.columns:
            for pos in range(4):
                col_name = f"{position_col}_{pos}"
                df_eng[col_name] = (df_eng[position_col] == pos).astype(int)
                onehot_cols.append(col_name)

    # Update feature columns
    player_position_cols = [f"player{i}_position" for i in range(4)]
    new_feature_cols = [
        col
        for col in feature_cols
        if col not in ["current_player", "num_players"]
        and col not in player_position_cols
    ]
    new_feature_cols.extend(onehot_cols)

    # Add turn_number as feature
    if "turn_number" not in new_feature_cols:
        new_feature_cols.append("turn_number")

    # Add strategic features (these are already in df_eng from the concat above)
    strategic_cols = list(strategic_df.columns)
    new_feature_cols.extend(strategic_cols)

    print(f"  Total features after engineering: {len(new_feature_cols)}")

    # Verify all feature columns exist in df_eng
    missing_cols = [col for col in new_feature_cols if col not in df_eng.columns]
    if missing_cols:
        print(f"  WARNING: Missing columns: {missing_cols[:10]}")
        # Filter to only existing columns
        new_feature_cols = [col for col in new_feature_cols if col in df_eng.columns]
        print(f"  Adjusted to {len(new_feature_cols)} features")

    # Convert labels list to dict of arrays
    labels_all = {}
    head_names = [
        "action_type",
        "card_selection",
        "card_reservation",
        "gem_take3",
        "gem_take2",
        "noble",
        "gems_removed",
    ]
    for head in head_names:
        labels_all[head] = np.array([labels[head] for labels in labels_list])

    # Convert masks list to dict of arrays
    masks_all = {}
    for head in head_names:
        masks_all[head] = np.stack([masks[head] for masks in masks_list], axis=0)

    # Split by game_id
    train_idx, val_idx, test_idx = split_by_game_id(
        df_eng,
        config["data"]["train_ratio"],
        config["data"]["val_ratio"],
        config["data"]["test_ratio"],
        config["seed"],
    )

    # Extract features
    print("\nExtracting features...")
    print(f"  DataFrame shape: {df_eng.shape}")
    print(f"  Feature columns count: {len(new_feature_cols)}")
    print("  Checking if all feature columns exist in DataFrame...")

    # Instead of tracking feature columns manually, just use all columns that aren't metadata/labels
    # This avoids issues with duplicates and missing columns
    exclude_cols = set(metadata_cols + label_cols)
    final_feature_cols = [col for col in df_eng.columns if col not in exclude_cols]

    # Check for duplicate columns in DataFrame
    if len(df_eng.columns) != len(set(df_eng.columns)):
        duplicates = [
            col for col in df_eng.columns if list(df_eng.columns).count(col) > 1
        ]
        unique_duplicates = list(set(duplicates))
        print(
            f"  WARNING: Found {len(unique_duplicates)} duplicate columns: {unique_duplicates[:5]}",
        )
        # Remove duplicates by keeping first occurrence
        df_eng = df_eng.loc[:, ~df_eng.columns.duplicated()]
        # Recalculate final_feature_cols
        final_feature_cols = [col for col in df_eng.columns if col not in exclude_cols]

    print(
        f"  Final feature columns count: {len(final_feature_cols)} (all non-metadata/label columns)",
    )

    X_all = df_eng[final_feature_cols].values
    print(f"  X_all shape: {X_all.shape}")

    # Create normalization mask AFTER we know the actual feature list
    norm_mask = create_normalization_mask(
        final_feature_cols, onehot_cols, strategic_cols,
    )
    print(f"  Normalization mask shape: {norm_mask.shape}")

    X_train = X_all[train_idx]
    X_val = X_all[val_idx]
    X_test = X_all[test_idx]

    # Split labels
    labels_train = {k: v[train_idx] for k, v in labels_all.items()}
    labels_val = {k: v[val_idx] for k, v in labels_all.items()}
    labels_test = {k: v[test_idx] for k, v in labels_all.items()}

    # Split masks
    masks_train = {k: v[train_idx] for k, v in masks_all.items()}
    masks_val = {k: v[val_idx] for k, v in masks_all.items()}
    masks_test = {k: v[test_idx] for k, v in masks_all.items()}

    # Validate masks on training set (unless skipped)
    if not skip_validation:
        df_for_validation = df_eng.iloc[train_idx].reset_index(drop=True)
        validation_report = validate_masks(masks_train, labels_train, df_for_validation)

        # Save validation report
        os.makedirs(config["data"]["processed_dir"], exist_ok=True)
        with open(
            os.path.join(
                config["data"]["processed_dir"], "mask_validation_report.json",
            ),
            "w",
        ) as f:
            json.dump(validation_report, f, indent=2)
        print("\n  Validation report saved")
    else:
        print("\n  Skipping mask validation (--skip-validation flag set)")

    # Normalize features
    X_train_norm, X_val_norm, X_test_norm, scaler = normalize_features(
        X_train, X_val, X_test, norm_mask,
    )

    # Prepare label mappings using pre-computed constants
    label_mappings = {
        "action_type": {
            "build": 0,
            "reserve": 1,
            "take 2 tokens": 2,
            "take 3 tokens": 3,
        },
        "gem_take3_classes": {str(k): list(v) for k, v in CLASS_TO_COMBO_TAKE3.items()},
        "gem_removal_classes": {str(k): list(v) for k, v in CLASS_TO_REMOVAL.items()},
    }

    # Save everything
    save_preprocessed_data(
        X_train_norm,
        X_val_norm,
        X_test_norm,
        labels_train,
        labels_val,
        labels_test,
        masks_train,
        masks_val,
        masks_test,
        scaler,
        final_feature_cols,
        label_mappings,
        config["data"]["processed_dir"],
    )

    print("\n✓ Merge and processing complete!")
    print(f"\n{'=' * 60}")
    print("Summary:")
    print(f"  Total samples: {len(df_eng):,}")
    print(f"  Input dimension: {X_train_norm.shape[1]}")
    print(f"  Train/Val/Test: {len(X_train):,} / {len(X_val):,} / {len(X_test):,}")
    print(f"{'=' * 60}\n")


def main():
    """Main entry point for standalone merge script."""
    parser = argparse.ArgumentParser(
        description="Merge preprocessed batch files into final arrays",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file",
    )
    parser.add_argument(
        "--intermediate-dir",
        type=str,
        default=None,
        help="Override intermediate directory (default: from config)",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete batch files after successful merge (default: False)",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip mask validation for faster iteration (default: False)",
    )
    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from {args.config}...")
    config = load_config(args.config)

    # Set seed for reproducibility
    set_seed(config["seed"])
    print(f"Set random seed: {config['seed']}")

    # Determine intermediate directory
    if args.intermediate_dir:
        intermediate_dir = args.intermediate_dir
    else:
        intermediate_dir = config.get("preprocessing", {}).get(
            "intermediate_dir", "data/intermediate",
        )

    print(f"Using intermediate directory: {intermediate_dir}")

    # Monitor memory if enabled
    monitor_memory = config.get("preprocessing", {}).get("monitor_memory", False)

    # Discover batch files
    batch_file_paths = discover_batch_files(intermediate_dir)

    # Merge batches
    df_compacted, strategic_features_list, labels_list, masks_list = merge_batches(
        batch_file_paths, config, monitor_memory=monitor_memory,
    )

    # Process merged data
    process_merged_data(
        df_compacted,
        strategic_features_list,
        labels_list,
        masks_list,
        config,
        skip_validation=args.skip_validation,
    )

    # Handle cleanup
    if args.cleanup:
        print("\nCleaning up batch files...")
        try:
            delete_batch_files(batch_file_paths)
            print(f"✓ Deleted {len(batch_file_paths)} batch files")
        except Exception as e:
            print(f"Warning: Failed to clean up batch files: {e}")
    else:
        print(f"\nBatch files kept in: {intermediate_dir}")
        print("  To clean up manually, run with --cleanup flag")


if __name__ == "__main__":
    main()
