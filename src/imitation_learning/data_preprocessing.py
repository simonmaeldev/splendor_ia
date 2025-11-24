"""Data preprocessing pipeline for Splendor imitation learning.

This module loads raw CSV files from MCTS self-play games, performs feature engineering,
encodes labels into classification tasks, splits data at game-level, normalizes features,
and saves preprocessed arrays ready for training.

Pipeline steps:
1. Load CSV files (preserving NaN for game state reconstruction)
2. Generate legal action masks (requires NaN values)
3. Fill NaN values with 0 (for feature engineering)
4. Compact visible cards and add position indices (NEW: improves positional invariance)
5. Feature engineering (one-hot encoding categorical features)
6. Normalization (excluding position indices and binary features)
7. Split by game_id and save preprocessed arrays

After card compaction, feature count changes:
- Old: 382 base features → ~450 after one-hot encoding
- New: 406 base features → ~474 after one-hot encoding (+24 position features)

Usage:
    python data_preprocessing.py --config config.yaml
"""

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from .constants import (
    COMBO_TO_CLASS_TAKE3,
    REMOVAL_TO_CLASS,
)
from .feature_engineering import extract_all_features, get_all_feature_names
from .utils import (
    encode_gem_take2,
    encode_gem_take3,
    encode_gems_removed,
    generate_all_masks_from_row,
    get_num_gem_removal_classes,
    set_seed,
)



def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def load_single_game(csv_path: str) -> pd.DataFrame:
    """Load a single game CSV file.

    Args:
        csv_path: Path to CSV file

    Returns:
        Dataframe with game_id column added

    Example:
        >>> df = load_single_game("data/games/3_games/5444.csv")
        >>> print(f"Loaded {len(df)} samples")

    """
    print(f"\nLoading single CSV file: {csv_path}...")

    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise ValueError(f"File not found: {csv_path}")

    df = pd.read_csv(csv_file)

    # Extract game_id from filename (e.g., "5444.csv" -> 5444)
    game_id = int(csv_file.stem)
    df["game_id"] = game_id

    print(f"Total samples loaded: {len(df)}")
    print(f"Game ID: {game_id}")

    return df


def load_all_games(data_root: str, max_games: int = None) -> pd.DataFrame:
    """Load all game CSVs from 2_games/, 3_games/, and 4_games/ directories.

    Args:
        data_root: Root directory containing game subdirectories
        max_games: Optional limit on number of games to load (for testing)

    Returns:
        Concatenated dataframe with game_id column added

    Example:
        >>> df = load_all_games("data/games")
        >>> print(f"Loaded {len(df)} samples from {df['game_id'].nunique()} games")

    """
    all_dfs = []
    total_loaded = 0

    for subdir in ["2_games", "3_games", "4_games"]:
        if max_games is not None and total_loaded >= max_games:
            break

        dir_path = Path(data_root) / subdir
        if not dir_path.exists():
            print(f"Warning: {dir_path} does not exist, skipping...")
            continue

        csv_files = list(dir_path.glob("*.csv"))

        # Limit files if max_games specified
        if max_games is not None:
            remaining = max_games - total_loaded
            csv_files = csv_files[:remaining]

        print(f"\nLoading {len(csv_files)} CSV files from {subdir}...")

        for csv_file in tqdm(csv_files, desc=f"Loading {subdir}"):
            try:
                df = pd.read_csv(csv_file)

                # Extract game_id from filename (e.g., "1.csv" -> 1)
                game_id = int(csv_file.stem)
                df["game_id"] = game_id

                all_dfs.append(df)
                total_loaded += 1

                if max_games is not None and total_loaded >= max_games:
                    break

            except Exception as e:
                print(f"Error loading {csv_file}: {e}")
                continue

    if not all_dfs:
        raise ValueError("No CSV files loaded successfully!")

    # Concatenate all dataframes
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # NOTE: Do NOT fill NaN here! The mask generation needs the original NaN values
    # to correctly reconstruct the game state. Filling NaN with 0 adds phantom nobles,
    # reserved cards, etc. that change what moves are legal.
    #
    # fillna will be done AFTER mask generation in main()

    print(f"\nTotal samples loaded: {len(combined_df):,}")
    print(f"Total games loaded: {combined_df['game_id'].nunique():,}")

    return combined_df


def fill_nan_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN values in feature columns (but keep label columns as NaN).

    This MUST be called AFTER mask generation, since masks need the original
    NaN values to correctly reconstruct game state.

    Args:
        df: DataFrame with NaN values

    Returns:
        DataFrame with NaN filled for feature columns only

    """
    print("\nFilling NaN values...")
    print(f"  Before fillna: {df.isna().sum().sum()} NaN values")

    # Label columns that should keep NaN (will be converted to -1 later)
    label_cols_to_keep_nan = [
        "card_selection",
        "card_reservation",
        "noble_selection",
        "gem_take3_white",
        "gem_take3_blue",
        "gem_take3_green",
        "gem_take3_red",
        "gem_take3_black",
        "gem_take2_white",
        "gem_take2_blue",
        "gem_take2_green",
        "gem_take2_red",
        "gem_take2_black",
        "gems_removed_white",
        "gems_removed_blue",
        "gems_removed_green",
        "gems_removed_red",
        "gems_removed_black",
        "gems_removed_gold",
    ]

    # Fill NaN for non-label columns (structured padding for fewer players/nobles)
    df_filled = df.copy()
    for col in df_filled.columns:
        if col not in label_cols_to_keep_nan and col != "action_type":
            df_filled[col] = df_filled[col].fillna(0)

    print(
        f"  After fillna: {df_filled.isna().sum().sum()} NaN values (label columns only)",
    )

    return df_filled


def compact_cards_and_add_position(df: pd.DataFrame) -> pd.DataFrame:
    """Compact visible cards (move zeros to end) and add position index feature.

    This function:
    1. Identifies non-zero visible cards (cards 0-11)
    2. Reorders them: non-zero first, zeros at end
    3. Adds position index feature: 0, 1, 2, ... for non-zero; -1 for zeros
    4. For reserved cards: adds position 12, 13, 14 per player; -1 for missing

    Args:
        df: DataFrame with filled NaN values (all zeros for missing cards)

    Returns:
        DataFrame with compacted cards and position features added

    """
    print("\nCompacting cards and adding position indices...")

    df_compacted = df.copy()

    # Card feature names (excluding position which we'll add)
    card_feature_names = [
        "vp",
        "level",
        "cost_white",
        "cost_blue",
        "cost_green",
        "cost_red",
        "cost_black",
        "bonus_white",
        "bonus_blue",
        "bonus_green",
        "bonus_red",
        "bonus_black",
    ]

    # Process visible cards (card0 to card11)
    print("  Processing visible cards (0-11)...")

    # Extract all visible card features as a 3D array: (n_samples, 12 cards, 12 features)
    visible_card_data = []
    for card_idx in range(12):
        card_cols = [f"card{card_idx}_{feat}" for feat in card_feature_names]
        card_values = df_compacted[card_cols].values  # (n_samples, 12)
        visible_card_data.append(card_values)

    # Stack to get (n_samples, 12 cards, 12 features)
    visible_card_data = np.stack(visible_card_data, axis=1)

    # Identify non-zero cards: a card is non-zero if ANY feature is non-zero
    # Shape: (n_samples, 12)
    is_nonzero = np.any(visible_card_data != 0, axis=2)

    # For each sample, compact cards
    n_samples = len(df_compacted)
    compacted_visible = np.zeros_like(visible_card_data)
    position_indices = np.full((n_samples, 12), -1, dtype=int)

    for sample_idx in range(n_samples):
        # Get indices of non-zero and zero cards for this sample
        nonzero_indices = np.where(is_nonzero[sample_idx])[0]
        zero_indices = np.where(~is_nonzero[sample_idx])[0]

        # Combine: non-zero first, then zeros
        reordered_indices = np.concatenate([nonzero_indices, zero_indices])

        # Reorder card features
        compacted_visible[sample_idx] = visible_card_data[sample_idx, reordered_indices]

        # Assign position indices: 0, 1, 2, ... for non-zero, -1 for zeros
        position_indices[sample_idx, : len(nonzero_indices)] = np.arange(
            len(nonzero_indices),
        )

    # Write back to dataframe
    # First, drop old card columns
    old_card_cols = []
    for card_idx in range(12):
        old_card_cols.extend([f"card{card_idx}_{feat}" for feat in card_feature_names])
    df_compacted = df_compacted.drop(columns=old_card_cols)

    # Build new columns efficiently using dict to avoid fragmentation
    new_cols = {}
    for card_idx in range(12):
        # Add position column
        new_cols[f"card{card_idx}_position"] = position_indices[:, card_idx]

        # Add feature columns
        for feat_idx, feat_name in enumerate(card_feature_names):
            new_cols[f"card{card_idx}_{feat_name}"] = compacted_visible[
                :,
                card_idx,
                feat_idx,
            ]

    # Concatenate all new columns at once
    df_compacted = pd.concat([df_compacted, pd.DataFrame(new_cols)], axis=1)

    print(f"    Compacted {np.sum(is_nonzero)} non-zero visible cards")

    # Process reserved cards (3 per player, positions 12, 13, 14)
    print("  Processing reserved cards (3 per player)...")

    # Collect all old and new columns for reserved cards
    old_reserved_cols = []
    new_reserved_cols = {}

    for player_idx in range(4):
        for reserved_idx in range(3):
            # Extract reserved card features
            reserved_cols = [
                f"player{player_idx}_reserved{reserved_idx}_{feat}"
                for feat in card_feature_names
            ]
            reserved_values = df_compacted[reserved_cols].values  # (n_samples, 12)
            old_reserved_cols.extend(reserved_cols)

            # Check if card is present (any non-zero feature)
            is_present = np.any(reserved_values != 0, axis=1)  # (n_samples,)

            # Assign position: 12, 13, or 14 if present, -1 if missing
            position = 12 + reserved_idx
            position_col = np.where(is_present, position, -1)

            # Add position column first
            col_prefix = f"player{player_idx}_reserved{reserved_idx}"
            new_reserved_cols[f"{col_prefix}_position"] = position_col

            # Add back feature columns
            for feat_idx, feat_name in enumerate(card_feature_names):
                new_reserved_cols[f"{col_prefix}_{feat_name}"] = reserved_values[
                    :,
                    feat_idx,
                ]

    # Drop old columns and add new ones efficiently
    df_compacted = df_compacted.drop(columns=old_reserved_cols)
    df_compacted = pd.concat([df_compacted, pd.DataFrame(new_reserved_cols)], axis=1)

    print("  Added position indices to all cards")
    print("  Visible cards: 12 cards × 13 features (position + 12) = 156 features")
    print("  Reserved cards per player: 3 cards × 13 features = 39 features")

    return df_compacted


def identify_column_groups(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """Identify column groups: metadata, labels, features.

    Args:
        df: DataFrame with all columns

    Returns:
        Tuple of (metadata_cols, label_cols, feature_cols)

    """
    all_cols = set(df.columns)

    # Metadata columns
    metadata_cols = ["game_id", "turn_number", "current_player", "num_players"]
    # Add playerX_position columns
    metadata_cols.extend([f"player{i}_position" for i in range(4)])

    # Label columns
    label_cols = [
        "action_type",
        "card_selection",
        "card_reservation",
        "noble_selection",
    ]
    # Add gem_take3 columns
    label_cols.extend(
        [f"gem_take3_{color}" for color in ["white", "blue", "green", "red", "black"]],
    )
    # Add gem_take2 columns
    label_cols.extend(
        [f"gem_take2_{color}" for color in ["white", "blue", "green", "red", "black"]],
    )
    # Add gems_removed columns
    label_cols.extend(
        [
            f"gems_removed_{color}"
            for color in ["white", "blue", "green", "red", "black", "gold"]
        ],
    )

    # Filter to only include columns that exist in the dataframe
    metadata_cols = [col for col in metadata_cols if col in all_cols]
    label_cols = [col for col in label_cols if col in all_cols]

    # Feature columns are everything else (excluding game_id which we'll drop from features)
    feature_cols = list(all_cols - set(metadata_cols) - set(label_cols))

    print("\nColumn groups identified:")
    print(f"  Metadata: {len(metadata_cols)} columns")
    print(f"  Labels: {len(label_cols)} columns")
    print(f"  Features: {len(feature_cols)} columns")

    return metadata_cols, label_cols, feature_cols


def engineer_features(
    df: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[pd.DataFrame, List[str], List[str], List[str]]:
    """Perform feature engineering including one-hot encoding and strategic features.

    Args:
        df: DataFrame with all data
        feature_cols: List of feature column names

    Returns:
        Tuple of (engineered_df, new_feature_cols, onehot_col_names, strategic_col_names)

    """
    print("\nEngineering features...")

    # Create a copy to avoid modifying original
    df_eng = df.copy()

    onehot_cols = []

    # One-hot encode current_player (0 to num_players-1, pad to 4)
    print("  One-hot encoding current_player...")
    for i in range(4):
        col_name = f"current_player_{i}"
        df_eng[col_name] = (df_eng["current_player"] == i).astype(int)
        onehot_cols.append(col_name)

    # One-hot encode num_players (2, 3, 4)
    print("  One-hot encoding num_players...")
    for n in [2, 3, 4]:
        col_name = f"num_players_{n}"
        df_eng[col_name] = (df_eng["num_players"] == n).astype(int)
        onehot_cols.append(col_name)

    # One-hot encode playerX_position (0 to num_players-1, pad to 4)
    print("  One-hot encoding player positions...")
    for player_idx in range(4):
        position_col = f"player{player_idx}_position"
        if position_col in df_eng.columns:
            for pos in range(4):
                col_name = f"{position_col}_{pos}"
                df_eng[col_name] = (df_eng[position_col] == pos).astype(int)
                onehot_cols.append(col_name)

    # Update feature columns: remove original categorical, add one-hot
    # Note: We keep card/reserved position features but exclude player position metadata
    player_position_cols = [f"player{i}_position" for i in range(4)]
    new_feature_cols = [
        col
        for col in feature_cols
        if col not in ["current_player", "num_players"]
        and col not in player_position_cols
    ]
    new_feature_cols.extend(onehot_cols)

    # Add turn_number as a feature (it's metadata but useful for prediction)
    if "turn_number" not in new_feature_cols:
        new_feature_cols.append("turn_number")

    print(f"  Created {len(onehot_cols)} one-hot encoded features")
    print(f"  Total features after one-hot encoding: {len(new_feature_cols)}")

    # Extract strategic features (token, card, noble, player comparison, game progression)
    print("\n  Extracting strategic features...")
    strategic_features_list = []

    for idx, row in tqdm(
        df_eng.iterrows(),
        total=len(df_eng),
        desc="  Extracting strategic features",
    ):
        features = extract_all_features(row)
        strategic_features_list.append(features)

    # Convert list of dicts to DataFrame
    strategic_df = pd.DataFrame(strategic_features_list)

    # Get expected feature names and fill missing columns with zeros
    expected_feature_names = get_all_feature_names()
    for feat_name in expected_feature_names:
        if feat_name not in strategic_df.columns:
            strategic_df[feat_name] = 0.0

    # Reorder columns to match expected order
    strategic_df = strategic_df[expected_feature_names]

    # Add strategic features to dataframe
    df_eng = pd.concat([df_eng, strategic_df], axis=1)

    # Add strategic feature names to feature list
    strategic_cols = list(strategic_df.columns)
    new_feature_cols.extend(strategic_cols)

    print(f"  Created {len(strategic_cols)} strategic features")
    print(f"  Total features after all engineering: {len(new_feature_cols)}")

    return df_eng, new_feature_cols, onehot_cols, strategic_cols


def create_normalization_mask(
    feature_cols: List[str],
    onehot_cols: List[str],
    strategic_cols: List[str],
) -> np.ndarray:
    """Create boolean mask indicating which features should be normalized.

    Args:
        feature_cols: List of all feature column names
        onehot_cols: List of one-hot encoded column names
        strategic_cols: List of strategic feature column names

    Returns:
        Boolean array where True = normalize this feature

    """
    mask = np.ones(len(feature_cols), dtype=bool)

    # Binary strategic features that should NOT be normalized
    binary_strategic_patterns = [
        "can_build",
        "must_use_gold",
        "acquirable",
        "can_take2_",
    ]

    for idx, col in enumerate(feature_cols):
        # Don't normalize one-hot encoded columns
        if col in onehot_cols or any(
            col.endswith(f"_bonus_{color}")
            for color in ["white", "blue", "green", "red", "black"]
        ):
            mask[idx] = False

        # Don't normalize position indices (discrete values: -1, 0, 1, 2, ...)
        if "position" in col:
            mask[idx] = False

        # Don't normalize binary strategic features
        if any(pattern in col for pattern in binary_strategic_patterns):
            mask[idx] = False

    print(
        f"\nNormalization mask: {np.sum(mask)} continuous features, {len(mask) - np.sum(mask)} binary/discrete features",
    )

    return mask


def encode_labels(df: pd.DataFrame, label_cols: List[str]) -> Dict[str, np.ndarray]:
    """Encode all label columns into classification targets.

    Args:
        df: DataFrame with all data
        label_cols: List of label column names

    Returns:
        Dict mapping label name to encoded array

    """
    print("\nEncoding labels...")

    labels = {}

    # Use pre-computed constant mappings
    combo_to_class_take3 = COMBO_TO_CLASS_TAKE3
    removal_to_class = REMOVAL_TO_CLASS

    # Action type: string to int mapping
    print("  Encoding action_type...")
    action_type_map = {
        "build": 0,
        "reserve": 1,
        "take 2 tokens": 2,
        "take 3 tokens": 3,
    }
    labels["action_type"] = df["action_type"].map(action_type_map).values
    print(f"    Action type distribution: {np.bincount(labels['action_type'])}")

    # Card selection: 0-14 or NaN -> -1
    print("  Encoding card_selection...")
    labels["card_selection"] = df["card_selection"].fillna(-1).astype(int).values
    valid_count = np.sum(labels["card_selection"] != -1)
    print(f"    Valid card selections: {valid_count}/{len(labels['card_selection'])}")

    # Card reservation: 0-14 or NaN -> -1
    print("  Encoding card_reservation...")
    labels["card_reservation"] = df["card_reservation"].fillna(-1).astype(int).values
    valid_count = np.sum(labels["card_reservation"] != -1)
    print(
        f"    Valid card reservations: {valid_count}/{len(labels['card_reservation'])}",
    )

    # Gem take 3: binary columns -> class index
    print("  Encoding gem_take3...")
    labels["gem_take3"] = df.apply(
        lambda row: encode_gem_take3(row, combo_to_class_take3),
        axis=1,
    ).values
    valid_count = np.sum(labels["gem_take3"] != -1)
    print(f"    Valid gem_take3 actions: {valid_count}/{len(labels['gem_take3'])}")

    # Gem take 2: binary columns -> class index
    print("  Encoding gem_take2...")
    labels["gem_take2"] = df.apply(encode_gem_take2, axis=1).values
    valid_count = np.sum(labels["gem_take2"] != -1)
    print(f"    Valid gem_take2 actions: {valid_count}/{len(labels['gem_take2'])}")

    # Noble selection: 0-4 or -1
    print("  Encoding noble_selection...")
    labels["noble"] = df["noble_selection"].fillna(-1).astype(int).values
    valid_count = np.sum(labels["noble"] != -1)
    print(f"    Valid noble selections: {valid_count}/{len(labels['noble'])}")

    # Gems removed: count columns -> class index
    print("  Encoding gems_removed...")
    labels["gems_removed"] = df.apply(
        lambda row: encode_gems_removed(row, removal_to_class),
        axis=1,
    ).values
    nonzero_count = np.sum(labels["gems_removed"] != 0)
    print(f"    Non-zero gems_removed: {nonzero_count}/{len(labels['gems_removed'])}")

    return labels


def generate_masks_for_dataframe(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Generate legal action masks for all rows in the dataframe.

    For each row, reconstructs the board state and generates masks indicating
    which actions are legal for each prediction head.

    Args:
        df: DataFrame with all game data

    Returns:
        Dict mapping head name to mask array of shape (n_samples, n_classes_for_head)
        Keys: {action_type, card_selection, card_reservation, gem_take3, gem_take2, noble, gems_removed}

    Example:
        >>> masks = generate_masks_for_dataframe(df)
        >>> masks['action_type'].shape  # (n_samples, 4)
        >>> masks['card_selection'].shape  # (n_samples, 15)

    """
    print("\nGenerating legal action masks...")

    # Initialize lists for each head
    head_names = [
        "action_type",
        "card_selection",
        "card_reservation",
        "gem_take3",
        "gem_take2",
        "noble",
        "gems_removed",
    ]
    masks_per_head = {head: [] for head in head_names}

    # Track failures
    failure_count = 0

    # Iterate through rows with progress bar
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating masks"):
        # Convert row to dictionary
        row_dict = row.to_dict()

        # Generate masks for this row
        try:
            masks = generate_all_masks_from_row(row_dict)

            # Append to lists
            for head in head_names:
                masks_per_head[head].append(masks[head])

        except Exception as e:
            # This should be caught by generate_all_masks_from_row, but double-check
            failure_count += 1
            game_id = row.get("game_id", "unknown")
            turn_num = row.get("turn_number", "unknown")
            print(
                f"\nERROR: Mask generation failed for game_id={game_id}, turn={turn_num}: {e}",
            )

            # Use all-ones fallback
            masks_per_head["action_type"].append(np.ones(4, dtype=np.int8))
            masks_per_head["card_selection"].append(np.ones(15, dtype=np.int8))
            masks_per_head["card_reservation"].append(np.ones(15, dtype=np.int8))
            masks_per_head["gem_take3"].append(np.ones(26, dtype=np.int8))
            masks_per_head["gem_take2"].append(np.ones(5, dtype=np.int8))
            masks_per_head["noble"].append(np.ones(5, dtype=np.int8))
            masks_per_head["gems_removed"].append(
                np.ones(get_num_gem_removal_classes(), dtype=np.int8),
            )

    # Convert lists to numpy arrays
    masks_dict = {}
    for head in head_names:
        masks_dict[head] = np.stack(masks_per_head[head], axis=0)
        print(f"  {head}: {masks_dict[head].shape}")

    if failure_count > 0:
        print(
            f"\n  WARNING: {failure_count} mask generation failures ({failure_count / len(df) * 100:.2f}%)",
        )
        print("  Failed samples use all-ones masks (allow all actions)")

    print(f"\n  Successfully generated masks for {len(df):,} samples")

    return masks_dict


def split_by_game_id(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split data at game level to prevent leakage.

    Args:
        df: DataFrame with game_id column
        train_ratio: Proportion for training (e.g., 0.8)
        val_ratio: Proportion for validation (e.g., 0.1)
        test_ratio: Proportion for test (e.g., 0.1)
        seed: Random seed

    Returns:
        Tuple of (train_indices, val_indices, test_indices)

    """
    print("\nSplitting data by game_id...")

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
        "Ratios must sum to 1.0"
    )

    game_ids = df["game_id"].values
    unique_games = df["game_id"].unique()

    print(f"  Total samples: {len(df):,}")
    print(f"  Unique games: {len(unique_games):,}")

    # First split: train vs temp (val + test)
    splitter1 = GroupShuffleSplit(n_splits=1, train_size=train_ratio, random_state=seed)
    train_idx, temp_idx = next(splitter1.split(df, groups=game_ids))

    # Second split: val vs test (50/50 of temp)
    temp_game_ids = df.iloc[temp_idx]["game_id"].values
    val_test_ratio = val_ratio / (val_ratio + test_ratio)
    splitter2 = GroupShuffleSplit(
        n_splits=1,
        train_size=val_test_ratio,
        random_state=seed,
    )
    val_idx_temp, test_idx_temp = next(
        splitter2.split(df.iloc[temp_idx], groups=temp_game_ids),
    )

    # Convert temp indices back to original dataframe indices
    val_idx = temp_idx[val_idx_temp]
    test_idx = temp_idx[test_idx_temp]

    print(
        f"  Train split: {len(train_idx):,} samples ({len(train_idx) / len(df) * 100:.1f}%)",
    )
    print(
        f"  Val split: {len(val_idx):,} samples ({len(val_idx) / len(df) * 100:.1f}%)",
    )
    print(
        f"  Test split: {len(test_idx):,} samples ({len(test_idx) / len(df) * 100:.1f}%)",
    )

    # Verify no game overlap
    train_games = set(df.iloc[train_idx]["game_id"].unique())
    val_games = set(df.iloc[val_idx]["game_id"].unique())
    test_games = set(df.iloc[test_idx]["game_id"].unique())

    assert len(train_games & val_games) == 0, "Train and val games overlap!"
    assert len(train_games & test_games) == 0, "Train and test games overlap!"
    assert len(val_games & test_games) == 0, "Val and test games overlap!"

    print("  ✓ No game overlap between splits")

    return train_idx, val_idx, test_idx


def normalize_features(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    norm_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Normalize continuous features using StandardScaler fitted on training data only.

    Args:
        X_train: Training features
        X_val: Validation features
        X_test: Test features
        norm_mask: Boolean mask indicating which features to normalize

    Returns:
        Tuple of (X_train_norm, X_val_norm, X_test_norm, fitted_scaler)

    """
    print("\nNormalizing features...")

    # Extract continuous features
    X_train_continuous = X_train[:, norm_mask]
    X_val_continuous = X_val[:, norm_mask]
    X_test_continuous = X_test[:, norm_mask]

    print(f"  Normalizing {np.sum(norm_mask)} continuous features")

    # Fit scaler on training data only
    scaler = StandardScaler()
    X_train_continuous_norm = scaler.fit_transform(X_train_continuous)
    X_val_continuous_norm = scaler.transform(X_val_continuous)
    X_test_continuous_norm = scaler.transform(X_test_continuous)

    # Reconstruct full feature matrices
    X_train_norm = X_train.copy()
    X_val_norm = X_val.copy()
    X_test_norm = X_test.copy()

    X_train_norm[:, norm_mask] = X_train_continuous_norm
    X_val_norm[:, norm_mask] = X_val_continuous_norm
    X_test_norm[:, norm_mask] = X_test_continuous_norm

    print(
        f"  Train features: mean={X_train_continuous_norm.mean():.4f}, std={X_train_continuous_norm.std():.4f}",
    )
    print(
        f"  Val features: mean={X_val_continuous_norm.mean():.4f}, std={X_val_continuous_norm.std():.4f}",
    )
    print(
        f"  Test features: mean={X_test_continuous_norm.mean():.4f}, std={X_test_continuous_norm.std():.4f}",
    )

    return X_train_norm, X_val_norm, X_test_norm, scaler


def save_preprocessed_data(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    labels_train: Dict[str, np.ndarray],
    labels_val: Dict[str, np.ndarray],
    labels_test: Dict[str, np.ndarray],
    masks_train: Dict[str, np.ndarray],
    masks_val: Dict[str, np.ndarray],
    masks_test: Dict[str, np.ndarray],
    scaler: StandardScaler,
    feature_cols: List[str],
    label_mappings: Dict,
    output_dir: str,
) -> None:
    """Save all preprocessed data to disk.

    Args:
        X_train, X_val, X_test: Feature arrays
        labels_train, labels_val, labels_test: Label dicts
        masks_train, masks_val, masks_test: Mask dicts
        scaler: Fitted StandardScaler
        feature_cols: List of feature column names
        label_mappings: Dict of encoding mappings
        output_dir: Directory to save files

    """
    print(f"\nSaving preprocessed data to {output_dir}...")

    os.makedirs(output_dir, exist_ok=True)

    # Save feature arrays
    np.save(os.path.join(output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(output_dir, "X_val.npy"), X_val)
    np.save(os.path.join(output_dir, "X_test.npy"), X_test)
    print(f"  Saved feature arrays: {X_train.shape}, {X_val.shape}, {X_test.shape}")

    # Save label arrays
    np.savez(os.path.join(output_dir, "labels_train.npz"), **labels_train)
    np.savez(os.path.join(output_dir, "labels_val.npz"), **labels_val)
    np.savez(os.path.join(output_dir, "labels_test.npz"), **labels_test)
    print(f"  Saved label arrays: {len(labels_train)} heads")

    # Save mask arrays
    np.savez(os.path.join(output_dir, "masks_train.npz"), **masks_train)
    np.savez(os.path.join(output_dir, "masks_val.npz"), **masks_val)
    np.savez(os.path.join(output_dir, "masks_test.npz"), **masks_test)
    print(f"  Saved mask arrays: {len(masks_train)} heads")

    # Save scaler
    with open(os.path.join(output_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    print("  Saved StandardScaler")

    # Save feature columns
    with open(os.path.join(output_dir, "feature_cols.json"), "w") as f:
        json.dump(feature_cols, f, indent=2)
    print(f"  Saved feature column names ({len(feature_cols)} features)")

    # Save label mappings
    with open(os.path.join(output_dir, "label_mappings.json"), "w") as f:
        json.dump(label_mappings, f, indent=2)
    print("  Saved label encoding mappings")

    # Compute mask statistics
    mask_stats = {}
    for head in masks_train:
        train_masks = masks_train[head]
        # Average number of legal actions per sample
        avg_legal = float(np.mean(np.sum(train_masks, axis=1)))
        min_legal = int(np.min(np.sum(train_masks, axis=1)))
        max_legal = int(np.max(np.sum(train_masks, axis=1)))

        mask_stats[head] = {
            "avg_legal_actions": avg_legal,
            "min_legal_actions": min_legal,
            "max_legal_actions": max_legal,
        }

    # Save preprocessing statistics
    stats = {
        "total_samples": int(len(X_train) + len(X_val) + len(X_test)),
        "train_size": len(X_train),
        "val_size": len(X_val),
        "test_size": len(X_test),
        "input_dim": int(X_train.shape[1]),
        "num_features": len(feature_cols),
        "action_type_distribution": {
            "train": labels_train["action_type"].tolist(),
            "val": labels_val["action_type"].tolist(),
            "test": labels_test["action_type"].tolist(),
        },
        "mask_statistics": mask_stats,
    }

    with open(os.path.join(output_dir, "preprocessing_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved preprocessing statistics (input_dim={stats['input_dim']})")
    print("\n  Mask statistics:")
    for head, stats_head in mask_stats.items():
        print(
            f"    {head}: avg={stats_head['avg_legal_actions']:.1f}, min={stats_head['min_legal_actions']}, max={stats_head['max_legal_actions']}",
        )


def validate_masks(
    masks: Dict[str, np.ndarray],
    labels: Dict[str, np.ndarray],
    df: pd.DataFrame,
) -> Dict[str, any]:
    """Validate that masks are correct and labeled actions are legal.

    Critical validation: 100% of labeled actions MUST be legal.
    Any failures indicate data issues or reconstruction bugs.

    Args:
        masks: Dict mapping head name to mask array (n_samples, n_classes)
        labels: Dict mapping head name to label array (n_samples,)
        df: DataFrame with game_id and turn_number for failure reporting

    Returns:
        Dict with validation results and statistics

    Raises:
        Warning if any validation failures occur

    """
    print("\nValidating masks...")

    expected_shapes = {
        "action_type": 4,
        "card_selection": 15,
        "card_reservation": 15,
        "gem_take3": 26,
        "gem_take2": 5,
        "noble": 5,
        "gems_removed": get_num_gem_removal_classes(),
    }

    validation_report = {}
    failures = []

    for head in expected_shapes:
        if head not in masks:
            print(f"  ERROR: Missing mask for head '{head}'")
            continue

        mask = masks[head]
        label = labels[head]
        expected_classes = expected_shapes[head]

        print(f"\n  Validating {head}...")

        # Check shape
        if mask.shape != (len(label), expected_classes):
            print(
                f"    ERROR: Shape mismatch. Expected {(len(label), expected_classes)}, got {mask.shape}",
            )
            continue

        # Check values are binary
        unique_vals = np.unique(mask)
        if not np.all(np.isin(unique_vals, [0, 1])):
            print(f"    ERROR: Mask contains non-binary values: {unique_vals}")
            continue

        # Check at least one legal action per sample (for action_type)
        if head == "action_type":
            zero_masks = np.sum(mask, axis=1) == 0
            if np.any(zero_masks):
                num_zeros = np.sum(zero_masks)
                print(f"    ERROR: {num_zeros} samples have no legal actions!")
                continue

        # Check labeled actions are legal (CRITICAL)
        # Exclude -1 labels (not applicable)
        valid_label_mask = label != -1
        valid_indices = np.where(valid_label_mask)[0]

        if len(valid_indices) > 0:
            # For each valid label, check if mask[sample_idx, label_value] == 1
            illegal_count = 0
            illegal_samples = []

            for idx in valid_indices:
                label_value = label[idx]
                is_legal = mask[idx, label_value] == 1

                if not is_legal:
                    illegal_count += 1
                    game_id = df.iloc[idx]["game_id"]
                    turn_num = df.iloc[idx]["turn_number"]
                    illegal_samples.append(
                        {
                            "sample_idx": int(idx),
                            "game_id": int(game_id),
                            "turn_number": int(turn_num),
                            "label_value": int(label_value),
                            "head": head,
                        },
                    )

            legal_rate = (len(valid_indices) - illegal_count) / len(valid_indices) * 100

            print(
                f"    Labeled actions legal: {len(valid_indices) - illegal_count}/{len(valid_indices)} ({legal_rate:.2f}%)",
            )

            if illegal_count > 0:
                print(f"    ❌ WARNING: {illegal_count} labeled actions are ILLEGAL!")
                failures.extend(illegal_samples)
        else:
            print("    No valid labels to check (all -1)")

        # Compute statistics
        avg_legal = np.mean(np.sum(mask, axis=1))
        min_legal = np.min(np.sum(mask, axis=1))
        max_legal = np.max(np.sum(mask, axis=1))

        validation_report[head] = {
            "shape_valid": bool(mask.shape == (len(label), expected_classes)),
            "binary_values": bool(np.all(np.isin(unique_vals, [0, 1]))),
            "avg_legal_actions": float(avg_legal),
            "min_legal_actions": int(min_legal),
            "max_legal_actions": int(max_legal),
            "illegal_count": int(illegal_count if len(valid_indices) > 0 else 0),
            "valid_labels_count": len(valid_indices),
            "legal_rate": float(legal_rate if len(valid_indices) > 0 else 100.0),
        }

        print(f"    Shape: {mask.shape} ✓")
        print("    Binary values: ✓")
        print(f"    Avg legal actions: {avg_legal:.1f}")
        print(f"    Min/Max legal: {min_legal}/{max_legal}")

    # Report overall results
    total_failures = len(failures)
    if total_failures > 0:
        print(f"\n❌ VALIDATION FAILED: {total_failures} labeled actions are illegal!")
        print("   This indicates data quality issues or reconstruction bugs.")
        print("   First 10 failures:")
        for failure in failures[:10]:
            print(
                f"     Game {failure['game_id']}, Turn {failure['turn_number']}, "
                f"Head '{failure['head']}', Label {failure['label_value']}",
            )

        # Save failure details
        with open("data/processed/mask_validation_failures.json", "w") as f:
            json.dump(failures, f, indent=2)
        print(
            "\n   Full failure list saved to: data/processed/mask_validation_failures.json",
        )

        validation_report["overall_status"] = "FAILED"
        validation_report["total_failures"] = total_failures
    else:
        print("\n✓ Mask validation PASSED! All labeled actions are legal.")
        validation_report["overall_status"] = "PASSED"
        validation_report["total_failures"] = 0

    return validation_report


def preprocess_with_parallel_processing(
    config: Dict,
    max_games: int = None,
    skip_merge: bool = False,
    cleanup: bool = False,
) -> None:
    """New optimized preprocessing pipeline using parallel processing.

    This pipeline:
    1. Discovers CSV files from data_root
    2. Processes files individually with optimized row processing:
       - Reconstructs board once per row
       - Reuses board for both masks and features (50% reduction in reconstructions)
    3. Saves batch files to intermediate directory
    4. Optionally merges batches and performs splitting, normalization, validation, and saving

    Args:
        config: Configuration dictionary
        max_games: Optional limit on number of games
        skip_merge: If True, stop after creating batches (don't merge)
        cleanup: If True, delete batch files after successful merge

    """
    from .parallel_processor import discover_csv_files, process_files_parallel
    from .merge_batches import merge_batches, process_merged_data

    print("\n" + "=" * 60)
    print("OPTIMIZED PARALLEL PREPROCESSING PIPELINE")
    print("=" * 60)

    # Discover CSV files
    print(f"\nDiscovering CSV files in {config['data']['data_root']}...")
    csv_files = discover_csv_files(config["data"]["data_root"], max_games=max_games)
    print(f"Found {len(csv_files)} CSV files")

    # Process files in parallel (now returns batch file paths)
    num_workers = config.get("preprocessing", {}).get("num_workers", None)

    batch_file_paths = process_files_parallel(
        csv_files,
        config,
        num_workers=num_workers,
    )

    if not batch_file_paths:
        raise ValueError("No batch files were created - file processing failed!")

    intermediate_dir = config.get("preprocessing", {}).get(
        "intermediate_dir",
        "data/intermediate",
    )
    print(f"\n{'=' * 60}")
    print("Batching complete!")
    print(f"  {len(batch_file_paths)} batch files saved to: {intermediate_dir}")
    print(f"{'=' * 60}\n")

    # Optionally skip merge (for inspection or debugging)
    if skip_merge:
        print("Skipping merge step (--skip-merge flag set)")
        print("To merge batches later, run:")
        print(
            "  python -m src.imitation_learning.merge_batches --config <config.yaml>",
        )
        return

    # Call merge script to complete processing
    print("Calling merge script to combine batches...")

    # Merge batches
    df_compacted, strategic_features_list, labels_list, masks_list = merge_batches(
        batch_file_paths, config
    )

    # Process merged data
    process_merged_data(
        df_compacted,
        strategic_features_list,
        labels_list,
        masks_list,
        config,
        skip_validation=False,
    )

    # Handle cleanup
    if cleanup:
        print("\nCleaning up batch files...")
        try:
            from .parallel_processor import delete_batch_files

            delete_batch_files(batch_file_paths)
            print(f"✓ Deleted {len(batch_file_paths)} batch files")
        except Exception as e:
            print(f"Warning: Failed to clean up batch files: {e}")
    else:
        print(f"\nBatch files kept in: {intermediate_dir}")
        print("  To clean up manually, run with --cleanup flag")


def main():
    """Main preprocessing pipeline."""
    parser = argparse.ArgumentParser(
        description="Preprocess Splendor game data for imitation learning",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="Maximum number of games to load (for testing)",
    )
    parser.add_argument(
        "--single-file",
        type=str,
        default=None,
        help="Process only a single CSV file (for debugging)",
    )
    parser.add_argument(
        "--parallel",
        type=str,
        default="true",
        help="Use parallel processing (true/false, default: true)",
    )
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="Stop after creating batches, don't merge (default: False)",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete batch files after successful merge (default: False)",
    )
    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from {args.config}...")
    config = load_config(args.config)

    # Set seed for reproducibility
    set_seed(config["seed"])
    print(f"Set random seed: {config['seed']}")

    # Parse parallel flag
    use_parallel = args.parallel.lower() in ["true", "1", "yes", "y"]

    # Route to appropriate preprocessing pipeline
    if use_parallel and not args.single_file:
        # Use new optimized parallel processing pipeline
        preprocess_with_parallel_processing(
            config,
            max_games=args.max_games,
            skip_merge=args.skip_merge,
            cleanup=args.cleanup,
        )
        return

    # Old pipeline (for single-file debugging or when parallel=false)
    print("\n" + "=" * 60)
    print("ORIGINAL PREPROCESSING PIPELINE")
    print("(Use --parallel true for optimized processing)")
    print("=" * 60)

    # Load game data (with NaN values preserved!)
    if args.single_file:
        print("\n⚠️  DEBUG MODE: Processing single file")
        df = load_single_game(args.single_file)
    elif args.max_games:
        print(f"\n⚠️  TEST MODE: Loading only {args.max_games} games")
        df = load_all_games(config["data"]["data_root"], max_games=args.max_games)
    else:
        df = load_all_games(config["data"]["data_root"])

    # Generate masks BEFORE filling NaN (masks need original game state)
    # This is critical: fillna adds phantom nobles/cards that change legal moves
    masks_all = generate_masks_for_dataframe(df)

    # Now fill NaN for feature engineering
    df_filled = fill_nan_values(df)

    # Compact cards and add position indices
    # This must be done AFTER fill_nan_values but BEFORE feature engineering
    df_compacted = compact_cards_and_add_position(df_filled)

    # Identify column groups
    metadata_cols, label_cols, feature_cols = identify_column_groups(df_compacted)

    # Engineer features (one-hot encoding + strategic features)
    df_eng, feature_cols_eng, onehot_cols, strategic_cols = engineer_features(
        df_compacted,
        feature_cols,
    )

    # Create normalization mask
    norm_mask = create_normalization_mask(feature_cols_eng, onehot_cols, strategic_cols)

    # Encode labels (from filled dataframe)
    labels_all = encode_labels(df_eng, label_cols)

    # Split by game_id
    train_idx, val_idx, test_idx = split_by_game_id(
        df_eng,
        config["data"]["train_ratio"],
        config["data"]["val_ratio"],
        config["data"]["test_ratio"],
        config["seed"],
    )

    # Extract features (exclude game_id)
    X_all = df_eng[feature_cols_eng].values

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

    # Reset index for validation DataFrame to match mask/label arrays
    df_for_validation = df_eng.iloc[train_idx].reset_index(drop=True)

    # Validate masks on training set
    validation_report = validate_masks(
        masks_train,
        labels_train,
        df_for_validation,
    )

    # Save validation report
    with open(
        os.path.join(config["data"]["processed_dir"], "mask_validation_report.json"),
        "w",
    ) as f:
        json.dump(validation_report, f, indent=2)
    print(
        f"\n  Validation report saved to: {config['data']['processed_dir']}/mask_validation_report.json",
    )

    # Normalize features
    X_train_norm, X_val_norm, X_test_norm, scaler = normalize_features(
        X_train,
        X_val,
        X_test,
        norm_mask,
    )

    # Prepare label mappings for saving
    class_to_combo_take3, combo_to_class_take3 = generate_gem_take3_classes()
    class_to_removal, removal_to_class = generate_gem_removal_classes()

    label_mappings = {
        "action_type": {
            "build": 0,
            "reserve": 1,
            "take 2 tokens": 2,
            "take 3 tokens": 3,
        },
        "gem_take3_classes": {str(k): list(v) for k, v in class_to_combo_take3.items()},
        "gem_removal_classes": {str(k): list(v) for k, v in class_to_removal.items()},
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
        feature_cols_eng,
        label_mappings,
        config["data"]["processed_dir"],
    )

    print("\n✓ Preprocessing complete!")
    print(f"\n{'=' * 60}")
    print("Summary:")
    print(f"  Total samples: {len(df_eng):,}")
    print(f"  Input dimension: {X_train_norm.shape[1]}")
    print(f"  Train/Val/Test: {len(X_train):,} / {len(X_val):,} / {len(X_test):,}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
