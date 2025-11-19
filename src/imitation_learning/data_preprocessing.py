"""
Data preprocessing pipeline for Splendor imitation learning.

This module loads raw CSV files from MCTS self-play games, performs feature engineering,
encodes labels into classification tasks, splits data at game-level, normalizes features,
and saves preprocessed arrays ready for training.

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

from .utils import (
    generate_gem_take3_classes,
    generate_gem_removal_classes,
    encode_gem_take3,
    encode_gem_take2,
    encode_gems_removed,
    set_seed,
)


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_all_games(data_root: str) -> pd.DataFrame:
    """
    Load all game CSVs from 2_games/, 3_games/, and 4_games/ directories.

    Args:
        data_root: Root directory containing game subdirectories

    Returns:
        Concatenated dataframe with game_id column added

    Example:
        >>> df = load_all_games("data/games")
        >>> print(f"Loaded {len(df)} samples from {df['game_id'].nunique()} games")
    """
    all_dfs = []

    for subdir in ['2_games', '3_games', '4_games']:
        dir_path = Path(data_root) / subdir
        if not dir_path.exists():
            print(f"Warning: {dir_path} does not exist, skipping...")
            continue

        csv_files = list(dir_path.glob('*.csv'))
        print(f"\nLoading {len(csv_files)} CSV files from {subdir}...")

        for csv_file in tqdm(csv_files, desc=f"Loading {subdir}"):
            try:
                df = pd.read_csv(csv_file)

                # Extract game_id from filename (e.g., "1.csv" -> 1)
                game_id = int(csv_file.stem)
                df['game_id'] = game_id

                all_dfs.append(df)

            except Exception as e:
                print(f"Error loading {csv_file}: {e}")
                continue

    if not all_dfs:
        raise ValueError("No CSV files loaded successfully!")

    # Concatenate all dataframes
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Handle missing values (structured padding for fewer players/nobles)
    print(f"\nBefore fillna: {combined_df.isna().sum().sum()} NaN values")
    combined_df = combined_df.fillna(0)
    print(f"After fillna: {combined_df.isna().sum().sum()} NaN values")

    print(f"\nTotal samples loaded: {len(combined_df):,}")
    print(f"Total games loaded: {combined_df['game_id'].nunique():,}")

    return combined_df


def identify_column_groups(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """
    Identify column groups: metadata, labels, features.

    Args:
        df: DataFrame with all columns

    Returns:
        Tuple of (metadata_cols, label_cols, feature_cols)
    """
    all_cols = set(df.columns)

    # Metadata columns
    metadata_cols = ['game_id', 'turn_number', 'current_player', 'num_players']
    # Add playerX_position columns
    metadata_cols.extend([f'player{i}_position' for i in range(4)])

    # Label columns
    label_cols = ['action_type', 'card_selection', 'card_reservation', 'noble_selection']
    # Add gem_take3 columns
    label_cols.extend([f'gem_take3_{color}' for color in ['white', 'blue', 'green', 'red', 'black']])
    # Add gem_take2 columns
    label_cols.extend([f'gem_take2_{color}' for color in ['white', 'blue', 'green', 'red', 'black']])
    # Add gems_removed columns
    label_cols.extend([f'gems_removed_{color}' for color in ['white', 'blue', 'green', 'red', 'black', 'gold']])

    # Filter to only include columns that exist in the dataframe
    metadata_cols = [col for col in metadata_cols if col in all_cols]
    label_cols = [col for col in label_cols if col in all_cols]

    # Feature columns are everything else (excluding game_id which we'll drop from features)
    feature_cols = list(all_cols - set(metadata_cols) - set(label_cols))

    print(f"\nColumn groups identified:")
    print(f"  Metadata: {len(metadata_cols)} columns")
    print(f"  Labels: {len(label_cols)} columns")
    print(f"  Features: {len(feature_cols)} columns")

    return metadata_cols, label_cols, feature_cols


def engineer_features(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Perform feature engineering including one-hot encoding of categorical variables.

    Args:
        df: DataFrame with all data
        feature_cols: List of feature column names

    Returns:
        Tuple of (engineered_df, new_feature_cols, onehot_col_names)
    """
    print("\nEngineering features...")

    # Create a copy to avoid modifying original
    df_eng = df.copy()

    onehot_cols = []

    # One-hot encode current_player (0 to num_players-1, pad to 4)
    print("  One-hot encoding current_player...")
    for i in range(4):
        col_name = f'current_player_{i}'
        df_eng[col_name] = (df_eng['current_player'] == i).astype(int)
        onehot_cols.append(col_name)

    # One-hot encode num_players (2, 3, 4)
    print("  One-hot encoding num_players...")
    for n in [2, 3, 4]:
        col_name = f'num_players_{n}'
        df_eng[col_name] = (df_eng['num_players'] == n).astype(int)
        onehot_cols.append(col_name)

    # One-hot encode playerX_position (0 to num_players-1, pad to 4)
    print("  One-hot encoding player positions...")
    for player_idx in range(4):
        position_col = f'player{player_idx}_position'
        if position_col in df_eng.columns:
            for pos in range(4):
                col_name = f'{position_col}_{pos}'
                df_eng[col_name] = (df_eng[position_col] == pos).astype(int)
                onehot_cols.append(col_name)

    # Update feature columns: remove original categorical, add one-hot
    new_feature_cols = [col for col in feature_cols
                       if col not in ['current_player', 'num_players']
                       and not col.endswith('_position')]
    new_feature_cols.extend(onehot_cols)

    # Add turn_number as a feature (it's metadata but useful for prediction)
    if 'turn_number' not in new_feature_cols:
        new_feature_cols.append('turn_number')

    print(f"  Created {len(onehot_cols)} one-hot encoded features")
    print(f"  Total features after engineering: {len(new_feature_cols)}")

    return df_eng, new_feature_cols, onehot_cols


def create_normalization_mask(feature_cols: List[str], onehot_cols: List[str]) -> np.ndarray:
    """
    Create boolean mask indicating which features should be normalized.

    Args:
        feature_cols: List of all feature column names
        onehot_cols: List of one-hot encoded column names

    Returns:
        Boolean array where True = normalize this feature
    """
    mask = np.ones(len(feature_cols), dtype=bool)

    for idx, col in enumerate(feature_cols):
        # Don't normalize one-hot encoded columns
        if col in onehot_cols:
            mask[idx] = False
        # Don't normalize binary bonus columns
        elif any(col.endswith(f'_bonus_{color}') for color in ['white', 'blue', 'green', 'red', 'black']):
            mask[idx] = False

    print(f"\nNormalization mask: {np.sum(mask)} continuous features, {len(mask) - np.sum(mask)} binary features")

    return mask


def encode_labels(df: pd.DataFrame, label_cols: List[str]) -> Dict[str, np.ndarray]:
    """
    Encode all label columns into classification targets.

    Args:
        df: DataFrame with all data
        label_cols: List of label column names

    Returns:
        Dict mapping label name to encoded array
    """
    print("\nEncoding labels...")

    labels = {}

    # Generate encoding mappings
    _, combo_to_class_take3 = generate_gem_take3_classes()
    _, removal_to_class = generate_gem_removal_classes()

    # Action type: string to int mapping
    print("  Encoding action_type...")
    action_type_map = {
        'build': 0,
        'reserve': 1,
        'take 2 tokens': 2,
        'take 3 tokens': 3
    }
    labels['action_type'] = df['action_type'].map(action_type_map).values
    print(f"    Action type distribution: {np.bincount(labels['action_type'])}")

    # Card selection: 0-14 or NaN -> -1
    print("  Encoding card_selection...")
    labels['card_selection'] = df['card_selection'].fillna(-1).astype(int).values
    valid_count = np.sum(labels['card_selection'] != -1)
    print(f"    Valid card selections: {valid_count}/{len(labels['card_selection'])}")

    # Card reservation: 0-14 or NaN -> -1
    print("  Encoding card_reservation...")
    labels['card_reservation'] = df['card_reservation'].fillna(-1).astype(int).values
    valid_count = np.sum(labels['card_reservation'] != -1)
    print(f"    Valid card reservations: {valid_count}/{len(labels['card_reservation'])}")

    # Gem take 3: binary columns -> class index
    print("  Encoding gem_take3...")
    labels['gem_take3'] = df.apply(
        lambda row: encode_gem_take3(row, combo_to_class_take3),
        axis=1
    ).values
    valid_count = np.sum(labels['gem_take3'] != -1)
    print(f"    Valid gem_take3 actions: {valid_count}/{len(labels['gem_take3'])}")

    # Gem take 2: binary columns -> class index
    print("  Encoding gem_take2...")
    labels['gem_take2'] = df.apply(encode_gem_take2, axis=1).values
    valid_count = np.sum(labels['gem_take2'] != -1)
    print(f"    Valid gem_take2 actions: {valid_count}/{len(labels['gem_take2'])}")

    # Noble selection: 0-4 or -1
    print("  Encoding noble_selection...")
    labels['noble'] = df['noble_selection'].fillna(-1).astype(int).values
    valid_count = np.sum(labels['noble'] != -1)
    print(f"    Valid noble selections: {valid_count}/{len(labels['noble'])}")

    # Gems removed: count columns -> class index
    print("  Encoding gems_removed...")
    labels['gems_removed'] = df.apply(
        lambda row: encode_gems_removed(row, removal_to_class),
        axis=1
    ).values
    nonzero_count = np.sum(labels['gems_removed'] != 0)
    print(f"    Non-zero gems_removed: {nonzero_count}/{len(labels['gems_removed'])}")

    return labels


def split_by_game_id(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data at game level to prevent leakage.

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

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    game_ids = df['game_id'].values
    unique_games = df['game_id'].unique()

    print(f"  Total samples: {len(df):,}")
    print(f"  Unique games: {len(unique_games):,}")

    # First split: train vs temp (val + test)
    splitter1 = GroupShuffleSplit(n_splits=1, train_size=train_ratio, random_state=seed)
    train_idx, temp_idx = next(splitter1.split(df, groups=game_ids))

    # Second split: val vs test (50/50 of temp)
    temp_game_ids = df.iloc[temp_idx]['game_id'].values
    val_test_ratio = val_ratio / (val_ratio + test_ratio)
    splitter2 = GroupShuffleSplit(n_splits=1, train_size=val_test_ratio, random_state=seed)
    val_idx_temp, test_idx_temp = next(splitter2.split(df.iloc[temp_idx], groups=temp_game_ids))

    # Convert temp indices back to original dataframe indices
    val_idx = temp_idx[val_idx_temp]
    test_idx = temp_idx[test_idx_temp]

    print(f"  Train split: {len(train_idx):,} samples ({len(train_idx)/len(df)*100:.1f}%)")
    print(f"  Val split: {len(val_idx):,} samples ({len(val_idx)/len(df)*100:.1f}%)")
    print(f"  Test split: {len(test_idx):,} samples ({len(test_idx)/len(df)*100:.1f}%)")

    # Verify no game overlap
    train_games = set(df.iloc[train_idx]['game_id'].unique())
    val_games = set(df.iloc[val_idx]['game_id'].unique())
    test_games = set(df.iloc[test_idx]['game_id'].unique())

    assert len(train_games & val_games) == 0, "Train and val games overlap!"
    assert len(train_games & test_games) == 0, "Train and test games overlap!"
    assert len(val_games & test_games) == 0, "Val and test games overlap!"

    print("  ✓ No game overlap between splits")

    return train_idx, val_idx, test_idx


def normalize_features(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    norm_mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Normalize continuous features using StandardScaler fitted on training data only.

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

    print(f"  Train features: mean={X_train_continuous_norm.mean():.4f}, std={X_train_continuous_norm.std():.4f}")
    print(f"  Val features: mean={X_val_continuous_norm.mean():.4f}, std={X_val_continuous_norm.std():.4f}")
    print(f"  Test features: mean={X_test_continuous_norm.mean():.4f}, std={X_test_continuous_norm.std():.4f}")

    return X_train_norm, X_val_norm, X_test_norm, scaler


def save_preprocessed_data(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    labels_train: Dict[str, np.ndarray],
    labels_val: Dict[str, np.ndarray],
    labels_test: Dict[str, np.ndarray],
    scaler: StandardScaler,
    feature_cols: List[str],
    label_mappings: Dict,
    output_dir: str
) -> None:
    """
    Save all preprocessed data to disk.

    Args:
        X_train, X_val, X_test: Feature arrays
        labels_train, labels_val, labels_test: Label dicts
        scaler: Fitted StandardScaler
        feature_cols: List of feature column names
        label_mappings: Dict of encoding mappings
        output_dir: Directory to save files
    """
    print(f"\nSaving preprocessed data to {output_dir}...")

    os.makedirs(output_dir, exist_ok=True)

    # Save feature arrays
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'X_val.npy'), X_val)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    print(f"  Saved feature arrays: {X_train.shape}, {X_val.shape}, {X_test.shape}")

    # Save label arrays
    np.savez(os.path.join(output_dir, 'labels_train.npz'), **labels_train)
    np.savez(os.path.join(output_dir, 'labels_val.npz'), **labels_val)
    np.savez(os.path.join(output_dir, 'labels_test.npz'), **labels_test)
    print(f"  Saved label arrays: {len(labels_train)} heads")

    # Save scaler
    with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    print(f"  Saved StandardScaler")

    # Save feature columns
    with open(os.path.join(output_dir, 'feature_cols.json'), 'w') as f:
        json.dump(feature_cols, f, indent=2)
    print(f"  Saved feature column names ({len(feature_cols)} features)")

    # Save label mappings
    with open(os.path.join(output_dir, 'label_mappings.json'), 'w') as f:
        json.dump(label_mappings, f, indent=2)
    print(f"  Saved label encoding mappings")

    # Save preprocessing statistics
    stats = {
        'total_samples': int(len(X_train) + len(X_val) + len(X_test)),
        'train_size': int(len(X_train)),
        'val_size': int(len(X_val)),
        'test_size': int(len(X_test)),
        'input_dim': int(X_train.shape[1]),
        'num_features': len(feature_cols),
        'action_type_distribution': {
            'train': labels_train['action_type'].tolist(),
            'val': labels_val['action_type'].tolist(),
            'test': labels_test['action_type'].tolist()
        }
    }

    with open(os.path.join(output_dir, 'preprocessing_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved preprocessing statistics (input_dim={stats['input_dim']})")


def main():
    """Main preprocessing pipeline."""
    parser = argparse.ArgumentParser(description='Preprocess Splendor game data for imitation learning')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from {args.config}...")
    config = load_config(args.config)

    # Set seed for reproducibility
    set_seed(config['seed'])
    print(f"Set random seed: {config['seed']}")

    # Load all game data
    df = load_all_games(config['data']['data_root'])

    # Identify column groups
    metadata_cols, label_cols, feature_cols = identify_column_groups(df)

    # Engineer features (one-hot encoding)
    df_eng, feature_cols_eng, onehot_cols = engineer_features(df, feature_cols)

    # Create normalization mask
    norm_mask = create_normalization_mask(feature_cols_eng, onehot_cols)

    # Encode labels
    labels_all = encode_labels(df_eng, label_cols)

    # Split by game_id
    train_idx, val_idx, test_idx = split_by_game_id(
        df_eng,
        config['data']['train_ratio'],
        config['data']['val_ratio'],
        config['data']['test_ratio'],
        config['seed']
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

    # Normalize features
    X_train_norm, X_val_norm, X_test_norm, scaler = normalize_features(
        X_train, X_val, X_test, norm_mask
    )

    # Prepare label mappings for saving
    class_to_combo_take3, combo_to_class_take3 = generate_gem_take3_classes()
    class_to_removal, removal_to_class = generate_gem_removal_classes()

    label_mappings = {
        'action_type': {'build': 0, 'reserve': 1, 'take 2 tokens': 2, 'take 3 tokens': 3},
        'gem_take3_classes': {str(k): list(v) for k, v in class_to_combo_take3.items()},
        'gem_removal_classes': {str(k): list(v) for k, v in class_to_removal.items()}
    }

    # Save everything
    save_preprocessed_data(
        X_train_norm, X_val_norm, X_test_norm,
        labels_train, labels_val, labels_test,
        scaler, feature_cols_eng, label_mappings,
        config['data']['processed_dir']
    )

    print("\n✓ Preprocessing complete!")
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total samples: {len(df_eng):,}")
    print(f"  Input dimension: {X_train_norm.shape[1]}")
    print(f"  Train/Val/Test: {len(X_train):,} / {len(X_val):,} / {len(X_test):,}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
