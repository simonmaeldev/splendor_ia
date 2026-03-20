#!/usr/bin/env python3
"""Export feature representation for a single CSV row with column names.

This CLI utility script takes a raw CSV file and row index, applies the full
preprocessing pipeline (feature engineering + normalization), and exports two
CSV files showing features before and after normalization with their column names.

Usage:
    python scripts/export_feature_row.py --csv_path data/games/3_games/3869.csv --row_index 1
    python scripts/export_feature_row.py --csv_path data/games/3_games/3869.csv --row_index 1 --output_dir outputs/
    python scripts/export_feature_row.py --csv_path data/games/3_games/3869.csv --row_index 1 --scaler_path data/preprocessed/scaler.pkl
"""

import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Add project root and src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from imitation_learning.feature_engineering import extract_all_features, get_all_feature_names
from imitation_learning.utils import generate_all_masks_from_row
from utils.state_reconstruction import reconstruct_board_from_csv_row


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Export feature representation for a single CSV row with column names",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export features for row 1 of game 3869
  python scripts/export_feature_row.py --csv_path data/games/3_games/3869.csv --row_index 1

  # Use a pre-fitted scaler for accurate normalization
  python scripts/export_feature_row.py --csv_path data/games/3_games/3869.csv --row_index 1 --scaler_path data/preprocessed/scaler.pkl

  # Save outputs to a specific directory
  python scripts/export_feature_row.py --csv_path data/games/3_games/3869.csv --row_index 1 --output_dir outputs/
        """,
    )

    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to the raw CSV file (e.g., data/games/3_games/3869.csv)",
    )
    parser.add_argument(
        "--row_index",
        type=int,
        required=True,
        help="Row index to extract (0 is header, 1 is first data row, etc.)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory to save output CSVs (default: current directory)",
    )
    parser.add_argument(
        "--scaler_path",
        type=str,
        default=None,
        help="Optional path to a pre-fitted scaler pickle file for normalization",
    )

    return parser.parse_args()


def load_csv_row(csv_path: str, row_index: int) -> tuple[pd.Series, int]:
    """Load a specific row from a CSV file.

    Args:
        csv_path: Path to the CSV file
        row_index: Row index (0 is header, 1 is first data row)

    Returns:
        Tuple of (row as Series, game_id)

    Raises:
        ValueError: If file not found or row index invalid
    """
    print(f"\n[1/5] Loading CSV file: {csv_path}")

    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise ValueError(f"File not found: {csv_path}")

    # Load the CSV
    df = pd.read_csv(csv_file)

    # Validate row index
    if row_index == 0:
        raise ValueError("Row index 0 is the header. Use row_index >= 1 for data rows.")
    if row_index > len(df):
        raise ValueError(f"Row index {row_index} exceeds the number of data rows ({len(df)})")

    # Extract the row (row_index=1 is the first data row, which is iloc[0])
    row = df.iloc[row_index - 1]

    # Extract game_id from filename (e.g., "3869.csv" -> 3869)
    game_id = int(csv_file.stem)

    print(f"  ✓ Loaded row {row_index} from game {game_id}")
    print(f"  ✓ CSV has {len(df)} data rows")

    return row, game_id


def engineer_features(row: pd.Series, game_id: int) -> tuple[np.ndarray, List[str]]:
    """Apply feature engineering pipeline to a single row.

    Args:
        row: CSV row as pandas Series
        game_id: Game ID extracted from filename

    Returns:
        Tuple of (feature vector, feature names)
    """
    print(f"\n[2/5] Applying feature engineering pipeline...")

    # Define label columns to exclude from features (same as data_preprocessing.py:381-401)
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

    # Add game_id to row (needed for feature engineering)
    row_with_id = row.copy()
    row_with_id['game_id'] = game_id

    # Convert row to dict for board reconstruction (needs dict-like access)
    row_dict = row_with_id.to_dict()

    # Reconstruct board object (needed for feature engineering)
    try:
        board = reconstruct_board_from_csv_row(row_dict)
    except Exception as e:
        print(f"  WARNING: Board reconstruction failed: {e}")
        print(f"  Continuing without board-dependent features...")
        board = None

    # Fill NaN values with 0 (same as main pipeline)
    row_filled = row_with_id.fillna(0)

    # Extract engineered features
    if board is not None:
        engineered_features = extract_all_features(row_filled, board)
    else:
        engineered_features = {}

    # Get feature names in the correct order
    feature_names = get_all_feature_names()

    # Get raw CSV column names (excluding game_id and label columns)
    raw_columns = [col for col in row.index if col not in label_cols]

    # Combine raw features + engineered features
    raw_values = row_filled[raw_columns].values
    engineered_values = np.array([engineered_features.get(name, 0.0) for name in feature_names])

    # Combine into single feature vector
    feature_vector = np.concatenate([raw_values, engineered_values])
    all_feature_names = raw_columns + feature_names

    print(f"  ✓ Raw CSV features: {len(raw_columns)}")
    print(f"  ✓ Engineered features: {len(feature_names)}")
    print(f"  ✓ Total features: {len(all_feature_names)}")

    return feature_vector, all_feature_names


def create_normalization_mask(feature_names: List[str]) -> np.ndarray:
    """Create boolean mask indicating which features should be normalized.

    This replicates the logic from data_preprocessing.py:523-564.

    Args:
        feature_names: List of all feature column names

    Returns:
        Boolean array where True = normalize this feature
    """
    mask = np.ones(len(feature_names), dtype=bool)

    # Binary strategic features that should NOT be normalized
    binary_strategic_patterns = [
        "can_build",
        "must_use_gold",
        "acquirable",
        "can_take2_",
    ]

    for idx, col in enumerate(feature_names):
        # Don't normalize one-hot encoded bonus columns
        if any(col.endswith(f"_bonus_{color}") for color in ["white", "blue", "green", "red", "black"]):
            mask[idx] = False

        # Don't normalize position indices (discrete values: -1, 0, 1, 2, ...)
        if "position" in col:
            mask[idx] = False

        # Don't normalize binary strategic features
        if any(pattern in col for pattern in binary_strategic_patterns):
            mask[idx] = False

    return mask


def normalize_features(
    feature_vector: np.ndarray,
    feature_names: List[str],
    scaler_path: Optional[str] = None,
) -> tuple[np.ndarray, int, int]:
    """Apply normalization to continuous features.

    Args:
        feature_vector: Feature vector to normalize
        feature_names: List of feature names
        scaler_path: Optional path to pre-fitted scaler

    Returns:
        Tuple of (normalized feature vector, num_normalized, num_excluded)
    """
    print(f"\n[3/5] Applying normalization...")

    # Create normalization mask
    norm_mask = create_normalization_mask(feature_names)
    num_normalized = np.sum(norm_mask)
    num_excluded = len(norm_mask) - num_normalized

    # Load or create scaler
    if scaler_path is not None:
        print(f"  Loading pre-fitted scaler from: {scaler_path}")
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
    else:
        print("  WARNING: No scaler provided - fitting on single row (NOT representative!)")
        print("  For accurate normalization, provide a scaler from a training run via --scaler_path")
        scaler = StandardScaler()
        # Fit on single row (will result in zero mean, undefined std)
        continuous_features = feature_vector[norm_mask].reshape(1, -1)
        scaler.fit(continuous_features)

    # Apply normalization to continuous features only
    feature_vector_norm = feature_vector.copy()
    continuous_features = feature_vector[norm_mask].reshape(1, -1)
    continuous_features_norm = scaler.transform(continuous_features)
    feature_vector_norm[norm_mask] = continuous_features_norm.flatten()

    print(f"  ✓ Normalized {num_normalized} continuous features")
    print(f"  ✓ Excluded {num_excluded} binary/discrete features from normalization")

    return feature_vector_norm, num_normalized, num_excluded


def export_csv_files(
    feature_vector_before: np.ndarray,
    feature_vector_after: np.ndarray,
    feature_names: List[str],
    game_id: int,
    row_index: int,
    output_dir: str,
) -> tuple[str, str]:
    """Export feature vectors to CSV files.

    Args:
        feature_vector_before: Features before normalization
        feature_vector_after: Features after normalization
        feature_names: List of feature column names
        game_id: Game ID
        row_index: Row index
        output_dir: Output directory

    Returns:
        Tuple of (before_path, after_path)
    """
    print(f"\n[4/5] Exporting CSV files...")

    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)

    # Create DataFrames with column names as header
    df_before = pd.DataFrame([feature_vector_before], columns=feature_names)
    df_after = pd.DataFrame([feature_vector_after], columns=feature_names)

    # Generate output filenames
    before_path = os.path.join(output_dir, f"features_before_norm_game{game_id}_row{row_index}.csv")
    after_path = os.path.join(output_dir, f"features_after_norm_game{game_id}_row{row_index}.csv")

    # Save to CSV
    df_before.to_csv(before_path, index=False)
    df_after.to_csv(after_path, index=False)

    print(f"  ✓ Saved: {before_path}")
    print(f"  ✓ Saved: {after_path}")

    return before_path, after_path


def print_summary(
    feature_names: List[str],
    num_normalized: int,
    num_excluded: int,
    before_path: str,
    after_path: str,
):
    """Print summary of the export operation."""
    print(f"\n[5/5] Summary")
    print(f"  Total features: {len(feature_names)}")
    print(f"  Features normalized: {num_normalized}")
    print(f"  Features excluded from normalization: {num_excluded}")
    print(f"\nOutput files:")
    print(f"  Before normalization: {before_path}")
    print(f"  After normalization: {after_path}")
    print("\n✓ Done!")


def main():
    """Main entry point."""
    args = parse_args()

    try:
        # Step 1: Load CSV row
        row, game_id = load_csv_row(args.csv_path, args.row_index)

        # Step 2: Apply feature engineering
        feature_vector, feature_names = engineer_features(row, game_id)

        # Step 3: Apply normalization
        feature_vector_norm, num_normalized, num_excluded = normalize_features(
            feature_vector,
            feature_names,
            args.scaler_path,
        )

        # Step 4: Export to CSV files
        before_path, after_path = export_csv_files(
            feature_vector,
            feature_vector_norm,
            feature_names,
            game_id,
            args.row_index,
            args.output_dir,
        )

        # Step 5: Print summary
        print_summary(feature_names, num_normalized, num_excluded, before_path, after_path)

    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
