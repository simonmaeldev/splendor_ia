"""Parallel processing infrastructure for data preprocessing optimization.

This module implements memory-efficient, parallelized preprocessing by:
1. Processing CSV files individually (not loading all into memory)
2. Reconstructing board state once per row (reusing for masks + features)
3. Parallelizing across files using multiprocessing
4. Accumulating results in lists, combining at the end

Expected benefits:
- 50% reduction in board reconstructions (2x per row → 1x per row)
- Significantly lower memory footprint (process files individually)
- Near-linear speedup from parallelization (N cores → ~N× faster)

Usage:
    from parallel_processor import process_files_parallel

    file_paths = discover_csv_files("data/games")
    results = process_files_parallel(file_paths, config, num_workers=6)
"""

import functools
import gc
import multiprocessing as mp
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from .utils import (
    encode_gem_take2,
    encode_gem_take3,
    encode_gems_removed,
    generate_all_masks_from_row,
)
from .constants import (
    COMBO_TO_CLASS_TAKE3,
    REMOVAL_TO_CLASS,
)
from .feature_engineering import extract_all_features, get_all_feature_names
from .memory_monitor import log_memory_usage
from utils.state_reconstruction import reconstruct_board_from_csv_row


# Head names for multi-head prediction
HEAD_NAMES = [
    "action_type",
    "card_selection",
    "card_reservation",
    "gem_take3",
    "gem_take2",
    "noble",
    "gems_removed",
]

# Number of classes for each mask head
NUM_CLASSES = {
    'action_type': 4,
    'card_selection': 15,
    'card_reservation': 15,
    'gem_take3': 26,
    'gem_take2': 5,
    'noble': 5,
    'gems_removed': 84,
}


def convert_strategic_features_to_array(
    strategic_features_list: List[Dict],
    feature_names: List[str],
    dtype: str = 'float32'
) -> np.ndarray:
    """Convert list of strategic feature dicts to 2D array.

    Args:
        strategic_features_list: List of dicts, each with 893 keys
        feature_names: Ordered list of feature names (from get_all_feature_names())
        dtype: Array dtype, 'float32' or 'float64'

    Returns:
        Array of shape (n_samples, 893)
    """
    # Use DataFrame for efficient conversion with column ordering
    df = pd.DataFrame(strategic_features_list)

    # Ensure all expected columns exist (fill missing with 0.0)
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0.0

    # Select columns in correct order and convert to array
    array = df[feature_names].values.astype(dtype)

    return array


def convert_labels_to_arrays(labels_list: List[Dict]) -> Dict[str, np.ndarray]:
    """Convert list of label dicts to dict of 1D arrays.

    Args:
        labels_list: List of dicts, each with 7 keys (one per head)

    Returns:
        Dict mapping head name to 1D array of shape (n_samples,)
    """
    labels_arrays = {}

    for head in HEAD_NAMES:
        labels_arrays[head] = np.array(
            [labels[head] for labels in labels_list],
            dtype=np.int16
        )

    return labels_arrays


def convert_masks_to_arrays(masks_list: List[Dict]) -> Dict[str, np.ndarray]:
    """Convert list of mask dicts to dict of 2D arrays.

    Args:
        masks_list: List of dicts, each with 7 keys (one per head)

    Returns:
        Dict mapping head name to 2D array of shape (n_samples, n_classes)
    """
    masks_arrays = {}

    for head in HEAD_NAMES:
        masks_arrays[head] = np.stack(
            [masks[head] for masks in masks_list],
            axis=0
        ).astype(np.int8)

    return masks_arrays


def convert_array_to_strategic_features(
    strategic_features_array: np.ndarray,
    feature_names: List[str]
) -> List[Dict]:
    """Convert strategic features array back to list of dicts.

    Args:
        strategic_features_array: Array of shape (n_samples, n_features)
        feature_names: Ordered list of feature names

    Returns:
        List of feature dicts
    """
    df = pd.DataFrame(strategic_features_array, columns=feature_names)
    return df.to_dict('records')


def convert_arrays_to_labels(labels_arrays: Dict[str, np.ndarray]) -> List[Dict]:
    """Convert dict of label arrays back to list of dicts.

    Args:
        labels_arrays: Dict mapping head name to 1D array

    Returns:
        List of label dicts
    """
    n_samples = len(labels_arrays[HEAD_NAMES[0]])
    labels_list = []

    for i in range(n_samples):
        labels_dict = {head: int(labels_arrays[head][i]) for head in HEAD_NAMES}
        labels_list.append(labels_dict)

    return labels_list


def convert_arrays_to_masks(masks_arrays: Dict[str, np.ndarray]) -> List[Dict]:
    """Convert dict of mask arrays back to list of dicts.

    Args:
        masks_arrays: Dict mapping head name to 2D array

    Returns:
        List of mask dicts
    """
    n_samples = len(masks_arrays[HEAD_NAMES[0]])
    masks_list = []

    for i in range(n_samples):
        masks_dict = {head: masks_arrays[head][i] for head in HEAD_NAMES}
        masks_list.append(masks_dict)

    return masks_list


def fill_nan_values_for_row(row_dict: Dict) -> Dict:
    """Fill NaN values for a single row dictionary.

    Args:
        row_dict: Row as dictionary

    Returns:
        Updated row dictionary with NaN filled for feature columns only
    """
    # Label columns that should keep NaN
    label_cols_to_keep_nan = [
        'card_selection', 'card_reservation', 'noble_selection',
        'gem_take3_white', 'gem_take3_blue', 'gem_take3_green', 'gem_take3_red', 'gem_take3_black',
        'gem_take2_white', 'gem_take2_blue', 'gem_take2_green', 'gem_take2_red', 'gem_take2_black',
        'gems_removed_white', 'gems_removed_blue', 'gems_removed_green', 'gems_removed_red',
        'gems_removed_black', 'gems_removed_gold',
    ]

    filled_row = {}
    for key, value in row_dict.items():
        if key not in label_cols_to_keep_nan and key != 'action_type':
            # Fill NaN with 0
            if pd.isna(value):
                filled_row[key] = 0
            else:
                filled_row[key] = value
        else:
            filled_row[key] = value

    return filled_row


def compact_cards_and_add_position_for_row(row_dict: Dict) -> Dict:
    """Compact visible cards and add position indices for a single row.

    This function:
    1. Identifies non-zero visible cards (cards 0-11)
    2. Reorders them: non-zero first, zeros at end
    3. Adds position index feature: 0, 1, 2, ... for non-zero; -1 for zeros
    4. For reserved cards: adds position 12, 13, 14 per player; -1 for missing

    Args:
        row_dict: Row as dictionary (with NaN filled)

    Returns:
        Updated row dictionary with compacted cards and position features
    """
    compacted_row = row_dict.copy()

    card_feature_names = ['vp', 'level', 'cost_white', 'cost_blue', 'cost_green',
                          'cost_red', 'cost_black', 'bonus_white', 'bonus_blue',
                          'bonus_green', 'bonus_red', 'bonus_black']

    # Process visible cards (card0 to card11)
    visible_cards = []
    for card_idx in range(12):
        card_data = {}
        for feat in card_feature_names:
            key = f'card{card_idx}_{feat}'
            card_data[feat] = row_dict.get(key, 0)

        # Check if card is non-zero
        is_nonzero = any(card_data[feat] != 0 for feat in card_feature_names)
        visible_cards.append((card_idx, card_data, is_nonzero))

    # Separate non-zero and zero cards
    nonzero_cards = []
    zero_cards = []
    for c in visible_cards:
        if c[2]:
            nonzero_cards.append(c)
        else:
            zero_cards.append(c)

    # Reorder: non-zero first, then zeros
    reordered_cards = nonzero_cards + zero_cards

    # Write back with position indices
    for new_idx, (old_idx, card_data, is_nonzero) in enumerate(reordered_cards):
        # Add position
        position = new_idx if is_nonzero else -1
        compacted_row[f'card{new_idx}_position'] = position

        # Add card features
        for feat in card_feature_names:
            compacted_row[f'card{new_idx}_{feat}'] = card_data[feat]

    # Process reserved cards (3 per player, positions 12, 13, 14)
    for player_idx in range(4):
        for reserved_idx in range(3):
            prefix = f'player{player_idx}_reserved{reserved_idx}'

            # Extract reserved card features
            card_data = {}
            for feat in card_feature_names:
                key = f'{prefix}_{feat}'
                card_data[feat] = row_dict.get(key, 0)

            # Check if card is present
            is_present = any(card_data[feat] != 0 for feat in card_feature_names)

            # Assign position: 12, 13, or 14 if present, -1 if missing
            position = 12 + reserved_idx if is_present else -1
            compacted_row[f'{prefix}_position'] = position

            # Features already in row, no need to update

    return compacted_row


def process_single_row(
    row_dict: Dict,
    board,
) -> Tuple[Dict, Dict[str, float], Dict[str, int], Dict[str, np.ndarray]]:
    """Process a single row: generate masks, fill NaN, compact cards, engineer features, encode labels.

    Args:
        row_dict: Row as dictionary
        board: Pre-reconstructed Board object (optimization!)

    Returns:
        Tuple of (row_compacted, strategic_features, labels_dict, masks_dict)
    """
    # Generate masks (reusing board)
    masks = generate_all_masks_from_row(row_dict, board=board)

    # Fill NaN values
    row_filled = fill_nan_values_for_row(row_dict)

    # Compact cards and add position indices
    row_compacted = compact_cards_and_add_position_for_row(row_filled)

    # Engineer features (reusing board)
    # Convert dict to Series for feature engineering
    row_series = pd.Series(row_compacted)
    strategic_features = extract_all_features(row_series, board=board)

    # Encode labels
    labels = {}

    # Action type
    action_type_map = {
        "build": 0,
        "reserve": 1,
        "take 2 tokens": 2,
        "take 3 tokens": 3,
    }
    action_type_str = row_compacted.get("action_type", "build")
    labels["action_type"] = action_type_map.get(action_type_str, 0)

    # Card selection
    card_selection = row_compacted.get("card_selection", np.nan)
    labels["card_selection"] = -1 if pd.isna(card_selection) else int(card_selection)

    # Card reservation
    card_reservation = row_compacted.get("card_reservation", np.nan)
    labels["card_reservation"] = -1 if pd.isna(card_reservation) else int(card_reservation)

    # Gem take3 - use constants
    labels["gem_take3"] = encode_gem_take3(row_series, COMBO_TO_CLASS_TAKE3)

    # Gem take2
    labels["gem_take2"] = encode_gem_take2(row_series)

    # Noble
    noble_selection = row_compacted.get("noble_selection", np.nan)
    labels["noble"] = -1 if pd.isna(noble_selection) else int(noble_selection)

    # Gems removed - use constants
    labels["gems_removed"] = encode_gems_removed(row_series, REMOVAL_TO_CLASS)

    return row_compacted, strategic_features, labels, masks


def process_single_file(file_path: str, config: Dict) -> Tuple[List, List, List, List]:
    """Process a single CSV file and return accumulated results.

    This function:
    1. Loads CSV file with pandas
    2. For each row:
       - Reconstructs board state ONCE
       - Generates masks (reusing board)
       - Fills NaN values
       - Compacts cards and adds position indices
       - Extracts strategic features (reusing board)
       - Encodes labels
    3. Returns accumulated lists of processed samples

    Args:
        file_path: Path to CSV file
        config: Configuration dictionary

    Returns:
        Tuple of (raw_rows_list, strategic_features_list, labels_list, masks_list)
    """
    try:
        # Load CSV file
        df = pd.read_csv(file_path)

        # Accumulate results
        raw_rows = []
        strategic_features_list = []
        labels_list = []
        masks_list = []

        # Process each row
        for idx, row in df.iterrows():
            row_dict = row.to_dict()

            try:
                # Reconstruct board ONCE
                board = reconstruct_board_from_csv_row(row_dict)

                # Process row (reusing board for masks and features)
                row_compacted, strategic_features, labels, masks = process_single_row(
                    row_dict, board
                )

                raw_rows.append(row_compacted)
                strategic_features_list.append(strategic_features)
                labels_list.append(labels)
                masks_list.append(masks)

            except Exception as e:
                # Skip this row on error
                game_id = row_dict.get("game_id", "unknown")
                turn_num = row_dict.get("turn_number", "unknown")
                print(f"ERROR: Row processing failed for game_id={game_id}, turn={turn_num}: {e}")
                continue

        return raw_rows, strategic_features_list, labels_list, masks_list

    except Exception as e:
        print(f"ERROR: File processing failed for {file_path}: {e}")
        return [], [], [], []


def save_batch_to_file(
    batch_num: int,
    df: pd.DataFrame,
    strategic_features: List[Dict],
    labels: List[Dict],
    masks: List[Dict],
    file_paths: List[str],
    intermediate_dir: str = "data/intermediate"
) -> str:
    """Save a batch of processed data to a compressed NPZ file.

    This now saves data as arrays instead of list-of-dicts for 82% memory reduction.

    Args:
        batch_num: Batch number for file naming
        df: DataFrame with compacted rows
        strategic_features: List of feature dictionaries
        labels: List of label dictionaries
        masks: List of mask dictionaries
        file_paths: List of file paths in this batch
        intermediate_dir: Directory to save batch files

    Returns:
        Path to saved batch file
    """
    os.makedirs(intermediate_dir, exist_ok=True)

    batch_file = os.path.join(intermediate_dir, f"batch_{batch_num:04d}.npz")

    # Convert DataFrame to pickle bytes for storage
    df_pickle = pickle.dumps(df)

    # Convert strategic features to array (Strategy A)
    feature_names = get_all_feature_names()
    strategic_features_array = convert_strategic_features_to_array(
        strategic_features, feature_names, dtype='float32'
    )

    # Convert labels to arrays
    labels_arrays = convert_labels_to_arrays(labels)

    # Convert masks to arrays
    masks_arrays = convert_masks_to_arrays(masks)

    # Create metadata
    metadata = {
        'batch_num': batch_num,
        'num_samples': len(df),
        'num_files': len(file_paths),
    }

    # Save to compressed NPZ with array format
    np.savez_compressed(
        batch_file,
        df_pickle=df_pickle,
        strategic_features=strategic_features_array,
        strategic_feature_names=feature_names,
        labels_action_type=labels_arrays['action_type'],
        labels_card_selection=labels_arrays['card_selection'],
        labels_card_reservation=labels_arrays['card_reservation'],
        labels_gem_take3=labels_arrays['gem_take3'],
        labels_gem_take2=labels_arrays['gem_take2'],
        labels_noble=labels_arrays['noble'],
        labels_gems_removed=labels_arrays['gems_removed'],
        masks_action_type=masks_arrays['action_type'],
        masks_card_selection=masks_arrays['card_selection'],
        masks_card_reservation=masks_arrays['card_reservation'],
        masks_gem_take3=masks_arrays['gem_take3'],
        masks_gem_take2=masks_arrays['gem_take2'],
        masks_noble=masks_arrays['noble'],
        masks_gems_removed=masks_arrays['gems_removed'],
        metadata=metadata,
    )

    # Calculate memory savings
    old_size_mb = (len(strategic_features) * 893 * 240) / (1024 * 1024)  # Approx list-of-dicts size
    new_size_mb = strategic_features_array.nbytes / (1024 * 1024)
    savings_pct = (1 - new_size_mb / old_size_mb) * 100

    print(f"  Saved batch {batch_num} to {batch_file} ({len(df):,} samples)")
    print(f"    Strategic features: {strategic_features_array.shape} ({new_size_mb:.1f} MB, {savings_pct:.0f}% memory savings)")
    log_memory_usage(f"After saving batch {batch_num}", force_gc=True)

    return batch_file


def load_batch_from_file(batch_file: str) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Load a batch of processed data from NPZ file.

    Supports both old list-of-dicts format (backward compatibility) and new array format.

    Args:
        batch_file: Path to batch NPZ file

    Returns:
        Tuple of (df, strategic_features_array, labels_arrays, masks_arrays)
    """
    data = np.load(batch_file, allow_pickle=True)

    # Unpickle DataFrame
    df = pickle.loads(data['df_pickle'].item())

    # Detect format by checking if strategic_features is 2D array (new format)
    is_new_format = 'strategic_features' in data and len(data['strategic_features'].shape) == 2

    metadata = data['metadata'].item()
    print(f"  Loaded batch {metadata['batch_num']} from {batch_file} ({metadata['num_samples']:,} samples, {'array' if is_new_format else 'legacy'} format)")

    if is_new_format:
        # NEW FORMAT: Load arrays directly
        strategic_features_array = data['strategic_features']

        # Load label arrays
        labels_arrays = {}
        for head in HEAD_NAMES:
            labels_arrays[head] = data[f'labels_{head}']

        # Load mask arrays
        masks_arrays = {}
        for head in HEAD_NAMES:
            masks_arrays[head] = data[f'masks_{head}']

        return df, strategic_features_array, labels_arrays, masks_arrays

    else:
        # OLD FORMAT: Convert from list-of-dicts to arrays for compatibility
        print(f"    Converting old format to arrays...")

        # Extract lists (stored as numpy arrays)
        strategic_features_list = list(data['strategic_features'])
        labels_list = list(data['labels'])
        masks_list = list(data['masks'])

        # Convert to arrays
        feature_names = get_all_feature_names()
        strategic_features_array = convert_strategic_features_to_array(
            strategic_features_list, feature_names, dtype='float32'
        )
        labels_arrays = convert_labels_to_arrays(labels_list)
        masks_arrays = convert_masks_to_arrays(masks_list)

        return df, strategic_features_array, labels_arrays, masks_arrays


def delete_batch_files(batch_files: List[str]) -> None:
    """Delete intermediate batch files.

    Args:
        batch_files: List of batch file paths to delete
    """
    deleted_count = 0
    failed_count = 0

    for batch_file in batch_files:
        try:
            if os.path.exists(batch_file):
                os.remove(batch_file)
                deleted_count += 1
        except Exception as e:
            print(f"  Warning: Failed to delete {batch_file}: {e}")
            failed_count += 1

    if deleted_count > 0:
        print(f"  Cleaned up {deleted_count} intermediate batch files")
    if failed_count > 0:
        print(f"  Warning: Failed to delete {failed_count} batch files")


def process_files_parallel(
    file_paths: List[str],
    config: Dict,
    num_workers: int = None
) -> List[str]:
    """Process multiple files in parallel using multiprocessing with batching.

    This function now processes files in batches to avoid memory overflow.
    Each batch is saved to disk and memory is cleared before the next batch.

    Args:
        file_paths: List of CSV file paths
        config: Configuration dictionary
        num_workers: Number of worker processes (default: cpu_count() - 2)

    Returns:
        List of batch file paths (e.g., ["data/intermediate/batch_0000.npz", ...])
    """
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 2)

    # Handle empty file list
    if not file_paths:
        return []

    # Get batch configuration
    batch_size = config.get('preprocessing', {}).get('batch_size', 500)
    intermediate_dir = config.get('preprocessing', {}).get('intermediate_dir', 'data/intermediate')
    monitor_memory = config.get('preprocessing', {}).get('monitor_memory', False)

    # Split files into batches
    num_batches = (len(file_paths) + batch_size - 1) // batch_size  # Ceiling division
    batches = [file_paths[i:i+batch_size] for i in range(0, len(file_paths), batch_size)]

    print(f"\nProcessing {len(file_paths)} files with {num_workers} workers...")
    print(f"Batch configuration: {batch_size} files/batch, {num_batches} batches")
    print(f"Intermediate directory: {intermediate_dir}")

    # Ensure intermediate directory exists
    os.makedirs(intermediate_dir, exist_ok=True)

    batch_file_paths = []

    # Process each batch
    for batch_num, batch_files in enumerate(batches):
        print(f"\n{'='*60}")
        print(f"Processing batch {batch_num + 1}/{num_batches} ({len(batch_files)} files)...")
        print(f"{'='*60}")

        if monitor_memory:
            log_memory_usage(f"Before batch {batch_num + 1}")

        # Accumulate results for this batch only
        batch_raw_rows = []
        batch_strategic_features = []
        batch_labels = []
        batch_masks = []

        # Use sequential processing for single worker or single file
        if num_workers == 1 or len(batch_files) == 1:
            for file_path in tqdm(batch_files, desc=f"Batch {batch_num + 1}/{num_batches}"):
                raw_rows, strategic_features, labels, masks = process_single_file(file_path, config)

                batch_raw_rows.extend(raw_rows)
                batch_strategic_features.extend(strategic_features)
                batch_labels.extend(labels)
                batch_masks.extend(masks)

                # Free memory
                gc.collect()
        else:
            # Use parallel processing with multiprocessing.Pool
            # Create a partial function with config bound
            worker_func = functools.partial(process_single_file, config=config)

            # Use context manager to ensure proper cleanup
            with mp.Pool(processes=num_workers) as pool:
                # Use imap_unordered for better memory efficiency and progress tracking
                results_iter = pool.imap_unordered(worker_func, batch_files)

                # Process results as they come in with progress bar
                for raw_rows, strategic_features, labels, masks in tqdm(
                    results_iter,
                    total=len(batch_files),
                    desc=f"Batch {batch_num + 1}/{num_batches}"
                ):
                    batch_raw_rows.extend(raw_rows)
                    batch_strategic_features.extend(strategic_features)
                    batch_labels.extend(labels)
                    batch_masks.extend(masks)

                    # Free memory periodically
                    gc.collect()

        # Convert batch raw rows to DataFrame
        if batch_raw_rows:
            batch_df = pd.DataFrame(batch_raw_rows)
        else:
            batch_df = pd.DataFrame()

        print(f"\nBatch {batch_num + 1} processed {len(batch_raw_rows):,} samples")

        # Save batch to file
        batch_file = save_batch_to_file(
            batch_num=batch_num,
            df=batch_df,
            strategic_features=batch_strategic_features,
            labels=batch_labels,
            masks=batch_masks,
            file_paths=batch_files,
            intermediate_dir=intermediate_dir
        )
        batch_file_paths.append(batch_file)

        # Clear batch data from memory
        del batch_raw_rows, batch_strategic_features, batch_labels, batch_masks, batch_df
        gc.collect()

    print(f"\n{'='*60}")
    print(f"All {num_batches} batches processed and saved")
    print(f"{'='*60}\n")

    return batch_file_paths


def discover_csv_files(data_root: str, max_games: int = None) -> List[str]:
    """Discover all CSV files in data_root subdirectories.

    Args:
        data_root: Root directory containing 2_games/, 3_games/, 4_games/
        max_games: Optional limit on number of games

    Returns:
        List of CSV file paths
    """
    csv_files = []

    for subdir in ["2_games", "3_games", "4_games"]:
        dir_path = Path(data_root) / subdir
        if not dir_path.exists():
            continue

        files = list(dir_path.glob("*.csv"))
        csv_files.extend(str(f) for f in files)

        if max_games is not None and len(csv_files) >= max_games:
            break

    # Limit if max_games specified
    if max_games is not None:
        csv_files = csv_files[:max_games]

    return csv_files
