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
    get_num_gem_removal_classes,
)
from .constants import (
    COMBO_TO_CLASS_TAKE3,
    REMOVAL_TO_CLASS,
    NUM_GEM_REMOVAL_CLASSES,
)
from .feature_engineering import extract_all_features, get_all_feature_names
from utils.state_reconstruction import reconstruct_board_from_csv_row


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
    nonzero_cards = [c for c in visible_cards if c[2]]
    zero_cards = [c for c in visible_cards if not c[2]]

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

        # Extract game_id from filename
        game_id = int(Path(file_path).stem)
        df["game_id"] = game_id

        # No longer generating encoding mappings per file - using constants!

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


def process_files_parallel(
    file_paths: List[str],
    config: Dict,
    num_workers: int = None
) -> Tuple[pd.DataFrame, List[Dict], List[Dict], List[Dict]]:
    """Process multiple files in parallel using multiprocessing.

    Args:
        file_paths: List of CSV file paths
        config: Configuration dictionary
        num_workers: Number of worker processes (default: cpu_count() - 2)

    Returns:
        Tuple of (combined_df, strategic_features_list, labels_list, masks_list)
    """
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 2)

    print(f"\nProcessing {len(file_paths)} files with {num_workers} workers...")

    # Handle empty file list
    if not file_paths:
        return pd.DataFrame(), [], [], []

    all_raw_rows = []
    all_strategic_features = []
    all_labels = []
    all_masks = []

    # Use sequential processing for single worker or single file
    if num_workers == 1 or len(file_paths) == 1:
        for file_path in tqdm(file_paths, desc="Processing files"):
            raw_rows, strategic_features, labels, masks = process_single_file(file_path, config)

            all_raw_rows.extend(raw_rows)
            all_strategic_features.extend(strategic_features)
            all_labels.extend(labels)
            all_masks.extend(masks)

            # Free memory
            gc.collect()
    else:
        # Use parallel processing with multiprocessing.Pool
        # Create a partial function with config bound
        worker_func = functools.partial(process_single_file, config=config)

        # Use context manager to ensure proper cleanup
        with mp.Pool(processes=num_workers) as pool:
            # Use imap_unordered for better memory efficiency and progress tracking
            results_iter = pool.imap_unordered(worker_func, file_paths)

            # Process results as they come in with progress bar
            for raw_rows, strategic_features, labels, masks in tqdm(
                results_iter,
                total=len(file_paths),
                desc="Processing files"
            ):
                all_raw_rows.extend(raw_rows)
                all_strategic_features.extend(strategic_features)
                all_labels.extend(labels)
                all_masks.extend(masks)

                # Free memory periodically
                gc.collect()

    # Convert raw rows to DataFrame
    if all_raw_rows:
        df = pd.DataFrame(all_raw_rows)
    else:
        df = pd.DataFrame()

    print(f"\nProcessed {len(all_raw_rows)} total samples")

    return df, all_strategic_features, all_labels, all_masks


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
