#!/usr/bin/env python3
"""
Test script for state reconstruction validation.

This script tests the reconstruction of Board objects from CSV features
and validates that getMoves() works correctly and actions are valid.

Usage:
    python scripts/test_reconstruction.py --csv data/games/3_games/3869.csv
    python scripts/test_reconstruction.py --csv data/games/2_games/*.csv --sample 1
    python scripts/test_reconstruction.py --csv data/games/4_games/*.csv --sample 5 --verbose
"""

import sys
import csv
import argparse
import glob
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from utils.state_reconstruction import reconstruct_board_from_csv_row
from utils.validate_reconstruction import (
    parse_action_from_csv,
    validate_action_in_moves,
    validate_round_trip,
    encode_board_state
)
from splendor.csv_exporter import generate_input_column_headers


def load_csv_rows(csv_path: str, max_rows: int = None) -> List[Dict[str, Any]]:
    """Load rows from CSV file."""
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_rows and i >= max_rows:
                break
            rows.append(row)
    return rows


def extract_features_from_row(row: Dict[str, Any]) -> List[float]:
    """Extract 382 input features from CSV row."""
    headers = generate_input_column_headers()
    features = []
    for header in headers:
        val = row[header]
        if val == '' or val == 'nan':
            features.append(float('nan'))
        else:
            features.append(float(val))
    return features


def test_single_row(row: Dict[str, Any], verbose: bool = False) -> Dict[str, Any]:
    """
    Test reconstruction for a single CSV row.

    Returns dict with test results.
    """
    result = {
        'reconstruction_success': False,
        'getmoves_success': False,
        'action_validation_success': False,
        'round_trip_success': False,
        'num_valid_moves': 0,
        'errors': []
    }

    try:
        # Test 1: Reconstruct board
        board = reconstruct_board_from_csv_row(row)
        result['reconstruction_success'] = True

        if verbose:
            print(f"  Reconstructed: turn={board.nbTurn}, current_player={board.currentPlayer}, "
                  f"num_players={len(board.players)}")

    except Exception as e:
        result['errors'].append(f"Reconstruction failed: {str(e)}")
        return result

    try:
        # Test 2: Call getMoves()
        valid_moves = board.getMoves()
        result['getmoves_success'] = True
        result['num_valid_moves'] = len(valid_moves)

        if verbose:
            print(f"  getMoves() returned {len(valid_moves)} valid moves")

    except Exception as e:
        result['errors'].append(f"getMoves() failed: {str(e)}")
        return result

    try:
        # Test 3: Validate action
        action = parse_action_from_csv(row, board)
        is_valid, message = validate_action_in_moves(board, action)
        result['action_validation_success'] = is_valid

        if verbose:
            if is_valid:
                print(f"  Action validation: ✓ {message}")
            else:
                print(f"  Action validation: ✗ {message}")

        if not is_valid:
            result['errors'].append(f"Action validation: {message}")

    except Exception as e:
        result['errors'].append(f"Action validation failed: {str(e)}")

    try:
        # Test 4: Round-trip encoding
        turn_num = int(float(row['turn_number']))
        original_features = extract_features_from_row(row)
        round_trip_result = validate_round_trip(original_features, board, turn_num)

        result['round_trip_success'] = round_trip_result['success']

        if verbose:
            if round_trip_result['success']:
                print(f"  Round-trip: ✓ Features match")
            else:
                num_diffs = round_trip_result['comparison']['num_diffs']
                print(f"  Round-trip: ✗ {num_diffs} feature differences")

        if not round_trip_result['success']:
            num_diffs = round_trip_result['comparison']['num_diffs']
            result['errors'].append(f"Round-trip: {num_diffs} feature differences")

    except Exception as e:
        result['errors'].append(f"Round-trip validation failed: {str(e)}")

    return result


def test_csv_file(csv_path: str, max_rows: int = None, verbose: bool = False) -> Dict[str, Any]:
    """
    Test reconstruction for all rows in a CSV file.

    Returns summary statistics.
    """
    print(f"\nTesting: {csv_path}")

    rows = load_csv_rows(csv_path, max_rows)
    print(f"  Loaded {len(rows)} rows")

    stats = {
        'total_rows': len(rows),
        'reconstruction_success': 0,
        'getmoves_success': 0,
        'action_validation_success': 0,
        'round_trip_success': 0,
        'all_tests_passed': 0,
        'failed_rows': []
    }

    for i, row in enumerate(rows):
        if verbose:
            print(f"\n  Row {i + 1}:")

        result = test_single_row(row, verbose)

        if result['reconstruction_success']:
            stats['reconstruction_success'] += 1
        if result['getmoves_success']:
            stats['getmoves_success'] += 1
        if result['action_validation_success']:
            stats['action_validation_success'] += 1
        if result['round_trip_success']:
            stats['round_trip_success'] += 1

        if (result['reconstruction_success'] and result['getmoves_success'] and
            result['action_validation_success'] and result['round_trip_success']):
            stats['all_tests_passed'] += 1
        else:
            stats['failed_rows'].append({
                'row': i + 1,
                'errors': result['errors']
            })

    return stats


def print_summary(all_stats: List[Dict[str, Any]], csv_files: List[str]):
    """Print summary of all tests."""
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    total_rows = sum(s['total_rows'] for s in all_stats)
    total_reconstruction = sum(s['reconstruction_success'] for s in all_stats)
    total_getmoves = sum(s['getmoves_success'] for s in all_stats)
    total_action_validation = sum(s['action_validation_success'] for s in all_stats)
    total_round_trip = sum(s['round_trip_success'] for s in all_stats)
    total_all_passed = sum(s['all_tests_passed'] for s in all_stats)

    print(f"\nTested {len(csv_files)} file(s) with {total_rows} total rows")
    print(f"\nResults:")
    print(f"  Reconstruction:      {total_reconstruction}/{total_rows} "
          f"({100*total_reconstruction/total_rows:.1f}%)")
    print(f"  getMoves():          {total_getmoves}/{total_rows} "
          f"({100*total_getmoves/total_rows:.1f}%)")
    print(f"  Action Validation:   {total_action_validation}/{total_rows} "
          f"({100*total_action_validation/total_rows:.1f}%)")
    print(f"  Round-trip:          {total_round_trip}/{total_rows} "
          f"({100*total_round_trip/total_rows:.1f}%)")
    print(f"  All Tests Passed:    {total_all_passed}/{total_rows} "
          f"({100*total_all_passed/total_rows:.1f}%)")

    # Print failed rows details
    failed_count = sum(len(s['failed_rows']) for s in all_stats)
    if failed_count > 0:
        print(f"\n{failed_count} row(s) failed one or more tests:")
        for i, stats in enumerate(all_stats):
            if stats['failed_rows']:
                print(f"\n  File: {csv_files[i]}")
                for failure in stats['failed_rows'][:5]:  # Show first 5 failures
                    print(f"    Row {failure['row']}: {failure['errors']}")
                if len(stats['failed_rows']) > 5:
                    print(f"    ... and {len(stats['failed_rows']) - 5} more")
    else:
        print("\n✓ All tests passed!")


def main():
    parser = argparse.ArgumentParser(description='Test state reconstruction')
    parser.add_argument('--csv', nargs='+', required=True,
                       help='CSV file(s) to test (supports wildcards)')
    parser.add_argument('--sample', type=int, default=None,
                       help='Number of rows to test per file (default: all)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')

    args = parser.parse_args()

    # Expand wildcards
    csv_files = []
    for pattern in args.csv:
        expanded = glob.glob(pattern)
        if expanded:
            csv_files.extend(expanded)
        else:
            csv_files.append(pattern)

    if not csv_files:
        print("Error: No CSV files found")
        sys.exit(1)

    # Test each file
    all_stats = []
    for csv_file in csv_files:
        try:
            stats = test_csv_file(csv_file, args.sample, args.verbose)
            all_stats.append(stats)
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            import traceback
            traceback.print_exc()

    # Print summary
    print_summary(all_stats, csv_files)


if __name__ == '__main__':
    main()
