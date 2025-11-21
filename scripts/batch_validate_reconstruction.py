#!/usr/bin/env python3
"""
Batch validation script for state reconstruction across dataset.

Samples games from each player count and runs comprehensive validation.

Usage:
    python scripts/batch_validate_reconstruction.py --sample-size 20 --player-counts 2 3 4
    python scripts/batch_validate_reconstruction.py --sample-size 10 --player-counts 3 --verbose
"""

import sys
import argparse
import random
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

# Import test_reconstruction functions
from test_reconstruction import test_csv_file


def find_games_for_player_count(data_dir: Path, player_count: int) -> List[Path]:
    """Find all game CSV files for a specific player count."""
    games_dir = data_dir / f"{player_count}_games"
    if not games_dir.exists():
        return []

    return list(games_dir.glob("*.csv"))


def sample_games(game_files: List[Path], sample_size: int) -> List[Path]:
    """Sample random games from list."""
    if len(game_files) <= sample_size:
        return game_files

    return random.sample(game_files, sample_size)


def batch_validate(data_dir: str, player_counts: List[int], sample_size: int,
                   verbose: bool = False) -> Dict[str, Any]:
    """
    Run batch validation across player counts.

    Args:
        data_dir: Base data directory (e.g., 'data/games')
        player_counts: List of player counts to test (e.g., [2, 3, 4])
        sample_size: Number of games to sample per player count
        verbose: Verbose output

    Returns:
        Dictionary with validation results
    """
    data_path = Path(data_dir)

    results = {
        'timestamp': datetime.now().isoformat(),
        'sample_size_per_player_count': sample_size,
        'player_counts': player_counts,
        'player_count_results': {},
        'overall': {
            'total_files': 0,
            'total_rows': 0,
            'reconstruction_success': 0,
            'getmoves_success': 0,
            'action_validation_success': 0,
            'round_trip_success': 0,
            'all_tests_passed': 0
        }
    }

    for player_count in player_counts:
        print(f"\n{'=' * 80}")
        print(f"Testing {player_count}-player games")
        print(f"{'=' * 80}")

        # Find games
        game_files = find_games_for_player_count(data_path, player_count)
        print(f"Found {len(game_files)} games for {player_count} players")

        if not game_files:
            print(f"  No games found, skipping")
            continue

        # Sample games
        sampled_games = sample_games(game_files, sample_size)
        print(f"Sampling {len(sampled_games)} games")

        # Test each game
        player_count_stats = {
            'total_files': len(sampled_games),
            'total_rows': 0,
            'reconstruction_success': 0,
            'getmoves_success': 0,
            'action_validation_success': 0,
            'round_trip_success': 0,
            'all_tests_passed': 0,
            'failed_files': []
        }

        for game_file in sampled_games:
            try:
                stats = test_csv_file(str(game_file), max_rows=None, verbose=verbose)

                player_count_stats['total_rows'] += stats['total_rows']
                player_count_stats['reconstruction_success'] += stats['reconstruction_success']
                player_count_stats['getmoves_success'] += stats['getmoves_success']
                player_count_stats['action_validation_success'] += stats['action_validation_success']
                player_count_stats['round_trip_success'] += stats['round_trip_success']
                player_count_stats['all_tests_passed'] += stats['all_tests_passed']

                if stats['failed_rows']:
                    player_count_stats['failed_files'].append({
                        'file': str(game_file),
                        'num_failures': len(stats['failed_rows']),
                        'failures': stats['failed_rows'][:3]  # Keep first 3 failures
                    })

            except Exception as e:
                print(f"  Error testing {game_file}: {e}")
                player_count_stats['failed_files'].append({
                    'file': str(game_file),
                    'error': str(e)
                })

        # Update overall stats
        results['overall']['total_files'] += player_count_stats['total_files']
        results['overall']['total_rows'] += player_count_stats['total_rows']
        results['overall']['reconstruction_success'] += player_count_stats['reconstruction_success']
        results['overall']['getmoves_success'] += player_count_stats['getmoves_success']
        results['overall']['action_validation_success'] += player_count_stats['action_validation_success']
        results['overall']['round_trip_success'] += player_count_stats['round_trip_success']
        results['overall']['all_tests_passed'] += player_count_stats['all_tests_passed']

        results['player_count_results'][player_count] = player_count_stats

    return results


def print_batch_summary(results: Dict[str, Any]):
    """Print batch validation summary."""
    print("\n" + "=" * 80)
    print("BATCH VALIDATION SUMMARY")
    print("=" * 80)

    print(f"\nTimestamp: {results['timestamp']}")
    print(f"Player counts tested: {results['player_counts']}")
    print(f"Sample size per player count: {results['sample_size_per_player_count']}")

    overall = results['overall']
    total_rows = overall['total_rows']

    if total_rows == 0:
        print("\nNo rows tested!")
        return

    print(f"\nOverall Results:")
    print(f"  Total files:         {overall['total_files']}")
    print(f"  Total rows:          {total_rows}")
    print(f"  Reconstruction:      {overall['reconstruction_success']}/{total_rows} "
          f"({100*overall['reconstruction_success']/total_rows:.1f}%)")
    print(f"  getMoves():          {overall['getmoves_success']}/{total_rows} "
          f"({100*overall['getmoves_success']/total_rows:.1f}%)")
    print(f"  Action Validation:   {overall['action_validation_success']}/{total_rows} "
          f"({100*overall['action_validation_success']/total_rows:.1f}%)")
    print(f"  Round-trip:          {overall['round_trip_success']}/{total_rows} "
          f"({100*overall['round_trip_success']/total_rows:.1f}%)")
    print(f"  All Tests Passed:    {overall['all_tests_passed']}/{total_rows} "
          f"({100*overall['all_tests_passed']/total_rows:.1f}%)")

    # Per-player-count breakdown
    print("\nPer-Player-Count Breakdown:")
    for player_count, stats in results['player_count_results'].items():
        total = stats['total_rows']
        if total == 0:
            continue

        print(f"\n  {player_count} players ({stats['total_files']} files, {total} rows):")
        print(f"    Reconstruction:      {stats['reconstruction_success']}/{total} "
              f"({100*stats['reconstruction_success']/total:.1f}%)")
        print(f"    getMoves():          {stats['getmoves_success']}/{total} "
              f"({100*stats['getmoves_success']/total:.1f}%)")
        print(f"    Action Validation:   {stats['action_validation_success']}/{total} "
              f"({100*stats['action_validation_success']/total:.1f}%)")
        print(f"    Round-trip:          {stats['round_trip_success']}/{total} "
              f"({100*stats['round_trip_success']/total:.1f}%)")
        print(f"    All Tests Passed:    {stats['all_tests_passed']}/{total} "
              f"({100*stats['all_tests_passed']/total:.1f}%)")

        if stats['failed_files']:
            print(f"    Failed files:        {len(stats['failed_files'])}")

    # Success check
    if overall['all_tests_passed'] == total_rows:
        print("\n✓ ALL VALIDATION TESTS PASSED!")
    else:
        failed = total_rows - overall['all_tests_passed']
        print(f"\n✗ {failed} row(s) failed one or more tests")


def save_report(results: Dict[str, Any], output_file: str):
    """Save validation report to file."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("BATCH VALIDATION REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Timestamp: {results['timestamp']}\n")
        f.write(f"Player counts tested: {results['player_counts']}\n")
        f.write(f"Sample size per player count: {results['sample_size_per_player_count']}\n\n")

        overall = results['overall']
        total_rows = overall['total_rows']

        f.write("Overall Results:\n")
        f.write(f"  Total files:         {overall['total_files']}\n")
        f.write(f"  Total rows:          {total_rows}\n")

        if total_rows > 0:
            f.write(f"  Reconstruction:      {overall['reconstruction_success']}/{total_rows} "
                   f"({100*overall['reconstruction_success']/total_rows:.1f}%)\n")
            f.write(f"  getMoves():          {overall['getmoves_success']}/{total_rows} "
                   f"({100*overall['getmoves_success']/total_rows:.1f}%)\n")
            f.write(f"  Action Validation:   {overall['action_validation_success']}/{total_rows} "
                   f"({100*overall['action_validation_success']/total_rows:.1f}%)\n")
            f.write(f"  Round-trip:          {overall['round_trip_success']}/{total_rows} "
                   f"({100*overall['round_trip_success']/total_rows:.1f}%)\n")
            f.write(f"  All Tests Passed:    {overall['all_tests_passed']}/{total_rows} "
                   f"({100*overall['all_tests_passed']/total_rows:.1f}%)\n\n")

        f.write("Per-Player-Count Breakdown:\n")
        for player_count, stats in results['player_count_results'].items():
            total = stats['total_rows']
            if total == 0:
                continue

            f.write(f"\n  {player_count} players ({stats['total_files']} files, {total} rows):\n")
            f.write(f"    Reconstruction:      {stats['reconstruction_success']}/{total} "
                   f"({100*stats['reconstruction_success']/total:.1f}%)\n")
            f.write(f"    getMoves():          {stats['getmoves_success']}/{total} "
                   f"({100*stats['getmoves_success']/total:.1f}%)\n")
            f.write(f"    Action Validation:   {stats['action_validation_success']}/{total} "
                   f"({100*stats['action_validation_success']/total:.1f}%)\n")
            f.write(f"    Round-trip:          {stats['round_trip_success']}/{total} "
                   f"({100*stats['round_trip_success']/total:.1f}%)\n")
            f.write(f"    All Tests Passed:    {stats['all_tests_passed']}/{total} "
                   f"({100*stats['all_tests_passed']/total:.1f}%)\n")

            if stats['failed_files']:
                f.write(f"\n    Failed files ({len(stats['failed_files'])}):\n")
                for failure in stats['failed_files'][:10]:
                    f.write(f"      {failure['file']}\n")
                    if 'error' in failure:
                        f.write(f"        Error: {failure['error']}\n")
                    elif 'num_failures' in failure:
                        f.write(f"        {failure['num_failures']} row(s) failed\n")
                if len(stats['failed_files']) > 10:
                    f.write(f"      ... and {len(stats['failed_files']) - 10} more\n")

    print(f"\nReport saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Batch validate state reconstruction')
    parser.add_argument('--data-dir', default='data/games',
                       help='Base data directory (default: data/games)')
    parser.add_argument('--sample-size', type=int, default=20,
                       help='Number of games to sample per player count (default: 20)')
    parser.add_argument('--player-counts', nargs='+', type=int, default=[2, 3, 4],
                       help='Player counts to test (default: 2 3 4)')
    parser.add_argument('--output', default='logs/reconstruction_validation_report.txt',
                       help='Output report file (default: logs/reconstruction_validation_report.txt)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')

    args = parser.parse_args()

    print("=" * 80)
    print("BATCH VALIDATION")
    print("=" * 80)
    print(f"Data directory: {args.data_dir}")
    print(f"Sample size per player count: {args.sample_size}")
    print(f"Player counts: {args.player_counts}")

    # Run batch validation
    results = batch_validate(args.data_dir, args.player_counts, args.sample_size, args.verbose)

    # Print summary
    print_batch_summary(results)

    # Save report
    save_report(results, args.output)


if __name__ == '__main__':
    main()
