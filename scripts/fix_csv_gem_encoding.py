#!/usr/bin/env python3
"""
Script to repair CSV files with gem take encoding mismatch.

Fixes rows where action_type = "take 3 tokens" but values are in gem_take2 columns
instead of gem_take3 columns.
"""

import pandas as pd
import os
import sys
from pathlib import Path
import argparse


def find_csv_files(base_dir: str) -> list:
    """Find all CSV files in the specified directories."""
    csv_files = []
    for subdir in ['2_games', '3_games', '4_games']:
        dir_path = Path(base_dir) / subdir
        if dir_path.exists():
            csv_files.extend(dir_path.glob('*.csv'))
    return sorted(csv_files)


def fix_csv_file(file_path: Path, dry_run: bool = False) -> int:
    """
    Fix a single CSV file by moving gem_take2 values to gem_take3 when action_type = "take 3 tokens".

    Returns:
        Number of rows fixed
    """
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Column names for gem_take3 and gem_take2
    gem_take3_cols = ['gem_take3_white', 'gem_take3_blue', 'gem_take3_green', 'gem_take3_red', 'gem_take3_black']
    gem_take2_cols = ['gem_take2_white', 'gem_take2_blue', 'gem_take2_green', 'gem_take2_red', 'gem_take2_black']

    # Find rows that need fixing:
    # - action_type = "take 3 tokens"
    # - All gem_take3 columns are NaN
    # - At least one gem_take2 column is not NaN
    mask = (
        (df['action_type'] == 'take 3 tokens') &
        (df[gem_take3_cols].isna().all(axis=1)) &
        (~df[gem_take2_cols].isna().any(axis=1))
    )

    rows_to_fix = mask.sum()

    if rows_to_fix > 0 and not dry_run:
        # Create a backup
        backup_path = file_path.with_suffix('.csv.backup')
        df.to_csv(backup_path, index=False)

        # Move values from gem_take2 to gem_take3
        df.loc[mask, gem_take3_cols] = df.loc[mask, gem_take2_cols].values

        # Set gem_take2 to NaN for those rows
        df.loc[mask, gem_take2_cols] = float('nan')

        # Write the corrected data back
        df.to_csv(file_path, index=False)

    return rows_to_fix


def main():
    parser = argparse.ArgumentParser(description='Fix CSV gem take encoding mismatch')
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview changes without modifying files')
    parser.add_argument('--base-dir', default='data/games',
                        help='Base directory containing game CSV files (default: data/games)')
    args = parser.parse_args()

    base_dir = Path(args.base_dir)

    if not base_dir.exists():
        print(f"Error: Directory '{base_dir}' does not exist")
        sys.exit(1)

    csv_files = find_csv_files(base_dir)

    if not csv_files:
        print(f"No CSV files found in {base_dir}/{{2_games,3_games,4_games}}")
        sys.exit(0)

    print(f"Found {len(csv_files)} CSV files to process")
    if args.dry_run:
        print("DRY RUN MODE - No files will be modified")
    print()

    total_rows_fixed = 0
    files_with_fixes = 0

    for csv_file in csv_files:
        try:
            rows_fixed = fix_csv_file(csv_file, dry_run=args.dry_run)
            if rows_fixed > 0:
                files_with_fixes += 1
                total_rows_fixed += rows_fixed
                status = "[DRY RUN] Would fix" if args.dry_run else "Fixed"
                print(f"{status} {rows_fixed} rows in {csv_file.relative_to(base_dir.parent)}")
        except Exception as e:
            print(f"Error processing {csv_file}: {e}", file=sys.stderr)

    print()
    print(f"Summary:")
    print(f"  Files processed: {len(csv_files)}")
    print(f"  Files with fixes: {files_with_fixes}")
    print(f"  Total rows fixed: {total_rows_fixed}")

    if not args.dry_run and total_rows_fixed > 0:
        print()
        print("Backups created with .csv.backup extension")


if __name__ == '__main__':
    main()
