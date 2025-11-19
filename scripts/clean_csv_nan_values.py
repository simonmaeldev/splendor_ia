#!/usr/bin/env python3
"""
Script to clean 'nan' string values from CSV files created after Nov 18, 2024 22:45:00.

Replaces all occurrences of the string 'nan' with empty strings to ensure data consistency.
"""

import os
import sys
from pathlib import Path
import argparse
from datetime import datetime


def find_csv_files_after_timestamp(base_dir: str, timestamp: datetime) -> list:
    """Find all CSV files in the specified directories created after the given timestamp."""
    csv_files = []
    timestamp_unix = timestamp.timestamp()

    for subdir in ['2_games', '3_games', '4_games']:
        dir_path = Path(base_dir) / subdir
        if dir_path.exists():
            for csv_file in dir_path.glob('*.csv'):
                # Get file modification time
                file_mtime = os.path.getmtime(csv_file)
                if file_mtime > timestamp_unix:
                    csv_files.append(csv_file)

    return sorted(csv_files)


def clean_csv_file(file_path: Path, dry_run: bool = False) -> int:
    """
    Clean a single CSV file by replacing all 'nan' strings with empty strings.

    Returns:
        Number of 'nan' replacements made
    """
    try:
        # Read the entire file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # Replace patterns where 'nan' appears as a CSV value
        # Loop until no more replacements can be made
        while True:
            new_content = content
            # Handle: ,nan, -> ,,
            new_content = new_content.replace(',nan,', ',,')
            # Handle: ,nan\n -> ,\n (end of line)
            new_content = new_content.replace(',nan\n', ',\n')
            # Handle: ,nan\r\n -> ,\r\n (Windows line endings)
            new_content = new_content.replace(',nan\r\n', ',\r\n')

            # If no changes were made, we're done
            if new_content == content:
                break
            content = new_content

        # Count actual changes made
        if content != original_content:
            # Count how many replacements were made by comparing before/after
            replacements = original_content.count(',nan,') + original_content.count(',nan\n') + original_content.count(',nan\r\n')

            if not dry_run:
                # Write the cleaned content back
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)

            return replacements

        return 0

    except Exception as e:
        print(f"Error processing {file_path}: {e}", file=sys.stderr)
        return 0


def main():
    parser = argparse.ArgumentParser(description='Clean CSV nan string values from files created after Nov 18 22:45')
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview changes without modifying files')
    parser.add_argument('--base-dir', default='data/games',
                        help='Base directory containing game CSV files (default: data/games)')
    parser.add_argument('--timestamp', default='2025-11-18 22:45:00',
                        help='Timestamp filter (default: 2025-11-18 22:45:00)')
    args = parser.parse_args()

    base_dir = Path(args.base_dir)

    if not base_dir.exists():
        print(f"Error: Directory '{base_dir}' does not exist")
        sys.exit(1)

    # Parse the timestamp
    try:
        timestamp = datetime.strptime(args.timestamp, '%Y-%m-%d %H:%M:%S')
    except ValueError:
        print(f"Error: Invalid timestamp format. Use YYYY-MM-DD HH:MM:SS")
        sys.exit(1)

    print(f"Finding CSV files created after {timestamp}")
    csv_files = find_csv_files_after_timestamp(base_dir, timestamp)

    if not csv_files:
        print(f"No CSV files found created after {timestamp}")
        sys.exit(0)

    print(f"Found {len(csv_files)} CSV files to process")
    if args.dry_run:
        print("DRY RUN MODE - No files will be modified")
    print()

    total_replacements = 0
    files_modified = 0

    for csv_file in csv_files:
        try:
            replacements = clean_csv_file(csv_file, dry_run=args.dry_run)
            if replacements > 0:
                files_modified += 1
                total_replacements += replacements
                status = "[DRY RUN] Would replace" if args.dry_run else "Replaced"
                print(f"{status} {replacements} 'nan' values in {csv_file.relative_to(base_dir.parent)}")
        except Exception as e:
            print(f"Error processing {csv_file}: {e}", file=sys.stderr)

    print()
    print(f"Summary:")
    print(f"  Files scanned: {len(csv_files)}")
    print(f"  Files modified: {files_modified}")
    print(f"  Total 'nan' replacements: {total_replacements}")


if __name__ == '__main__':
    main()
