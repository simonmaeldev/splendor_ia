#!/usr/bin/env python3
"""
Dataset Validation Script

Validates the exported ML dataset for correctness, consistency, and ML-readiness.
"""

import csv
import sys
import sqlite3
from pathlib import Path

def validate_dataset(csv_path: str, db_path: str):
    """Validate the exported dataset."""
    print(f"Validating dataset: {csv_path}")
    print("=" * 60)

    # Check file exists
    if not Path(csv_path).exists():
        print(f"ERROR: File not found: {csv_path}")
        return False

    # Read CSV
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        rows = list(reader)

    print(f"✓ File exists and is readable")
    print(f"  Rows: {len(rows):,}")
    print(f"  Columns: {len(headers)}")

    # Validate column count
    if len(headers) != 402:
        print(f"ERROR: Expected 402 columns, got {len(headers)}")
        return False
    print(f"✓ Correct number of columns (402)")

    # Validate all rows have same column count
    for i, row in enumerate(rows):
        if len(row) != 402:
            print(f"ERROR: Row {i} has {len(row)} columns, expected 402")
            return False
    print(f"✓ All rows have consistent column count")

    # Check for expected headers
    expected_start = ['game_id', 'num_players', 'turn_number']
    if headers[:3] != expected_start:
        print(f"ERROR: First 3 headers should be {expected_start}, got {headers[:3]}")
        return False
    print(f"✓ Headers start with expected fields")

    # Validate action types
    action_type_idx = headers.index('action_type')
    action_types = set()
    for row in rows:
        action_types.add(row[action_type_idx])

    expected_action_types = {'build', 'reserve', 'take 2 tokens', 'take 3 tokens'}
    if not action_types.issubset(expected_action_types):
        unexpected = action_types - expected_action_types
        print(f"ERROR: Unexpected action types: {unexpected}")
        return False
    print(f"✓ All action types are valid: {action_types}")

    # Validate game_id references
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    game_ids_csv = set(int(row[0]) for row in rows)
    game_ids_db = set(row[0] for row in cursor.execute("SELECT DISTINCT IDGame FROM Action"))

    if not game_ids_csv.issubset(game_ids_db):
        extra = game_ids_csv - game_ids_db
        print(f"ERROR: CSV contains game IDs not in database: {extra}")
        return False
    print(f"✓ All game IDs exist in database")
    print(f"  Unique games in CSV: {len(game_ids_csv)}")
    print(f"  Unique games in DB: {len(game_ids_db)}")

    # Validate sample count
    db_action_count = cursor.execute("SELECT COUNT(*) FROM Action").fetchone()[0]
    csv_sample_count = len(rows)
    print(f"  DB actions: {db_action_count:,}")
    print(f"  CSV samples: {csv_sample_count:,}")

    if csv_sample_count > db_action_count:
        print(f"ERROR: More CSV samples than DB actions")
        return False
    print(f"✓ Sample count is reasonable")

    conn.close()

    # Check for NaN patterns (basic check)
    nan_count = sum(1 for row in rows for cell in row if cell.lower() in ['nan', ''])
    print(f"  NaN/empty cells: {nan_count:,}")

    print("=" * 60)
    print("✓ Dataset validation PASSED")
    return True


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'data/training_dataset.csv'
    db_path = sys.argv[2] if len(sys.argv) > 2 else 'data/games.db'

    success = validate_dataset(csv_path, db_path)
    sys.exit(0 if success else 1)
