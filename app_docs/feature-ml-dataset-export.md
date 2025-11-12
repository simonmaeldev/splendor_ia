# ML Dataset Export Script

**ADW ID:** N/A
**Date:** 2025-11-12
**Specification:** specs/ml_dataset_export.md

## Overview

A comprehensive Python script that extracts game state and action data from the SQLite database and transforms it into a machine learning-ready CSV format. The script encodes 1.7M+ state-action pairs into 381-dimensional input vectors with proper player rotation and NaN padding, paired with 7-head multi-task outputs for supervised learning.

## What Was Built

- **Export Script** (`scripts/export_ml_dataset.py`): Main 1,166-line script with complete ML dataset transformation pipeline
- **Validation Script** (`scripts/validate_dataset.py`): Dataset integrity checker for structure, consistency, and correctness
- **381-dimensional State Encoding**: Complete game state representation with:
  - Player rotation (current player always first)
  - NaN padding for 2-3 player games (supports all player counts)
  - Consistent color ordering across all features
  - 12 visible cards, 5 nobles, 4 players, board state
- **7-head Action Encoding**: Multi-task output structure with:
  - Action type classification (build/reserve/take tokens)
  - Card selection/reservation indices
  - Gem take patterns (2-hot/3-hot vectors)
  - Noble selection
  - Gem removal counts
- **Progress Tracking**: Real-time progress bar with percentage completion
- **Comprehensive Summary**: Dataset statistics, feature descriptions, and usage guidance
- **CLI Interface**: Command-line options for testing, validation, and customization

## Technical Implementation

### Files Modified

- `scripts/export_ml_dataset.py`: Complete ML dataset export implementation (new file, 1,166 lines)
  - Database connection and caching layer (cards/nobles loaded into memory)
  - Game state encoding functions (381 features)
  - Action encoding functions (7 output heads)
  - Progress tracking with ANSI terminal updates
  - CSV generation with descriptive column headers
  - Dataset summary report generator
  - Command-line interface with argparse

- `scripts/validate_dataset.py`: Dataset validation script (new file, 108 lines)
  - Column count verification (402 total)
  - Action type validation
  - Game ID cross-reference with database
  - Sample count consistency checks
  - NaN pattern detection

- `scripts/main.py`: Minor update to simulation config path (2 lines changed)
- `data/simulation_config.txt`: Minor config adjustment (2 lines changed)

### Key Changes

- **State Encoding Architecture**: Built complete 381-dimensional state encoder that:
  - Rotates players so current player is position 0
  - Pads missing players (2-3 player games) with NaN
  - Encodes visible cards (12 × 12 features)
  - Encodes nobles (5 × 6 features)
  - Encodes player states including reserved cards (4 × 49 features)
  - Maintains consistent color ordering: WHITE, BLUE, GREEN, RED, BLACK, GOLD

- **Action Encoding System**: Implemented 7-head multi-task output:
  - action_type: String classification ("build", "reserve", "take 2 tokens", "take 3 tokens")
  - card_selection: [0-14] for build actions (0-11 visible, 12-14 reserved)
  - card_reservation: [0-14] for reserve actions (0-11 visible, 12-14 top deck by level)
  - gem_take_3: 5-element 3-hot vector for taking 3 different tokens
  - gem_take_2: 5-element one-hot vector for taking 2 same tokens
  - noble_selection: [0-4] for noble position selection
  - gems_removed: 6-element count vector for overflow handling

- **Reserved Card Tracking**: Complex state reconstruction logic that:
  - Queries all RESERVE actions by player up to current turn
  - Queries all BUILD actions to identify builds from reserve
  - Computes current reserved cards = reserved - built
  - Encodes up to 3 reserved cards per player (36 features)

- **Performance Optimizations**:
  - In-memory caching of all cards and nobles (avoids repeated queries)
  - Batch progress updates (every 100 rows)
  - Efficient player state reconstruction
  - CSV buffering for faster writes

- **Comprehensive Validation**:
  - Input vector length validation (exactly 381)
  - Action encoding consistency checks
  - Database sample count verification
  - Column header validation (402 total)
  - Action type correctness

## How to Use

### Basic Export

Export the full dataset from the database:

```bash
python scripts/export_ml_dataset.py
```

This creates `data/training_dataset.csv` with all ~1.7M samples.

### Test with Limited Samples

Export only the first 100 samples for testing:

```bash
python scripts/export_ml_dataset.py --limit 100 --output data/test_export.csv
```

### Enable Validation Mode

Run with strict validation (slower but catches encoding errors):

```bash
python scripts/export_ml_dataset.py --validate
```

### Validate Exported Dataset

After export, validate the dataset structure and consistency:

```bash
python scripts/validate_dataset.py data/training_dataset.csv
```

### Command-Line Options

- `--db-path PATH`: Custom database path (default: `data/games.db`)
- `--output PATH`: Custom output CSV path (default: `data/training_dataset.csv`)
- `--validate`: Enable strict validation mode
- `--limit N`: Export only first N samples (for testing)
- `--quiet`: Suppress progress output
- `--help`: Show all options

### Loading the Dataset

Example Python code to load and use the dataset:

```python
import pandas as pd
import numpy as np

# Load CSV
df = pd.read_csv('data/training_dataset.csv')

# Separate inputs and outputs
input_cols = [col for col in df.columns if col.startswith(('num_players', 'turn_number', 'gems_', 'card', 'deck_', 'noble', 'player'))]
output_cols = [col for col in df.columns if col.startswith(('action_', 'card_selection', 'card_reservation', 'gem_take', 'noble_selection', 'gems_removed'))]

X = df[input_cols].values  # Shape: (N, 381)
y = df[output_cols].values  # Shape: (N, 20)

# Split by game_id to prevent data leakage
unique_games = df['game_id'].unique()
train_games, test_games = train_test_split(unique_games, test_size=0.2)

train_df = df[df['game_id'].isin(train_games)]
test_df = df[df['game_id'].isin(test_games)]
```

## Configuration

### Dataset Structure

- **Input Space**: 381 dimensions
  - `num_players` (1): Number of players in game
  - `turn_number` (1): Current turn number
  - `gems_board_*` (6): Tokens available on board (white, blue, green, red, black, gold)
  - `card{i}_*` (144): 12 visible cards × 12 features each (vp, level, costs, bonus)
  - `deck_level{i}_remaining` (3): Cards left in each deck
  - `noble{i}_*` (30): 5 nobles × 6 features each (vp, cost requirements)
  - `player{i}_*` (196): 4 players × 49 features each (position, vp, gems, reductions, reserved cards)

- **Output Space**: 20 dimensions across 7 heads
  - `action_type` (1): String classification
  - `card_selection` (1): Integer [0-14] or NaN
  - `card_reservation` (1): Integer [0-14] or NaN
  - `gem_take3_*` (5): 3-hot binary vector
  - `gem_take2_*` (5): One-hot binary vector
  - `noble_selection` (1): Integer [0-4] or NaN
  - `gems_removed_*` (6): Integer count vector

### Color Ordering

All color-based features use consistent ordering:
```python
COLOR_ORDER = [WHITE, BLUE, GREEN, RED, BLACK, GOLD]
# Indices:      0      1      2      3      4      5
```

### NaN Handling

- Missing players (2-3 player games): 49 NaN values per missing player slot
- Exhausted card slots: 12 NaN values per missing card
- Missing nobles: 6 NaN values per missing noble
- Inapplicable actions: NaN for output heads not relevant to current action

## Testing

### Unit Tests

The implementation includes validation for:
- Card encoding (12 features)
- Noble encoding (6 features)
- Player rotation algorithm
- Card position calculations
- Gem take encodings (2-hot and 3-hot)
- Input vector length (381)
- Output completeness (7 heads)

### Integration Tests

Run with limited samples to verify export works:

```bash
# Export 100 samples
python scripts/export_ml_dataset.py --limit 100 --output data/test_export.csv

# Validate structure
python -c "import pandas as pd; df = pd.read_csv('data/test_export.csv'); \
  assert len(df.columns) == 402, 'Expected 402 columns'; \
  assert df.shape[0] <= 100, 'Expected max 100 rows'; \
  print('✓ Test export validated')"
```

### Validation Commands

Complete validation workflow from the specification:

```bash
# 1. Test export with limited samples
python scripts/export_ml_dataset.py --limit 100 --output data/test_export.csv

# 2. Validate test export
python scripts/validate_dataset.py data/test_export.csv

# 3. Full export
python scripts/export_ml_dataset.py

# 4. Validate full dataset
python scripts/validate_dataset.py data/training_dataset.csv

# 5. Verify sample count matches database
python -c "import sqlite3; import pandas as pd; \
  conn = sqlite3.connect('data/games.db'); \
  db_count = conn.execute('SELECT COUNT(*) FROM Action').fetchone()[0]; \
  df = pd.read_csv('data/training_dataset.csv'); \
  print(f'DB: {db_count}, CSV: {len(df)}'); \
  assert db_count == len(df)"
```

## Notes

### Performance

- Processing ~1.7M samples takes several minutes
- In-memory card/noble caching significantly improves speed
- Progress updates every 100 rows to minimize terminal I/O
- CSV file size: ~5-10 GB (text format)

### Edge Cases Handled

- **Variable Player Counts**: Supports 2, 3, and 4 player games with NaN padding
- **Exhausted Decks**: Missing visible cards encoded as NaN
- **Reserved Card Tracking**: Complex state reconstruction from action history
- **Top Deck Reservations**: Distinguishes visible vs. top-deck reserves by checking card visibility
- **Build from Reserve**: Identifies builds from reserve (encoded as positions 12-14)
- **Noble Selection**: Handles actions with and without noble acquisition
- **Gem Overflow**: Tracks gem removal when player exceeds 10 token limit

### Train/Val/Test Splitting

The dataset includes `game_id` to enable proper splitting without data leakage:

```python
# Group by game_id to prevent same game in train and test
unique_games = df['game_id'].unique()
train_games, temp = train_test_split(unique_games, test_size=0.3)
val_games, test_games = train_test_split(temp, test_size=0.5)

train_df = df[df['game_id'].isin(train_games)]  # 70%
val_df = df[df['game_id'].isin(val_games)]      # 15%
test_df = df[df['game_id'].isin(test_games)]    # 15%
```

### Multi-Task Learning

The 7 output heads require task-specific loss masking during training:
- **build** actions: Use card_selection, possibly noble_selection and gems_removed
- **reserve** actions: Use card_reservation
- **take tokens** actions: Use gem_take_2 or gem_take_3, possibly gems_removed

Implement masking in training loop:
```python
# Pseudocode
if action_type == "build":
    loss = loss_card_selection + loss_noble_selection + loss_gems_removed
elif action_type == "reserve":
    loss = loss_card_reservation
# ... etc
```

### Future Enhancements

Potential improvements identified in the specification:
- Export to HDF5 or Parquet for better compression and faster loading
- Data augmentation through player permutations
- Stratified sampling to balance action type distribution
- Feature scaling/normalization utilities
- Visualization of feature distributions
- Separate train/val/test file generation with proper game-based grouping
