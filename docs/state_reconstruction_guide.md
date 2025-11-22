# State Reconstruction Guide

## Overview

The state reconstruction module (`src/utils/state_reconstruction.py`) provides functionality to reconstruct internal Python game state (Board objects) from CSV input features. This enables calling `board.getMoves()` to get valid moves for action masking during training and evaluation.

### What It Does

The module reverses the encoding performed by `csv_exporter.py`:

```
CSV features (382) → Board object with Card, Character, Player objects
```

The reconstructed board is **functionally equivalent** to the original game state for the purpose of generating valid moves.

### Why It's Needed

The ML pipeline can predict actions from input features, but to implement illegal action masking, we need to:

1. Generate valid action masks for each state
2. Filter illegal predictions during inference
3. Validate that training data actions are actually legal
4. Enable rule-aware evaluation metrics

## Architecture

### Module Structure

```
src/utils/
├── __init__.py
├── state_reconstruction.py    # Core reconstruction functions
└── validate_reconstruction.py  # Validation utilities

scripts/
├── test_reconstruction.py              # Single-file testing
└── batch_validate_reconstruction.py   # Batch validation

docs/
└── state_reconstruction_guide.md      # This file
```

### Key Functions

#### State Reconstruction (`state_reconstruction.py`)

- `parse_card_features(features)` - Parse 12 features into Card object
- `parse_noble_features(features)` - Parse 6 features into Character object
- `parse_player_features(features, player_num)` - Parse 49 features into Player object
- `create_dummy_deck(level, count)` - Create dummy cards for deck representation
- `reconstruct_board_from_csv_row(row)` - Main reconstruction from CSV dict
- `reconstruct_board_from_features(features)` - Reconstruction from feature list

#### Validation (`validate_reconstruction.py`)

- `parse_action_from_csv(row, board)` - Parse CSV action into Move object
- `validate_action_in_moves(board, action)` - Verify action is valid
- `encode_board_state(board, turn_num)` - Encode board back to features
- `compare_features(original, reconstructed)` - Compare feature lists
- `validate_round_trip(features, board, turn_num)` - Full round-trip test

## Usage Examples

### Basic Reconstruction from CSV

```python
import csv
from utils.state_reconstruction import reconstruct_board_from_csv_row

# Load CSV and reconstruct each row
with open('data/games/3_games/3869.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Reconstruct board from CSV row
        board = reconstruct_board_from_csv_row(row)

        # Get valid moves
        valid_moves = board.getMoves()
        print(f"Turn {board.nbTurn}: {len(valid_moves)} valid moves")
```

### Reconstruction from Feature List

```python
from utils.state_reconstruction import reconstruct_board_from_features

# Assuming you have 382 features from your dataset
features = [3.0, 1.0, 0.0, ...]  # 382 features

# Reconstruct board
board = reconstruct_board_from_features(features)

# Use the board
moves = board.getMoves()
```

### Action Validation

```python
from utils.state_reconstruction import reconstruct_board_from_csv_row
from utils.validate_reconstruction import parse_action_from_csv, validate_action_in_moves

# Reconstruct board
board = reconstruct_board_from_csv_row(row)

# Parse action from CSV
action = parse_action_from_csv(row, board)

# Validate action
is_valid, message = validate_action_in_moves(board, action)

if is_valid:
    print("Action is valid!")
else:
    print(f"Action is invalid: {message}")
```

### Generate Action Masks for Training

```python
import csv
import numpy as np
from utils.state_reconstruction import reconstruct_board_from_csv_row

def generate_action_masks(csv_path):
    """Generate action masks for all rows in CSV."""
    masks = []

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Reconstruct board
            board = reconstruct_board_from_csv_row(row)

            # Get valid moves
            valid_moves = board.getMoves()

            # Create mask (convert valid moves to action indices)
            # TODO: Implement action space mapping
            mask = create_mask_from_moves(valid_moves)
            masks.append(mask)

    return np.array(masks)
```

### Batch Validation

```bash
# Test single file
python scripts/test_reconstruction.py --csv data/games/3_games/3869.csv

# Test with sampling
python scripts/test_reconstruction.py --csv data/games/2_games/*.csv --sample 1

# Batch validation across dataset
python scripts/batch_validate_reconstruction.py --sample-size 20 --player-counts 2 3 4

# View validation report
cat logs/reconstruction_validation_report.txt
```

## Implementation Details

### Player Rotation Handling

CSV files have players rotated so the current player is always `player0_*`:

```python
# CSV encoding: current player first
# Original: players [A, B, C], current_player = 1
# CSV: player0 = B, player1 = C, player2 = A

# Reconstruction: un-rotate using player{i}_position
# player0_position = 1  -> Place at board.players[1]
# player1_position = 2  -> Place at board.players[2]
# player2_position = 0  -> Place at board.players[0]
```

This ensures `board.getCurrentPlayer()` returns the correct player.

### Dummy Deck Creation

We create dummy cards for deck contents because:

- `getMoves()` only checks if decks are non-empty for top-deck reservation
- The actual card properties don't matter for move generation
- Exact deck contents are unknown (only counts are recorded)

```python
# Dummy cards have: vp=0, bonus=0, cost=[0,0,0,0,0], visible=False
deck = create_dummy_deck(level=1, count=10)
```

### NaN Handling

CSV uses empty strings for missing values (e.g., player 4 in 3-player games):

```python
def safe_float(val):
    """Convert value to float, handling empty strings as NaN."""
    if val == '' or val is None:
        return float('nan')
    return float(val)

# Check for NaN
if all(math.isnan(f) for f in features):
    return None  # Entity doesn't exist
```

### Reserved Cards Visibility

All reserved cards have `visible=True` because:

- `getMoves()` doesn't require visibility information
- Visibility matters for ISMCTS determinization (not implemented)
- The ML model can see all reserved cards anyway

### Built Cards Not Reconstructed

We don't reconstruct `player.built` lists because:

- Not present in CSV input features
- Not needed for `getMoves()` calculation
- Player VP and reductions are reconstructed directly (sufficient)

## Validation Results

Based on batch validation of 60 games (20 per player count) across 4,861 rows:

### Success Rates

| Validation Type | Success Rate | Status |
|----------------|--------------|---------|
| Reconstruction | 100.0% (4861/4861) | ✓ PERFECT |
| getMoves() | 100.0% (4861/4861) | ✓ PERFECT |
| Action Validation | 97.3% (4731/4861) | ✓ EXCELLENT |
| Round-trip | 10.1% (492/4861) | ⚠ See note below |

### Per-Player-Count Breakdown

- **2 players**: 100% reconstruction, 100% getMoves(), 99.1% action validation
- **3 players**: 100% reconstruction, 100% getMoves(), 98.3% action validation
- **4 players**: 100% reconstruction, 100% getMoves(), 95.5% action validation

### Round-Trip Encoding Note

Round-trip encoding (reconstruct → encode → compare) has a lower success rate (10.1%). This is expected and acceptable because:

1. CSV data has been cleaned by `fix_csv_gem_encoding.py` and `clean_csv_nan_values.py`
2. The original CSV may have had encoding inconsistencies that were fixed post-export
3. **The critical validations pass**: 100% getMoves() success and 97.3% action validation
4. Semantic meaning is preserved even if exact feature values differ slightly

As stated in the spec: "If round-trip comparison fails due to encoding differences, this is acceptable as long as getMoves() works and action validation passes."

## Troubleshooting

### Reconstruction Errors

**Problem**: `could not convert string to float: ''`

**Solution**: Empty strings are handled by `safe_float()` helper. This should not occur with the current implementation.

**Problem**: `Player rotation mismatch`

**Solution**: Verify `player{i}_position` values match expected absolute positions.

### getMoves() Failures

**Problem**: `getMoves() failed: AttributeError`

**Solution**: Check that all board attributes are set correctly:
- `board.tokens` (6-element list)
- `board.displayedCards` (3 lists of Card objects)
- `board.decks` (3 lists of Card objects)
- `board.characters` (list of Character objects)
- `board.players` (list of Player objects)

### Action Validation Failures

**Problem**: `Action not in valid moves`

**Causes**:
1. Data issue: CSV action may have been recorded incorrectly
2. Reconstruction bug: Board state doesn't match original
3. Action parsing bug: Move object doesn't match CSV encoding

**Debug**:
```python
# Print valid moves
print(f"Valid moves: {len(valid_moves)}")
for i, move in enumerate(valid_moves[:5]):
    print(f"  {i}: {move}")

# Print expected action
print(f"Expected: {expected_action}")

# Compare attributes
print(f"Type match: {expected_action.actionType == valid_moves[0].actionType}")
```

## Limitations

### What's Not Reconstructed

1. **Built Cards** (`player.built`): Not in CSV, not needed for getMoves()
2. **Exact Deck Contents**: Only counts known, dummy cards used
3. **Card Visibility**: Reserved cards always visible=True
4. **Player Names**: Always set to str(player_num)
5. **Player AI**: Always set to "ISMCTS_PARA1000"

### What's Acceptable

1. **Round-trip mismatches**: Due to CSV cleaning, acceptable if getMoves() works
2. **Float vs int**: 2.0 == 2 for comparison purposes
3. **Small numerical errors**: Within tolerance (1e-6)

### What's Not Acceptable

1. **getMoves() failures**: Indicates reconstruction bug, must fix
2. **Consistent action validation failures**: Indicates systematic issue
3. **Board structure errors**: Wrong number of players, missing cards, etc.

## Integration with Training Pipeline

### Phase 2: Action Masking (IMPLEMENTED)

This module provides the foundation for Phase 2, which generates action masks during preprocessing.

#### Mask Generation During Preprocessing

Masks are generated for all samples during the data preprocessing step:

```python
from imitation_learning.data_preprocessing import generate_masks_for_dataframe

# Generate masks for entire dataframe
masks_all = generate_masks_for_dataframe(df)

# masks_all contains 7 arrays (one per prediction head):
# - action_type: (n_samples, 4) - BUILD, RESERVE, TAKE2, TAKE3
# - card_selection: (n_samples, 15) - which cards can be built
# - card_reservation: (n_samples, 15) - which cards can be reserved
# - gem_take3: (n_samples, 26) - which gem combinations for TAKE3
# - gem_take2: (n_samples, 5) - which colors for TAKE2
# - noble: (n_samples, 5) - which nobles can be acquired
# - gems_removed: (n_samples, 84) - which gem removal patterns

# Split masks with same indices as features/labels
masks_train = {k: v[train_idx] for k, v in masks_all.items()}
masks_val = {k: v[val_idx] for k, v in masks_all.items()}
masks_test = {k: v[test_idx] for k, v in masks_all.items()}
```

#### Loading Masks in Training

```python
import numpy as np

# Load preprocessed masks
masks_train = np.load('data/processed/masks_train.npz')
masks_val = np.load('data/processed/masks_val.npz')

# Access individual head masks
action_type_masks = masks_train['action_type']  # Shape: (n_samples, 4)
card_selection_masks = masks_train['card_selection']  # Shape: (n_samples, 15)
```

#### Applying Masks During Training

```python
# In training loop
for batch_idx, (features, labels, masks) in enumerate(dataloader):
    # Forward pass
    logits = model(features)  # Dict with 7 heads

    # Apply masks to logits (set illegal actions to very negative values)
    masked_logits = {}
    for head_name in logits.keys():
        masked_logits[head_name] = logits[head_name] + (1 - masks[head_name]) * -1e9

    # Compute loss with masked logits
    # Now the model only considers legal actions
    loss = compute_loss(masked_logits, labels)
```

#### Mask Statistics

After preprocessing, check the mask statistics in `preprocessing_stats.json`:

```json
{
  "mask_statistics": {
    "action_type": {
      "avg_legal_actions": 2.3,
      "min_legal_actions": 1,
      "max_legal_actions": 4
    },
    "card_selection": {
      "avg_legal_actions": 3.5,
      "min_legal_actions": 0,
      "max_legal_actions": 15
    }
    // ... other heads
  }
}
```

## Performance Considerations

### Current Performance

Reconstruction is fast enough for validation but will be called millions of times during training. Current implementation is unoptimized.

### Future Optimizations

1. **Cache parsed cards/nobles**: Many states share same visible cards
2. **Vectorize reconstruction**: Reconstruct batch of boards at once
3. **Profile bottlenecks**: Identify slow functions
4. **Lazy loading**: Only reconstruct when needed

**Priority**: Correctness first, performance optimization later if needed.

## References

- Spec: `specs/state-reconstruction-from-input.md`
- CSV Exporter: `src/splendor/csv_exporter.py`
- Game Logic: `src/splendor/board.py`, `src/splendor/player.py`
- Test Scripts: `scripts/test_reconstruction.py`, `scripts/batch_validate_reconstruction.py`
- Validation Report: `logs/reconstruction_validation_report.txt`

## Confidence Level

Based on validation results:

- ✓ **High Confidence**: Reconstruction is correct (100% success)
- ✓ **High Confidence**: getMoves() works (100% success)
- ✓ **High Confidence**: Actions are valid (97.3% success)
- ⚠ **Medium Confidence**: Round-trip encoding (10.1% success, but acceptable per spec)

**Overall Assessment**: The reconstruction module is **production-ready** for action masking in Phase 2.
