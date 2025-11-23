# Board State Reconstruction from CSV

**ADW ID:** N/A
**Date:** 2025-11-23
**Specification:** N/A

## Overview

This feature enables reconstruction of complete Splendor game board states from CSV row data. The reconstructed board objects are functionally equivalent to the original game state, allowing the codebase to call `board.getMoves()` to generate legal moves for action masking during training and evaluation.

The reconstruction module reverses the encoding performed by `csv_exporter.py`, converting 382 CSV features back into Python game objects (Board, Card, Character, Player).

## What Was Built

- **CSV-to-Board Reconstruction**: Core functionality to parse CSV rows and rebuild complete Board objects
- **Card Validation**: Validation that reconstructed cards exist in game deck constants (DECK1, DECK2, DECK3)
- **Variable-Length Card Lists**: Support for depleted decks with shrinking card lists (no None padding)
- **Level-Based Card Positioning**: Proper card placement using intrinsic card level rather than CSV position
- **Integration with Utils Module**: Main reconstruction function accessible from `utils.state_reconstruction`

## Technical Implementation

### Files Modified

- `src/utils/state_reconstruction.py`: Enhanced `parse_card_features()` to validate cards against deck constants; improved `reconstruct_board_from_csv_row()` to handle variable-length displayed card lists using intrinsic card levels
- `src/splendor/board.py`: Minor fix to ensure compatibility with reconstruction (line 2)
- `src/splendor/csv_exporter.py`: Updated to align with reconstruction expectations (4 lines changed)
- `src/utils/validate_reconstruction.py`: Enhanced validation utilities to work with new reconstruction logic
- `docs/state_reconstruction_guide.md`: Updated documentation with new validation and card handling details

### Key Changes

1. **Card Validation Against Deck Constants**
   - Added validation in `parse_card_features()` to verify cards exist in DECK1/DECK2/DECK3 constants (src/utils/state_reconstruction.py:80-90)
   - Prevents invalid card reconstruction by checking [vp, bonus, cost, level] signature against deck data
   - Raises `ValueError` with detailed message if card doesn't exist in appropriate deck

2. **Variable-Length Displayed Card Lists**
   - Changed from fixed 4-cards-per-level with None padding to variable-length lists (src/utils/state_reconstruction.py:257-277)
   - Only appends non-None cards to appropriate level list
   - Uses card's intrinsic level (`card.lvl - 1`) rather than calculating from CSV position
   - Matches core Splendor engine behavior where depleted decks have shrinking lists

3. **Robust Card-to-Index Mapping**
   - Reconstruction now properly handles boards where decks are partially depleted
   - Ensures `board.displayedCards[level]` contains only actual cards, no placeholders
   - Critical for accurate legal move generation in masking pipeline

## How to Use

### Basic Reconstruction

```python
from utils.state_reconstruction import reconstruct_board_from_csv_row
import pandas as pd

# Load game CSV data
df = pd.read_csv('data/games/3_games/5444.csv')

# Reconstruct board from first row
row_dict = df.iloc[0].to_dict()
board = reconstruct_board_from_csv_row(row_dict)

# Now you can use the board object
legal_moves = board.getMoves()
print(f"Found {len(legal_moves)} legal moves")
```

### Integration with Preprocessing Pipeline

```python
# In data_preprocessing.py or similar
from utils.state_reconstruction import reconstruct_board_from_csv_row

def generate_masks_for_row(row_dict):
    # Step 1: Reconstruct board state
    board = reconstruct_board_from_csv_row(row_dict)

    # Step 2: Get legal moves
    moves = board.getMoves()

    # Step 3: Convert moves to masks
    # (see illegal action masking feature)
    ...
```

### Validation Example

```python
from utils.validate_reconstruction import validate_round_trip

# Validate that reconstruction preserves all features
features = [...] # 382 features from dataset
board = reconstruct_board_from_features(features)
is_valid = validate_round_trip(features, board, turn_num=5)

if is_valid:
    print("Reconstruction successful!")
```

## Configuration

No special configuration required. The reconstruction uses the same deck constants (DECK1, DECK2, DECK3) defined in `src/splendor/constants.py`.

## Testing

### Manual Testing

```bash
# Test single file reconstruction
python scripts/test_reconstruction.py data/games/3_games/5444.csv

# Batch validate multiple games
python scripts/batch_validate_reconstruction.py data/games/3_games/
```

### Automated Validation

The preprocessing pipeline includes built-in validation:
- Checks that all CSV features successfully reconstruct to valid boards
- Verifies that reconstructed boards can generate legal moves
- Reports any reconstruction failures with game_id and turn_number

## Notes

### Important Considerations

1. **NaN Handling**: CSV rows must preserve original NaN values for nobles, reserved cards, etc. Filling NaN with 0 before reconstruction will add phantom game elements and corrupt the board state.

2. **Card Identity**: Reconstruction uses Python object identity for card matching. The same card object must be used consistently across `displayedCards`, player reserved lists, etc.

3. **Deck Depletion**: When decks are depleted, `displayedCards[level]` lists will have fewer than 4 cards. This is expected behavior and matches the core game engine.

4. **Performance**: Reconstruction is relatively expensive (~100-200ms per row). For large datasets (1.7M samples), expect 20-40 minutes for full mask generation.

### Limitations

- Reconstruction creates "dummy" cards for remaining deck counts (not full deck state)
- Some internal game state (like action history) is not preserved
- Round-trip encoding may have minor floating-point precision differences

### Future Considerations

- Consider caching reconstructed boards if processing the same state multiple times
- Could optimize by avoiding reconstruction when legal moves are not needed
- May want to add reconstruction statistics (success rate, timing) to preprocessing logs
