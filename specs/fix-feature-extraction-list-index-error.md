# Bug: Feature Extraction List Index Out of Range Error

## Bug Description
During data preprocessing for imitation learning, the strategic feature extraction step fails with thousands of "list index out of range" warnings:
```
WARNING: Feature extraction failed for game_id=968, turn=16: list index out of range
WARNING: Feature extraction failed for game_id=686, turn=1: list index out of range
```

These warnings occur when `extract_all_features()` attempts to extract player-specific features from games with fewer than 4 players (2 or 3-player games). The feature extraction code incorrectly assumes all games have 4 players and attempts to access `board.players[player_idx]` for indices 0-3, causing index out of range errors when there are only 2 or 3 players.

**Expected behavior**: Feature extraction should successfully extract strategic features for all game states, respecting the actual number of players in each game.

**Actual behavior**: Feature extraction fails for all 2 and 3-player games when trying to access non-existent player indices, returning empty feature dictionaries that get filled with zeros, degrading model training quality.

## Problem Statement
The feature extraction functions in `feature_engineering.py` hardcode loops over 4 players:
```python
for player_idx in range(4):
    player = board.players[player_idx]  # IndexError when player_idx >= len(board.players)
```

In Splendor, games can have 2, 3, or 4 players. The `board.players` list only contains the actual number of players (e.g., 2 players for a 2-player game), not always 4. When the code tries to access player indices beyond the actual player count, Python raises an `IndexError: list index out of range`.

## Solution Statement
Modify all feature extraction functions to:
1. Use the actual number of players from `len(board.players)` instead of hardcoding `range(4)`
2. Pad missing player features with zeros to maintain consistent feature dimensionality across all games
3. Ensure all player-indexed features (card, noble, comparison features) respect the actual player count

This approach:
1. Fixes the immediate index out of range errors
2. Maintains consistent feature dimensions (4 players × features) for the ML model
3. Is minimally invasive - only changes the loop ranges and adds padding logic
4. Preserves semantic meaning - missing players contribute zero features

## Steps to Reproduce
1. Run data preprocessing: `uv run python -m src.imitation_learning.data_preprocessing --config src/imitation_learning/configs/config_small.yaml`
2. Observe thousands of warnings during "Extracting strategic features" step
3. Check warnings have pattern: `list index out of range` for various game_ids and turns
4. Run debug script: `uv run python debug_feature_extraction.py`
5. Observe error occurs for 2-player games when accessing player indices 2 and 3

## Root Cause Analysis
The root cause is in `src/imitation_learning/feature_engineering.py`:

1. **extract_card_features()** (line 173): Loops `for player_idx in range(4)` but `board.players` may have only 2-3 elements
2. **extract_noble_features()** (line 280): Same issue - loops over range(4)
3. **extract_card_noble_synergy()** (line 347): Same issue - loops over range(4)
4. **extract_player_comparison_features()** (line 436): Same issue - loops over range(4)
5. **extract_game_progression_features()** (line 531): Same issue - loops over range(4)

All these functions assume 4 players exist, but `board.players` is constructed with actual player count from the CSV (2, 3, or 4 players).

## Relevant Files
Files needed to fix this bug:

- **src/imitation_learning/feature_engineering.py** (lines 153-536): Contains all feature extraction functions that assume 4 players. Need to modify:
  - `extract_card_features()` - lines 173-254
  - `extract_noble_features()` - lines 280-327
  - `extract_card_noble_synergy()` - lines 347-396
  - `extract_player_comparison_features()` - lines 436-493
  - `extract_game_progression_features()` - lines 531-534

## Step by Step Tasks

### 1. Fix extract_card_features() to respect actual player count
- Change `for player_idx in range(4):` to `for player_idx in range(len(board.players)):`
- Add padding loop to fill features for missing players (indices `len(board.players)` to 3) with zeros
- Ensure 540 features are always generated (4 players × 15 cards × 9 features)

### 2. Fix extract_noble_features() to respect actual player count
- Change `for player_idx in range(4):` to `for player_idx in range(len(board.players)):`
- Add padding loop to fill features for missing players with zeros
- Ensure 148 features are always generated (4 players × 37 features)

### 3. Fix extract_card_noble_synergy() to respect actual player count
- Change `for player_idx in range(4):` to `for player_idx in range(len(board.players)):`
- Add padding loop to fill features for missing players with zeros
- Ensure 120 features are always generated (4 players × 15 cards × 2 features)

### 4. Fix extract_player_comparison_features() to respect actual player count
- Change all `for player_idx in range(4):` loops to use actual player count
- Handle missing players in VP ranking and reduction ranking calculations
- Pad missing player features with zeros
- Ensure 50 features are always generated

### 5. Fix extract_game_progression_features() to respect actual player count
- Change `for player_idx in range(4):` to `for player_idx in range(len(board.players)):`
- Add padding loop for missing players
- Ensure 9 features are always generated (4 player features + 5 global features)

### 6. Test fix with debug script
- Run: `uv run python debug_feature_extraction.py`
- Verify no "list index out of range" errors
- Verify correct number of features extracted (893 total)
- Verify features are correctly padded with zeros for 2-player games

### 7. Run full data preprocessing pipeline
- Execute: `uv run python -m src.imitation_learning.data_preprocessing --config src/imitation_learning/configs/config_small.yaml`
- Verify no "list index out of range" warnings
- Check that preprocessing completes successfully with "✓ Preprocessing complete!"

## Validation Commands
Execute every command to validate the bug is fixed with zero regressions.

```bash
# 1. Run debug script to test specific failing game
uv run python debug_feature_extraction.py

# Expected: No "list index out of range" errors, 893 features extracted for each turn

# 2. Run data preprocessing and check for warnings
uv run python -m src.imitation_learning.data_preprocessing --config src/imitation_learning/configs/config_small.yaml 2>&1 | grep "list index out of range"

# Expected: No output (no warnings about list index errors)

# 3. Run preprocessing and verify completion
uv run python -m src.imitation_learning.data_preprocessing --config src/imitation_learning/configs/config_small.yaml

# Expected: Completes successfully with "✓ Preprocessing complete!"

# 4. Verify feature count consistency
uv run python -c "from imitation_learning.feature_engineering import get_all_feature_names; names = get_all_feature_names(); print(f'Total features: {len(names)}'); assert len(names) == 893, f'Expected 893 features, got {len(names)}'"

# Expected: "Total features: 893"
```

## Notes
- The fix maintains consistent feature dimensionality by padding missing players with zeros
- This is ML-friendly: missing players contribute zero signal, allowing the model to learn from variable player counts
- No changes needed to CSV format, data pipeline, or training code
- The 4-player assumption in `get_all_feature_names()` is correct - it defines the feature schema, not runtime validation
- Padding strategy: missing players get all-zero features, which is semantically correct (no player = no stats)
