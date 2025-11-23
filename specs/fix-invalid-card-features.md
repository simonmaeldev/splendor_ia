# Bug: Invalid Card Features During Feature Extraction

## Bug Description
During data preprocessing, the feature extraction step fails with thousands of warnings:
```
WARNING: Feature extraction failed for game_id=1270, turn=31: Invalid card features: vp=0, level=0, bonus=0, cost=[0, 0, 0, 0, 0]. Card does not exist in DECK0 constants.
```

These warnings indicate that `extract_all_features()` is attempting to reconstruct board states with invalid card data (all zeros), which causes `parse_card_features()` in `state_reconstruction.py` to fail validation against the deck constants.

**Expected behavior**: Feature extraction should successfully extract strategic features for all game states, or properly handle missing/invalid cards without attempting reconstruction.

**Actual behavior**: Feature extraction fails for thousands of samples, returning empty feature dictionaries that get filled with zeros, potentially degrading model training quality.

## Problem Statement
After the `compact_cards_and_add_position()` function rearranges cards and the `fillna(0)` operation fills NaN values, cards that don't exist (were originally NaN) now appear as cards with all-zero features including `level=0`. When `extract_all_features()` attempts to reconstruct the board state using `reconstruct_board_from_csv_row()`, these zero-value cards fail validation in `parse_card_features()` because:
1. Cards in Splendor have levels 1, 2, or 3 (never 0)
2. The card signature `[vp=0, bonus=0, cost=[0,0,0,0,0], level=0]` doesn't exist in any deck constant (DECK1, DECK2, DECK3)

## Solution Statement
Modify the card parsing logic to treat cards with `level=0` as missing cards (None) rather than attempting to validate them against deck constants. This approach:
1. Is minimally invasive - only changes the validation logic
2. Maintains backward compatibility - existing valid cards (level 1-3) are unaffected
3. Aligns with the semantic meaning - level=0 explicitly marks non-existent cards after fillna
4. Fixes the immediate issue without requiring changes to the data pipeline

## Steps to Reproduce
1. Run data preprocessing: `uv run python -m src.imitation_learning.data_preprocessing --config src/imitation_learning/configs/config_small.yaml`
2. Observe thousands of warnings during "Extracting strategic features" step
3. Check that warnings have pattern: `vp=0, level=0, bonus=0, cost=[0, 0, 0, 0, 0]`

## Root Cause Analysis
The root cause is in the data preprocessing pipeline:

1. **Original CSV data**: Cards that don't exist are represented as NaN
2. **Mask generation** (line 1082): Requires NaN values to correctly reconstruct game state
3. **fillna(0)** (line 1085): Converts NaN to 0 for feature engineering
4. **Card compaction** (line 1089): Rearranges cards but preserves zero-value cards
5. **Feature engineering** (line 1095): For each row, calls `extract_all_features()`
   - This calls `reconstruct_board_from_csv_row()` which calls `parse_card_features()`
   - `parse_card_features()` sees cards with `level=0` and tries to validate against DECK0
   - Validation fails because level=0 is invalid and DECK0 doesn't exist
   - Exception is caught, warning is logged, empty features returned

The fundamental issue: after `fillna(0)`, there's no way to distinguish a non-existent card from a card with level=0 in the reconstruction logic.

## Relevant Files
Files needed to fix this bug:

- **src/utils/state_reconstruction.py** (lines 40-98): Contains `parse_card_features()` function that validates card signatures against deck constants. Needs modification to treat level=0 as "missing card".
  - Currently throws ValueError for invalid cards at line 87-90
  - Should return None for level=0 cards before validation

## Step by Step Tasks

### 1. Modify card parsing to handle level=0 as missing cards
- Update `parse_card_features()` in `src/utils/state_reconstruction.py`
- Add early return for level=0 cards (after NaN check, before validation)
- This treats level=0 as semantically equivalent to "no card exists here"

### 2. Run data preprocessing and verify fix
- Execute preprocessing with config: `uv run python -m src.imitation_learning.data_preprocessing --config src/imitation_learning/configs/config_small.yaml`
- Verify no warnings about "Invalid card features" with level=0
- Check that feature extraction completes successfully

### 3. Run unit tests
- Execute existing tests to ensure no regressions: `uv run pytest tests/test_feature_engineering.py -v`

## Validation Commands
Execute every command to validate the bug is fixed with zero regressions.

```bash
# 1. Run data preprocessing and check for warnings
uv run python -m src.imitation_learning.data_preprocessing --config src/imitation_learning/configs/config_small.yaml 2>&1 | grep "Invalid card features"

# Expected: No output (no warnings about invalid card features with level=0)

# 2. Run preprocessing and verify completion
uv run python -m src.imitation_learning.data_preprocessing --config src/imitation_learning/configs/config_small.yaml

# Expected: Completes successfully with "✓ Preprocessing complete!"

# 3. Run unit tests for feature engineering
uv run pytest tests/test_feature_engineering.py -v

# Expected: All tests pass
```

## Notes
- The fix is surgical and minimal: only adds 3-4 lines to check for level=0 before validation
- No changes needed to data pipeline, preprocessing order, or feature extraction logic
- Cards with level 1-3 continue to be validated against deck constants as before
- This maintains the semantic meaning: level=0 = "no card", just like NaN originally meant
- The warning message in `extract_all_features()` will still catch other types of failures (e.g., corrupted data with invalid level 1-3 cards)
