# Bug: AI Player Feature Dimension Mismatch

## Bug Description
When using the ModelPlayer in `ai_player.py`, there is a dimension mismatch between the features extracted during inference and the features expected by the trained model. The error shows:

```
[ModelPlayer] Expected 893 engineered features
[ModelPlayer] Model input dim: 1308
WARNING: Feature tensor shape torch.Size([1, 893]) doesn't match model input_dim 1308
```

The model expects 1308 features (CSV features + engineered features), but the `get_model_input_features` method only provides 893 features (engineered features only). This occurs because:

1. During **training** (in `merge_batches.py` line 369): The input features are created by concatenating `df_compacted` (original CSV features ~415 columns) + `strategic_df` (engineered features 893 columns) = ~1308 total features
2. During **inference** (in `ai_player.py` line 417): Only the engineered features (893) are extracted via `extract_all_features()`, missing the original CSV features

## Problem Statement
The `ModelPlayer.get_model_input_features()` method needs to produce the same feature tensor structure that was used during training: original CSV features concatenated with engineered strategic features.

## Solution Statement
Modify `ai_player.py` to concatenate the CSV row features with the engineered strategic features, matching the exact same feature engineering pipeline used during training in `merge_batches.py`. The solution must:

1. Extract CSV features from the board state (via `board_to_csv_row`)
2. Apply the same transformations done in training: fill NaN values, compact cards, add position indices, create one-hot encodings
3. Extract engineered features (via `extract_all_features`)
4. Concatenate both feature sets in the correct order to match training data

## Steps to Reproduce
1. Train a model using the current data preprocessing pipeline (which creates 1308-dimensional inputs)
2. Save the model checkpoint to `data/models/*/best_model.pth`
3. Run inference using `ModelPlayer`:
   ```python
   from src.splendor.ai_player import ModelPlayer
   from src.splendor.board import Board

   player = ModelPlayer("data/models/202511241159_config_tuning/best_model.pth")
   board = Board()  # or load a game state
   move = player.get_action(board)
   ```
4. Observe the warning: `WARNING: Feature tensor shape torch.Size([1, 893]) doesn't match model input_dim 1308`

## Root Cause Analysis
The root cause is in `src/splendor/ai_player.py` lines 380-426 in the `get_model_input_features()` method:

**Current implementation (line 417):**
```python
feature_dict = extract_all_features(csv_row, board)
```

This only extracts the 893 engineered features, ignoring the CSV features.

**Training pipeline (src/imitation_learning/merge_batches.py lines 363-372):**
```python
# Create strategic DataFrame from array
strategic_df = pd.DataFrame(
    strategic_features_array,
    columns=expected_feature_names
)

# Add strategic features to dataframe
df_eng = pd.concat(
    [df_compacted.reset_index(drop=True), strategic_df.reset_index(drop=True)],
    axis=1,
)
```

The training pipeline concatenates:
1. `df_compacted`: CSV features after NaN filling, card compaction, and position indexing (~415 features)
2. `strategic_df`: Engineered features from `extract_all_features` (893 features)
3. Plus one-hot encodings for `current_player`, `num_players`, and player positions (~20 features)
4. **Total: ~1308 features**

The inference code only provides the 893 engineered features, missing ~415 CSV features and one-hot encodings.

## Relevant Files
Use these files to fix the bug:

- **`src/splendor/ai_player.py`** (lines 380-440)
  - Contains `get_model_input_features()` that needs fixing
  - Currently only extracts engineered features (893)
  - Needs to replicate the full training feature pipeline

- **`src/imitation_learning/merge_batches.py`** (lines 336-438)
  - Contains `process_merged_data()` showing the complete feature engineering pipeline used during training
  - Shows how CSV features, one-hot encodings, and strategic features are combined
  - Line 369: The key concatenation of df_compacted + strategic_df

- **`src/imitation_learning/parallel_processor.py`** (lines 206-313)
  - Contains `fill_nan_values_for_row()` and `compact_cards_and_add_position_for_row()`
  - These transformations are applied to CSV features before training
  - Must be replicated during inference

- **`src/imitation_learning/data_preprocessing.py`** (if exists)
  - May contain helper functions for identifying column groups and creating feature lists
  - Use `identify_column_groups()` to separate metadata/labels from features

## Step by Step Tasks

### Step 1: Read and understand the complete training feature pipeline
- Read `src/imitation_learning/merge_batches.py` lines 336-438 to understand the exact sequence of transformations
- Read `src/imitation_learning/parallel_processor.py` lines 206-313 for NaN filling and card compaction logic
- Identify all feature transformations: NaN filling, card compaction, position indices, one-hot encodings
- Document the exact order of features in the final training tensor

### Step 2: Create a helper function to replicate training preprocessing
- In `src/splendor/ai_player.py`, create a new function `preprocess_csv_row()` that:
  - Takes a CSV row dictionary as input
  - Applies `fill_nan_values_for_row()` (import from parallel_processor)
  - Applies `compact_cards_and_add_position_for_row()` (import from parallel_processor)
  - Returns the preprocessed row dictionary
- Add necessary imports from `imitation_learning.parallel_processor`

### Step 3: Create a helper function to generate one-hot features
- In `src/splendor/ai_player.py`, create a new function `create_onehot_features()` that:
  - Takes a preprocessed row dictionary as input
  - Creates one-hot encodings for:
    - `current_player` (4 values: 0, 1, 2, 3)
    - `num_players` (3 values: 2, 3, 4)
    - `player{i}_position` for i in 0-3 (4 values: 0, 1, 2, 3)
  - Returns a dictionary of one-hot feature values
- This replicates the logic in `merge_batches.py` lines 387-406

### Step 4: Extract and identify CSV feature columns
- In `src/splendor/ai_player.py`, create a helper function `get_csv_feature_columns()` that:
  - Returns an ordered list of CSV feature column names
  - Excludes metadata columns: `game_id`, `turn_number`, `action_type`
  - Excludes label columns: `card_selection`, `card_reservation`, `noble_selection`, gem/noble labels
  - Excludes position index columns that get replaced: `current_player`, `num_players`, `player{i}_position`
  - Includes all game state features: tokens, cards, players, nobles, deck counts
- Use the same logic as `identify_column_groups()` from data_preprocessing if available

### Step 5: Rewrite get_model_input_features to match training pipeline
- Modify `ModelPlayer.get_model_input_features()` to:
  1. Convert board to CSV row using `board_to_csv_row()`
  2. Preprocess the CSV row: `preprocessed_row = preprocess_csv_row(csv_row)`
  3. Extract CSV feature values in correct order using `get_csv_feature_columns()`
  4. Create one-hot features: `onehot_features = create_onehot_features(preprocessed_row)`
  5. Extract engineered features: `strategic_features = extract_all_features(preprocessed_row, board)`
  6. Concatenate all features in order: CSV features + one-hot features + strategic features
  7. Convert to tensor with shape (1, input_dim)
- Ensure the feature order exactly matches the training pipeline in `merge_batches.py`

### Step 6: Add validation and error checking
- Add validation to ensure the feature tensor shape matches `model_input_dim`
- Add helpful error messages if dimensions mismatch
- Print debug information showing:
  - Number of CSV features
  - Number of one-hot features
  - Number of strategic features
  - Total features vs expected model input dim
- If dimensions still mismatch, print which features might be missing/extra

### Step 7: Update model loading logic for backward compatibility
- In `load_model()` function (lines 40-137), ensure that models trained with different feature sets are handled
- The existing logic at line 401 (`if model_input_dim < 500`) tries to detect old vs new format
- Update this logic to be more robust and clearly documented
- Consider adding a flag in the model checkpoint to indicate which features were used

### Step 8: Run validation tests
- Run the `Validation Commands` below to verify the fix works correctly
- Test with the existing model at `data/models/202511241159_config_tuning/best_model.pth`
- Verify no dimension mismatch warning appears
- Verify the model can successfully predict actions

## Validation Commands
Execute every command to validate the bug is fixed with zero regressions.

- `uv run python -c "from src.imitation_learning.feature_engineering import get_all_feature_names; print(f'Engineered features: {len(get_all_feature_names())}')"` - Verify engineered feature count is 893

- `uv run python -c "from src.splendor.ai_player import board_to_csv_row, Board; row = board_to_csv_row(Board()); print(f'CSV columns: {len(row)}')"` - Verify CSV row feature count

- `uv run python -c "from src.splendor.ai_player import ModelPlayer; import torch; player = ModelPlayer('data/models/202511241159_config_tuning/best_model.pth'); print(f'Model input dim: {player.config[\"input_dim\"]}')"` - Verify model loads and check expected input dimension

- `uv run python scripts/test_model_player.py` - Run the test script to verify ModelPlayer works without dimension mismatch warnings

- `uv run python scripts/play_with_model.py` - Run a full game with the model to ensure no runtime errors

- `grep -n "WARNING: Feature tensor shape" <output_file>` - After running tests, verify no dimension mismatch warnings appear (should return empty)

## Notes

### Critical Implementation Details
1. **Feature Order Matters**: The exact order of features in the tensor must match training. The order is:
   - CSV features (after preprocessing)
   - One-hot encodings (current_player, num_players, player positions)
   - Turn number
   - Strategic/engineered features (893)

2. **NaN Handling**: Some CSV features contain NaN for missing players/cards. During training, these are filled with 0 for features but preserved for labels. Replicate this exact behavior.

3. **Card Compaction**: During training, visible cards are reordered (non-zero first, zeros last) and position indices are added. This must be replicated exactly during inference.

4. **Column Naming**: The CSV column names must match exactly what's in the training data. Use the same column ordering as `board_to_csv_row()` produces.

### Testing Strategy
1. First verify the fix with the existing trained model
2. Then train a new model from scratch to ensure end-to-end compatibility
3. Compare predictions before/after the fix to ensure model behavior is preserved
4. Test edge cases: 2-player games, 3-player games, 4-player games

### Backward Compatibility
The fix should maintain backward compatibility with models trained using only CSV features (input_dim < 500). The existing check at line 401 should be preserved and enhanced.

### Performance Considerations
The feature extraction happens on every inference call. Keep the implementation efficient:
- Avoid unnecessary DataFrame operations
- Cache column name lists if possible
- Use numpy operations instead of Python loops where possible
