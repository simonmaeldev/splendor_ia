# Chore: Compact Card Features with Position Index

## Chore Description
Modify the data preprocessing pipeline to compact card features by moving zero-padded (missing) cards to the end of the feature vector, and add a position index feature to each card indicating its position in the compacted list. This change applies only during X_*.npy creation, NOT to the CSV export.

**Current behavior:**
- Visible cards in CSV: `(lvl1, lvl1, zeros, zeros, lvl2, lvl2, lvl2, lvl2, lvl3, lvl3, lvl3, zeros)`
- After fillna: Same order with NaN→0
- 12 cards × 12 features = 144 features

**New behavior:**
- Visible cards in CSV: Still `(lvl1, lvl1, zeros, zeros, lvl2, ...)`  ← **NO CHANGE**
- After compaction in X_*.npy: `(lvl1, lvl1, lvl2, lvl2, lvl2, lvl2, lvl3, lvl3, lvl3, zeros, zeros, zeros)`
- Each card gets +1 new feature (position index): 12 cards × 13 features = 156 features
- Position index for visible cards: 0, 1, 2, ..., (up to 11)
- Position index for missing cards: -1
- Reserved cards per player: 3 cards × 13 features = 39 features each
- Reserved card position index: 12, 13, 14 (continues from visible cards)

**Scope:**
- Compact **visible cards only** (12 cards total across all 3 levels)
- Each player's **reserved cards stay in their own section** (may have zeros if <3 reserved)
- Each player's reserved cards independently use positions 12, 13, 14

## Relevant Files
Use these files to resolve the chore:

### Core Files to Modify

- **`src/imitation_learning/data_preprocessing.py`** - Main preprocessing pipeline
  - Contains `fill_nan_values()` which converts NaN to 0 (line 147-178)
  - Contains `engineer_features()` which adds one-hot encodings (line 236-296)
  - Contains `main()` which orchestrates the preprocessing pipeline (line 861-1002)
  - **Changes needed**: Add new function `compact_and_add_position_index()` to be called after `fill_nan_values()` but before feature engineering
  - Must handle visible cards (12 × 12 features → 12 × 13 features) and reserved cards per player (3 × 12 features → 3 × 13 features)

- **`src/imitation_learning/utils.py`** - Utility functions
  - **Changes needed**: No changes required (mask generation uses board.getMoves() which doesn't depend on feature order)

### Files to Review (No Changes Expected)

- **`src/splendor/csv_exporter.py`** - CSV export logic
  - **No changes**: CSV export must remain unchanged per requirements
  - Currently exports cards in tier-based order with NaN padding (line 258-268)

- **`src/imitation_learning/dataset.py`** - Dataset loader for training
  - **Review needed**: Verify it correctly loads the new feature dimension (156 + 39×4 instead of 144 + 36×4)
  - Should automatically work if only uses array shapes from loaded files

- **`src/imitation_learning/model.py`** - Neural network model
  - **Review needed**: Input dimension may need updating
  - Check if `input_dim` is hardcoded or read from preprocessing_stats.json

- **`src/imitation_learning/train.py`** - Training script
  - **Review needed**: Verify it reads input_dim from config or auto-detects from data

### Configuration Files

- **`src/imitation_learning/configs/config_small.yaml`** - Small dataset config
  - **Review needed**: Check if `input_dim` is specified; update if hardcoded

- **`src/imitation_learning/configs/config_medium.yaml`** - Medium dataset config
  - **Review needed**: Check if `input_dim` is specified; update if hardcoded

- **`src/imitation_learning/configs/config_large.yaml`** - Large dataset config
  - **Review needed**: Check if `input_dim` is specified; update if hardcoded

## Step by Step Tasks
IMPORTANT: Execute every step in order, top to bottom.

### Step 1: Implement Card Compaction Function

Add a new function to `src/imitation_learning/data_preprocessing.py` that:

1. **Extract card features from DataFrame** (before feature engineering):
   - Visible cards: Extract all columns matching `card{0-11}_{vp,level,cost_*,bonus_*}`
   - Reserved cards per player: Extract columns matching `player{0-3}_reserved{0-2}_{vp,level,cost_*,bonus_*}`

2. **Identify non-zero cards** for visible cards:
   - For each card (12 features), check if all features are 0
   - Non-zero cards: Keep in order
   - Zero cards: Move to end

3. **Create position index**:
   - Non-zero visible cards: Assign positions 0, 1, 2, ... (sequential)
   - Zero visible cards: Assign position -1
   - Reserved cards per player: Assign positions 12, 13, 14 (independent per player)
   - Missing reserved cards (all zeros): Assign position -1

4. **Rebuild DataFrame** with new columns:
   - `card{0-11}_position` - Position index feature (NEW)
   - `card{0-11}_{vp,level,cost_*,bonus_*}` - Existing 12 features (reordered)
   - `player{0-3}_reserved{0-2}_position` - Position index feature (NEW)
   - `player{0-3}_reserved{0-2}_{vp,level,cost_*,bonus_*}` - Existing 12 features (same order)

5. **Update feature_cols list** to include new position columns

Function signature:
```python
def compact_cards_and_add_position(df: pd.DataFrame) -> pd.DataFrame:
    """Compact visible cards (move zeros to end) and add position index feature.

    This function:
    1. Identifies non-zero visible cards (cards 0-11)
    2. Reorders them: non-zero first, zeros at end
    3. Adds position index feature: 0, 1, 2, ... for non-zero; -1 for zeros
    4. For reserved cards: adds position 12, 13, 14 per player; -1 for missing

    Args:
        df: DataFrame with filled NaN values (all zeros for missing cards)

    Returns:
        DataFrame with compacted cards and position features added
    """
```

### Step 2: Integrate into Preprocessing Pipeline

Modify `main()` in `src/imitation_learning/data_preprocessing.py`:

1. **Call order** (lines 861-1002):
   ```python
   # Load data
   df = load_all_games(...)

   # Generate masks BEFORE filling NaN (line 897)
   masks_all = generate_masks_for_dataframe(df)

   # Fill NaN → 0 (line 900)
   df_filled = fill_nan_values(df)

   # NEW: Compact cards and add position index
   df_compacted = compact_cards_and_add_position(df_filled)

   # Identify column groups (line 903)
   metadata_cols, label_cols, feature_cols = identify_column_groups(df_compacted)

   # Engineer features (one-hot encoding) (line 906)
   df_eng, feature_cols_eng, onehot_cols = engineer_features(df_compacted, feature_cols)

   # Continue with normalization, etc.
   ```

2. **Update comments** to document the new step

3. **Verify feature count**:
   - Old: 382 input features (before one-hot) → ~450 after one-hot
   - New: 382 - 144 - 144 + 156 + 156 = 406 input features → ~474 after one-hot
   - Calculation: Remove 12 old visible cards (144 features) + 4 players × 3 reserved (144 features), add 12 new visible (156 features) + 4 players × 3 reserved (156 features)

### Step 3: Update Normalization Mask

Modify `create_normalization_mask()` in `src/imitation_learning/data_preprocessing.py` (lines 299-326):

1. **Add position features to non-normalized list**:
   - Position indices are discrete (0, 1, 2, ..., -1), should NOT be normalized
   - Add check: `if 'position' in col: mask[idx] = False`

2. **Verify bonus one-hot columns** are still excluded from normalization

### Step 4: Update Model Input Dimension

Check and update model configuration:

1. **Review `src/imitation_learning/model.py`**:
   - Find where `input_dim` is defined
   - If hardcoded, update calculation
   - Preferred: Read from `preprocessing_stats.json` (auto-generated by preprocessing)

2. **Review config files** (`src/imitation_learning/configs/*.yaml`):
   - If `input_dim` is specified, update or remove (let it auto-detect)
   - Document the change in config comments

3. **Verify `src/imitation_learning/dataset.py`**:
   - Ensure it loads arrays without assuming specific dimensions
   - Should work automatically with new shapes

### Step 5: Update Documentation

Add comments and docstrings:

1. **Add docstring** to `compact_cards_and_add_position()` explaining:
   - Purpose: Improve model performance by removing positional bias from zero-padding
   - Position index: Helps model understand card availability
   - Scope: Visible cards only; reserved cards stay in player sections

2. **Update `data_preprocessing.py` module docstring** (lines 1-9):
   - Add note about card compaction step
   - Mention new feature count

3. **Update `README.md`** if it documents feature counts:
   - Search for "382 features" or similar
   - Update to new count (406 base features)

## Validation Commands
Execute every command to validate the chore is complete with zero regressions.

### Preprocessing Validation

```bash
# Test preprocessing on small dataset (debug mode)
uv run python -m src.imitation_learning.data_preprocessing \
  --config src/imitation_learning/configs/config_small.yaml \
  --max-games 10

# Verify output files were created
ls -lh data/processed/

# Check feature dimensions in saved arrays
uv run python -c "
import numpy as np
import json

# Load feature arrays
X_train = np.load('data/processed/X_train.npy')
X_val = np.load('data/processed/X_val.npy')
X_test = np.load('data/processed/X_test.npy')

# Load stats
with open('data/processed/preprocessing_stats.json') as f:
    stats = json.load(f)

print(f'X_train shape: {X_train.shape}')
print(f'X_val shape: {X_val.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'Input dim from stats: {stats[\"input_dim\"]}')
print(f'Number of features: {stats[\"num_features\"]}')

# Verify shapes match
assert X_train.shape[1] == stats['input_dim'], 'Shape mismatch!'
print('✓ All shapes match')
"

# Verify position indices are correct
uv run python -c "
import numpy as np
import json

# Load feature columns
with open('data/processed/feature_cols.json') as f:
    feature_cols = json.load(f)

# Find position columns
position_cols = [col for col in feature_cols if 'position' in col]
print(f'Position columns found: {len(position_cols)}')
print(f'Expected: 12 (visible) + 12 (reserved) = 24')
print(f'Position columns: {position_cols}')

# Load data to check position values
X_train = np.load('data/processed/X_train.npy')

# Find indices of position columns
position_indices = [i for i, col in enumerate(feature_cols) if 'position' in col]

# Check position values for first few samples
print('\\nSample position values (first 5 rows):')
for idx in position_indices[:5]:  # First 5 position columns
    col_name = feature_cols[idx]
    values = X_train[:5, idx]
    print(f'{col_name}: {values}')

# Verify position values are in expected range or -1
position_values = X_train[:, position_indices]
unique_vals = np.unique(position_values)
print(f'\\nUnique position values: {sorted(unique_vals)}')
print('Expected: -1, 0, 1, 2, ..., 14')
"
```

### Training Validation

```bash
# Test training with new features (1 epoch)
uv run python -m src.imitation_learning.train \
  --config src/imitation_learning/configs/config_small.yaml \
  --epochs 1

# Verify training completes without errors
echo "✓ Training validation passed"
```

### Model Loading Validation

```bash
# Test model can load preprocessed data
uv run python -c "
from src.imitation_learning.dataset import SplendorDataset
import json

# Load stats
with open('data/processed/preprocessing_stats.json') as f:
    stats = json.load(f)

# Create dataset
train_dataset = SplendorDataset('data/processed', split='train')

print(f'Dataset size: {len(train_dataset)}')
print(f'Input dim: {stats[\"input_dim\"]}')

# Get sample
sample = train_dataset[0]
print(f'Sample X shape: {sample[\"X\"].shape}')

# Verify shape matches
assert sample['X'].shape[0] == stats['input_dim'], 'Input dimension mismatch!'
print('✓ Dataset loads correctly')
"
```

### Regression Testing

```bash
# Run full preprocessing pipeline on small dataset
uv run python -m src.imitation_learning.data_preprocessing \
  --config src/imitation_learning/configs/config_small.yaml \
  --max-games 100

# Verify mask validation passes
uv run python -c "
import json

with open('data/processed/mask_validation_report.json') as f:
    report = json.load(f)

print(f'Mask validation status: {report[\"overall_status\"]}')
print(f'Total failures: {report[\"total_failures\"]}')

assert report['overall_status'] == 'PASSED', 'Mask validation failed!'
print('✓ Mask validation passed')
"

# Verify no duplicate position indices in non-zero cards
uv run python -c "
import numpy as np
import json

with open('data/processed/feature_cols.json') as f:
    feature_cols = json.load(f)

X_train = np.load('data/processed/X_train.npy')

# Get visible card position columns (card0-11)
visible_pos_cols = [i for i, col in enumerate(feature_cols)
                    if 'card' in col and 'position' in col and 'player' not in col]

print(f'Found {len(visible_pos_cols)} visible card position columns')

# For each sample, check that non-(-1) positions are unique and sequential
for sample_idx in range(min(100, len(X_train))):
    positions = X_train[sample_idx, visible_pos_cols]
    non_missing = positions[positions != -1]

    if len(non_missing) > 0:
        # Check sequential: should be 0, 1, 2, ...
        expected = np.arange(len(non_missing))
        if not np.array_equal(sorted(non_missing), expected):
            print(f'ERROR: Sample {sample_idx} has non-sequential positions: {sorted(non_missing)}')
            raise AssertionError('Position indices are not sequential!')

print('✓ All position indices are valid and sequential')
"
```

## Notes

### Implementation Considerations

1. **Feature dimension changes**:
   - Old total: ~450 features after one-hot encoding
   - New total: ~474 features after one-hot encoding (+24 position features)
   - Position features are discrete integers, not normalized

2. **Backward compatibility**:
   - Existing CSV files remain unchanged (per requirements)
   - New preprocessing required for all existing datasets
   - Saved model checkpoints will be incompatible (different input_dim)

3. **Position index semantics**:
   - Visible cards: 0-11 (or fewer if deck depleted)
   - Reserved cards: 12, 13, 14 (per player, independent)
   - Missing cards: -1 (sentinel value)
   - Compaction only affects visible cards (12 slots)

4. **Testing strategy**:
   - Use `--max-games 10` for quick iteration during development
   - Verify position values are correct before running full preprocessing
   - Check mask validation passes (ensures game state reconstruction still works)

5. **Performance impact**:
   - Minimal: Only affects preprocessing (one-time operation)
   - Array operations should be vectorized (avoid Python loops)
   - Training should be faster (fewer zero features to learn)

### Edge Cases to Handle

1. **All cards missing** (deck completely depleted at some level):
   - All 4 cards at that level have position -1
   - Compaction moves all 4 to the end

2. **No reserved cards** (player has 0 reserved):
   - All 3 reserved card positions are -1
   - All features are 0

3. **Partial reserved cards** (player has 1-2 reserved):
   - First N positions are 12, 13, 14
   - Remaining positions are -1

4. **Column ordering**:
   - Maintain relative order within metadata/feature/label groups
   - Position column should be first feature for each card (before vp, level, etc.)
