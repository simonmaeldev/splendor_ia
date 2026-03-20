# Chore: Legacy Data Loading Compatibility

## Chore Description
Adapt the code in `train.py`, `evaluate.py`, and `evaluate_multiple_models.py` to support loading legacy preprocessed data that was generated without the masking feature. The user has regenerated the baseline and masking datasets using the original code that existed before the `enable_masking` flag was added.

**Problem**: The legacy `processed_baseline` data directory does NOT contain `masks_*.npz` files (only has features, labels, and metadata), while the current code expects masks to always be present.

**Solution**: Make the dataset loading code backward-compatible by:
1. Detecting when mask files are absent (legacy format)
2. Generating dummy all-ones masks on-the-fly when loading legacy data
3. Ensuring all downstream code (training, evaluation) works seamlessly with both legacy and new data formats

## Relevant Files
Use these files to resolve the chore:

- `src/imitation_learning/dataset.py` - Contains the data loading logic that needs to be made backward-compatible
  - `load_preprocessed_data()` function currently assumes masks files exist
  - `SplendorDataset` class expects masks to be provided
  - Need to add fallback logic when masks files are missing

- `src/imitation_learning/train.py` - Training script that uses the dataset
  - Already passes `enable_masking` flag to training/validation functions
  - Should work seamlessly once dataset loading is fixed
  - No changes needed here

- `src/imitation_learning/evaluate.py` - Evaluation script
  - Uses `create_dataloaders()` which needs the fix
  - Already has `enable_masking` parameter support
  - No changes needed here

- `scripts/evaluate_multiple_models.py` - Multi-model evaluation script
  - Uses `create_dataloaders()` which needs the fix
  - Should work once dataset loading is fixed
  - No changes needed here

### New Files
None - this is purely a compatibility fix for existing code.

## Step by Step Tasks
IMPORTANT: Execute every step in order, top to bottom.

### Step 1: Understand the Legacy Format
- Read the current `load_preprocessed_data()` function in `src/imitation_learning/dataset.py:79-117`
- Understand what files exist in legacy format:
  - `X_train.npy`, `X_val.npy`, `X_test.npy` (features)
  - `labels_train.npz`, `labels_val.npz`, `labels_test.npz` (labels)
  - NO `masks_train.npz`, `masks_val.npz`, `masks_test.npz` (missing in legacy)
- Understand what files exist in new format:
  - All of the above PLUS the masks files

### Step 2: Add Backward-Compatible Mask Loading
- Modify `load_preprocessed_data()` in `src/imitation_learning/dataset.py` to:
  - Check if mask files exist using `os.path.exists()`
  - If masks exist: Load them as currently done
  - If masks DON'T exist: Generate dummy all-ones masks with correct shapes
- Generate dummy masks based on the number of classes for each head:
  - `action_type`: 4 classes (BUILD, RESERVE, TAKE2, TAKE3)
  - `card_selection`: 15 classes (13 visible cards + 2 reserved)
  - `card_reservation`: 13 classes (12 visible + 1 from deck)
  - `gem_take3`: 10 classes (combinations of 3 different colors from 5)
  - `gem_take2`: 5 classes (which color to take 2 of)
  - `noble`: 6 classes (5 nobles + 1 for "no noble")
  - `gems_removed`: 56 classes (combinations of gems to remove when overflow)
- Ensure the shapes match: `(n_samples, n_classes_for_head)` filled with 1.0 values
- Add informative print statements indicating whether legacy or new format was detected

### Step 3: Test with Legacy Data (processed_baseline)
- Run the dataset loading with `data/processed_baseline` directory
- Verify that:
  - Features and labels load correctly
  - Dummy all-ones masks are generated with correct shapes
  - No errors or shape mismatches occur
- Test with a simple Python script or by running the dataset module directly:
  ```bash
  python -m src.imitation_learning.dataset data/processed_baseline
  ```

### Step 4: Test with New Data (processed_masking)
- Run the dataset loading with `data/processed_masking` directory
- Verify that:
  - Features, labels, AND masks load correctly from files
  - No regression in functionality
  - Masks contain actual legal action information (not all-ones)
- Test with a simple Python script or by running the dataset module directly:
  ```bash
  python -m src.imitation_learning.dataset data/processed_masking
  ```

### Step 5: Verify Training Works with Legacy Data
- Ensure `train.py` can load and train on legacy data
- The code already has `enable_masking` parameter support in training/validation
- No code changes needed here, just verification that it works end-to-end

### Step 6: Verify Evaluation Works with Both Formats
- Test `evaluate.py` with a model trained on legacy data
- Test `evaluate_multiple_models.py` with models from both legacy and new formats
- Ensure no errors occur during data loading

### Step 7: Run Validation Commands
- Execute all validation commands below to ensure zero regressions
- Verify that both legacy and new data formats work seamlessly

## Validation Commands
Execute every command to validate the chore is complete with zero regressions.

```bash
# Test legacy data loading (processed_baseline - no masks)
python -c "
import sys
sys.path.insert(0, 'src')
from imitation_learning.dataset import load_preprocessed_data
X_train, X_val, X_test, labels_train, labels_val, labels_test, masks_train, masks_val, masks_test = load_preprocessed_data('data/processed_baseline')
assert masks_train is not None, 'masks_train should not be None'
assert 'action_type' in masks_train, 'action_type mask missing'
print(f'✓ Legacy data loading test passed')
print(f'  Train samples: {X_train.shape[0]}, masks shape: {masks_train[\"action_type\"].shape}')
"

# Test new data loading (processed_masking - with masks)
python -c "
import sys
sys.path.insert(0, 'src')
from imitation_learning.dataset import load_preprocessed_data
X_train, X_val, X_test, labels_train, labels_val, labels_test, masks_train, masks_val, masks_test = load_preprocessed_data('data/processed_masking')
assert masks_train is not None, 'masks_train should not be None'
assert 'action_type' in masks_train, 'action_type mask missing'
print(f'✓ New data loading test passed')
print(f'  Train samples: {X_train.shape[0]}, masks shape: {masks_train[\"action_type\"].shape}')
"

# Test dataloader creation with legacy data
python -c "
import sys
sys.path.insert(0, 'src')
from imitation_learning.dataset import create_dataloaders
train_loader, val_loader, test_loader = create_dataloaders('data/processed_baseline', batch_size=32, num_workers=0)
states, labels, masks = next(iter(train_loader))
assert masks is not None, 'masks should not be None'
assert 'action_type' in masks, 'action_type mask missing'
print(f'✓ DataLoader test with legacy data passed')
print(f'  Batch shape: {states.shape}, action_type mask shape: {masks[\"action_type\"].shape}')
"

# Test dataloader creation with new data
python -c "
import sys
sys.path.insert(0, 'src')
from imitation_learning.dataset import create_dataloaders
train_loader, val_loader, test_loader = create_dataloaders('data/processed_masking', batch_size=32, num_workers=0)
states, labels, masks = next(iter(train_loader))
assert masks is not None, 'masks should not be None'
assert 'action_type' in masks, 'action_type mask missing'
print(f'✓ DataLoader test with new data passed')
print(f'  Batch shape: {states.shape}, action_type mask shape: {masks[\"action_type\"].shape}')
"
```

## Notes
- The key insight is that the `enable_masking` flag in the config controls whether masks are USED during training/evaluation, but the dataset loading code currently REQUIRES masks to exist
- This chore makes the dataset loading gracefully handle BOTH cases: when masks exist (new format) and when they don't (legacy format)
- Models trained on legacy data (processed_baseline) will still work because the dummy all-ones masks effectively disable masking (all actions appear legal)
- Models trained on new data (processed_masking) will continue to work as before
- No changes to training/evaluation logic are needed - only the data loading layer needs to be backward-compatible
- The number of classes for each head is hardcoded based on the game rules and feature engineering design - these are stable constants that won't change
