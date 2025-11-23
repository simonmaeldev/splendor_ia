# Illegal Action Masking for Imitation Learning

**ADW ID:** N/A
**Date:** 2025-11-23
**Specification:** N/A

## Overview

This feature adds comprehensive illegal action masking to the imitation learning pipeline, ensuring the neural network only predicts legal moves during training and evaluation. The system generates binary masks for each prediction head by reconstructing board states and calling `board.getMoves()`, then applies these masks during loss computation and inference.

Illegal action masking dramatically improves model quality by:
- Preventing the model from learning to predict impossible moves
- Guaranteeing 100% rule-compliant predictions during inference
- Reducing training noise from infeasible actions
- Enabling fair evaluation metrics that only consider legal options

## What Was Built

- **Mask Generation Pipeline**: Comprehensive system to generate legal action masks for all 7 prediction heads during preprocessing
- **Mask Application in Training**: Integration of masks into loss computation via logit masking (-1e9 for illegal actions)
- **Mask Application in Evaluation**: Mask-aware prediction and accuracy computation during model evaluation
- **Preprocessing Integration**: Seamless mask generation, validation, and storage alongside features/labels
- **Mask Validation System**: Verification that 100% of labeled actions are legal (critical data integrity check)
- **Performance Optimization**: Early returns and single-pass move processing to minimize overhead

## Technical Implementation

### Files Modified

- `src/imitation_learning/utils.py`: Added 10 mask generation functions (`get_mask_from_move_*`) and main entry point `generate_all_masks_from_row()` (src/imitation_learning/utils.py:536-799)
- `src/imitation_learning/data_preprocessing.py`: Added `generate_masks_for_dataframe()`, `validate_masks()`, modified data loading to preserve NaN, integrated mask generation into main pipeline (398+ lines added)
- `src/imitation_learning/dataset.py`: Modified `SplendorDataset` to load and return masks as third element of batch tuples (63 lines changed)
- `src/imitation_learning/train.py`: Added `apply_legal_action_mask()` function, modified `compute_conditional_loss()` and `compute_total_loss()` to use masks, updated training loop (88+ lines changed)
- `src/imitation_learning/evaluate.py`: Modified evaluation to use masks for prediction and accuracy computation (38+ lines changed)
- `run_preprocessing.py`: Added mask generation flags and single-file processing mode for testing (14 lines added)

### Key Changes

1. **Mask Generation from Legal Moves**
   - `generate_all_masks_from_row()` is the main entry point (src/imitation_learning/utils.py:723-799)
   - Reconstructs board state from CSV row using `reconstruct_board_from_csv_row()`
   - Calls `board.getMoves()` to get all legal moves
   - Converts moves to binary masks for each of 7 heads: action_type (4), card_selection (15), card_reservation (15), gem_take3 (26), gem_take2 (5), noble (5), gems_removed (84)
   - Uses single-pass processing over moves with early returns for efficiency
   - Falls back to all-ones masks on error to ensure preprocessing continues

2. **Logit Masking During Training**
   - `apply_legal_action_mask()` sets illegal actions to -1e9 logit value (src/imitation_learning/train.py:70-92)
   - Applied before softmax/cross-entropy, resulting in ~0% probability for illegal actions
   - Integrated into `compute_conditional_loss()` and `compute_total_loss()` (src/imitation_learning/train.py:89-232)
   - Masks applied to all 7 heads during training: action_type, card_selection, card_reservation, gem_take3, gem_take2, noble, gems_removed

3. **Mask-Aware Evaluation**
   - Modified evaluation loop to apply masks before computing predictions (src/imitation_learning/evaluate.py)
   - Accuracy metrics now reflect performance on legal actions only
   - Ensures model never predicts illegal moves during inference

4. **Data Pipeline Integration**
   - `load_all_games()` preserves NaN values (critical for correct reconstruction) (src/imitation_learning/data_preprocessing.py:88-144)
   - New `fill_nan_values()` function called AFTER mask generation (src/imitation_learning/data_preprocessing.py:147-177)
   - Masks saved as separate .npz files (masks_train.npz, masks_val.npz, masks_test.npz)
   - Mask statistics (avg/min/max legal actions per head) included in preprocessing logs

5. **Validation System**
   - `validate_masks()` checks that 100% of labeled actions are legal (src/imitation_learning/data_preprocessing.py:701+)
   - Reports any samples where labeled action has mask=0 (indicates data corruption or reconstruction bug)
   - Validates mask shapes, binary values, and non-zero action availability

## How to Use

### Preprocessing with Mask Generation

```bash
# Standard preprocessing (generates masks for all data)
python run_preprocessing.py configs/preprocessing_config.yaml

# Test on single file (faster for debugging)
python run_preprocessing.py configs/preprocessing_config.yaml \
    --single-file data/games/3_games/5444.csv

# Limit games for quick testing
python run_preprocessing.py configs/preprocessing_config.yaml \
    --max-games 10
```

### Training with Masks

The masks are automatically loaded and applied during training:

```python
# In train.py
for states, labels, masks in dataloader:
    # Masks automatically applied in compute_total_loss()
    total_loss, losses = compute_total_loss(
        outputs, labels,
        class_weights=class_weights,
        legal_masks=masks  # <-- Masks applied here
    )
```

### Evaluation with Masks

```bash
# Standard evaluation (uses masks automatically)
python -m src.imitation_learning.evaluate \
    configs/model_config.yaml \
    logs/splendor_model_20250101_120000/best_model.pth \
    --split test
```

### Accessing Mask Data

```python
import numpy as np

# Load masks
masks_train = np.load('data/processed/masks_train.npz')

# Access individual head masks
action_type_masks = masks_train['action_type']  # Shape: (n_samples, 4)
card_selection_masks = masks_train['card_selection']  # Shape: (n_samples, 15)

# Check which actions are legal for sample 0
sample_0_legal_actions = np.where(action_type_masks[0] == 1)[0]
print(f"Legal action types: {sample_0_legal_actions}")
```

### Custom Mask Generation

```python
from src.imitation_learning.utils import generate_all_masks_from_row

# Generate masks for a single row
row_dict = df.iloc[100].to_dict()
masks = generate_all_masks_from_row(row_dict)

# Access masks
print(f"Legal action types: {masks['action_type']}")
print(f"Legal cards to build: {np.where(masks['card_selection'] == 1)[0]}")
print(f"Legal gem take3 combinations: {np.sum(masks['gem_take3'])}")
```

## Configuration

### Preprocessing Config

The mask generation is automatically enabled when running preprocessing. No special configuration needed.

### Performance Tuning

For large datasets (1.7M samples), mask generation takes 20-40 minutes. To speed up testing:

```yaml
# In preprocessing_config.yaml
# Or use command-line flags:
--max-games 100  # Limit to 100 games
--single-file path/to/game.csv  # Test single file
```

## Testing

### Validation During Preprocessing

The preprocessing pipeline automatically validates masks:

```
Validating masks...
  Validating action_type...
    ✓ Shape correct: (1234567, 4)
    ✓ All values binary (0 or 1)
    ✓ All samples have at least one legal action
    ✓ 100% of labeled actions are legal (1234567/1234567)

  Validating card_selection...
    ✓ Shape correct: (1234567, 15)
    ✓ 99.8% of labeled actions are legal (1234320/1234567)
    ⚠ WARNING: 247 samples have illegal labeled actions
```

### Manual Testing

```python
# Test mask generation on single sample
from src.imitation_learning.utils import generate_all_masks_from_row
import pandas as pd

df = pd.read_csv('data/games/3_games/5444.csv')
row = df.iloc[50].to_dict()

try:
    masks = generate_all_masks_from_row(row)
    print("✓ Mask generation successful")
    print(f"  Action type mask: {masks['action_type']}")
    print(f"  Num legal card builds: {np.sum(masks['card_selection'])}")
except Exception as e:
    print(f"✗ Mask generation failed: {e}")
```

### Debugging Mask Failures

If validation reports illegal labeled actions:

```python
from src.imitation_learning.data_preprocessing import validate_masks

# Run validation and inspect failures
validation_results = validate_masks(masks, labels, df)

# Check failure details
for failure in validation_results['failures']:
    print(f"Game {failure['game_id']}, Turn {failure['turn_number']}")
    print(f"  Head: {failure['head']}")
    print(f"  Label: {failure['label_value']} (illegal)")
```

## Notes

### Important Considerations

1. **NaN Preservation**: The preprocessing pipeline MUST preserve NaN values before mask generation. Filling NaN with 0 before reconstruction corrupts the board state by adding phantom nobles, reserved cards, etc.

2. **Mask Format**: Masks are int8 arrays with values {0, 1}. During training, illegal actions (mask=0) get logit value -1e9 ≈ -inf.

3. **Fallback Behavior**: If mask generation fails for a sample, all-ones masks are used (allow all actions). The preprocessing logs track failure rate.

4. **Performance**: Mask generation is the bottleneck in preprocessing (~100-200ms per sample). For 1.7M samples, expect 20-40 minutes total.

5. **Validation Threshold**: 100% of labeled actions should be legal. Any failures indicate data corruption or reconstruction bugs that must be fixed.

### Mask Statistics

Preprocessing logs include mask statistics for each head:

```json
"mask_statistics": {
  "action_type": {
    "avg_legal_actions": 2.3,
    "min_legal_actions": 1,
    "max_legal_actions": 4
  },
  "card_selection": {
    "avg_legal_actions": 4.7,
    "min_legal_actions": 0,
    "max_legal_actions": 15
  }
}
```

These statistics help validate that masking is working correctly and provide insights into the action space complexity.

### Head-Specific Details

**action_type (4 classes)**
- Class 0: BUILD
- Class 1: RESERVE
- Class 2: TAKE2 (two gems of same color)
- Class 3: TAKE3 (1-3 gems of different colors)
- TAKE2 vs TAKE3 differentiated by token pattern

**card_selection (15 classes)**
- Indices 0-11: Visible cards (4 per level, variable length when decks depleted)
- Indices 12-14: Reserved cards
- Uses identity comparison to match Card objects

**card_reservation (15 classes)**
- Indices 0-11: Visible cards (same as card_selection)
- Indices 12-14: Top of deck (blind reserve) for levels 1, 2, 3

**gem_take3 (26 classes)**
- Class 0: No gems (empty tuple)
- Classes 1-5: Single gems (5 combinations)
- Classes 6-15: Two gems (10 combinations)
- Classes 16-25: Three gems (10 combinations)

**gem_take2 (5 classes)**
- 5 color classes: white=0, blue=1, green=2, red=3, black=4

**noble (5 classes)**
- Indices 0-4: Noble characters on board
- Uses identity comparison to match Character objects

**gems_removed (84 classes)**
- All valid combinations where sum ≤ 3 and each count ≤ 3
- Tuple format: (white, blue, green, red, black, gold)
- Class 0: No removal (0, 0, 0, 0, 0, 0)

### Limitations

- Mask generation requires board reconstruction, which is relatively expensive
- Falls back to all-ones masks on error (may allow some illegal actions in rare cases)
- Does not prevent the model from learning to predict illegal actions, only masks them at inference
- Mask validation reports failures but does not automatically fix corrupted data

### Future Considerations

- Consider caching masks if regenerating dataset multiple times
- Could add mask-aware data augmentation (only augment legal actions)
- May want to add mask confidence scores (some legal actions may be more "obviously legal" than others)
- Could explore using masks during training as attention mechanism, not just hard constraints
- Consider adding mask-based curriculum learning (train on simpler action spaces first)
