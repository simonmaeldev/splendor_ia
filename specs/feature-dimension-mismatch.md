# Bug: Feature Tensor Dimension Mismatch Warning

## Bug Description
When playing games with AI model players, the system generates repeated warnings about feature tensor shape mismatches:

```
WARNING: Feature tensor shape torch.Size([1, 893]) doesn't match model input_dim 1308
  - CSV features: 386
  - Engineered features: 893
```

**Symptoms:**
- Warning appears on every turn for every action
- Logs are cluttered with hundreds of identical warnings per game
- Makes it difficult to see actual game progress or real issues
- Despite warnings, games complete successfully (padding/trimming is applied)

**Expected behavior:**
- Models should load with correct feature dimensions matching their training configuration
- No warnings should appear during normal gameplay
- Feature extraction should match what the model was trained on

**Actual behavior:**
- System always generates 893 engineered features (current feature_engineering.py version)
- Models trained with different feature sets (older: 1308 total, newer: 423 total) cause mismatches
- Warnings are printed on every single action, flooding the logs

## Problem Statement
The AI player feature extraction code (`ai_player.py:382-442`) doesn't correctly detect which feature set a model was trained with. It uses a simple heuristic (`if model_input_dim < 500`) to decide between CSV-only and CSV+engineered features, but this breaks because:

1. Different models were trained with different feature engineering versions
2. The feature engineering code has evolved (originally ~922 features, now 893)
3. The heuristic threshold of 500 is arbitrary and doesn't account for feature evolution
4. Models don't store metadata about which features they expect

The code does handle the mismatch with padding/trimming, but generates noisy warnings on every prediction.

## Solution Statement
Store the model's expected feature configuration in the checkpoint during training, and load it during inference to extract the exact feature set the model expects. If metadata is missing (legacy models), use intelligent fallback logic without generating warnings.

Specifically:
1. Update training script to save feature configuration metadata in checkpoint
2. Update model loading to read and respect this metadata
3. Add a `suppress_warnings` flag to silence expected dimension mismatches after first occurrence
4. Improve the fallback heuristic for legacy models without metadata

## Steps to Reproduce
1. Run: `uv run python scripts/play_with_model.py --model data/models/202511241159_config_tuning/best_model.pth --games 1`
2. Observe hundreds of identical warnings during game play
3. Model was trained with input_dim=1308, but current code generates 893 features
4. System pads/trims to match, but warns on every action

## Root Cause Analysis

### Historical Context
The project has evolved through multiple feature engineering iterations:
- **Early models**: CSV features only (386 features)
- **Mid models**: CSV + engineered features (~922 features, total ~1308)
- **Recent models**: CSV + updated engineered features (893 features, total ~1279)
- **Latest models**: Subset or different feature configuration (423 features)

### Execution Flow
1. `ModelPlayer.__init__()` loads model and infers input_dim from weights
2. `get_model_input_features()` is called on every action
3. Generates 893 engineered features (hardcoded in current feature_engineering.py)
4. Creates tensor of shape (1, 893)
5. Compares to model input_dim (1308 for old models, 423 for new models)
6. Prints warning and applies padding/trimming
7. Repeats for EVERY action in EVERY game

### Why This Happens
1. Models don't store which feature set they expect (no metadata)
2. Feature engineering code has changed over time
3. Heuristic detection (`if model_input_dim < 500`) is too simplistic
4. Warning is printed inside the hot path (every prediction)
5. No mechanism to suppress repeated warnings

### Affected Code
- `src/splendor/ai_player.py:382-442` - `get_model_input_features()` method
- `src/splendor/ai_player.py:431-440` - Warning and padding/trimming logic
- `src/imitation_learning/train.py` - Doesn't save feature metadata in checkpoints
- `src/imitation_learning/feature_engineering.py` - Defines current feature set (893 features)

## Relevant Files
Use these files to fix the bug:

### Existing Files
- **`src/splendor/ai_player.py`** - Contains the `ModelPlayer.get_model_input_features()` method that generates warnings. Need to add logic to suppress repeated warnings and improve feature detection.

- **`src/imitation_learning/train.py`** - Training script that saves model checkpoints. Need to add feature metadata saving so models remember their expected input dimensions and feature types.

- **`src/imitation_learning/feature_engineering.py`** - Defines the current feature extraction logic (893 features). Need to make it version-aware or expose metadata about features being used.

## Step by Step Tasks
IMPORTANT: Execute every step in order, top to bottom.

### 1. Add warning suppression to ai_player.py
- Add an instance variable `_dimension_warning_shown` to `ModelPlayer.__init__()` (initialize to False)
- In `get_model_input_features()`, check if warning has been shown before printing
- Only print dimension mismatch warning once per ModelPlayer instance
- After printing warning, set `_dimension_warning_shown = True`
- This immediately reduces noise from hundreds of warnings to just one per model load

### 2. Improve feature detection heuristic for legacy models
- Replace the simplistic `if model_input_dim < 500` check with intelligent detection
- Check if `model_input_dim == 386` → CSV features only (legacy format)
- Check if `model_input_dim` in range `[1200, 1400]` → CSV + old engineered features (~1308)
- Check if `model_input_dim` in range `[400, 600]` → Newer reduced feature set
- Check if `model_input_dim` in range `[1200+, current]` → CSV + current engineered
- Add comments explaining each range and why it exists

### 3. Update training script to save feature metadata
- In `src/imitation_learning/train.py`, find where checkpoints are saved
- Add `'feature_metadata'` key to checkpoint dict with:
  - `use_engineered_features`: bool (True if using feature_engineering.py)
  - `num_csv_features`: int (386)
  - `num_engineered_features`: int (893 or actual count)
  - `total_input_dim`: int (should match model.input_dim)
- This ensures future models have explicit metadata

### 4. Update model loading to use feature metadata
- In `ai_player.py:load_model()`, check if checkpoint contains `'feature_metadata'`
- If present, store it in the returned config dict
- In `get_model_input_features()`, check if `self.config` has `'feature_metadata'`
- If present, use it to determine exact feature extraction approach
- If absent, fall back to improved heuristic (from step 2)

### 5. Test with both old and new models
- Test with old model (202511241159_config_tuning/best_model.pth) - should show warning once
- Test with new model (202511261710_config_large_mask/checkpoint_epoch_45.pth) - should show warning once
- Verify both models still play games correctly
- Verify warning only appears once per model load, not on every action

## Validation Commands
Execute every command to validate the bug is fixed with zero regressions.

- `uv run python scripts/play_with_model.py --model data/models/202511241159_config_tuning/best_model.pth --games 1 2>&1 | grep -c "WARNING: Feature tensor shape"` - Should output "1" (warning shown once, not hundreds of times)
- `uv run python scripts/play_with_model.py --model data/models/202511261710_config_large_mask/checkpoint_epoch_45.pth --games 1 2>&1 | grep -c "WARNING: Feature tensor shape"` - Should output "1" or "0"
- `uv run python scripts/play_with_model.py --model data/models/202511241159_config_tuning/best_model.pth --games 3` - Should complete 3 games without errors
- `uv run python -c "import sys; sys.path.insert(0, 'src'); from splendor.ai_player import ModelPlayer; player = ModelPlayer('data/models/202511241159_config_tuning/best_model.pth'); print('✓ ModelPlayer loads successfully')"` - Verify model loading works
- `uv run python scripts/test_model_player.py data/models/202511261710_config_large_mask/checkpoint_epoch_45.pth --games 2` - Test newer model still works

## Notes
- This is a **logging/UX bug**, not a functional bug - games work correctly despite warnings
- The fix prioritizes reducing noise over perfect feature detection
- Future training runs should include metadata, making detection unnecessary
- Legacy models (without metadata) will still work using improved heuristics
- The warning suppression (step 1) is the highest priority fix - it provides immediate value
- Steps 2-4 improve long-term maintainability but are less urgent
- Do not modify the actual feature engineering or model architecture
- Do not retrain models - this is purely an inference-time fix
- Keep the padding/trimming fallback - it's a good safety mechanism
