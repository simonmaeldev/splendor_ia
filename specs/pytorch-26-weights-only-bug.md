# Bug: PyTorch 2.6 weights_only Default Change Breaks Model Loading

## Bug Description
When running the evaluation script on a model checkpoint, PyTorch 2.6's new default `weights_only=True` parameter causes a `_pickle.UnpicklingError` because the saved checkpoints contain numpy objects (`numpy._core.multiarray.scalar`) which are not allowed with the stricter security setting.

**Symptoms:**
- Error: `_pickle.UnpicklingError: Weights only load failed`
- Error occurs in `src/imitation_learning/evaluate.py` at line 361
- Error message: "Unsupported global: GLOBAL numpy._core.multiarray.scalar was not an allowed global by default"

**Expected Behavior:**
The evaluation script should successfully load model checkpoints and run evaluation metrics.

**Actual Behavior:**
The script crashes with a pickle unpickling error when attempting to load checkpoints saved by the training script.

## Problem Statement
In PyTorch 2.6, the default value of the `weights_only` argument in `torch.load()` was changed from `False` to `True` for security reasons. This breaks checkpoint loading in `src/imitation_learning/evaluate.py` and `src/imitation_learning/train.py` because the checkpoints contain numpy scalar objects that are not allowed with `weights_only=True`.

## Solution Statement
Add the `weights_only=False` parameter to all `torch.load()` calls in the imitation learning module to explicitly allow loading of numpy objects in checkpoints. This solution is already implemented in other parts of the codebase (`scripts/evaluate_multiple_models.py:49` and `src/splendor/ai_player.py:67`) and should be applied consistently across all torch.load calls.

## Steps to Reproduce
1. Train a model using the training script (which saves checkpoints with numpy objects)
2. Run evaluation on the saved checkpoint:
   ```bash
   uv run python -m src.imitation_learning.evaluate --config data/models/202511250809_config_tuning/config.yaml --checkpoint data/models/202511250809_config_tuning/best_model.pth --split test
   ```
3. Observe the `_pickle.UnpicklingError` at line 361 in evaluate.py

## Root Cause Analysis
The root cause is a breaking change in PyTorch 2.6 that changed the default behavior of `torch.load()`:
- **Before PyTorch 2.6**: `weights_only=False` (default) - allows loading arbitrary Python objects
- **After PyTorch 2.6**: `weights_only=True` (default) - only allows loading tensors and basic types for security

The training script (`src/imitation_learning/train.py:606`) saves checkpoints that contain:
- Model state dict (tensors)
- Optimizer state dict (tensors + numpy scalars)
- Metadata (epoch, val_loss, val_acc - may contain numpy types)

When the optimizer state or metrics contain numpy scalars, they cannot be loaded with `weights_only=True`.

## Relevant Files
Use these files to fix the bug:

### Files to Modify
- **`src/imitation_learning/evaluate.py:361`** - Main evaluation script that loads checkpoints for model evaluation. Needs `weights_only=False` parameter added to `torch.load()`.

- **`src/imitation_learning/train.py:615`** - Training script's checkpoint loading function used when resuming training. Needs `weights_only=False` parameter added to `torch.load()` for consistency.

### Reference Files (Already Fixed)
- **`scripts/evaluate_multiple_models.py:49`** - Already uses `weights_only=False` parameter. This is the correct pattern to follow.

- **`src/splendor/ai_player.py:67`** - Already uses `weights_only=False` parameter with a helpful comment explaining why.

## Step by Step Tasks
IMPORTANT: Execute every step in order, top to bottom.

### Step 1: Fix evaluate.py checkpoint loading
- Modify `src/imitation_learning/evaluate.py` line 361
- Change `torch.load(args.checkpoint, map_location=device)` to `torch.load(args.checkpoint, map_location=device, weights_only=False)`
- Add a comment above explaining the parameter is needed for numpy object compatibility

### Step 2: Fix train.py checkpoint loading
- Modify `src/imitation_learning/train.py` line 615
- Change `torch.load(checkpoint_path)` to `torch.load(checkpoint_path, weights_only=False)`
- Add a comment above explaining the parameter is needed for numpy object compatibility

### Step 3: Run validation commands
- Execute all validation commands listed below to confirm the bug is fixed with zero regressions
- Verify the evaluation script runs successfully on an existing model checkpoint
- Verify no other torch.load calls in the codebase are affected

## Validation Commands
Execute every command to validate the bug is fixed with zero regressions.

- `uv run python -m src.imitation_learning.evaluate --config data/models/202511250809_config_tuning/config.yaml --checkpoint data/models/202511250809_config_tuning/best_model.pth --split test` - Verify the evaluation script now works without errors
- `grep -r "torch\.load" src/imitation_learning/ --include="*.py"` - Verify all torch.load calls in imitation_learning module are updated
- `grep -r "torch\.load" src/ scripts/ --include="*.py" | grep -v "weights_only"` - Check if any other torch.load calls need updating (should only show lines that don't need it)

## Notes
- This is a minimal fix that only addresses the immediate bug by adding the `weights_only=False` parameter
- The `weights_only=False` parameter is safe in this context because:
  1. The checkpoints are generated by our own training script
  2. They are stored locally in the `data/models/` directory
  3. Users should only load checkpoints they trust (as noted in PyTorch documentation)
- Alternative solution (not recommended): Modify the save checkpoint logic to avoid numpy objects, but this would be a larger change and could introduce compatibility issues with existing checkpoints
- The fix follows the same pattern already established in `scripts/evaluate_multiple_models.py` and `src/splendor/ai_player.py`
- No new libraries are needed; this is a parameter change only
