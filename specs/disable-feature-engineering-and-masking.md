# Chore: Disable Feature Engineering and Masking of Illegal Actions

## Chore Description
Disable feature engineering and masking of illegal actions in the data preprocessing and training pipeline to recreate a baseline dataset for training on the initial, unmodified data. This is needed because the models trained on the original data (without feature engineering and masking) were not saved, and we need to recreate them for comparison purposes.

The changes should be configuration-driven where possible, allowing easy toggling between:
1. **Original pipeline**: No feature engineering, no masking
2. **Enhanced pipeline**: With feature engineering and masking (current default)

This involves:
- Adding configuration flags to disable feature engineering during preprocessing
- Adding configuration flags to disable masking during training/evaluation
- Ensuring the pipeline produces compatible outputs regardless of configuration
- Maintaining backward compatibility with existing preprocessed data

## Relevant Files
Use these files to resolve the chore:

- **`src/imitation_learning/configs/config_tuning.yaml`** (lines 1-63)
  - Main configuration file where we'll add flags to control feature engineering and masking
  - Currently has preprocessing, model, training, compute, logging, and checkpointing sections

- **`src/imitation_learning/data_preprocessing.py`** (lines 1-1436)
  - Main preprocessing pipeline containing the `engineer_features()` function (lines 418-516)
  - Contains `generate_masks_for_dataframe()` function (lines 644-728)
  - Contains `main()` function that orchestrates preprocessing (lines 1234-1432)
  - Need to conditionally skip feature engineering and mask generation based on config

- **`src/imitation_learning/parallel_processor.py`**
  - Handles parallel processing of CSV files and calls feature engineering per batch
  - Need to ensure feature engineering can be disabled here as well

- **`src/imitation_learning/merge_batches.py`**
  - Merges preprocessed batches and may reference strategic features
  - Need to handle cases where strategic features are absent

- **`src/imitation_learning/train.py`** (lines 1-802)
  - Training loop that uses `compute_masked_predictions()` for accuracy computation (lines 306-362, 457-513)
  - Need to conditionally disable masking based on config flag
  - Masking is used in `train_one_epoch()` and `validate()` functions

- **`src/imitation_learning/evaluate.py`** (lines 1-100+)
  - Evaluation script that uses `apply_legal_action_mask()` (line 70)
  - Need to conditionally disable masking during evaluation

- **`src/imitation_learning/utils.py`** (lines 324-349)
  - Contains `compute_masked_predictions()` function that applies masking
  - This function should accept an optional parameter to disable masking

### New Files
None - all changes will be modifications to existing files.

## Step by Step Tasks
IMPORTANT: Execute every step in order, top to bottom.

### Step 1: Add Configuration Flags
- Add `enable_feature_engineering` flag to the `preprocessing` section in `config_tuning.yaml` (default: `true`)
- Add `enable_masking` flag to the `preprocessing` section in `config_tuning.yaml` (default: `true`)
- Add `enable_masking` flag to the `training` section in `config_tuning.yaml` (default: `true`)
- Add `enable_masking` flag to the `logging` section or create an `evaluation` section in `config_tuning.yaml` (default: `true`)

### Step 2: Modify Utility Functions to Support Optional Masking
- Update `compute_masked_predictions()` in `src/imitation_learning/utils.py` to accept an `enable_masking` parameter (default: `True`)
- When `enable_masking=False`, the function should return `logits.argmax(dim=1)` without applying masks
- Update the docstring to document this new parameter

### Step 3: Update Data Preprocessing Pipeline
- Modify `engineer_features()` in `src/imitation_learning/data_preprocessing.py` to check `config.get('preprocessing', {}).get('enable_feature_engineering', True)`
- When disabled, skip the strategic feature extraction loop (lines 482-512) but still perform one-hot encoding
- Modify `generate_masks_for_dataframe()` to check `config.get('preprocessing', {}).get('enable_masking', True)`
- When masking is disabled, generate all-ones masks (all actions legal) instead of reconstructing board states
- Update `main()` function to pass config to these functions and handle both modes

### Step 4: Update Parallel Processing Pipeline
- Modify the parallel processing code to respect the `enable_feature_engineering` and `enable_masking` flags
- Ensure batch files are created correctly whether or not features are engineered or masks are generated
- Update intermediate file format to be compatible with both modes

### Step 5: Update Training Loop
- Modify `train_one_epoch()` in `src/imitation_learning/train.py` to read `config['training'].get('enable_masking', True)`
- Update all calls to `compute_masked_predictions()` to pass the `enable_masking` parameter
- Locations: lines 306, 317, 331, 344, 357, 370, 382 (approximate based on current code)
- Modify `validate()` function similarly for validation loop
- Locations: lines 457, 468, 482, 495, 508, 520, 532 (approximate)

### Step 6: Update Evaluation Script
- Modify `evaluate_action_type()` and other evaluation functions in `src/imitation_learning/evaluate.py`
- Add masking configuration to evaluation config or read from training config
- Update calls to `apply_legal_action_mask()` to be conditional based on config
- When masking is disabled, use raw logits for predictions

### Step 7: Handle Backward Compatibility
- Ensure existing preprocessed data (with feature engineering and masks) still loads correctly
- Add validation in data loading to check if masks exist and warn if config mismatch detected
- Document the configuration options in comments

### Step 8: Create Baseline Configuration File
- Create a new config file `src/imitation_learning/configs/config_baseline.yaml` as a copy of `config_tuning.yaml`
- Set `enable_feature_engineering: false` and `enable_masking: false` in all relevant sections
- Document this as the "baseline" configuration for comparison experiments

### Step 9: Test Preprocessing Pipeline
- Run preprocessing with baseline config on a small test set (e.g., `--max-games 10`)
- Verify that feature engineering is skipped and masks are all-ones
- Check output dimensions are correct
- Verify no errors during preprocessing

### Step 10: Test Training Pipeline
- Run training for a few epochs with baseline config on preprocessed baseline data
- Verify masking is disabled during accuracy computation
- Check that training completes without errors
- Verify metrics are computed correctly

### Step 11: Run Validation Commands
Execute validation commands to ensure zero regressions.

## Validation Commands
Execute every command to validate the chore is complete with zero regressions.

- `uv run python -m src.imitation_learning.data_preprocessing --config src/imitation_learning/configs/config_baseline.yaml --max-games 10 --parallel true` - Test baseline preprocessing pipeline
- `uv run python -m src.imitation_learning.data_preprocessing --config src/imitation_learning/configs/config_tuning.yaml --max-games 10 --parallel true` - Test enhanced preprocessing pipeline (regression test)
- `uv run python -c "import yaml; config = yaml.safe_load(open('src/imitation_learning/configs/config_baseline.yaml')); assert config['preprocessing']['enable_feature_engineering'] == False; assert config['preprocessing']['enable_masking'] == False; print('✓ Baseline config validated')"` - Verify baseline config flags
- `uv run python -c "import yaml; config = yaml.safe_load(open('src/imitation_learning/configs/config_tuning.yaml')); assert config['preprocessing'].get('enable_feature_engineering', True) == True; assert config['preprocessing'].get('enable_masking', True) == True; print('✓ Enhanced config validated')"` - Verify enhanced config flags
- `uv run pytest tests/test_feature_engineering.py -v` - Run existing feature engineering tests (if they exist)

## Notes

### Important Design Decisions

1. **Feature Engineering vs One-Hot Encoding**: When `enable_feature_engineering=false`, we still perform one-hot encoding of categorical features (current_player, num_players, player positions) because this is necessary for the model to learn. We only skip the strategic feature extraction (lines 482-512 in data_preprocessing.py).

2. **Masking During Training**: Masking is currently ONLY used for accuracy computation, not for loss computation (by design - see comments at lines 105-108, 282 in train.py). When disabled, accuracy metrics will be computed on raw predictions without enforcing legality.

3. **All-Ones Masks**: When masking is disabled during preprocessing, we generate all-ones masks instead of skipping mask generation entirely. This maintains compatibility with the existing data loading pipeline which expects mask arrays.

4. **Backward Compatibility**: Existing preprocessed data with feature engineering and masks will continue to work. The config flags only affect NEW preprocessing runs.

5. **Performance Impact**: Disabling feature engineering will significantly speed up preprocessing (no expensive strategic feature extraction per row). Disabling masking will slightly speed up both preprocessing (no board reconstruction) and training (no mask application in accuracy computation).

### Configuration Flag Summary

| Flag | Location | Purpose | Default |
|------|----------|---------|---------|
| `preprocessing.enable_feature_engineering` | config YAML | Skip strategic feature extraction | `true` |
| `preprocessing.enable_masking` | config YAML | Generate all-ones masks instead of legal action masks | `true` |
| `training.enable_masking` | config YAML | Apply masking during training accuracy computation | `true` |
| `evaluation.enable_masking` | config YAML | Apply masking during evaluation | `true` |

### Expected Behavior

**With baseline config (all flags = false)**:
- Preprocessing: No strategic features, all-ones masks, faster processing
- Training: Raw predictions for accuracy, no legal action enforcement
- Model: Learns from basic features + one-hot encoding only
- Use case: Reproduce original baseline models

**With enhanced config (all flags = true, default)**:
- Preprocessing: Full strategic features, legal action masks
- Training: Masked predictions for accuracy, legal action enforcement
- Model: Learns from rich features and legal action constraints
- Use case: Current state-of-the-art approach
