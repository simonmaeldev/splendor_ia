# Chore: Comment Out Noisy Model Heads

## Chore Description
Comment out the `gems_removed`, `noble`, and `card_reservation` prediction heads in the training and evaluation scripts. These heads are noisy, not converging properly, and preventing other heads from converging effectively, which is negatively impacting the model's overall performance. The heads should only be disabled in the `src/imitation_learning/train.py` and `src/imitation_learning/evaluate.py` files, which means some labels will be ignored during training and evaluation. The model architecture itself and the config file remain unchanged - we're only modifying the loss computation and accuracy tracking logic.

## Relevant Files
The following files are relevant to this chore:

- **`src/imitation_learning/train.py`** (lines 133-233, 236-395, 398-537)
  - Contains `compute_total_loss()` function that computes loss for all heads including the noisy ones
  - Contains `train_one_epoch()` function that computes accuracies for all heads including the noisy ones
  - Contains `validate()` function that computes accuracies for all heads including the noisy ones
  - Need to comment out loss computation and accuracy tracking for `gems_removed`, `noble`, and `card_reservation`

- **`src/imitation_learning/evaluate.py`** (lines 104-167, 170-260, 310-461)
  - Contains `evaluate_conditional_head()` function used to evaluate individual heads
  - Contains `evaluate_overall_action_accuracy()` function that checks target correctness for all action types
  - Contains `main()` function that evaluates all heads and computes metrics
  - Need to comment out evaluation calls for `gems_removed`, `noble`, and `card_reservation`

- **`src/imitation_learning/configs/config_tuning.yaml`** (read-only for reference)
  - Contains the model configuration including num_classes for all heads
  - This file is NOT modified - the heads remain in the model architecture
  - The model will still output predictions for these heads, but they won't contribute to the loss

## Step by Step Tasks
IMPORTANT: Execute every step in order, top to bottom.

### 1. Comment out noisy heads in `compute_total_loss()` function
- Open `src/imitation_learning/train.py`
- Locate the `compute_total_loss()` function (around lines 133-233)
- Comment out the loss computation for:
  - `card_reservation` (lines 175-184)
  - `noble` (lines 208-216)
  - `gems_removed` (lines 218-225)
- Add explanatory comments indicating these heads are disabled due to noise/convergence issues
- Ensure the commented-out losses are NOT added to the `total_loss` sum

### 2. Comment out noisy heads accuracy tracking in `train_one_epoch()`
- In `src/imitation_learning/train.py`
- Locate the `train_one_epoch()` function (around lines 236-395)
- Comment out the accuracy computation for:
  - `card_reservation` (lines 320-331)
  - `noble` (lines 359-369)
  - `gems_removed` (lines 371-380)
- Add explanatory comments for each commented section

### 3. Comment out noisy heads accuracy tracking in `validate()`
- In `src/imitation_learning/train.py`
- Locate the `validate()` function (around lines 398-537)
- Comment out the accuracy computation for:
  - `card_reservation` (lines 467-478)
  - `noble` (lines 506-516)
  - `gems_removed` (lines 518-527)
- Add explanatory comments for each commented section

### 4. Comment out noisy heads evaluation in `evaluate.py` main function
- Open `src/imitation_learning/evaluate.py`
- Locate the `main()` function (around lines 310-461)
- Comment out the evaluation calls for:
  - `card_reservation` (lines 379-383)
  - `noble` (lines 397-401)
  - `gems_removed` (lines 403-407)
- Add explanatory comments for each commented section

### 5. Comment out noisy heads in `evaluate_overall_action_accuracy()`
- In `src/imitation_learning/evaluate.py`
- Locate the `evaluate_overall_action_accuracy()` function (around lines 170-260)
- Comment out the target correctness checks for:
  - `card_reservation` in RESERVE action handling (lines 231-237)
  - `noble` is not explicitly checked in this function (it's part of BUILD action, only card_selection is checked)
- Add explanatory comments indicating these target checks are disabled

### 6. Update `train_one_epoch()` and `validate()` losses_accum and accuracies_accum dictionaries
- In `src/imitation_learning/train.py`
- Update the initialization dictionaries in both `train_one_epoch()` (lines 259-261) and `validate()` (lines 417-419)
- Comment out or remove the disabled heads from these dictionaries to avoid KeyErrors
- Ensure the dictionaries only contain: `action_type`, `card_selection`, `gem_take3`, `gem_take2`

### 7. Update validation history tracking in `main()` training loop
- In `src/imitation_learning/train.py`
- Locate the `main()` function around line 706
- Update the `val_accuracies` initialization to only include active heads
- Remove `card_reservation`, `noble`, `gems_removed` from the dictionary initialization

### 8. Run the Training Script to Validate Changes
Execute the training script with the tuning config to ensure:
- The model loads correctly
- Loss computation works without the commented heads
- Training proceeds without errors
- Only active heads are logged to wandb
- No KeyErrors or missing dictionary keys

### 9. Run the Evaluation Script to Validate Changes
Execute the evaluation script to ensure:
- Model checkpoint loads correctly
- Evaluation runs without errors for active heads
- Commented heads are skipped without issues
- Results are saved correctly with only active heads included

## Validation Commands
Execute every command to validate the chore is complete with zero regressions.

- `cd /home/apprentyr/projects/splendor_ia && timeout 60 uv run python -m src.imitation_learning.train --config src/imitation_learning/configs/config_tuning.yaml` - Run training script for 60 seconds to validate it starts correctly, computes losses only for active heads, and trains without errors
- `cd /home/apprentyr/projects/splendor_ia && uv run python -c "import sys; sys.path.insert(0, 'src'); from imitation_learning.train import compute_total_loss; print('compute_total_loss import successful')"` - Validate the train module imports correctly after changes
- `cd /home/apprentyr/projects/splendor_ia && uv run python -c "import sys; sys.path.insert(0, 'src'); from imitation_learning.evaluate import main; print('evaluate module import successful')"` - Validate the evaluate module imports correctly after changes

## Notes
- The model architecture remains unchanged - all heads (including `gems_removed`, `noble`, `card_reservation`) are still present in the `MultiHeadSplendorNet` and will still produce outputs
- Only the loss computation and accuracy tracking are disabled for these heads
- This means the model will waste some computation on these heads during forward passes, but they won't affect the training gradients
- Labels for these heads will be ignored (won't contribute to loss), which is acceptable per the user's requirements
- The config file `config_tuning.yaml` is not modified - it still defines all 7 heads
- If in the future these heads need to be re-enabled, simply uncomment the relevant sections in train.py and evaluate.py
- This approach allows for quick experimentation with different head combinations without modifying the model architecture or config files
