# Bug: Action Masking During Training Causing Poor Convergence

## Bug Description

The imitation learning model exhibits poor convergence with high variance and worse accuracy than baseline when action masking is applied during training. The model shows:
- **Worse accuracy** compared to training without masks
- **High variance** in metrics across epochs
- **No convergence** - loss and accuracy do not stabilize
- **Unstable training dynamics** with unpredictable behavior

Expected behavior: Action masking should prevent illegal action predictions and improve model performance.

Actual behavior: Action masking is actively degrading model performance, preventing convergence, and creating conflicting optimization signals.

## Problem Statement

The current implementation has **three critical bugs** working together to prevent model convergence:

1. **Masking during training prevents learning state→legality relationships**: The model never learns which actions are legal from the game state because masks are always applied, creating a dependency on external masks rather than learned patterns.

2. **Class weights + action masking create conflicting optimization signals**: Class weights attempt to balance frequencies across the entire dataset while masks restrict valid actions per-sample, creating contradictory gradient signals.

3. **Inconsistent loss computation between training and validation**: Training loop passes class_weights, validation loop doesn't, causing different loss landscapes during training vs evaluation.

4. **Corrupted gradient flow through masked logits**: Adding `-1e9` to illegal action logits causes the model to waste capacity learning to output large negative values instead of discriminating among legal actions.

## Solution Statement

**Remove all action masking from loss computation during training**, relying on the fact that expert demonstrations only contain legal actions. The model will naturally learn to assign low probability to illegal actions through the data distribution. Apply masking only during inference (accuracy calculation) as a safety mechanism.

Additionally, **remove all class weights** which conflict with the conditional loss masking already in place (e.g., only computing card_selection loss when action_type==BUILD).

This approach:
- Allows the model to learn state→legality mappings naturally
- Eliminates conflicting optimization signals
- Creates consistent training/validation behavior
- Enables clean gradient flow
- Maintains safety through inference-time masking

## Steps to Reproduce

1. Train the model with current configuration:
   ```bash
   cd /home/apprentyr/projects/splendor_ia
   uv run python -m src.imitation_learning.train --config src/imitation_learning/configs/config_small.yaml
   ```

2. Observe training metrics in wandb:
   - Validation accuracy fluctuates wildly (high variance)
   - Training loss doesn't decrease smoothly
   - No clear convergence pattern after 20+ epochs
   - Accuracy worse than previous runs without masking

3. Check the mask validation report:
   ```bash
   cat data/processed/mask_validation_report.json
   ```
   - Confirms masks are correct (100% of labeled actions are legal)
   - Shows the masks themselves are not the issue

## Root Cause Analysis

### Root Cause 1: Training/Inference Mismatch

During training, the model sees:
```python
logits = model(state)  # e.g., [0.5, 1.2, 0.8, -0.3]
masked_logits = logits + (1-mask) * -1e9  # e.g., [0.5, 1.2, -1e9, -0.3]
loss = cross_entropy(masked_logits, label)
```

The model never learns to make raw logits sensible because they're always masked. During inference, if masking is removed or applied differently, the model's raw outputs are meaningless.

### Root Cause 2: Model Wastes Capacity on Illegal Actions

Gradients flow through the `-1e9` masked values:
```python
∂loss/∂logits[illegal_action] ≈ 0.0000001 (tiny gradient pushing it more negative)
```

The model learns to output increasingly negative values for illegal actions, wasting model parameters that should discriminate among legal actions.

### Root Cause 3: Expert Data Already Encodes Legality

Since the expert (MCTS) only picks legal actions:
- Action A appears 1000 times when state has pattern X → Model learns P(A|X) is high
- Action B appears 0 times when state has pattern X → Model learns P(B|X) is low

**The absence of illegal actions in training data already teaches the model what's legal.** Additional masking prevents this natural learning.

### Root Cause 4: Class Weights Conflict with Masking

- Class weights balance action frequencies across the entire dataset
- Masks change which actions are valid per sample
- These create opposing gradient directions
- Optimization becomes chaotic

### Root Cause 5: Training/Validation Inconsistency

`train.py:300` passes `class_weights` to `compute_total_loss()`:
```python
total_loss, per_head_losses = compute_total_loss(outputs, labels, class_weights, masks)
```

`train.py:440` (validation) does NOT pass `class_weights`:
```python
total_loss, per_head_losses = compute_total_loss(outputs, labels, legal_masks=masks)
```

Training and validation are computing **different loss functions**, making validation metrics unreliable.

## Relevant Files

Use these files to fix the bug:

- **src/imitation_learning/train.py** (primary fix location)
  - Line 71-93: `apply_legal_action_mask()` - Remove or deprecate this function
  - Line 96-132: `compute_conditional_loss()` - Remove `legal_masks` parameter and masking logic
  - Line 135-252: `compute_total_loss()` - Remove `class_weights` and `legal_masks` parameters
  - Line 255-400: `train_one_epoch()` - Remove class_weights logic, add masked accuracy computation
  - Line 403-526: `validate()` - Add masked accuracy computation
  - Line 600-774: `main()` - Remove class weight computation entirely

- **src/imitation_learning/config.yaml** (configuration updates)
  - Line 38: `use_class_weights: false` - Verify this is false (already correct)

- **src/imitation_learning/configs/config_small.yaml**
  - Line 32: `use_class_weights: false` - Verify this is false (already correct)

- **src/imitation_learning/configs/config_medium.yaml**
  - Line 32: `use_class_weights: false` - Verify this is false (already correct)

- **src/imitation_learning/configs/config_large.yaml**
  - Line 32: `use_class_weights: false` - Verify this is false (already correct)

- **src/imitation_learning/utils.py** (add helper for masked predictions)
  - Add new function: `compute_masked_predictions()` to apply masks before argmax
  - Existing `compute_accuracy()` at line 324 already supports masking - no changes needed

### New Files

- **src/imitation_learning/utils.py** (new function only, not a new file)
  - Add `compute_masked_predictions(logits, legal_masks)` helper function

## Step by Step Tasks

### Step 1: Add masked prediction helper to utils.py

Add a new utility function that applies legal action masks before computing predictions:
- Function should accept logits tensor and legal_masks tensor
- Apply strong negative mask (-1e10) to illegal actions
- Return argmax predictions over masked logits
- This will be used ONLY for accuracy computation, NOT for loss

### Step 2: Remove masking from loss computation in train.py

Update `compute_conditional_loss()`:
- Remove `legal_masks` parameter entirely
- Remove the `if legal_masks is not None:` block (lines 122-123)
- Function should only compute cross_entropy on sample-masked logits
- Remove the call to `apply_legal_action_mask()`

Update `compute_total_loss()`:
- Remove `class_weights` parameter entirely
- Remove `legal_masks` parameter entirely
- Remove all calls to `apply_legal_action_mask()` (lines 163, throughout function)
- Remove class weight arguments from all `F.cross_entropy()` calls
- Simplify to pure conditional loss based on action types and valid labels

### Step 3: Remove class weight computation from main()

Update `main()` function:
- Remove all code related to `compute_class_weights()` (it's defined but should not be called)
- Ensure class_weights is never passed to training or validation functions
- Verify config's `use_class_weights: false` is respected

### Step 4: Add mask-aware accuracy computation to train_one_epoch()

Update accuracy calculation in `train_one_epoch()`:
- For each prediction head, apply masks BEFORE argmax
- Use the new `compute_masked_predictions()` helper
- Example: `action_type_preds = compute_masked_predictions(outputs['action_type'], masks['action_type'])`
- Apply this pattern to all 7 prediction heads
- This ensures training accuracy reflects real-world performance (where masks are applied)

**Important**: This happens AFTER loss computation and backprop, so it doesn't affect gradients.

### Step 5: Add mask-aware accuracy computation to validate()

Update accuracy calculation in `validate()`:
- Apply same masked prediction logic as in training
- Use `compute_masked_predictions()` for all heads
- Ensures train/val accuracy metrics are comparable

### Step 6: Update train_one_epoch() signature to remove class_weights

- Remove `class_weights` parameter from function signature (line 260)
- Remove `class_weights` argument from `compute_total_loss()` call (line 300)
- Remove any references to class_weights in the function

### Step 7: Verify configuration files

Check all config files to ensure `use_class_weights: false`:
- src/imitation_learning/config.yaml
- src/imitation_learning/configs/config_small.yaml
- src/imitation_learning/configs/config_medium.yaml
- src/imitation_learning/configs/config_large.yaml

All should have `use_class_weights: false` (already correct based on reads).

### Step 8: Add code comments explaining the design decision

Add documentation to key functions:
- Document in `compute_total_loss()` why masking is NOT applied during loss
- Document in accuracy computation why masking IS applied there
- Add comments explaining that expert data only contains legal actions
- Reference this design pattern for future maintainers

### Step 9: Run validation tests

Execute validation commands to ensure:
- Code runs without errors
- Training starts successfully
- No crashes during forward/backward pass
- Metrics are logged correctly to wandb
- Training and validation use identical loss computation

## Validation Commands

Execute every command to validate the bug is fixed with zero regressions.

### Pre-fix validation (reproduce the bug):
```bash
# Check current mask usage in loss computation
cd /home/apprentyr/projects/splendor_ia
grep -n "legal_masks" src/imitation_learning/train.py | head -20

# Verify class weights configuration
grep -n "use_class_weights" src/imitation_learning/configs/*.yaml
```

### Post-fix validation (verify the fix):

```bash
# 1. Verify no masking in loss computation
cd /home/apprentyr/projects/splendor_ia
grep -A 5 "def compute_conditional_loss" src/imitation_learning/train.py
# Should NOT contain legal_masks parameter

# 2. Verify compute_total_loss signature
grep -A 10 "def compute_total_loss" src/imitation_learning/train.py
# Should NOT contain class_weights or legal_masks parameters

# 3. Verify masked predictions are used for accuracy
grep -B 2 -A 2 "compute_masked_predictions" src/imitation_learning/train.py
# Should appear in both train_one_epoch and validate functions

# 4. Run a quick training test (1 epoch)
cd /home/apprentyr/projects/splendor_ia
timeout 300 uv run python -m src.imitation_learning.train \
    --config src/imitation_learning/configs/config_small.yaml 2>&1 | head -100
# Should complete one epoch without errors, show consistent train/val loss computation

# 5. Verify no class weights are being used
grep -n "class_weights" src/imitation_learning/train.py
# Should only appear in the compute_class_weights function definition (not called)

# 6. Check that masks are still generated and passed to dataloader
grep -n "masks" src/imitation_learning/dataset.py
# Should show masks are still loaded and returned by dataset

# 7. Verify config files
cat src/imitation_learning/configs/config_small.yaml | grep use_class_weights
# Should show: use_class_weights: false

# 8. Run full training for 5 epochs to verify convergence improvement
cd /home/apprentyr/projects/splendor_ia
timeout 600 uv run python -m src.imitation_learning.train \
    --config src/imitation_learning/configs/config_small.yaml 2>&1 | tee /tmp/training_output.log
# Should show:
# - Smooth loss decrease
# - Consistent train/val losses
# - Improving accuracy
# - No NaN or inf values
# - Training completes without errors
```

## Notes

### Why This Fix Works

1. **Expert data contains only legal actions** - The model naturally learns P(illegal_action|state) ≈ 0 from the data distribution, without explicit masking needed during training.

2. **Masking at inference provides safety** - Even though the model learns legality from data, applying masks during prediction (argmax) guarantees illegal actions are never selected.

3. **No conflicting optimization signals** - Removing class weights and loss-time masking creates a clean, unambiguous optimization objective.

4. **Consistent train/val behavior** - Both loops use identical loss computation, making validation metrics reliable indicators of generalization.

5. **Clean gradient flow** - No corrupted gradients through large negative values; model learns meaningful logit values for all actions.

### Why Masked Accuracy Doesn't Hurt Training

**Question**: Can we apply masking to accuracy during training without impacting the loss?

**Answer**: **YES, absolutely!** Here's why:

```python
# Forward pass
outputs = model(states)

# Backward pass (loss computation) - NO MASKING
loss = F.cross_entropy(outputs['action_type'], labels['action_type'])
loss.backward()  # Gradients computed
optimizer.step()  # Weights updated

# Accuracy computation (after optimizer.step()) - YES MASKING
masked_logits = outputs['action_type'] + (1 - masks['action_type']) * -1e10
preds = masked_logits.argmax(dim=1)  # This is just evaluation, no gradients
accuracy = (preds == labels).float().mean()  # Just a metric
```

The key insight: **Accuracy is computed AFTER the backward pass**. PyTorch doesn't track gradients through the accuracy calculation because:
1. It happens after `optimizer.step()`
2. It's wrapped in a context where gradients aren't needed
3. It's just numpy operations in the current code

So yes, you can and should apply masks for accuracy to get comparable train/val metrics without affecting the loss or gradients at all.

### Performance Expectations

After this fix, you should observe:
- **Faster convergence** - Loss decreases smoothly within first 5-10 epochs
- **Stable training** - Low variance in metrics across epochs
- **Better accuracy** - Improved performance compared to masked training
- **Consistent train/val metrics** - Similar loss curves, predictable generalization gap
- **Higher final accuracy** - Model reaches better performance ceiling

### Testing Strategy

1. **Short test** (5 epochs, config_small.yaml) - Verify no crashes, basic convergence
2. **Medium test** (20 epochs, config_medium.yaml) - Verify full convergence behavior
3. **Compare to baseline** - If you have old runs without masking, new runs should match or exceed them
4. **Ablation study** - If curious, try adding back ONLY class weights or ONLY masking to confirm they each hurt performance

### Future Considerations

If illegal actions still appear in predictions after this fix (unlikely), consider:
1. **Data quality** - Verify expert demonstrations are truly optimal
2. **State representation** - Ensure state features contain enough information to determine legality
3. **Model capacity** - Increase model size if underfitting
4. **Inference-time masking** - Already included as safety mechanism in this fix

Do NOT re-introduce masking during training unless you have strong empirical evidence it's needed after trying this fix.
