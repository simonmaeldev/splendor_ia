# Bug: Gems Removed Evaluation Returns 0 Samples

## Bug Description
The `src/imitation_learning/evaluate.py` script reports 0 gems removed samples when evaluating the model, even though the test split contains 5,370 samples (9.94%) requiring gem removal. The evaluation output shows:
```
gems_removed: acc=0.0000, samples=0
```

However, analysis of the test data confirms gems_removed is not empty:
- Actions requiring gem removal: 5,370 (9.94% of 54,031 samples)
- Actions without gem removal: 48,661 (90.06%)

The `scripts/play_with_model.py` demonstrates that the model correctly predicts gems_removed using the `decode_gems_removed()` function from `action_decoder.py`, but the evaluation script doesn't properly evaluate this head.

## Problem Statement
The `evaluate_conditional_head()` function in `evaluate.py` is designed to evaluate heads that are conditional on specific action types (BUILD, RESERVE, TAKE2, TAKE3). However, gems_removed is NOT conditional on a single action type - it can occur with ANY action type when the player has > 10 tokens after taking an action.

The current implementation at lines 409-413 in `evaluate.py` uses a placeholder that doesn't actually evaluate the gems_removed head:

```python
# Gems removed (overflow)
gems_rem_results = {'accuracy': 0.0, 'num_samples': 0}  # Placeholder
results['gems_removed'] = gems_rem_results
print(f"  gems_removed: acc={gems_rem_results['accuracy']:.4f}, "
      f"samples={gems_rem_results['num_samples']}")
```

## Solution Statement
Create a new evaluation function `evaluate_gems_removed_head()` that:
1. Evaluates gems_removed across ALL action types (not filtered by a specific action_type)
2. Only evaluates samples where `gems_removed != 0` (i.e., where removal was actually required)
3. Applies legal action masking for gems_removed predictions
4. Computes accuracy by comparing predictions to ground truth labels

This approach mirrors how the model is used in production (in `action_decoder.py` and `ai_player.py`), where gems_removed is decoded unconditionally and then validated.

## Steps to Reproduce
1. Run evaluation on the test split:
   ```bash
   uv run python -m imitation_learning.evaluate --checkpoint data/models/202511250809_config_tuning/best_model.pth --config data/models/202511250809_config_tuning/config.yaml --split test
   ```

2. Observe output shows:
   ```
   gems_removed: acc=0.0000, samples=0
   ```

3. Verify gems_removed data exists in test split:
   ```bash
   uv run python << 'EOF'
   import numpy as np
   labels_test = np.load('data/processed/labels_test.npz')
   gems_removed = labels_test['gems_removed']
   actions_with_removal = np.sum(gems_removed > 0)
   print(f"Actions requiring gem removal: {actions_with_removal:,}")
   EOF
   ```

   Output: `Actions requiring gem removal: 5,370`

## Root Cause Analysis
The root cause is that `evaluate.py` treats gems_removed as a conditional head similar to card_selection (only applies to BUILD) or gem_take3 (only applies to TAKE3). However, gems_removed is fundamentally different:

1. **gems_removed is universal**: It can apply to ANY of the 4 action types (BUILD, RESERVE, TAKE2, TAKE3)
2. **gems_removed is conditional on token overflow**: It only applies when `sum(player_tokens_after_action) > 10`
3. **The current placeholder doesn't evaluate anything**: Lines 410 hardcode the results to `{'accuracy': 0.0, 'num_samples': 0}`

The proper way to evaluate gems_removed is shown in `action_decoder.py:383-387`:
```python
# Check if token removal is needed (sum > 10)
tokens_to_remove = decode_gems_removed(predictions.get('gems_removed', 0))
if sum(tokens_after_action) > MAX_NB_TOKENS:
    tokens_to_remove = validate_token_removal(tokens_after_action, tokens_to_remove)
else:
    tokens_to_remove = NO_TOKENS
```

This shows gems_removed should be evaluated on samples where removal is actually needed, regardless of action type.

## Relevant Files
Use these files to fix the bug:

### src/imitation_learning/evaluate.py
- Line 106-171: `evaluate_conditional_head()` - existing function for conditional heads
- Line 409-413: **BUG LOCATION** - Placeholder that needs replacement
- Line 452: gems_removed accuracy is included in visualization but always 0.0
- Need to add new function `evaluate_gems_removed_head()` before line 174

**Why relevant**: This is where the bug exists and where the fix needs to be implemented.

### src/imitation_learning/train.py
- Contains `apply_legal_action_mask()` function used by evaluate.py
- Shows how masking is applied during training

**Why relevant**: The new evaluation function needs to use the same masking approach for consistency.

### src/splendor/action_decoder.py
- Line 120-141: `decode_gems_removed()` - shows how to decode class indices
- Line 239-290: `validate_token_removal()` - shows validation logic
- Line 383-387: Shows conditional logic for when gems_removed applies

**Why relevant**: Demonstrates the correct logic for when gems_removed is relevant and how it should be processed.

## Step by Step Tasks

### Step 1: Create evaluate_gems_removed_head function
- Add new function `evaluate_gems_removed_head()` in `src/imitation_learning/evaluate.py` after `evaluate_conditional_head()` (around line 173)
- Function signature: `def evaluate_gems_removed_head(model: nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device, enable_masking: bool = True) -> Dict:`
- Loop through dataloader batches
- For each sample, check if `labels['gems_removed'][i] != 0` (indicating removal was needed)
- Collect predictions and labels only for samples requiring removal
- Apply legal action masking to gems_removed logits
- Compute accuracy using `compute_accuracy()`
- Return dict with `accuracy`, `num_samples`, and `frequency` keys

### Step 2: Replace placeholder in main() function
- Remove lines 409-413 (the placeholder code)
- Replace with call to `evaluate_gems_removed_head()`:
  ```python
  gems_rem_results = evaluate_gems_removed_head(model, dataloader, device)
  results['gems_removed'] = gems_rem_results
  print(f"  gems_removed: acc={gems_rem_results['accuracy']:.4f}, "
        f"samples={gems_rem_results['num_samples']}")
  ```

### Step 3: Test the fix
- Run evaluation on test split and verify gems_removed now shows correct sample count
- Verify accuracy is computed (non-zero value)
- Check that results.json contains proper gems_removed metrics

## Validation Commands
Execute every command to validate the bug is fixed with zero regressions.

```bash
# Before fix: Confirm bug exists
uv run python -m imitation_learning.evaluate --checkpoint data/models/202511250809_config_tuning/best_model.pth --config data/models/202511250809_config_tuning/config.yaml --split test 2>&1 | grep "gems_removed:"

# After fix: Verify gems_removed evaluation works
uv run python -m imitation_learning.evaluate --checkpoint data/models/202511250809_config_tuning/best_model.pth --config data/models/202511250809_config_tuning/config.yaml --split test 2>&1 | grep "gems_removed:"

# Verify sample count is approximately 5,370 (9.94% of 54,031)
uv run python << 'EOF'
import json
with open('logs/evaluation_results_test.json', 'r') as f:
    results = json.load(f)
    gems_removed = results['gems_removed']
    print(f"gems_removed accuracy: {gems_removed['accuracy']:.4f}")
    print(f"gems_removed samples: {gems_removed['num_samples']}")
    print(f"gems_removed frequency: {gems_removed.get('frequency', 0):.4f}")
    assert gems_removed['num_samples'] > 5000, f"Expected ~5,370 samples, got {gems_removed['num_samples']}"
    assert gems_removed['num_samples'] < 6000, f"Expected ~5,370 samples, got {gems_removed['num_samples']}"
    print("✓ gems_removed evaluation is working correctly!")
EOF

# Verify no regressions in other heads
uv run python << 'EOF'
import json
with open('logs/evaluation_results_test.json', 'r') as f:
    results = json.load(f)
    required_heads = ['action_type', 'card_selection', 'card_reservation', 'gem_take3', 'gem_take2', 'noble', 'gems_removed']
    for head in required_heads:
        assert head in results, f"Missing head: {head}"
        assert 'accuracy' in results[head], f"Missing accuracy for {head}"
        print(f"✓ {head}: acc={results[head]['accuracy']:.4f}")
    print("✓ All heads evaluated successfully!")
EOF
```

## Notes
- The gems_removed head is unique because it's the only head that can apply to any action type
- During training, gems_removed labels are populated regardless of action type, but only when player tokens > 10 after the action
- The label value 0 in gems_removed means "no removal needed", not "invalid/not applicable" like -1 in other heads
- This bug was hidden because the placeholder looked like a valid result (dict with accuracy and num_samples keys)
- The fix is minimal: just need to add one new evaluation function and replace the placeholder call
