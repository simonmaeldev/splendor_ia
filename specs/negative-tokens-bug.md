# Bug: Negative Tokens Error in Model Player Actions

## Bug Description
When playing games with the AI model player, the game crashes with a "negatif tokens" error during token-taking actions. The error occurs in the `removeTooManyTokens` method when the AI player attempts to remove more tokens of a specific color than it possesses after taking new tokens.

**Symptoms:**
- Game crashes mid-play with exception: "negatif tokens"
- Occurs specifically during TOKENS actions (TAKE2/TAKE3)
- The error message shows the player trying to remove tokens they don't have

**Expected behavior:**
- The model should only predict legal token removal combinations
- The game should execute moves without errors
- Token counts should never go negative

**Actual behavior:**
- Model predicts illegal token removals
- Game crashes with "negatif tokens" exception
- Example from error log:
  - Player has: `[4, 3, 1, 1, 1, 0]` tokens
  - Tries to take: `[0, 1, 1, 0, 1, 0]` → results in `[4, 4, 2, 1, 2, 0]`
  - Tries to remove: `[0, 0, 0, 2, 1, 0]` → would result in `[4, 4, 2, -1, 1, 0]` ❌

## Problem Statement
The AI model's predicted `gems_removed` action is not being properly validated or constrained to ensure that the player actually has the tokens to remove AFTER taking new tokens. The legal move masking system in `imitation_learning/utils.py` generates masks for token removal based on the current token count, but the action decoder doesn't validate that the removal is still valid after tokens are taken.

## Solution Statement
Add validation in the action decoder to ensure that predicted token removals are actually possible given the player's token count after taking new tokens. If the prediction is invalid, fall back to a safe removal pattern (e.g., removing gems in order of abundance until under the 10-token limit).

## Steps to Reproduce
1. Run: `uv run python scripts/play_with_model.py --model data/models/202511241159_config_tuning/best_model.pth --games 1 --display`
2. Observe game play until a TOKENS action occurs
3. Game will eventually crash with "negatif tokens" error when model predicts an invalid token removal

## Root Cause Analysis

### Execution Flow
1. Model predicts action_type=TOKENS (TAKE2 or TAKE3)
2. Model predicts gems to take (e.g., `[0, 1, 1, 0, 1, 0]`)
3. Model predicts gems to remove (e.g., `[0, 0, 0, 2, 1, 0]`)
4. Action decoder converts predictions to Move object without validation
5. Game executes:
   - `takeTokens()`: Adds tokens to player
   - `removeTooManyTokens()`: Tries to remove tokens → **CRASHES if player doesn't have enough**

### Root Cause
The action decoder (`src/splendor/action_decoder.py`) in the `decode_predictions_to_move` function does not validate that the predicted token removal is valid given the player's token count AFTER taking new tokens. The masking system generates masks based on the CURRENT token count, but by the time tokens are removed, the player has already taken new tokens, which may change what removals are valid.

### Why This Happens
1. Legal move masks are generated based on current state
2. The `gems_removed` mask is based on current player tokens
3. But tokens are TAKEN first, THEN removed
4. The removal that was legal before taking tokens may be illegal after
5. The model has no way to learn this dependency during training

### Affected Code
- `src/splendor/action_decoder.py:242-324` - `decode_predictions_to_move()` function
- Specifically lines 315-322 (TAKE2) and 320-324 (TAKE3)
- No validation of tokens_to_remove against player's final token count

## Relevant Files
Use these files to fix the bug:

### Existing Files
- **`src/splendor/action_decoder.py`** - Contains the `decode_predictions_to_move()` function that needs validation logic added to ensure token removals are valid after tokens are taken.
- **`src/splendor/board.py`** - Contains the game logic for token management (`takeTokens`, `removeTooManyTokens`) to understand the execution order and validation rules.
- **`src/splendor/constants.py`** - Contains game constants like `MAX_NB_TOKENS` (10) needed for validation logic.

## Step by Step Tasks

### 1. Add helper function to validate and fix token removal
- Create a new function `validate_token_removal()` in `src/splendor/action_decoder.py`
- Function should take: `tokens_after_taking` (player tokens after taking), `tokens_to_remove` (predicted removal)
- Function should return: validated `tokens_to_remove` list
- Logic:
  - Check if predicted removal is valid (no negative tokens)
  - If valid, return as-is
  - If invalid, generate a safe removal pattern:
    - Calculate how many tokens need to be removed (total - MAX_NB_TOKENS)
    - Remove tokens in order of abundance, starting with most plentiful colors
    - Never remove more tokens of a color than the player has

### 2. Update TAKE2 action decoding
- In `decode_predictions_to_move()`, when `action_type == 2` (TAKE2)
- Calculate player's tokens after taking: `player_tokens_after = add(player.tokens, tokens)`
- Validate removal: `tokens_to_remove = validate_token_removal(player_tokens_after, tokens_to_remove)`
- Create Move with validated removal

### 3. Update TAKE3 action decoding
- In `decode_predictions_to_move()`, when `action_type == 3` (TAKE3)
- Calculate player's tokens after taking: `player_tokens_after = add(player.tokens, tokens)`
- Validate removal: `tokens_to_remove = validate_token_removal(player_tokens_after, tokens_to_remove)`
- Create Move with validated removal

### 4. Test the fix with actual gameplay
- Run the original failing command with `--display` flag
- Verify no "negatif tokens" errors occur
- Check that token removals are always valid
- Verify game completes successfully

## Validation Commands
Execute every command to validate the bug is fixed with zero regressions.

- `uv run python scripts/play_with_model.py --model data/models/202511241159_config_tuning/best_model.pth --games 10` - Run 10 games to ensure no "negatif tokens" errors occur
- `uv run python scripts/play_with_model.py --model data/models/202511241159_config_tuning/best_model.pth --games 1 --display` - Run 1 game with display to manually verify token removals are valid
- `uv run python -c "from src.splendor.action_decoder import decode_predictions_to_move, validate_token_removal; print('Import successful')"` - Verify the new function can be imported
- `uv run python scripts/test_model_player.py data/models/202511241159_config_tuning/best_model.pth --games 5` - Run with different test script if available

## Notes
- The fix should be minimal and surgical - only add validation to token removal, don't change other logic
- Use the existing `add()` and `substract()` helper functions from `src/splendor/custom_operators.py` for token arithmetic
- Import `MAX_NB_TOKENS` from `src/splendor/constants.py` (value is 10)
- The validation function should be defensive and always return a valid removal, even if the model prediction is completely wrong
- This is a runtime fix; the model training pipeline may need to be updated separately to prevent the model from learning invalid patterns
