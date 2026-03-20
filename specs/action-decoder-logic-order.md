# Bug: Action Decoder Logic Order Mismatch

## Bug Description
The `decode_predictions_to_move` function in `src/splendor/action_decoder.py` does not follow the same logical structure as `doMove` in `src/splendor/board.py`. Specifically:

1. **Noble selection logic is incorrect**: The function currently calls `decode_noble_selection` unconditionally only for BUILD actions (line 340), but it should check for available nobles AFTER ANY action is performed. This is a bug fix for the core game logic - nobles should be checkable after any action (BUILD/RESERVE/TOKENS), not just BUILD.

2. **Token removal validation is incorrect**: The function currently validates token removal for TAKE2/TAKE3 actions (lines 368, 379) but doesn't properly validate for BUILD and RESERVE actions. The check should only happen if the player would have more than 10 tokens after the action.

3. **Missing gold token logic for RESERVE**: When reserving a card, a player automatically receives a gold token from the board if gold tokens are available (as seen in `board.py:291`). This affects the token count calculation for determining if token removal is needed.

4. **No simulation on copy**: The decoder doesn't simulate actions on a deep copy of the board state, risking potential side effects.

**Expected behavior**: Match the structure of `doMove` in `board.py` where:
- Main action (BUILD/RESERVE/TOKENS) is executed first (on a deep copy for simulation)
- Token removal check happens second (only if player has > 10 tokens)
- Noble selection happens last (for ALL action types, only if nobles are available to the player)

**Actual behavior**: The decoder checks nobles only for BUILD, validates token removal prematurely, doesn't use deep copies for simulation, and doesn't account for state changes from the main action.

## Problem Statement
The action decoder must match the game logic flow in `board.py:104-119` to correctly predict which nobles are available and whether token removal is needed based on the post-action game state, not the pre-action state.

## Solution Statement
Restructure `decode_predictions_to_move` to:
1. Create a deep copy of the board to simulate actions without side effects
2. Simulate the main action (BUILD/RESERVE/TAKE_TOKENS) on the copy to determine the player's token state and bonus after the action
3. Check if token removal is needed (sum of tokens > 10) and only then validate the token removal prediction
4. Calculate the player's bonus after ANY action and check if any nobles are available, only calling the noble head if nobles exist
5. Ensure RESERVE actions account for the gold token that will be automatically taken
6. Return a Move object based on the simulated state, not the pre-action state

## Steps to Reproduce
1. Train a model that uses the action decoder
2. Have the model predict a BUILD action where:
   - The player doesn't have enough bonus to attract any nobles before building
   - The noble head predicts a noble selection anyway
3. Observe that the decoder creates a Move with an invalid noble
4. Similarly, have the model predict actions where token removal is checked even when the player has ≤ 10 tokens after the action

## Root Cause Analysis
The root cause is a mismatch between the training data generation (which follows `board.py` logic) and the inference-time action decoding. The `board.py:doMove` function executes actions in this order:

```python
# board.py:104-119
def doMove(self, move: Move) -> None:
    if move.actionType == BUILD:
        self.build(move)
    elif move.actionType == RESERVE:
        self.reserve(move)
    elif move.actionType == TOKENS:
        self.takeTokens(move)

    self.removeTooManyTokens(move)  # Line 117 - happens AFTER main action
    self.takeCharacter(move)        # Line 118 - happens AFTER removal
    self.nextPlayer()
```

The `reserve` function also shows that gold tokens are automatically taken:
```python
# board.py:285-291
def reserve(self, move: Move) -> None:
    card = self.getCard(move)
    self.removeCard(move)
    player = self.getCurrentPlayer()
    player.reserved.append(card)
    player.updateStateAfterReserve()
    if self.tokens[GOLD]: self.takeTokens(Move(move.actionType, TAKEONEGOLD, move.tokensToRemove, move.character))
```

The action decoder currently:
1. Only checks nobles for BUILD actions (should check for ALL actions)
2. Checks nobles and token removal based on the state BEFORE the action, not after
3. Doesn't use deep copies when simulating state changes

## Relevant Files
Use these files to fix the bug:

### Existing Files
- **`src/splendor/action_decoder.py`** (lines 297-382) - Main file to fix
  - Contains `decode_predictions_to_move` function that needs restructuring
  - Lines 338-347: BUILD action handling - needs to calculate bonus after build and check noble availability
  - Lines 349-359: RESERVE action handling - needs to account for gold token addition
  - Lines 361-381: TAKE2/TAKE3 handling - token removal logic is correct but needs to be applied to all action types

- **`src/splendor/board.py`** (lines 104-119, 285-291) - Reference implementation
  - Shows the correct order: main action → token removal → noble selection
  - Lines 200-206: `makeMovesBuild` shows how to check for available nobles using the post-build bonus
  - Line 291: Shows gold token is automatically taken when reserving

- **`src/splendor/player.py`** (lines 185-186) - Helper methods
  - `getTotalBonus()` method used to calculate player's current bonus
  - Returns a copy of the reductions list representing card bonuses

- **`src/splendor/characters.py`** - Noble/Character class definition
  - Characters have `cost` attribute representing required bonuses
  - Used to determine if a player can attract a noble

- **`src/splendor/custom_operators.py`** (line 8) - Utility functions
  - `substract()` function used for calculating cost differences
  - Used in noble availability check

- **`src/splendor/constants.py`** - Game constants
  - `MAX_NB_TOKENS = 10` - Token limit that triggers removal requirement
  - `GOLD = 5`, `TAKEONEGOLD = [0,0,0,0,0,1]` - Gold token constants
  - Action type constants: `BUILD = 0`, `RESERVE = 1`, `TOKENS = 2`

## Step by Step Tasks

### Step 1: Add imports for deep copy functionality
- Ensure `from copy import deepcopy` is imported (already present in board.py but verify in action_decoder.py)
- Add import if not present

### Step 2: Refactor the main decode function structure
- Create a deep copy of the board at the start: `board_copy = board.clone()` or use `deepcopy(board)`
- Use `board.clone()` method since it's available from the Board class
- All simulations should happen on `board_copy`, not the original `board`

### Step 3: Refactor BUILD action handling
- Remove the unconditional `decode_noble_selection` call on line 340
- Simulate the BUILD action on the board copy
- Calculate the player's bonus AFTER building the card on the copy
- Calculate tokens after paying for the card (subtract the real gold cost)
- Move noble checking to the end (Step 6)

### Step 4: Refactor RESERVE action handling
- Simulate the RESERVE action on the board copy
- Account for the automatic gold token addition (if `board.tokens[GOLD] > 0`)
- Calculate `tokens_after_action` by adding `[0,0,0,0,0,1]` to current tokens if gold is available
- Move noble checking to the end (Step 6)

### Step 5: Apply consistent token removal validation across all action types
- For ALL action types (BUILD/RESERVE/TOKENS), calculate `tokens_after_action` based on the simulated state
- Check if `sum(tokens_after_action) > 10`
- Only validate token removal predictions if this condition is true
- If the player has ≤ 10 tokens after the action, set `tokens_to_remove = NO_TOKENS` and ignore the gems_removed head
- Use `validate_token_removal(tokens_after_action, tokens_to_remove)` for validation

### Step 6: Add noble checking for ALL action types
- After simulating the main action, extract the current player's bonus from the board copy
- Use the pattern from `board.py:200-206` to check for available nobles:
  ```python
  current_player_bonus = board_copy.getCurrentPlayer().getTotalBonus()
  possible_nobles = list(filter(lambda c: all(color >= 0 for color in substract(current_player_bonus, c.cost)), board_copy.characters))
  ```
- Only call `decode_noble_selection` if `possible_nobles` is not empty
- Set `noble = None` if no nobles are available or if the list is empty
- Apply this check to BUILD, RESERVE, and TOKENS action types

### Step 7: Update TAKE2/TAKE3 to include noble checking
- Keep the existing token removal validation logic
- Add noble checking after calculating `tokens_after_taking`
- Ensure pattern is consistent with BUILD/RESERVE

### Step 8: Create comprehensive tests
- Test BUILD action with no available nobles before build but noble available after build
- Test RESERVE action attracting a noble (edge case but possible)
- Test TOKENS action attracting a noble (should return None since tokens don't change bonus)
- Test RESERVE action with gold tokens available (token count = 10 before reserve)
- Test all action types with token count at exactly 10 and 11 to verify threshold behavior
- Test token removal validation is skipped when player has ≤ 10 tokens after action

### Step 9: Run validation commands
- Execute all validation commands to ensure no regressions
- Verify the decoder now matches the `doMove` structure from `board.py`
- Verify deep copies prevent side effects on the original board

## Validation Commands
Execute every command to validate the bug is fixed with zero regressions.

- `cd /home/apprentyr/projects/splendor_ia && uv run python src/splendor/action_decoder.py` - Run the action decoder's built-in tests to verify basic functionality
- `cd /home/apprentyr/projects/splendor_ia && uv run python -c "from src.splendor.action_decoder import decode_predictions_to_move; from src.splendor.board import Board; b = Board(2, [None, None]); predictions = {'action_type': 0, 'card_selection': 0, 'card_reservation': 0, 'gem_take3': 0, 'gem_take2': 0, 'noble': 0, 'gems_removed': 0}; move = decode_predictions_to_move(predictions, b); print('✓ Decoder works with basic BUILD action')"` - Test basic BUILD action decoding
- `cd /home/apprentyr/projects/splendor_ia && uv run python -c "from src.splendor.action_decoder import decode_predictions_to_move; from src.splendor.board import Board; b = Board(2, [None, None]); b.tokens[5] = 5; predictions = {'action_type': 1, 'card_selection': 0, 'card_reservation': 0, 'gem_take3': 0, 'gem_take2': 0, 'noble': 0, 'gems_removed': 0}; move = decode_predictions_to_move(predictions, b); print('✓ Decoder works with RESERVE action with gold available')"` - Test RESERVE with gold tokens
- `cd /home/apprentyr/projects/splendor_ia && uv run python -c "from src.splendor.action_decoder import decode_predictions_to_move; from src.splendor.board import Board; b = Board(2, [None, None]); b.getCurrentPlayer().tokens = [2,2,2,2,2,0]; predictions = {'action_type': 2, 'card_selection': 0, 'card_reservation': 0, 'gem_take3': 0, 'gem_take2': 0, 'noble': 0, 'gems_removed': 0}; move = decode_predictions_to_move(predictions, b); print('✓ Decoder works with TAKE2 requiring token removal')"` - Test TAKE2 with token overflow
- `cd /home/apprentyr/projects/splendor_ia && uv run pytest tests/` - Run all unit tests if they exist (optional, will gracefully fail if no tests directory exists)

## Notes
- **CRITICAL**: The bug fix requires careful simulation of game state changes without actually modifying the board state - ALWAYS use `board.clone()` or `deepcopy(board)` before simulating
- **CRITICAL**: Noble availability should be checked for ALL action types (BUILD/RESERVE/TOKENS), not just BUILD - this fixes a bug in the core game logic
- Use `board.clone()` method since Board class provides this functionality (see `board.py:52-64`)
- Performance optimization: Extract `getCurrentPlayer().getTotalBonus()` from the board copy after simulating the action
- The noble availability check uses the same lambda filter pattern from `board.py:204`: `filter(lambda c: all(color >= 0 for color in substract(bonus, c.cost)), self.characters)`
- Gold token constant is `TAKEONEGOLD = [0,0,0,0,0,1]` from `constants.py:9`
- Token limit is `MAX_NB_TOKENS = 10` from `constants.py:20`
- The validation should use `sum(tokens) > 10`, not `sum(tokens) >= 10` (based on `board.py:136` and `player.py:207`)
- This is a surgical fix - only modify the logic order in `decode_predictions_to_move`, don't change the individual decoder functions
- No new libraries needed - all required imports are already in the file
- For TOKENS actions, noble check will always return empty list since taking tokens doesn't change bonus, but we check anyway for consistency
