# Feature: Game State Reconstruction from CSV Input Features

## Feature Description
This feature implements a robust system to reconstruct the internal Python game state (Board object) from CSV input features. The reconstructed board state must be functionally equivalent to the original game state, meaning it can generate the same valid moves using the existing `board.getMoves()` function. This is Phase 1 of implementing illegal action masking for training and evaluation.

The reconstruction process reverses the encoding performed by `csv_exporter.py`, transforming 382 CSV features back into the nested object structure used by the game simulation (Board, Player, Card, Character objects).

## User Story
As a machine learning engineer training a Splendor AI
I want to reconstruct the internal game state from CSV input features
So that I can call `board.getMoves()` to get valid actions and mask illegal predictions during training and evaluation

## Problem Statement
Currently, the ML pipeline can only predict actions from input features but cannot validate whether predictions are legal according to game rules. The game simulation has a `board.getMoves()` function that returns all valid moves for a given state, but this requires the internal Board object representation, not the flat CSV format.

The CSV export process (simulation → CSV) is one-way. We need the reverse transformation (CSV → simulation) to:
1. Generate valid action masks for each state
2. Filter illegal predictions during inference
3. Validate that training data actions are actually legal
4. Enable rule-aware evaluation metrics

## Solution Statement
We implement a reconstruction module (`src/utils/state_reconstruction.py`) that:
1. Parses 382 CSV input features into structured dictionaries
2. Creates Card, Character, Player, and Board objects matching the internal representation
3. Uses dummy cards for deck contents (only counts matter for valid move generation)
4. Handles NaN padding for variable player counts (2-4 players)
5. Validates reconstruction by:
   - Calling `board.getMoves()` successfully (no errors)
   - Confirming the CSV action is in the valid moves list
   - Round-trip testing (reconstruct → encode → compare)

The module never modifies `src/splendor/*` files - all validation confirms our reconstruction is correct, not that the game logic needs changes.

## Relevant Files

### Existing Files (READ ONLY - DO NOT MODIFY)
- `src/splendor/board.py` - Board class with `getMoves()` function we'll call
  - Contains game state: tokens, displayedCards, decks, characters, players
  - Line 121-140: `getMoves()` returns all valid moves for current state

- `src/splendor/player.py` - Player class with state attributes
  - Contains: tokens, built, reserved, characters, vp, reductions

- `src/splendor/cards.py` - Card class definition
  - Attributes: vp, bonus, cost, lvl, visible

- `src/splendor/characters.py` - Character (noble) class definition
  - Attributes: vp, cost

- `src/splendor/move.py` - Move class representing actions
  - Attributes: actionType, action, tokensToRemove, character

- `src/splendor/constants.py` - Game constants (colors, action types, deck definitions)
  - Lines 3-8: Color indices (WHITE=0, BLUE=1, GREEN=2, RED=3, BLACK=4, GOLD=5)
  - Lines 16-18: Action types (BUILD=0, RESERVE=1, TOKENS=2)
  - Lines 77-166: DECK1, DECK2, DECK3 card definitions

- `src/splendor/csv_exporter.py` - Forward encoding (simulation → CSV)
  - Lines 36-62: `encode_card()` - 12 features per card
  - Lines 65-82: `encode_noble()` - 6 features per noble
  - Lines 176-220: `encode_player_state_from_board()` - 49 features per player
  - Lines 223-300: `encode_game_state_from_board()` - 382 features total
  - Lines 339-444: `encode_action_from_move()` - 20 output features

- `data/games/3_games/3869.csv` - Example CSV file with 82 rows, 403 columns

### New Files

#### Core Reconstruction Module
- `src/utils/__init__.py` - Package initialization
- `src/utils/state_reconstruction.py` - Main reconstruction module
  - `parse_card_features(features: List[float]) -> Optional[Card]`
  - `parse_noble_features(features: List[float]) -> Optional[Character]`
  - `parse_player_features(features: List[float], player_num: int) -> Optional[Player]`
  - `create_dummy_deck(level: int, count: int) -> List[Card]`
  - `reconstruct_board_from_csv_row(row: Dict[str, Any]) -> Board`
  - `reconstruct_board_from_features(features: List[float]) -> Board`

#### Validation Module
- `src/utils/validate_reconstruction.py` - Validation utilities
  - `parse_action_from_csv(row: Dict[str, Any]) -> Move`
  - `validate_action_in_moves(board: Board, expected_action: Move) -> Tuple[bool, str]`
  - `encode_board_state(board: Board, turn_num: int) -> List[float]`
  - `validate_round_trip(original_features: List[float], board: Board, turn_num: int) -> Dict[str, Any]`
  - `compare_features(original: List[float], reconstructed: List[float], tolerance: float = 1e-6) -> Dict[str, Any]`

#### Testing & Scripts
- `scripts/test_reconstruction.py` - Comprehensive test suite
  - Load sample CSV rows from each player count (2/3/4 players)
  - Test basic reconstruction (no errors)
  - Test getMoves() can be called
  - Test action validation (CSV action is in valid moves)
  - Test round-trip (reconstruct → encode → compare)
  - Generate validation report with pass/fail statistics

- `scripts/batch_validate_reconstruction.py` - Batch validation across dataset
  - Sample N games from each player count
  - Run all validations
  - Report success rate and failure analysis

#### Documentation
- `docs/state_reconstruction_guide.md` - Usage guide and architecture documentation

## Implementation Plan

### Phase 1: Foundation (Parsing & Object Creation)
Set up the utilities module structure and implement parsers for the atomic game objects (Card, Character). These parsers must handle NaN values correctly and create objects matching the exact structure expected by the game simulation.

### Phase 2: Core Implementation (Board Reconstruction)
Implement player and board reconstruction, handling the complexities of player rotation, NaN padding, and dummy deck creation. The reconstructed Board must be functionally equivalent to the original state for the purpose of calling `getMoves()`.

### Phase 3: Validation & Testing
Implement comprehensive validation to ensure reconstruction correctness. This includes verifying that `getMoves()` works, the CSV action is valid, and round-trip encoding produces matching features.

## Step by Step Tasks

### Setup and Structure

#### Create utilities package structure
- Run `mkdir -p src/utils` to create utilities directory
- Create `src/utils/__init__.py` as empty file to make it a package
- This module is separate from splendor core logic and ML code

### Core Parsing Functions (`src/utils/state_reconstruction.py`)

#### Implement card parsing from 12 features
- Write `parse_card_features(features: List[float]) -> Optional[Card]` that:
  - Takes 12 features: [vp, level, cost_white, cost_blue, cost_green, cost_red, cost_black, bonus_white, bonus_blue, bonus_green, bonus_red, bonus_black]
  - Returns None if features are all NaN (empty slot)
  - Extracts vp (int), level (int), cost list (5 ints)
  - Finds bonus index from one-hot encoding (0-4)
  - Creates `Card(vp, bonus, cost, level)` object
  - Sets card.visible = True (per user requirement)
  - Returns Card object
- Add type hints and docstring with example
- Handle edge case: cost might have float values like 2.0, convert to int

#### Implement noble parsing from 6 features
- Write `parse_noble_features(features: List[float]) -> Optional[Character]` that:
  - Takes 6 features: [vp, req_white, req_blue, req_green, req_red, req_black]
  - Returns None if features are all NaN (empty noble slot)
  - Extracts vp (int), cost list (5 ints)
  - Creates `Character(vp, cost)` object
  - Returns Character object
- Add type hints and docstring
- Handle float to int conversion

#### Implement dummy deck creation
- Write `create_dummy_deck(level: int, count: int) -> List[Card]` that:
  - Creates `count` dummy Card objects for the specified level
  - Dummy cards have: vp=0, bonus=0, cost=[0,0,0,0,0], lvl=level
  - Sets visible=False (cards in deck are not visible)
  - Returns list of dummy cards
  - Purpose: `getMoves()` likely checks `if self.decks[level]` to allow top-deck reservation
- Add docstring explaining why dummy cards are sufficient

#### Implement player parsing from 49 features
- Write `parse_player_features(features: List[float], player_num: int) -> Optional[Player]` that:
  - Takes 49 features: [position, vp, gems(6), reductions(5), reserved_cards(3x12)]
  - Returns None if features are all NaN (player doesn't exist in this game)
  - Extracts position (int, 0-3, should match player_num)
  - Creates `Player(str(player_num), "ISMCTS_PARA1000")` object
  - Sets player.tokens to gems list (6 ints)
  - Sets player.vp to vp (int)
  - Sets player.reductions to reductions list (5 ints)
  - Parses 3 reserved cards using `parse_card_features()` for each 12-feature chunk
  - Adds non-None cards to player.reserved list
  - Returns Player object
- Note: We don't reconstruct player.built cards (not in CSV, not needed for getMoves)
- Add type hints and comprehensive docstring

### Board Reconstruction (`src/utils/state_reconstruction.py`)

#### Implement main reconstruction function from dict
- Write `reconstruct_board_from_csv_row(row: Dict[str, Any]) -> Board` that:
  - Takes a dictionary with column names as keys (from CSV reader)
  - Extracts num_players from row['num_players']
  - Extracts turn_number from row['turn_number']
  - Extracts current_player from row['current_player']
  - Creates Board object: `Board(num_players, ["ISMCTS_PARA1000"] * num_players)`
  - Sets board.nbTurn = turn_number
  - Sets board.currentPlayer = current_player
  - Reconstructs board state (tokens, cards, nobles, players)
  - Returns Board object
- Implement each reconstruction step in helper functions (see below)
- Add comprehensive docstring explaining the reconstruction process

#### Reconstruct board tokens
- Extract gems_board_white through gems_board_gold from row
- Create list of 6 integers: [white, blue, green, red, black, gold]
- Set board.tokens = gems_list
- Handle potential float values (convert to int)

#### Reconstruct visible cards (12 cards: 4 per level)
- For each level (1-3):
  - For each position (0-3):
    - Calculate card index: level * 4 + position (0-11)
    - Extract 12 features for card{index}_*
    - Parse using `parse_card_features()`
    - If card is not None, add to board.displayedCards[level]
- Result: board.displayedCards is List[List[Card]] with 3 sublists
- Some sublists may have < 4 cards if deck is depleted

#### Reconstruct deck counts
- Extract deck_level1_remaining, deck_level2_remaining, deck_level3_remaining
- For each level (1-3):
  - Create dummy deck using `create_dummy_deck(level, count)`
  - Set board.deckLVL1, board.deckLVL2, board.deckLVL3
  - Set board.decks = [board.deckLVL1, board.deckLVL2, board.deckLVL3]
- Ensure decks are lists, not other iterables

#### Reconstruct nobles
- For each noble position (0-4):
  - Extract 6 features for noble{i}_*
  - Parse using `parse_noble_features()`
  - If noble is not None, add to board.characters list
- Result: board.characters is List[Character] with 0-5 nobles

#### Reconstruct players with rotation handling
- Extract num_players to know how many players exist
- The CSV has players rotated so current player is first (player0_*)
- We need to un-rotate to get absolute player positions
- For each player slot (0-3):
  - Extract 49 features for player{i}_*
  - Parse using `parse_player_features()`
  - If player is not None:
    - Get absolute position from player{i}_position
    - Place player at board.players[absolute_position]
- Initialize board.players with correct size before placing
- Validate that current_player index is valid

#### Implement reconstruction from feature list
- Write `reconstruct_board_from_features(features: List[float]) -> Board` that:
  - Takes flat list of 382 features
  - Creates dictionary mapping feature names to values
  - Uses column headers from csv_exporter.generate_input_column_headers()
  - Calls `reconstruct_board_from_csv_row(row_dict)`
  - Returns Board object
- This function is for testing and programmatic use (no CSV file needed)

### Action Parsing (`src/utils/validate_reconstruction.py`)

#### Implement action parsing from CSV output columns
- Write `parse_action_from_csv(row: Dict[str, Any]) -> Move` that:
  - Extracts action_type string ("build", "reserve", "take 2 tokens", "take 3 tokens")
  - Maps to constant: BUILD, RESERVE, or TOKENS
  - For BUILD actions:
    - Extract card_selection (0-14)
    - Determine if from visible (0-11) or reserved (12-14)
    - Create Move with action=Card (need to reconstruct card reference)
  - For RESERVE actions:
    - Extract card_reservation (0-14)
    - If 12-14: top deck reservation (action = deck level)
    - If 0-11: visible card reservation (action = Card reference)
  - For TOKENS actions:
    - Extract gem_take3_* or gem_take2_* based on action_type
    - Create token list [white, blue, green, red, black, gold]
  - Extract gems_removed_* for tokensToRemove
  - Extract noble_selection (0-4 or -1)
  - If noble_selection >= 0, get Character reference from board
  - Create Move(actionType, action, tokensToRemove, character)
  - Returns Move object
- This is complex: need board reference to get Card/Character objects
- Add comprehensive docstring with examples for each action type

#### Handle card/character reference resolution
- For card references (BUILD from visible, RESERVE from visible):
  - Use card position to index into board.displayedCards
  - Level 1 cards: positions 0-3
  - Level 2 cards: positions 4-7
  - Level 3 cards: positions 8-11
  - Reserved cards: positions 12-14 (index into current player's reserved list)
- For noble references:
  - Use noble_selection index to get from board.characters[index]
- Handle edge cases: card/noble might not exist (validation will catch this)

### Validation Functions (`src/utils/validate_reconstruction.py`)

#### Implement getMoves() validation
- Write `validate_action_in_moves(board: Board, expected_action: Move) -> Tuple[bool, str]` that:
  - Calls `valid_moves = board.getMoves()` (may raise exception if reconstruction is wrong)
  - Wraps in try/except to catch reconstruction errors
  - If exception: return (False, f"getMoves() failed: {error}")
  - If no exception: check if expected_action is in valid_moves
  - Use Move.__eq__() for comparison (already implemented in move.py)
  - If found: return (True, "Action is valid")
  - If not found: return (False, f"Action not in {len(valid_moves)} valid moves")
- Add detailed logging of what was compared
- Return success boolean and descriptive message

#### Implement round-trip encoding
- Write `encode_board_state(board: Board, turn_num: int) -> List[float]` that:
  - Calls the forward encoding function from csv_exporter
  - `encode_game_state_from_board(board, turn_num)`
  - Returns 382 features
  - This tests that our reconstruction produces the same features
- Import from csv_exporter or replicate encoding logic

#### Implement feature comparison
- Write `compare_features(original: List[float], reconstructed: List[float], tolerance: float = 1e-6) -> Dict[str, Any]` that:
  - Compares two feature lists element-by-element
  - Handles NaN comparisons correctly (math.isnan() for both)
  - Handles integer vs float (2.0 == 2)
  - Tracks indices where features differ
  - Returns dict with:
    - 'match': bool (all features match within tolerance)
    - 'num_diffs': int (count of differing features)
    - 'diff_indices': List[int] (indices where features differ)
    - 'diff_details': List[Tuple[int, float, float]] (index, original, reconstructed)
- Add tolerance parameter for floating point comparison

#### Implement full round-trip validation
- Write `validate_round_trip(original_features: List[float], board: Board, turn_num: int) -> Dict[str, Any]` that:
  - Encodes the reconstructed board using `encode_board_state()`
  - Compares with original features using `compare_features()`
  - Returns comprehensive report dict
  - This confirms reconstruction is exact (no information loss)

### Testing Scripts

#### Implement basic reconstruction test (`scripts/test_reconstruction.py`)
- Write main() function that:
  - Loads a sample CSV file (e.g., data/games/3_games/3869.csv)
  - Reads first 10 rows
  - For each row:
    - Reconstructs board using `reconstruct_board_from_csv_row()`
    - Prints basic board info (turn, current_player, num_players)
    - Calls `board.getMoves()` to verify it works
    - Prints number of valid moves
  - Reports success/failure for each row
- Add CLI arguments for CSV file path
- Make script executable with `if __name__ == "__main__"`

#### Implement action validation test
- Add to test_reconstruction.py:
  - Parse action from CSV using `parse_action_from_csv()`
  - Call `validate_action_in_moves()`
  - Print whether CSV action is in valid moves
  - Log warnings if action is not valid (indicates data issue or reconstruction bug)

#### Implement round-trip test
- Add to test_reconstruction.py:
  - Extract original input features from CSV row
  - Reconstruct board
  - Encode board back to features
  - Compare using `validate_round_trip()`
  - Print detailed diff report if features don't match
  - Track pass rate across all rows

#### Implement batch validation script (`scripts/batch_validate_reconstruction.py`)
- Write script that:
  - Samples N games from each player count directory (e.g., 10 games each)
  - For each game:
    - Load all rows
    - Run all three validations (reconstruction, action, round-trip)
    - Track pass/fail statistics
  - Generate summary report:
    - Total rows tested
    - Reconstruction success rate (getMoves() works)
    - Action validation success rate (CSV action is valid)
    - Round-trip success rate (features match)
    - List of failing games for debugging
  - Save report to logs/reconstruction_validation_report.txt
- Add CLI arguments for sample size, player counts to test
- Add verbose flag for detailed output

### Documentation

#### Create usage guide (`docs/state_reconstruction_guide.md`)
- Write guide with sections:
  - Overview: What reconstruction does and why
  - Architecture: Module structure and key functions
  - Usage examples:
    - Reconstruct from CSV row dict
    - Reconstruct from feature list
    - Validate a single reconstruction
    - Batch validation
  - Implementation details:
    - How player rotation is handled
    - Why dummy cards are sufficient
    - NaN handling strategy
  - Troubleshooting:
    - Common reconstruction errors
    - How to debug failed validations
  - Limitations:
    - Built cards not reconstructed (not needed for getMoves)
    - Exact deck contents unknown (only counts)
    - Reserved card visibility always True
- Include code examples for each usage pattern

### Integration & Validation

#### Test with 2-player games
- Run test_reconstruction.py on sample 2-player game
- Verify player3 and player4 features are NaN and ignored
- Verify board.players has length 2
- Verify getMoves() works correctly
- Verify round-trip encoding matches

#### Test with 3-player games
- Run test_reconstruction.py on sample 3-player game
- Verify player3 exists, player4 is NaN
- Verify board.players has length 3
- Verify all validations pass

#### Test with 4-player games
- Run test_reconstruction.py on sample 4-player game
- Verify all 4 players exist
- Verify all validations pass

#### Test edge cases
- Test with depleted decks (late game states):
  - Verify displayedCards has < 4 cards per level
  - Verify deck_remaining = 0 creates empty dummy deck
  - Verify getMoves() still works (no top-deck reservation available)
- Test with maximum tokens (player has 10 tokens):
  - Verify token reconstruction is correct
  - Verify getMoves() restricts to BUILD actions or token removal
- Test with reserved cards:
  - Verify reserved cards are reconstructed
  - Verify BUILD from reserved is in valid moves
- Test first turn (turn_number = 1):
  - Verify empty built cards, empty reserved
  - Verify getMoves() returns TOKENS actions

#### Run batch validation across dataset
- Run batch_validate_reconstruction.py with sample size = 20 per player count
- Generate validation report
- Target: 100% success rate for all validations
- If failures occur:
  - Investigate root cause (data issue vs reconstruction bug)
  - Fix reconstruction logic or document data limitations
  - Re-run validation until 100% pass rate achieved

#### Document validation results
- Add section to state_reconstruction_guide.md:
  - Validation results summary
  - Known issues or limitations discovered
  - Edge cases that required special handling
  - Confidence level in reconstruction correctness

## Testing Strategy

### Unit Tests

#### Test Card Parsing
- Test parse_card_features() with:
  - Valid card features (vp=1, level=2, costs, bonus)
  - All NaN features (should return None)
  - Edge case: vp=0, level=1 (valid level 1 card)
  - Verify one-hot bonus decoding (each bonus value 0-4)
  - Verify visible flag is set to True

#### Test Noble Parsing
- Test parse_noble_features() with:
  - Valid noble features (vp=3, costs)
  - All NaN features (should return None)
  - Various cost combinations

#### Test Player Parsing
- Test parse_player_features() with:
  - Valid player with no reserved cards
  - Player with 1-3 reserved cards
  - Player with tokens and reductions
  - All NaN features (should return None)
  - Verify player.vp and player.reductions are set correctly

#### Test Dummy Deck Creation
- Test create_dummy_deck() with:
  - Each level (1, 2, 3)
  - Various counts (0, 1, 10, 40)
  - Verify all cards have correct level
  - Verify visible=False

### Integration Tests

#### Test Complete Board Reconstruction
- Test reconstruct_board_from_csv_row() with:
  - Complete CSV row from 2-player game
  - Complete CSV row from 3-player game
  - Complete CSV row from 4-player game
  - Verify all board attributes are set correctly
  - Verify player rotation is handled correctly

#### Test Action Parsing
- Test parse_action_from_csv() with:
  - BUILD action from visible card (positions 0-11)
  - BUILD action from reserved card (positions 12-14)
  - RESERVE action from visible card
  - RESERVE action from top deck (positions 12-14)
  - TAKE 3 tokens action
  - TAKE 2 tokens action
  - Action with noble selection
  - Action with gems removed

#### Test getMoves() Execution
- For reconstructed boards:
  - Verify getMoves() can be called without errors
  - Verify it returns a non-empty list
  - Verify returned moves have correct structure

#### Test Round-Trip Encoding
- For sample rows:
  - Reconstruct board
  - Encode back to features
  - Compare with original
  - Verify 100% feature match (within tolerance for floats)

### Edge Cases

#### Data Edge Cases
- Empty decks (deck_remaining = 0):
  - Verify dummy deck list is empty []
  - Verify getMoves() doesn't include top-deck reservation
- Depleted card rows (< 4 cards per level):
  - Verify displayedCards sublists have correct length
  - Verify NaN card slots are skipped
- Missing nobles (< 5 nobles):
  - Verify board.characters has correct count
  - Verify NaN noble slots are skipped
- Early game (turn 1):
  - No built cards, no reserved cards
  - All players have 0 VP, 0 reductions
- Late game (turn 20+):
  - Players have many built cards (not in CSV, not reconstructed)
  - High VP values
  - Depleted decks

#### Reconstruction Edge Cases
- Player rotation edge cases:
  - current_player = 0 (no rotation needed)
  - current_player = 3 (maximum rotation)
  - Verify absolute positions are preserved
- Float vs int handling:
  - CSV might have 2.0 instead of 2
  - Verify conversion to int doesn't cause issues
- NaN handling:
  - Verify math.isnan() check works correctly
  - Verify NaN propagation doesn't break object creation

#### Validation Edge Cases
- Action not in valid moves (data issue):
  - Should return False with descriptive message
  - Log for investigation but don't crash
- getMoves() raises exception:
  - Catch and report as reconstruction failure
  - Don't let exception crash validation script
- Round-trip mismatch:
  - Small floating point errors (acceptable within tolerance)
  - Genuine data differences (indicates bug)

## Acceptance Criteria

1. **Reconstruction Module Complete**
   - `src/utils/state_reconstruction.py` implements all parsing functions
   - Can reconstruct Board from CSV row dict
   - Can reconstruct Board from feature list
   - Handles NaN padding for all player counts (2-4)
   - Creates dummy decks with correct counts
   - No modifications to `src/splendor/*` files

2. **Validation Module Complete**
   - `src/utils/validate_reconstruction.py` implements all validation functions
   - Can parse actions from CSV output columns
   - Can validate action is in valid moves list
   - Can perform round-trip encoding test
   - Can compare features with NaN-aware logic

3. **getMoves() Works Correctly**
   - Reconstructed boards can call `board.getMoves()` without errors
   - Returns non-empty list of Move objects
   - For 100% of test cases across all player counts

4. **Action Validation Passes**
   - CSV action can be parsed into Move object
   - Parsed action is found in `board.getMoves()` list
   - Target: 95%+ validation rate (some data may have issues)
   - Any failures are logged for investigation

5. **Round-Trip Encoding Matches**
   - Reconstruct → Encode produces same 382 features
   - Comparison accounts for:
    - NaN equality (math.isnan() for both)
    - Float vs int (2.0 == 2)
    - Floating point tolerance (1e-6)
   - Target: 100% feature match for all test cases
   - Any mismatches indicate reconstruction bugs

6. **Comprehensive Testing**
   - Test script runs on all player counts (2, 3, 4)
   - Batch validation samples diverse games
   - Edge cases tested (empty decks, turn 1, late game)
   - Validation report generated with statistics

7. **Code Quality**
   - All functions have type hints
   - All functions have docstrings with examples
   - Error handling for invalid input
   - Logging for debugging
   - No hardcoded paths (use Path objects)

8. **Documentation Complete**
   - Usage guide explains architecture and usage
   - Examples for common use cases
   - Troubleshooting section
   - Validation results documented

## Validation Commands
Execute every command to validate the feature works correctly with zero regressions.

- `python scripts/test_reconstruction.py --csv data/games/2_games/*.csv --sample 1` - Test 2-player reconstruction
- `python scripts/test_reconstruction.py --csv data/games/3_games/3869.csv` - Test 3-player reconstruction (all rows)
- `python scripts/test_reconstruction.py --csv data/games/4_games/*.csv --sample 1` - Test 4-player reconstruction
- `python scripts/batch_validate_reconstruction.py --sample-size 20 --player-counts 2 3 4` - Batch validation across all player counts
- `cat logs/reconstruction_validation_report.txt` - Review validation results
- `python -c "from src.utils.state_reconstruction import reconstruct_board_from_csv_row; from src.splendor.board import Board; print('Imports successful')"` - Verify imports work
- `python -c "from src.utils.validate_reconstruction import validate_action_in_moves; print('Validation imports successful')"` - Verify validation imports

## Notes

### Key Design Decisions

1. **Dummy Decks**: We create dummy cards only to satisfy `getMoves()` checks like `if self.decks[level]`. The actual card properties don't matter because:
   - Top-deck reservation doesn't require knowing the specific card
   - Valid moves only check if deck is non-empty
   - Actual card will be drawn randomly during gameplay (not relevant for ML)

2. **Player Rotation**: CSV has players rotated so current player is always player0. We must un-rotate to get absolute positions because:
   - player{i}_position contains the absolute player number (0-3)
   - Board.players is indexed by absolute position
   - Board.currentPlayer is an absolute index
   - This ensures `board.getCurrentPlayer()` returns the correct player

3. **Reserved Cards Always Visible**: Setting `card.visible = True` for all reserved cards simplifies reconstruction because:
   - `getMoves()` doesn't require visibility information for reserved cards
   - Visibility matters for ISMCTS determinization (which we're not doing)
   - The ML model can see all reserved cards in the CSV anyway

4. **Built Cards Not Reconstructed**: We don't reconstruct `player.built` list because:
   - Not present in CSV input features
   - Not needed for `getMoves()` calculation
   - Player VP and reductions are reconstructed directly (sufficient)

5. **No Modifications to Game Logic**: We never modify `src/splendor/*` because:
   - If `getMoves()` fails, the bug is in our reconstruction, not the game
   - The game logic is battle-tested from thousands of simulations
   - Our goal is to match the internal representation, not change it

### CSV Data Cleaning Notes

The user mentioned that CSV data has been cleaned by:
- `scripts/fix_csv_gem_encoding.py` - Fixed gem encoding issues
- `scripts/clean_csv_nan_values.py` - Cleaned NaN value representation

This means the original raw CSV may have had inconsistencies that were fixed post-export. Our reconstruction works with the cleaned CSV format. If round-trip comparison fails due to encoding differences, this is acceptable as long as:
1. `getMoves()` works correctly
2. Action validation passes
3. The semantic meaning is preserved

### Future Work (Part 2)

This module provides the foundation for Part 2, which will:
- Use `board.getMoves()` to generate action masks during training
- Filter illegal predictions during inference
- Implement masked loss functions
- Add masking to evaluation metrics

The reconstruction module will be called for each training/eval batch to generate masks on-the-fly.

### Performance Considerations

Reconstruction will be called for every training sample (potentially millions of times). Optimizations to consider:
- Cache parsed cards/nobles within a batch (many states share same visible cards)
- Vectorize reconstruction if possible (reconstruct batch of boards at once)
- Profile bottlenecks after initial implementation

However, correctness is the priority for Part 1. Performance optimization can come later if needed.
