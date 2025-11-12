# Feature: ML Dataset Export Script

## Feature Description
Create a Python script that extracts game state and action data from the SQLite database and exports it to a CSV file formatted for machine learning training. The script transforms raw game data into a standardized 381-dimensional input vector representing the complete board state, paired with multi-head output vectors representing the action taken. The export includes progress tracking, data validation, and comprehensive documentation of the dataset structure.

## User Story
As a machine learning engineer
I want to export game data from the database into a properly formatted CSV file
So that I can train a neural network model to play Splendor using supervised learning from MCTS gameplay

## Problem Statement
The current database stores game states and actions in a relational format optimized for storage and querying, but machine learning models require fixed-dimensional numerical vectors. The transformation from relational data to ML-ready format is complex, involving:
- Ordering players relative to the current player
- Encoding cards with consistent color ordering
- Handling variable player counts (2-4 players) with NaN padding
- Converting actions into multi-head classification targets
- Maintaining consistency across multiple encoding schemes (cards, nobles, gems)

## Solution Statement
Implement a standalone Python script (`scripts/export_ml_dataset.py`) that:
1. Queries all game states and actions from the SQLite database
2. For each state-action pair, constructs a 381-dimensional input vector with proper ordering and NaN padding
3. Encodes the corresponding action into 7 output heads for multi-head classification
4. Writes the data to CSV with clear column headers
5. Displays real-time progress with percentage completion
6. Provides a summary report explaining the dataset structure and statistics

## Relevant Files

### Existing Files to Reference

- **`scripts/create_database.py`** (lines 1-166)
  - Defines the complete database schema
  - Documents all tables: Game, StateGame, StatePlayer, Action, Card, Character
  - Essential for understanding data structure and foreign key relationships

- **`src/splendor/constants.py`** (lines 1-188)
  - Defines color constants: WHITE=0, BLUE=1, GREEN=2, RED=3, BLACK=4, GOLD=5
  - Defines action type constants: BUILD=0, RESERVE=1, TOKENS=2
  - Contains all 40 cards (DECK1, DECK2, DECK3) and 10 nobles (CHARACTERS)
  - Critical for consistent color ordering throughout the export

- **`src/splendor/board.py`** (lines 1-357)
  - Shows how game state is structured during gameplay
  - Documents visible cards layout: 3 levels × 4 cards = 12 cards
  - Explains token distribution and deck management

- **`src/splendor/player.py`** (lines 1-188)
  - Defines player state structure: tokens, built cards, reserved cards, characters
  - Shows bonus/reduction calculation from owned cards
  - Documents maximum reserve limit (MAX_RESERVE = 3)

- **`src/splendor/cards.py`** (lines 1-36)
  - Card structure: vp, bonus, cost (5 colors), lvl
  - Cards have visibility flag for reserved cards

- **`src/splendor/characters.py`** (lines 1-23)
  - Character/Noble structure: vp (always 3), cost (5 color requirements)

- **`scripts/main.py`** (lines 66-102)
  - Shows how actions are currently saved to database
  - Documents action encoding: type, card, take tokens, give tokens, character
  - Essential for understanding how to decode actions from database

- **`README.md`** (lines 134-207)
  - Complete database schema documentation
  - Explains entity relationships and foreign keys
  - Documents dataset statistics and game counts

### New Files

- **`scripts/export_ml_dataset.py`**
  - Main export script with all transformation logic
  - Database querying and data extraction
  - Feature vector construction (379 dimensions)
  - Action encoding (7 output heads)
  - Progress tracking and reporting
  - CSV generation with proper headers

## Implementation Plan

### Phase 1: Foundation
Set up the script structure with database connection, constant definitions, and helper functions for color ordering, player rotation, and NaN handling. This establishes the core utilities needed for data transformation.

### Phase 2: Core Implementation
Implement the main data extraction logic:
- Query all game states and actions from database
- Build the 379-dimensional input vector for each state
- Encode actions into 7 output heads
- Handle edge cases (missing players, empty decks, missing nobles)

### Phase 3: Integration
Add progress tracking with real-time percentage updates, CSV export with clear column headers, and final summary report explaining dataset structure and providing statistics.

## Step by Step Tasks

### 1. Create script foundation

- Create `scripts/export_ml_dataset.py` with imports and constants
- Import necessary modules: `sqlite3`, `csv`, `sys`, `typing`, `pathlib`
- Import game constants from `src/splendor/constants.py` (color indices, action types)
- Define color order list: `COLOR_ORDER = [WHITE, BLUE, GREEN, RED, BLACK, GOLD]` for consistency
- Define database path constant: `DB_PATH = 'data/games.db'`
- Define output CSV path: `OUTPUT_PATH = 'data/training_dataset.csv'`

### 2. Implement database helper functions

- Create `connect_to_database(db_path: str) -> sqlite3.Connection` function
- Create `get_card_by_id(cursor, card_id) -> dict` to fetch card details (vp, bonus, cost, level)
- Create `get_character_by_id(cursor, char_id) -> dict` to fetch noble details (vp, cost)
- Create `get_total_state_count(cursor) -> int` to query total number of state-action pairs for progress tracking
- Create `load_all_cards_into_memory(cursor) -> dict` to cache all cards keyed by IDCard
- Create `load_all_characters_into_memory(cursor) -> dict` to cache all nobles keyed by IDCharacter
- Add error handling for database connection failures

### 3. Implement card encoding function

- Create `encode_card(card: dict) -> list[float]` function returning 12 features:
  - Victory points (int)
  - Card level (int)
  - Cost for each color in COLOR_ORDER (5 ints, excluding gold)
  - Bonus as one-hot vector (5 values: 0 or 1)
- If card is None, return list of 12 NaN values
- Add unit test helper to verify encoding produces exactly 12 values
- Ensure color ordering matches COLOR_ORDER consistently

### 4. Implement noble encoding function

- Create `encode_noble(noble: dict) -> list[float]` function returning 6 features:
  - Victory points (always 3)
  - Cost requirements for each color in COLOR_ORDER (5 ints, excluding gold)
- If noble is None, return list of 6 NaN values
- Verify encoding produces exactly 6 values

### 5. Implement player state extraction and encoding

- Create `get_player_built_cards(cursor, game_id, player_num, turn_num) -> list[dict]`
  - Query all cards built by player up to and including this turn
  - Join Action table with Card table where Type=BUILD
- Create `get_player_reserved_cards(cursor, game_id, player_num, turn_num) -> list[dict]`
  - Track reservations (Type=RESERVE) and builds from reserve
  - Return currently reserved cards at this turn (max 3)
- Create `calculate_player_reductions(built_cards: list[dict]) -> list[int]`
  - Count bonus of each color from built cards
  - Return list of 5 ints (one per non-gold color)
- Create `calculate_player_victory_points(built_cards, characters) -> int`
  - Sum VP from cards and nobles
- Create `encode_player_state(cursor, game_id, player_num, turn_num, turn_position: int, state_player_row, cards_cache, chars_cache) -> list[float]` returning 49 features:
  - Turn position relative to current player (0=current, 1=next, etc.)
  - Total victory points
  - Gems possessed (6 ints in COLOR_ORDER)
  - Reductions per color (5 ints, excluding gold)
  - 3 reserved cards × 12 features = 36 features (use NaN for missing slots)

### 6. Implement game state encoding function

- Create `rotate_players_by_current(player_numbers: list[int], current_player: int) -> list[int]`
  - Reorder player list so current player is first
  - Example: players=[0,1,2,3], current=2 → returns [2,3,0,1]
- Create `encode_game_state(cursor, state_game_row, state_player_rows, cards_cache, chars_cache) -> list[float]` returning 381 features:
  - Extract basic info: nb_players, turn_number, current_player from StateGame
  - Encode number of players (1 int)
  - Encode current turn number (1 int)
  - Encode gems on board (6 ints from TokensWhite through TokensGold)
  - Encode 12 visible cards using Card1_1 through Card3_4 (12 × 12 = 144 features)
    - Order by level: level 1 cards first, then level 2, then level 3
    - Use NaN if card slot is NULL (deck exhausted)
  - Encode number of cards left in each deck (3 ints for levels 1, 2, 3)
    - Query: `SELECT COUNT(*) FROM Action WHERE IDGame=? AND TurnNumber<? AND IDCard IN (SELECT IDCard FROM Card WHERE Level=?)`
    - Subtract from initial deck size
  - Encode 5 nobles using Character1 through Character5 (5 × 6 = 30 features)
    - Use NaN if character slot is NULL
  - Encode 4 players (4 × 49 = 196 features)
    - Rotate player order using `rotate_players_by_current`
    - Fill missing players (for 2-3 player games) with NaN
- Verify total is exactly 381 features
- Add assertion to check length

### 7. Implement action encoding function

- Create `encode_action_type(action_type: int) -> str`
  - BUILD=0 → "build"
  - RESERVE=1 → "reserve"
  - TOKENS=2 with sum(take)==2 → "take 2 tokens"
  - TOKENS=2 with sum(take)==3 → "take 3 tokens"
- Create `encode_card_selection(action_row, cards_cache) -> int`
  - Used for BUILD action from visible cards
  - Card position: 4×(level-1) + position_in_level (0-3) = [0-11]
  - Top deck reserve: not applicable for build (can't build from top deck)
  - Reserved card build: 12 + position_in_reserved (0-2) = [12-14]
  - Return -1 or NaN if not applicable
- Create `encode_card_reservation(action_row, cards_cache) -> int`
  - Used for RESERVE action
  - Visible card: 4×(level-1) + position_in_level = [0-11]
  - Top deck: 12 + level (1-3) = [13-15]
  - Return -1 or NaN if not applicable
- Create `encode_gem_take_3(take_tokens: list[int]) -> list[int]`
  - Return 5-element list (excluding gold)
  - Each element is 0 or 1 (3-hot vector)
  - Return [NaN]*5 if not a take-3 action
- Create `encode_gem_take_2(take_tokens: list[int]) -> list[int]`
  - Return 5-element list (excluding gold)
  - Each element is 0 or 2 (one-hot vector)
  - Return [NaN]*5 if not a take-2 action
- Create `encode_noble_selection(character_id: int, state_game_row, chars_cache) -> int`
  - Find position of character in Character1-Character5 fields
  - Return position (0-4)
  - Return -1 or NaN if no character selected
- Create `encode_gems_removed(give_tokens: list[int]) -> list[int]`
  - Return 6-element list (including gold) in COLOR_ORDER
  - Each element is count of gems removed for that color
  - Return [0]*6 if no gems removed
- Create `encode_action(action_row, state_game_row, cards_cache, chars_cache) -> dict`
  - Return dictionary with all 7 output heads:
    - 'action_type': string
    - 'card_selection': int (for build)
    - 'card_reservation': int (for reserve)
    - 'gem_take_3': list[int] (5 elements)
    - 'gem_take_2': list[int] (5 elements)
    - 'noble_selection': int
    - 'gems_removed': list[int] (6 elements)

### 8. Implement main data extraction loop

- Create `extract_all_training_data(db_path: str) -> list[tuple]`
  - Connect to database
  - Load all cards and characters into memory for faster lookup
  - Query total count of state-action pairs for progress tracking
  - Query all games with JOIN across Game, StateGame, StatePlayer, Action tables
  - For each turn in each game:
    - Get StateGame row for this turn
    - Get all StatePlayer rows for this turn (one per player)
    - Get all Action rows for this turn (one per player)
    - For each player's action:
      - Build 381-dim input vector using `encode_game_state`
      - Build action output using `encode_action`
      - Yield (game_id, input_vector, action_dict)
  - Return list of all training samples

### 9. Implement CSV column header generation

- Create `generate_input_column_headers() -> list[str]`
  - Generate 381 descriptive column names:
    - "num_players"
    - "turn_number"
    - "gems_board_white", "gems_board_blue", ..., "gems_board_gold" (6 columns)
    - For 12 visible cards: "card{i}_vp", "card{i}_level", "card{i}_cost_white", ..., "card{i}_bonus_white", ... (144 columns)
    - "deck_level1_remaining", "deck_level2_remaining", "deck_level3_remaining" (3 columns)
    - For 5 nobles: "noble{i}_vp", "noble{i}_req_white", ... (30 columns)
    - For 4 players: "player{i}_position", "player{i}_vp", "player{i}_gems_white", ..., "player{i}_reduction_white", ..., "player{i}_reserved{j}_vp", ... (196 columns)
  - Verify total is 381
- Create `generate_output_column_headers() -> list[str]`
  - Return column names for outputs:
    - "action_type" (1 column)
    - "card_selection" (1 column)
    - "card_reservation" (1 column)
    - "gem_take3_white", "gem_take3_blue", ..., "gem_take3_black" (5 columns)
    - "gem_take2_white", "gem_take2_blue", ..., "gem_take2_black" (5 columns)
    - "noble_selection" (1 column)
    - "gems_removed_white", ..., "gems_removed_gold" (6 columns)
  - Total: 20 output columns
- Create `generate_all_headers() -> list[str]`
  - Return: ["game_id"] + input headers + output headers
  - Total: 1 + 381 + 20 = 402 columns

### 10. Implement progress tracking

- Create `ProgressTracker` class with methods:
  - `__init__(self, total: int)` - initialize with total number of items
  - `update(self, current: int)` - update progress
  - `display(self)` - print progress bar with percentage (overwrite same line)
- Use ANSI escape codes to update same terminal line: `\r` for carriage return
- Display format: `"Progress: [=====>          ] 45.2% (123456/273890)"`
- Update every 100 rows to avoid excessive terminal writes

### 11. Implement CSV export function

- Create `write_to_csv(data: list[tuple], output_path: str)`
  - Open CSV file for writing
  - Write header row using `generate_all_headers()`
  - Initialize progress tracker
  - For each (game_id, input_vec, action_dict) in data:
    - Flatten action_dict into list matching output column order
    - Combine: [game_id] + input_vec + action_outputs
    - Write row to CSV
    - Update progress tracker
  - Close CSV file
- Handle NaN values: write as empty string or "NaN" (decide based on ML library preference)

### 12. Implement dataset summary function

- Create `generate_dataset_summary(data: list[tuple], output_path: str) -> str`
  - Calculate statistics:
    - Total number of training samples
    - Number of unique games
    - Distribution of action types (count each type)
    - Distribution of player counts (2, 3, 4 players)
    - Percentage of NaN values in input features
  - Generate human-readable report:
    - Dataset file location
    - Total rows and columns
    - Input space description (379 features)
    - Output space description (7 heads with class counts)
    - Action type distribution
    - Player count distribution
    - Example of how to load the data
  - Return formatted string for console output

### 13. Implement main execution function

- Create `main()` function:
  - Print "Starting dataset export..."
  - Call `extract_all_training_data(DB_PATH)` with progress tracking
  - Call `write_to_csv(data, OUTPUT_PATH)` with progress tracking
  - Call `generate_dataset_summary(data, OUTPUT_PATH)`
  - Print summary to console
  - Print "Export complete!"
- Add `if __name__ == "__main__":` block to call main()
- Add try-except for graceful error handling

### 14. Add data validation

- Create `validate_input_vector(vec: list[float])` function:
  - Assert length is exactly 381
  - Check for expected NaN patterns (e.g., missing players)
  - Verify value ranges (e.g., gems >= 0, levels in 1-3)
- Create `validate_action_encoding(action_dict: dict)` function:
  - Verify action_type is one of the 4 valid strings
  - Check that appropriate fields are set based on action type
  - Verify vector lengths for multi-element outputs
- Call validation functions during extraction (with option to disable for performance)
- Add `--validate` command line flag to enable strict validation mode

### 15. Handle edge cases

- **Empty deck scenario**: When Card{X}_{Y} is NULL in StateGame
  - Use 12 NaN values for card encoding
- **Missing players**: For 2-3 player games
  - Use 49 NaN values for each missing player slot
  - Always encode 4 player slots (pad with NaN)
- **Missing nobles**: When Character{X} is NULL
  - Use 6 NaN values for noble encoding
- **Top deck reservation**: When action reserves from top deck
  - IDCard will be in database but need to track separately from visible cards
  - Check if card was visible at time of reservation
- **Build from reserved**:
  - Track which cards are in player's reserved pile
  - Encode as 12 + position (giving 12, 13, or 14)
- **No character selected**: Most actions don't involve nobles
  - Use -1 or NaN for noble_selection output
- **No gems removed**: Most actions don't remove gems
  - Use [0, 0, 0, 0, 0, 0] for gems_removed

### 16. Add command line interface

- Use `argparse` module to add CLI options:
  - `--db-path`: Custom database path (default: 'data/games.db')
  - `--output`: Custom output CSV path (default: 'data/training_dataset.csv')
  - `--validate`: Enable strict validation mode (slower but catches errors)
  - `--limit`: Limit number of samples for testing (default: no limit)
  - `--quiet`: Suppress progress output
- Update `main()` to use parsed arguments

### 17. Test with sample data

- Create `test_export_ml_dataset.py` in scripts directory
- Write unit tests for encoding functions:
  - `test_encode_card()` - verify 12 features with correct values
  - `test_encode_noble()` - verify 6 features
  - `test_rotate_players()` - verify correct player ordering
  - `test_encode_card_selection()` - verify position calculations
  - `test_encode_gem_take_3()` - verify 3-hot encoding
  - `test_encode_gem_take_2()` - verify one-hot encoding
  - `test_input_vector_length()` - verify 381 total features
  - `test_output_encoding_completeness()` - verify all 7 heads present
- Write integration test:
  - `test_full_export_with_limit()` - run export with --limit 100
  - Verify CSV has correct structure
  - Verify all expected columns present
  - Verify no data corruption
- Run tests: `python scripts/test_export_ml_dataset.py`

### 18. Document the script

- Add comprehensive docstring to `export_ml_dataset.py` module:
  - Purpose and overview
  - Input/output specification
  - Feature encoding details
  - Action encoding details
  - Usage examples
- Add docstrings to all functions with:
  - Parameter descriptions
  - Return value descriptions
  - Example usage where helpful
- Add inline comments for complex logic:
  - Player rotation algorithm
  - Card position calculation
  - Action type disambiguation

### 19. Run full export and validation

- Execute: `python scripts/export_ml_dataset.py`
- Monitor progress output for completion
- Verify CSV file created at `data/training_dataset.csv`
- Check CSV file size is reasonable (should be large for 1.7M samples)
- Open CSV and manually inspect:
  - Header row has all 402 columns
  - First few data rows have expected structure
  - NaN values appear where expected (missing players, etc.)
  - Action encodings look correct
- Verify dataset summary output matches expectations:
  - Total samples around 1.7M
  - Action type distribution reasonable
  - Player count distribution matches database

### 20. Validate dataset is ML-ready

- Write validation script `scripts/validate_dataset.py`:
  - Load CSV using pandas
  - Check for data type consistency
  - Verify no unexpected NaN patterns
  - Check value ranges make sense
  - Verify action encodings are valid
  - Count unique games matches database
- Optionally create example loader:
  - `scripts/load_dataset_example.py` showing how to:
    - Load CSV with pandas
    - Split into train/validation/test by game_id
    - Separate input features from output targets
    - Handle NaN values appropriately
    - Prepare data for PyTorch/TensorFlow

## Testing Strategy

### Unit Tests

- **Encoding Functions**: Test each encoding function in isolation
  - `encode_card()`: Verify 12-element output with correct values for known card
  - `encode_noble()`: Verify 6-element output with correct VP and costs
  - `encode_player_state()`: Verify 49-element output with proper ordering
  - `rotate_players_by_current()`: Test all rotation scenarios (2, 3, 4 players)
  - `encode_action_type()`: Verify string outputs for all action types
  - `encode_card_selection()`: Test position calculation for all card locations
  - `encode_card_reservation()`: Test visible and top-deck encoding
  - `encode_gem_take_3()`: Verify 3-hot encoding correctness
  - `encode_gem_take_2()`: Verify one-hot encoding correctness
  - `encode_noble_selection()`: Test position finding in state
  - `encode_gems_removed()`: Verify 6-element output

- **Helper Functions**: Test database and utility functions
  - `get_card_by_id()`: Mock database query, verify dict structure
  - `connect_to_database()`: Test connection success and failure handling
  - `generate_input_column_headers()`: Verify exactly 379 headers
  - `generate_output_column_headers()`: Verify correct output structure

- **Validation Functions**: Test data validation
  - `validate_input_vector()`: Test with valid and invalid vectors
  - `validate_action_encoding()`: Test all action type combinations

### Integration Tests

- **Small Dataset Export**:
  - Export first 100 samples from database
  - Verify CSV structure is correct
  - Check all columns present
  - Verify game_id column matches database
  - Ensure no exceptions during export

- **Round-trip Verification**:
  - Export sample, read it back, verify values match database
  - Check that input features can be reconstructed from database state
  - Verify action encoding matches original action in database

- **Edge Case Coverage**:
  - Test export with 2-player games only (verify NaN padding for players 3-4)
  - Test export with games where decks are exhausted
  - Test export with games with no nobles available
  - Test export with all action types represented

### Edge Cases

- **Missing Data**:
  - Games with only 2 players (verify 2 player slots filled, 2 with NaN)
  - Games with only 3 players (verify 3 player slots filled, 1 with NaN)
  - Exhausted decks (no cards remaining for a level)
  - Fewer than 5 nobles available (early in 2-player game might have only 3)
  - Player with no reserved cards (verify 36 NaN values)
  - Player with 1-2 reserved cards (verify partial NaN padding)

- **Action Variations**:
  - BUILD from visible card (levels 1, 2, 3)
  - BUILD from reserved card (positions 0, 1, 2 in reserve)
  - RESERVE visible card
  - RESERVE from top deck (levels 1, 2, 3)
  - TAKE 3 different tokens
  - TAKE 2 same tokens
  - Actions with noble selection
  - Actions without noble selection
  - Actions requiring gem removal
  - Actions without gem removal

- **Color Ordering**:
  - Verify consistent COLOR_ORDER in all encoding functions
  - Check card costs use correct order
  - Check card bonuses use correct order
  - Check gems use correct order (including gold)
  - Check noble requirements use correct order
  - Check player reductions use correct order
  - Check gem take/removal use correct order

- **Boundary Conditions**:
  - First turn of game (turn 0)
  - Final turn of game (endgame state)
  - Player with 10 victory points (high VP)
  - Player with 0 victory points (low VP)
  - Player with maximum tokens (10)
  - Player with no tokens
  - Board with no tokens of a color remaining

## Acceptance Criteria

- [ ] Script successfully connects to `data/games.db` without errors
- [ ] All game states are extracted and processed without exceptions
- [ ] Each input vector is exactly 381 dimensions
- [ ] Input vector encoding follows specification:
  - Number of players (1 int)
  - Turn number (1 int)
  - Gems on board (6 ints in COLOR_ORDER)
  - 12 visible cards (144 features, 12 per card)
  - Deck remaining counts (3 ints)
  - 5 nobles (30 features, 6 per noble)
  - 4 players (196 features, 49 per player)
- [ ] Player ordering places current player first, followed by next players in turn order
- [ ] Missing players (in 2-3 player games) are filled with NaN values
- [ ] Color ordering is consistent across all features (WHITE, BLUE, GREEN, RED, BLACK, GOLD)
- [ ] Action encoding includes all 7 output heads:
  - action_type (string: "build", "reserve", "take 2 tokens", "take 3 tokens")
  - card_selection (int 0-14 for build actions)
  - card_reservation (int 0-15 for reserve actions)
  - gem_take_3 (5-element 3-hot vector)
  - gem_take_2 (5-element one-hot vector)
  - noble_selection (int 0-4 or -1/NaN)
  - gems_removed (6-element count vector)
- [ ] Card selection encoding: visible cards [0-11], reserved cards [12-14]
- [ ] Card reservation encoding: visible cards [0-11], top deck [13-15]
- [ ] CSV file has clear column headers (402 total: 1 game_id + 381 input + 20 output)
- [ ] CSV file includes game_id column for train/val/test splitting
- [ ] Progress tracking displays percentage with format: `Progress: [====>    ] 45.2%`
- [ ] Progress updates use static mode (single line, overwrite with \r)
- [ ] Dataset summary is displayed after export completes, including:
  - Total number of samples
  - Input space description (381 dimensions)
  - Output space description (7 heads with class counts)
  - Action type distribution
  - Player count distribution
  - File location and size
- [ ] Script handles all edge cases without crashing:
  - 2, 3, and 4 player games
  - Missing visible cards (exhausted decks)
  - Missing nobles
  - All action types (build, reserve, take 2, take 3)
  - Actions with/without noble selection
  - Actions with/without gem removal
- [ ] Validation mode (--validate flag) catches encoding errors
- [ ] Unit tests pass for all encoding functions
- [ ] Integration test with --limit 100 produces valid CSV
- [ ] Full export completes successfully on entire database (~1.7M samples)
- [ ] Exported CSV file can be loaded with pandas
- [ ] No data corruption or formatting errors in CSV
- [ ] Documentation is complete with usage examples and feature descriptions

## Validation Commands

Execute every command to validate the feature works correctly with zero regressions.

- `python scripts/export_ml_dataset.py --help` - Display help message with all CLI options
- `python scripts/export_ml_dataset.py --limit 100 --output data/test_export.csv` - Test export with 100 samples
- `python -c "import pandas as pd; df = pd.read_csv('data/test_export.csv'); print(f'Shape: {df.shape}'); print(f'Columns: {len(df.columns)}'); assert len(df.columns) == 402, 'Expected 402 columns'; assert df.shape[0] <= 100, 'Expected max 100 rows'; print('✓ Test export validated')"` - Validate test export structure
- `python scripts/test_export_ml_dataset.py` - Run unit tests for encoding functions
- `python scripts/export_ml_dataset.py --validate --limit 1000` - Run validation mode on 1000 samples to catch errors
- `python scripts/export_ml_dataset.py` - Run full export on entire database
- `python -c "import pandas as pd; df = pd.read_csv('data/training_dataset.csv'); print(f'Total samples: {len(df)}'); print(f'Unique games: {df.game_id.nunique()}'); print(f'Columns: {len(df.columns)}'); print(f'Action types:\\n{df.action_type.value_counts()}'); assert len(df.columns) == 402; print('✓ Full export validated')"` - Validate full export
- `python scripts/validate_dataset.py` - Comprehensive dataset validation (data types, ranges, consistency)
- `python -c "import sqlite3; conn = sqlite3.connect('data/games.db'); cursor = conn.cursor(); cursor.execute('SELECT COUNT(*) FROM Action'); db_count = cursor.fetchone()[0]; import pandas as pd; df = pd.read_csv('data/training_dataset.csv'); csv_count = len(df); print(f'Database actions: {db_count}'); print(f'CSV samples: {csv_count}'); assert db_count == csv_count, f'Mismatch: DB has {db_count}, CSV has {csv_count}'; print('✓ Sample count matches database')"` - Verify sample count matches database

## Notes

### Feature Dimension Verification

The input vector has **381 dimensions**. The calculation breakdown is:
- Basic info: 1 (num_players) + 1 (turn_number) = 2
- Gems on board: 6 (white, blue, green, red, black, gold)
- Visible cards: 12 cards × 12 features = 144
  - Each card: 1 (VP) + 1 (level) + 5 (cost) + 5 (bonus one-hot) = 12
- Deck remaining: 3 (one per level)
- Nobles: 5 nobles × 6 features = 30
  - Each noble: 1 (VP, always 3) + 5 (cost requirements) = 6
- Players: 4 players × 49 features = 196
  - Each player: 1 (position) + 1 (VP) + 6 (gems) + 5 (reductions) + 36 (3 reserved cards × 12) = 49

**Total: 2 + 6 + 144 + 3 + 30 + 196 = 381 ✓**

### Color Ordering Consistency

Critical requirement: ALL color-based features must use the same ordering:
```python
COLOR_ORDER = [WHITE, BLUE, GREEN, RED, BLACK, GOLD]  # Indices: 0, 1, 2, 3, 4, 5
```

This applies to:
- Gems on board
- Card costs (excluding gold, so indices 0-4)
- Card bonus one-hot (excluding gold, so indices 0-4)
- Noble cost requirements (excluding gold, so indices 0-4)
- Player gems (including gold, indices 0-5)
- Player reductions (excluding gold, so indices 0-4)
- Gem take actions (excluding gold for take-2 and take-3, so indices 0-4)
- Gems removed (including gold, indices 0-5)

### Reserved Cards Tracking Challenge

Tracking which cards are reserved by a player at a given turn requires reconstructing state:
1. Find all RESERVE actions by this player up to this turn
2. Find all BUILD actions from reserve by this player up to this turn
3. Currently reserved = (all reserved) - (built from reserve)

This is complex because the database doesn't track "current reserved cards" per turn explicitly. Need to reconstruct from action history.

### Top Deck Reservations

When a player reserves from the top deck, the card becomes visible in their reserve but wasn't previously visible on the board. The action will have IDCard set, but we need to identify this is from top deck vs. visible cards by checking if the card was in the displayed cards at that turn.

### Multi-Head Output Structure

The 7 output heads create a multi-task learning setup:
1. **action_type**: 4 classes (categorical)
2. **card_selection**: 15 classes [0-14] (only for build actions)
3. **card_reservation**: 15 classes [0-14] (only for reserve actions)
4. **gem_take_3**: 5 binary outputs (3-hot encoding)
5. **gem_take_2**: 5 binary outputs (one-hot encoding)
6. **noble_selection**: 5 classes [0-4] (only when noble available)
7. **gems_removed**: 6 count outputs (0-3 per color typically)

During training, the model will need masking to only compute loss on relevant heads for each action type.

### Dataset Size Estimation

With 1,743,688 state-action pairs:
- Input: 381 features × 4 bytes (float32) = 1,524 bytes per sample
- Output: ~20 features × 4 bytes = 80 bytes per sample
- Total per sample: ~1,600 bytes
- Total dataset: 1,743,688 × 1,600 ≈ 2.6 GB raw data

CSV format will be larger due to text encoding, potentially 5-10 GB.

### Future Enhancements

Potential improvements for future iterations:
- Add `--format` option to export as HDF5 or Parquet for efficiency
- Add `--augment` flag to include data augmentation (player permutations)
- Add `--stratify` option to balance action type distribution
- Create separate train/val/test splits with game-based grouping
- Add visualization script to plot feature distributions
- Include feature scaling/normalization utilities
- Generate data statistics report (mean, std, min, max per feature)
- Add schema validation against expected feature ranges

### Dependencies

The script requires only standard library modules:
- `sqlite3` - Database access
- `csv` - CSV file writing
- `sys` - System operations
- `typing` - Type hints
- `pathlib` - Path handling
- `argparse` - Command line interface

For validation and testing:
- Consider adding `pandas` for data validation (optional, not required for main script)
- Consider adding `numpy` for numerical validation (optional)

### Performance Considerations

Processing 1.7M samples may take significant time:
- Estimate: ~100-1000 samples per second depending on query optimization
- Total time: 30 minutes to several hours
- Optimizations:
  - Load all cards/characters into memory (avoid repeated queries)
  - Use batch queries where possible
  - Consider multiprocessing for parallel encoding
  - Use csv.writer buffering for faster writes
  - Disable validation mode for production runs

### Train/Val/Test Splitting

The CSV includes `game_id` specifically to enable proper splitting:
```python
# Example splitting approach
unique_games = df['game_id'].unique()
train_games, temp_games = train_test_split(unique_games, test_size=0.3)
val_games, test_games = train_test_split(temp_games, test_size=0.5)

train_df = df[df['game_id'].isin(train_games)]
val_df = df[df['game_id'].isin(val_games)]
test_df = df[df['game_id'].isin(test_games)]
```

This ensures no game appears in multiple splits, preventing data leakage.
