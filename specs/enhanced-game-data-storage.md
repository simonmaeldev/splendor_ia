# Feature: Enhanced Game Data Storage and Real-Time CSV Export

## Feature Description
Enhance the game data storage system to save comprehensive state information directly in the database (matching the ML dataset CSV format) and automatically export game data to CSV files during gameplay. This eliminates the need for complex post-processing and ensures all ML-ready features are available immediately. The system will save each game's data in three locations: the SQLite database with enriched schema, a game-specific CSV file, and an aggregated all_games.csv file.

## User Story
As a machine learning researcher
I want game data automatically saved in ML-ready format during gameplay
So that I can train models immediately without running expensive export scripts, and I can recover data even if the simulation crashes

## Problem Statement
The current system has several critical issues:
1. **Database schema is incomplete** - Missing calculated features like victory points per turn, gem reductions, reserved cards state, and deck remaining counts
2. **Post-processing is expensive** - export_ml_dataset.py must recalculate ~1.7M state-action pairs from Action history, taking several minutes
3. **State capture timing is inconsistent** - StatePlayer is captured AFTER action execution, but we need state BEFORE action for ML training
4. **No real-time CSV export** - Must manually run export script after simulations complete; data lost if process crashes
5. **Reference bugs possible** - Code like `[[]] * len(Players)` creates shared references instead of independent copies
6. **Data extraction is tedious** - No easy way to access specific games without database queries

## Solution Statement
Implement a comprehensive enhancement that:
1. **Expands database schema** to store all 381 ML input features and 20 output features directly in tables (StatePlayer, StateGame, Action)
2. **Fixes state capture timing** so StatePlayer represents state BEFORE the player takes their action
3. **Adds real-time CSV export** that writes two CSV files at the end of each game:
   - `/data/games/<nb_players>_games/<game_id>.csv` - Game-specific file without NaN padding
   - `/data/games/all_games.csv` - Aggregated file with NaN padding for 4 players
4. **Ensures data consistency** with deep copies and careful state management to avoid reference bugs
5. **Maintains backward compatibility** with existing simulation infrastructure

## Relevant Files
Use these files to implement the feature:

### Existing Files - Database Schema

- **`scripts/create_database.py`** (lines 1-166)
  - Current database schema definition
  - Need to add columns to StatePlayer (VictoryPoints, gem reductions, reserved card IDs)
  - Need to add columns to StateGame (deck remaining counts)
  - Need to add new columns to Action table for denormalized data

### Existing Files - Data Collection

- **`scripts/main.py`** (lines 83-134)
  - `saveIntoBdd()` function (lines 83-103) - saves game data to database
  - `PlayGame()` function (lines 105-134) - main game loop with state capture
  - **Line 114**: Initial state capture (BEFORE any actions)
  - **Line 118**: StateGame capture (BEFORE action) - timing is correct
  - **Line 129**: StatePlayer capture (AFTER action) - **NEEDS FIXING**
  - Need to move player state capture to before doMove()
  - Need to add CSV export calls at end of game

- **`scripts/data_collector.py`** (lines 1-200+)
  - Safe data collection wrapper around main.py
  - Progress tracking and logging infrastructure
  - Need to ensure CSV directories are created

### Existing Files - Game Engine

- **`src/splendor/board.py`** (lines 1-357)
  - `Board` class with game state management
  - `getState()` method - returns state for StateGame table
  - `getPlayerState(player_num)` method - returns player token counts
  - Need to add new methods: `getPlayerFullState()`, `getDeckRemainingCounts()`, `getPlayerReservedCards()`, etc.

- **`src/splendor/player.py`** (lines 1-188)
  - `Player` class with player state
  - Properties: tokens, built cards, reserved cards, characters
  - Methods: `getVictoryPoints()`, `getReductions()`, etc.
  - Need to ensure deep copy safety for all state methods

- **`src/splendor/move.py`**
  - Move/Action representation
  - Need to verify action encoding matches export_ml_dataset.py

### Existing Files - ML Export Reference

- **`scripts/export_ml_dataset.py`** (lines 1-1166)
  - **CRITICAL REFERENCE** - defines exact format for CSV export
  - Line 571-618: `generate_input_column_headers()` - defines 381 input column names
  - Line 621-641: `generate_output_column_headers()` - defines 20 output column names
  - Line 122-149: `encode_card()` - card encoding logic (12 features)
  - Line 152-165: `encode_noble()` - noble encoding logic (6 features)
  - Line 241-281: `encode_player_state()` - player encoding logic (49 features)
  - Line 306-392: `encode_game_state()` - full game state encoding (381 features)
  - Line 543-568: `encode_action()` - action encoding (7 output heads)
  - Need to reuse this encoding logic for real-time CSV export

### Existing Files - Constants

- **`src/splendor/constants.py`** (lines 1-188)
  - Color constants: WHITE=0, BLUE=1, GREEN=2, RED=3, BLACK=4, GOLD=5
  - Action types: BUILD=0, RESERVE=1, TOKENS=2
  - Deck sizes: DECK1 (40 cards), DECK2 (30 cards), DECK3 (20 cards)
  - Important for consistent encoding

### New Files

- **`src/splendor/csv_exporter.py`**
  - New module for real-time CSV export functionality
  - Reuses encoding logic from export_ml_dataset.py
  - Functions: `export_game_to_csv()`, `write_game_specific_csv()`, `append_to_all_games_csv()`
  - Handles directory creation, NaN padding, and file appending

- **`data/games/2_games/.gitkeep`**
  - Directory structure for 2-player game CSVs

- **`data/games/3_games/.gitkeep`**
  - Directory structure for 3-player game CSVs

- **`data/games/4_games/.gitkeep`**
  - Directory structure for 4-player game CSVs

## Implementation Plan

### Phase 1: Foundation
Update the database schema to include all missing columns (victory points, reductions, reserved cards, deck remaining), add validation to prevent reference bugs, and create the directory structure for CSV files. This establishes the infrastructure needed for enhanced data storage.

### Phase 2: Core Implementation
Modify the game engine to capture complete state information before actions are executed, implement deep copy safety throughout state capture, add new methods to Board and Player classes for full state extraction, and ensure all 381 input features can be computed from the game state.

### Phase 3: Integration
Create the CSV exporter module that reuses encoding logic from export_ml_dataset.py, integrate real-time CSV writing into the game loop (called at end of each game), update saveIntoBdd() to save enriched data to the new schema, and add comprehensive validation to ensure data consistency across database and CSV files.

## Step by Step Tasks
IMPORTANT: Execute every step in order, top to bottom.

### 1. Update database schema

- Read `scripts/create_database.py` to understand current schema
- Add new columns to `StatePlayer` table:
  - `VictoryPoints INT NOT NULL` - total VP at this turn (from cards + nobles)
  - `ReductionWhite INT NOT NULL` - gem reduction from built cards
  - `ReductionBlue INT NOT NULL`
  - `ReductionGreen INT NOT NULL`
  - `ReductionRed INT NOT NULL`
  - `ReductionBlack INT NOT NULL`
  - `ReservedCard1 INT` - nullable foreign key to Card table
  - `ReservedCard2 INT` - nullable foreign key to Card table
  - `ReservedCard3 INT` - nullable foreign key to Card table
  - Add foreign key constraints for reserved cards
- Add new columns to `StateGame` table:
  - `DeckLevel1Remaining INT NOT NULL` - cards left in level 1 deck
  - `DeckLevel2Remaining INT NOT NULL` - cards left in level 2 deck
  - `DeckLevel3Remaining INT NOT NULL` - cards left in level 3 deck
- Update the schema with proper SQL formatting
- Add comments documenting the new columns

### 2. Create directory structure for CSV files

- Create directory structure in `data/games/`:
  - `data/games/2_games/` for 2-player games
  - `data/games/3_games/` for 3-player games
  - `data/games/4_games/` for 4-player games
- Add `.gitkeep` files to each directory to ensure they're tracked
- Add `data/games/all_games.csv` to `.gitignore` if not already present (it will be large)
- Add `data/games/*/` pattern to `.gitignore` for individual game CSVs

### 3. Create CSV exporter module - Part 1: Setup and utilities

- Create new file `src/splendor/csv_exporter.py`
- Import necessary modules: csv, pathlib, typing, constants
- Copy color ordering and constants from export_ml_dataset.py:
  - `COLOR_ORDER = [WHITE, BLUE, GREEN, RED, BLACK, GOLD]`
  - `COLOR_NAMES = ['white', 'blue', 'green', 'red', 'black', 'gold']`
  - `DECK_SIZES = {1: 40, 2: 30, 3: 20}`
- Copy helper functions from export_ml_dataset.py:
  - `encode_card(card_obj) -> list[float]` (12 features)
  - `encode_noble(noble_obj) -> list[float]` (6 features)
  - `rotate_players_by_current(player_numbers, current_player) -> list[int]`
- Add function `generate_input_column_headers() -> list[str]` (copy from export_ml_dataset.py line 571)
- Add function `generate_output_column_headers() -> list[str]` (copy from export_ml_dataset.py line 621)
- Add function `generate_all_headers() -> list[str]` that returns ["game_id"] + input_headers + output_headers

### 4. Create CSV exporter module - Part 2: State encoding

- In `csv_exporter.py`, add function `encode_player_state_from_board(board, player_num, turn_position) -> list[float]`:
  - Get player object from board
  - Extract: position (turn_position), victory points, tokens (6), reductions (5)
  - Extract reserved cards (up to 3) and encode each with `encode_card()`
  - Pad with NaN if fewer than 3 reserved cards
  - Return 49 features total
  - Use deep copies to avoid reference issues
- Add function `encode_game_state_from_board(board, action_turn_num) -> list[float]`:
  - Extract num_players, turn_number
  - Extract board tokens (6)
  - Extract and encode 12 visible cards from board.displayedCards
  - Calculate deck remaining counts from board.decks lengths
  - Extract and encode 5 nobles from board.characters
  - Get all players, rotate by current player
  - Encode 4 player states (pad with NaN for missing players)
  - Return 381 features total
- Add validation: assert length is exactly 381

### 5. Create CSV exporter module - Part 3: Action encoding

- In `csv_exporter.py`, add function `encode_action_from_move(move, board) -> dict`:
  - Extract action type from move.actionType
  - Encode action_type string: BUILD → "build", RESERVE → "reserve", TOKENS → "take 2/3 tokens"
  - For BUILD actions:
    - Find card position in visible cards (0-11) or reserved (12-14)
    - Set card_selection, leave card_reservation as NaN
  - For RESERVE actions:
    - Find card position in visible cards (0-11) or top deck (12-14)
    - Set card_reservation, leave card_selection as NaN
  - For TOKENS actions:
    - Encode gem_take_2 or gem_take_3 based on total taken
    - Leave card fields as NaN
  - Encode noble_selection if move.character is not None
  - Encode gems_removed from move.tokensToRemove
  - Return dict with 7 output heads
- Add function `flatten_action_dict(action_dict) -> list[Any]`:
  - Convert action dict to flat list matching output column order
  - Return 20 values

### 6. Create CSV exporter module - Part 4: CSV writing functions

- In `csv_exporter.py`, add function `write_game_specific_csv(game_id, nb_players, data_rows, output_dir)`:
  - Create output directory if it doesn't exist
  - Open file: `{output_dir}/{nb_players}_games/{game_id}.csv`
  - Write headers (without NaN padding - only actual player columns)
  - Write each data row (game state + action pairs)
  - Close file
- Add function `append_to_all_games_csv(game_id, nb_players, data_rows, output_file)`:
  - Check if file exists; if not, write headers with full 4-player padding
  - Open file in append mode
  - Pad data rows to 4 players with NaN if nb_players < 4
  - Write each padded row
  - Close file
- Add function `export_game_to_csv(game_id, nb_players, states_and_actions, output_dir)`:
  - Convert states_and_actions to CSV rows
  - Call `write_game_specific_csv()`
  - Call `append_to_all_games_csv()`
  - Return success status

### 7. Fix state capture timing in main.py

- Read `scripts/main.py` lines 105-134 to understand current flow
- **CRITICAL FIX**: Modify the game loop to capture StatePlayer BEFORE action:
  - Currently line 129 captures AFTER doMove(): `historyPlayers[currentPlayer].append(state.getPlayerState(currentPlayer))`
  - Need to capture BEFORE doMove()
- Update the loop structure:
  ```python
  while (state.getMoves() != []):
      state.show()
      historyState.append(state.getState())
      currentPlayer = state.currentPlayer

      # NEW: Capture player state BEFORE action
      # (Initial state already captured at line 114)

      # Get action from AI
      if Players[currentPlayer] == "ISMCTS_PARA":
          m = ISMCTS_para(rootstate = state, itermax = nbIte[state.currentPlayer], verbose = False)
      elif Players[currentPlayer] == "ISMCTS":
          m = ISMCTS(rootstate = state, itermax = nbIte[state.currentPlayer], verbose = False)

      print ("Best Move: " + str(m) + "\n")

      # NEW: Capture state BEFORE executing action
      historyPlayers[currentPlayer].append(state.getPlayerState(currentPlayer))

      # Execute action
      state.doMove(m)

      # Store action
      historyActionPlayers[currentPlayer].append(m)
  ```
- Ensure initial state (line 114) remains unchanged

### 8. Add deep copy safety to Board methods

- Read `src/splendor/board.py` to understand getState() and getPlayerState()
- Update `getPlayerState(self, player_num)` to return deep copy of token list
- Add new method `getPlayerFullState(self, player_num) -> dict`:
  - Return dict with: tokens (deep copy), victory_points, reductions, reserved_cards (deep copy list), built_cards (deep copy list), characters (deep copy list)
  - Use `deepcopy()` from copy module for all lists
- Add new method `getDeckRemainingCounts(self) -> list[int]`:
  - Return [len(self.deckLVL1), len(self.deckLVL2), len(self.deckLVL3)]
- Add new method `getFullGameState(self) -> dict`:
  - Return comprehensive state including all board information
  - Use deep copies for all mutable data structures
- Add validation: ensure no shared references with assertion checks

### 9. Verify Player class methods

- Read `src/splendor/player.py` to check existing methods
- Verify `getVictoryPoints()` method exists and works correctly
- Add or verify `getReductions() -> list[int]` method:
  - Count bonus of each color from self.built
  - Return list of 5 ints (white, blue, green, red, black)
- Add or verify `getReservedCards() -> list[Card]` method:
  - Return deep copy of self.reserved list
  - Use `deepcopy()` to avoid reference issues
- Add or verify `getBuiltCards() -> list[Card]` method:
  - Return deep copy of self.built list
- Ensure all methods use deep copies when returning mutable data

### 10. Update saveIntoBdd to save enriched data

- Read current `saveIntoBdd()` implementation in scripts/main.py (lines 83-103)
- Update `savePlayerState()` function signature and implementation:
  - Change to: `savePlayerState(cursor, gameID, playerPos, history, board, historyActionPlayers)`
  - For each turn, calculate and save:
    - VictoryPoints (query built cards + nobles from actions up to this turn)
    - ReductionWhite through ReductionBlack (count bonuses from built cards)
    - ReservedCard1, ReservedCard2, ReservedCard3 (track from reserve/build actions)
  - Use the board state to compute these values correctly
- Update `saveGamesState()` function to save deck remaining counts:
  - Add DeckLevel1Remaining, DeckLevel2Remaining, DeckLevel3Remaining to insert
  - Calculate from board.getDeckRemainingCounts() at each state
- Ensure all INSERT statements include new columns
- Add error handling for database operations

### 11. Integrate CSV export into PlayGame()

- In `scripts/main.py`, import csv_exporter module:
  - `from splendor.csv_exporter import export_game_to_csv, encode_game_state_from_board, encode_action_from_move`
- After game completes (after line 132 `winner = state.getVictorious(True)`):
  - Build list of (state, action) pairs from historyState and historyActionPlayers
  - Convert each pair to CSV row format:
    - game_id, 381 input features (from encode_game_state_from_board), 20 output features (from encode_action_from_move)
  - Call `export_game_to_csv(gameID, len(Players), csv_data, 'data/games')`
- This happens BEFORE saveIntoBdd() call to ensure CSV is written even if DB save fails
- Add try-except around CSV export to prevent crashes

### 12. Update saveIntoBdd to use enriched state data

- Modify `saveIntoBdd()` to accept additional parameters:
  - `saveIntoBdd(state, winner, historyState, historyPlayers, historyActionPlayers, nbIte, Players, board)`
- Pass the board object through the call chain
- Use board methods to extract enriched state data
- Ensure consistency between CSV export and database save

### 13. Handle reference bugs prevention

- Review all list/dict operations in data collection code
- Replace any instances of `[[]] * n` with `[[] for _ in range(n)]`
- Add validation in PlayGame():
  - After creating `historyActionPlayers = [[] for _ in range(len(Players))]`
  - Assert each player's list is a unique object: `assert id(historyActionPlayers[0]) != id(historyActionPlayers[1])`
- Add similar validation for historyPlayers initialization
- Use deepcopy() when appending state to history lists

### 14. Test with single game

- Create test script `scripts/test_enhanced_storage.py`:
  - Run single 2-player game
  - Verify database saves correctly with new columns
  - Verify CSV files are created in correct locations
  - Verify CSV has correct structure (402 columns)
  - Verify data consistency between DB and CSV
- Run test: `python scripts/test_enhanced_storage.py`
- Check output files:
  - `data/games/2_games/1.csv` should exist
  - `data/games/all_games.csv` should exist with one game's data
- Manually inspect CSV headers and first few rows
- Verify no NaN padding in game-specific CSV
- Verify NaN padding present in all_games.csv for 2-player game

### 15. Test state capture timing fix

- In test script, add validation:
  - Compare StatePlayer tokens at turn N with tokens after action at turn N
  - Tokens should match state BEFORE action, not after
  - Check that if action is TOKENS with take [2,0,0,0,0,0], the StatePlayer shows tokens before taking
- Add debug output showing:
  - Player tokens before action
  - Action taken
  - Player tokens in StatePlayer record
  - Player tokens after action
- Verify timing is correct: StatePlayer = before, Action = what was done, next StatePlayer = result

### 16. Test with multiple games of different player counts

- Run 3 test games: one 2-player, one 3-player, one 4-player
- Verify directory structure:
  - `data/games/2_games/` contains 2-player game CSV
  - `data/games/3_games/` contains 3-player game CSV
  - `data/games/4_games/` contains 4-player game CSV
  - `data/games/all_games.csv` contains all 3 games
- Verify CSV structure:
  - 2-player game CSV has columns for 2 players only (no player3/player4 columns)
  - 3-player game CSV has columns for 3 players only
  - 4-player game CSV has columns for 4 players
  - all_games.csv has columns for 4 players (with NaN for missing players)
- Verify data correctness:
  - No shared references (each game has independent data)
  - Victory points calculated correctly
  - Reductions match built cards
  - Reserved cards tracked accurately

### 17. Test CSV data consistency

- Create validation script `scripts/validate_enhanced_storage.py`:
  - Load game from database
  - Load same game from game-specific CSV
  - Load same game from all_games.csv
  - Compare all three sources field-by-field
  - Verify exact match (accounting for NaN padding differences)
- Check specific fields:
  - Player victory points at each turn
  - Gem reductions at each turn
  - Reserved cards at each turn
  - Deck remaining counts
  - Action encodings
- Report any discrepancies with detailed error messages

### 18. Update data_collector.py integration

- Read `scripts/data_collector.py` to understand progress tracking
- Ensure CSV directory creation happens before data collection starts
- Add progress tracking for CSV writes (optional, since it happens at game end)
- Update error handling to catch CSV export failures separately from DB failures
- Add cleanup code to handle partial CSV writes on interrupt (Ctrl+C)
- Test that Ctrl+C during simulation doesn't corrupt all_games.csv

### 19. Add comprehensive documentation

- Update docstrings in `csv_exporter.py`:
  - Module-level docstring explaining purpose and usage
  - Function docstrings with parameter types and return values
  - Examples of CSV format and encoding
- Add comments to modified sections of `main.py`:
  - Explain state capture timing fix
  - Document CSV export integration
  - Clarify deep copy usage
- Update `scripts/create_database.py` with schema documentation:
  - Document new StatePlayer columns and their meaning
  - Document new StateGame columns
  - Explain relationship between DB and CSV data
- Create README section explaining the enhanced storage system

### 20. Run validation commands

- Reinitialize database: `python scripts/create_database.py && python scripts/load_database.py`
- Run small simulation: configure `data/simulation_config.txt` with small counts (e.g., 2:5, 3:5, 4:5)
- Run data collection: `python scripts/main.py`
- Verify outputs:
  - Database contains games with new columns populated
  - CSV files exist in correct directories
  - all_games.csv contains all games with proper NaN padding
- Run validation: `python scripts/validate_enhanced_storage.py`
- Check data integrity:
  - `sqlite3 data/games.db "SELECT COUNT(*) FROM Game"` - should show 15 games
  - `wc -l data/games/all_games.csv` - should show header + sum of all turns from all games
  - `ls data/games/*/` - should show individual CSV files
- Verify no errors or warnings
- Test that export_ml_dataset.py still works with new schema (should be faster now)

## Testing Strategy

### Unit Tests

- **CSV Encoding Functions**:
  - Test `encode_card()` with various card types, verify 12 features
  - Test `encode_noble()` with various nobles, verify 6 features
  - Test `encode_player_state_from_board()` with different player states, verify 49 features
  - Test `encode_game_state_from_board()` with different game states, verify 381 features
  - Test `encode_action_from_move()` with all action types, verify correct encoding

- **State Capture Functions**:
  - Test `Board.getPlayerFullState()` returns deep copies (modify returned list shouldn't affect original)
  - Test `Board.getDeckRemainingCounts()` returns correct counts
  - Test `Player.getReductions()` calculates bonuses correctly
  - Test `Player.getReservedCards()` returns deep copy

- **Reference Safety**:
  - Test that `historyActionPlayers = [[] for _ in range(n)]` creates independent lists
  - Test that appending to history doesn't create shared references
  - Test deep copy safety in all state extraction methods

### Integration Tests

- **Single Game Export**:
  - Run one 2-player game
  - Verify database has complete data with new columns
  - Verify CSV files created correctly
  - Verify data consistency between DB and CSV

- **Multi-Game Export**:
  - Run games with 2, 3, and 4 players
  - Verify correct directory structure
  - Verify NaN padding in all_games.csv
  - Verify no NaN padding in game-specific CSVs

- **State Timing Verification**:
  - Capture state before and after action
  - Verify StatePlayer matches state before action
  - Verify subsequent state reflects action results

- **Data Collection Integration**:
  - Run through data_collector.py workflow
  - Test interrupt handling (Ctrl+C)
  - Verify progress tracking works correctly
  - Verify CSV files are complete after interruption

### Edge Cases

- **Reference Bugs**:
  - Create historyActionPlayers with list multiplication: verify error is caught
  - Modify a player state in history: verify original board state unchanged
  - Test with shared reference detection assertions

- **Variable Player Counts**:
  - Test 2-player game: verify 2 players in game CSV, 4 in all_games.csv
  - Test 3-player game: verify 3 players in game CSV, 4 in all_games.csv
  - Test 4-player game: verify 4 players in both CSVs
  - Verify NaN padding only in all_games.csv, not game-specific

- **Empty/Exhausted States**:
  - Test when deck is exhausted (no visible cards): verify NaN encoding
  - Test with no reserved cards: verify NaN padding in reserved slots
  - Test with max reserved cards (3): verify all slots populated

- **CSV File Management**:
  - Test when CSV file already exists: verify append works correctly
  - Test when directory doesn't exist: verify creation happens
  - Test disk full scenario: verify error handling
  - Test file permissions issue: verify error handling

- **State Consistency**:
  - Verify victory points match sum of card VPs + noble VPs
  - Verify reductions match built card bonuses
  - Verify deck remaining equals DECK_SIZE - used cards
  - Verify reserved cards list matches action history

## Acceptance Criteria

- [ ] Database schema updated with new columns:
  - [ ] StatePlayer has VictoryPoints, Reduction* (5 colors), ReservedCard* (3 slots)
  - [ ] StateGame has DeckLevel*Remaining (3 levels)
- [ ] State capture timing fixed:
  - [ ] StatePlayer captured BEFORE action execution (not after)
  - [ ] StateGame remains captured before action (no change needed)
  - [ ] Action still captured correctly
- [ ] CSV export functionality implemented:
  - [ ] Game-specific CSV created in `/data/games/<nb_players>_games/<game_id>.csv`
  - [ ] No NaN padding in game-specific CSV
  - [ ] all_games.csv created/appended in `/data/games/all_games.csv`
  - [ ] NaN padding to 4 players in all_games.csv
  - [ ] CSV format exactly matches export_ml_dataset.py (381 input + 20 output + game_id)
  - [ ] CSV written at end of each game (batch mode, not per turn)
- [ ] Data consistency guaranteed:
  - [ ] No reference bugs (all state uses deep copies)
  - [ ] Database data matches CSV data
  - [ ] Victory points calculated correctly at each turn
  - [ ] Gem reductions match built cards
  - [ ] Reserved cards tracked accurately
  - [ ] Deck remaining counts correct
- [ ] Directory structure created:
  - [ ] `/data/games/2_games/` exists
  - [ ] `/data/games/3_games/` exists
  - [ ] `/data/games/4_games/` exists
  - [ ] Directories created automatically if missing
- [ ] Integration with existing system:
  - [ ] data_collector.py works with new system
  - [ ] Ctrl+C interruption handled gracefully
  - [ ] Progress tracking still functions
  - [ ] Database saves complete successfully
- [ ] Validation passing:
  - [ ] Single game test passes
  - [ ] Multi-game test passes (2, 3, 4 players)
  - [ ] State timing test passes
  - [ ] Data consistency test passes
  - [ ] No reference bug assertions fail
- [ ] Performance acceptable:
  - [ ] CSV export adds minimal overhead to game save time
  - [ ] No memory leaks from deep copies
  - [ ] Simulation speed not significantly impacted

## Validation Commands
Execute every command to validate the feature works correctly with zero regressions.

- `python scripts/create_database.py` - Recreate database with new schema
- `python scripts/load_database.py` - Load cards and characters
- `python scripts/test_enhanced_storage.py` - Run single game test
- `ls data/games/2_games/` - Verify game-specific CSV created
- `head -1 data/games/2_games/1.csv | tr ',' '\n' | wc -l` - Verify column count (should be 402)
- `head -1 data/games/all_games.csv | tr ',' '\n' | wc -l` - Verify column count (should be 402)
- `python -c "import pandas as pd; df = pd.read_csv('data/games/2_games/1.csv'); print(f'Columns: {len(df.columns)}'); print(f'Rows: {len(df)}'); assert len(df.columns) == 402"` - Validate game CSV structure
- `python -c "import pandas as pd; df = pd.read_csv('data/games/all_games.csv'); print(f'Columns: {len(df.columns)}'); print(f'Has NaN: {df.isna().any().any()}'); assert len(df.columns) == 402"` - Validate all_games CSV structure
- `python scripts/validate_enhanced_storage.py` - Validate data consistency between DB and CSV
- `python scripts/test_enhanced_storage.py --test-reference-safety` - Test deep copy safety
- `python scripts/test_enhanced_storage.py --test-state-timing` - Verify state captured before action
- Configure `data/simulation_config.txt` with small counts: `echo "2: 5\n3: 5\n4: 5" > data/simulation_config.txt`
- `python scripts/main.py` - Run small data collection
- `sqlite3 data/games.db "SELECT COUNT(*) FROM Game"` - Verify 15 games saved
- `sqlite3 data/games.db "SELECT VictoryPoints, ReductionWhite FROM StatePlayer LIMIT 5"` - Verify new columns populated
- `sqlite3 data/games.db "SELECT DeckLevel1Remaining, DeckLevel2Remaining, DeckLevel3Remaining FROM StateGame LIMIT 5"` - Verify deck counts
- `ls data/games/2_games/ | wc -l` - Count 2-player game CSVs (should be ~5)
- `ls data/games/3_games/ | wc -l` - Count 3-player game CSVs (should be ~5)
- `ls data/games/4_games/ | wc -l` - Count 4-player game CSVs (should be ~5)
- `wc -l data/games/all_games.csv` - Verify all_games.csv has data from all 15 games
- `python -c "import pandas as pd; df = pd.read_csv('data/games/all_games.csv'); print(f'Total turns: {len(df)}'); print(f'Unique games: {df.game_id.nunique()}'); assert df.game_id.nunique() == 15"` - Verify all games in all_games.csv
- `python scripts/export_ml_dataset.py --limit 100 --output data/test_export_new.csv` - Verify old export script still works
- `python -c "import pandas as pd; old = pd.read_csv('data/test_export_new.csv'); new = pd.read_csv('data/games/all_games.csv'); print(f'Old columns: {len(old.columns)}'); print(f'New columns: {len(new.columns)}'); assert len(old.columns) == len(new.columns)"` - Verify format compatibility

## Notes

### State Capture Timing - Critical Fix

The current code has a timing inconsistency:
- Line 118: `historyState.append(state.getState())` - captures game state BEFORE action ✓
- Line 129: `historyPlayers[currentPlayer].append(state.getPlayerState(currentPlayer))` - captures player state AFTER doMove() ✗

For ML training, we need the state BEFORE the player takes their action. The fix moves line 129 to before `state.doMove(m)` at line 126.

### Reference Bug Prevention

Python's list multiplication creates references, not copies:
```python
# WRONG - creates 3 references to same list
history = [[]] * 3
history[0].append(1)  # All three become [1]!

# CORRECT - creates 3 independent lists
history = [[] for _ in range(3)]
history[0].append(1)  # Only history[0] becomes [1]
```

All state capture must use deep copies to avoid this bug.

### CSV Format Consistency

The CSV format must exactly match `export_ml_dataset.py`:
- 402 columns total: game_id (1) + input (381) + output (20)
- Input features in exact order from `generate_input_column_headers()`
- Output features in exact order from `generate_output_column_headers()`
- Same NaN encoding for missing data
- Same color ordering (WHITE, BLUE, GREEN, RED, BLACK, GOLD)

### NaN Padding Strategy

- **Game-specific CSV** (`data/games/<nb_players>_games/<game_id>.csv`):
  - Only includes columns for actual players in game
  - 2-player game: player0 and player1 columns only
  - 3-player game: player0, player1, player2 columns only
  - No NaN padding needed

- **Aggregated CSV** (`data/games/all_games.csv`):
  - Always includes columns for 4 players
  - Pads missing players with NaN values
  - 2-player game: player2 and player3 all NaN
  - 3-player game: player3 all NaN
  - Allows consistent schema across all games

### Performance Considerations

- CSV export happens at end of game (batch write), not per turn
- Database save still happens per game (atomic transaction)
- Deep copies add memory overhead but ensure correctness
- In-memory encoding avoids repeated database queries
- File I/O buffered for efficiency

### Future Enhancements

- Add compression for CSV files (gzip) to save disk space
- Add checksum validation for data integrity
- Add incremental export mode (resume from partial all_games.csv)
- Add data deduplication (skip games already in all_games.csv)
- Add CSV file rotation (split all_games.csv when it gets too large)
- Add parallel CSV writing for faster exports
- Add validation reports after each simulation run
