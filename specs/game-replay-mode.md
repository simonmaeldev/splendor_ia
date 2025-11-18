# Feature: Game Replay Mode

## Feature Description
A visually appealing terminal-based replay system that loads game data from CSV files and displays the complete game progression turn-by-turn. The replay will show all game state information (board tokens, visible cards, nobles, player states) and actions taken at each turn, using colors and clear formatting to make the game state easy to comprehend when scrolling through the terminal output.

## User Story
As a **Splendor AI researcher/developer**
I want to **replay games from CSV files in a visually rich terminal format**
So that **I can analyze game progression, understand AI decision-making, and debug issues by scrolling through complete game history**

## Problem Statement
Currently, game data is stored in CSV files with 402 columns of encoded numerical data. While this format is excellent for machine learning, it's impossible for humans to understand game progression, analyze strategies, or debug issues. There's no way to visualize what actually happened during a game - which cards were available, what actions were taken, why players made certain decisions, or how the game state evolved turn by turn.

## Solution Statement
Create a Python script (`scripts/replay_game.py`) that:
1. Loads game data from CSV files in the existing format (402 columns)
2. Decodes the numerical encodings back into game objects (cards, nobles, tokens, actions)
3. Prints each turn's complete state to the terminal with:
   - Color-coded output using ANSI color codes (already available in constants.py)
   - Clear visual separation between turns
   - Board state (tokens, visible cards at each level, available nobles)
   - All player states (tokens, victory points, built cards, reserved cards)
   - Action taken with detailed breakdown
   - Statistics (turn number, current player, game progress)
4. Outputs all turns sequentially so users can scroll up/down to analyze game flow

## Relevant Files
Use these files to implement the feature:

- **src/splendor/csv_exporter.py** - Contains all encoding logic that must be reversed for decoding
  - `encode_game_state_from_board()` - 382 input features encoding (lines 223-300)
  - `encode_action_from_move()` - 20 output features encoding (lines 339-442)
  - `generate_input_column_headers()` - Column names for CSV (lines 98-146)
  - `generate_output_column_headers()` - Action column names (lines 149-168)
  - This file is the **primary reference** for understanding the CSV structure

- **src/splendor/constants.py** - Game constants and color utilities
  - Color constants: WHITE, BLUE, GREEN, RED, BLACK, GOLD (lines 2-7)
  - Action types: BUILD, RESERVE, TOKENS (lines 15-17)
  - `bcolors` class for terminal colors (lines 49-57)
  - `getColor()` function for color mapping (lines 60-74)
  - `fullStrColor()` for color names (lines 37-46)
  - Card and noble definitions (DECK1, DECK2, DECK3, CHARACTERS)

- **src/splendor/cards.py** - Card class definition
  - Understand card attributes (vp, level, cost, bonus)

- **src/splendor/characters.py** - Noble/Character class definition
  - Understand noble attributes (vp, cost requirements)

- **src/splendor/move.py** - Move class definition
  - Understand action structure (actionType, action, tokensToRemove, character)

- **src/splendor/board.py** - Board state representation
  - Understanding game state structure is helpful for context

- **data/games/[2-4]_games/*.csv** - Existing CSV game files to replay

### New Files

- **scripts/replay_game.py** - Main replay script with:
  - CSV loading and parsing
  - Decoding logic (inverse of encoding in csv_exporter.py)
  - Terminal rendering with colors
  - Command-line interface

## Implementation Plan

### Phase 1: Foundation
Create the core decoding infrastructure that reverses the encoding logic from csv_exporter.py. This involves reading CSV files, parsing the 402 columns, and reconstructing game objects (cards, nobles, tokens, board state, player states, actions).

### Phase 2: Core Implementation
Build the visual rendering system that takes decoded game states and actions, then formats them for terminal display using colors, proper spacing, and clear labels. This includes creating helper functions for displaying cards, nobles, player states, and actions.

### Phase 3: Integration
Integrate the decoder with the renderer and create the command-line interface. Add the main script entry point that accepts CSV file paths and orchestrates the complete replay from start to finish.

## Step by Step Tasks

### Step 1: Create CSV Decoder Module
- Create `scripts/replay_game.py` with basic structure
- Implement CSV loading functionality using Python's `csv` module
- Create function to parse CSV headers (402 columns: game_id + 382 inputs + 20 outputs)
- Implement basic data validation (check column count, handle missing values)

### Step 2: Implement Input State Decoder
- Create `decode_game_state()` function that reverses `encode_game_state_from_board()`
- Decode basic info: num_players, turn_number, current_player (3 features)
- Decode board tokens: gems_board_* (6 features)
- Decode visible cards: card0-11 with vp, level, costs, bonus (12 cards × 12 features = 144)
- Decode deck remaining counts: deck_level1-3_remaining (3 features)
- Decode nobles: noble0-4 with vp and requirements (5 nobles × 6 features = 30)
- Decode players: player0-3 with position, vp, gems, reductions, reserved cards (4 players × 49 features = 196)
- Return structured dictionary with all decoded state information
- Handle NaN values appropriately (missing cards, players in 2-3 player games)

### Step 3: Implement Action Decoder
- Create `decode_action()` function that reverses `encode_action_from_move()`
- Decode action_type: build/reserve/take 2 tokens/take 3 tokens (1 feature)
- Decode card_selection: 0-14 position or NaN (1 feature)
- Decode card_reservation: 0-14 position or NaN (1 feature)
- Decode gem_take_3: which gems taken in take-3 action (5 features)
- Decode gem_take_2: which gems taken in take-2 action (5 features)
- Decode noble_selection: 0-4 position or -1 for none (1 feature)
- Decode gems_removed: tokens returned to board (6 features)
- Return structured dictionary with all action details

### Step 4: Create Visual Display Functions
- Implement `render_card()` - Display card with colors, costs, bonus, VP
- Implement `render_noble()` - Display noble with requirements and VP
- Implement `render_tokens()` - Display token counts with color coding
- Implement `render_player_state()` - Display player info: name, VP, tokens, built cards count, reserved cards
- Implement `render_board_state()` - Display complete board: tokens, visible cards (3 levels × 4 cards), nobles
- Implement `render_action()` - Display action details with reasoning (what was done and why)
- Use `bcolors` from constants.py for terminal coloring
- Add proper spacing, borders, and section headers for readability

### Step 5: Create Turn-by-Turn Renderer
- Implement `render_turn()` - Orchestrate display of complete turn
- Display turn separator/header with turn number and current player
- Show board state before action
- Show all player states (rotated so current player is first)
- Show action taken with detailed breakdown
- Show game statistics (progress toward VP goal, tokens remaining)
- Add visual separators between turns for easy scrolling

### Step 6: Implement Main Replay Loop
- Create `replay_game()` main function
- Load CSV file and validate format
- Extract game_id and number of players
- Iterate through all turns in CSV (each row = one turn)
- For each turn: decode state + action, then render
- Display game summary at the end (winner, total turns, final scores)

### Step 7: Add Command-Line Interface
- Add argument parsing using `argparse`
- Support `--file` or positional argument for CSV file path
- Add `--game-id` option to load specific game from data/games/N_games/ directory
- Add `--help` with usage instructions and examples
- Validate file existence and handle errors gracefully
- Add example usage in docstring

### Step 8: Test with Sample Games
- Test with 2-player game CSV
- Test with 3-player game CSV
- Test with 4-player game CSV
- Verify all action types display correctly (build, reserve, take tokens)
- Verify noble attractions display correctly
- Test edge cases: depleted decks, empty reserved slots, NaN values

### Step 9: Add Enhanced Visual Features
- Add color legend at start of replay
- Add turn navigation hints (e.g., "Scroll up/down to review turns")
- Use Unicode box-drawing characters for better visual structure
- Add summary statistics: average VP per turn, token efficiency
- Highlight important events (noble acquired, game-winning move)

### Step 10: Documentation and Examples
- Add comprehensive docstrings to all functions
- Create usage examples in script header
- Document CSV format expectations
- Add troubleshooting section for common issues

### Step 11: Run Validation Commands
- Execute all validation commands listed below
- Fix any errors or issues discovered
- Verify output is visually appealing and comprehensible
- Ensure zero regressions in existing functionality

## Testing Strategy

### Unit Tests
- **CSV Loading**: Test parsing of valid CSV files, handling of malformed CSVs
- **State Decoder**: Test decoding of each state component independently with known values
- **Action Decoder**: Test decoding of all action types (build, reserve, take 2, take 3)
- **NaN Handling**: Test proper handling of missing cards, players, nobles
- **Color Mapping**: Test color code generation for all gem types

### Integration Tests
- **Complete Turn Rendering**: Test full turn render with all components
- **Multi-Turn Replay**: Test replaying complete game from start to finish
- **Player Count Variations**: Test 2, 3, and 4 player games
- **Edge Cases**: Test games with depleted decks, maximum reserved cards, noble attractions

### Edge Cases
- CSV with missing columns or extra columns
- Games that end before VP goal (if applicable)
- All visible card slots depleted (deck empty)
- Player with 0 tokens
- Player with maximum tokens requiring discard
- Multiple nobles available to same player
- Action with gems_removed (token discard)
- Reserved card built from hand vs visible card

## Acceptance Criteria

1. Script successfully loads and parses any valid game CSV from data/games/
2. All 382 input features are correctly decoded back to interpretable game state
3. All 20 output features are correctly decoded to human-readable actions
4. Terminal output uses colors effectively to distinguish gem types
5. Each turn displays complete information: board state, all player states, action taken
6. Output is readable when scrolling up/down in terminal
7. Script handles 2, 3, and 4 player games correctly
8. NaN values (missing cards, players) are handled without errors
9. All action types (build, reserve, take 2, take 3) display correctly with details
10. Noble attractions are clearly indicated when they occur
11. Command-line interface is intuitive with helpful error messages
12. Script runs without errors on sample CSV files
13. Game summary at end shows winner and final state

## Validation Commands

Execute every command to validate the feature works correctly with zero regressions.

```bash
# Test replay with different player counts
python scripts/replay_game.py --file data/games/2_games/1.csv
python scripts/replay_game.py --file data/games/3_games/1.csv
python scripts/replay_game.py --file data/games/4_games/1.csv

# Test with game-id shorthand (if implemented)
python scripts/replay_game.py --game-id 1 --players 2

# Verify no syntax errors
python -m py_compile scripts/replay_game.py

# Check that script is executable and shows help
python scripts/replay_game.py --help

# Test with multiple games to verify consistency
python scripts/replay_game.py --file data/games/2_games/5.csv
python scripts/replay_game.py --file data/games/2_games/10.csv
```

## Notes

### Technical Considerations
- **Decoding accuracy**: The decoder must be the exact inverse of the encoder in csv_exporter.py. Any mismatch will cause incorrect rendering.
- **NaN handling**: Use `math.isnan()` to check for NaN values. These indicate missing data (e.g., 4th player in 2-player game, depleted card slot).
- **Color compatibility**: Terminal colors may not work in all environments. Consider detecting terminal support or adding a `--no-color` flag.
- **Performance**: Loading and rendering large games (50+ turns) should be fast. Consider optimizing if needed.

### Future Enhancements (Not in Current Scope)
- Interactive mode with pause/resume controls
- Export replay to HTML for sharing
- Side-by-side comparison of multiple games
- Filtering/searching for specific game situations
- Integration with database to replay by game_id without CSV export
- Animated transitions between turns
- Strategy annotations based on AI reasoning

### Dependencies
- No new external libraries required
- Uses only Python standard library:
  - `csv` for file parsing
  - `argparse` for CLI
  - `math` for NaN handling
  - `pathlib` for file paths
  - `sys` for terminal output

### Color Scheme Reference
From constants.py:
- WHITE: `\u001b[37m`
- BLUE: `\u001b[34m`
- GREEN: `\u001b[32m`
- RED: `\u001b[31m`
- BLACK: `\u001b[30m`
- GOLD (Yellow): `\u001b[33m`
- RESET: `\u001b[0m`

### CSV Structure Reference (402 columns)
1. **game_id** (1 column)
2. **Input features** (382 columns): num_players, turn_number, current_player, board tokens, visible cards, deck counts, nobles, players
3. **Output features** (20 columns): action_type, card_selection, card_reservation, gem_take_3 (5), gem_take_2 (5), noble_selection, gems_removed (6)
