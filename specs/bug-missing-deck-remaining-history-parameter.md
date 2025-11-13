# Bug: Missing deck_remaining_history parameter in data_collector.py

## Bug Description
When running game simulations through `data_collector.py`, the program crashes with the error:
```
Error during game execution: saveIntoBdd() missing 1 required positional argument: 'deck_remaining_history'
```

The `saveIntoBdd()` function signature was updated to include a `deck_remaining_history` parameter (added in scripts/main.py:148), but the call to this function in `data_collector.py:435` was not updated to include this parameter. This causes the data collection system to fail when trying to save games to the database.

**Symptoms:**
- Game simulations crash during database save
- Error message indicates missing required argument
- Data collection process is completely blocked

**Expected behavior:**
- Games should be saved to database successfully with deck remaining counts
- Data collection should complete without errors

**Actual behavior:**
- TypeError is raised when calling saveIntoBdd without deck_remaining_history
- No games are saved to database
- Data collection fails

## Problem Statement
The `data_collector.py` module's `run_single_game()` function (line 435) calls `saveIntoBdd()` with only 7 arguments, but the function signature in `scripts/main.py:146-148` requires 8 arguments including `deck_remaining_history: List[List[int]]`. This parameter was added to support enhanced game data storage (tracking remaining deck counts per turn), but the data_collector module was not updated accordingly.

## Solution Statement
Update the `run_single_game()` function in `data_collector.py` to:
1. Track `deck_remaining_history` during game execution (capture deck sizes after each turn)
2. Pass the `deck_remaining_history` parameter when calling `saveIntoBdd()`

This matches the implementation pattern already present in `scripts/main.py:184-197,230` where deck remaining counts are tracked and passed to `saveIntoBdd()`.

## Steps to Reproduce
1. Ensure database is initialized: `python scripts/create_database.py && python scripts/load_database.py`
2. Configure simulation targets in `data/simulation_config.txt` (e.g., "2: 1")
3. Run data collection: `python scripts/main.py`
4. Observe the error: "saveIntoBdd() missing 1 required positional argument: 'deck_remaining_history'"

## Root Cause Analysis
The root cause is a **function signature mismatch** caused by incomplete refactoring:

1. **Change History**: The `deck_remaining_history` parameter was added to `saveIntoBdd()` in scripts/main.py:148 as part of the enhanced game data storage feature (specs/enhanced-game-data-storage.md)

2. **Incomplete Update**: The `PlayGame()` function in main.py was updated to track and pass this parameter (lines 184, 197, 230), but the `run_single_game()` function in data_collector.py was not updated

3. **Why It Happens**:
   - data_collector.py line 435 calls: `saveIntoBdd(state, winner, historyState, historyPlayers, historyActionPlayers, iterations, players)` (7 args)
   - main.py line 146-148 expects: `saveIntoBdd(state, winner, historyState, historyPlayers, historyActionPlayers, nbIte, Players, deck_remaining_history)` (8 args)

4. **Missing Logic**: The data_collector.py game loop (lines 418-429) does not capture deck remaining counts like main.py does (line 197)

## Relevant Files
Use these files to fix the bug:

- **scripts/data_collector.py** (lines 376-450) - Contains the `run_single_game()` function that calls `saveIntoBdd()` without the deck_remaining_history parameter
  - Line 413: Initialize historyActionPlayers (need to add deck_remaining_history initialization nearby)
  - Line 418-429: Game execution loop (need to capture deck counts per turn)
  - Line 435: Call to saveIntoBdd (need to add deck_remaining_history parameter)

- **scripts/main.py** (lines 146-230) - Reference implementation showing correct usage
  - Line 148: Function signature with deck_remaining_history parameter
  - Line 184: Initialization of deck_remaining_history list
  - Line 197: Capturing deck counts during game loop
  - Line 230: Correct call to saveIntoBdd with all parameters

## Step by Step Tasks

### 1. Initialize deck_remaining_history in run_single_game
- In scripts/data_collector.py around line 413, add initialization of `deck_remaining_history` list after other history lists are initialized
- Use the same pattern as in scripts/main.py:184: `deck_remaining_history: List[List[int]] = []`

### 2. Capture deck remaining counts during game loop
- In scripts/data_collector.py in the game execution loop (after line 419 where `historyState.append(state.getState())` would be)
- Add deck count capture: `deck_remaining_history.append([len(state.deckLVL1), len(state.deckLVL2), len(state.deckLVL3)])`
- This should happen once per turn, similar to scripts/main.py:197
- Place it right after `historyState.append(state.getState())` to maintain consistency

### 3. Pass deck_remaining_history to saveIntoBdd
- Update line 435 in scripts/data_collector.py
- Change from: `saveIntoBdd(state, winner, historyState, historyPlayers, historyActionPlayers, iterations, players)`
- Change to: `saveIntoBdd(state, winner, historyState, historyPlayers, historyActionPlayers, iterations, players, deck_remaining_history)`
- This matches the call in scripts/main.py:229-230

### 4. Validate the fix
- Run the validation commands to ensure the bug is fixed with zero regressions
- Test with a small number of games (2 players: 1 game) to verify the fix works
- Verify database is populated with deck remaining counts
- Ensure no other errors occur during game execution and saving

## Validation Commands
Execute every command to validate the bug is fixed with zero regressions.

- `python scripts/create_database.py` - Initialize database schema
- `python scripts/load_database.py` - Load cards and characters
- `echo "2: 1" > data/simulation_config.txt` - Configure single test game
- `python scripts/main.py` - Run data collection (should complete without errors)
- `python -c "import sqlite3; conn = sqlite3.connect('data/games.db'); cursor = conn.cursor(); cursor.execute('SELECT COUNT(*) FROM Game'); print(f'Games saved: {cursor.fetchone()[0]}'); cursor.execute('SELECT COUNT(*) FROM StateGame WHERE Deck1Remaining IS NOT NULL'); print(f'States with deck data: {cursor.fetchone()[0]}'); conn.close()"` - Verify game was saved with deck data

## Notes
- The bug was introduced when the enhanced game data storage feature was implemented, which added deck remaining tracking to the database schema
- The fix is minimal and surgical: only adds the missing tracking and parameter passing to data_collector.py
- No changes are needed to the database schema or saveIntoBdd function itself
- The fix follows the exact same pattern already implemented in main.py's PlayGame function
- This is a critical bug that blocks all data collection, so it should be fixed with high priority
