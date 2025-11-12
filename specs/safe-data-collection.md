# Feature: Safe Data Collection with Progress Tracking

## Feature Description
A robust data collection system for running MCTS game simulations that safely persists game data to the database, tracks progress through logs, and provides detailed terminal output with time estimates. The system is designed to handle interruptions (Ctrl+C, shutdowns) gracefully by ensuring only complete games are committed to the database. It reads target game counts from a configuration file and automatically continues across different player counts until all targets are met.

## User Story
As a machine learning researcher
I want to safely collect large amounts of game simulation data without corrupting the database
So that I can gather training data while being able to interrupt the process at any time without losing data integrity

## Problem Statement
The current implementation runs an infinite loop and commits games to the database, but lacks:
1. **No progress tracking**: Users cannot see how much data has been collected or how long until completion
2. **No target-based collection**: No way to specify how many games are needed for different player counts
3. **No interruption safety**: While database commits are atomic, there's no guarantee about partial game states
4. **No persistence of progress**: After interruption, there's no record of what has been run
5. **No time estimates**: Users cannot plan around simulation durations

## Solution Statement
Implement a configuration-driven data collection system that:
1. Reads target game counts from `data/simulation_config.txt` with format `nb_player: nb_games`
2. Queries the database to determine remaining games needed for each player count
3. Runs simulations only until targets are met, automatically moving between player counts
4. Logs each completed game to `data/simulation_log.txt` with timing information
5. Only commits complete games to the database (atomic transactions)
6. Displays a rich terminal UI with progress bars, time estimates, and average game duration
7. Parses log files to compute accurate statistics and time estimates

## Relevant Files

### Existing Files
- **scripts/main.py** (lines 105-143)
  - Contains the main game execution loop that needs to be replaced
  - Contains `PlayGame()` function that runs individual games
  - Contains `saveIntoBdd()` that commits games atomically (already safe)

- **scripts/create_database.py**
  - Contains database schema creation
  - No changes needed but important for understanding table structure

- **README.md** (lines 115-187)
  - Documents the database schema used for queries
  - Important reference for understanding Game table structure

### New Files

#### data/simulation_config.txt
Configuration file specifying target game counts per player count. Format:
```
2: 500
3: 1000
4: 500
```

#### data/simulation_log.txt
Log file tracking completed games. Format (one line per game):
```
<nb_players> players, <nb_turns> turns, started: <time_start>, ended: <time_end>, duration: <duration_seconds>s, <sim_num>/<total_sims>
```

#### scripts/data_collector.py
New module containing:
- Configuration file parsing
- Database query functions for game counts
- Log file parsing and statistics computation
- Progress tracking and display formatting
- Main simulation orchestration loop

## Implementation Plan

### Phase 1: Foundation
Create the configuration and logging infrastructure:
- Define configuration file format and parsing logic
- Define log file format and parsing logic
- Create database query functions to count existing games by player count
- Implement log statistics computation (average duration, game counts)

### Phase 2: Core Implementation
Build the safe data collection loop:
- Create main simulation orchestrator that reads config and determines what to run
- Implement progress display with completion percentage and time estimates
- Wrap game execution in try-except to ensure atomic commits
- Add log writing after successful database commits
- Implement graceful shutdown on KeyboardInterrupt

### Phase 3: Integration
Connect the new system to existing game execution:
- Import existing `PlayGame()` and `saveIntoBdd()` functions
- Modify main.py to use the new data collector when run
- Add command-line arguments for flexibility (optional)
- Test end-to-end with small game counts

## Step by Step Tasks

### 1. Create configuration file format and parser
- Create `data/simulation_config.txt` with sample configuration (2:10, 3:10, 4:10 for testing)
- Implement `parse_config(filepath: str) -> Dict[int, int]` function that reads and parses the config file
- Handle file not found, invalid format, and empty file cases
- Add validation: player counts must be 2-4, game counts must be positive integers

### 2. Create log file format and parser
- Define log entry dataclass with fields: nb_players, nb_turns, time_start, time_end, duration, sim_num, total_sims
- Implement `parse_log_file(filepath: str) -> List[LogEntry]` function that reads and parses existing logs
- Implement `write_log_entry(filepath: str, entry: LogEntry) -> None` that appends one line to the log
- Handle missing log file (create new), malformed lines (warn and skip with count), and parse datetime strings

### 3. Implement database query functions
- Create `get_game_count_by_players(db_path: str, nb_players: int) -> int` that queries the Game table
- SQL: `SELECT COUNT(*) FROM Game WHERE NbPlayers = ?`
- Test with existing database to ensure correct counts

### 4. Implement log statistics computation
- Create `compute_avg_duration(log_entries: List[LogEntry], nb_players: int) -> Optional[float]` that filters by player count and computes mean duration
- Return None if no entries found for that player count (display as "NaN")
- Handle edge cases: empty log file, no entries for specific player count

### 5. Implement progress display formatting
- Create `format_progress_display(...)` function that takes current state and returns formatted string
- Include: `{completed}/{total} games for {nb_players} players`
- Include: `Started on: {day} {month}, {hh:mm}` in 24-hour format
- Include: `Average duration: {avg}s` or "NaN" if no data
- Include: `Estimated time remaining: {days}d, {hh:mm}`
- Include: `Estimated completion: {day} {month}, {hh:mm}`
- Include: ASCII progress bar: `[========>         ] 45%`
- Use datetime module for all time formatting and calculations

### 6. Create main simulation orchestrator
- Create `scripts/data_collector.py` with main orchestration logic
- Implement `run_data_collection(config_path: str, db_path: str, log_path: str) -> None`
- Loop through each player count in config (sorted order: 2, 3, 4)
- For each player count: query database, calculate remaining games, run simulations
- Store session start time for time estimates
- After each player count completes, automatically move to next

### 7. Implement safe game execution wrapper
- Create `run_single_game_safe(nb_players: int, iterations: List[int], algorithm: str, db_path: str, log_path: str, sim_num: int, total_sims: int, session_start: datetime) -> bool`
- Wrap `PlayGame()` call in try-except to catch any exceptions
- Record start time before game, end time after game
- Only commit to database if game completes successfully (saveIntoBdd already atomic)
- Only write log entry after successful database commit
- Return True if successful, False if failed
- On failure: print error but continue to next game (game data is not committed)

### 8. Add progress display between games
- After each successful game, parse log file to get updated statistics
- Compute and display progress information using `format_progress_display()`
- Clear previous display and redraw (use `\r` and ANSI codes or simple print)
- Update every game completion

### 9. Implement graceful shutdown handling
- Wrap main loop in try-except for KeyboardInterrupt
- On Ctrl+C: print summary of work completed
- Display: "Interrupted. {completed} games saved. Database is safe."
- Exit cleanly without partial commits

### 10. Integrate with existing main.py
- Modify `scripts/main.py` to import and call new data collector
- Replace infinite loop (lines 135-143) with call to `run_data_collection()`
- Set default paths: `data/simulation_config.txt`, `data/games.db`, `data/simulation_log.txt`
- Keep backward compatibility: if config file doesn't exist, print helpful message

### 11. Add configuration file creation helper
- If `data/simulation_config.txt` doesn't exist on first run, print example format
- Suggest user create the file before running
- Provide clear error message with example

### 12. Create test configuration and validate
- Create test config: `2: 5, 3: 5, 4: 5` (15 games total)
- Run validation tests:
  - Test config parsing with valid and invalid files
  - Test log parsing with valid and invalid entries
  - Test database queries with existing data
  - Test progress calculations with mock data

### 13. Run end-to-end test with small dataset
- Clear any test games from database or use fresh test database
- Run simulation with test config (15 games)
- Verify: config is read correctly, progress displays update, log file is written, database contains exactly 15 new games
- Test interruption: start run, Ctrl+C after 2 games, verify only 2 games committed, restart and verify it continues from where it left off

### 14. Add edge case handling
- Handle case: target already met (skip that player count, display message)
- Handle case: all targets met (display "All targets complete!" and exit)
- Handle case: database connection errors (retry once, then fail with clear message)
- Handle case: log file write errors (warn but don't fail the game commit)

### 15. Final validation and documentation
- Run with realistic configuration (100+ games)
- Verify time estimates are accurate
- Verify progress bar displays correctly
- Test graceful shutdown multiple times
- Verify log file format is consistent and parseable
- Run validation commands to ensure no regressions

## Testing Strategy

### Unit Tests
- **Config parsing**: Test valid formats, invalid formats, missing file, empty file, out-of-range values
- **Log parsing**: Test valid entries, malformed entries, empty file, mixed valid/invalid
- **Database queries**: Test with 0 games, some games, verify counts match expected
- **Statistics computation**: Test with no data (returns None/NaN), single entry, multiple entries, filtering by player count
- **Time formatting**: Test datetime parsing, duration calculations, estimate computations
- **Progress bar**: Test 0%, 50%, 100%, edge cases

### Integration Tests
- **Full simulation run**: Start with clean database, run 10 games, verify all commits successful
- **Resume after interruption**: Run 5 games, interrupt, verify exactly 5 committed, restart, verify continues correctly
- **Multi-player-count**: Config with 2:3, 3:3, verify it runs 3 of each automatically
- **Log persistence**: Run games, verify log file contains correct number of entries with valid format
- **Database safety**: Simulate crash during game (exception), verify database not corrupted

### Edge Cases
- **Config file missing**: Should display helpful error message with example format
- **Log file corrupted**: Should warn about skipped lines but continue
- **Database locked**: Should retry and/or fail gracefully with clear error
- **Target already met**: Should skip and move to next player count
- **Zero games remaining**: Should display completion message and exit
- **Invalid player count in config**: Should validate and reject (only 2-4 allowed)
- **Negative or zero game count**: Should validate and reject
- **Interrupt during database commit**: Database transaction should handle atomicity

## Acceptance Criteria
1. ✅ Configuration file `data/simulation_config.txt` can specify targets like `3: 1000`
2. ✅ System reads database to determine how many games already exist for each player count
3. ✅ System only runs remaining games needed to meet targets
4. ✅ Each completed game is logged to `data/simulation_log.txt` with all required fields
5. ✅ Log entries are only written after successful database commits
6. ✅ Progress display shows between each game with all required information:
   - Games completed/total for current player count
   - Average game duration (or NaN)
   - Start time in "Day Month, HH:MM" format
   - Estimated time remaining in "Xd, HH:MM" format
   - Estimated completion date in "Day Month, HH:MM" format
   - Completion percentage with progress bar
7. ✅ System handles Ctrl+C gracefully without corrupting database
8. ✅ System automatically moves to next player count when current target is met
9. ✅ System exits when all targets are met with completion message
10. ✅ Invalid log entries are warned about (with count) but skipped, allowing continuation
11. ✅ System works correctly after interruption and restart (resumes from current state)

## Validation Commands
Execute every command to validate the feature works correctly with zero regressions.

- `python -c "import scripts.data_collector as dc; config = dc.parse_config('data/simulation_config.txt'); print('Config valid' if config else 'Config invalid')"` - Validate config parsing
- `python -c "import scripts.data_collector as dc; entries = dc.parse_log_file('data/simulation_log.txt'); print(f'Parsed {len(entries)} log entries')"` - Validate log parsing
- `python -c "import scripts.data_collector as dc; count = dc.get_game_count_by_players('data/games.db', 3); print(f'Found {count} 3-player games')"` - Validate database queries
- `python scripts/main.py` - Run main data collection (interrupt with Ctrl+C to test graceful shutdown)
- `python -c "import sqlite3; conn = sqlite3.connect('data/games.db'); cursor = conn.cursor(); cursor.execute('SELECT COUNT(*) FROM Game'); print(f'Total games in database: {cursor.fetchone()[0]}'); conn.close()"` - Verify database integrity
- `cat data/simulation_log.txt | tail -5` - Verify log file format
- `python -c "import scripts.data_collector as dc; dc.validate_all_systems('data/simulation_config.txt', 'data/games.db', 'data/simulation_log.txt')"` - Run comprehensive system validation

## Notes

### Implementation Details
- Use `datetime` module for all time tracking and formatting (strftime/strptime)
- Use `sqlite3` module for database queries (read-only queries for counting)
- Keep `saveIntoBdd()` as the single point of database writes (already atomic)
- Log file is append-only, never modify existing entries
- Progress bar can be simple ASCII: `[=====>    ]` with percentage

### Algorithm Selection
- Default to ISMCTS_PARA for all simulations (fastest)
- Default iterations: 1000 per player (as per current main.py)
- These can be configured in data_collector.py constants

### Time Estimate Accuracy
- Average duration computed from log file only for matching player count
- Estimates improve as more games complete
- Display "NaN" when no historical data available
- Consider: first few games may be slower (cold start), averages stabilize over time

### Database Safety
- `saveIntoBdd()` already uses single `conn.commit()` at end (atomic)
- Games are fully constructed in memory before commit
- If exception during game execution, saveIntoBdd is never called
- Database remains in consistent state even with interruptions

### Future Enhancements
- Command-line arguments for config/log/db paths
- Support for different algorithms per player count in config
- Support for different iteration counts per player in config
- Parallel game execution (multiple games at once)
- Web dashboard for monitoring progress
- Email/notification on completion

### Example Terminal Output
```
=== Splendor Data Collection ===
Config: 2 players: 500 games | 3 players: 1000 games | 4 players: 500 games

[2 players] 245/500 games (49%)
[===========================>                         ] 49%
Started on: 31 Oct, 14:30
Average duration: 45.3s
Estimated remaining: 0d, 03:12
Estimated completion: 31 Oct, 17:42

Running game 246/500...
```

### Dependencies
- No new external dependencies required
- Uses Python standard library: `sqlite3`, `datetime`, `typing`, `pathlib`, `re`, `dataclasses`
