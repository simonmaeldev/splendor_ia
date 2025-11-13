# Enhanced Game Data Storage and Real-Time CSV Export

**ADW ID:** N/A
**Date:** 2025-11-12
**Specification:** specs/enhanced-game-data-storage.md

## Overview

This feature enhances the game data storage system to save comprehensive state information directly in the database and automatically export ML-ready data to CSV files during gameplay. This eliminates the need for expensive post-processing (which previously took several minutes to recalculate ~1.7M state-action pairs) and ensures all ML training features are available immediately after each game completes.

## What Was Built

- **Enhanced database schema** with 17 new columns capturing victory points, gem reductions, reserved cards, and deck remaining counts
- **Real-time CSV exporter module** (`csv_exporter.py`) that encodes game states into 381 input features + 20 output features
- **Fixed state capture timing** to record player state BEFORE action execution (critical for ML training)
- **Dual CSV export system** generating both game-specific CSVs and an aggregated all_games.csv file
- **Deep copy safety** throughout data collection to prevent reference bugs
- **Automatic directory structure** for organizing games by player count (2/3/4 players)

## Technical Implementation

### Files Modified

- `scripts/create_database.py`: Updated schema with new columns
  - Added to StatePlayer: VictoryPoints, ReductionWhite/Blue/Green/Red/Black, ReservedCard1/2/3
  - Added to StateGame: DeckLevel1/2/3Remaining

- `scripts/main.py`: Enhanced game loop and data collection
  - Moved player state capture to BEFORE doMove() (line 127) - critical timing fix
  - Added deck remaining count tracking
  - Integrated real-time CSV export at end of each game
  - Updated saveIntoBdd() to reconstruct and save enriched state data

- `src/splendor/csv_exporter.py`: New 545-line module for CSV export
  - Encoding functions for cards (12 features), nobles (6 features), player states (49 features)
  - Complete game state encoder (381 features total)
  - Action encoder (20 output features across 7 heads)
  - CSV writing functions with NaN padding for variable player counts
  - Uses integers for discrete values, floats only for NaN

- `.gitignore`: Added patterns to exclude large CSV files from version control

- Directory structure created:
  - `data/games/2_games/` - Individual 2-player game CSVs
  - `data/games/3_games/` - Individual 3-player game CSVs
  - `data/games/4_games/` - Individual 4-player game CSVs
  - `data/games/all_games.csv` - Aggregated CSV with all games

### Key Changes

1. **Database Schema Enhancement**: Added 17 new columns to capture computed state (victory points from cards+nobles, gem reductions from built card bonuses, reserved card references, and remaining cards in each deck level)

2. **State Capture Timing Fix**: Critical change - moved player state capture from AFTER action execution to BEFORE action execution. This ensures ML models train on the state that led to a decision, not the resulting state.

3. **Real-Time CSV Export**: Game data is now encoded and written to CSV files immediately after each game completes, using the same 381+20 feature format as export_ml_dataset.py but without expensive recalculation.

4. **Data Type Optimization**: All discrete values (counts, IDs, positions) are stored as integers in CSV, with floats only used for NaN padding. This reduces file size and improves ML pipeline efficiency.

5. **Reference Bug Prevention**: Implemented deep copies throughout (using `deepcopy()` and list comprehensions instead of `[[]] * n`) to ensure independent state snapshots.

## How to Use

### Running Games with Enhanced Storage

1. **Initialize the database** with the new schema:
   ```bash
   python scripts/create_database.py
   python scripts/load_database.py
   ```

2. **Run a game** - CSV export happens automatically:
   ```bash
   python scripts/main.py
   ```
   Or run a quick test:
   ```bash
   python scripts/test_quick.py
   ```

3. **Locate the exported data**:
   - Game-specific CSV: `data/games/{nb_players}_games/{game_id}.csv`
   - Aggregated CSV: `data/games/all_games.csv`
   - Database: `data/games.db` (with enriched StatePlayer and StateGame tables)

### CSV File Structure

Each CSV file contains **402 columns**:
- 1 column: `game_id`
- 381 columns: Input features (game state)
  - num_players, turn_number
  - gems_board (6)
  - visible_cards (12 cards × 12 features = 144)
  - deck_remaining (3)
  - nobles (5 nobles × 6 features = 30)
  - players (4 players × 49 features = 196, with NaN padding for 2-3 player games)
- 20 columns: Output features (action taken)
  - action_type (string)
  - card_selection (int or NaN)
  - card_reservation (int or NaN)
  - gem_take_3 (5 ints or NaN)
  - gem_take_2 (5 ints or NaN)
  - noble_selection (int)
  - gems_removed (6 ints)

### Verifying Data Integrity

Run the test suite to verify everything works:

```bash
# Test database schema
sqlite3 data/games.db "SELECT VictoryPoints, ReductionWhite FROM StatePlayer LIMIT 5;"

# Test CSV structure
head -1 data/games/all_games.csv | tr ',' '\n' | wc -l  # Should be 402

# Run full integration test
python scripts/test_enhanced_storage.py
```

## Configuration

No additional configuration required. The system integrates seamlessly with existing game simulation infrastructure:

- **Backward compatible**: Existing code continues to work
- **Automatic directory creation**: CSV directories are created if they don't exist
- **Graceful error handling**: CSV export failures are caught and logged without crashing the game

### .gitignore Patterns

Large CSV files are automatically excluded from version control:
```
data/games/all_games.csv
data/games/2_games/*.csv
data/games/3_games/*.csv
data/games/4_games/*.csv
```

## Testing

### Quick Test
```bash
python scripts/test_quick.py
# Runs a 2-player game with 10 ISMCTS iterations
# Verifies CSV export completes successfully
```

### Comprehensive Test
```bash
python scripts/test_enhanced_storage.py
# Tests:
# - Database schema correctness
# - CSV file creation and structure (402 columns)
# - Data consistency between DB and CSV
# - Reference safety (no shared references)
# - State capture timing
```

### Manual Verification
```bash
# Count games
sqlite3 data/games.db "SELECT COUNT(*) FROM Game"

# Check enriched columns
sqlite3 data/games.db "SELECT VictoryPoints, ReductionWhite, ReservedCard1 FROM StatePlayer LIMIT 5"

# Check deck counts
sqlite3 data/games.db "SELECT DeckLevel1Remaining, DeckLevel2Remaining FROM StateGame LIMIT 5"

# Verify CSV structure
wc -l data/games/2_games/*.csv
head -2 data/games/all_games.csv | tail -1 | cut -d',' -f1-20
```

## Notes

### Performance Impact

- **CSV export overhead**: Minimal (~100ms per game). Export happens BEFORE database save to preserve data even if DB save fails.
- **Memory usage**: Deep copies add overhead but ensure data integrity. Not significant for individual games.
- **Disk space**: Each game generates ~100KB CSV. For large simulations (1000s of games), expect several hundred MB.

### Data Consistency

- **State timing**: StatePlayer now represents state BEFORE action (fixed from AFTER)
- **Victory points**: Reconstructed from action history (built cards + nobles)
- **Reductions**: Calculated from built card bonuses at each turn
- **Reserved cards**: Tracked through reserve/build action history

### Comparison to export_ml_dataset.py

The new real-time CSV export:
- ✅ Produces identical 402-column format
- ✅ Uses same encoding logic (cards, nobles, players, actions)
- ✅ Handles NaN padding identically
- ✅ Much faster (no database queries, direct state encoding)
- ✅ More reliable (data saved even if simulation crashes)

### Future Enhancements

Potential improvements noted in specification but not yet implemented:
- CSV file compression (gzip) to save disk space
- Checksum validation for data integrity
- Incremental export mode (resume from partial all_games.csv)
- Data deduplication (skip already-exported games)
- CSV file rotation (split all_games.csv when it gets too large)
- Parallel CSV writing for faster bulk exports

### Migration Notes

If upgrading from the old system:
1. Recreate database: `python scripts/create_database.py`
2. Reload reference data: `python scripts/load_database.py`
3. Old games in database won't have enriched columns (will be NULL/0)
4. New games automatically get full enriched data
5. Can still use export_ml_dataset.py on old data if needed
