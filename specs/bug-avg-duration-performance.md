# Bug: Average Duration Computation Performance Issue

## Bug Description
The `compute_avg_duration` function in `data_collector.py` is called on every single game iteration to calculate the average game duration. This function processes all log entries each time, which becomes increasingly expensive as the number of games grows (e.g., with 1000 games, it processes 1000 entries 1000 times). The calculation uses all historical games regardless of count, which is unnecessary since game duration stabilizes after ~50 games.

**Expected behavior**: Compute average duration efficiently by:
1. Limiting the calculation to the first 50 games only (ignoring subsequent games)
2. Computing the average either once at the start of the script or every 10 games while the game count for that player configuration is <= 50
3. Caching the result to avoid redundant calculations

**Actual behavior**: The average is recalculated from ALL log entries before every single game, causing O(n²) time complexity.

## Problem Statement
Performance degrades significantly as the simulation progresses because:
1. The average duration is computed before every game (line 515 in `data_collector.py`)
2. Each computation iterates through ALL log entries for that player count
3. With 1000 target games, this results in ~500,000 filtering operations
4. The average doesn't change meaningfully after ~50 games, making most calculations wasteful

## Solution Statement
Optimize the average duration computation by:
1. Modifying `compute_avg_duration` to limit its calculation to the first 50 games only
2. Caching the computed average after 50 games are reached
3. Implementing a strategy to update the average either:
   - Once at the start when game count < 50
   - Every 10 games while game count <= 50
   - Never after 50 games (use cached value)

## Steps to Reproduce
1. Start with a fresh database or fewer than 50 games
2. Run `python scripts/main.py` with a target of 1000+ games
3. Observe that every game triggers a full parse and filter of all log entries
4. Performance degradation becomes noticeable after ~100 games
5. With 1000 games, each game iteration processes increasingly large log files

## Root Cause Analysis
The root cause is in the main orchestration loop in `run_data_collection` (lines 508-548):

**Line 514-515**: On every game loop iteration:
```python
log_entries = parse_log_file(log_path)
avg_duration = compute_avg_duration(log_entries, nb_players)
```

**Line 230-247**: The `compute_avg_duration` function:
```python
def compute_avg_duration(log_entries: List[LogEntry], nb_players: int) -> Optional[float]:
    filtered = [e for e in log_entries if e.nb_players == nb_players]
    if not filtered:
        return None
    total_duration = sum(e.duration for e in filtered)
    return total_duration / len(filtered)
```

This causes:
1. Full log file parsing on every iteration (can be thousands of lines)
2. Full filtering of all entries by player count
3. Sum calculation over all matching entries
4. No caching or memoization
5. No limit on the number of games considered (should cap at 50)

## Relevant Files

### Modified Files

- **`scripts/data_collector.py:230-247`** - Contains the `compute_avg_duration` function that needs to be modified to:
  - Accept a `limit` parameter (default 50) to cap the number of entries used
  - Only use the first N entries matching the player count

- **`scripts/data_collector.py:456-575`** - Contains the `run_data_collection` function that needs to be modified to:
  - Cache the computed average duration once >= 50 games are available
  - Only recompute every 10 games when count <= 50
  - Use cached value when count > 50
  - Initialize the cache at the start of processing each player count

## Step by Step Tasks

### Step 1: Modify `compute_avg_duration` to limit entries
- Update function signature to accept `limit` parameter (default 50)
- Modify the filtering logic to only take the first `limit` entries that match the player count
- Update docstring to document the new parameter and behavior
- Ensure backward compatibility if `limit=None` means no limit (for testing)

### Step 2: Add caching logic to `run_data_collection`
- Initialize an `avg_duration_cache` variable before the main game loop for each player count
- Add logic to determine when to recompute:
  - When `current_count < 50`: compute every 10 games (i.e., when `current_count % 10 == 0` or `current_count == 0`)
  - When `current_count >= 50`: compute once and cache permanently for that player count
- Replace the unconditional computation on line 515 with conditional logic
- Use cached value when available instead of recomputing

### Step 3: Optimize log file parsing
- Consider caching the parsed log entries at the start of each player count processing
- Only re-parse the log file when needed (every 10 games for count <= 50, never for count > 50)
- Update both progress display calls (lines 514-526 and 551-561) to use the same cached values

### Step 4: Update progress display calls
- Ensure the progress display at line 518-526 uses the cached average
- Ensure the final progress display at line 551-561 uses the cached average
- Pass the cached average duration instead of recomputing each time

### Step 5: Test the changes
- Test with a fresh database (0 games)
- Test with < 50 games in database
- Test with exactly 50 games in database
- Test with > 50 games in database
- Test with 1000+ games to verify performance improvement
- Verify average duration is computed correctly in all scenarios
- Verify cache invalidation works correctly between different player counts

### Step 6: Run validation commands
- Execute all validation commands listed below to ensure zero regressions
- Verify that the average duration matches between old and new implementations
- Verify performance improvement by timing a run with 100+ games

## Validation Commands
Execute every command to validate the bug is fixed with zero regressions.

- `python -c "import sys; sys.path.insert(0, 'scripts'); import data_collector as dc; entries = dc.parse_log_file('data/simulation_log.txt'); print(f'Test 1: Parse log - {len(entries)} entries'); avg_old = dc.compute_avg_duration(entries, 2); print(f'Test 2: Old avg (all entries): {avg_old:.1f}s' if avg_old else 'Test 2: No data'); avg_new = dc.compute_avg_duration(entries, 2, limit=50); print(f'Test 3: New avg (50 entries): {avg_new:.1f}s' if avg_new else 'Test 3: No data')"` - Verify the modified `compute_avg_duration` works with the limit parameter

- `python -c "import sys; sys.path.insert(0, 'scripts'); import data_collector as dc; entries = dc.parse_log_file('data/simulation_log.txt'); filtered = [e for e in entries if e.nb_players == 2][:50]; manual_avg = sum(e.duration for e in filtered) / len(filtered) if filtered else None; computed_avg = dc.compute_avg_duration(entries, 2, limit=50); print(f'Manual avg: {manual_avg:.1f}s'); print(f'Computed avg: {computed_avg:.1f}s'); print(f'Match: {abs(manual_avg - computed_avg) < 0.01}' if manual_avg and computed_avg else 'No data')"` - Verify the limit logic is correctly implemented

- `python scripts/data_collector.py` - Run a full data collection session to verify caching works correctly (can interrupt after a few games)

- `python -c "import sys; sys.path.insert(0, 'scripts'); import data_collector as dc; dc.validate_all_systems('data/simulation_config.txt', 'data/games.db', 'data/simulation_log.txt')"` - Run comprehensive system validation

## Notes

### Performance Impact
- **Before**: O(n²) - for n games, we parse and filter n entries n times = n² operations
- **After**: O(n) - we parse once per 10 games until 50, then cache = ~5-10 parse operations total

### Why limit to 50 games?
Game duration stabilizes quickly in Monte Carlo simulations. After ~50 games, the average duration changes minimally (<1-2%). Computing from all games provides negligible accuracy improvement while significantly hurting performance.

### Cache invalidation
The cache is naturally invalidated between different player counts because we process them sequentially (2-player games, then 3-player, then 4-player). Each player count gets its own cache that lives only during that player count's processing phase.

### Backward compatibility
By adding `limit` as an optional parameter with a sensible default (50), the function remains backward compatible for any other code that might call it without the parameter.
