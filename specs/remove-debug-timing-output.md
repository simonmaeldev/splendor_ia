# Bug: Remove Verbose Debug Timing Output

## Bug Description
The terminal displays excessive debug output during game simulations showing timing information for each game iteration. The output includes:
- `time simulation : X.XXX` - Time spent in parallel simulation
- `total time: X.XXX` - Total time including tree merging

This output clutters the terminal and makes it difficult to track the actual game progress. The user wants to keep the `running game X/Y` progress indicator but remove the timing debug lines.

## Problem Statement
Two print statements in the ISMCTS_para function (`src/splendor/ISMCTS.py:76` and `:82`) output timing information to the console for every MCTS iteration. These debug statements are taking up too much screen space and need to be disabled while preserving the game progress indicator that shows which game is currently being simulated.

## Solution Statement
Comment out or remove the two print statements that output timing information in the `ISMCTS_para` function in `src/splendor/ISMCTS.py`. This will eliminate the verbose timing output while keeping the "running game X/Y" message that comes from the data collector.

## Steps to Reproduce
1. Run the data collection script: `python scripts/main.py`
2. Observe the terminal output
3. See repeated timing lines (`time simulation` and `total time`) for each MCTS call
4. Note that these lines appear multiple times per game (once per player turn)

## Root Cause Analysis
The root cause is two debug print statements in the `ISMCTS_para` function:
- Line 76: `print(f'time simulation : {timer() - start}')` - prints time after parallel tree computation
- Line 82: `print(f'total time: {end - start}')` - prints total execution time

These were likely added during development for performance profiling but were left in the production code. They execute on every call to `ISMCTS_para`, which happens multiple times per turn (once per player), resulting in excessive console output.

## Relevant Files
Use these files to fix the bug:

- `src/splendor/ISMCTS.py:76,82` - Contains the two print statements outputting timing debug information
  - Line 76: Prints simulation time after parallel MCTS tree computation
  - Line 82: Prints total execution time including tree merging
  - These are the only lines that need to be modified to fix the bug

## Step by Step Tasks

### 1. Remove timing debug output from ISMCTS_para function
- Open `src/splendor/ISMCTS.py`
- Comment out or remove line 76: `print(f'time simulation : {timer() - start}')`
- Comment out or remove line 82: `print(f'total time: {end - start}')`
- Preserve all other functionality in the function

### 2. Verify the fix
- Run the validation commands to ensure the bug is fixed
- Confirm that "running game X/Y" still appears but timing lines do not
- Ensure no other functionality is affected

## Validation Commands
Execute every command to validate the bug is fixed with zero regressions.

- `python scripts/main.py` - Start a simulation and verify timing lines are gone but "running game X/Y" remains (can interrupt with Ctrl+C after seeing a few games)
- `grep -n "print.*time simulation" src/splendor/ISMCTS.py` - Verify the timing print statement is commented out or removed
- `grep -n "print.*total time" src/splendor/ISMCTS.py` - Verify the total time print statement is commented out or removed

## Notes
- The fix only requires modifying 2 lines in a single file
- No logic changes are needed - only removing debug output
- The timer variables and calculations can remain in the code (they're harmless even if unused)
- If timing information is needed in the future for profiling, these lines can be easily uncommented or controlled via a debug flag
- The "running game X/Y" message comes from `scripts/data_collector.py:527` and will not be affected by this change
