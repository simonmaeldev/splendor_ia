# Bug: CSV Export Gem Take Encoding Mismatch

## Bug Description
The CSV export logic incorrectly encodes token-taking actions when MCTS intends to "take 3 tokens" but can only take 2 tokens due to board state. In this scenario:
- The `action_type` correctly shows "take 3 tokens" (reflecting the MCTS intent)
- But the token values are placed in `gem_take2` columns instead of `gem_take3` columns
- This creates a mismatch where `action_type = "take 3 tokens"` but `gem_take3` columns are NaN and `gem_take2` columns contain the actual values

The bug does NOT occur when:
- Taking 2 tokens of the same color (action_type = "take 2 tokens") - correctly uses `gem_take2` columns
- Taking 3 tokens successfully (action_type = "take 3 tokens") - correctly uses `gem_take3` columns

## Problem Statement
The encoding logic in `src/splendor/csv_exporter.py` uses the actual token count (`sum(take_tokens[:5])`) to determine which columns to populate, rather than using the `action_type_str` that correctly identifies the MCTS intent. This causes a mismatch between the action type label and the actual encoding columns used.

## Solution Statement
Fix the encoding logic to populate `gem_take3` columns when `action_type_str = "take 3 tokens"`, regardless of how many tokens were actually taken. The action_type is already correctly determined based on the pattern (contains a 2 = "take 2 tokens", else = "take 3 tokens"), so we should use this same logic for column selection.

Additionally, create a script to repair existing CSV files by:
1. Finding rows where `action_type = "take 3 tokens"` AND `gem_take3` columns are all NaN AND `gem_take2` columns are not all NaN
2. Moving the values from `gem_take2` columns to `gem_take3` columns
3. Setting `gem_take2` columns to NaN

## Steps to Reproduce
1. Examine `data/games/2_games/5.csv` at line 11 (CSV row 12):
   ```bash
   cut -d',' -f384,387-396 data/games/2_games/5.csv | sed -n '12p'
   ```
   Expected output shows: `action_type = "take 3 tokens"` but values in `gem_take2` columns

2. Compare with line 15 (CSV row 16) which correctly has `action_type = "take 2 tokens"` with values in `gem_take2` columns

## Root Cause Analysis
In `src/splendor/csv_exporter.py`, the `encode_action_from_move()` function (lines 339-442):

1. Lines 374-378: Correctly determines `action_type_str` based on token pattern:
   - If any token value is 2 → "take 2 tokens"
   - Otherwise → "take 3 tokens"

2. Lines 411-419: INCORRECTLY uses actual token sum to decide which columns to populate:
   ```python
   # Gem take 3 - integers (0 or 1)
   gem_take_3 = [float('nan')] * 5
   if action_type == TOKENS and sum(take_tokens[:5]) == 3:
       gem_take_3 = [take_tokens[i] for i in range(5)]

   # Gem take 2 - integers (0 or 2)
   gem_take_2 = [float('nan')] * 5
   if action_type == TOKENS and sum(take_tokens[:5]) == 2:
       gem_take_2 = [take_tokens[i] for i in range(5)]
   ```

The issue: When MCTS wants to take 3 tokens but only 2 are available:
- `take_tokens` = [0, 0, 1, 0, 1] (sum = 2)
- Line 375 check: `any(t == 2 for t in take_tokens[:5])` = False → `action_type_str = "take 3 tokens"` ✓
- Line 413 check: `sum(take_tokens[:5]) == 3` = False → `gem_take_3` stays as NaN ✗
- Line 418 check: `sum(take_tokens[:5]) == 2` = True → `gem_take_2` gets populated ✗

## Relevant Files
Use these files to fix the bug:

### `src/splendor/csv_exporter.py`
- Lines 339-442: `encode_action_from_move()` function contains the buggy logic
- Lines 374-378: Correct action_type determination
- Lines 411-419: Incorrect column selection based on actual token sum

### New Files

#### `scripts/fix_csv_gem_encoding.py`
- Script to repair existing CSV files in `data/games/2_games/`, `data/games/3_games/`, and `data/games/4_games/`
- Should find and fix rows where action_type is "take 3 tokens" but values are in gem_take2 columns

## Step by Step Tasks
IMPORTANT: Execute every step in order, top to bottom.

### Step 1: Fix the CSV Exporter Logic
- Modify `src/splendor/csv_exporter.py` in the `encode_action_from_move()` function
- Change lines 411-419 to use the same logic as action_type determination (lines 374-378)
- If `any(t == 2 for t in take_tokens[:5])` is True, populate `gem_take_2`
- Otherwise, populate `gem_take_3`
- Ensure the logic matches: action_type determination and column population should be consistent

### Step 2: Create CSV Repair Script
- Create `scripts/fix_csv_gem_encoding.py` that:
  - Iterates through all CSV files in `data/games/2_games/`, `data/games/3_games/`, `data/games/4_games/`
  - For each CSV file:
    - Read all rows
    - Find rows where `action_type == "take 3 tokens"` AND all `gem_take3` columns are NaN AND any `gem_take2` column is not NaN
    - For those rows: swap `gem_take2` values into `gem_take3` columns and set `gem_take2` to NaN
    - Write the corrected data back to the same file
  - Print summary of files processed and rows fixed

### Step 3: Test the Fix on Sample Data
- Run the repair script on `data/games/2_games/5.csv` first
- Verify that line 11 now has values in `gem_take3` columns instead of `gem_take2`
- Verify that line 15 remains unchanged (action_type = "take 2 tokens" should keep values in `gem_take2`)
- Use command:
  ```bash
  cut -d',' -f384,387-396 data/games/2_games/5.csv | sed -n '1p;12p;16p'
  ```

### Step 4: Run Repair Script on All CSV Files
- Execute the repair script on all CSV files in the three directories
- Review the summary output to confirm:
  - Number of files processed
  - Number of rows fixed per file
  - Total rows fixed across all files

### Step 5: Validate the Fixes
- Randomly sample 5-10 CSV files from different directories
- Verify that no rows have `action_type = "take 3 tokens"` with values in `gem_take2` columns
- Verify that rows with `action_type = "take 2 tokens"` still have values in `gem_take2` columns
- Run validation commands from the "Validation Commands" section

## Validation Commands
Execute every command to validate the bug is fixed with zero regressions.

- `python -c "import sys; sys.path.insert(0, 'src'); from splendor.csv_exporter import encode_action_from_move; print('CSV exporter module loads successfully')"` - Verify the module loads without errors

- `cut -d',' -f384,387-396 data/games/2_games/5.csv | sed -n '12p'` - Verify line 11 (row 12) now has values in gem_take3 columns

- `cut -d',' -f384,387-396 data/games/2_games/5.csv | sed -n '16p'` - Verify line 15 (row 16) still has values in gem_take2 columns (unchanged)

- `grep -h "take 3 tokens" data/games/2_games/*.csv | cut -d',' -f384,387-396 | awk -F',' '{if ($2 == "nan" && $3 == "nan" && $4 == "nan" && $5 == "nan" && $6 == "nan" && ($7 != "nan" || $8 != "nan" || $9 != "nan" || $10 != "nan" || $11 != "nan")) print "BUG FOUND: " $0}' | head -5` - Check for any remaining bugs in 2_games directory (should find 0 matches after fix)

- `grep -h "take 2 tokens" data/games/2_games/*.csv | cut -d',' -f384,387-396 | awk -F',' '{if ($7 == "nan" && $8 == "nan" && $9 == "nan" && $10 == "nan" && $11 == "nan") print "REGRESSION: " $0}' | head -5` - Check that "take 2 tokens" rows weren't incorrectly modified (should find 0 matches)

- `python scripts/fix_csv_gem_encoding.py --dry-run` - Run repair script in dry-run mode to preview changes without modifying files

- `python scripts/fix_csv_gem_encoding.py` - Execute the actual repair

## Notes
- The bug only affects the encoding of token-taking actions, not build or reserve actions
- The fix must preserve backward compatibility with the existing 382 input features + 20 output features format
- The repair script should create backups of files before modifying them (optional but recommended)
- After running the repair script, the `all_games.csv` file should also be regenerated or repaired
- This is a data quality issue that could affect machine learning model training if left unfixed
- The root cause is a logic inconsistency where action_type determination uses token pattern (has a 2?) but column selection uses token sum
