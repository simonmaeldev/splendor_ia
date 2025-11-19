# Chore: Clean CSV NaN Values

## Chore Description
Create a script that will identify and process CSV files in the `data/*_games/` directories that were created after "Nov 18 22:45" and replace all occurrences of the string 'nan' with empty strings. The script should provide a summary report showing the number of files affected.

The issue is that files created after Nov 18 22:45 contain the string "nan" in many columns (particularly in player reserved card fields and player3 fields for 2-player games), while older files correctly use empty strings for these values. This cleanup ensures data consistency across all CSV files.

Based on initial investigation, approximately 740 files need to be processed.

## Relevant Files
Use these files to resolve the chore:

- **`data/games/2_games/*.csv`** - CSV files from 2-player games, some created after Nov 18 22:45 containing 'nan' values
- **`data/games/3_games/*.csv`** - CSV files from 3-player games, some created after Nov 18 22:45 containing 'nan' values
- **`data/games/4_games/*.csv`** - CSV files from 4-player games, some created after Nov 18 22:45 containing 'nan' values
- **`scripts/fix_csv_gem_encoding.py`** - Existing similar CSV fix script that can serve as a reference for the structure

### New Files

- **`scripts/clean_csv_nan_values.py`** - New Python script to clean 'nan' values from CSV files created after Nov 18 22:45

## Step by Step Tasks
IMPORTANT: Execute every step in order, top to bottom.

### Step 1: Examine existing fix script for reference
- Read the existing `scripts/fix_csv_gem_encoding.py` to understand the pattern for CSV processing scripts in this project
- Identify best practices for file discovery, processing, and reporting

### Step 2: Create the cleanup script
- Create `scripts/clean_csv_nan_values.py` with the following functionality:
  - Use `find` command or Python's `os.path.getmtime()` to identify files in `data/*_games/` directories created after "Nov 18 22:45" (timestamp: Nov 18 2024 22:45:00)
  - For each identified file:
    - Read the CSV content
    - Replace all occurrences of the string 'nan' with empty strings (handle both standalone 'nan' values and 'nan' within CSV fields)
    - Write the cleaned content back to the file
  - Track statistics: number of files processed, number of files with changes, total 'nan' replacements
- Include proper error handling for file operations
- Add a dry-run mode option to preview changes without modifying files
- Make the script executable with `chmod +x`

### Step 3: Test the script in dry-run mode
- Run the script in dry-run mode to verify it correctly identifies the ~740 files created after Nov 18 22:45
- Examine sample output to ensure the replacement logic is correct
- Verify one or two specific files to ensure 'nan' values would be properly replaced

### Step 4: Execute the cleanup
- Run the script to perform the actual cleanup of all affected files
- Capture and display the final report showing:
  - Total number of files scanned
  - Number of files modified
  - Total number of 'nan' replacements made
  - Any errors encountered

### Step 5: Validate the cleanup
- Sample a few files that were modified to verify 'nan' strings are removed
- Ensure CSV structure remains intact
- Verify that files created before Nov 18 22:45 were not modified

## Validation Commands
Execute every command to validate the chore is complete with zero regressions.

- `python scripts/clean_csv_nan_values.py --dry-run` - Run in dry-run mode to verify file identification
- `python scripts/clean_csv_nan_values.py` - Execute the actual cleanup
- `find /home/apprentyr/projects/splendor_ia/data -type f -name "*.csv" -newermt "Nov 18 22:45" | wc -l` - Verify we're processing the correct number of files (~740)
- `grep -l "nan" /home/apprentyr/projects/splendor_ia/data/games/2_games/7075.csv` - Verify sample file no longer contains 'nan' after cleanup
- `head -5 /home/apprentyr/projects/splendor_ia/data/games/2_games/7075.csv` - Verify CSV structure is intact after cleanup

## Notes
- The script should handle both Python (`.py`) and shell script (`.sh`) implementations - Python is recommended for better CSV handling
- Be careful to replace 'nan' as a complete value, not as part of other words (though in CSV context this is unlikely)
- The timestamp "Nov 18 22:45" should be interpreted as Nov 18, 2024 22:45:00 in the local timezone
- Consider creating backups before modification, though the files appear to be generated data that can be regenerated if needed
- The script will be useful for the user to understand the scope of the issue through the summary report
