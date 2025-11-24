# Bug: Parallel Processing Not Actually Running in Parallel

## Bug Description
The `process_files_parallel` function in `src/imitation_learning/parallel_processor.py` claims to process files in parallel using multiprocessing, but it actually processes files sequentially in a for loop. The function accepts a `num_workers` parameter (defaulting to `cpu_count() - 2`) but never uses it. Instead, it has a comment on line 305-306 stating "For now, process sequentially to avoid multiprocessing complexity".

**Symptoms:**
- Processing runs much slower than expected (no parallelization speedup)
- Only one CPU core is heavily utilized during processing
- The progress bar shows files being processed one at a time

**Expected behavior:**
- Files should be processed in parallel across multiple CPU cores
- Near-linear speedup with number of workers (N cores → ~N× faster)
- Multiple CPU cores should show high utilization

**Actual behavior:**
- Files are processed sequentially in a single for loop
- Only one CPU core is utilized
- No performance improvement from multiprocessing

## Problem Statement
The `process_files_parallel` function needs to be modified to actually use multiprocessing.Pool to process files in parallel, rather than processing them sequentially in a for loop.

## Solution Statement
Implement true parallel processing using `multiprocessing.Pool` with the specified number of workers. Use `pool.imap_unordered()` or `pool.map()` to distribute file processing across worker processes, allowing multiple files to be processed simultaneously.

## Steps to Reproduce
1. Run the preprocessing pipeline with `num_workers=6` (or any value)
2. Monitor CPU usage during processing
3. Observe that only one core is heavily utilized
4. Notice that files are processed sequentially (no speedup from multiple workers)

## Root Cause Analysis
The function was initially implemented with sequential processing as a placeholder, with the intent to add parallelization later. The comment on lines 305-306 explicitly states this:

```python
# For now, process sequentially to avoid multiprocessing complexity
# We can add true parallelization later if needed
```

However, the sequential implementation was never replaced with actual parallel processing. The `num_workers` parameter is calculated but never used in the actual processing logic (lines 300-331).

## Relevant Files
Use these files to fix the bug:

### `src/imitation_learning/parallel_processor.py` (lines 285-331)
- Contains the `process_files_parallel` function that needs to be fixed
- Currently processes files sequentially despite accepting `num_workers` parameter
- Has helper function `process_single_file` that is already designed to work as a worker function

### `src/imitation_learning/data_preprocessing.py` (lines 1082-1093)
- Calls `process_files_parallel` with `num_workers` from config
- No changes needed - this file correctly passes the parameter

## Step by Step Tasks

### 1. Implement actual parallel processing in `process_files_parallel`
- Replace the sequential for loop (lines 312-318) with `multiprocessing.Pool`
- Use `pool.imap_unordered()` or `pool.map()` to distribute work across workers
- Ensure proper handling of the `tqdm` progress bar with parallel processing
- Use a wrapper function or `functools.partial` to pass the `config` parameter to workers

### 2. Handle edge cases
- Ensure `num_workers=1` still works (falls back to sequential processing)
- Handle empty file list gracefully
- Ensure proper pool cleanup (use context manager or try/finally)

### 3. Test the parallel processing implementation
- Create a test script that processes a small subset of files
- Verify that multiple CPU cores are utilized during processing
- Confirm that results are identical to sequential processing
- Measure speedup with different `num_workers` values

### 4. Run validation commands
- Execute all validation commands to ensure no regressions
- Verify output correctness and performance improvement

## Validation Commands
Execute every command to validate the bug is fixed with zero regressions.

```bash
# Test 1: Run a small preprocessing test to verify parallel processing works
timeout 180 uv run python -c "
import sys
sys.path.insert(0, 'src')
from imitation_learning.parallel_processor import discover_csv_files, process_files_parallel

# Create minimal config
config = {
    'data': {'data_root': 'data/games'},
    'preprocessing': {'num_workers': 4}
}

# Discover and process a small subset
csv_files = discover_csv_files(config['data']['data_root'], max_games=5)
print(f'Processing {len(csv_files)} files with 4 workers...')

df, strategic_features, labels, masks = process_files_parallel(
    csv_files, config, num_workers=4
)

print(f'Successfully processed {len(df)} samples')
print('Parallel processing test PASSED')
"

# Test 2: Verify single worker mode still works
timeout 60 uv run python -c "
import sys
sys.path.insert(0, 'src')
from imitation_learning.parallel_processor import discover_csv_files, process_files_parallel

config = {'data': {'data_root': 'data/games'}}
csv_files = discover_csv_files(config['data']['data_root'], max_games=2)

df, strategic_features, labels, masks = process_files_parallel(
    csv_files, config, num_workers=1
)

print(f'Single worker mode processed {len(df)} samples')
print('Single worker test PASSED')
"

# Test 3: Check that multiprocessing module is properly imported and used
uv run python -c "
import sys
sys.path.insert(0, 'src')
from imitation_learning import parallel_processor
import inspect

source = inspect.getsource(parallel_processor.process_files_parallel)
assert 'Pool' in source or 'pool' in source, 'Pool not found in function'
print('Code inspection PASSED: Pool usage detected')
"
```

## Notes
- The `process_single_file` function is already designed to work as a worker function - it takes a file path and config as parameters and returns results
- Use `functools.partial` to bind the `config` parameter when mapping the function across files
- Consider using `pool.imap_unordered()` instead of `pool.map()` for better memory efficiency and ability to show progress with tqdm
- For tqdm progress tracking with parallel processing, use: `tqdm(pool.imap_unordered(...), total=len(file_paths))`
- The multiprocessing module is already imported at the top of the file (line 22: `import multiprocessing as mp`)
- No new dependencies are required - multiprocessing is part of Python's standard library
