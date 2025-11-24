# Feature: Batched Preprocessing with Enhanced Memory Monitoring and Worker Feedback

## Feature Description
Implement batch-based file processing to prevent memory overflow during data preprocessing. The current implementation loads all 7,259 CSV files into memory simultaneously (causing 60+ GB RAM usage), leading to system slowdown and swap usage. This feature adds:

1. **Batched Processing**: Process files in configurable batches (default: 500 files), saving intermediate results to disk and clearing memory between batches
2. **Memory Monitoring**: Display RAM state before/after each batch and during critical operations
3. **Enhanced Worker Feedback**: Show which file each worker is processing without flooding the terminal
4. **Batch Merging**: Load and combine batch files one at a time to create final preprocessed arrays

This reduces peak memory usage from 60-90 GB to 15-25 GB while maintaining data quality.

## User Story
As a machine learning engineer preprocessing large game datasets
I want to process CSV files in batches with memory monitoring and worker visibility
So that I can complete preprocessing without running out of RAM and understand system resource usage during execution

## Problem Statement
The current parallel preprocessing pipeline (`parallel_processor.py`) has a critical memory issue:
- Accumulates ALL 547K rows in memory across 4 lists: `all_raw_rows`, `all_strategic_features`, `all_labels`, `all_masks`
- Strategic features alone consume 33 GB (547K dicts × 65 KB each)
- No intermediate persistence or memory clearing
- Peak memory reaches 60-90 GB, causing swap usage and system slowdown
- `batch_size: 500` is configured in `config_small.yaml` but completely unused
- No visibility into which worker is processing which file
- No feedback on memory state during processing

## Solution Statement
Refactor the preprocessing pipeline to implement true batched processing:

1. **Split files into batches** based on `config.preprocessing.batch_size`
2. **Process each batch completely**:
   - Accumulate results for batch only
   - Save batch to compressed NPZ file in `data/intermediate/`
   - Display memory state after batch save
   - Clear batch data and run garbage collection
3. **Enhanced worker feedback**:
   - Use multiprocessing.Manager to share current file being processed by each worker
   - Display worker status in a compact format (one line per worker, updated in place)
   - Maintain global progress bar with tqdm
4. **Batch merging phase**:
   - Load batch files one at a time
   - Combine into final arrays
   - Display memory state during merge
   - Delete intermediate files after successful merge
5. **Memory monitoring**:
   - Log memory at: pipeline start, after each batch save, before/after merge, pipeline end
   - Use existing `memory_monitor.py` infrastructure

## Relevant Files

### Core Implementation Files
- **`src/imitation_learning/parallel_processor.py`** (Lines 286-359)
  - Contains `process_files_parallel()` that needs batching logic
  - Currently accumulates all results in memory without clearing
  - Needs to split files into batches, save intermediate results, and return batch paths

- **`src/imitation_learning/data_preprocessing.py`** (Lines 1051-1272)
  - Contains `preprocess_with_parallel_processing()` main pipeline
  - Calls `process_files_parallel()` and expects combined results
  - Needs to handle batch file loading and merging

- **`src/imitation_learning/configs/config_small.yaml`** (Line 17)
  - Already has `batch_size: 500` configured
  - Also has `intermediate_dir: "data/intermediate"` configured (line 20)
  - These settings need to be actually used

### Supporting Files
- **`src/imitation_learning/memory_monitor.py`**
  - Already implements `log_memory_usage()` and `MemoryTracker` context manager
  - Will be used to monitor memory at batch boundaries

- **`src/imitation_learning/constants.py`**
  - Contains gem class constants used during encoding
  - Already imported by parallel_processor.py

### New Files
- **`data/intermediate/.gitkeep`**
  - Ensure intermediate directory exists in git
  - Batch NPZ files will be saved here temporarily

## Implementation Plan

### Phase 1: Foundation
Set up infrastructure for batched processing:
- Create intermediate directory structure
- Add helper functions for batch file I/O (save/load NPZ batches)
- Verify memory monitoring utilities work correctly

### Phase 2: Core Implementation
Implement batching logic in parallel processor:
- Split file list into batches based on config
- Process each batch and accumulate results in memory
- Save batch results to NPZ file with compression
- Clear memory and log usage after each batch
- Add enhanced worker feedback with shared state

### Phase 3: Integration
Update main preprocessing pipeline to handle batches:
- Modify pipeline to load batch files one at a time
- Combine batches into final arrays
- Clean up intermediate files after successful merge
- Add memory logging throughout the pipeline

## Step by Step Tasks

### Step 1: Create Intermediate Directory Structure
- Create `data/intermediate/` directory
- Add `.gitkeep` file to track in git
- Add `.gitignore` entry to ignore `*.npz` batch files

### Step 2: Add Batch I/O Helper Functions
- Add `save_batch_to_file()` function in `parallel_processor.py`
  - Takes batch data (df, features, labels, masks)
  - Saves to compressed NPZ file: `data/intermediate/batch_{batch_num:04d}.npz`
  - Returns file path
  - Logs memory before and after save
- Add `load_batch_from_file()` function in `parallel_processor.py`
  - Takes batch file path
  - Loads and returns all arrays/dataframes
  - Logs memory after load
- Add `delete_batch_files()` function to clean up intermediate files

### Step 3: Implement Batch Splitting in `process_files_parallel()`
- Extract `batch_size` from config: `config.get('preprocessing', {}).get('batch_size', 500)`
- Split `file_paths` into batches: `[file_paths[i:i+batch_size] for i in range(0, len(file_paths), batch_size)]`
- Create `data/intermediate/` directory if it doesn't exist
- Loop over batches instead of processing all files at once

### Step 4: Add Enhanced Worker Feedback System
- Create `WorkerStatus` class using `multiprocessing.Manager().dict()`
- Each worker updates its current file in shared dict
- Create `display_worker_status()` function:
  - Shows one compact line per worker: `Worker 0: game_5444.csv [████░░░] 67%`
  - Updates in place using carriage return `\r`
  - Called periodically during processing
- Integrate with existing tqdm progress bar

### Step 5: Process Each Batch with Memory Monitoring
- For each batch of files:
  - Log memory at batch start
  - Process files in batch (reuse existing parallel/sequential logic)
  - Accumulate results in temporary lists (batch only, not all files)
  - Log memory after batch processing
  - Save batch to NPZ file
  - Log memory after batch save
  - Clear batch data: `del all_raw_rows, all_strategic_features, all_labels, all_masks`
  - Run garbage collection: `gc.collect()`
  - Log memory after cleanup
- Update progress display: `Processing batch 3/15 (500 files)...`

### Step 6: Return Batch File Paths
- Change `process_files_parallel()` return signature:
  - OLD: `Tuple[pd.DataFrame, List[Dict], List[Dict], List[Dict]]`
  - NEW: `List[str]` (list of batch file paths)
- Return list of saved batch file paths instead of accumulated data

### Step 7: Implement Batch Merging in Main Pipeline
- Update `preprocess_with_parallel_processing()` to handle batch files:
  - Receive list of batch file paths from `process_files_parallel()`
  - Initialize empty lists for final data
  - Log memory before merge phase
  - Loop through batch files one at a time:
    - Load batch file
    - Log memory after load
    - Extend final lists with batch data
    - Delete batch data from memory
    - Run `gc.collect()`
    - Log memory after cleanup
  - Convert final lists to DataFrames/arrays
  - Delete batch files from disk

### Step 8: Add Memory Monitoring Throughout Pipeline
- Add `log_memory_usage()` calls at key points:
  - Pipeline start (already exists line 1075)
  - Start of each batch processing
  - After each batch save
  - After each batch cleanup
  - Before merge phase
  - After loading each batch file during merge
  - After final merge completion
  - Pipeline end
- Use `MemoryTracker` context manager for major operations

### Step 9: Update Configuration Validation
- Ensure `batch_size` is read from config
- Ensure `intermediate_dir` is read from config
- Add validation for these config values
- Print batch configuration at pipeline start

### Step 10: Add Cleanup Error Handling
- Wrap batch file deletion in try/except
- If batch merge fails, keep intermediate files for debugging
- Print warning if cleanup fails
- Add `--keep-batches` CLI flag for debugging

### Step 11: Test with Small Subset
- Test with `--max-games 100` (should create 1 batch)
- Verify batch file is created in `data/intermediate/`
- Verify batch file contains correct data
- Verify memory logging appears
- Verify worker status display works
- Verify batch file is deleted after merge

### Step 12: Test with Medium Subset
- Test with `--max-games 1000` (should create 2 batches)
- Verify memory usage stays below 25 GB
- Verify batch files are created and deleted
- Verify final output matches expected shape

### Step 13: Run Full Validation
- Execute all validation commands
- Verify no regressions in output quality
- Verify memory usage reduction
- Verify processing completes successfully

## Testing Strategy

### Unit Tests
- Test `save_batch_to_file()` and `load_batch_from_file()` round-trip
- Test batch splitting logic with various file counts
- Test batch merging produces identical results to non-batched
- Test cleanup handles missing files gracefully

### Integration Tests
- Test full pipeline with 100 files (1 batch)
- Test full pipeline with 1000 files (2 batches)
- Verify output arrays match expected shapes and dtypes
- Verify no data loss or corruption during batching

### Edge Cases
- Test with file count < batch_size (single batch)
- Test with file count = batch_size exactly
- Test with file count = batch_size + 1 (two batches, second has 1 file)
- Test with empty file list
- Test with batch_size = 1 (maximum batches)
- Test batch file I/O failure handling
- Test interrupted processing (Ctrl+C during batch)

## Acceptance Criteria
1. ✅ `batch_size` configuration parameter is actually used
2. ✅ Files are split into batches of configured size
3. ✅ Each batch is saved to NPZ file in `data/intermediate/`
4. ✅ Memory is logged before/after each batch operation
5. ✅ Peak memory usage stays below 25 GB for full dataset (down from 60-90 GB)
6. ✅ Worker status shows which file each worker is processing
7. ✅ Worker display doesn't flood terminal (compact, in-place updates)
8. ✅ Batch files are merged correctly to produce final arrays
9. ✅ Intermediate batch files are deleted after successful merge
10. ✅ Output arrays are identical to non-batched processing (same shapes, dtypes, values)
11. ✅ No swap usage occurs during preprocessing
12. ✅ All validation commands pass without errors

## Validation Commands
Execute every command to validate the feature works correctly with zero regressions.

```bash
# Test 1: Verify configuration is loaded correctly
uv run python -c "
import yaml
config = yaml.safe_load(open('src/imitation_learning/configs/config_small.yaml'))
batch_size = config['preprocessing']['batch_size']
intermediate_dir = config['preprocessing']['intermediate_dir']
print(f'✓ batch_size: {batch_size}')
print(f'✓ intermediate_dir: {intermediate_dir}')
assert batch_size == 500, 'batch_size should be 500'
assert intermediate_dir == 'data/intermediate', 'intermediate_dir should be data/intermediate'
"

# Test 2: Small subset test (100 files, 1 batch)
uv run python -m src.imitation_learning.data_preprocessing \
  --config src/imitation_learning/configs/config_small.yaml \
  --max-games 100 \
  --parallel true

# Test 3: Verify output shapes and dtypes
uv run python -c "
import numpy as np
X_train = np.load('data/processed/X_train.npy')
X_val = np.load('data/processed/X_val.npy')
X_test = np.load('data/processed/X_test.npy')
print(f'✓ X_train shape: {X_train.shape}, dtype: {X_train.dtype}')
print(f'✓ X_val shape: {X_val.shape}, dtype: {X_val.dtype}')
print(f'✓ X_test shape: {X_test.shape}, dtype: {X_test.dtype}')
assert X_train.dtype == np.float64 or X_train.dtype == np.float32, 'Should be float32 or float64'
assert len(X_train.shape) == 2, 'Should be 2D array'
print('✓ All shapes and dtypes valid')
"

# Test 4: Verify no NaN values in output
uv run python -c "
import numpy as np
X_train = np.load('data/processed/X_train.npy')
X_val = np.load('data/processed/X_val.npy')
X_test = np.load('data/processed/X_test.npy')
assert not np.isnan(X_train).any(), 'X_train contains NaN'
assert not np.isnan(X_val).any(), 'X_val contains NaN'
assert not np.isnan(X_test).any(), 'X_test contains NaN'
print('✓ No NaN values in outputs')
"

# Test 5: Verify batch files are cleaned up
uv run python -c "
import os
batch_files = [f for f in os.listdir('data/intermediate') if f.endswith('.npz')]
if batch_files:
    print(f'⚠ Warning: {len(batch_files)} batch files remain: {batch_files[:5]}')
else:
    print('✓ All batch files cleaned up')
"

# Test 6: Medium subset test (1000 files, 2 batches)
uv run python -m src.imitation_learning.data_preprocessing \
  --config src/imitation_learning/configs/config_small.yaml \
  --max-games 1000 \
  --parallel true

# Test 7: Monitor memory during full test
# Run this in a separate terminal: watch -n 2 'free -h'
# Then run full preprocessing
uv run python -m src.imitation_learning.data_preprocessing \
  --config src/imitation_learning/configs/config_small.yaml \
  --parallel true

# Test 8: Verify final outputs
uv run python -c "
import numpy as np
import json

# Load arrays
X_train = np.load('data/processed/X_train.npy')
X_val = np.load('data/processed/X_val.npy')
X_test = np.load('data/processed/X_test.npy')

# Load labels
labels_train = np.load('data/processed/labels_train.npz')
labels_val = np.load('data/processed/labels_val.npz')
labels_test = np.load('data/processed/labels_test.npz')

# Load masks
masks_train = np.load('data/processed/masks_train.npz')
masks_val = np.load('data/processed/masks_val.npz')
masks_test = np.load('data/processed/masks_test.npz')

# Load stats
with open('data/processed/preprocessing_stats.json') as f:
    stats = json.load(f)

print('✓ Features:')
print(f'  Train: {X_train.shape}')
print(f'  Val: {X_val.shape}')
print(f'  Test: {X_test.shape}')
print(f'  Total samples: {stats[\"total_samples\"]}')
print(f'  Input dim: {stats[\"input_dim\"]}')

print('\\n✓ Labels:')
for key in labels_train.files:
    print(f'  {key}: train={labels_train[key].shape}, val={labels_val[key].shape}, test={labels_test[key].shape}')

print('\\n✓ Masks:')
for key in masks_train.files:
    print(f'  {key}: train={masks_train[key].shape}, val={masks_val[key].shape}, test={masks_test[key].shape}')

print('\\n✓ All preprocessing outputs valid')
"

# Test 9: Verify no swap usage occurred (run after preprocessing)
free -h | grep Swap

# Test 10: Check preprocessing stats
cat data/processed/preprocessing_stats.json | python -m json.tool
```

## Notes

### Memory Savings Expected
- **Before**: 60-90 GB peak memory (all files in memory)
- **After**: 15-25 GB peak memory (one batch at a time)
- **Reduction**: 60-75% memory savings

### Batch File Format
Each batch NPZ file contains:
- `df_compacted`: DataFrame with compacted rows (pickle)
- `strategic_features`: List of feature dicts
- `labels`: List of label dicts
- `masks`: List of mask dicts
- `metadata`: Dict with batch info (batch_num, num_samples, file_paths)

### Worker Feedback Display
Example compact display (updates in place):
```
Processing batch 3/15 (500 files)...
Worker 0: data/games/3_games/5444.csv
Worker 1: data/games/3_games/5445.csv
Worker 2: data/games/2_games/1234.csv
Worker 3: data/games/4_games/8901.csv
Worker 4: data/games/3_games/5446.csv
Worker 5: data/games/2_games/1235.csv
[████████████████████░░░░░░░░░] 315/500 files (63%)
```

### Future Enhancements
- Add progress persistence: save batch file list to JSON, allowing resume after interruption
- Add parallel batch merging: load multiple batches in parallel if memory allows
- Add batch verification: compute checksums to detect corruption
- Add incremental processing: process new files without reprocessing existing batches

### Psutil Dependency
The `memory_monitor.py` module requires `psutil` for memory tracking. If not installed:
```bash
uv add psutil
```

This is already documented in the spec at line 410: "uv add psutil"
