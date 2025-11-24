# Bug: Intermediate Files Not Persisted and Merge Logic Needs Separation

## Bug Description
During preprocessing with batching and memory monitoring, all batches are processed successfully, but the process quits unexpectedly during the merge phase. The intermediate batch files (*.npz) in `data/intermediate/` disappear, making it impossible to retry the merge without reprocessing all files. Additionally, the merging logic is embedded in the main preprocessing pipeline, preventing users from modifying or re-running only the merge step.

**Symptoms:**
- Batches process successfully (all files converted to batch_*.npz files)
- Process crashes/quits during merge phase (lines 1099-1158 in data_preprocessing.py)
- Intermediate files are not visible in `data/intermediate/` after the crash
- Must reprocess all files to retry, wasting significant computation time

**Expected behavior:**
- Intermediate batch files should persist in `data/intermediate/` even after merge completion
- Merging should be a separate, independently runnable script
- Users should be able to modify and re-run merge logic without reprocessing files

## Problem Statement
The current implementation has two critical issues:

1. **Premature deletion of intermediate files:** The `delete_batch_files()` function is called immediately after successful merge (line 1152), and if the merge fails partway through, files may already be deleted. This makes debugging and recovery impossible.

2. **Tight coupling of concerns:** The merge logic (lines 1099-1158) is embedded in `preprocess_with_parallel_processing()`, mixing file processing with data combination. This violates separation of concerns and prevents:
   - Re-running merge with different strategies
   - Debugging merge issues without full reprocessing
   - Testing merge logic independently
   - Iterating on merge implementation

## Solution Statement
Refactor the preprocessing pipeline to:

1. **Keep intermediate files by default:** Remove automatic deletion of batch files after merge. Add an optional `--cleanup` flag to delete them only when explicitly requested.

2. **Extract merge to separate script:** Create `src/imitation_learning/merge_batches.py` that:
   - Discovers batch files in `data/intermediate/`
   - Loads them one at a time
   - Combines into final arrays
   - Performs splitting, normalization, validation, and saving
   - Can be run independently without reprocessing

3. **Modify main pipeline:** Update `data_preprocessing.py` to:
   - Process files and save batches (stopping after batching)
   - Optionally call merge script if `--skip-merge` is not specified
   - Report where batch files are saved for manual merging

## Steps to Reproduce
1. Run preprocessing with batching:
   ```bash
   uv run python -m src.imitation_learning.data_preprocessing \
     --config src/imitation_learning/configs/config_small.yaml \
     --parallel true
   ```
2. Observe batches being processed and saved to `data/intermediate/`
3. Process enters merge phase (loading and combining batches)
4. Process crashes or quits unexpectedly (e.g., OOM, keyboard interrupt, bug)
5. Check `data/intermediate/` directory - no batch files remain
6. Must restart entire preprocessing from scratch

## Root Cause Analysis

**Root Cause 1: Premature Cleanup**
- Lines 1149-1155 in `data_preprocessing.py` call `delete_batch_files()` immediately after successful merge
- If merge fails partway (OOM, crash, exception), files may already be partially deleted
- No way to resume or debug without full reprocessing

**Root Cause 2: Monolithic Design**
- Merge logic is embedded in the main `preprocess_with_parallel_processing()` function
- No separation between:
  - File processing (can take hours)
  - Data merging (quick but error-prone)
  - Normalization and saving
- Cannot modify merge strategy without modifying main pipeline
- Cannot test merge independently

**Root Cause 3: No User Control**
- Users have no option to keep intermediate files for inspection/debugging
- No `--keep-batches` flag or configuration option
- Cleanup is automatic and non-optional

## Relevant Files

### Files to Modify

- **`src/imitation_learning/parallel_processor.py`** (Lines 366-388)
  - Contains `delete_batch_files()` function that removes intermediate files
  - Need to make cleanup optional, not automatic

- **`src/imitation_learning/data_preprocessing.py`** (Lines 1099-1158)
  - Contains merge logic embedded in main pipeline
  - Need to extract this into separate function/script
  - Need to add `--skip-merge` and `--cleanup` flags

- **`.gitignore`** (Lines with `data/intermediate/*.npz`)
  - Already configured to ignore batch files
  - No changes needed, but worth noting

### New Files

- **`src/imitation_learning/merge_batches.py`**
  - New standalone script for merging batch files
  - Discovers batch files in intermediate directory
  - Loads batches one at a time with memory monitoring
  - Combines into final arrays
  - Performs splitting, normalization, validation
  - Saves to `data/processed/`
  - Can be run independently: `python -m src.imitation_learning.merge_batches --config config.yaml`

## Step by Step Tasks

### Step 1: Create Standalone Merge Script
- Create `src/imitation_learning/merge_batches.py` with:
  - `discover_batch_files(intermediate_dir)` - finds all batch_*.npz files
  - `merge_batches(batch_files, config, monitor_memory)` - main merge logic
  - `main()` - CLI entry point with argparse
- Accept arguments:
  - `--config` - path to config file
  - `--intermediate-dir` - override intermediate directory (default: from config)
  - `--cleanup` - delete batch files after successful merge (default: False)
  - `--skip-validation` - skip mask validation (for faster iteration)
- Reuse existing functions from `data_preprocessing.py`:
  - `split_by_game_id()`
  - `create_normalization_mask()`
  - `normalize_features()`
  - `validate_masks()`
  - `save_preprocessed_data()`

### Step 2: Extract Merge Logic from Main Pipeline
- Move merge logic (lines 1099-1158) from `preprocess_with_parallel_processing()` to new `merge_batches.py`
- Keep the logic exactly the same initially (no behavioral changes)
- Ensure memory monitoring is preserved
- Ensure garbage collection is preserved

### Step 3: Update Main Preprocessing Pipeline
- Modify `preprocess_with_parallel_processing()` to:
  - Process files and save batches (lines 1067-1094)
  - Print summary: "Batches saved to {intermediate_dir}, {num_batches} files"
  - Add `--skip-merge` flag to stop after batching
  - If `--skip-merge` is False (default), call merge script
  - Pass through `--cleanup` flag to merge script
- Add argparse arguments to `data_preprocessing.py`:
  - `--skip-merge` (default: False) - stop after creating batches
  - `--cleanup` (default: False) - delete batch files after merge

### Step 4: Make Cleanup Optional in delete_batch_files()
- Modify `delete_batch_files()` in `parallel_processor.py`:
  - Keep function signature but document it's now only called when explicitly requested
  - No functional changes needed, just usage changes

### Step 5: Update Config to Support Cleanup Option
- Add to `config_small.yaml` under `preprocessing`:
  ```yaml
  preprocessing:
    batch_size: 500
    intermediate_dir: "data/intermediate"
    cleanup_batches: false  # New: keep batches by default
    monitor_memory: true
  ```
- Make merge script respect this config option

### Step 6: Add Batch Discovery Function
- In `merge_batches.py`, create `discover_batch_files()`:
  - Scans intermediate directory for `batch_*.npz` files
  - Sorts by batch number (batch_0000.npz, batch_0001.npz, ...)
  - Returns sorted list of paths
  - Validates batch files exist and are readable

### Step 7: Implement Main Merge Function
- In `merge_batches.py`, create `merge_batches()`:
  - Takes list of batch file paths and config
  - Implements exact logic from lines 1099-1158 of data_preprocessing.py
  - Loads batches one at a time with memory monitoring
  - Combines DataFrames and extends lists
  - Clears memory between batches
  - Returns combined DataFrame and lists

### Step 8: Implement Post-Merge Processing
- In `merge_batches.py`, after merge:
  - Convert strategic features list to DataFrame
  - Engineer one-hot features
  - Identify column groups
  - Convert labels and masks to arrays
  - Split by game_id
  - Create normalization mask
  - Normalize features
  - Validate masks (unless `--skip-validation`)
  - Save preprocessed data

### Step 9: Handle Cleanup Logic
- In `merge_batches.py`, after successful save:
  - Check if `--cleanup` flag is set
  - If True, call `delete_batch_files(batch_file_paths)`
  - If False, print message: "Batch files kept in {intermediate_dir}"
  - Log cleanup decision

### Step 10: Update Documentation
- Update `specs/implement-batched-preprocessing-with-monitoring.md`:
  - Document new `--skip-merge` and `--cleanup` flags
  - Add section on running merge independently
  - Add example workflow: process → inspect batches → merge
- Add docstrings to new `merge_batches.py` module

### Step 11: Test Batching Without Merge
- Run preprocessing with `--skip-merge`:
  ```bash
  uv run python -m src.imitation_learning.data_preprocessing \
    --config src/imitation_learning/configs/config_small.yaml \
    --max-games 100 \
    --parallel true \
    --skip-merge
  ```
- Verify batches are created in `data/intermediate/`
- Verify process stops after batching (no merge)

### Step 12: Test Standalone Merge Script
- Run merge script on existing batches:
  ```bash
  uv run python -m src.imitation_learning.merge_batches \
    --config src/imitation_learning/configs/config_small.yaml
  ```
- Verify batches are loaded correctly
- Verify final arrays are saved to `data/processed/`
- Verify batch files remain in `data/intermediate/` (no cleanup)

### Step 13: Test Full Pipeline with Cleanup
- Run full pipeline with cleanup:
  ```bash
  uv run python -m src.imitation_learning.data_preprocessing \
    --config src/imitation_learning/configs/config_small.yaml \
    --max-games 100 \
    --parallel true \
    --cleanup
  ```
- Verify batches are created, merged, and then deleted
- Verify final arrays are correct

### Step 14: Test Error Recovery
- Create batches with `--skip-merge`
- Manually corrupt one batch file (to simulate crash)
- Run merge script and verify it handles error gracefully
- Verify uncorrupted batches are still available for retry

### Step 15: Run Validation Commands
- Execute all validation commands from below
- Verify no regressions in output quality
- Verify batch files can be inspected and reused

## Validation Commands
Execute every command to validate the bug is fixed with zero regressions.

```bash
# Test 1: Process batches without merging
uv run python -m src.imitation_learning.data_preprocessing \
  --config src/imitation_learning/configs/config_small.yaml \
  --max-games 100 \
  --parallel true \
  --skip-merge

# Test 2: Verify batch files exist
uv run python -c "
import os
batch_files = sorted([f for f in os.listdir('data/intermediate') if f.endswith('.npz')])
print(f'✓ Found {len(batch_files)} batch files:')
for f in batch_files[:5]:
    print(f'  {f}')
assert len(batch_files) > 0, 'No batch files found!'
"

# Test 3: Run standalone merge script
uv run python -m src.imitation_learning.merge_batches \
  --config src/imitation_learning/configs/config_small.yaml

# Test 4: Verify batch files still exist (no cleanup by default)
uv run python -c "
import os
batch_files = [f for f in os.listdir('data/intermediate') if f.endswith('.npz')]
print(f'✓ Batch files still present: {len(batch_files)} files')
assert len(batch_files) > 0, 'Batch files were deleted!'
"

# Test 5: Verify output files are correct
uv run python -c "
import numpy as np
X_train = np.load('data/processed/X_train.npy')
X_val = np.load('data/processed/X_val.npy')
X_test = np.load('data/processed/X_test.npy')
print(f'✓ X_train shape: {X_train.shape}')
print(f'✓ X_val shape: {X_val.shape}')
print(f'✓ X_test shape: {X_test.shape}')
assert X_train.shape[0] > 0, 'X_train is empty'
"

# Test 6: Test merge with cleanup flag
uv run python -m src.imitation_learning.merge_batches \
  --config src/imitation_learning/configs/config_small.yaml \
  --cleanup

# Test 7: Verify batch files are deleted after cleanup
uv run python -c "
import os
batch_files = [f for f in os.listdir('data/intermediate') if f.endswith('.npz')]
print(f'✓ Batch files after cleanup: {len(batch_files)} files')
assert len(batch_files) == 0, 'Batch files were not cleaned up!'
"

# Test 8: Test full pipeline end-to-end (with cleanup)
uv run python -m src.imitation_learning.data_preprocessing \
  --config src/imitation_learning/configs/config_small.yaml \
  --max-games 200 \
  --parallel true \
  --cleanup

# Test 9: Verify no batch files remain after full pipeline with cleanup
uv run python -c "
import os
batch_files = [f for f in os.listdir('data/intermediate') if f.endswith('.npz')]
print(f'✓ Batch files after full pipeline: {len(batch_files)} files')
assert len(batch_files) == 0, 'Cleanup flag did not work in full pipeline!'
"

# Test 10: Test full pipeline without cleanup (default behavior)
uv run python -m src.imitation_learning.data_preprocessing \
  --config src/imitation_learning/configs/config_small.yaml \
  --max-games 100 \
  --parallel true

# Test 11: Verify batch files remain after full pipeline (default)
uv run python -c "
import os
batch_files = [f for f in os.listdir('data/intermediate') if f.endswith('.npz')]
print(f'✓ Batch files kept by default: {len(batch_files)} files')
assert len(batch_files) > 0, 'Default behavior should keep batch files!'
"

# Test 12: Verify final outputs are correct
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

print('✓ All outputs present and valid')
print(f'  Total samples: {stats[\"total_samples\"]}')
print(f'  Input dim: {stats[\"input_dim\"]}')
print(f'  Train/Val/Test: {len(X_train)}/{len(X_val)}/{len(X_test)}')

assert not np.isnan(X_train).any(), 'X_train contains NaN'
assert not np.isnan(X_val).any(), 'X_val contains NaN'
assert not np.isnan(X_test).any(), 'X_test contains NaN'
print('✓ No NaN values in outputs')
"

# Test 13: Verify merge script works with different intermediate directories
mkdir -p data/test_intermediate
uv run python -m src.imitation_learning.data_preprocessing \
  --config src/imitation_learning/configs/config_small.yaml \
  --max-games 50 \
  --parallel true \
  --skip-merge

# Move batches to test directory
mv data/intermediate/batch_*.npz data/test_intermediate/ 2>/dev/null || true

# Run merge on custom directory
uv run python -m src.imitation_learning.merge_batches \
  --config src/imitation_learning/configs/config_small.yaml \
  --intermediate-dir data/test_intermediate

# Cleanup test directory
rm -rf data/test_intermediate

# Test 14: Verify help messages work
uv run python -m src.imitation_learning.data_preprocessing --help | grep -q "skip-merge"
uv run python -m src.imitation_learning.merge_batches --help | grep -q "cleanup"

echo "✓ All validation tests passed!"
```

## Notes

### Design Decisions

1. **Keep batches by default:** Users can inspect intermediate files for debugging, verify batch integrity, and retry merge without reprocessing. This is safer and more transparent.

2. **Separate merge script:** Follows Unix philosophy (do one thing well). Users can:
   - Process files overnight → inspect batches → merge next day
   - Modify merge logic without touching file processing
   - Test different merge strategies on same batches
   - Debug merge issues without expensive reprocessing

3. **Optional cleanup:** Power users can delete batches with `--cleanup`, but novice users get safer default behavior (keep files).

### Expected Benefits

1. **Robustness:** Intermediate files survive crashes, OOM, interruptions
2. **Debuggability:** Can inspect batch files, verify data quality at intermediate stage
3. **Flexibility:** Can modify merge logic independently
4. **Separation of concerns:** File processing ≠ data merging ≠ normalization
5. **Developer experience:** Faster iteration on merge logic

### Disk Space Considerations

- Batch files are compressed NPZ (numpy's compressed format)
- For 7,259 CSV files with 547K rows:
  - Expected batch file size: ~5-10 GB total (compressed)
  - Final output size: ~2-3 GB
  - Users with disk space constraints can use `--cleanup`

### Migration Path

- Old behavior preserved with `--cleanup` flag
- New default is safer (keep batches)
- No breaking changes to existing scripts
- Backwards compatible with existing configs

### Future Enhancements

- Add `--resume` flag to merge script (skip already-merged batches)
- Add batch integrity checking (checksums, sample counts)
- Add parallel batch loading if memory allows
- Add incremental processing (merge new batches without reprocessing old ones)
