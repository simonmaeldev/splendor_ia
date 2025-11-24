# Feature: Memory-Efficient Batch Processing and Merging

## Feature Description
Redesign the parallel preprocessing pipeline's batch storage and merge operations to eliminate out-of-memory (OOM) errors when processing the 547K sample dataset. The current implementation stores strategic features as inefficient list-of-dicts structures and accumulates all batches in memory during merging, causing memory usage to exceed 64GB RAM + 16GB swap. This feature will:

1. **Strategy A**: Convert list-of-dicts to NumPy arrays before saving to batch files (82% memory reduction)
2. **Strategy B**: Implement sequential two-pass merging that preallocates arrays and writes directly without accumulation (eliminates memory growth)

The solution maintains backward compatibility with the intermediate `.npz` file format (essential for debugging and experimentation) while drastically reducing peak memory usage from ~70GB+ to ~3-4GB.

## User Story
As a machine learning researcher
I want to preprocess my 547K sample Splendor game dataset without running out of memory
So that I can train imitation learning models on my 64GB RAM machine without OOM crashes

## Problem Statement
The current preprocessing pipeline has severe memory inefficiencies:

1. **List-of-Dicts Storage (Strategy A Problem)**:
   - Strategic features: 547K samples × 893 features stored as list of 547K Python dicts
   - Each dict has massive overhead: ~240 bytes base + per-key overhead
   - Memory waste: ~50GB for strategic features alone vs ~2GB as NumPy arrays
   - Also affects labels (7 heads) and masks (7 heads) stored as list-of-dicts

2. **Accumulative Merging (Strategy B Problem)**:
   - `merge_batches()` loads 15 batches sequentially but extends Python lists
   - Lists grow from 0 → 547K elements, causing reallocation overhead
   - No memory is released until final conversion to arrays
   - Peak memory: all 15 batches worth of list-of-dicts (~50GB+) + DataFrame concat copies (~2GB) + temporary arrays during conversion (~2GB) = **70GB+ peak**

3. **OOM Failures**:
   - System has 64GB RAM + 16GB swap = 80GB total
   - Pipeline peaks at 70GB+ during merge operations
   - Unpredictable OOM due to fragmentation and temporary allocations
   - Cannot complete preprocessing on otherwise capable hardware

## Solution Statement
Implement a two-part optimization that maintains the `.npz` intermediate file format while slashing memory usage:

**Strategy A - Array-Based Storage**: Modify `parallel_processor.py` to convert list-of-dicts to structured NumPy arrays before saving batches. This changes storage from Python objects (massive overhead) to contiguous array blocks (minimal overhead):
- Strategic features: list of 29K dicts → (29K, 893) float32 array (720 MB → 104 MB per batch)
- Labels: list of 29K dicts → 7 separate int16 arrays (one per head)
- Masks: list of 29K dicts → 7 separate int8 2D arrays (one per head)

**Strategy B - Sequential Two-Pass Merge**: Rewrite `merge_batches.py` to eliminate accumulation:
- **Pass 1**: Scan all batch files to calculate total samples and dimensions
- Preallocate final arrays once: `X = np.zeros((547K, 893), dtype=float32)`
- **Pass 2**: Load each batch, write directly to preallocated arrays at correct offset, immediately release batch memory
- Result: Peak memory = size of final arrays (~3GB) vs current 70GB+

Both strategies work together synergistically:
- Strategy A enables Strategy B (can't efficiently concatenate list-of-dicts)
- Strategy A alone: 70GB → 12GB (83% reduction)
- Strategy A + B: 70GB → 3GB (96% reduction) 🎯

## Relevant Files
Use these files to implement the feature:

- **`src/imitation_learning/parallel_processor.py`** (Lines 285-336, 338-361)
  - `save_batch_to_file()`: Convert list-of-dicts to arrays before saving
  - `load_batch_from_file()`: Load arrays and reconstruct expected format for compatibility
  - Core of Strategy A implementation

- **`src/imitation_learning/merge_batches.py`** (Lines 90-159, 162-394)
  - `merge_batches()`: Rewrite to use two-pass sequential merge
  - `process_merged_data()`: Update to work with arrays from new merge
  - Core of Strategy B implementation

- **`src/imitation_learning/data_preprocessing.py`** (Lines 1125-1221)
  - `preprocess_with_parallel_processing()`: Entry point that calls merge functions
  - May need minor updates to handle new array formats

- **`src/imitation_learning/constants.py`** (Lines 14-19)
  - Provides constant mappings used for label encoding
  - No changes needed, but referenced for label array creation

- **`src/imitation_learning/utils.py`**
  - Contains encoding functions: `encode_gem_take3`, `encode_gem_take2`, `encode_gems_removed`
  - No changes needed, but needed to understand label structure

- **`src/imitation_learning/feature_engineering.py`**
  - `get_all_feature_names()`: Returns ordered list of 893 strategic feature names
  - Critical for ensuring column order consistency in arrays

- **`src/imitation_learning/configs/config_small.yaml`** (Lines 13-20)
  - Configuration for batch processing parameters
  - May add new config options for validation

### New Files
- **`tests/test_batch_storage.py`**
  - Unit tests for array conversion and batch save/load round-trip
  - Tests for memory efficiency and data integrity

- **`tests/test_sequential_merge.py`**
  - Unit tests for two-pass merge algorithm
  - Tests for correctness of preallocated array writes

- **`scripts/validate_batch_migration.py`**
  - Utility script to validate old vs new batch format produce identical results
  - Loads old batch, converts, saves, loads new batch, compares outputs

## Implementation Plan

### Phase 1: Foundation
Create array-based storage infrastructure and comprehensive test suite before modifying production code. This phase ensures we have a safety net and can validate that the new format produces identical results to the old format.

**Key Deliverables**:
- Test infrastructure for batch format validation
- Helper functions for converting between list-of-dicts and arrays
- Validation scripts to compare old vs new batch outputs

### Phase 2: Core Implementation
Implement Strategy A (array storage) in the batch save/load functions. This is done first because it's a prerequisite for Strategy B and provides immediate memory benefits even if used alone.

**Key Deliverables**:
- Modified `save_batch_to_file()` that saves arrays instead of list-of-dicts
- Modified `load_batch_from_file()` that loads arrays
- Backward compatibility to handle both old and new batch formats during transition

### Phase 3: Sequential Merge Implementation
Implement Strategy B (two-pass merge) that eliminates memory accumulation. This builds on Strategy A's array format to achieve maximum memory efficiency.

**Key Deliverables**:
- New `merge_batches_sequential()` function with two-pass algorithm
- Integration with existing preprocessing pipeline
- End-to-end validation that full pipeline works without OOM

## Step by Step Tasks

### 1. Create Test Infrastructure

- Create `tests/test_batch_storage.py` with fixtures for sample batch data
  - Create fixture that generates 1000 sample rows of synthetic data (strategic features, labels, masks)
  - Ensure synthetic data has realistic structure (893 strategic features, 7 label heads, 7 mask heads)
  - Include edge cases: NaN values, -1 labels, all-zeros masks

- Implement test for dict-to-array conversion round-trip
  - Test `convert_strategic_features_to_array()` and reverse conversion
  - Verify no data loss: dict → array → dict should be identical
  - Test with float32 vs float64 (config option)

- Implement test for label dict-to-arrays conversion
  - Test each of 7 label heads separately
  - Verify dtypes: action_type, card_selection, etc. should be int16
  - Verify -1 values preserved (indicates "not applicable")

- Implement test for mask dict-to-arrays conversion
  - Test each of 7 mask heads separately
  - Verify dtype: int8 (memory efficiency)
  - Verify shapes match expected classes (action_type: 4, card_selection: 15, etc.)

### 2. Implement Array Conversion Helper Functions

- Add `convert_strategic_features_to_array()` in `parallel_processor.py` (after imports, before `fill_nan_values_for_row`)
  ```python
  def convert_strategic_features_to_array(
      strategic_features_list: List[Dict],
      feature_names: List[str],
      dtype: str = 'float32'
  ) -> np.ndarray:
      """Convert list of strategic feature dicts to 2D array.

      Args:
          strategic_features_list: List of dicts, each with 893 keys
          feature_names: Ordered list of feature names (from get_all_feature_names())
          dtype: Array dtype, 'float32' or 'float64'

      Returns:
          Array of shape (n_samples, 893)
      """
  ```
  - Use `pd.DataFrame(strategic_features_list)[feature_names].values` for column ordering
  - Cast to specified dtype (float32 by default per config)
  - Handle missing keys by filling with 0.0

- Add `convert_labels_to_arrays()` in `parallel_processor.py`
  ```python
  def convert_labels_to_arrays(labels_list: List[Dict]) -> Dict[str, np.ndarray]:
      """Convert list of label dicts to dict of 1D arrays.

      Args:
          labels_list: List of dicts, each with 7 keys (one per head)

      Returns:
          Dict mapping head name to 1D array of shape (n_samples,)
      """
  ```
  - Extract each head separately: `action_type`, `card_selection`, `card_reservation`, `gem_take3`, `gem_take2`, `noble`, `gems_removed`
  - Use int16 dtype (sufficient for all label ranges: max is 84 classes for gems_removed)
  - Preserve -1 values (not applicable indicators)

- Add `convert_masks_to_arrays()` in `parallel_processor.py`
  ```python
  def convert_masks_to_arrays(masks_list: List[Dict]) -> Dict[str, np.ndarray]:
      """Convert list of mask dicts to dict of 2D arrays.

      Args:
          masks_list: List of dicts, each with 7 keys (one per head)

      Returns:
          Dict mapping head name to 2D array of shape (n_samples, n_classes)
      """
  ```
  - Use `np.stack()` for each head separately
  - Use int8 dtype (masks are binary 0/1)
  - Shapes: action_type (N,4), card_selection (N,15), card_reservation (N,15), gem_take3 (N,26), gem_take2 (N,5), noble (N,5), gems_removed (N,84)

- Add reverse conversion functions for load compatibility
  - `convert_array_to_strategic_features()`: array → list of dicts
  - `convert_arrays_to_labels()`: dict of arrays → list of dicts
  - `convert_arrays_to_masks()`: dict of arrays → list of dicts
  - These enable loading new format but returning old format for backward compatibility

### 3. Update `save_batch_to_file()` for Strategy A

- Modify `parallel_processor.py:285-336` (`save_batch_to_file`)
  - Import `get_all_feature_names` from `feature_engineering` at top of file
  - After receiving `strategic_features` list, convert to array:
    ```python
    feature_names = get_all_feature_names()
    strategic_features_array = convert_strategic_features_to_array(
        strategic_features, feature_names, dtype='float32'
    )
    ```
  - Convert labels list to dict of arrays: `labels_arrays = convert_labels_to_arrays(labels)`
  - Convert masks list to dict of arrays: `masks_arrays = convert_masks_to_arrays(masks)`
  - Update `np.savez_compressed()` call to save arrays with descriptive keys:
    ```python
    np.savez_compressed(
        batch_file,
        df_pickle=df_pickle,
        strategic_features=strategic_features_array,
        strategic_feature_names=feature_names,
        labels_action_type=labels_arrays['action_type'],
        labels_card_selection=labels_arrays['card_selection'],
        labels_card_reservation=labels_arrays['card_reservation'],
        labels_gem_take3=labels_arrays['gem_take3'],
        labels_gem_take2=labels_arrays['gem_take2'],
        labels_noble=labels_arrays['noble'],
        labels_gems_removed=labels_arrays['gems_removed'],
        masks_action_type=masks_arrays['action_type'],
        masks_card_selection=masks_arrays['card_selection'],
        masks_card_reservation=masks_arrays['card_reservation'],
        masks_gem_take3=masks_arrays['gem_take3'],
        masks_gem_take2=masks_arrays['gem_take2'],
        masks_noble=masks_arrays['noble'],
        masks_gems_removed=masks_arrays['gems_removed'],
        metadata=metadata,
    )
    ```
  - Update print statement to show array shapes and memory savings

### 4. Update `load_batch_from_file()` for Strategy A

- Modify `parallel_processor.py:338-361` (`load_batch_from_file`)
  - Detect batch format by checking keys:
    ```python
    data = np.load(batch_file, allow_pickle=True)

    # Detect format
    is_new_format = 'strategic_features' in data and len(data['strategic_features'].shape) == 2
    ```
  - For NEW format (arrays):
    - Load strategic features array directly: `strategic_features_array = data['strategic_features']`
    - Load labels arrays: `labels_arrays = {head: data[f'labels_{head}'] for head in HEAD_NAMES}`
    - Load masks arrays: `masks_arrays = {head: data[f'masks_{head}'] for head in HEAD_NAMES}`
    - Return tuple of `(df, strategic_features_array, labels_arrays, masks_arrays)` with new signature
  - For OLD format (list-of-dicts, backward compatibility):
    - Load as before: `strategic_features = list(data['strategic_features'])`
    - Convert to new format: `strategic_features_array = convert_strategic_features_to_array(strategic_features, feature_names)`
    - Return converted arrays
  - Update docstring to document new return types

### 5. Update `merge_batches()` Callers for Array Format

- Modify `merge_batches.py:90-159` (`merge_batches`)
  - Update function signature to indicate it returns arrays:
    ```python
    def merge_batches(
        batch_file_paths: List[str],
        config: Dict,
        monitor_memory: bool = False,
    ) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Merge batch files into combined arrays (NOT lists)."""
    ```
  - Update accumulation to handle arrays:
    ```python
    df_compacted_list = []
    strategic_features_list = []  # List of arrays, not dicts!
    labels_list = {head: [] for head in HEAD_NAMES}  # Separate list per head
    masks_list = {head: [] for head in HEAD_NAMES}

    for batch_file in batch_file_paths:
        batch_df, batch_features_array, batch_labels_arrays, batch_masks_arrays = load_batch_from_file(batch_file)

        df_compacted_list.append(batch_df)
        strategic_features_list.append(batch_features_array)  # Append array

        for head in HEAD_NAMES:
            labels_list[head].append(batch_labels_arrays[head])
            masks_list[head].append(batch_masks_arrays[head])

        del batch_df, batch_features_array, batch_labels_arrays, batch_masks_arrays
        gc.collect()
    ```
  - Concatenate arrays instead of extending lists:
    ```python
    df_compacted = pd.concat(df_compacted_list, axis=0, ignore_index=True)
    strategic_features_array = np.concatenate(strategic_features_list, axis=0)

    labels_arrays = {head: np.concatenate(labels_list[head], axis=0) for head in HEAD_NAMES}
    masks_arrays = {head: np.concatenate(masks_list[head], axis=0) for head in HEAD_NAMES}

    return df_compacted, strategic_features_array, labels_arrays, masks_arrays
    ```
  - Update print statements to show array shapes

- Modify `merge_batches.py:162-394` (`process_merged_data`)
  - Update function signature to accept arrays:
    ```python
    def process_merged_data(
        df_compacted: pd.DataFrame,
        strategic_features_array: np.ndarray,  # Changed from list
        labels_arrays: Dict[str, np.ndarray],  # Changed from list
        masks_arrays: Dict[str, np.ndarray],   # Changed from list
        config: Dict,
        skip_validation: bool = False,
    ) -> None:
    ```
  - Remove conversion steps (lines 182-192, 255-272) since data is already arrays
  - Update strategic features handling:
    ```python
    # Get expected feature names
    expected_feature_names = get_all_feature_names()

    # Create strategic DataFrame from array
    strategic_df = pd.DataFrame(
        strategic_features_array,
        columns=expected_feature_names
    )
    ```
  - Labels and masks already in correct format, use directly

### 6. Implement Strategy B - Sequential Two-Pass Merge

- Add new function `merge_batches_sequential()` in `merge_batches.py` (after `merge_batches()`, before `process_merged_data()`)
  ```python
  def merge_batches_sequential(
      batch_file_paths: List[str],
      config: Dict,
      monitor_memory: bool = False,
  ) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
      """Merge batches using sequential two-pass algorithm (memory-efficient).

      This eliminates memory accumulation by:
      1. Pass 1: Scan all batches to get total size
      2. Preallocate final arrays once
      3. Pass 2: Load each batch, write to preallocated arrays, release immediately

      Peak memory: ~size of final arrays (3-4GB) vs ~70GB for accumulative merge.

      Args:
          batch_file_paths: List of batch file paths
          config: Configuration dictionary
          monitor_memory: Whether to log memory usage

      Returns:
          Tuple of (df_compacted, strategic_features_array, labels_arrays, masks_arrays)
      """
  ```

- Implement Pass 1: Scan batches to calculate dimensions
  ```python
  # PASS 1: Calculate total size and get dimensions
  print(f"\n{'='*60}")
  print("PASS 1: Scanning batches to calculate dimensions...")
  print(f"{'='*60}")

  total_samples = 0
  num_features = None
  num_classes = {
      'action_type': 4,
      'card_selection': 15,
      'card_reservation': 15,
      'gem_take3': 26,
      'gem_take2': 5,
      'noble': 5,
      'gems_removed': 84,
  }

  for i, batch_file in enumerate(batch_file_paths):
      data = np.load(batch_file, allow_pickle=True)
      metadata = data['metadata'].item()
      n_samples = metadata['num_samples']
      total_samples += n_samples

      if num_features is None:
          num_features = data['strategic_features'].shape[1]

      print(f"  Batch {i+1}/{len(batch_file_paths)}: {n_samples:,} samples")

      del data
      gc.collect()

  print(f"\nTotal samples: {total_samples:,}")
  print(f"Feature dimension: {num_features}")
  ```

- Implement Pass 2: Preallocate arrays and fill sequentially
  ```python
  # PASS 2: Preallocate and fill arrays
  print(f"\n{'='*60}")
  print("PASS 2: Preallocating arrays and filling sequentially...")
  print(f"{'='*60}")

  if monitor_memory:
      log_memory_usage("Before array preallocation")

  # Preallocate all arrays at once
  strategic_features_array = np.zeros((total_samples, num_features), dtype=np.float32)
  labels_arrays = {head: np.zeros(total_samples, dtype=np.int16) for head in num_classes.keys()}
  masks_arrays = {
      head: np.zeros((total_samples, n_classes), dtype=np.int8)
      for head, n_classes in num_classes.items()
  }
  df_list = []  # Still need to collect DataFrames for concat

  if monitor_memory:
      log_memory_usage("After array preallocation")

  # Fill arrays batch by batch
  offset = 0
  for i, batch_file in enumerate(batch_file_paths):
      print(f"\nLoading and writing batch {i+1}/{len(batch_file_paths)}...")

      # Load batch
      batch_df, batch_features, batch_labels, batch_masks = load_batch_from_file(batch_file)
      n_samples = len(batch_df)

      # Write directly to preallocated arrays
      strategic_features_array[offset:offset+n_samples] = batch_features

      for head in num_classes.keys():
          labels_arrays[head][offset:offset+n_samples] = batch_labels[head]
          masks_arrays[head][offset:offset+n_samples] = batch_masks[head]

      df_list.append(batch_df)

      # Update offset
      offset += n_samples

      # Immediately release batch memory
      del batch_df, batch_features, batch_labels, batch_masks
      gc.collect()

      if monitor_memory:
          log_memory_usage(f"After batch {i+1}")

      print(f"  Written {offset:,} / {total_samples:,} samples ({offset/total_samples*100:.1f}%)")

  # Combine DataFrames (unavoidable, but much smaller than strategic features)
  print(f"\nCombining DataFrames...")
  df_compacted = pd.concat(df_list, axis=0, ignore_index=True)
  del df_list
  gc.collect()

  print(f"\n{'='*60}")
  print("Sequential merge complete!")
  print(f"  Total samples: {total_samples:,}")
  print(f"  Strategic features: {strategic_features_array.shape}")
  print(f"{'='*60}\n")

  return df_compacted, strategic_features_array, labels_arrays, masks_arrays
  ```

### 7. Add Configuration Option for Merge Strategy

- Modify `src/imitation_learning/configs/config_small.yaml` (after line 20)
  ```yaml
  merge_strategy: "sequential"  # "sequential" or "accumulative" (default: sequential)
  ```

- Update `preprocess_with_parallel_processing()` in `data_preprocessing.py` (around line 1193)
  ```python
  # Call appropriate merge strategy
  merge_strategy = config.get('preprocessing', {}).get('merge_strategy', 'sequential')

  if merge_strategy == 'sequential':
      print("Using sequential two-pass merge (memory-efficient)...")
      df_compacted, strategic_features_array, labels_arrays, masks_arrays = merge_batches_sequential(
          batch_file_paths, config, monitor_memory=monitor_memory
      )
  else:
      print("Using accumulative merge (legacy)...")
      df_compacted, strategic_features_array, labels_arrays, masks_arrays = merge_batches(
          batch_file_paths, config, monitor_memory=monitor_memory
      )
  ```

### 8. Create Batch Format Validation Script

- Create `scripts/validate_batch_migration.py`
  ```python
  """Validate that new array-based batch format produces identical results to old format.

  This script:
  1. Loads a batch file
  2. If old format, converts to new format and saves
  3. Loads both versions
  4. Compares all arrays to ensure bit-exact equality
  5. Reports memory usage differences

  Usage:
      python scripts/validate_batch_migration.py data/intermediate/batch_0000.npz
      python scripts/validate_batch_migration.py --all  # Validate all batches
  """
  ```

- Implement comparison logic:
  - Load old format batch (if it exists)
  - Convert to new format and save as `batch_XXXX_new.npz`
  - Load new format batch
  - Compare DataFrame: `pd.testing.assert_frame_equal(df_old, df_new)`
  - Compare arrays: `np.testing.assert_array_equal(array_old, array_new)`
  - Compare memory usage: report old vs new file sizes and in-memory sizes
  - Exit with error code if any differences found

### 9. Run Tests and Validation

- Run unit tests for array conversion functions
  ```bash
  uv run pytest tests/test_batch_storage.py -v
  ```
  - Verify all conversion functions work correctly
  - Verify round-trip conversions preserve data
  - Verify dtypes are correct (float32, int16, int8)

- Run test for sequential merge algorithm
  ```bash
  uv run pytest tests/test_sequential_merge.py -v
  ```
  - Verify two-pass algorithm produces correct results
  - Verify preallocated arrays are filled correctly
  - Verify offset calculations are accurate

- Validate batch format on existing batches (if old format)
  ```bash
  python scripts/validate_batch_migration.py --all
  ```
  - Should report any differences (expect none if implementation correct)
  - Should show memory savings (expect ~80-85% reduction)

### 10. Generate New Batch Files with Array Format

- Run preprocessing pipeline to regenerate batch files with new format
  ```bash
  uv run python -m src.imitation_learning.data_preprocessing \
    --config src/imitation_learning/configs/config_small.yaml \
    --parallel true \
    --skip-merge
  ```
  - This will create new batch files in `data/intermediate/` with array format
  - Verify batch files are smaller on disk (expect ~20-30% compression improvement)
  - Verify batch files load quickly with `load_batch_from_file()`

- Inspect new batch files
  ```bash
  uv run python -c "
  import numpy as np
  data = np.load('data/intermediate/batch_0000.npz', allow_pickle=True)
  print('Keys:', list(data.keys()))
  print('Strategic features shape:', data['strategic_features'].shape)
  print('Strategic features dtype:', data['strategic_features'].dtype)
  print('Labels action_type shape:', data['labels_action_type'].shape)
  print('Masks action_type shape:', data['masks_action_type'].shape)
  "
  ```

### 11. Test Sequential Merge End-to-End

- Run merge with sequential strategy
  ```bash
  uv run python -m src.imitation_learning.merge_batches \
    --config src/imitation_learning/configs/config_small.yaml \
    --intermediate-dir data/intermediate
  ```
  - Monitor memory usage (should peak at ~3-4GB, not 70GB+)
  - Verify merge completes without OOM
  - Verify final arrays are created in `data/processed/`

- Verify final output files
  ```bash
  ls -lh data/processed/
  # Should see: X_train.npy, X_val.npy, X_test.npy, labels_train.npz, etc.
  ```

- Check final array shapes
  ```bash
  uv run python -c "
  import numpy as np
  X_train = np.load('data/processed/X_train.npy')
  labels_train = np.load('data/processed/labels_train.npz')
  masks_train = np.load('data/processed/masks_train.npz')
  print('X_train shape:', X_train.shape)
  print('Labels keys:', list(labels_train.keys()))
  print('Masks keys:', list(masks_train.keys()))
  "
  ```

### 12. Run Full Preprocessing Pipeline End-to-End

- Run complete pipeline with new sequential merge
  ```bash
  uv run python -m src.imitation_learning.data_preprocessing \
    --config src/imitation_learning/configs/config_small.yaml \
    --parallel true
  ```
  - Monitor memory usage throughout (use `htop` or `watch -n1 free -h`)
  - Verify no OOM errors occur
  - Verify peak memory stays under 10GB (target: ~3-4GB)
  - Verify pipeline completes successfully and produces all expected output files

- Time the pipeline
  ```bash
  time uv run python -m src.imitation_learning.data_preprocessing \
    --config src/imitation_learning/configs/config_small.yaml \
    --parallel true
  ```
  - Compare timing vs old approach (should be similar or faster)

### 13. Validate Final Outputs Match Expected Format

- Load and inspect training data
  ```bash
  uv run python -c "
  import numpy as np
  import json

  # Load data
  X_train = np.load('data/processed/X_train.npy')
  labels_train = np.load('data/processed/labels_train.npz')
  masks_train = np.load('data/processed/masks_train.npz')

  with open('data/processed/feature_cols.json') as f:
      feature_cols = json.load(f)

  with open('data/processed/preprocessing_stats.json') as f:
      stats = json.load(f)

  # Verify shapes
  print('Dataset shapes:')
  print(f'  X_train: {X_train.shape}')
  print(f'  Expected features: {len(feature_cols)}')
  print(f'  Labels heads: {list(labels_train.keys())}')
  print(f'  Masks heads: {list(masks_train.keys())}')
  print()

  # Verify statistics
  print('Preprocessing stats:')
  print(f'  Total samples: {stats[\"total_samples\"]}')
  print(f'  Train size: {stats[\"train_size\"]}')
  print(f'  Val size: {stats[\"val_size\"]}')
  print(f'  Test size: {stats[\"test_size\"]}')
  print(f'  Input dim: {stats[\"input_dim\"]}')
  "
  ```

### 14. Run All Validation Commands

Execute every validation command listed in the Validation Commands section to ensure:
- No regressions in existing tests
- New functionality works correctly
- Memory usage is within acceptable limits
- All outputs match expected formats

## Testing Strategy

### Unit Tests

**Test File: `tests/test_batch_storage.py`**

1. **Test Array Conversion Functions**:
   - `test_convert_strategic_features_to_array()`: Verify list of 893-key dicts → (N, 893) array
   - `test_convert_strategic_features_round_trip()`: dict → array → dict should be identical
   - `test_convert_labels_to_arrays()`: Verify 7 label heads extracted correctly
   - `test_convert_labels_dtypes()`: Verify all labels are int16
   - `test_convert_masks_to_arrays()`: Verify 7 mask heads with correct shapes
   - `test_convert_masks_dtypes()`: Verify all masks are int8

2. **Test Batch Save/Load Round-Trip**:
   - `test_save_load_batch_array_format()`: Save batch with arrays, load, verify identical
   - `test_batch_file_size()`: Verify new format is smaller on disk
   - `test_batch_load_memory()`: Verify loading new format uses less memory

3. **Test Backward Compatibility**:
   - `test_load_old_format_batch()`: Load old list-of-dicts format, verify converts correctly
   - `test_mixed_format_merge()`: Merge mix of old and new format batches, verify works

**Test File: `tests/test_sequential_merge.py`**

1. **Test Two-Pass Algorithm**:
   - `test_calculate_total_size()`: Verify Pass 1 calculates correct dimensions
   - `test_preallocate_arrays()`: Verify arrays preallocated with correct shapes and dtypes
   - `test_sequential_write()`: Verify batches written to correct offsets
   - `test_offset_calculation()`: Verify offsets calculated correctly across batches

2. **Test Merge Correctness**:
   - `test_sequential_vs_accumulative()`: Verify sequential merge produces identical results to accumulative
   - `test_sequential_merge_preserves_order()`: Verify samples maintain correct order
   - `test_sequential_merge_no_data_loss()`: Verify all samples present in final arrays

3. **Test Memory Efficiency**:
   - `test_sequential_merge_memory_usage()`: Verify peak memory is bounded by final array size
   - `test_no_accumulation()`: Verify intermediate lists don't grow during merge

### Integration Tests

1. **End-to-End Preprocessing Pipeline**:
   - Run full pipeline on small subset (10 games)
   - Verify all output files created
   - Verify shapes and dtypes correct
   - Verify training can load and use the data

2. **Memory Monitoring Integration**:
   - Run pipeline with `monitor_memory: true` in config
   - Verify memory logs show bounded usage
   - Verify no memory leaks (usage returns to baseline after operations)

3. **Merge Strategy Switching**:
   - Run with `merge_strategy: accumulative` and `merge_strategy: sequential`
   - Verify both produce identical outputs
   - Verify sequential uses significantly less memory

### Edge Cases

1. **Empty Batches**: Batch with 0 samples (skip gracefully)
2. **Single Batch**: Only 1 batch file (should work without special casing)
3. **Very Large Batch**: Batch with 100K+ samples (test memory scaling)
4. **Missing Features**: Strategic features dict missing some keys (fill with 0.0)
5. **NaN Values**: Labels/masks with NaN (should preserve as -1)
6. **Mixed Dtypes**: Old batches with float64, new with float32 (convert consistently)
7. **Corrupt Batch File**: Handle gracefully with error message
8. **Disk Full**: Handle write failures during batch save
9. **Interrupted Merge**: Can resume or restart without corruption

## Acceptance Criteria

1. **Memory Efficiency**:
   - ✅ Peak memory usage during preprocessing ≤ 10GB (target: 3-4GB)
   - ✅ No OOM errors on 64GB RAM system with 547K sample dataset
   - ✅ Memory usage bounded and predictable (no accumulation)

2. **Correctness**:
   - ✅ Sequential merge produces bit-exact identical results to accumulative merge
   - ✅ All 547,224 samples present in final arrays (no data loss)
   - ✅ Sample order preserved (can map back to game_id and turn_number)
   - ✅ All label and mask arrays have correct shapes and dtypes
   - ✅ All tests pass with 100% success rate

3. **File Format**:
   - ✅ Batch files use array format (not list-of-dicts)
   - ✅ Batch files are 80-85% smaller in memory than old format
   - ✅ Batch files are 20-30% smaller on disk (better compression)
   - ✅ Backward compatibility: can load old format batches (if needed for debugging)

4. **Performance**:
   - ✅ Preprocessing pipeline completes in reasonable time (≤ 1.5x old pipeline time)
   - ✅ Batch save/load operations are faster (array I/O vs pickle)
   - ✅ Merge operation is not significantly slower (target: within 10% of old time)

5. **Usability**:
   - ✅ Can inspect batch files with simple NumPy commands (no complex unpickling)
   - ✅ Configuration option to choose merge strategy (sequential vs accumulative)
   - ✅ Clear logging of memory usage and progress
   - ✅ Validation scripts to verify correctness

6. **Documentation**:
   - ✅ Docstrings for all new functions
   - ✅ Comments explaining two-pass algorithm
   - ✅ README updates (if needed) explaining new batch format
   - ✅ This spec document serves as implementation guide

## Validation Commands

Execute every command to validate the feature works correctly with zero regressions.

- `uv run pytest tests/test_batch_storage.py -v` - Test array conversion and batch save/load functions
- `uv run pytest tests/test_sequential_merge.py -v` - Test two-pass merge algorithm
- `uv run pytest tests/test_feature_engineering.py -v` - Verify existing feature engineering tests still pass (no regressions)
- `python scripts/validate_batch_migration.py --all` - Validate old vs new batch format produce identical results
- `uv run python -m src.imitation_learning.data_preprocessing --config src/imitation_learning/configs/config_small.yaml --parallel true --skip-merge` - Generate new array-based batch files
- `uv run python -m src.imitation_learning.merge_batches --config src/imitation_learning/configs/config_small.yaml` - Test sequential merge in isolation
- `uv run python -m src.imitation_learning.data_preprocessing --config src/imitation_learning/configs/config_small.yaml --parallel true` - Full end-to-end pipeline test
- `uv run python -c "import numpy as np; X = np.load('data/processed/X_train.npy'); print(f'X_train shape: {X.shape}, dtype: {X.dtype}, memory: {X.nbytes / 1024 / 1024:.1f} MB')"` - Verify final training data
- `uv run python -c "import numpy as np; labels = np.load('data/processed/labels_train.npz'); print('Labels heads:', list(labels.keys())); print('Action type shape:', labels['action_type'].shape)"` - Verify labels format
- `uv run python -c "import numpy as np; masks = np.load('data/processed/masks_train.npz'); print('Masks heads:', list(masks.keys())); print('Action type shape:', masks['action_type'].shape)"` - Verify masks format

## Notes

### Memory Estimation Clarification
The initial analysis estimated 12-13GB total memory for the old approach. However, with 64GB RAM + 16GB swap (80GB total), experiencing OOM indicates the actual peak is higher. Contributing factors:

1. **Python Memory Overhead**: List-of-dicts have ~3-4x overhead beyond theoretical minimum
2. **Pandas Concat Copies**: Creates temporary copy of entire DataFrame during concat
3. **Fragmentation**: Long-running Python process with many allocations/deallocations
4. **Temporary Allocations**: During array conversions, brief periods with 2x memory
5. **OS and Other Processes**: ~10-15GB reserved for system and other applications

Realistic peak with old approach: **65-75GB**, exceeding available memory.

With optimizations:
- **Strategy A only**: Reduces to ~15-20GB (feasible but tight)
- **Strategy A + B**: Reduces to ~3-4GB (plenty of headroom)

### Future Enhancements
After this feature is stable, consider:

1. **HDF5 Format**: For even better scalability and convenience
   - Eliminates need for intermediate batch files
   - Direct chunked writing during preprocessing
   - Better inspection tools (h5ls, HDFView)

2. **Compression Tuning**: Experiment with compression levels
   - Current: `np.savez_compressed` uses gzip level 6
   - Try levels 3-4 for faster I/O with acceptable compression

3. **Float16 for Strategic Features**: Some features could use half precision
   - 50% more memory savings
   - Need to verify model accuracy not affected

4. **Parallel Merge**: Parallelize array writes within sequential merge
   - Multiple batches read and written concurrently
   - Requires careful offset management and locking

### Rollback Plan
If issues arise:
1. Set `merge_strategy: accumulative` in config to use old merge logic
2. Old batch files can still be loaded (backward compatibility)
3. Regenerate batches without changes if needed (original CSV files unchanged)

### Dependencies
No new dependencies required. Uses only:
- `numpy` (already in use)
- `pandas` (already in use)
- `pickle` (standard library)
- `gc` (standard library)
