# Bug: Memory Leak and Inefficient Architecture in Data Preprocessing Pipeline

## Bug Description

The data preprocessing pipeline has severe memory and performance issues when processing 7,259 CSV files (~547,000 rows):

**Memory Symptoms:**
- Starts fast but progressively slows down
- RAM fills completely (60GB consumed)
- Swap usage reaches 10GB+
- Eventually processes only 1 file at a time
- System becomes extremely slow

**Performance Issues:**
- Computes gem class mappings 7,259 times (should be once)
- Does one-hot encoding AFTER accumulation (should be per-row)
- Multiple DataFrame copies and redundant operations
- Converts between data structures unnecessarily

## Problem Statement

The pipeline has **fundamental architectural inefficiencies**:

### 1. Repeated Constant Computation (30M wasted iterations)
- `generate_gem_take3_classes()` called 7,259 times (once per file)
- `generate_gem_removal_classes()` called 7,259 times, each with 4^6 = 4,096 loop iterations
- **Total waste**: 7,259 × 4,096 = 29.7 million iterations computing the SAME 84 mappings
- These are pure functions based on game rules - should be constants

### 2. Post-Accumulation Processing (should be per-row)
After accumulating all 547K rows, the pipeline does:
- Lines 1102-1125: One-hot encoding (could be done per-row during file processing)
- Lines 1127-1150: Column identification and feature list updates
- Lines 1152-1161: Converting lists to dicts of arrays
- Lines 1084-1096: DataFrame reformatting and concatenation
- **All of this is row-independent** and could happen during file processing

### 3. Unbounded Memory Accumulation
- All 547K rows kept in memory as lists
- No batching or intermediate persistence
- Multiple full DataFrame copies (lines 185, 212, 389, 773-775)
- Peak memory: 60-90GB (should be <25GB)

### 4. Inefficient Data Types
- Uses float64 (8 bytes) instead of float32 (4 bytes)
- Uses int64 (8 bytes) instead of int32 (4 bytes)
- **Wastes 50% memory** for no benefit (game features don't need double precision)

### 5. Excessive DataFrame Operations
- Multiple `.copy()` operations creating full duplicates
- `pd.concat()` instead of in-place column addition
- Converting lists → DataFrame → reorder → extract → convert back

## Solution Statement

**Restructure pipeline to be clean, efficient, and memory-conscious:**

1. **Move gem class generation to constants** (compute once, never again)
2. **Do all row-independent processing per-row** (one-hot encoding, feature formatting)
3. **Return properly formatted numpy arrays** (no post-processing needed)
4. **Add batched processing** (prevent memory overflow)
5. **Use appropriate dtypes** (float32, int32, int8)
6. **Minimize DataFrame copies** (in-place operations where possible)

**Architecture: Process → Batch → Combine → Normalize → Split → Save**

```
For each batch of files:
  For each file:
    For each row:
      1. Load and parse CSV row
      2. Generate masks (reconstruct board once)
      3. Fill NaN, compact cards
      4. Extract strategic features (reuse board)
      5. One-hot encode categorical features ← DO THIS NOW
      6. Encode labels
      7. Assemble complete feature vector ← FINAL FORMAT
      8. Append to batch arrays (numpy, proper dtype)
  Save batch to disk (compressed NPZ)
  Clear memory

Combine batches (load one at a time)
Split by game_id
Normalize (fit on train, transform all)
Save final arrays
```

This eliminates all post-processing redundancy and keeps memory bounded.

## Steps to Reproduce

1. Set up environment with 60GB RAM
2. Prepare dataset: 7,259 CSV files
3. Run preprocessing:
   ```bash
   cd /home/apprentyr/projects/splendor_ia
   uv run python -m src.imitation_learning.data_preprocessing \
     --config src/imitation_learning/configs/config_small.yaml \
     --parallel true
   ```
4. Monitor memory:
   ```bash
   watch -n 2 'free -h && ps aux | grep python | grep -v grep | awk "{print \$6/1024\" MB\"}"'
   ```
5. Observe: Memory fills, processing slows, swap usage increases

## Root Cause Analysis

### Memory Growth Timeline

```
Stage                           Memory Usage
====================================================
1. Load CSVs                    ~10 GB (raw DataFrames)
2. Accumulate dicts             ~10 + 33 = 43 GB
3. Convert to DataFrame         43 + 3.64 = 46.64 GB
4. Reorder columns              46.64 + 3.64 = 50.28 GB
5. Concat with df_eng           50.28 + 5.58 = 55.86 GB
6. One-hot encoding             55.86 + ? = 60+ GB
7. Garbage collection           May free some, but fragmented
8. Swap starts                  System desperate for memory
```

### PRIMARY ISSUE: Dict Accumulation (33 GB!)

**Code: `data_preprocessing.py:437-441`**
```python
# MEMORY ISSUE: Accumulates 547K dicts in memory (~33 GB)
strategic_features_list = []  # Empty list

for idx, row in tqdm(df_eng.iterrows(), total=len(df_eng)):
    features = extract_all_features(row)  # Returns Dict[str, float] with 893 entries
    strategic_features_list.append(features)  # List grows: 0 → 547K dicts
    # Each dict: ~65 KB (893 keys + values + overhead)
    # Total: 547K × 65 KB = 33 GB
```

**Why dicts are expensive:**
- **String keys**: 893 keys × ~40 bytes = 35.7 KB
- **Float64 values**: 893 values × 8 bytes = 7.1 KB
- **Dict overhead**: Hash table, buckets ~20-30 KB
- **Total per dict**: ~65 KB
- **For 547K rows**: 33 GB vs numpy array (1.82 GB with float32)

**What's in the dict:** (`feature_engineering.py:74`)
```python
def extract_all_features(row: pd.Series, board: 'Board' = None) -> FeatureDict:
    features: FeatureDict = {}  # Dict[str, float]

    features.update(extract_token_features(row, board))      # 26 items
    features.update(extract_card_features(row, board))       # 540 items
    features.update(extract_noble_features(row, board))      # 148 items
    features.update(extract_card_noble_synergy(row, board))  # 120 items
    features.update(extract_player_comparison_features(...)) # 50 items
    features.update(extract_game_progression_features(...))  # 9 items

    return features  # 893 string keys → 43 KB just for keys!
```

### SECONDARY ISSUE: DataFrame Conversion While List in Memory

**Code: `data_preprocessing.py:444`**
```python
# MEMORY ISSUE: Creates 3.64 GB DataFrame WHILE list (33 GB) still in memory
strategic_df = pd.DataFrame(strategic_features_list)
# DataFrame: 547K rows × 893 cols × 8 bytes = 3.64 GB
# List STILL in memory: 33 GB
# PEAK: 33 + 3.64 = 36.64 GB
```

### TERTIARY ISSUE: Column Reordering Creates Copy

**Code: `data_preprocessing.py:453`**
```python
# MEMORY ISSUE: Column indexing creates a COPY of the DataFrame
strategic_df = strategic_df[expected_feature_names]
# Creates NEW DataFrame (3.64 GB)
# Old DataFrame still in memory until gc
# PEAK: 36.64 + 3.64 = 40.28 GB
```

### QUATERNARY ISSUE: pd.concat Creates Another Copy

**Code: `data_preprocessing.py:456`**
```python
# MEMORY ISSUE: pd.concat creates YET ANOTHER DataFrame
df_eng = pd.concat([df_eng, strategic_df], axis=1)
# df_eng before: 547K × 474 features = 1.94 GB
# strategic_df: 547K × 893 features = 3.64 GB
# Creates NEW combined: 5.58 GB
# Old df_eng still referenced
# PEAK: 40.28 + 5.58 = 45.86 GB
```

### Additional Issues:

**5. Unbounded accumulation in parallel processor**
**Code: `parallel_processor.py:315-326`**
```python
# MEMORY ISSUE: Accumulates ALL files in memory
all_raw_rows = []              # Grows to 547K dicts
all_strategic_features = []    # Grows to 547K dicts (33 GB!)
all_labels = []                # Grows to 547K dicts
all_masks = []                 # Grows to 547K arrays

for file_path in tqdm(file_paths, desc="Processing files"):
    raw_rows, strategic_features, labels, masks = process_single_file(...)

    all_raw_rows.extend(raw_rows)
    all_strategic_features.extend(strategic_features)  # 33 GB here!
    all_labels.extend(labels)
    all_masks.extend(masks)

    gc.collect()  # Too late! Data still referenced in lists
```

**6. Gem class enumeration every file**
**Code: `parallel_processor.py:252-253`**
```python
def process_single_file(file_path: str, config: Dict):
    # PERFORMANCE ISSUE: Called 7,259 times (once per file)
    _, combo_to_class_take3 = generate_gem_take3_classes()  # 26 iterations
    _, removal_to_class = generate_gem_removal_classes()    # 4,096 iterations

    # Total: 7,259 × 4,122 = 29.9 million wasted iterations
```

**7. df.iterrows() inefficiency**
**Code: `data_preprocessing.py:439`**
```python
# PERFORMANCE + MEMORY ISSUE
for idx, row in tqdm(df_eng.iterrows(), total=len(df_eng)):
#              ↑↑↑↑↑↑↑↑
# Creates a NEW pd.Series object for each row
# Series overhead: ~1 KB per row
# 547K Series objects: 547 MB
# Also SLOW: 100x slower than itertuples() or numpy iteration
```

**8. Player reductions copy**
**Code: `feature_engineering.py:246, 407`**
```python
# In extract_card_features():
new_reductions = player.reductions.copy()  # Creates new list (40 bytes)

# In extract_card_noble_synergy():
new_reductions = player.reductions.copy()  # Another copy

# Called for EVERY card, EVERY row:
# 547K rows × 15 cards × 2 copies × 40 bytes = 656 MB
```

## Relevant Files

### Primary Files to Modify:

- **`src/imitation_learning/constants.py`** (NEW FILE)
  - Add `GEM_TAKE3_CLASSES` constant (computed once at import)
  - Add `GEM_REMOVAL_CLASSES` constant (computed once at import)
  - Eliminates 7,259 redundant function calls

- **`src/imitation_learning/parallel_processor.py`**
  - Lines 149-219: `process_single_row()` - Add one-hot encoding here
  - Lines 222-290: `process_single_file()` - Return numpy arrays with proper dtypes
  - Lines 293-339: `process_files_parallel()` - Add batching, save to disk
  - Import constants instead of generating mappings

- **`src/imitation_learning/data_preprocessing.py`**
  - Lines 1046-1248: `preprocess_with_parallel_processing()` - Simplify (no post-processing)
  - Lines 1102-1161: DELETE redundant post-processing code
  - Lines 437-464: Replace dict accumulation with numpy array pre-allocation
  - Lines 185, 212, 389: Review `.copy()` necessity
  - Import and use gem class constants

- **`src/imitation_learning/utils.py`**
  - Lines 63-121: `generate_gem_take3_classes()` - Keep for constant initialization
  - Lines 124-165: `generate_gem_removal_classes()` - Keep for constant initialization
  - Update all call sites to use constants

### Configuration File:

- **`src/imitation_learning/configs/config_small.yaml`**
  - Add `preprocessing.batch_size: 500`
  - Add `preprocessing.use_float32: true`
  - Add `preprocessing.monitor_memory: true`
  - Add `preprocessing.intermediate_dir: "data/intermediate"`

## Step by Step Tasks

### Step 1: Create Constants Module
- Create `src/imitation_learning/constants.py`
- Move gem class generation to module-level initialization:
  ```python
  from .utils import generate_gem_take3_classes, generate_gem_removal_classes

  # Computed once at import time
  CLASS_TO_COMBO_TAKE3, COMBO_TO_CLASS_TAKE3 = generate_gem_take3_classes()
  CLASS_TO_REMOVAL, REMOVAL_TO_CLASS = generate_gem_removal_classes()
  NUM_GEM_REMOVAL_CLASSES = len(CLASS_TO_REMOVAL)
  ```
- Update imports in `parallel_processor.py`, `data_preprocessing.py`, `utils.py`
- **Memory savings**: Eliminates 29.9M redundant loop iterations

### Step 2: Fix Strategic Feature Accumulation (Biggest Win!)
- **Replace dict accumulation with numpy array pre-allocation**
- Update `data_preprocessing.py:437-464`:
  ```python
  # BEFORE (33 GB memory):
  strategic_features_list = []
  for idx, row in tqdm(df_eng.iterrows(), total=len(df_eng)):
      features = extract_all_features(row)
      strategic_features_list.append(features)
  strategic_df = pd.DataFrame(strategic_features_list)

  # AFTER (1.82 GB memory):
  n_rows = len(df_eng)
  strategic_features = np.zeros((n_rows, 893), dtype=np.float32)
  feature_names = get_all_feature_names()

  for idx in tqdm(range(n_rows), desc="Extracting strategic features"):
      row = df_eng.iloc[idx]
      features_dict = extract_all_features(row)
      # Convert dict to array in consistent order
      strategic_features[idx] = np.array(
          [features_dict.get(name, 0.0) for name in feature_names],
          dtype=np.float32
      )

      # Periodic cleanup
      if idx % 10000 == 0:
          gc.collect()

  # Add to df_eng as numpy array (no DataFrame conversion needed)
  ```
- **Memory savings**: 33 GB → 1.82 GB (18x reduction!)

### Step 3: Refactor Per-Row Processing
- Update `process_single_row()` in `parallel_processor.py`:
  - Add one-hot encoding for `current_player` (4 features)
  - Add one-hot encoding for `num_players` (3 features: 2, 3, 4)
  - Add one-hot encoding for player positions (16 features: 4 players × 4 positions)
  - Build complete feature vector: raw features + one-hot + strategic features
  - Return as numpy arrays with dtypes: `float32`, `int32`, `int8`

- Signature change:
  ```python
  def process_single_row(...) -> Tuple[
      np.ndarray,  # features: shape (n_features,), dtype float32
      Dict[str, int],  # labels: dtype int32
      Dict[str, np.ndarray]  # masks: dtype int8
  ]
  ```

### Step 4: Refactor Per-File Processing
- Update `process_single_file()` in `parallel_processor.py`:
  - Use gem class constants (not function calls)
  - Pre-allocate numpy arrays for file rows:
    ```python
    n_rows = len(df)
    n_features = 1367  # Known from architecture

    feature_array = np.zeros((n_rows, n_features), dtype=np.float32)
    label_arrays = {head: np.zeros(n_rows, dtype=np.int32) for head in head_names}
    mask_arrays = {head: np.zeros((n_rows, n_classes[head]), dtype=np.int8)
                   for head in head_names}
    ```
  - Return numpy arrays (not lists of dicts):
    ```python
    return (
        feature_array,  # (n_rows, 1367) float32
        label_arrays,   # {head: (n_rows,) int32}
        mask_arrays,    # {head: (n_rows, n_classes) int8}
        metadata_df     # Small DF with game_id, turn_number for splitting
    )
    ```

### Step 5: Implement Batched Processing
- Refactor `process_files_parallel()` in `parallel_processor.py`:
  - Split files into batches of 500
  - Process each batch completely
  - Save batch to compressed NPZ file
  - Clear memory after each batch
  - Return list of batch file paths

### Step 6: Simplify Main Pipeline
- Update `preprocess_with_parallel_processing()` in `data_preprocessing.py`:
  - Load and combine batch files one at a time
  - DELETE lines 1102-1161 (redundant post-processing)
  - Simplify to: load batches → split by game_id → normalize → save

### Step 7: Add Memory Monitoring
- Create `log_memory_usage()` helper
- Call at key points: start, after each batch, end

### Step 8: Update Configuration
- Add to `config_small.yaml`:
  - `batch_size: 500`
  - `use_float32: true`
  - `monitor_memory: true`
  - `intermediate_dir: "data/intermediate"`

### Step 9: Testing and Validation
- Test with 100, 1000, then 7,259 files
- Verify output correctness, dtypes, memory usage

## Validation Commands

```bash
# Install memory monitoring dependency
uv add psutil

# Test small subset (100 files)
uv run python -m src.imitation_learning.data_preprocessing \
  --config src/imitation_learning/configs/config_small.yaml \
  --max-games 100 \
  --parallel true

# Verify output shape and dtype
uv run python -c "
import numpy as np
X = np.load('data/processed/X_train.npy')
print(f'Shape: {X.shape}')
print(f'Dtype: {X.dtype}')  # Should be float32
print(f'Memory: {X.nbytes / 1024**2:.1f} MB')
"

# Monitor memory during full run
watch -n 2 'free -h'

# Full test
uv run python -m src.imitation_learning.data_preprocessing \
  --config src/imitation_learning/configs/config_small.yaml \
  --parallel true

# Verify no swap usage
free -h | grep Swap

# Check output correctness
uv run python -c "
import numpy as np
X_train = np.load('data/processed/X_train.npy')
X_val = np.load('data/processed/X_val.npy')
X_test = np.load('data/processed/X_test.npy')
print(f'Train: {X_train.shape}, dtype: {X_train.dtype}')
print(f'Val: {X_val.shape}, dtype: {X_val.dtype}')
print(f'Test: {X_test.shape}, dtype: {X_test.dtype}')
print(f'No NaN: {not (np.isnan(X_train).any() or np.isnan(X_val).any() or np.isnan(X_test).any())}')
"
```

## Notes

### Memory Breakdown (547K rows):

| Component | Before (dict) | After (numpy float32) | Savings |
|-----------|--------------|----------------------|---------|
| Strategic features | 33 GB | 1.82 GB | 31 GB (95%) |
| Base features | 1.94 GB | 0.97 GB | 0.97 GB (50%) |
| Labels | ~300 MB | ~150 MB | 150 MB (50%) |
| Masks | ~55 MB | ~55 MB | 0 MB (already int8) |
| **Total** | **~35 GB** | **~3 GB** | **~32 GB (91%)** |

### Key Improvements:

1. **Eliminate dict accumulation** (BIGGEST WIN):
   - Before: 547K dicts × 65 KB = 33 GB
   - After: Numpy array 547K × 893 × 4 bytes = 1.82 GB
   - **Savings**: 31 GB (95% reduction!)

2. **Eliminate redundant computation**:
   - Before: 29.7M loop iterations computing gem classes
   - After: Computed once at import (0.001 seconds)

3. **Overall memory efficiency**:
   - Before: 60-90 GB peak
   - After: 15-25 GB peak
   - **Savings**: 60-75% memory reduction
