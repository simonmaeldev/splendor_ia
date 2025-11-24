"""Constants for imitation learning preprocessing.

This module pre-computes expensive gem class mappings at import time,
eliminating the need to recompute them for every CSV file processed.

Memory and performance impact:
- Before: 7,259 files × 4,122 iterations = 29.9M wasted loop iterations
- After: Computed once at module import (< 0.001 seconds)
- Savings: 29.9M redundant iterations eliminated

These mappings are pure functions based on game rules and never change.
"""

from .utils import generate_gem_take3_classes, generate_gem_removal_classes

# Computed once at import time (not per-file, not per-row)
CLASS_TO_COMBO_TAKE3, COMBO_TO_CLASS_TAKE3 = generate_gem_take3_classes()
CLASS_TO_REMOVAL, REMOVAL_TO_CLASS = generate_gem_removal_classes()
NUM_GEM_REMOVAL_CLASSES = len(CLASS_TO_REMOVAL)

# For reference and validation
NUM_GEM_TAKE3_CLASSES = len(CLASS_TO_COMBO_TAKE3)  # Should be 26

__all__ = [
    'CLASS_TO_COMBO_TAKE3',
    'COMBO_TO_CLASS_TAKE3',
    'CLASS_TO_REMOVAL',
    'REMOVAL_TO_CLASS',
    'NUM_GEM_REMOVAL_CLASSES',
    'NUM_GEM_TAKE3_CLASSES',
]
