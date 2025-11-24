"""Unit tests for batch storage array conversion functions."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from imitation_learning.parallel_processor import (
    HEAD_NAMES,
    NUM_CLASSES,
    convert_arrays_to_labels,
    convert_arrays_to_masks,
    convert_array_to_strategic_features,
    convert_labels_to_arrays,
    convert_masks_to_arrays,
    convert_strategic_features_to_array,
)


def test_convert_strategic_features_to_array():
    """Test strategic features list-of-dicts to array conversion."""
    # Create sample data
    feature_names = [f"feature_{i}" for i in range(10)]
    strategic_features_list = [
        {f"feature_{i}": float(i + j) for i in range(10)}
        for j in range(5)
    ]

    # Convert to array
    array = convert_strategic_features_to_array(
        strategic_features_list, feature_names, dtype='float32'
    )

    # Verify shape and dtype
    assert array.shape == (5, 10)
    assert array.dtype == np.float32

    # Verify values
    for j in range(5):
        for i in range(10):
            assert array[j, i] == float(i + j)


def test_convert_strategic_features_round_trip():
    """Test strategic features round-trip conversion (dict -> array -> dict)."""
    feature_names = [f"feature_{i}" for i in range(10)]
    original_list = [
        {f"feature_{i}": float(i + j) for i in range(10)}
        for j in range(3)
    ]

    # Convert to array and back
    array = convert_strategic_features_to_array(original_list, feature_names)
    converted_list = convert_array_to_strategic_features(array, feature_names)

    # Verify identical (within floating point precision)
    assert len(converted_list) == len(original_list)
    for orig, conv in zip(original_list, converted_list):
        for key in feature_names:
            assert abs(orig[key] - conv[key]) < 1e-6


def test_convert_labels_to_arrays():
    """Test labels list-of-dicts to arrays conversion."""
    labels_list = [
        {head: i + idx for idx, head in enumerate(HEAD_NAMES)}
        for i in range(5)
    ]

    # Convert to arrays
    labels_arrays = convert_labels_to_arrays(labels_list)

    # Verify structure
    assert len(labels_arrays) == len(HEAD_NAMES)
    for head in HEAD_NAMES:
        assert head in labels_arrays
        assert labels_arrays[head].shape == (5,)
        assert labels_arrays[head].dtype == np.int16

    # Verify values
    for i in range(5):
        for idx, head in enumerate(HEAD_NAMES):
            assert labels_arrays[head][i] == i + idx


def test_convert_labels_round_trip():
    """Test labels round-trip conversion."""
    original_list = [
        {head: i + idx for idx, head in enumerate(HEAD_NAMES)}
        for i in range(3)
    ]

    # Convert to arrays and back
    arrays = convert_labels_to_arrays(original_list)
    converted_list = convert_arrays_to_labels(arrays)

    # Verify identical
    assert len(converted_list) == len(original_list)
    for orig, conv in zip(original_list, converted_list):
        for head in HEAD_NAMES:
            assert orig[head] == conv[head]


def test_convert_masks_to_arrays():
    """Test masks list-of-dicts to arrays conversion."""
    # Create sample masks with correct shapes
    masks_list = [
        {
            head: np.random.randint(0, 2, size=NUM_CLASSES[head])
            for head in HEAD_NAMES
        }
        for _ in range(5)
    ]

    # Convert to arrays
    masks_arrays = convert_masks_to_arrays(masks_list)

    # Verify structure
    assert len(masks_arrays) == len(HEAD_NAMES)
    for head in HEAD_NAMES:
        assert head in masks_arrays
        assert masks_arrays[head].shape == (5, NUM_CLASSES[head])
        assert masks_arrays[head].dtype == np.int8

    # Verify values (check that they're 0 or 1)
    for head in HEAD_NAMES:
        assert np.all((masks_arrays[head] == 0) | (masks_arrays[head] == 1))


def test_convert_masks_round_trip():
    """Test masks round-trip conversion."""
    original_list = [
        {
            head: np.random.randint(0, 2, size=NUM_CLASSES[head])
            for head in HEAD_NAMES
        }
        for _ in range(3)
    ]

    # Convert to arrays and back
    arrays = convert_masks_to_arrays(original_list)
    converted_list = convert_arrays_to_masks(arrays)

    # Verify identical
    assert len(converted_list) == len(original_list)
    for orig, conv in zip(original_list, converted_list):
        for head in HEAD_NAMES:
            assert np.array_equal(orig[head], conv[head])


def test_missing_features_filled_with_zeros():
    """Test that missing features are filled with zeros."""
    feature_names = [f"feature_{i}" for i in range(10)]
    # Only provide 5 out of 10 features
    strategic_features_list = [
        {f"feature_{i}": float(i) for i in range(5)}
    ]

    # Convert to array
    array = convert_strategic_features_to_array(
        strategic_features_list, feature_names
    )

    # Verify shape
    assert array.shape == (1, 10)

    # Verify first 5 features have values, rest are zeros
    for i in range(5):
        assert array[0, i] == float(i)
    for i in range(5, 10):
        assert array[0, i] == 0.0


def test_labels_preserve_negative_one():
    """Test that -1 values in labels are preserved (not applicable indicators)."""
    labels_list = [
        {head: -1 for head in HEAD_NAMES}
    ]

    # Convert to arrays
    labels_arrays = convert_labels_to_arrays(labels_list)

    # Verify -1 is preserved
    for head in HEAD_NAMES:
        assert labels_arrays[head][0] == -1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
