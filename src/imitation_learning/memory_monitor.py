"""Memory monitoring utilities for data preprocessing.

This module provides tools to track memory usage during preprocessing,
helping identify memory leaks and optimize memory efficiency.
"""

import gc
import os
from typing import Optional

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("WARNING: psutil not available. Memory monitoring will be limited.")


def get_memory_usage() -> dict:
    """Get current memory usage statistics.

    Returns:
        Dict with memory statistics in MB:
        - rss: Resident Set Size (physical memory used)
        - vms: Virtual Memory Size (total virtual memory)
        - percent: Percentage of system memory used
        - available: Available system memory
        - total: Total system memory
    """
    if not PSUTIL_AVAILABLE:
        return {
            'rss_mb': 0,
            'vms_mb': 0,
            'percent': 0,
            'available_mb': 0,
            'total_mb': 0,
        }

    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    system_memory = psutil.virtual_memory()

    return {
        'rss_mb': memory_info.rss / 1024 / 1024,  # Physical memory
        'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual memory
        'percent': process.memory_percent(),
        'available_mb': system_memory.available / 1024 / 1024,
        'total_mb': system_memory.total / 1024 / 1024,
    }


def log_memory_usage(label: str = "", force_gc: bool = False) -> None:
    """Log current memory usage with optional garbage collection.

    Args:
        label: Descriptive label for this measurement
        force_gc: If True, run garbage collection before measuring
    """
    if force_gc:
        gc.collect()

    memory = get_memory_usage()

    if label:
        print(f"\n[Memory: {label}]")
    else:
        print("\n[Memory Usage]")

    print(f"  Process RSS: {memory['rss_mb']:.1f} MB")
    print(f"  Process VMS: {memory['vms_mb']:.1f} MB")
    print(f"  Process %: {memory['percent']:.2f}%")
    print(f"  System Available: {memory['available_mb']:.1f} MB")
    print(f"  System Total: {memory['total_mb']:.1f} MB")


class MemoryTracker:
    """Context manager for tracking memory usage during operations."""

    def __init__(self, operation_name: str, log_interval: Optional[int] = None):
        """Initialize memory tracker.

        Args:
            operation_name: Name of operation being tracked
            log_interval: If set, log memory every N seconds during operation
        """
        self.operation_name = operation_name
        self.log_interval = log_interval
        self.start_memory = None
        self.end_memory = None

    def __enter__(self):
        """Start tracking memory."""
        gc.collect()
        self.start_memory = get_memory_usage()
        log_memory_usage(f"{self.operation_name} - START")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop tracking and report memory delta."""
        gc.collect()
        self.end_memory = get_memory_usage()

        delta_rss = self.end_memory['rss_mb'] - self.start_memory['rss_mb']
        delta_vms = self.end_memory['vms_mb'] - self.start_memory['vms_mb']

        log_memory_usage(f"{self.operation_name} - END")
        print(f"\n[Memory Delta: {self.operation_name}]")
        print(f"  RSS Change: {delta_rss:+.1f} MB")
        print(f"  VMS Change: {delta_vms:+.1f} MB")

        if delta_rss > 1000:  # Warning if using more than 1GB
            print(f"  ⚠️  WARNING: High memory increase detected!")


def check_memory_available(required_mb: int = 1000) -> bool:
    """Check if sufficient memory is available.

    Args:
        required_mb: Required memory in MB

    Returns:
        True if enough memory available
    """
    if not PSUTIL_AVAILABLE:
        return True  # Can't check, assume OK

    memory = get_memory_usage()
    available = memory['available_mb']

    if available < required_mb:
        print(f"\n⚠️  WARNING: Low memory!")
        print(f"  Required: {required_mb:.1f} MB")
        print(f"  Available: {available:.1f} MB")
        return False

    return True


if __name__ == "__main__":
    # Test memory monitoring
    print("Testing memory monitoring...")
    log_memory_usage("Initial")

    # Test with context manager
    with MemoryTracker("Test Operation"):
        # Allocate some memory
        import numpy as np
        data = np.zeros((1000, 1000), dtype=np.float32)
        print(f"  Allocated array: {data.nbytes / 1024 / 1024:.1f} MB")

    print("\n✓ Memory monitoring test complete")
