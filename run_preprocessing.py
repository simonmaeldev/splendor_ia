#!/usr/bin/env python3
"""Wrapper script to run data preprocessing from project root."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Now import and run the preprocessing
from imitation_learning.data_preprocessing import main

if __name__ == "__main__":
    main()
