#!/bin/bash

# Script to train a model and organize the output directory
# Usage: ./train_and_organize.sh <path_to_config.yaml>

set -e  # Exit on error

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <path_to_config.yaml>"
    exit 1
fi

CONFIG_PATH="$1"

# Check if config file exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config file not found: $CONFIG_PATH"
    exit 1
fi

# Extract config filename without extension
CONFIG_FILENAME=$(basename "$CONFIG_PATH" .yaml)
echo "Training with config: $CONFIG_FILENAME"

# Store the timestamp before training starts to match the created directory
TIMESTAMP=$(date +%Y%m%d%H%M)
echo "Starting training at: $TIMESTAMP"

# Run training
uv run python -m src.imitation_learning.train --config "$CONFIG_PATH"

# Find the most recently created directory in data/models/
# that matches the timestamp pattern
MODELS_DIR="data/models"
CREATED_DIR=$(ls -dt "$MODELS_DIR"/${TIMESTAMP}_config_* 2>/dev/null | head -1)

if [ -z "$CREATED_DIR" ]; then
    echo "Error: Could not find created model directory with timestamp $TIMESTAMP"
    echo "Trying to find most recent directory..."
    CREATED_DIR=$(ls -dt "$MODELS_DIR"/????????_config_* 2>/dev/null | head -1)

    if [ -z "$CREATED_DIR" ]; then
        echo "Error: Could not find any created model directory"
        exit 1
    fi
    echo "Found directory: $CREATED_DIR"
fi

# Copy the config file into the directory and rename it to config.yaml
echo "Copying config file to $CREATED_DIR/config.yaml"
mv "$CONFIG_PATH" "$CREATED_DIR/config.yaml"

echo "✓ Training complete and organized!"
echo "  Model directory: $CREATED_DIR"
echo "  Config saved as: $CREATED_DIR/config.yaml"
