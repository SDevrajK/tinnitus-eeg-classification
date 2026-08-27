#!/bin/bash

# Script to create Conda environment from environment.yml and install eeg-processor package in editable mode
# Usage: bash scripts/setup_environment.sh

set -euo pipefail

ENV_NAME="eeg-processor"
ENV_YAML="$(dirname "$0")/../environment.yml"
EEG_PROCESSOR_PATH="/home/sdevrajk/projects/personal/eeg-processor"
CONDA_BIN="conda"

# Get the base conda directory
CONDA_BASE="$(conda info --base)"

# Check if environment already exists
if [ -d "$CONDA_BASE/envs/$ENV_NAME" ]; then
    echo "Environment '$ENV_NAME' already exists; skipping creation."
else
    conda env create -f "$ENV_YAML"
fi

# Install the local eeg-processor package in editable mode using the environment's pip
"$CONDA_BASE/envs/$ENV_NAME/bin/pip" install -e "$EEG_PROCESSOR_PATH"

echo "Setup complete. Activate with: conda activate $ENV_NAME"