#!/usr/bin/env bash
set -eo pipefail

ENV_NAME="arc_env"
ENV_FILE="environment.yml"

# Initialize conda for bash
source "$(conda info --base)/etc/profile.d/conda.sh"

# Check if the environment exists
if conda env list | grep -qw "$ENV_NAME"; then
    echo "✔️ Environment '$ENV_NAME' exists. Updating from $ENV_FILE..."
    conda env update -n "$ENV_NAME" -f "$ENV_FILE"
else
    echo "📦 Creating environment '$ENV_NAME' from $ENV_FILE..."
    conda env create -n "$ENV_NAME" -f "$ENV_FILE"
fi

echo "✅ ARC environment setup complete."
