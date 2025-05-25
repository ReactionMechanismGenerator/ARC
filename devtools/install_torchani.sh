#!/bin/bash -l
set -eo pipefail

# Detect package manager
if command -v micromamba &> /dev/null; then
    echo "✔️ Micromamba is installed."
    COMMAND_PKG=micromamba
elif command -v mamba &> /dev/null; then
    echo "✔️ Mamba is installed."
    COMMAND_PKG=mamba
elif command -v conda &> /dev/null; then
    echo "✔️ Conda is installed."
    COMMAND_PKG=conda
else
    echo "❌ Micromamba, Mamba, or Conda is required. Please install one."
    exit 1
fi

# Initialize shell hook
if [ "$COMMAND_PKG" = "micromamba" ]; then
    eval "$(micromamba shell hook --shell=bash)"
else
    BASE=$($COMMAND_PKG info --base)
    source "$BASE/etc/profile.d/conda.sh"
fi

ENV_FILE="devtools/tani_environment.yml"
ENV_NAME="tani_env"

# Verify environment file exists
if [ ! -f "$ENV_FILE" ]; then
    echo "❌ Environment file not found: $ENV_FILE"
    exit 1
fi

# Check if environment exists
if $COMMAND_PKG env list | grep -q "^$ENV_NAME\s"; then
    echo ">>> Updating existing $ENV_NAME..."
    $COMMAND_PKG env update -n "$ENV_NAME" -f "$ENV_FILE" --prune
else
    echo ">>> Creating new $ENV_NAME..."
    $COMMAND_PKG env create -n "$ENV_NAME" -f "$ENV_FILE"
fi

# Temporarily activate environment
if [ "$COMMAND_PKG" = "micromamba" ]; then
    micromamba activate "$ENV_NAME"
else
    conda activate "$ENV_NAME"
fi

# Validate TorchANI installation
if python -c 'import torchani; print("TorchANI version:", torchani.__version__)'; then
    echo "✔️ TorchANI is installed and importable."
else
    echo "❌ TorchANI is not importable. Please check the environment setup."
    exit 1
fi

# Deactivate environment
if [ "$COMMAND_PKG" = "micromamba" ]; then
    micromamba deactivate
else
    conda deactivate
fi

echo "✅ Done installing TorchANI ($ENV_NAME)."
