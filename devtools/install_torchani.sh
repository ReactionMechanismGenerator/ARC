#!/bin/bash -l
set -eo pipefail

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

if [ "$COMMAND_PKG" = "micromamba" ]; then
    eval "$(micromamba shell hook --shell=bash)"
else
    BASE=$($COMMAND_PKG info --base)
    . "$BASE/etc/profile.d/conda.sh"
fi

ENV_FILE="devtools/tani_environment.yml"

if [ ! -f "$ENV_FILE" ]; then
    echo "❌ Environment file not found: $ENV_FILE"
    exit 1
fi

if $COMMAND_PKG env list | grep -q '^tani_env\s'; then
    echo ">>> Updating existing tani_env..."
    $COMMAND_PKG env update -n tani_env -f "$ENV_FILE" --prune -y
else
    echo ">>> Creating new tani_env..."
    $COMMAND_PKG env create -n tani_env -f "$ENV_FILE" -y
fi

# Activate environment temporarily for the check
if [ "$COMMAND_PKG" = "micromamba" ]; then
    micromamba activate tani_env
else
    conda activate tani_env
fi

if python -c 'import torchani'; then
    echo "✔️ TorchANI is installed and importable."
else
    echo "❌ TorchANI is not importable. Please check the environment setup."
    exit 1
fi

if [ "$COMMAND_PKG" = "micromamba" ]; then
    micromamba deactivate
else
    conda deactivate
fi

echo "✅ Done installing TorchANI (tani_env)."
