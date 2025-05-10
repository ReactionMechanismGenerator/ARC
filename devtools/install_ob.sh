#!/bin/bash -l
set -eo pipefail

COMMAND_PKG=""

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
elif [ "$COMMAND_PKG" = "mamba" ] || [ "$COMMAND_PKG" = "conda" ]; then
    BASE=$(conda info --base)
    source "$BASE/etc/profile.d/conda.sh"
fi

ENV_FILE="devtools/ob_environment.yml"

if [ ! -f "$ENV_FILE" ]; then
    echo "❌ $ENV_FILE not found!"
    exit 1
fi

ENV_EXISTS=$($COMMAND_PKG env list | grep -q '^ob_env\s' && echo "true" || echo "false")

if [ "$ENV_EXISTS" = "true" ]; then
    echo ">>> Updating existing environment ob_env..."
    $COMMAND_PKG env update -n ob_env -f "$ENV_FILE" --prune -y
else
    echo ">>> Creating new environment ob_env..."
    $COMMAND_PKG env create -n ob_env -f "$ENV_FILE" -y
fi

echo "✅ OpenBabel environment (ob_env) is ready."
