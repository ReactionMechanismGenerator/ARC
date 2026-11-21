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

source "$(dirname "${BASH_SOURCE[0]}")/shell_init.sh"
arc_shell_init


ENV_FILE="devtools/xtb_environment.yml"

if [ ! -f "$ENV_FILE" ]; then
    echo "❌ File not found: $ENV_FILE"
    exit 1
fi

if $COMMAND_PKG env list | grep -q '^xtb_env\s'; then
    echo ">>> Updating existing xtb_env..."
    $COMMAND_PKG env update -n xtb_env -f "$ENV_FILE" --prune
else
    echo ">>> Creating new xtb_env..."
    $COMMAND_PKG env create -n xtb_env -f "$ENV_FILE" -y
fi

# Activate environment temporarily for the check
$COMMAND_PKG activate xtb_env

echo ">>> Checking xTB installation..."

if xtb --version &> /dev/null; then
    xtb --version
    echo "✔️ xTB is successfully installed."
else
    echo "❌ xTB is not found in PATH. Please check the environment."
    exit 1
fi

$COMMAND_PKG deactivate

echo "✅ Done installing xTB (xtb_env)."
