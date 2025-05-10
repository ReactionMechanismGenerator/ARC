#!/bin/bash -l
set -e

echo ">>> Checking available package manager..."

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
    BASE=$(conda info --base)
    . "$BASE/etc/profile.d/conda.sh"
fi

echo ">>> Creating ob_env from devtools/ob_environment.yml..."
if [ ! -f devtools/ob_environment.yml ]; then
    echo "❌ devtools/ob_environment.yml not found!"
    exit 1
fi

$COMMAND_PKG env create -n ob_env -f devtools/ob_environment.yml || true

echo "✅ OpenBabel environment (ob_env) ready."
