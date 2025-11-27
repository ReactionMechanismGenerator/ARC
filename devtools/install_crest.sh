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
    BASE=$(conda info --base)
    . "$BASE/etc/profile.d/conda.sh"
fi

ENV_FILE="devtools/crest_environment.yml"

if [ ! -f "$ENV_FILE" ]; then
    echo "❌ File not found: $ENV_FILE"
    exit 1
fi

if $COMMAND_PKG env list | grep -q '^crest_env\s'; then
    echo ">>> Updating existing crest_env..."
    $COMMAND_PKG env update -n crest_env -f "$ENV_FILE" --prune
else
    echo ">>> Creating new crest_env..."
    $COMMAND_PKG env create -n crest_env -f "$ENV_FILE" -y
fi

if [ "$COMMAND_PKG" = "micromamba" ]; then
    micromamba activate crest_env
else
    conda activate crest_env
fi

echo ">>> Checking CREST installation..."

if crest --version &> /dev/null; then
    version_output=$(crest --version 2>&1 | head -n 1)
    echo "$version_output"
    if ! grep -q "2\\.12" <<< "$version_output"; then
        echo "❌ CREST version mismatch (expected 2.12)."
        exit 1
    fi
    echo "✔️ CREST 2.12 is successfully installed."
else
    echo "❌ CREST is not found in PATH. Please check the environment."
    exit 1
fi

$COMMAND_PKG deactivate

echo "✅ Done installing CREST (crest_env)."
