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

echo ">>> Checking CREST installation..."

if [ "$COMMAND_PKG" = "micromamba" ]; then
    CREST_RUNNER="micromamba run -n crest_env"
    CREST_LISTER="micromamba list -n crest_env"
else
    CREST_RUNNER="conda run -n crest_env"
    CREST_LISTER="conda list -n crest_env"
fi

if $CREST_RUNNER crest --version &> /dev/null; then
    version_output=$($CREST_RUNNER crest --version 2>&1)
    echo "$version_output"
    installed_version=$(printf '%s' "$version_output" | tr '\n' ' ' | sed -n 's/.*Version[[:space:]]\+\([0-9.][0-9.]*\).*/\1/p')
    if [ "$installed_version" != "2.12" ]; then
        echo "❌ CREST version mismatch (expected 2.12)."
        exit 1
    fi
    echo "✔️ CREST 2.12 is successfully installed."
else
    echo "❌ CREST is not found in PATH. Please check the environment."
    exit 1
fi

echo "✅ Done installing CREST (crest_env)."
