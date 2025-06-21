#!/bin/bash -l
set -e

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
    . "$BASE/etc/profile.d/conda.sh"       # shellcheck source=/dev/null
fi

pushd ..

echo ">>> Cloning or updating TS-GCN..."
if [ -d TS-GCN ]; then
    cd TS-GCN
    git fetch origin
    git checkout main
    git pull origin main
else
    git clone https://github.com/ReactionMechanismGenerator/TS-GCN
    cd TS-GCN
fi

export PYTHONPATH="$PYTHONPATH:$(pwd)"

GCN_LINE="export PYTHONPATH=\$PYTHONPATH:$(pwd)"
if [ -d TS-GCN ]; then
    cd TS-GCN
    CURRENT_BRANCH=$(git symbolic-ref --short HEAD)
    if [ "$CURRENT_BRANCH" = "main" ]; then
        git fetch origin
        git pull origin main
    else
        echo "⚠️ TS-GCN is on branch '$CURRENT_BRANCH'. Skipping update."
    fi
else
    echo "ℹ️ GCN path already present in ~/.bashrc"
fi
echo "PYTHONPATH=$PYTHONPATH"

if grep -q '^conda_env:' Makefile; then
    echo ">>> Creating GCN conda environment via Makefile"
    make conda_env
else
    echo "❌ Makefile target 'conda_env' not found. Please check TS-GCN repo."
    exit 1
fi

popd > /dev/null
echo "✅ Done installing GCN."
