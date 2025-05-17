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
    . "$BASE/etc/profile.d/conda.sh"
fi

pushd ..
echo ">>> Cloning or updating AutoTST..."
if [ -d AutoTST ]; then
    cd AutoTST
    CURRENT_BRANCH=$(git symbolic-ref --short HEAD)
    if [ "$CURRENT_BRANCH" = "main" ]; then
        git fetch origin
        git pull origin main
    else
        echo "⚠️ AutoTST is on branch '$CURRENT_BRANCH'. Skipping update."
    fi
else
    git clone https://github.com/ReactionMechanismGenerator/AutoTST
    cd AutoTST
fi

AUTO_PATH_LINE="export PYTHONPATH=\$PYTHONPATH:$(pwd)"
if ! grep -Fxq "$AUTO_PATH_LINE" ~/.bashrc; then
    echo "$AUTO_PATH_LINE" >> ~/.bashrc
    echo "✔️ Added AutoTST path to ~/.bashrc"
else
    echo "ℹ️ AutoTST path already exists in ~/.bashrc"
fi

echo ">>> Creating or updating AutoTST environment (tst_env)..."
if [ "$COMMAND_PKG" = "micromamba" ]; then
    $COMMAND_PKG create -n tst_env -f environment.yml || true
else
    $COMMAND_PKG env create -n tst_env -f environment.yml --skip-if-exists || true
fi

echo ">>> Installing extra dependencies into tst_env..."
$COMMAND_PKG install -y -n tst_env -c conda-forge pyyaml

popd > /dev/null
echo "✅ Done installing AutoTST."
