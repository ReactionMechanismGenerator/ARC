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
elif [ "$COMMAND_PKG" = "mamba" ] || [ "$COMMAND_PKG" = "conda" ]; then
    BASE=$(conda info --base)
    source "$BASE/etc/profile.d/conda.sh"
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

if $COMMAND_PKG env list | grep -q '^tst_env\s'; then
    echo ">>> Updating existing environment tst_env..."
    if [ "$COMMAND_PKG" != "conda" ]; then
        $COMMAND_PKG env update -n tst_env -f environment.yml --prune -y
    else
        $COMMAND_PKG env update -n tst_env -f environment.yml --prune
    fi
else
    echo ">>> Creating new environment tst_env..."
    if [ "$COMMAND_PKG" != "conda" ]; then
        $COMMAND_PKG env create -n tst_env -f environment.yml -y
    else
        $COMMAND_PKG env create -n tst_env -f environment.yml
    fi
fi

echo ">>> Installing extra dependencies into tst_env..."
if [ "$COMMAND_PKG" != "conda" ]; then
    $COMMAND_PKG install -n tst_env -c conda-forge pyyaml -y
else
    $COMMAND_PKG install -n tst_env -c conda-forge pyyaml
fi

popd > /dev/null
echo "✅ Done installing AutoTST."
