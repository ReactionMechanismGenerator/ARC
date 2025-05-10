#!/bin/bash -l
set -e

INSTALL_MAIN=false
KINBOT_VERSION=2.0.6
KINBOT_TAR="v${KINBOT_VERSION}.tar.gz"
KINBOT_URL="https://github.com/zadorlab/KinBot/archive/refs/tags/${KINBOT_TAR}"

if [ "$1" == "--main" ]; then
    INSTALL_MAIN=true
    echo "📦 Installing the latest KinBot from 'main' branch"
fi

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
    micromamba activate base
else
    BASE=$(conda info --base)
    . "$BASE/etc/profile.d/conda.sh"
    conda activate base
fi

pushd .. > /dev/null

if $INSTALL_MAIN; then
    echo ">>> Cloning KinBot from main branch..."
    if [ -d KinBot ]; then
        cd KinBot
        git fetch origin
        git checkout main
        git pull origin main
    else
        git clone https://github.com/zadorlab/KinBot
        cd KinBot
    fi
else
    echo ">>> Downloading KinBot v${KINBOT_VERSION}..."
    wget -q --show-progress "$KINBOT_URL" -O "$KINBOT_TAR"
    echo ">>> Extracting $KINBOT_TAR..."
    tar -xzf "$KINBOT_TAR"
    rm "$KINBOT_TAR"
    cd "KinBot-${KINBOT_VERSION}"
fi

echo ">>> Activating arc_env temporarily..."
if [ "$COMMAND_PKG" = "micromamba" ]; then
    micromamba activate arc_env
else
    conda activate arc_env
fi

echo ">>> Installing KinBot..."
python setup.py build
python setup.py install

echo ">>> Deactivating arc_env..."
if [ "$COMMAND_PKG" = "micromamba" ]; then
    micromamba deactivate
else
    conda deactivate
fi

KINBOT_ABS_PATH=$(pwd)
export PYTHONPATH="$PYTHONPATH:$KINBOT_ABS_PATH"
KINBOT_LINE="export PYTHONPATH=\$PYTHONPATH:$KINBOT_ABS_PATH"
if ! grep -Fxq "$KINBOT_LINE" ~/.bashrc; then
    echo "$KINBOT_LINE" >> ~/.bashrc
    echo "✔️ Added KinBot path to ~/.bashrc"
else
    echo "ℹ️ KinBot path already present in ~/.bashrc"
fi
echo "PYTHONPATH=$PYTHONPATH"

popd > /dev/null

echo "✅ Done installing KinBot."
