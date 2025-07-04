#!/bin/bash -l
set -eo pipefail

INSTALL_MAIN=false
KINBOT_VERSION=2.0.6
KINBOT_TAR="v${KINBOT_VERSION}.tar.gz"
KINBOT_URL="https://github.com/zadorlab/KinBot/archive/refs/tags/${KINBOT_TAR}"

if [ "$1" == "--main" ]; then
    INSTALL_MAIN=true
    echo "📦 Installing the latest KinBot from 'main' branch"
fi

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
    if [ -d KinBot ]; then
        cd KinBot
        CURRENT_BRANCH=$(git symbolic-ref --short HEAD)
        if [ "$CURRENT_BRANCH" = "main" ]; then
            git fetch origin
            git pull origin main
        else
            echo "⚠️ KinBot is on branch '$CURRENT_BRANCH'. Skipping update."
        fi
    else
        git clone https://github.com/zadorlab/KinBot
        cd KinBot
    fi
else
    wget -q --show-progress "$KINBOT_URL" -O "$KINBOT_TAR"
    tar -xzf "$KINBOT_TAR"
    rm "$KINBOT_TAR"
    cd "KinBot-${KINBOT_VERSION}"
fi

if [ "$COMMAND_PKG" = "micromamba" ]; then
    micromamba activate arc_env
else
    conda activate arc_env
fi

echo ">>> Installing KinBot..."
python setup.py build
python setup.py install

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
