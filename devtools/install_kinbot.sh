#!/bin/bash -l
set -eo pipefail
ENV_NAME=arc_env
KINBOT_VERSION=2.1.1

echo "📦 Installing Kinbot..."

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
    eval "$($COMMAND_PKG shell hook --shell=bash)"
fi


# Activate arc_env

if ! $COMMAND_PKG env list | awk '{print $1}' | sed 's/^\*//' | grep -Fxq "$ENV_NAME"; then
    echo "❌ Environment '$ENV_NAME' not found. Please create it first."
    exit 1
fi

echo ">>> Installing KinBot version $KINBOT_VERSION in environment '$ENV_NAME'..."
$COMMAND_PKG install -n "$ENV_NAME" -c conda-forge kinbot=$KINBOT_VERSION -y

echo "✅ Done installing KinBot."
