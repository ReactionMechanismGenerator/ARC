#!/bin/bash -l
set -eo pipefail

echo ">>> Starting TS-GCN installation and environment setup..."

if command -v mamba &> /dev/null; then
    COMMAND_PKG=mamba
elif command -v conda &> /dev/null; then
    COMMAND_PKG=conda
else
    echo "❌ Mamba or Conda is required. Micromamba is not supported for TS-GCN."
    exit 1
fi
echo "✔️ Using package manager: $COMMAND_PKG"

ENV_FILE="$(pwd)/devtools/gcn_environment.yml"

TS_GCN_DIR="../TS-GCN"
echo ">>> Cloning or updating TS-GCN in $TS_GCN_DIR..."
pushd .. > /dev/null

if [ -d TS-GCN ]; then
    echo "✔️ TS-GCN already exists. Updating..."
    cd TS-GCN
    git fetch origin
    git checkout main
    git pull origin main
else
    echo "📦 Cloning TS-GCN..."
    git clone https://github.com/ReactionMechanismGenerator/TS-GCN
    cd TS-GCN
fi

TS_GCN_CLONED_PATH=$(pwd)
popd > /dev/null

echo "✅ TS-GCN repository is ready."
echo "🔎 Using ARC environment file for TS-GCN: $ENV_FILE"

if [ ! -f "$ENV_FILE" ]; then
    echo "❌ Environment file not found: $ENV_FILE"
    exit 1
fi
echo "✔️ Environment file exists."

echo ">>> Creating or updating conda environment 'ts_gcn'..."
source "$($COMMAND_PKG info --base)/etc/profile.d/conda.sh"

$COMMAND_PKG env create -n ts_gcn -f "$ENV_FILE" -y || \
$COMMAND_PKG env update -n ts_gcn -f "$ENV_FILE" -y

echo ">>> Adding TS-GCN to PYTHONPATH..."
LINE="export PYTHONPATH=\$PYTHONPATH:$TS_GCN_CLONED_PATH"
if ! grep -Fxq "$LINE" ~/.bashrc; then
    echo "$LINE" >> ~/.bashrc
    echo "✔️ Added TS-GCN to ~/.bashrc"
else
    echo "ℹ️ TS-GCN already present in ~/.bashrc"
fi
export PYTHONPATH="$PYTHONPATH:$TS_GCN_CLONED_PATH"

echo "✅ TS-GCN installation and environment setup complete."
