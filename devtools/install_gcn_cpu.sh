#!/bin/bash -l
set -e

echo ">>> Starting TS-GCN installation and environment setup..."

# --- Detect package manager ---
echo ">>> Checking available package manager..."
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

# --- Clone or update TS-GCN ---
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

# --- Validate environment file ---
if [ ! -f "$ENV_FILE" ]; then
    echo "❌ Environment file not found: $ENV_FILE"
    exit 1
fi
echo "✔️ Environment file exists."

# --- Create or update the ts_gcn environment ---
echo ">>> Creating or updating conda environment 'ts_gcn'..."
source "$($COMMAND_PKG info --base)/etc/profile.d/conda.sh"

$COMMAND_PKG env create -n ts_gcn -f "$ENV_FILE" -y || \
$COMMAND_PKG env update -n ts_gcn -f "$ENV_FILE" -y
echo "✅ Environment 'ts_gcn' created or updated."

# --- Reminder to activate manually if needed ---
echo "ℹ️ To use ts_gcn, activate it manually:"
echo "   conda activate ts_gcn"

# --- Add TS-GCN to PYTHONPATH ---
echo ">>> Adding TS-GCN to PYTHONPATH..."
LINE="export PYTHONPATH=\$PYTHONPATH:$TS_GCN_CLONED_PATH"
if ! grep -Fxq "$LINE" ~/.bashrc; then
    echo "$LINE" >> ~/.bashrc
    echo "✔️ Added TS-GCN to ~/.bashrc"
else
    echo "ℹ️ TS-GCN already present in ~/.bashrc"
fi
export PYTHONPATH="$PYTHONPATH:$TS_GCN_CLONED_PATH"
echo "PYTHONPATH set in current session: $PYTHONPATH"

echo "✅ TS-GCN installation and environment setup complete."
