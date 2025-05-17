#!/usr/bin/env bash
set -euo pipefail

echo ">>> Installing molecule and PyRDL..."

ARC_ROOT=$(pwd)
DEFAULT_PARENT=$(dirname "$ARC_ROOT")

cleanup() {
    echo "--- Running cleanup trap ---"
    if [[ -n "${COMMAND_PKG:-}" ]] && command -v conda &>/dev/null && conda info --envs | grep -q "^arc_env "; then
        echo "🔌 Deactivating arc_env during cleanup..."
        if [[ "$COMMAND_PKG" = "micromamba" ]]; then
            micromamba deactivate || true
        else
            conda deactivate || true
        fi
    fi
    echo "--- Cleanup complete ---"
}
trap cleanup EXIT

if [[ -n "${DEBUG_INSTALL:-}" ]]; then
    echo "--- Debug mode enabled ---"
    set -x
fi

# === Detect conda-compatible package manager ===
if command -v micromamba &>/dev/null; then
    echo "✔️ Using package manager: micromamba"
    COMMAND_PKG=micromamba
elif command -v mamba &>/dev/null; then
    echo "✔️ Using package manager: mamba"
    COMMAND_PKG=mamba
elif command -v conda &>/dev/null; then
    echo "✔️ Using package manager: conda"
    COMMAND_PKG=conda
else
    echo "❌ No conda-compatible package manager found."
    exit 1
fi

# === Shell activation ===
if [[ "$COMMAND_PKG" = "micromamba" ]]; then
    eval "$(micromamba shell hook --shell=bash)"
else
    BASE=$($COMMAND_PKG info --base)
    source "$BASE/etc/profile.d/conda.sh"
fi

# === Activate arc_env ===
echo "✔️ Activating arc_env"
$COMMAND_PKG activate arc_env

# === 1. Clone and build RingDecomposerLib ===
echo "📦 Cloning/updating RingDecomposerLib (branch setup3)..."
RDL_PARENT_DIR=${RDL_PARENT_DIR:-$DEFAULT_PARENT}
cd "$RDL_PARENT_DIR"

if [[ -d RingDecomposerLib ]]; then
    cd RingDecomposerLib
    git fetch origin setup3
    git checkout setup3
    git pull origin setup3
else
    git clone -b setup3 https://github.com/DanaResearchGroup/RingDecomposerLib
    cd RingDecomposerLib
fi

# === Build RingDecomposerLib ===
echo "🛠 Building RingDecomposerLib..."
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build .

# === Install the Python wrapper into arc_env ===
echo "📦 Installing py_rdl into arc_env via pip…"
# move to the Python wrapper directory
cd ../src/python
# use --no-build-isolation so we pick up the static lib from build/
$COMMAND_PKG run -n arc_env pip install --no-build-isolation .

# === Quick smoke-test of the installed extension ===
echo "🔬 Verifying py_rdl.wrapper.DataInternal import…"
# return to ARC root so that $ARC_ROOT is unchanged
cd "$ARC_ROOT"
$COMMAND_PKG run -n arc_env python -c "import py_rdl.wrapper.DataInternal; print('✅ py_rdl.wrapper.DataInternal import successful')"

# Extra check: ensure the .so is in site-packages
echo "🔍 Checking for DataInternal.so in site-packages…"
$COMMAND_PKG run -n arc_env python - <<'PYCODE'
import glob, os, py_rdl.wrapper
search = os.path.join(os.path.dirname(py_rdl.wrapper.__file__), "DataInternal*.so")
matches = glob.glob(search)
assert matches, f"❌ DataInternal.so not found; looked for: {search}"
print("✅ DataInternal.so present at", matches[0])
PYCODE

# === 2. Clone and build molecule ===
echo "📦 Cloning/updating molecule…"
cd "$ARC_ROOT/.."
if [[ -d molecule ]]; then
    cd molecule
    git fetch origin
    git checkout main
    git pull origin main
else
    git clone https://github.com/ReactionMechanismGenerator/molecule
    cd molecule
fi

MOLECULE_PATH=$(pwd)

echo "🧪 Cythonizing molecule…"
if [[ -f Makefile ]]; then
    $COMMAND_PKG run -n arc_env make
else
    echo "❌ Makefile not found in molecule; aborting."
    exit 1
fi

echo "🌱 Adding molecule to PYTHONPATH…"
LINE="export PYTHONPATH=\${PYTHONPATH:-}:$MOLECULE_PATH"
if ! grep -Fxq "$LINE" ~/.bashrc; then
    echo "$LINE" >> ~/.bashrc
    echo "✔️ Added molecule to ~/.bashrc"
else
    echo "ℹ️ molecule already present in ~/.bashrc"
fi
export PYTHONPATH="${PYTHONPATH:-}:$MOLECULE_PATH"

echo "🔬 Verifying molecule import…"
$COMMAND_PKG run -n arc_env python -c "import molecule; print('✅ molecule import successful')"

echo "✅ Done installing molecule and PyRDL."
