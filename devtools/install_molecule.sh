#!/usr/bin/env bash
set -eo pipefail

ARC_ROOT=$(pwd)
DEFAULT_PARENT=$(dirname "$ARC_ROOT")
RDL_PARENT_DIR=${RDL_PARENT_DIR:-$DEFAULT_PARENT}
export RDL_REPO_PATH="$RDL_PARENT_DIR/RingDecomposerLib"

cleanup() {
    if [[ -n "${COMMAND_PKG:-}" ]] && command -v conda &>/dev/null && conda info --envs | grep -q "^arc_env "; then
        if [[ "$COMMAND_PKG" = "micromamba" ]]; then
            micromamba deactivate || true
        else
            conda deactivate || true
        fi
    fi
}
trap cleanup EXIT

if command -v micromamba &>/dev/null; then
    COMMAND_PKG=micromamba
elif command -v mamba &>/dev/null; then
    COMMAND_PKG=mamba
elif command -v conda &>/dev/null; then
    COMMAND_PKG=conda
else
    echo "No conda-compatible package manager found."
    exit 1
fi

if [[ "$COMMAND_PKG" = "micromamba" ]]; then
    eval "$(micromamba shell hook --shell=bash)"
    micromamba activate arc_env
else
    source "$($COMMAND_PKG info --base)/etc/profile.d/conda.sh"
    conda activate arc_env
fi

# RingDecomposerLib
cd "$RDL_PARENT_DIR"
if [[ -d RingDecomposerLib ]]; then
    cd RingDecomposerLib
    CURRENT_BRANCH=$(git symbolic-ref --short HEAD)
    if [ "$CURRENT_BRANCH" = "main" ]; then
        git fetch origin main
        git pull origin main
    else
        echo "⚠️ RingDecomposerLib is on branch '$CURRENT_BRANCH'. Skipping update."
    fi
else
    git clone https://github.com/DanaResearchGroup/RingDecomposerLib
    cd RingDecomposerLib
fi

mkdir -p build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build .

cd ../src/python
$COMMAND_PKG run -n arc_env python -m pip install --no-build-isolation .

cd "$ARC_ROOT"
$COMMAND_PKG run -n arc_env python -c "import py_rdl.wrapper.DataInternal"

# molecule
cd "$ARC_ROOT/.."
if [[ -d molecule ]]; then
    cd molecule
    CURRENT_BRANCH=$(git symbolic-ref --short HEAD)
    if [ "$CURRENT_BRANCH" = "main" ]; then
        git fetch origin
        git pull origin main
    else
        echo "⚠️ molecule is on branch '$CURRENT_BRANCH'. Skipping update."
    fi
else
    git clone https://github.com/ReactionMechanismGenerator/molecule
    cd molecule
fi

MOLECULE_PATH=$(pwd)
if [[ -f Makefile ]]; then
    $COMMAND_PKG run -n arc_env make
else
    echo "Makefile not found in molecule; aborting."
    exit 1
fi

LINE="export PYTHONPATH=\${PYTHONPATH:-}:$MOLECULE_PATH"
if ! grep -Fxq "$LINE" ~/.bashrc; then
    echo "$LINE" >> ~/.bashrc
fi
export PYTHONPATH="${PYTHONPATH:-}:$MOLECULE_PATH"

$COMMAND_PKG run -n arc_env python -c "import molecule"

echo "Installation of molecule and PyRDL completed successfully."
