#!/usr/bin/env bash
set -eo pipefail
set -x                                   # echo commands as they run

# Enable debug symbols and disable optimizations in any C/C++ extension
export CFLAGS="-g -O0"
export CXXFLAGS="$CFLAGS"
export CYTHON_TRACE=1                    # generate Cython tracing hooks

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

# Choose conda-compatible package manager
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

# Activate arc_env
if [[ "$COMMAND_PKG" = "micromamba" ]]; then
    eval "$(micromamba shell hook --shell=bash)"
    micromamba activate arc_env
else
    source "$($COMMAND_PKG info --base)/etc/profile.d/conda.sh"
    conda activate arc_env
fi

###############################################################################
# RingDecomposerLib
cd "$RDL_PARENT_DIR"
if [[ -d RingDecomposerLib ]]; then
    cd RingDecomposerLib
    CURRENT_BRANCH=$(git symbolic-ref --short HEAD)
    if [[ "$CURRENT_BRANCH" = "main" ]]; then
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
cmake --build . 2>&1 | tee "$ARC_ROOT/rdl_cmake_build.log"

cd ../src/python
$COMMAND_PKG run -n arc_env python -m pip install --no-build-isolation . -vvv \
    2>&1 | tee "$ARC_ROOT/rdl_python_install.log"

cd "$ARC_ROOT"
$COMMAND_PKG run -n arc_env python -c "import py_rdl.wrapper.DataInternal"

###############################################################################
# molecule
cd "$ARC_ROOT/.."
if [[ -d molecule ]]; then
    cd molecule
    CURRENT_BRANCH=$(git symbolic-ref --short HEAD)
    if [[ "$CURRENT_BRANCH" = "main" ]]; then
        git fetch origin
        git pull origin main
    else
        echo "⚠️ molecule is on branch '$CURRENT_BRANCH'. Skipping update."
    fi
else
    git clone https://github.com/ReactionMechanismGenerator/molecule
    cd molecule
fi

# (Temporary) switch to the May_25 branch
git fetch origin
git checkout May_25 || {
    echo "⚠️ Failed to switch to May_25 branch. Ensure it exists."
    exit 1
}

MOLECULE_PATH=$(pwd)
if [[ -f Makefile ]]; then
    # run make single-threaded and verbose, capture output
    $COMMAND_PKG run -n arc_env make VERBOSE=1 -j1 \
        2>&1 | tee "$ARC_ROOT/molecule_make.log"
else
    echo "Makefile not found in molecule; aborting."
    exit 1
fi

# Persist PYTHONPATH
LINE="export PYTHONPATH=\${PYTHONPATH:-}:$MOLECULE_PATH"
if ! grep -Fxq "$LINE" ~/.bashrc; then
    echo "$LINE" >> ~/.bashrc
fi
export PYTHONPATH="${PYTHONPATH:-}:$MOLECULE_PATH"

# Smoke-test the import
$COMMAND_PKG run -n arc_env python -c "import molecule" \
    2>&1 | tee "$ARC_ROOT/molecule_import_test.log"

echo "✅ Installation of molecule and PyRDL completed successfully."
