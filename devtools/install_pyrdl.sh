#!/usr/bin/env bash
set -eo pipefail
set -x  # Echo commands for debug

LOGDIR="$HOME/molecule_build_logs"
mkdir -p "$LOGDIR"

# Save working directory and setup RDL path
ARC_ROOT=$(pwd)
DEFAULT_PARENT=$(dirname "$ARC_ROOT")
RDL_PARENT_DIR=${RDL_PARENT_DIR:-$DEFAULT_PARENT}
export RDL_REPO_PATH="$RDL_PARENT_DIR/RingDecomposerLib"

# Detect environment tool: micromamba, mamba, or conda
if command -v micromamba &> /dev/null; then
    echo "✔️ Micromamba is installed."
    COMMAND_PKG=micromamba
    eval "$(micromamba shell hook -s bash)"
    micromamba activate arc_env
elif command -v mamba &> /dev/null; then
    echo "✔️ Mamba is installed."
    COMMAND_PKG=mamba
    source "$(mamba info --base)/etc/profile.d/conda.sh"
    conda activate arc_env
elif command -v conda &> /dev/null; then
    echo "✔️ Conda is installed."
    COMMAND_PKG=conda
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate arc_env
else
    echo "❌ Micromamba, Mamba, or Conda is required. Please install one."
    exit 1
fi

# Clone and build RingDecomposerLib
cd "$RDL_PARENT_DIR"
if [[ -d RingDecomposerLib ]]; then
    cd RingDecomposerLib
    git fetch origin main || true
    git checkout main || true
    git pull origin main || true
else
    git clone https://github.com/DanaResearchGroup/RingDecomposerLib
    cd RingDecomposerLib
fi

mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release |& tee "$LOGDIR/rdl_cmake.log"
cmake --build . --verbose |& tee "$LOGDIR/rdl_build.log"

# Install the Python wrapper
cd ../src/python
python -m pip install --no-build-isolation --verbose . |& tee "$LOGDIR/rdl_python_install.log"

# Verify install
cd "$ARC_ROOT"
python -c "import py_rdl.wrapper.DataInternal" |& tee "$LOGDIR/rdl_import_test.log"

echo "✅ PyRDL installation complete."
