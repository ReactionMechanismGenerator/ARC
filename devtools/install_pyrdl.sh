#!/usr/bin/env bash
set -eo pipefail


# Create log directory
LOGDIR="$HOME/molecule_build_logs"
mkdir -p "$LOGDIR"

# Save working directory and setup RDL path
# Determine where this script lives
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# Try to find the ARC repo root via git, otherwise use parent dir
if ARC_ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null); then
  :
else
  ARC_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi

# Compute clone root (one level above ARC_ROOT) and export RDL path
CLONE_ROOT="$(dirname "$ARC_ROOT")"
RDL_PARENT_DIR="${RDL_PARENT_DIR:-$CLONE_ROOT}"
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
    BASE=$(conda info --base)
    source "$BASE/etc/profile.d/conda.sh"
    eval "$($COMMAND_PKG shell hook --shell bash)"
    conda activate arc_env
elif command -v conda &> /dev/null; then
    echo "✔️ Conda is installed."
    COMMAND_PKG=conda
    BASE=$(conda info --base)
    source "$BASE/etc/profile.d/conda.sh"
    eval "$($COMMAND_PKG shell hook --shell bash)"
    conda activate arc_env
else
    echo "❌ Micromamba, Mamba, or Conda is required. Please install one."
    exit 1
fi

# Ensure CMake is installed in the environment
if ! command -v cmake &> /dev/null; then
    echo "Installing CMake..."
    "$COMMAND_PKG" install -y cmake
fi

# Clone and build RingDecomposerLib
cd "$RDL_PARENT_DIR"
if [[ -d RingDecomposerLib ]]; then
    echo "Updating existing RingDecomposerLib repository..."
    cd RingDecomposerLib
    git fetch origin main || true
    git checkout main || true
    git pull origin main || true
else
    echo "Cloning RingDecomposerLib repository..."
    git clone https://github.com/DanaResearchGroup/RingDecomposerLib
    cd RingDecomposerLib
fi

# Configure and build C library
mkdir -p build
cd build
echo "Configuring RingDecomposerLib..."
cmake .. -DCMAKE_BUILD_TYPE=Release |& tee "$LOGDIR/rdl_cmake.log"
echo "Building RingDecomposerLib..."
cmake --build . --verbose |& tee "$LOGDIR/rdl_build.log"

# Install Python wrapper with include/library paths
cd ../src/python
echo "Installing PyRDL wrapper..."
export C_INCLUDE_PATH="../..:$RDL_REPO_PATH/include:$C_INCLUDE_PATH"
export LIBRARY_PATH="$RDL_REPO_PATH/build:$LIBRARY_PATH"
python -m pip install --no-build-isolation --verbose . |& tee "$LOGDIR/rdl_python_install.log"

# Verify install
cd "$ARC_ROOT"
echo "Verifying installation..."
if python -c "import py_rdl.wrapper.DataInternal" |& tee "$LOGDIR/rdl_import_test.log"; then
    echo "✅ PyRDL installation complete."
else
    echo "❌ PyRDL installation verification failed!"
    exit 1
fi
