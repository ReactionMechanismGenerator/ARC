#!/usr/bin/env bash
set -eo pipefail
set -x  # Echo commands

LOGDIR="$HOME/molecule_build_logs"
mkdir -p "$LOGDIR"

# Save working dir
ARC_ROOT=$(pwd)
DEFAULT_PARENT=$(dirname "$ARC_ROOT")
RDL_PARENT_DIR=${RDL_PARENT_DIR:-$DEFAULT_PARENT}
export RDL_REPO_PATH="$RDL_PARENT_DIR/RingDecomposerLib"

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate arc_env

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

cd "$ARC_ROOT"
python -c "import py_rdl.wrapper.DataInternal" |& tee "$LOGDIR/rdl_import_test.log"

echo "✅ PyRDL installation complete."
