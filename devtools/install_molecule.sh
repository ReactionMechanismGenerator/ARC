#!/usr/bin/env bash
set -eo pipefail
set -x  # echo commands

# where to dump logs
LOGDIR="$HOME/molecule_build_logs"
mkdir -p "$LOGDIR"

# remember where we started
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

# 1) Pick and activate the conda front‐end
if command -v micromamba &>/dev/null; then
    COMMAND_PKG=micromamba
elif command -v mamba &>/dev/null; then
    COMMAND_PKG=mamba
elif command -v conda &>/dev/null; then
    COMMAND_PKG=conda
else
    echo "❌ No conda/micromamba/mamba found."
    exit 1
fi

if [[ "$COMMAND_PKG" = "micromamba" ]]; then
    eval "$(micromamba shell hook --shell=bash)"
    micromamba activate arc_env
else
    source "$($COMMAND_PKG info --base)/etc/profile.d/conda.sh"
    conda activate arc_env
fi

# ----------------------------------------
# 2) Build RingDecomposerLib (unchanged)
# ----------------------------------------
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
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    |& tee "$LOGDIR/rdl_cmake.log"
cmake --build . --verbose \
    |& tee "$LOGDIR/rdl_build.log"

cd ../src/python
$COMMAND_PKG run -n arc_env python -m pip install --no-build-isolation --verbose . \
    |& tee "$LOGDIR/rdl_python_install.log"

cd "$ARC_ROOT"
$COMMAND_PKG run -n arc_env python -c "import py_rdl.wrapper.DataInternal" \
    |& tee "$LOGDIR/rdl_import_test.log"

# ----------------------------------------
# 3) Build & enable your local molecule
# ----------------------------------------
cd "$ARC_ROOT/.."
if [[ -d molecule ]]; then
    cd molecule
    git fetch origin || true
    git checkout main || true
    git pull origin main || true
else
    git clone https://github.com/ReactionMechanismGenerator/molecule
    cd molecule
fi

# switch to the branch/tag you need
git fetch origin
git checkout June_13 || echo "⚠️ branch June_13 not found, staying on default"

MOLECULE_PATH=$(pwd)

# compile the Cython extensions in-place
$COMMAND_PKG run -n arc_env python setup.py build_ext --inplace --verbose \
    2>&1 | tee "$LOGDIR/molecule_build.log"

# **skip** pip installing the package—just add it to PYTHONPATH
export PYTHONPATH="$MOLECULE_PATH:${PYTHONPATH:-}"
echo "Added $MOLECULE_PATH to PYTHONPATH"

# sanity‐check imports
$COMMAND_PKG run -n arc_env python - <<'EOF' |& tee "$LOGDIR/molecule_import.log"
import molecule
print("molecule __file__:", molecule.__file__)
import molecule.kinetics.model
print("Loaded kinetics.model OK")
EOF

# surface any build errors
grep -Ei 'error|failed' "$LOGDIR/molecule_build.log" || true

echo "✅ molecule and PyRDL installation complete. Logs in $LOGDIR"
