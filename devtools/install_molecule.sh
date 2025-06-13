#!/usr/bin/env bash
set -eo pipefail
set -x  # echo commands

# where to dump logs
LOGDIR="$HOME/molecule_build_logs"
mkdir -p "$LOGDIR"

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

# 1) pick and activate the conda front-end
if command -v micromamba &>/dev/null; then
    COMMAND_PKG=micromamba
elif command -v mamba &>/dev/null; then
    COMMAND_PKG=mamba
elif command -v conda &>/dev/null; then
    COMMAND_PKG=conda
else
    echo "❌ No conda-compatible package manager found."
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
# 2) Build RingDecomposerLib
# ----------------------------------------
cd "$RDL_PARENT_DIR"
if [[ -d RingDecomposerLib ]]; then
    cd RingDecomposerLib
    CURRENT_BRANCH=$(git symbolic-ref --short HEAD)
    if [[ "$CURRENT_BRANCH" == "main" ]]; then
        git fetch origin main && git pull origin main
    else
        echo "⚠️ RingDecomposerLib on branch '$CURRENT_BRANCH', skipping update."
    fi
else
    git clone https://github.com/DanaResearchGroup/RingDecomposerLib
    cd RingDecomposerLib
fi

mkdir -p build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
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
# 3) Build & install molecule
# ----------------------------------------
cd "$ARC_ROOT/.."
if [[ -d molecule ]]; then
    cd molecule
    CURRENT_BRANCH=$(git symbolic-ref --short HEAD)
    if [[ "$CURRENT_BRANCH" == "main" ]]; then
        git fetch origin && git pull origin main
    else
        echo "⚠️ molecule on branch '$CURRENT_BRANCH', skipping update."
    fi
else
    git clone https://github.com/ReactionMechanismGenerator/molecule
    cd molecule
fi

git fetch origin
git checkout May_25 || {
    echo "⚠️ Failed to switch to branch May_25."
    exit 1
}

MOLECULE_PATH=$(pwd)

# compile the Cython extensions in place
$COMMAND_PKG run -n arc_env python setup.py build_ext --inplace --verbose \
    2>&1 | tee "$LOGDIR/molecule_build.log" || {
    echo ">>> Build failed; last 50 lines of log:"
    tail -n50 "$LOGDIR/molecule_build.log"
    exit 1
}

# install the package, reusing the in-place build
$COMMAND_PKG run -n arc_env python -m pip install . --no-deps --no-build-isolation --verbose \
    2>&1 | tee "$LOGDIR/molecule_python_install.log"

# immediately add the cloned source to PYTHONPATH
export PYTHONPATH="$MOLECULE_PATH:${PYTHONPATH:-}"
echo "Added $MOLECULE_PATH to PYTHONPATH"

# sanity-check import
$COMMAND_PKG run -n arc_env python - <<'EOF' |& tee "$LOGDIR/molecule_import.log"
import molecule
print("molecule version:", molecule.__version__)
EOF

# persist for future shells
LINE="export PYTHONPATH=\${PYTHONPATH:-}:$MOLECULE_PATH"
if ! grep -Fxq "$LINE" ~/.bashrc; then
    echo "$LINE" >> ~/.bashrc
fi

echo "✅ molecule and PyRDL installation complete. Logs in $LOGDIR"


micromamba activate arc_env   # or conda activate arc_env
python -c "import molecule; print(molecule.__file__)"
python - <<EOF
import sys, pprint
pprint.pprint(sys.path)
EOF


grep -Ei 'error|failed' "$HOME/molecule_build_logs/molecule_build.log"

