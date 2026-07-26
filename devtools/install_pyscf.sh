#!/usr/bin/env bash
#
# install_pyscf.sh - Set up the 'pyscf_env' environment for ARC's PySCF ESS adapter.
#
# PySCF is a Python quantum-chemistry package that ARC runs as a local, in-core ESS (no external
# scheduler), shelling out to a dedicated 'pyscf_env' conda environment from arc_env. This script
# wraps every step needed to get PySCF working:
#
#   1. Create the 'pyscf_env' conda env from devtools/pyscf_environment.yml (CPU baseline: the
#      pyscf and pyscf-dispersion PyPI wheels installed via the yml's pip: block).
#   2. With --cuda/--gpu, additionally install the GPU stack (gpu4pyscf-cuda12x) on top.
#   3. Verify the pyscf (and, for GPU, gpu4pyscf) imports the adapter relies on.
#
# The CPU baseline is wired into devtools/install_all.sh, so a plain `make install` yields a
# working CPU pyscf_env. The GPU stack is opt-in (a CUDA 12.x toolkit + NVIDIA GPU are required):
#
# Usage:
#   bash devtools/install_pyscf.sh                 # install + verify the CPU baseline (default)
#   bash devtools/install_pyscf.sh --cuda          # also install the gpu4pyscf-cuda12x GPU stack
#   bash devtools/install_pyscf.sh --gpu           # alias for --cuda
#
# Re-running is safe: an existing 'pyscf_env' is updated in place.

set -eo pipefail

# gpu4pyscf-cuda12x pin for the optional GPU stack (CUDA 12.x wheels; see gpu4pyscf docs).
GPU4PYSCF_PKG="gpu4pyscf-cuda12x==1.7.6"

DEVICE="cpu"
for arg in "$@"; do
    case "$arg" in
        --cpu) DEVICE="cpu" ;;
        --cuda|--gpu) DEVICE="gpu" ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

# Resolve repo paths from this script's location (no hard-coded paths).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_YAML="$SCRIPT_DIR/pyscf_environment.yml"
ENV_NAME="$(grep -E '^ *name:' "$ENV_YAML" | head -1 | awk '{print $2}')"

# 1) Pick a conda front-end and initialize shell integration.
if command -v micromamba &>/dev/null; then
    COMMAND_PKG=micromamba
    eval "$(micromamba shell hook --shell=bash)"
elif command -v mamba &>/dev/null; then
    COMMAND_PKG=mamba
    BASE=$(conda info --base); source "$BASE/etc/profile.d/conda.sh"
elif command -v conda &>/dev/null; then
    COMMAND_PKG=conda
    BASE=$(conda info --base); source "$BASE/etc/profile.d/conda.sh"
else
    echo "❌  No micromamba/mamba/conda found in PATH." >&2
    exit 1
fi
echo "✔️  Using $COMMAND_PKG"

# 2) Create or update the CPU environment (pyscf + pyscf-dispersion via the yml's pip: block).
if $COMMAND_PKG env list | grep -qE "^\s*${ENV_NAME}\s"; then
    echo ">>> Updating existing '$ENV_NAME' from $ENV_YAML"
    $COMMAND_PKG env update -n "$ENV_NAME" -f "$ENV_YAML" --prune
else
    echo ">>> Creating '$ENV_NAME' from $ENV_YAML"
    $COMMAND_PKG env create -n "$ENV_NAME" -f "$ENV_YAML" -y
fi

# 3) Optionally layer the GPU stack on top.
if [ "$DEVICE" = "gpu" ]; then
    echo ">>> Installing the GPU stack ($GPU4PYSCF_PKG) into '$ENV_NAME'"
    $COMMAND_PKG run -n "$ENV_NAME" pip install "$GPU4PYSCF_PKG"
fi

# 4) Verify the imports the PySCF adapter / pyscf_script.py depend on.
echo ">>> Verifying pyscf imports in '$ENV_NAME'"
$COMMAND_PKG run -n "$ENV_NAME" python - "$DEVICE" <<'PYCODE'
import sys
import pyscf
print("pyscf", pyscf.__version__, "imports OK")
if sys.argv[1] == "gpu":
    import gpu4pyscf
    print("gpu4pyscf", gpu4pyscf.__version__, "imports OK")
PYCODE

echo ""
echo "✅  '$ENV_NAME' is ready. ARC discovers it via find_executable('$ENV_NAME')."
if [ "$DEVICE" = "cpu" ]; then
    echo "    Installed the CPU baseline; re-run with --cuda to add the gpu4pyscf GPU stack."
fi
echo "✅  PySCF setup script finished."
