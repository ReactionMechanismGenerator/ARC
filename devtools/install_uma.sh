#!/usr/bin/env bash
#
# install_uma.sh - Set up the 'uma_env' environment for ARC's UMA engine (USERS, not CI).
#
# UMA (Universal Models for Atoms) is Meta FAIR's fairchem-core foundation MLIP. ARC runs it in
# a dedicated 'uma_env' conda environment (fairchem-core + sella + ase), shelling out to it from
# arc_env via arc/job/env_run.py. This script wraps every step needed to get UMA working:
#
#   1. Create the 'uma_env' conda env from devtools/uma_environment.yml.
#   2. Verify the fairchem / Sella (incl. IRC) / ASE imports the UMA adapter relies on.
#   3. Authenticate to HuggingFace for the GATED uma-s-1p1 model (one-time, interactive).
#   4. Print (and, with --test, use) the environment exports needed to run UMA from arc_env.
#
# This script is intentionally NOT part of devtools/install_all.sh / `make install-ci`: the UMA
# model is gated behind a Meta license + HuggingFace token and is heavy to download, so it is a
# manual, user-driven setup rather than a CI dependency.
#
# Prerequisite (do this once, in a browser logged into HuggingFace):
#   Accept the model license at  https://huggingface.co/facebook/UMA
#   and create an access token with "read access to gated repos".
#
# Usage:
#   bash devtools/install_uma.sh                 # install + verify + HuggingFace login (defaults to CPU)
#   bash devtools/install_uma.sh --cpu           # install UMA optimized for CPU only machines (default)
#   bash devtools/install_uma.sh --gpu           # install UMA with GPU (CUDA) support
#   bash devtools/install_uma.sh --test          # also run the UMA model-dependent unit tests
#   bash devtools/install_uma.sh --skip-hf-login # skip the HuggingFace login step (CI/non-interactive)
#
# Re-running is safe: an existing 'uma_env' is updated in place.

set -eo pipefail

RUN_TESTS=0
SKIP_HF_LOGIN=0
DEVICE="cpu"
for arg in "$@"; do
    case "$arg" in
        --test) RUN_TESTS=1 ;;
        --skip-hf-login) SKIP_HF_LOGIN=1 ;;
        --cpu) DEVICE="cpu" ;;
        --gpu) DEVICE="gpu" ;;
        -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

# Resolve repo paths from this script's location (no hard-coded paths).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARC_DIR="$(dirname "$SCRIPT_DIR")"
ENV_YAML="$SCRIPT_DIR/uma_environment.yml"
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

# 2) Create or update the environment.
if $COMMAND_PKG env list | grep -qE "^\s*${ENV_NAME}\s"; then
    echo ">>> Updating existing '$ENV_NAME' from $ENV_YAML"
    $COMMAND_PKG env update -n "$ENV_NAME" -f "$ENV_YAML" --prune
else
    echo ">>> Creating '$ENV_NAME' from $ENV_YAML"
    $COMMAND_PKG env create -n "$ENV_NAME" -f "$ENV_YAML" -y
fi

# Install UMA dependencies BEFORE PyTorch: fairchem-core pins a torch version
# (currently torch~=2.8.0), so installing torch first only to have pip uninstall it
# again a moment later makes --cpu/--gpu a no-op. Let fairchem settle the version,
# then swap in the build for the requested device.
echo ">>> Installing fairchem-core, sella, and ase"
if [ "$DEVICE" = "cpu" ]; then
    # Offer the PyTorch CPU index alongside PyPI so the torch that fairchem pins is resolved as a
    # '+cpu' build in this very step: its local version sorts above the plain PyPI wheel, which
    # bundles CUDA and drags in ~3 GB of unusable nvidia-*-cu12 packages. Should that wheel be
    # unavailable, pip falls back to PyPI and the swap below cleans up after it.
    $COMMAND_PKG run -n "$ENV_NAME" pip install fairchem-core sella ase \
        --extra-index-url https://download.pytorch.org/whl/cpu
else
    $COMMAND_PKG run -n "$ENV_NAME" pip install fairchem-core sella ase
fi

TORCH_CUDA="$($COMMAND_PKG run -n "$ENV_NAME" python -c 'import torch; print(torch.version.cuda or "")')"
if [ "$DEVICE" = "cpu" ] && [ -n "$TORCH_CUDA" ]; then
    echo ">>> Reinstalling PyTorch as a CPU-only build"
    TORCH_VERSION="$($COMMAND_PKG run -n "$ENV_NAME" python -c 'import torch; print(torch.__version__.split("+")[0])')"
    # --no-deps: torch's dependencies are already in place from the install above, and
    # re-resolving them against the PyTorch index only re-downloads what is already there.
    $COMMAND_PKG run -n "$ENV_NAME" pip install --force-reinstall --no-deps \
        "torch==${TORCH_VERSION}" --index-url https://download.pytorch.org/whl/cpu
    # The CUDA wheel's nvidia-*-cu12 packages are orphaned by the swap, not removed by it.
    NVIDIA_PKGS="$($COMMAND_PKG run -n "$ENV_NAME" pip list --format=freeze | sed -n 's/^\(nvidia-[^=]*\)==.*/\1/p')"
    if [ -n "$NVIDIA_PKGS" ]; then
        echo ">>> Removing the now-unused CUDA runtime packages"
        $COMMAND_PKG run -n "$ENV_NAME" pip uninstall -y $NVIDIA_PKGS
    fi
elif [ "$DEVICE" = "gpu" ]; then
    echo ">>> Checking the installed PyTorch exposes a GPU"
fi

# Report what actually landed — the fairchem pin decides the version, not this script.
# No --no-capture-output: micromamba rejects that flag (see arc/job/env_run.py).
$COMMAND_PKG run -n "$ENV_NAME" python -c \
    'import torch; print(f"    torch {torch.__version__} | cuda build: {torch.version.cuda} | available: {torch.cuda.is_available()}")'
if [ "$DEVICE" = "gpu" ]; then
    $COMMAND_PKG run -n "$ENV_NAME" python -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)' \
        || echo "⚠️   --gpu was requested but torch reports no usable CUDA device. UMA will fall back to CPU."
fi

# 3) Verify the imports the UMA adapter / uma_script.py depend on.
echo ">>> Verifying fairchem / Sella / ASE imports in '$ENV_NAME'"
$COMMAND_PKG run -n "$ENV_NAME" python - <<'PYCODE'
from fairchem.core import FAIRChemCalculator, pretrained_mlip  # noqa: F401
from sella import Sella, IRC                                   # noqa: F401
import ase
print("fairchem + Sella (incl. IRC) + ASE", ase.__version__, "imports OK")
PYCODE

# 4) HuggingFace authentication for the gated uma-s-1p1 model.
if [ "$SKIP_HF_LOGIN" -eq 0 ]; then
    # Use the env's `hf` binary directly rather than `<conda> run ... hf`: `conda run`
    # gives the child no TTY, so the interactive token prompt cannot read input.
    # (`huggingface-cli` is gone in current huggingface_hub — it now errors, not warns.)
    UMA_ENV_PREFIX="$($COMMAND_PKG run -n "$ENV_NAME" python -c 'import sys; print(sys.prefix)')"
    HF_BIN="$UMA_ENV_PREFIX/bin/hf"
    if [ -n "$HF_TOKEN" ]; then
        echo ">>> Using HF_TOKEN from environment for HuggingFace authentication."
        "$HF_BIN" auth login --token "$HF_TOKEN"
    elif "$HF_BIN" auth whoami &>/dev/null; then
        echo "✔️  Already authenticated to HuggingFace."
    else
        echo ">>> HuggingFace login is required for the gated model 'facebook/UMA'."
        echo "    If you have not yet accepted the license, open https://huggingface.co/facebook/UMA first."
        echo "    Then create a token with read access to gated repos:"
        echo "    https://huggingface.co/settings/tokens"
        "$HF_BIN" auth login
    fi
fi

# 5) Runtime environment for invoking UMA from arc_env.
# These exports let arc_env's Python load OpenBabel correctly when invoked non-interactively
# (calling the env's python directly, rather than via an activated shell). PYTHONPATH points at
# the ARC checkout. Computed dynamically so there are no hard-coded paths.
ARC_ENV_PY="$($COMMAND_PKG run -n arc_env python -c 'import sys; print(sys.executable)')"
ARC_ENV_PREFIX="$(dirname "$(dirname "$ARC_ENV_PY")")"
BABEL_VERSION_DIR="$(ls -d "$ARC_ENV_PREFIX"/lib/openbabel/*/ 2>/dev/null | head -1)"

export_block() {
    echo "export BABEL_LIBDIR=${BABEL_VERSION_DIR%/}"
    echo "export BABEL_DATADIR=${ARC_ENV_PREFIX}/share/openbabel/$(basename "${BABEL_VERSION_DIR%/}")"
    echo "export PYTHONPATH=${ARC_DIR}:\$PYTHONPATH"
}

echo ""
echo "✅  '$ENV_NAME' is ready. ARC discovers it via find_executable('$ENV_NAME')."
echo ""
echo "To run a UMA job, activate arc_env and set 'method' to 'uma' (resolves to the latest model)."
echo "To run the UMA model-dependent unit tests, export the following and run pytest with UMA_RUN_MODEL=1:"
echo "----------------------------------------------------------------------"
export_block
echo "UMA_RUN_MODEL=1 ${ARC_ENV_PY} -m pytest arc/job/adapters/uma_test.py -v"
echo "----------------------------------------------------------------------"

# 6) Optionally run the model-dependent tests now, using the exports above.
if [ "$RUN_TESTS" -eq 1 ]; then
    echo ">>> Running the UMA model-dependent unit tests (first run downloads the model; this is slow)..."
    export BABEL_LIBDIR="${BABEL_VERSION_DIR%/}"
    export BABEL_DATADIR="${ARC_ENV_PREFIX}/share/openbabel/$(basename "${BABEL_VERSION_DIR%/}")"
    export PYTHONPATH="${ARC_DIR}:${PYTHONPATH}"
    UMA_RUN_MODEL=1 "$ARC_ENV_PY" -m pytest "$ARC_DIR/arc/job/adapters/uma_test.py" -v
fi

echo "✅  UMA setup script finished."
