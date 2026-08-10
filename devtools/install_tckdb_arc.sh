#!/bin/bash -l
set -eo pipefail

# -------------------------------------------------------------------------
# Install the OPTIONAL tckdb-arc upload adapter.
#
# WORK IN PROGRESS: the adapter is under active development, so this
# installer tracks a moving target and is expected to change as the adapter
# does. It is deliberately NOT wired into install_all.sh — ARC runs fine
# without the adapter, and `ARC.py::run_tckdb_upload` warns once and
# continues when the package is absent.
#
# Unlike the other devtools installers, this one does NOT create its own
# conda environment. The adapter is imported in-process by ARC, so it must
# live in the SAME environment ARC runs from (default: arc_env).
#
# Usage:
#   devtools/install_tckdb_arc.sh                     # from GitHub into arc_env
#   devtools/install_tckdb_arc.sh --env my_env        # into a different env
#   devtools/install_tckdb_arc.sh --ref some-branch   # a specific branch/tag
#   devtools/install_tckdb_arc.sh --editable ~/code/tckdb-adapters/tckdb_arc
# -------------------------------------------------------------------------

REPO_URL="https://github.com/calvinp0/tckdb-adapters.git"
SUBDIRECTORY="tckdb_arc"
TARGET_ENV="arc_env"
REF="main"
EDITABLE_PATH=""
MINIMUM_PYTHON="3.11"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env)      TARGET_ENV="$2"; shift 2 ;;
        --ref)      REF="$2"; shift 2 ;;
        --editable) EDITABLE_PATH="$2"; shift 2 ;;
        --help|-h)
            sed -n '4,23p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *)
            echo "❌ Unknown argument: $1 (try --help)"
            exit 1 ;;
    esac
done

if command -v micromamba &> /dev/null; then
    echo "✔️ Micromamba is installed."
    COMMAND_PKG=micromamba
elif command -v mamba &> /dev/null; then
    echo "✔️ Mamba is installed."
    COMMAND_PKG=mamba
elif command -v conda &> /dev/null; then
    echo "✔️ Conda is installed."
    COMMAND_PKG=conda
else
    echo "❌ Micromamba, Mamba, or Conda is required. Please install one."
    exit 1
fi

if [ "$COMMAND_PKG" = "micromamba" ]; then
    eval "$(micromamba shell hook --shell=bash)"
    ENV_RUNNER="micromamba run -n $TARGET_ENV"
else
    BASE=$(conda info --base)
    . "$BASE/etc/profile.d/conda.sh"
    ENV_RUNNER="conda run -n $TARGET_ENV"
fi

if ! $ENV_RUNNER python -V &> /dev/null; then
    echo "❌ Environment '$TARGET_ENV' not found or has no Python."
    echo "   Create it first (devtools/install_arc.sh) or pass --env <name>."
    exit 1
fi

# The adapter and its tckdb-client dependency both require Python >= 3.11.
# Check before pip resolves, so the failure names the real problem.
echo ">>> Checking Python in '$TARGET_ENV' (need >= $MINIMUM_PYTHON)..."
if ! $ENV_RUNNER python -c "
import sys
minimum = tuple(int(part) for part in '$MINIMUM_PYTHON'.split('.'))
sys.exit(0 if sys.version_info[:2] >= minimum else 1)
" 2>/dev/null; then
    FOUND=$($ENV_RUNNER python -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
    echo "❌ '$TARGET_ENV' runs Python $FOUND, but tckdb-arc requires >= $MINIMUM_PYTHON."
    echo "   Install into an environment that meets the floor, or pass --env <name>."
    exit 1
fi
echo "✔️ Python check passed."

if [ -n "$EDITABLE_PATH" ]; then
    if [ ! -f "$EDITABLE_PATH/pyproject.toml" ]; then
        echo "❌ No pyproject.toml under '$EDITABLE_PATH'."
        echo "   Point --editable at the tckdb_arc package directory of a tckdb-adapters checkout."
        exit 1
    fi
    echo ">>> Installing tckdb-arc (editable) from $EDITABLE_PATH into $TARGET_ENV..."
    $ENV_RUNNER python -m pip install -e "$EDITABLE_PATH"
else
    echo ">>> Installing tckdb-arc from $REPO_URL@$REF into $TARGET_ENV..."
    $ENV_RUNNER python -m pip install "git+$REPO_URL@$REF#subdirectory=$SUBDIRECTORY"
fi

echo ">>> Verifying the adapter imports..."
if $ENV_RUNNER python -c "import tckdb_arc" &> /dev/null; then
    VERSION=$($ENV_RUNNER python -c "
from importlib.metadata import version
print(version('tckdb-arc'))
" 2>/dev/null || echo "unknown")
    echo "✔️ tckdb-arc $VERSION is importable from '$TARGET_ENV'."
else
    echo "❌ tckdb-arc installed but does not import. Check the pip output above."
    exit 1
fi

echo "✅ Done installing tckdb-arc (WORK IN PROGRESS — expect this installer to change)."
