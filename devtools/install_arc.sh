#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Command-line flags
###############################################################################
DO_COMPILE=true
PRUNE=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-compile) DO_COMPILE=false ;;
        --prune)      PRUNE=true       ;;
        --help|-h)
            echo "Usage: $0 [--no-compile] [--prune]"
            exit 0
            ;;
        *) echo "Unknown flag $1" ; exit 1 ;;
    esac
    shift
done
PRUNE_FLAG=""

###############################################################################
# Locate ARC repo root
###############################################################################
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if ARC_ROOT_GIT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null); then
    ARC_ROOT="$ARC_ROOT_GIT"
else
    ARC_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
echo "📂  ARC root: $ARC_ROOT"

###############################################################################
# Paths / filenames
###############################################################################
LOGDIR="$ARC_ROOT/build"
mkdir -p "$LOGDIR"
BUILD_LOG="$LOGDIR/compile.log"
IMPORT_LOG="$LOGDIR/import.log"
ENV_NAME=arc_env
ENV_FILE="$ARC_ROOT/environment.yml"

###############################################################################
# Detect conda front-end
###############################################################################
if command -v micromamba &>/dev/null; then
    echo "✔️ Micromamba is installed."
    COMMAND_PKG=micromamba
elif command -v mamba &>/dev/null; then
    echo "✔️ Mamba is installed."
    COMMAND_PKG=mamba
elif command -v conda &>/dev/null; then
    echo "✔️ Conda is installed."
    COMMAND_PKG=conda
else
    echo "❌ Micromamba, Mamba, or Conda is required."
    exit 1
fi

# Shell hook
if [[ $COMMAND_PKG == micromamba ]]; then
    eval "$(micromamba shell hook --shell bash)"
else
    BASE=$(conda info --base)
    source "$BASE/etc/profile.d/conda.sh"
    eval "$($COMMAND_PKG shell hook --shell bash)"
fi

###############################################################################
# Pick create/update sub-commands
###############################################################################
if [[ $COMMAND_PKG == micromamba ]]; then
    CREATE_CMD="create"
    if $COMMAND_PKG env --help 2>&1 | grep -q "update"; then
        UPDATE_CMD="env update"
    else
        UPDATE_CMD="install"
    fi
else
    CREATE_CMD="env create"
    UPDATE_CMD="env update"
fi

###############################################################################
# Create or update environment
###############################################################################
if $COMMAND_PKG env list | awk '{print $1}' | sed 's/^\*//' | grep -Fxq "$ENV_NAME"; then
    echo "🔄 Updating $ENV_NAME"
    $PRUNE && [[ $COMMAND_PKG != micromamba ]] && PRUNE_FLAG="--prune"
    $COMMAND_PKG $UPDATE_CMD -n "$ENV_NAME" -f "$ENV_FILE" $PRUNE_FLAG -y
else
    echo "📦 Creating $ENV_NAME"
    $COMMAND_PKG $CREATE_CMD -n "$ENV_NAME" -f "$ENV_FILE" -y
fi

###############################################################################
# Compile Cython extensions
###############################################################################
if $DO_COMPILE; then
    cd "$ARC_ROOT"
    echo "🔨 Compiling Cython extensions … (log → $BUILD_LOG)"
    mkdir -p "$(dirname "$BUILD_LOG")"

    # Run make *inside* the env without activating the shell session
    $COMMAND_PKG run -n "$ENV_NAME" \
        make -j"$(nproc)" BUILD_DIR=build compile 2>&1 | tee "$BUILD_LOG"

    status=${PIPESTATUS[0]}        # real exit code from make
    if (( status != 0 )); then
        echo "❌ Build failed – check $BUILD_LOG"
        exit $status
    fi
fi


###############################################################################
# Sanity-import check
###############################################################################
$COMMAND_PKG run -n "$ENV_NAME" python - <<'PY' | tee "$IMPORT_LOG"
import arc.molecule, sys
print("arc.molecule imported from:", arc.molecule.__file__)
PY

###############################################################################
# Final message
###############################################################################
grep -Ei ': error:|failed' "$BUILD_LOG" || true
echo "✅ ARC environment and molecule module setup complete."
