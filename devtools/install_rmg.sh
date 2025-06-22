#!/usr/bin/env bash
set -euo pipefail

echo ">>> Starting RMG-Py installer..."

###############################################################################
# CONFIGURATION
###############################################################################
USE_SSH=false
ENV_NAME="rmg_env"

# Parse flags
for arg in "$@"; do
    case $arg in
        --ssh) USE_SSH=true ;;
    esac
done

# Detect conda frontend
if command -v micromamba &>/dev/null; then
    COMMAND=micromamba
elif command -v mamba &>/dev/null; then
    COMMAND=mamba
elif command -v conda &>/dev/null; then
    COMMAND=conda
else
    echo "❌ No conda/mamba/micromamba found in PATH."
    exit 1
fi

echo "✔️ Using $COMMAND to manage environments"

###############################################################################
# CLONE REPOS (shallow)
###############################################################################
ARC_PATH="$(pwd)"
PARENT_DIR="$(realpath "$ARC_PATH/..")"
cd "$PARENT_DIR"

clone_repo() {
    local repo_name=$1
    local ssh_url=$2
    local https_url=$3
    if [[ -d "$repo_name" ]]; then
        echo "✔️ $repo_name already exists, skipping clone"
    else
        local url="$([ "$USE_SSH" == true ] && echo "$ssh_url" || echo "$https_url")"
        echo "📦 Cloning $repo_name with --depth 1"
        git clone --depth 1 "$url" "$repo_name"
    fi
}

clone_repo RMG-Py \
    git@github.com:ReactionMechanismGenerator/RMG-Py.git \
    https://github.com/ReactionMechanismGenerator/RMG-Py.git

clone_repo RMG-database \
    git@github.com:ReactionMechanismGenerator/RMG-database.git \
    https://github.com/ReactionMechanismGenerator/RMG-database.git

export RMG_PY_PATH="$(realpath RMG-Py)"
export RMG_DB_PATH="$(realpath RMG-database)"

###############################################################################
# CREATE ENVIRONMENT
###############################################################################
cd "$RMG_PY_PATH"
echo "📚 Creating conda env: $ENV_NAME"
$COMMAND env create -n "$ENV_NAME" -f environment.yml || {
    echo "⚠️ Environment likely exists; trying update..."
    $COMMAND env update -n "$ENV_NAME" -f environment.yml
}

###############################################################################
# COMPILE RMG
###############################################################################
echo "🔧 Compiling RMG-Py..."
$COMMAND run -n "$ENV_NAME" make -j"$(nproc)"

###############################################################################
# UPDATE SHELL PATH
###############################################################################
case "$SHELL" in
    */zsh) RC=~/.zshrc ;;
    *)     RC=~/.bashrc ;;
esac

if ! grep -q "$RMG_PY_PATH" "$RC"; then
    echo "📌 Adding RMG-Py to \$PATH in $RC"
    echo "export PATH=\"$RMG_PY_PATH:\$PATH\"" >> "$RC"
fi

###############################################################################
# INSTALL JULIA INTO THE SAME ENV
###############################################################################
echo "📦 Installing Julia 1.10 in $ENV_NAME"
$COMMAND install -n "$ENV_NAME" -c conda-forge julia=1.10

###############################################################################
# INSTALL RMS
###############################################################################
echo "🚀 Running install_rms.sh"
source install_rms.sh

echo "✅ RMG-Py installation complete."
