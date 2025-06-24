#!/usr/bin/env bash
set -euo pipefail

echo ">>> Starting RMG-Py installer..."

###############################################################################
# CONFIGURATION
###############################################################################
# default values
PATH_MODE=path        # {path|pip}
INSTALL_RMS=false
USE_SSH=false
ENV_NAME="rmg_env"   # conda environment name

while getopts ":prsh" opt; do
  case $opt in
      p) PATH_MODE=pip   ;;   # -p  → pip-install
      r) INSTALL_RMS=true ;;  # -r  → install RMS
      s) USE_SSH=true    ;;   # -s  → clone via SSH
      h|\?) cat <<EOF
Usage: $0 [-p] [-r] [-s]
  -p   pip-install RMG-Py instead of adding to PATH
  -r   install Reaction Mechanism Simulator
  -s   use SSH to clone repos
EOF
          exit 0 ;;
  esac
done
shift $((OPTIND-1))   # remove parsed flags from $@

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
# ── locate ARC root ───────────────────────────────────────────────────────
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ARC_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || \
            printf '%s\n' "$(dirname "$SCRIPT_DIR")")"

# ── set clone root to the parent of ARC ───────────────────────────────────
CLONE_ROOT="$(dirname "$ARC_ROOT")"      # ← this is ~/code when ARC is ~/code/ARC
cd "$CLONE_ROOT"
echo "📂  Cloning into: $CLONE_ROOT"

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
# CREATE OR UPDATE rmg_env
###############################################################################
cd "$RMG_PY_PATH"

if [[ ! -f environment.yml ]]; then
    echo "❌ environment.yml not found in RMG-Py directory."
    exit 1
fi

if $COMMAND env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo ">>> Updating existing environment: $ENV_NAME"
    $COMMAND env update -n "$ENV_NAME" -f environment.yml --prune -y
else
    echo ">>> Creating new environment: $ENV_NAME"
    $COMMAND env create -n "$ENV_NAME" -f environment.yml -y
fi

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

if [[ ! -f "$RC" ]]; then
    echo "❌ Shell configuration file not found: $RC"
    exit 1
fi

ACTIVE_RE="^[[:space:]]*[^#].*${RMG_PY_PATH//\//\\/}"   # uncommented, contains path
COMMENT_RE="^[[:space:]]*#.*${RMG_PY_PATH//\//\\/}"     # commented-out, contains path
NEW_LINE='export PATH="$PATH:'"$RMG_PY_PATH"'"'


# If PATH_ADD is true, add RMG-Py to PATH
if [ "$PATH_MODE" == path ]; then
    if grep -Eq "$ACTIVE_RE" "$RC"; then
        printf 'ℹ️  RMG-Py already active in %s\n' "$RC"

    elif grep -Eq "$COMMENT_RE" "$RC"; then
        printf '✅  Found commented entry; adding new active line\n'
        printf '\n# RMG-Py added on %s\n%s\n' "$(date +%F)" "$NEW_LINE" >> "$RC"

    else
        printf '✅  No entry found; adding new active line\n'
        printf '\n# RMG-Py added on %s\n%s\n' "$(date +%F)" "$NEW_LINE" >> "$RC"
    fi
else
    # pip install
    $COMMAND run -n "$ENV_NAME" pip install -e "$RMG_PY_PATH"
    echo "📦 RMG-Py installed via pip in $ENV_NAME"
fi


###############################################################################
# INSTALL JULIA & RMS
###############################################################################
if [ "$INSTALL_RMS" = true ]; then
    echo "📦 Installing RMS (Reaction Mechanism Simulator) in $ENV_NAME"
    # Check if juliaup is installed - juliaup &> /dev/null
    if ! command -v juliaup &> /dev/null; then
        echo "❌ juliaup not found. Installing it now."
        curl -fsSL https://install.julialang.org | bash
        export PATH="$HOME/.juliaup/bin:$PATH"
        # Add it to bashrc/zshrc
        echo 'export PATH="$HOME/.juliaup/bin:$PATH"' >> "$RC"
        # juliaup 1.10
        juliaup add 1.10
        juliaup default 1.10
        juliaup remove release
    else
        echo "✔️ juliaup is already installed."
        echo "Checking for Julia 1.10..."
        if ! julia --version | grep -q "1\.10"; then
            echo "❌ Julia 1.10 not found. Installing it now."
            juliaup add 1.10
            juliaup default 1.10
        else
            echo "✔️ Julia 1.10 is already installed."
        fi
        # Get julia path
        JULIA_PATH=$(which julia)
        echo "Using Julia at: $JULIA_PATH"
        # Set the micromamba/conda/mamba env config vars
        # Find the base path - check COMMAND is micromamba, mamba, or conda
        if [ "$COMMAND" = "micromamba" ]; then
            BASE_PATH=$(micromamba info --base | awk -F': +' '/base environment/ {print $2; exit}')
        elif [ "$COMMAND" = "mamba" ] || [ "$COMMAND" = "conda" ]; then
            BASE_PATH=$(conda info --base)
        else
            echo "❌ Unknown command: $COMMAND"
            exit 1
        fi

        # Check if COMMAND is not micromamba
        if [ "$COMMAND" != "micromamba" ]; then
            conda env config vars set -n "$ENV_NAME" \
                JULIA_CONDAPKG_BACKEND=Null \
                JULIA_PYTHONCALL_EXE=$BASE_PATH/envs/$ENV_NAME/bin/python \
                PYTHON_JULIAPKG_EXE=$JULIA_PATH \
                PYTHON_JULIAPKG_PROJECT=$BASE_PATH/envs/$ENV_NAME/julia_env
        else
            mkdir -p "$BASE_PATH/envs/$ENV_NAME/etc/conda/activate.d"
            cat > "$BASE_PATH/envs/$ENV_NAME/etc/conda/activate.d/julia_vars.sh" <<'EOF'
export JULIA_CONDAPKG_BACKEND=Null
export JULIA_PYTHONCALL_EXE="$CONDA_PREFIX/bin/python"
export PYTHON_JULIAPKG_EXE="$(command -v julia)"
export PYTHON_JULIAPKG_PROJECT="$CONDA_PREFIX/julia_env"
EOF
        
        fi

        # Now export the variables to the current shell
        export JULIA_CONDAPKG_BACKEND=Null
        export JULIA_PYTHONCALL_EXE=$BASE_PATH/envs/$ENV_NAME/bin/python
        export PYTHON_JULIAPKG_EXE=$JULIA_PATH
        export PYTHON_JULIAPKG_PROJECT=$BASE_PATH/envs/$ENV_NAME/julia_env

        # install pyjuliacall
        echo "📦 Installing PyJuliaCall in $ENV_NAME"
        $COMMAND install -n "$ENV_NAME" -c conda-forge pyjuliacall -y
        export RMS_BRANCH=${RMS_BRANCH:-for_rmg}
        echo "📦 Installing ReactionMechanismSimulator - BRANCH: $RMS_BRANCH"
        $COMMAND run -n "$ENV_NAME" julia -e '
            using Pkg;
            Pkg.add(Pkg.PackageSpec(
                name="ReactionMechanismSimulator",
                url="https://github.com/ReactionMechanismGenerator/ReactionMechanismSimulator.jl.git",
                rev=ENV["RMS_BRANCH"]
            ));
            using ReactionMechanismSimulator;
            Pkg.instantiate();
        ' || echo "RMS install error – continuing anyway ¯\_(ツ)_/¯"   # conda-run executes in env :contentReference[oaicite:1]{index=1}
        echo "Checking if RMS is installed..."
$COMMAND run -n "$ENV_NAME" python - <<'PY'
from juliacall import Main
import sys
pk_ok = Main.seval('Base.identify_package("ReactionMechanismSimulator") !== nothing')
print("RMS visible to Python in rmg_env" if pk_ok else "RMS NOT found")
sys.exit(0 if pk_ok else 1)
PY
fi
else
    echo "ℹ️ Skipping RMS installation as INSTALL_RMS is set to false."
fi

echo "✅ RMG-Py installation complete."
