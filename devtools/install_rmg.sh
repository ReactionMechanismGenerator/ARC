#!/usr/bin/env bash
set -euo pipefail

echo ">>> Starting RMG-Py installer..."

###############################################################################
# CONFIGURATION
###############################################################################
# default values ────────────────────────────────────────────────────────────
MODE=package        # {package|source}
SOURCE_MODE=path    # {path|pip|conda} (only used when MODE=source)
USE_SSH=false
ENV_NAME="rmg_env"
RMG_VERSION="${RMG_VERSION:-3.3.0}"

TEMP=$(getopt -o pschS --long pip,conda,ssh,source,help -- "$@")
[[ $? -eq 0 ]] || { echo "Flag parsing failed"; exit 1; }
eval set -- "$TEMP"
while true; do
    case "$1" in
        -p|--pip)    MODE=source; SOURCE_MODE=pip; shift ;;
        -c|--conda)  MODE=source; SOURCE_MODE=conda; shift ;;
        -s|--ssh)    USE_SSH=true; shift ;;
        -S|--source) MODE=source; shift ;;
        -h|--help)
cat <<EOF
Usage: $0 [--source] [--pip] [--conda] [--ssh]

  default       install packaged RMG (RMG-Py + database + Arkane) into rmg_env
  -S, --source  install RMG-Py from source and build in rmg_env
  -p, --pip     (source) install RMG-Py as an editable pip package
  -c, --conda   (source) export RMG-Py & database into conda activation/deactivation hooks
  -s, --ssh     clone via git@… instead of https://

Notes:
  - Examples: --source --conda | --source --pip | --source --ssh
  - Override rmg version with RMG_VERSION=3.3.0 (or similar) in the environment.
EOF
            exit 0 ;;
        --) shift; break ;;
        *)  echo "Internal getopt error"; exit 1 ;;
    esac
done

# Detect conda frontend
if command -v micromamba &>/dev/null; then
    COMMAND_PKG=micromamba
elif command -v mamba &>/dev/null; then
    COMMAND_PKG=mamba
elif command -v conda &>/dev/null; then
    COMMAND_PKG=conda
else
    echo "❌ No conda/mamba/micromamba found in PATH."
    exit 1
fi

echo "✔️ Using $COMMAND_PKG to manage environments"

###############################################################################
# Create/update rmg_env (package install)
###############################################################################
if [[ "$MODE" == "package" ]]; then
    if $COMMAND_PKG env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
        echo ">>> Updating existing environment: $ENV_NAME"
        $COMMAND_PKG install -n "$ENV_NAME" -y -c rmg -c conda-forge \
            "python=3.9" "conda-forge::numpy>=1.10.0,<2" "blas=*=openblas" "conda-forge::numdifftools" \
            "rmg=${RMG_VERSION}" "connie::symmetry"
    else
        echo ">>> Creating new environment: $ENV_NAME"
        $COMMAND_PKG create -n "$ENV_NAME" -y -c rmg -c conda-forge \
            "python=3.9" "conda-forge::numpy>=1.10.0,<2" "blas=*=openblas" "conda-forge::numdifftools" \
            "rmg=${RMG_VERSION}" "connie::symmetry"
    fi
fi

###############################################################################
# Paths and clones
###############################################################################
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if ARC_ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null); then
    :  # got it
else
    ARC_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
fi
CLONE_ROOT="$(dirname "$ARC_ROOT")"
cd "$CLONE_ROOT" || exit 1
echo "📂  Clone root: $CLONE_ROOT"

clone_repo () {
    local repo_name=$1 ssh_url=$2 https_url=$3
    [[ -d $repo_name/.git ]] && { echo "✔️  $repo_name exists"; return; }

    local url
    if $USE_SSH; then
        url=$ssh_url
    else
        url=$https_url
    fi

    echo "📦  Cloning $repo_name → $url"
    git clone --depth 1 "$url" "$repo_name"
}

if [[ "$MODE" == "source" ]]; then
    # ---------- actual clones ---------------------------------------------------
    clone_repo RMG-Py \
               git@github.com:ReactionMechanismGenerator/RMG-Py.git \
               https://github.com/ReactionMechanismGenerator/RMG-Py.git

    clone_repo RMG-database \
               git@github.com:ReactionMechanismGenerator/RMG-database.git \
               https://github.com/ReactionMechanismGenerator/RMG-database.git

    export RMG_PY_PATH=$CLONE_ROOT/RMG-Py
    export RMG_DB_PATH=$CLONE_ROOT/RMG-database

    ###############################################################################
    # CREATE OR UPDATE rmg_env
    ###############################################################################
    cd "$RMG_PY_PATH"

    if [[ ! -f environment.yml ]]; then
        echo "❌ environment.yml not found in RMG-Py directory."
        exit 1
    fi

    # Prepare flags - --strict-channel-priority only works with mamba/micromamba on create
    CHANNEL_PRIORITY_FLAG=""
    if [[ "$COMMAND_PKG" != "conda" ]]; then
        CHANNEL_PRIORITY_FLAG="--strict-channel-priority"
    fi

    if $COMMAND_PKG env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
        echo ">>> Updating existing environment: $ENV_NAME"
        $COMMAND_PKG env update -n "$ENV_NAME" -f environment.yml --prune -y
    else
        echo ">>> Creating new environment: $ENV_NAME"
        $COMMAND_PKG env create -n "$ENV_NAME" -f environment.yml -y $CHANNEL_PRIORITY_FLAG
    fi

    ###############################################################################
    # COMPILE RMG
    ###############################################################################
    echo "🔧 Compiling RMG-Py..."
    $COMMAND_PKG run -n "$ENV_NAME" make -j"$(nproc)"
fi


###############################################################################
# Inject (PY)PATH hooks into any conda-like env
###############################################################################
add_rmg_hooks () {
    local env_name="$1"
    if [[ $COMMAND_PKG != micromamba && $COMMAND_PKG != mamba && $COMMAND_PKG != conda ]]; then
        echo "❌ Cannot add hooks: $COMMAND_PKG is not a conda-like package manager."
        return 1
    elif [[ $COMMAND_PKG == micromamba ]]; then
        local env_prefix=$(micromamba info --base)/envs/"$env_name"
    else
        local env_prefix=$(conda info --base)/envs/"$env_name"
    fi
    local act_dir="$env_prefix/etc/conda/activate.d"
    local deact_dir="$env_prefix/etc/conda/deactivate.d"
    mkdir -p "$act_dir" "$deact_dir"

    # ---------- activation hook ----------
    cat >"$act_dir/99-rmg.sh" <<EOF
# added by installer on $(date +%F)
export RMG_PY_PATH="$RMG_PY_PATH"
export RMG_DB_PATH="$RMG_DB_PATH"

# prepend once
case ":\$PATH:" in *":\$RMG_PY_PATH:"*) ;; \
  *) export PATH="\$RMG_PY_PATH:\$PATH" ;; esac

case "\${PYTHONPATH:+:\$PYTHONPATH:}" in *":\$RMG_PY_PATH:"*) ;; \
  *) export PYTHONPATH="\${PYTHONPATH:+\$RMG_PY_PATH:\$PYTHONPATH}" || \
     export PYTHONPATH="\$RMG_PY_PATH" ;; esac
EOF

    # ---------- deactivation hook ----------
    cat >"$deact_dir/99-rmg.sh" <<'EOF'
_strip () { local n=":$1:"; local s=":$2:"; echo "${s//$n/:}" | sed 's/^://;s/:$//'; }
export PATH=$(_strip "$RMG_PY_PATH" "$PATH")
export PYTHONPATH=$(_strip "$RMG_PY_PATH" "${PYTHONPATH:-}")
unset RMG_PY_PATH RMG_DB_PATH
EOF

    echo "🔗 RMG hooks added to $env_name"
}


###############################################################################
# UPDATE SHELL PATH (source installs only)
###############################################################################
if [[ "$MODE" == "source" ]]; then
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
    if [ "$SOURCE_MODE" == path ]; then
        if grep -Eq "$ACTIVE_RE" "$RC"; then
            printf 'ℹ️  RMG-Py already active in %s\n' "$RC"

        elif grep -Eq "$COMMENT_RE" "$RC"; then
            printf '✅  Found commented entry; adding new active line\n'
            printf '\n# RMG-Py added on %s\n%s\n' "$(date +%F)" "$NEW_LINE" >> "$RC"

        else
            printf '✅  No entry found; adding new active line\n'
            printf '\n# RMG-Py added on %s\n%s\n' "$(date +%F)" "$NEW_LINE" >> "$RC"
        fi
    elif [ "$SOURCE_MODE" == conda ]; then
        # conda envs already have the RMG_PY_PATH in PATH, so no need to add it
        add_rmg_hooks "$ENV_NAME"
    else
        # pip install
        $COMMAND_PKG run -n "$ENV_NAME" pip install -e "$RMG_PY_PATH"
        echo "📦 RMG-Py installed via pip in $ENV_NAME"
    fi

    ARC_ENV=arc_env
    if [[ "$SOURCE_MODE" == "conda" ]]; then
        echo "📦 Adding RMG hooks ARC to $ARC_ENV"
        add_rmg_hooks "$ARC_ENV"
    else
        echo "ℹ️  Skipping ARC hooks as ARC is not cloned or not in conda mode."
    fi
fi


echo "✅ RMG-Py installation complete."
