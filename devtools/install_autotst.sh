#!/bin/bash -l
set -e

MODE=path   # {path|conda}

TEMP=$(getopt -o ch --long conda,help -- "$@")
eval set -- "$TEMP"
while true; do
  case "$1" in
      -c|--conda)  MODE=conda; shift ;;
      -h|--help)
cat <<EOF
Usage: $0 [--conda]

  -c, --conda   write PYTHONPATH hooks into tst_env (and arc_env if it exists)
                instead of modifying ~/.bashrc
EOF
          exit 0 ;;
      --) shift; break ;;
      *)  echo "Internal getopt error"; exit 1 ;;
  esac
done

# ── functions ───────────────────────────────────────────────────────────────
# This function writes the AutoTST hook files for a given conda environment.
# It checks if the environment exists, creates the necessary directories,
# and writes the activation and deactivation scripts to set/unset the
# AUTOTST_ROOT environment variable and modify PYTHONPATH accordingly.
# Usage: write_hook <env_name> <repo_path>
# Example: write_hook tst_env "$(pwd)"
# where "$(pwd)" is the path to the AutoTST repository.
write_hook () {
    local env="$1" repo_path="$2"            # repo_path="$(pwd)" in AutoTST
    $COMMAND_PKG env list | awk '{print $1}' | grep -qx "$env" || return 0

    # env prefix
    if [[ $COMMAND_PKG == micromamba ]]; then
        prefix="$(micromamba info --base)/envs/$env"
    else
        prefix="$(conda info --base)/envs/$env"
    fi

    local act="$prefix/etc/conda/activate.d/zzz-autotst.sh"
    local deact="$prefix/etc/conda/deactivate.d/zzz-autotst.sh"
    mkdir -p "${act%/*}" "${deact%/*}"

    # --- delete any previous hook files ------------------------------------
    rm -f "$act" "$deact"

    # --- activation --------------------------------------------------------
    cat >"$act" <<EOF
# AutoTST hook – $(date +%F)
export AUTOTST_ROOT="$repo_path"
case ":\$PYTHONPATH:" in *":\$AUTOTST_ROOT:"*) ;; \
  *) export PYTHONPATH="\$AUTOTST_ROOT:\${PYTHONPATH:-}" ;; esac
EOF

    # --- de-activation -----------------------------------------------------
    cat >"$deact" <<'EOF'
_strip () { local n=":$1:"; local s=":$2:"; echo "${s//$n/:}" | sed 's/^://;s/:$//'; }
export PYTHONPATH=$(_strip "$AUTOTST_ROOT" ":${PYTHONPATH:-}:")
unset AUTOTST_ROOT
EOF
    echo "🔗  AutoTST hook refreshed in $env"
}



# ── locate folders relative to this script ────────────────────────────────
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if ARC_ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null); then
    :  # got it
else
    ARC_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
fi
CLONE_ROOT="$(dirname "$ARC_ROOT")"
cd "$CLONE_ROOT" || exit 1
echo "📂  Clone root: $CLONE_ROOT"

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
elif [ "$COMMAND_PKG" = "mamba" ] || [ "$COMMAND_PKG" = "conda" ]; then
    BASE=$(conda info --base)
    source "$BASE/etc/profile.d/conda.sh"
fi

echo ">>> Cloning or updating AutoTST..."
if [ -d AutoTST ]; then
    cd AutoTST
    CURRENT_BRANCH=$(git symbolic-ref --short HEAD)
    if [ "$CURRENT_BRANCH" = "main" ]; then
        git fetch origin
        git pull origin main
    else
        echo "⚠️ AutoTST is on branch '$CURRENT_BRANCH'. Skipping update."
    fi
else
    git clone https://github.com/ReactionMechanismGenerator/AutoTST
    cd AutoTST
fi

if [[ $MODE == "path" ]]; then

    AUTO_PATH_LINE="export PYTHONPATH=\"\$PYTHONPATH:$(pwd)\""
    if ! grep -Fqx "$AUTO_PATH_LINE" ~/.bashrc; then
        echo "$AUTO_PATH_LINE" >> ~/.bashrc
        echo "✔️ Added AutoTST path to ~/.bashrc"
    else
        echo "ℹ️ AutoTST path already exists in ~/.bashrc"
    fi
elif [[ $MODE == "conda" ]]; then
    write_hook tst_env   "$(pwd)"
    # add to arc_env if that env exists
    if $COMMAND_PKG env list | awk '{print $1}' | grep -qx arc_env; then
        write_hook arc_env "$(pwd)"
    fi
fi

if $COMMAND_PKG env list | awk '{print $1}' | sed 's/^\*//' | grep -Fxq 'tst_env'; then
    echo ">>> Updating existing environment tst_env..."
    if [ "$COMMAND_PKG" != "conda" ]; then
        $COMMAND_PKG env update -n tst_env -f environment.yml -y
    else
        $COMMAND_PKG env update -n tst_env -f environment.yml -y
    fi
else
    echo ">>> Creating new environment tst_env..."
    if [ "$COMMAND_PKG" != "conda" ]; then
        $COMMAND_PKG env create -n tst_env -f environment.yml -y
    else
        $COMMAND_PKG env create -n tst_env -f environment.yml
    fi
fi

echo ">>> Checking for pyyaml. Will be installing extra dependencies into tst_env if not there..."
if $COMMAND_PKG list -n tst_env pyyaml | grep -Eq '^\*?\s*pyyaml\s'; then
    echo "ℹ️  PyYAML already in tst_env – skipping install."
else
    echo "➜  Installing PyYAML into tst_env…"
    $COMMAND_PKG install -n tst_env -c conda-forge pyyaml -y
fi

echo "✅ Done installing AutoTST."
