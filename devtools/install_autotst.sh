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

AutoTST is always registered in tst_env through an autotst.pth file in its
site-packages, so it is importable regardless of the caller's PYTHONPATH
(ARC runs the AutoTST subprocess with PYTHONPATH removed). ~/.bashrc is not
modified.

  -c, --conda   additionally write PYTHONPATH activation hooks into tst_env
                (and arc_env if it exists)
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



# Registers AutoTST in an environment through a .pth file in its site-packages, which
# (unlike PYTHONPATH) cannot be shadowed or stripped by the caller's environment.
# Usage: write_pth <env_name> <repo_path>
write_pth () {
    local env="$1" repo_path="$2" site_dir
    $COMMAND_PKG env list | awk '{print $1}' | sed 's/^\*//' | grep -Fxq "$env" || return 0
    site_dir="$($COMMAND_PKG run -n "$env" python -c 'import site; print(site.getsitepackages()[0])')"
    write_pth_file "$site_dir" "$repo_path"
    echo "🔗  AutoTST registered in $env via $site_dir/autotst.pth"
}

# Writes <site_dir>/autotst.pth containing <repo_path>.
write_pth_file () {
    local site_dir="$1" repo_path="$2"
    [[ -d "$site_dir" ]] || { echo "❌ site-packages not found: $site_dir"; return 1; }
    printf '%s\n' "$repo_path" > "$site_dir/autotst.pth"
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

AUTOTST_DIR="$(pwd)"

if $COMMAND_PKG env list | awk '{print $1}' | sed 's/^\*//' | grep -Fxq 'tst_env'; then
    echo ">>> Updating existing environment tst_env..."
    if [ "$COMMAND_PKG" != "conda" ]; then
        $COMMAND_PKG env update -n tst_env -f environment.yml
    else
        $COMMAND_PKG env update -n tst_env -f environment.yml
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

# Register AutoTST only now that tst_env exists.
write_pth tst_env "$AUTOTST_DIR"
if [[ $MODE == "conda" ]]; then
    write_hook tst_env "$AUTOTST_DIR"
    # add to arc_env if that env exists
    if $COMMAND_PKG env list | awk '{print $1}' | grep -qx arc_env; then
        write_hook arc_env "$AUTOTST_DIR"
    fi
fi

if grep -Fqx "export PYTHONPATH=\"\$PYTHONPATH:$AUTOTST_DIR\"" ~/.bashrc 2>/dev/null; then
    echo "ℹ️ ~/.bashrc still has an AutoTST PYTHONPATH line from an earlier install. It is no longer needed and can be removed."
fi

echo "✅ Done installing AutoTST."
