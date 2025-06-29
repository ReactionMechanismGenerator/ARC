#!/bin/bash -l
set -e

# ── command-line flags ────────────────────────────────────────────────
MODE=path   # {path|conda}

TEMP=$(getopt -o ch --long conda,help -- "$@")
eval set -- "$TEMP"
while true; do
  case "$1" in
      -c|--conda) MODE=conda; shift ;;
      -h|--help)
cat <<EOF
Usage: $0 [--conda]

  -c, --conda  write PYTHONPATH hooks into ts_gcn (and arc_env if it exists)
               instead of modifying ~/.bashrc
EOF
          exit 0 ;;
      --) shift; break ;;
      *)  echo "Internal getopt error"; exit 1 ;;
  esac
done

# ── functions ─────────────────────────────────────────────────────────────
write_hook () {             # env_name  repo_path
    local env="$1" repo="$2"
    # skip if env missing
    $COMMAND_PKG env list | awk '{print $1}' | grep -qx "$env" || return 0

    local prefix
    if [[ $COMMAND_PKG == micromamba ]]; then
        prefix="$(micromamba info --base)/envs/$env"
    else
        prefix="$(conda info --base)/envs/$env"
    fi
    local act="$prefix/etc/conda/activate.d/zzz-tsgcn.sh"
    local deact="$prefix/etc/conda/deactivate.d/zzz-tsgcn.sh"
    mkdir -p "${act%/*}" "${deact%/*}"
    rm -f "$act" "$deact"                  # ensure fresh copy each run

    # --- activate ----------------------------------------------------------
    cat >"$act" <<EOF
# TS-GCN hook – $(date +%F)
export TSGCN_ROOT="$repo"
case ":\$PYTHONPATH:" in *":\$TSGCN_ROOT:"*) ;; \
  *) export PYTHONPATH="\$TSGCN_ROOT:\${PYTHONPATH:-}" ;; esac
EOF

    # --- deactivate --------------------------------------------------------
    cat >"$deact" <<'EOF'
_strip () { local n=":$1:"; local s=":$2:"; echo "${s//$n/:}" | sed 's/^://;s/:$//'; }
export PYTHONPATH=$(_strip "$TSGCN_ROOT" ":${PYTHONPATH:-}:")
unset TSGCN_ROOT
EOF
    echo "🔗 PYTHONPATH hook refreshed in $env"
}

# ── locate folders relative to this script ────────────────────────────────
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ARC_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"      # …/ARC_Mol
CLONE_ROOT="$(cd "$ARC_ROOT/.." && pwd)"      # directory that *contains* ARC_Mol
cd "$CLONE_ROOT"

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
else
    BASE=$(conda info --base)
    # shellcheck source=/dev/null
    . "$BASE/etc/profile.d/conda.sh"
fi


echo ">>> Cloning or updating TS-GCN..."
if [ -d TS-GCN ]; then
    cd TS-GCN
    git fetch origin
    git checkout main
    git pull origin main
else
    git clone https://github.com/ReactionMechanismGenerator/TS-GCN
    cd TS-GCN
fi

# 3. PATH vs hooks ----------------------------------------------------------
if [[ $MODE == path ]]; then
    GCN_LINE="export PYTHONPATH=\$PYTHONPATH:$(pwd)"
    if ! grep -Fqx "$GCN_LINE" ~/.bashrc; then
        echo "$GCN_LINE" >> ~/.bashrc
        echo "✔️ Added TS-GCN path to ~/.bashrc"
    else
        echo "ℹ️ TS-GCN path already exists in ~/.bashrc"
    fi
fi

# ---------------------------------------------------------------------------
# create / update env *here* (unchanged)
if grep -q '^conda_env:' Makefile; then
    echo ">>> Creating GCN conda environment via Makefile"
    # --- pick the fastest Conda frontend just for create_env.sh ---------------
    if command -v micromamba >/dev/null; then
        _backend="micromamba"
    elif command -v mamba >/dev/null; then
        _backend="mamba"
    else
        _backend="conda"          # fallback to classic
    fi
    echo "⚡ Using $_backend for create_env.sh"
    # run make in a subshell so the alias doesn't leak
    (
    alias conda="$_backend" # This alias is only for this subshell - so we can use the fastest Conda frontend
    make conda_env
    )

else
    echo "❌ Makefile target 'conda_env' not found. Please check TS-GCN repo."
    exit 1
fi

# 4. write hooks *after* env exists -----------------------------------------
if [[ $MODE == conda ]]; then
    write_hook ts_gcn "$(pwd)"
    if $COMMAND_PKG env list | awk '{print $1}' | grep -qx arc_env; then
        write_hook arc_env "$(pwd)"
    fi
fi

echo "✅ TS-GCN installation complete."
