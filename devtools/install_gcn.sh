#!/usr/bin/env bash
set -euo pipefail

# ── defaults ───────────────────────────────────────────────────────────────
MODE="path"           # or "conda"
TSGCN_CUDA_REQ=""
FORCE_CPU=false

# ── parse flags ────────────────────────────────────────────────────────────
TEMP=$(getopt -o h --long cuda:,cpu,conda,path,help -- "$@")
eval set -- "$TEMP"
while true; do
  case "$1" in
    --cuda)
      TSGCN_CUDA_REQ="$2"
      shift 2
      ;;
    --cpu)
      FORCE_CPU=true
      shift
      ;;
    --conda)
      MODE="conda"
      shift
      ;;
    --path)
      MODE="path"
      shift
      ;;
    -h|--help)
      cat <<EOF
Usage: $0 [--cuda <9.2|10.1|10.2|11.0>] [--cpu] [--conda|--path] [--help]

  --cuda   request a specific CUDA version (overrides auto-detect)
  --cpu    force a CPU-only install
  --conda  install hooks into conda activate/deactivate
  --path   append TS-GCN to ~/.bashrc
  -h       this help
EOF
      exit 0
      ;;
    --) shift; break ;;
    *)  echo "Invalid flag: $1" >&2; exit 1 ;;
  esac
done

# ── determine CUDA vs CPU ───────────────────────────────────────────────────
if [[ -n "$TSGCN_CUDA_REQ" ]]; then
  # user override
  case "$TSGCN_CUDA_REQ" in
    9.2|10.1|10.2|11.0)
      CUDA="cudatoolkit=${TSGCN_CUDA_REQ}"
      CUDA_VERSION="cu${TSGCN_CUDA_REQ/./}"
      ;;
    *)
      echo "Error: unsupported --cuda version: $TSGCN_CUDA_REQ" >&2
      exit 1
      ;;
  esac

elif $FORCE_CPU; then
  CUDA="cpuonly"
  CUDA_VERSION="cpu"

else
  # auto-detect via nvcc
  if command -v nvcc &>/dev/null; then
    VER=$(nvcc --version | grep -oP "release \K[0-9]+\.[0-9]+")
    echo "Detected nvcc CUDA $VER"
    CUDA="cudatoolkit=$VER"
    CUDA_VERSION="cu${VER/./}"

  # or via nvidia-smi
  elif command -v nvidia-smi &>/dev/null; then
    VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1 | cut -d. -f1-2)
    echo "Detected NVIDIA-driver CUDA $VER"
    CUDA="cudatoolkit=$VER"
    CUDA_VERSION="cu${VER/./}"

  else
    echo "No CUDA toolchain found: defaulting to CPU build"
    CUDA="cpuonly"
    CUDA_VERSION="cpu"
  fi
fi

echo "→ Installing with $CUDA on platform $CUDA_VERSION"

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
  alias conda="$_backend"
  export CUDA CUDA_VERSION

  # map CUDA_VERSION → select index in create_env.sh’s menu
  case "$CUDA_VERSION" in
    cu92)  pick=1 ;;
    cu101) pick=2 ;;
    cu102) pick=3 ;;
    cu110) pick=4 ;;
    cpu)   pick=5 ;;
    *)     pick=5 ;;
  esac

  echo "→ Auto-selecting menu item #$pick for CUDA install"
  # pipe the choice into create_env.sh via make
  printf '%s\n' "$pick" | make conda_env
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
