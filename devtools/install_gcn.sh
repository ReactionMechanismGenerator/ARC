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
  if command -v nvcc &>/dev/null; then
    VER=$(nvcc --version | grep -oP "release \K[0-9]+\.[0-9]+")
    CUDA="cudatoolkit=$VER"
    CUDA_VERSION="cu${VER/./}"
  elif command -v nvidia-smi &>/dev/null; then
    VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1 | cut -d. -f1-2)
    CUDA="cudatoolkit=$VER"
    CUDA_VERSION="cu${VER/./}"
  else
    CUDA="cpuonly"
    CUDA_VERSION="cpu"
  fi
fi

echo "→ Installing with $CUDA on platform $CUDA_VERSION"

# ── functions ─────────────────────────────────────────────────────────────
write_hook () {             # env_name  repo_path
    local env="$1" repo="$2"
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
    rm -f "$act" "$deact"

    # activate hook
    cat >"$act" <<EOF
# TS-GCN hook – $(date +%F)
export TSGCN_ROOT="$repo"
case ":\$PYTHONPATH:" in *":\$TSGCN_ROOT:") ;; *) export PYTHONPATH="\$TSGCN_ROOT:\${PYTHONPATH:-}" ;; esac
EOF

    # deactivate hook
    cat >"$deact" <<'EOF'
_strip () { local n=":$1:"; local s=":$2:"; echo "${s//$n/:}" | sed 's/^://;s/:$//'; }
export PYTHONPATH=$(_strip "$TSGCN_ROOT" ":${PYTHONPATH:-}:")
unset TSGCN_ROOT
EOF
    echo "🔗 PYTHONPATH hook refreshed in $env"
}

# ── locate folders ─────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CLONE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$CLONE_ROOT"

# choose backend
if command -v micromamba &>/dev/null; then
    COMMAND_PKG=micromamba
elif command -v mamba &>/dev/null; then
    COMMAND_PKG=mamba
else
    COMMAND_PKG=conda
fi

eval "\$($COMMAND_PKG shell hook --shell=bash)"

# clone/update repo
if [ -d TS-GCN ]; then cd TS-GCN && git fetch && git checkout main && git pull; else git clone https://github.com/ReactionMechanismGenerator/TS-GCN && cd TS-GCN; fi

# 3. PATH vs hooks
if [[ $MODE == path ]]; then
    GCN_LINE="export PYTHONPATH=\\$PYTHONPATH:$(pwd)"
    grep -Fqx "$GCN_LINE" ~/.bashrc || { echo "$GCN_LINE" >> ~/.bashrc; echo "✔️ Added TS-GCN path to ~/.bashrc"; }
fi

# 4. inline env creation & install
if [[ -f environment.yml ]]; then
    echo "Creating/updating ts_gcn environment"
    if $COMMAND_PKG env list | awk '{print $1}' | grep -qx ts_gcn; then
        $COMMAND_PKG env update -n ts_gcn -f environment.yml --prune -y
    else
        $COMMAND_PKG env create -n ts_gcn -f environment.yml -y
    fi

    conda activate ts_gcn
    echo "Installing PyTorch + torchvision with $CUDA"
    $COMMAND_PKG install -n ts_gcn pytorch torchvision $CUDA -c pytorch -y

    TORCH_VER=$(python -c "import torch; print(torch.__version__)" | cut -c1-4)0
    WHEEL_URL="https://pytorch-geometric.com/whl/torch-${TORCH_VER}+${CUDA_VERSION}.html"
    pip install torch-scatter   -f "$WHEEL_URL"
    pip install torch-sparse    -f "$WHEEL_URL"
    pip install torch-cluster   -f "$WHEEL_URL"
    pip install torch-spline-conv -f "$WHEEL_URL"
    pip install torch-geometric
    echo "✅ ts_gcn environment ready"
else
    echo "❌ environment.yml not found."
    exit 1
fi

# 5. write hooks
if [[ $MODE == conda ]]; then
    write_hook ts_gcn "$(pwd)"
    if $COMMAND_PKG env list | awk '{print $1}' | grep -qx arc_env; then
        write_hook arc_env "$(pwd)"
    fi
fi

echo "✅ TS-GCN installation complete."
