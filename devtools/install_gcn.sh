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
    *) echo "Invalid flag: $1" >&2; exit 1 ;;
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
write_hook() {  # env_name  repo_path
  local env="$1" repo="$2"
  $COMMAND_PKG env list | awk '{print $1}' | grep -qx "$env" || return 0
  if [[ $COMMAND_PKG == micromamba ]]; then
    prefix="$(micromamba info --base)/envs/$env"
  else
    prefix="$(conda info --base)/envs/$env"
  fi
  act_dir="$prefix/etc/conda/activate.d"
  deact_dir="$prefix/etc/conda/deactivate.d"
  mkdir -p "$act_dir" "$deact_dir"
  act="$act_dir/zzz-tsgcn.sh"
  deact="$deact_dir/zzz-tsgcn.sh"
  rm -f "$act" "$deact"

  # --- activation hook -----------------------------------------------------
  cat <<ACTHOOK >"$act"
# TS-GCN hook – $(date +%F)
export TSGCN_ROOT="$repo"
case ":\$PYTHONPATH:" in
  *":\$TSGCN_ROOT:") ;; \
  *) export PYTHONPATH="\$TSGCN_ROOT:\${PYTHONPATH:-}" ;; 
esac
ACTHOOK

  # --- deactivation hook ---------------------------------------------------
  cat <<'DEACTHOOK' >"$deact"
_strip() { local n=":$1:"; local s=":$2:"; echo "${s//$n/:}" | sed 's/^://;s/:$//'; }
export PYTHONPATH=$(_strip "$TSGCN_ROOT" ":${PYTHONPATH:-}:")
unset TSGCN_ROOT
DEACTHOOK

  echo "🔗 PYTHONPATH hook refreshed in $env"
}

# ── locate and enter repo root ────────────────────────────────────────────
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if ARC_ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null); then
    :  # got it
else
    ARC_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
fi
CLONE_ROOT="$(dirname "$ARC_ROOT")"
cd "$CLONE_ROOT" || exit 1
echo "📂  Clone root: $CLONE_ROOT"

# ── choose fastest Conda frontend and init shell ---------------------------
if command -v micromamba &>/dev/null; then
  COMMAND_PKG=micromamba
elif command -v mamba &>/dev/null; then
  COMMAND_PKG=mamba
else
  COMMAND_PKG=conda
fi

# Shell hook
if [[ $COMMAND_PKG == micromamba ]]; then
    eval "$(micromamba shell hook --shell bash)"
else
    BASE=$(conda info --base)
    source "$BASE/etc/profile.d/conda.sh"
    eval "$($COMMAND_PKG shell hook --shell bash)"
fi

# ── clone or update TS-GCN ------------------------------------------------
if [[ -d TS-GCN ]]; then
  cd TS-GCN && git fetch origin && git checkout main && git pull
else
  git clone https://github.com/ReactionMechanismGenerator/TS-GCN && cd TS-GCN
fi

# ── PATH vs. hooks mode ---------------------------------------------------
if [[ $MODE == path ]]; then
  GCN_LINE="export PYTHONPATH=\$PYTHONPATH:$(pwd)"
  grep -Fqx "$GCN_LINE" ~/.bashrc || {
    echo "$GCN_LINE" >> ~/.bashrc
    echo "✔️ Added TS-GCN path to ~/.bashrc"
  }
fi

CORE_PKGS=(
  python=3.8
  numpy
  matplotlib-base
  pandas
  scikit-learn
  hyperopt
  h5py
  psutil
  rdkit
  scipy
  openbabel
  py3dmol
  glew
  pymol-open-source
  pyyaml
  seaborn
  tqdm
  xlrd
  xlwt
)


# ── inline env creation & unified PyTorch install --------------------------
if $COMMAND_PKG env list | awk '{print $1}' | grep -qx ts_gcn; then
  $COMMAND_PKG install -n ts_gcn \
    -c schrodinger -c conda-forge \
    --channel-priority flexible \
    "${CORE_PKGS[@]}" \
    --yes
else
  $COMMAND_PKG create -n ts_gcn \
    -c schrodinger -c conda-forge \
    --channel-priority flexible \
    "${CORE_PKGS[@]}" \
    --yes
fi

# 2) pip‐install exactly the CPU or CUDA wheels (no ROCm on that index)
PIP_RUN=("$COMMAND_PKG" run -n ts_gcn)
WHEEL=https://download.pytorch.org/whl/torch_stable.html
if [[ $CUDA_VERSION == cpu ]]; then
  "${PIP_RUN[@]}" pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f $WHEEL
else
  "${PIP_RUN[@]}" pip install torch==1.7.1+${CUDA_VERSION} \
                              torchvision==0.8.2+${CUDA_VERSION} \
                              torchaudio==0.7.2+${CUDA_VERSION} \
    -f $WHEEL
fi
# for PyG wheels use the official PyG index—with a real '+' in the URL
TORCH_VER=1.7.1
WHEEL_URL="https://pytorch-geometric.com/whl/torch-${TORCH_VER}+${CUDA_VERSION}.html"

# install ONLY the prebuilt binaries, never fall back to source
"${PIP_RUN[@]}" pip install torch-scatter     -f "$WHEEL_URL" --only-binary torch-scatter
"${PIP_RUN[@]}" pip install torch-sparse      -f "$WHEEL_URL" --only-binary torch-sparse
"${PIP_RUN[@]}" pip install torch-cluster     -f "$WHEEL_URL" --only-binary torch-cluster
"${PIP_RUN[@]}" pip install torch-spline-conv -f "$WHEEL_URL" --only-binary torch-spline-conv

# finally the meta‐package (this one can install from PyPI)
"${PIP_RUN[@]}" pip install torch-geometric
echo "✅ ts_gcn environment ready"

# ── write hooks into conda envs if required -------------------------------
if [[ $MODE == conda ]]; then
  write_hook ts_gcn "$(pwd)"
  if $COMMAND_PKG env list | awk '{print $1}' | grep -qx arc_env; then
    write_hook arc_env "$(pwd)"
  fi
fi

echo "✅ TS-GCN installation complete."
