#!/usr/bin/env bash
set -euo pipefail

# ── defaults ───────────────────────────────────────────────────────────────
RITS_REPO_URL="https://github.com/isayevlab/RitS.git"
RITS_ENV_NAME="rits_env"
FORCE_CPU=false
RITS_PATH=""
SKIP_CKPT=false
CUDA_VARIANT=""        # one of: cpu, cu118, cu121, cu124, cu126 (empty → autodetect)
TORCH_VERSION="2.7.0"  # must match RitS's pinned torch version

# Pretrained checkpoint mirror (Dana Research Group, Zenodo)
# Google Drive checkpoint file source: https://drive.google.com/drive/folders/1DD2hmWx3E1klM3Ljon5r4gdquGoN_4v6
# Source paper: https://github.com/isayevlab/RitS, 10.26434/chemrxiv.15001681/v1
# Mirror DOI : https://doi.org/10.5281/zenodo.19474153
RITS_CKPT_URL="https://zenodo.org/records/19474153/files/rits.ckpt?download=1"
RITS_CKPT_MD5="884121fcf7a5bfcfb826b7d5e28d379a"

# ── parse flags ────────────────────────────────────────────────────────────
TEMP=$(getopt -o h --long cpu,cuda:,path:,no-ckpt,help -- "$@")
eval set -- "$TEMP"
while true; do
  case "$1" in
    --cpu)
      FORCE_CPU=true
      shift
      ;;
    --cuda)
      CUDA_VARIANT="$2"
      shift 2
      ;;
    --path)
      RITS_PATH="$2"
      shift 2
      ;;
    --no-ckpt)
      SKIP_CKPT=true
      shift
      ;;
    -h|--help)
      cat <<EOF
Usage: $0 [--cpu] [--cuda <variant>] [--path <existing RitS checkout>] [--no-ckpt] [--help]

  --cpu             force a CPU-only PyTorch install (shortcut for --cuda cpu)
  --cuda <variant>  pick a specific PyG wheel variant: cpu, cu118, cu121, cu124, cu126
                    (default: autodetect via nvcc / nvidia-smi)
  --path <dir>      use an existing RitS checkout instead of cloning
  --no-ckpt         skip the pretrained checkpoint download (offline installs)
  -h                this help

By default the script clones (or updates) RitS as a sibling of the ARC repo,
creates the '${RITS_ENV_NAME}' conda env with python=3.10, autodetects the
host CUDA version, installs torch=${TORCH_VERSION} + matching PyTorch Geometric
companion wheels (torch-scatter / torch-sparse / torch-cluster /
torch-spline-conv / pyg-lib) from PyG's wheel index, runs 'pip install -e .'
so that 'import megalodon' works inside the env, and downloads + verifies the
pretrained 'rits.ckpt' from Zenodo
(${RITS_CKPT_URL%%\?*}).

No training is required — RitS ships pretrained weights.
EOF
      exit 0
      ;;
    --) shift; break ;;
    *) echo "Invalid flag: $1" >&2; exit 1 ;;
  esac
done

# ── pick a CUDA variant for the PyG wheels ───────────────────────────────
# PyG publishes wheels for torch ${TORCH_VERSION} against these variants only:
SUPPORTED_VARIANTS=(cpu cu118 cu121 cu124 cu126)

map_cuda_to_variant() {  # X.Y → cu118|cu121|cu124|cu126|cpu
  local ver="$1"
  local major minor
  major=${ver%%.*}
  minor=${ver#*.}
  minor=${minor%%.*}
  if [[ -z "$major" || -z "$minor" ]]; then echo cpu; return; fi
  if   (( major > 12 )) || { (( major == 12 )) && (( minor >= 6 )); }; then echo cu126
  elif (( major == 12 )) && (( minor >= 4 )); then                          echo cu124
  elif (( major == 12 )) && (( minor >= 1 )); then                          echo cu121
  elif { (( major == 12 )) && (( minor == 0 )); } || \
       { (( major == 11 )) && (( minor >= 8 )); }; then                     echo cu118
  else                                                                       echo cpu
  fi
}

if [[ -n "$CUDA_VARIANT" ]]; then
  if $FORCE_CPU && [[ "$CUDA_VARIANT" != cpu ]]; then
    echo "❌  --cpu and --cuda $CUDA_VARIANT are contradictory" >&2
    exit 1
  fi
  # validate against the supported set
  if ! printf '%s\n' "${SUPPORTED_VARIANTS[@]}" | grep -qx "$CUDA_VARIANT"; then
    echo "❌  Unsupported --cuda variant: $CUDA_VARIANT" >&2
    echo "    Supported: ${SUPPORTED_VARIANTS[*]}" >&2
    exit 1
  fi
elif $FORCE_CPU; then
  CUDA_VARIANT="cpu"
elif command -v nvcc &>/dev/null; then
  VER=$(nvcc --version | grep -oP "release \K[0-9]+\.[0-9]+" | head -n1)
  CUDA_VARIANT=$(map_cuda_to_variant "$VER")
  echo "🔍  nvcc reports CUDA $VER → using PyG variant '$CUDA_VARIANT'"
elif command -v nvidia-smi &>/dev/null; then
  # The 'CUDA Version' field in nvidia-smi is the *driver's max supported* CUDA, which is the
  # right ceiling for binary wheel compatibility (not driver_version, which is a different number).
  VER=$(nvidia-smi 2>/dev/null | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" | head -n1 || true)
  if [[ -n "$VER" ]]; then
    CUDA_VARIANT=$(map_cuda_to_variant "$VER")
    echo "🔍  nvidia-smi reports max CUDA $VER → using PyG variant '$CUDA_VARIANT'"
  else
    CUDA_VARIANT="cpu"
    echo "🔍  Could not parse CUDA version from nvidia-smi → falling back to CPU"
  fi
else
  CUDA_VARIANT="cpu"
  echo "🔍  No nvcc / nvidia-smi found → falling back to CPU"
fi
echo "→  PyG wheel variant: $CUDA_VARIANT"

# ── locate ARC repo and the sibling clone root ────────────────────────────
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if ARC_ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null); then
    :
else
    ARC_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
fi
CLONE_ROOT="$(dirname "$ARC_ROOT")"
echo "📂  ARC root  : $ARC_ROOT"
echo "📂  Clone root: $CLONE_ROOT"

# ── pick a conda frontend ─────────────────────────────────────────────────
if command -v micromamba &>/dev/null; then
  COMMAND_PKG=micromamba
elif command -v mamba &>/dev/null; then
  COMMAND_PKG=mamba
elif command -v conda &>/dev/null; then
  COMMAND_PKG=conda
else
  echo "❌  No micromamba/mamba/conda found in PATH" >&2
  exit 1
fi
echo "✔️  Using $COMMAND_PKG"

# Initialize shell integration so 'activate' works in this script
if [[ $COMMAND_PKG == micromamba ]]; then
    eval "$(micromamba shell hook --shell=bash)"
else
    BASE=$(conda info --base)
    source "$BASE/etc/profile.d/conda.sh"
fi

# ── clone or update RitS ──────────────────────────────────────────────────
if [[ -n "$RITS_PATH" ]]; then
  if [[ ! -d "$RITS_PATH" ]]; then
    echo "❌  --path was given but directory does not exist: $RITS_PATH" >&2
    exit 1
  fi
  RITS_DIR="$(cd "$RITS_PATH" && pwd)"
  echo "📂  Using existing RitS checkout at: $RITS_DIR"
else
  RITS_DIR="$CLONE_ROOT/RitS"
  if [[ -d "$RITS_DIR/.git" ]]; then
    echo "🔄  Updating existing RitS clone at $RITS_DIR"
    git -C "$RITS_DIR" fetch origin
    git -C "$RITS_DIR" pull --ff-only || echo "⚠️  Could not fast-forward; leaving working tree as-is."
  else
    echo "⬇️  Cloning RitS into $RITS_DIR"
    git clone "$RITS_REPO_URL" "$RITS_DIR"
  fi
fi

# ── create / update the rits_env conda environment ───────────────────────
if $COMMAND_PKG env list | awk '{print $1}' | grep -qx "$RITS_ENV_NAME"; then
  echo "♻️  '$RITS_ENV_NAME' already exists — updating in place."
else
  echo "🆕  Creating '$RITS_ENV_NAME' (python=3.10)"
  $COMMAND_PKG create -n "$RITS_ENV_NAME" -c conda-forge python=3.10 -y
fi

set +u; $COMMAND_PKG activate "$RITS_ENV_NAME"; set -u

# RDKit & OpenBabel are far smoother via conda-forge than pip
echo "📦  Installing rdkit + openbabel from conda-forge"
$COMMAND_PKG install -n "$RITS_ENV_NAME" -c conda-forge -y \
  "rdkit=2025.3.2" openbabel

# Install PyTorch + PyTorch Geometric companion wheels for the chosen variant.
# We deliberately do NOT use RitS's requirements.txt because it pins +pt27cu126
# specifically — we install the variant-matched companion wheels instead so the
# install works on CPU runners and on GPUs with CUDA != 12.6.
python -m pip install --upgrade pip

if [[ "$CUDA_VARIANT" == "cpu" ]]; then
  TORCH_INDEX="https://download.pytorch.org/whl/cpu"
else
  TORCH_INDEX="https://download.pytorch.org/whl/${CUDA_VARIANT}"
fi
PYG_WHEELS="https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_VARIANT}.html"

echo "🚀  Installing torch==${TORCH_VERSION} (${CUDA_VARIANT}) from $TORCH_INDEX"
python -m pip install "torch==${TORCH_VERSION}" --index-url "$TORCH_INDEX"

echo "🧮  Installing PyG companion wheels from $PYG_WHEELS"
# --only-binary :all: forces wheels, never source builds (those would need a CUDA toolkit)
python -m pip install \
  pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
  --only-binary :all: -f "$PYG_WHEELS"

echo "📦  Installing pure-Python megalodon dependencies from PyPI"
python -m pip install \
  "torch_geometric==2.6.1" \
  "hydra-core==1.3.2" \
  "lightning==2.5.1.post0" \
  "einops==0.8.1" \
  "wandb==0.19.11" \
  "pandas==2.2.3" \
  "tqdm==4.67.1"

# Editable install of the megalodon package (this is what puts 'import megalodon' on path)
echo "🧷  pip install -e . (megalodon, src layout)"
python -m pip install -e "$RITS_DIR"

# Sanity check — import megalodon AND the PyG companions, since a successful pip
# install does not guarantee the .so files actually load against the host's CUDA.
echo "🔍  Verifying inference stack inside $RITS_ENV_NAME"
python - <<'PYEOF'
import importlib, sys
mods = ["torch", "torch_geometric", "torch_scatter", "torch_sparse",
        "torch_cluster", "torch_spline_conv", "megalodon"]
for m in mods:
    try:
        mod = importlib.import_module(m)
        ver = getattr(mod, "__version__", "?")
        print(f"  ✔️  {m:<22} {ver}")
    except Exception as e:
        print(f"  ❌  {m:<22} FAILED: {e}", file=sys.stderr)
        sys.exit(1)
import torch
print(f"  ℹ️  torch.cuda.is_available() = {torch.cuda.is_available()}")
PYEOF

set +u; $COMMAND_PKG deactivate; set -u

# ── download + verify pretrained checkpoint ──────────────────────────────
CKPT_DIR="$RITS_DIR/data"
CKPT_PATH="$CKPT_DIR/rits.ckpt"

verify_md5() {  # path expected_md5
  local path="$1" expected="$2"
  local actual
  if command -v md5sum &>/dev/null; then
    actual=$(md5sum "$path" | awk '{print $1}')
  elif command -v md5 &>/dev/null; then
    actual=$(md5 -q "$path")
  else
    echo "❌  Neither md5sum nor md5 found in PATH; cannot verify checkpoint." >&2
    return 2
  fi
  if [[ "$actual" != "$expected" ]]; then
    echo "❌  Checksum mismatch for $path" >&2
    echo "       expected: $expected" >&2
    echo "       actual  : $actual"   >&2
    return 1
  fi
  return 0
}

if $SKIP_CKPT; then
  echo "ℹ️  --no-ckpt set, skipping checkpoint download."
elif [[ -f "$CKPT_PATH" ]]; then
  echo "📦  Existing checkpoint found at $CKPT_PATH — verifying MD5..."
  if verify_md5 "$CKPT_PATH" "$RITS_CKPT_MD5"; then
    echo "✔️  Checkpoint MD5 OK ($RITS_CKPT_MD5)"
  else
    echo "❌  Existing checkpoint does not match the expected MD5." >&2
    echo "    Refusing to overwrite — move it aside or delete it and re-run." >&2
    exit 1
  fi
else
  mkdir -p "$CKPT_DIR"
  if ! command -v curl &>/dev/null; then
    echo "❌  curl is required to download the RitS checkpoint." >&2
    exit 1
  fi
  echo "⬇️  Downloading rits.ckpt (~364 MB) from Zenodo:"
  echo "       $RITS_CKPT_URL"
  TMP_CKPT="$(mktemp "${CKPT_DIR}/rits.ckpt.XXXXXX")"
  if ! curl -fL --retry 3 --retry-delay 5 -o "$TMP_CKPT" "$RITS_CKPT_URL"; then
    rm -f "$TMP_CKPT"
    echo "❌  Download failed. Re-run the install, or pass --no-ckpt to skip." >&2
    exit 1
  fi
  if verify_md5 "$TMP_CKPT" "$RITS_CKPT_MD5"; then
    mv "$TMP_CKPT" "$CKPT_PATH"
    echo "✔️  Checkpoint verified and saved to $CKPT_PATH"
  else
    rm -f "$TMP_CKPT"
    echo "❌  Downloaded checkpoint failed MD5 verification — aborting." >&2
    exit 1
  fi
fi

# ── final notes ───────────────────────────────────────────────────────────
echo ""
echo "✅  RitS installation complete."
echo "    Repo : $RITS_DIR"
echo "    Env  : $RITS_ENV_NAME"
echo "    Ckpt : $([[ -f $CKPT_PATH ]] && echo $CKPT_PATH || echo '(not installed)')"
echo ""
echo "    Mirror DOI : https://doi.org/10.5281/zenodo.19474153"
echo "    Source     : https://github.com/isayevlab/RitS"
