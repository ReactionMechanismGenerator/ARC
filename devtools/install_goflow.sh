#!/usr/bin/env bash
set -euo pipefail

# ── defaults ───────────────────────────────────────────────────────────────
GOFLOW_REPO_URL="https://github.com/heid-lab/goflow_lean.git"
GOFLOW_ENV_NAME="goflow_env"
FORCE_CPU=false
GOFLOW_PATH=""
SKIP_CKPT_CHECK=false
USER_CKPT_PATH=""
USER_FEAT_DICT_PATH=""
CUDA_VARIANT=""        # one of: cpu, cu118, cu121, cu124, cu126 (empty → autodetect)
TORCH_VERSION="2.6.0"  # must match GoFlow's pinned torch version (see goflow_lean README)
TORCHVISION_VERSION="0.21.0"
SKIP_CKPT=false

# Pretrained checkpoint mirror (Dana Research Group, Zenodo)
# Self-trained reproduction of the paper's epoch_337.ckpt with seed=1, same
# hyperparameters; see goflow/README.md in DanaResearchGroup/training_checkpoints
# for full provenance.
# Source paper : Galustian et al., Digital Discovery 2025, 10.1039/D5DD00283D
# Source repo  : https://github.com/heid-lab/goflow_lean
GOFLOW_CKPT_URL="https://zenodo.org/records/20073635/files/epoch_316.ckpt?download=1"
GOFLOW_CKPT_SHA256="f0db9762687e4c9e5ce8af54c62e77b087bbacdb48374fa5fb6c6ecda16f13b8"

# Pretrained checkpoint policy:
#
#   GoFlow's published `goflow_lean` repo stores `data/RDB7/epoch_337.ckpt` and
#   `data/RDB7/feat_dict_organic.pkl` as Git LFS pointers / placeholders (45 B
#   and 387 B respectively at the time of writing). The 387-byte
#   feat_dict_organic.pkl IS a real (small) pickle — feat_dicts are inherently
#   tiny — but the 45-byte epoch_337.ckpt is a placeholder, NOT usable.
#
#   This installer therefore:
#     1. accepts a user-supplied ckpt via --ckpt or ARC_GOFLOW_CKPT
#     2. otherwise downloads from Zenodo + verifies SHA-256
#     3. validates the in-repo feat_dict_organic.pkl by size + pickle.load
#     4. accepts --no-ckpt-check to skip download/validation entirely (CI
#        smoke installs only — adapter will skip cleanly until real ckpt
#        is available)

# ── parse flags ────────────────────────────────────────────────────────────
TEMP=$(getopt -o h --long cpu,cuda:,path:,no-ckpt-check,no-ckpt,ckpt:,feat-dict:,help -- "$@")
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
      GOFLOW_PATH="$2"
      shift 2
      ;;
    --no-ckpt-check)
      SKIP_CKPT_CHECK=true
      shift
      ;;
    --no-ckpt)
      SKIP_CKPT=true
      shift
      ;;
    --ckpt)
      USER_CKPT_PATH="$2"
      shift 2
      ;;
    --feat-dict)
      USER_FEAT_DICT_PATH="$2"
      shift 2
      ;;
    -h|--help)
      cat <<EOF
Usage: $0 [--cpu] [--cuda <variant>] [--path <existing goflow_lean checkout>]
          [--ckpt <ckpt-path>] [--feat-dict <pkl-path>]
          [--no-ckpt] [--no-ckpt-check] [--help]

  --cpu                force a CPU-only PyTorch install (shortcut for --cuda cpu)
  --cuda <variant>     pick a specific PyG wheel variant: cpu, cu118, cu121, cu124, cu126
                       (default: autodetect via nvcc / nvidia-smi)
  --path <dir>         use an existing goflow_lean checkout instead of cloning
  --ckpt <path>        copy this file into <repo>/data/RDB7/epoch_316.ckpt
                       (overrides ARC_GOFLOW_CKPT and the Zenodo download)
  --feat-dict <path>   copy this file into <repo>/data/RDB7/feat_dict_organic.pkl
                       (overrides ARC_GOFLOW_FEAT_DICT)
  --no-ckpt            skip the Zenodo checkpoint download (offline installs)
  --no-ckpt-check      build the env without validating ckpt+feat_dict — useful for
                       CI smoke installs. The ARC adapter will skip GoFlow at runtime
                       until real artifacts are placed at the expected paths.
  -h                   this help

By default the script clones (or updates) goflow_lean as a sibling of the ARC
repo, creates the '${GOFLOW_ENV_NAME}' conda env with python=3.11, autodetects
the host CUDA version, installs torch=${TORCH_VERSION} + matching PyTorch
Geometric companion wheels (torch-scatter / torch-sparse / torch-cluster /
torch-spline-conv / pyg-lib / torch-geometric) from PyG's wheel index, runs
'pip install -e .' so that 'import goflow' works inside the env, and validates
the checkpoint + feature-dictionary artifacts in place.

By default the GoFlow checkpoint is downloaded from Zenodo
(${GOFLOW_CKPT_URL%%\?*}) and SHA-256-verified. Override with --ckpt or
ARC_GOFLOW_CKPT to use a local file, --no-ckpt to skip the download
(offline installs), or --no-ckpt-check to skip download AND validation
entirely (CI smoke).

Citation:
  Galustian, L. et al. GoFlow: efficient transition state geometry prediction
  with flow matching and E(3)-equivariant neural networks.
  Digital Discovery 2025. https://doi.org/10.1039/D5DD00283D
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
  if ! printf '%s\n' "${SUPPORTED_VARIANTS[@]}" | grep -qx "$CUDA_VARIANT"; then
    echo "❌  Unsupported --cuda variant: $CUDA_VARIANT" >&2
    echo "    Supported: ${SUPPORTED_VARIANTS[*]}" >&2
    exit 1
  fi
elif $FORCE_CPU; then
  CUDA_VARIANT="cpu"
elif command -v nvcc &>/dev/null; then
  # POSIX-friendly: works with both GNU and BSD grep (grep -oP / \K is GNU-only).
  VER=$(nvcc --version | sed -nE 's/.*release ([0-9]+\.[0-9]+).*/\1/p' | head -n1)
  CUDA_VARIANT=$(map_cuda_to_variant "$VER")
  echo "🔍  nvcc reports CUDA $VER → using PyG variant '$CUDA_VARIANT'"
elif command -v nvidia-smi &>/dev/null; then
  VER=$(nvidia-smi 2>/dev/null | sed -nE 's/.*CUDA Version: ([0-9]+\.[0-9]+).*/\1/p' | head -n1 || true)
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

if [[ $COMMAND_PKG == micromamba ]]; then
    eval "$(micromamba shell hook --shell=bash)"
else
    BASE=$(conda info --base)
    source "$BASE/etc/profile.d/conda.sh"
fi

# ── clone or update goflow_lean ───────────────────────────────────────────
if [[ -n "$GOFLOW_PATH" ]]; then
  if [[ ! -d "$GOFLOW_PATH" ]]; then
    echo "❌  --path was given but directory does not exist: $GOFLOW_PATH" >&2
    exit 1
  fi
  GOFLOW_DIR="$(cd "$GOFLOW_PATH" && pwd)"
  echo "📂  Using existing goflow_lean checkout at: $GOFLOW_DIR"
else
  GOFLOW_DIR="$CLONE_ROOT/goflow_lean"
  if [[ -d "$GOFLOW_DIR/.git" ]]; then
    echo "🔄  Updating existing goflow_lean clone at $GOFLOW_DIR"
    git -C "$GOFLOW_DIR" fetch origin
    git -C "$GOFLOW_DIR" pull --ff-only || echo "⚠️  Could not fast-forward; leaving working tree as-is."
  else
    echo "⬇️  Cloning goflow_lean into $GOFLOW_DIR"
    git clone "$GOFLOW_REPO_URL" "$GOFLOW_DIR"
  fi
fi

GOFLOW_COMMIT="$(git -C "$GOFLOW_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "🔖  goflow_lean commit: $GOFLOW_COMMIT"

# ── create / update the goflow_env conda environment ─────────────────────
# We deliberately do NOT use 'conda env create -f environment.yml' because the
# upstream env name is 'goflow' and we want our ARC-managed env at 'goflow_env'.
if $COMMAND_PKG env list | awk '{print $1}' | grep -qx "$GOFLOW_ENV_NAME"; then
  echo "♻️  '$GOFLOW_ENV_NAME' already exists — updating in place."
else
  echo "🆕  Creating '$GOFLOW_ENV_NAME' (python=3.11)"
  $COMMAND_PKG create -n "$GOFLOW_ENV_NAME" -c conda-forge python=3.11 -y
fi

set +u; $COMMAND_PKG activate "$GOFLOW_ENV_NAME"; set -u

# RDKit / OpenBabel / ASE are far smoother via conda-forge than pip.
echo "📦  Installing rdkit + ase + numpy/pandas/tqdm/ipykernel from conda-forge"
$COMMAND_PKG install -n "$GOFLOW_ENV_NAME" -c conda-forge -y \
  rdkit ase numpy pandas tqdm ipykernel

python -m pip install --upgrade pip

if [[ "$CUDA_VARIANT" == "cpu" ]]; then
  TORCH_INDEX="https://download.pytorch.org/whl/cpu"
else
  TORCH_INDEX="https://download.pytorch.org/whl/${CUDA_VARIANT}"
fi
PYG_WHEELS="https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_VARIANT}.html"

echo "🚀  Installing torch==${TORCH_VERSION} (${CUDA_VARIANT}) from $TORCH_INDEX"
python -m pip install \
  "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
  --index-url "$TORCH_INDEX"

echo "🧮  Installing PyG companion wheels from $PYG_WHEELS"
# --only-binary :all: forces wheels, never source builds (those would need a CUDA toolkit)
# torch-geometric is pinned to 2.7.0: 2.8.0 dropped the torch-cluster fallback for
# radius_graph and requires pyg-lib>=0.6.0, which the torch-2.6.0 wheel index lacks.
python -m pip install \
  pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv "torch-geometric==2.7.0" \
  --only-binary :all: -f "$PYG_WHEELS"

echo "📦  Installing pure-Python goflow dependencies from PyPI"
python -m pip install \
  hydra-core lightning torchdiffeq wandb pymatgen einops rich

# Editable install of the goflow package (this is what puts 'import goflow' on path)
echo "🧷  pip install -e . (goflow, src layout)"
python -m pip install -e "$GOFLOW_DIR"

# Sanity check — import goflow AND the PyG companions, since a successful pip
# install does not guarantee the .so files actually load against the host's CUDA.
echo "🔍  Verifying inference stack inside $GOFLOW_ENV_NAME"
python - <<'PYEOF'
import importlib, sys
mods = ["torch", "torch_geometric", "torch_scatter", "torch_sparse",
        "torch_cluster", "torch_spline_conv", "lightning", "torchdiffeq",
        "ase", "goflow"]
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
# Functional probe: imports alone don't prove radius_graph is usable (e.g.,
# torch-geometric 2.8.0 imports fine but raises at call time without pyg-lib>=0.6.0).
from torch_geometric.nn import radius_graph
radius_graph(torch.rand(8, 3), r=1.5)
print("  ✔️  radius_graph functional probe passed")
PYEOF

# Capture absolute path to the env's python so the artifact-validation
# block (after deactivation) can still reach `import torch`.
ENV_PY="$(which python)"

set +u; $COMMAND_PKG deactivate; set -u

# ── checkpoint + feat_dict acquisition + validation ──────────────────────
CKPT_DIR="$GOFLOW_DIR/data/RDB7"
CKPT_PATH="$CKPT_DIR/epoch_316.ckpt"
FEAT_DICT_PATH="$CKPT_DIR/feat_dict_organic.pkl"
mkdir -p "$CKPT_DIR"

verify_sha256() {  # path expected_sha256
  local path="$1" expected="$2"
  local actual
  if command -v sha256sum &>/dev/null; then
    actual=$(sha256sum "$path" | awk '{print $1}')
  elif command -v shasum &>/dev/null; then
    actual=$(shasum -a 256 "$path" | awk '{print $1}')
  else
    echo "❌  Neither sha256sum nor shasum found in PATH; cannot verify checkpoint." >&2
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

# Highest-priority overrides: CLI flag, then env var, then Zenodo download.
# If a path is supplied, copy it into the canonical in-repo location so
# settings discovery picks it up.
if [[ -n "$USER_CKPT_PATH" ]]; then
  if [[ ! -f "$USER_CKPT_PATH" ]]; then
    echo "❌  --ckpt: file not found: $USER_CKPT_PATH" >&2
    exit 1
  fi
  echo "📥  Copying user-supplied checkpoint into $CKPT_PATH"
  cp "$USER_CKPT_PATH" "$CKPT_PATH"
elif [[ -n "${ARC_GOFLOW_CKPT:-}" ]]; then
  if [[ ! -f "$ARC_GOFLOW_CKPT" ]]; then
    echo "❌  ARC_GOFLOW_CKPT: file not found: $ARC_GOFLOW_CKPT" >&2
    exit 1
  fi
  echo "📥  Copying \$ARC_GOFLOW_CKPT into $CKPT_PATH"
  cp "$ARC_GOFLOW_CKPT" "$CKPT_PATH"
elif $SKIP_CKPT || $SKIP_CKPT_CHECK; then
  echo "ℹ️  --no-ckpt / --no-ckpt-check: skipping Zenodo checkpoint download."
elif [[ -f "$CKPT_PATH" ]] && \
     [[ $(stat -c%s "$CKPT_PATH" 2>/dev/null || stat -f%z "$CKPT_PATH" 2>/dev/null) -ge 1000000 ]] && \
     verify_sha256 "$CKPT_PATH" "$GOFLOW_CKPT_SHA256" 2>/dev/null; then
  echo "✔️  Existing checkpoint at $CKPT_PATH matches expected SHA-256; skipping download."
else
  if ! command -v curl &>/dev/null; then
    echo "❌  curl is required to download the GoFlow checkpoint." >&2
    exit 1
  fi
  echo "⬇️  Downloading epoch_316.ckpt (~57 MB) from Zenodo:"
  echo "       $GOFLOW_CKPT_URL"
  TMP_CKPT="$(mktemp "${CKPT_DIR}/epoch_316.ckpt.XXXXXX")"
  if ! curl -fL --retry 3 --retry-delay 5 -o "$TMP_CKPT" "$GOFLOW_CKPT_URL"; then
    rm -f "$TMP_CKPT"
    echo "❌  Download failed. Re-run the install, or pass --no-ckpt to skip." >&2
    exit 1
  fi
  if verify_sha256 "$TMP_CKPT" "$GOFLOW_CKPT_SHA256"; then
    mv "$TMP_CKPT" "$CKPT_PATH"
    echo "✔️  Checkpoint verified and saved to $CKPT_PATH"
  else
    rm -f "$TMP_CKPT"
    echo "❌  Downloaded checkpoint failed SHA-256 verification — aborting." >&2
    exit 1
  fi
fi

# feat_dict_organic.pkl: unlike the ckpt, the in-repo file in goflow_lean
# IS a real (small) pickle — no Zenodo download needed by default. Only
# acquire from --feat-dict / env var if the user wants to override.
if [[ -n "$USER_FEAT_DICT_PATH" ]]; then
  if [[ ! -f "$USER_FEAT_DICT_PATH" ]]; then
    echo "❌  --feat-dict: file not found: $USER_FEAT_DICT_PATH" >&2
    exit 1
  fi
  echo "📥  Copying user-supplied feat_dict into $FEAT_DICT_PATH"
  cp "$USER_FEAT_DICT_PATH" "$FEAT_DICT_PATH"
elif [[ -n "${ARC_GOFLOW_FEAT_DICT:-}" ]]; then
  if [[ ! -f "$ARC_GOFLOW_FEAT_DICT" ]]; then
    echo "❌  ARC_GOFLOW_FEAT_DICT: file not found: $ARC_GOFLOW_FEAT_DICT" >&2
    exit 1
  fi
  echo "📥  Copying \$ARC_GOFLOW_FEAT_DICT into $FEAT_DICT_PATH"
  cp "$ARC_GOFLOW_FEAT_DICT" "$FEAT_DICT_PATH"
fi

if $SKIP_CKPT_CHECK; then
  echo "ℹ️  --no-ckpt-check set; skipping artifact validation."
else
  echo "🔬  Validating $CKPT_PATH and $FEAT_DICT_PATH"
  if ! "$ENV_PY" - "$CKPT_PATH" "$FEAT_DICT_PATH" <<'PYEOF'
import os, pickle, sys

ckpt_path, feat_path = sys.argv[1], sys.argv[2]
errors = []

# Checkpoint
if not os.path.isfile(ckpt_path):
    errors.append(f"missing: {ckpt_path}")
else:
    sz = os.path.getsize(ckpt_path)
    if sz < 1_000_000:
        errors.append(
            f"too small ({sz} B; expected >=1 MB — likely an LFS placeholder): {ckpt_path}"
        )
    else:
        try:
            import torch
            # weights_only=False because Lightning ckpts embed an
            # omegaconf.DictConfig in 'hyper_parameters' that PyTorch 2.6+'s
            # safe-by-default unpickler refuses.
            obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except Exception as e:
            errors.append(f"torch.load failed for {ckpt_path}: {e}")
        else:
            if not isinstance(obj, dict) or "state_dict" not in obj:
                errors.append(
                    f"not a Lightning-style checkpoint (no 'state_dict' key): {ckpt_path}"
                )

# Feat dict
if not os.path.isfile(feat_path):
    errors.append(f"missing: {feat_path}")
else:
    sz = os.path.getsize(feat_path)
    if sz < 100:
        errors.append(
            f"too small ({sz} B; expected >=100 B — likely an LFS placeholder): {feat_path}"
        )
    else:
        try:
            with open(feat_path, "rb") as f:
                fd = pickle.load(f)
        except Exception as e:
            errors.append(f"pickle.load failed for {feat_path}: {e}")
        else:
            if not isinstance(fd, dict):
                errors.append(f"feat_dict is not a dict: {feat_path}")

if errors:
    print("\n".join("  ❌ " + e for e in errors), file=sys.stderr)
    sys.exit(1)
print("  ✔️  GoFlow artifacts validated.")
PYEOF
  then
    echo "" >&2
    echo "❌  GoFlow artifact validation failed." >&2
    echo "" >&2
    echo "    Provide real files via one of:" >&2
    echo "      --ckpt <path>         and  --feat-dict <path>" >&2
    echo "      ARC_GOFLOW_CKPT=...   and  ARC_GOFLOW_FEAT_DICT=..." >&2
    echo "    OR re-run with --no-ckpt-check to install the env without" >&2
    echo "    validation (the adapter will then skip GoFlow at runtime)." >&2
    exit 1
  fi
fi

# ── final notes ───────────────────────────────────────────────────────────
echo ""
echo "✅  GoFlow installation complete."
echo "    Repo      : $GOFLOW_DIR (commit $GOFLOW_COMMIT)"
echo "    Env       : $GOFLOW_ENV_NAME"
if [[ -f "$CKPT_PATH" && $(stat -c%s "$CKPT_PATH" 2>/dev/null || stat -f%z "$CKPT_PATH" 2>/dev/null) -ge 1000000 ]]; then
  echo "    Ckpt      : $CKPT_PATH"
else
  echo "    Ckpt      : (not yet installed — set ARC_GOFLOW_CKPT or use --ckpt)"
fi
if [[ -f "$FEAT_DICT_PATH" && $(stat -c%s "$FEAT_DICT_PATH" 2>/dev/null || stat -f%z "$FEAT_DICT_PATH" 2>/dev/null) -ge 100 ]]; then
  echo "    Feat dict : $FEAT_DICT_PATH"
else
  echo "    Feat dict : (not yet installed — set ARC_GOFLOW_FEAT_DICT or use --feat-dict)"
fi
echo ""
echo "    Source     : https://github.com/heid-lab/goflow_lean"
echo "    Paper DOI  : https://doi.org/10.1039/D5DD00283D"
echo "    Ckpt mirror: https://zenodo.org/records/20073635"
