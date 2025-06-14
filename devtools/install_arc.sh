#!/usr/bin/env bash
set -eo pipefail
set -x  # Echo commands

ENV_NAME="arc_env"
ENV_FILE="environment.yml"
LOGDIR="$HOME/molecule_build_logs"
mkdir -p "$LOGDIR"

# Detect available conda frontend
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

# Initialize shell hooks
if [[ "$COMMAND_PKG" == "micromamba" ]]; then
    eval "$(micromamba shell hook --shell=bash)"
    micromamba activate "$ENV_NAME"
else
    source "$($COMMAND_PKG info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
fi

# Create or update the environment
if $COMMAND_PKG env list | grep -qw "$ENV_NAME"; then
    echo "✔️ Environment '$ENV_NAME' exists. Updating from $ENV_FILE..."
    $COMMAND_PKG env update -n "$ENV_NAME" -f "$ENV_FILE"
else
    echo "📦 Creating environment '$ENV_NAME' from $ENV_FILE..."
    $COMMAND_PKG env create -n "$ENV_NAME" -f "$ENV_FILE"
fi

# Compile ARC’s in-repo Cython module
cd "$(dirname "${BASH_SOURCE[0]}")/arc/molecule"
python setup.py build_ext --inplace --verbose 2>&1 | tee "$LOGDIR/arc_molecule_build.log"
cd ../../  # Return to ARC root

# Sanity check import
python - <<'EOF' |& tee "$LOGDIR/arc_molecule_import.log"
import arc.molecule
print("arc.molecule __file__:", arc.molecule.__file__)
EOF

# Print errors if any
grep -Ei 'error|failed' "$LOGDIR/arc_molecule_build.log" || true

echo "✅ ARC environment and molecule module setup complete."
