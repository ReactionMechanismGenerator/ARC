#!/usr/bin/env bash
set -eo pipefail

ENV_NAME="arc_env"

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

# Activate the environment
if [[ "$COMMAND_PKG" == "micromamba" ]]; then
    eval "$(micromamba shell hook --shell=bash)"
    micromamba activate "$ENV_NAME"
elif [[ "$COMMAND_PKG" == "mamba" ]]; then
    eval "$(mamba shell hook --shell=bash)"
    mamba activate "$ENV_NAME"
else
    source "$($COMMAND_PKG info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
fi

# Compile ARC’s in-repo Cython module
cd "$(dirname "${BASH_SOURCE[0]}")/../"
python setup.py build_ext --inplace --verbose
cd ../../  # Return to ARC root

# Sanity check import
python - <<'EOF'
import arc.molecule
print("arc.molecule __file__:", arc.molecule.__file__)
EOF

echo "✅ ARC molecule module compiled successfully."
