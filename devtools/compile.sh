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
source "$(dirname "${BASH_SOURCE[0]}")/shell_init.sh"
arc_shell_init
"$COMMAND_PKG" activate "$ENV_NAME"

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
