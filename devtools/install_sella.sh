#!/bin/bash -l
set -e

echo ">>> Checking available package manager..."

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
    BASE=$($COMMAND_PKG info --base)
    . "$BASE/etc/profile.d/conda.sh"
fi

echo ">>> Creating the Sella environment..."

if [ ! -f devtools/sella_environment.yml ]; then
    echo "❌ File not found: devtools/sella_environment.yml"
    exit 1
fi

$COMMAND_PKG env create -n sella_env -f devtools/sella_environment.yml || true

# Dynamically determine the sella_env install path
ENV_PATH=$($COMMAND_PKG env list | awk '$1 == "sella_env" {print $2}' | head -n1)

if [ -z "$ENV_PATH" ]; then
    echo "❌ Could not locate path for sella_env"
    exit 1
fi

echo ">>> Configuring LD_LIBRARY_PATH for sella_env at: $ENV_PATH"

ACTIVATE_HOOK="$ENV_PATH/etc/conda/activate.d/env_vars.sh"
DEACTIVATE_HOOK="$ENV_PATH/etc/conda/deactivate.d/env_vars.sh"

mkdir -p "$(dirname "$ACTIVATE_HOOK")"
mkdir -p "$(dirname "$DEACTIVATE_HOOK")"

cat <<EOF > "$ACTIVATE_HOOK"
export OLD_LD_LIBRARY_PATH=\$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=\${LD_LIBRARY_PATH}:$ENV_PATH/lib
EOF

cat <<EOF > "$DEACTIVATE_HOOK"
export LD_LIBRARY_PATH=\$OLD_LD_LIBRARY_PATH
unset OLD_LD_LIBRARY_PATH
EOF

echo "✔️ LD_LIBRARY_PATH hooks installed."

echo "✅ Done installing Sella."
