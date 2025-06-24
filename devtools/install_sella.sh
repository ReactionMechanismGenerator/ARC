#!/bin/bash -l
set -eo pipefail

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
    . "$BASE/etc/profile.d/conda.sh"
fi

echo ">>> Creating or updating the Sella environment..."

if [ ! -f devtools/sella_environment.yml ]; then
    echo "❌ File not found: devtools/sella_environment.yml"
    exit 1
fi

if $COMMAND_PKG env list | grep -q '^sella_env\s'; then
    echo ">>> Updating existing sella_env..."
    if [ "$COMMAND_PKG" != "conda" ]; then
        $COMMAND_PKG env update -n sella_env -f devtools/sella_environment.yml --prune -y
    else
        $COMMAND_PKG env update -n sella_env -f devtools/sella_environment.yml --prune
    fi
else
    echo ">>> Creating new sella_env..."
    if [ "$COMMAND_PKG" != "conda" ]; then
        $COMMAND_PKG env create -n sella_env -f devtools/sella_environment.yml -y
    else
        $COMMAND_PKG env create -n sella_env -f devtools/sella_environment.yml 
    fi
fi

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

echo "✅ Done installing Sella."
