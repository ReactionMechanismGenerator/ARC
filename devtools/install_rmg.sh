#!/usr/bin/env bash
set -euo pipefail

echo ">>> Installing RMG-Py and RMG-database..."

RMG_PY_REPO="https://github.com/ReactionMechanismGenerator/RMG-Py.git"
RMG_DB_REPO="https://github.com/ReactionMechanismGenerator/RMG-database.git"
INSTALL_DIR="$(pwd)/.."

install_repo() {
    local repo_url=$1
    local repo_name=$2

    cd "$INSTALL_DIR"
    if [ -d "$repo_name" ]; then
        echo "✔️ $repo_name exists. Checking for updates..."
        cd "$repo_name"

        REMOTE=$(git remote | grep -E '^(origin|official)$' | head -n1)
        if [ -z "$REMOTE" ]; then
            echo "❌ No valid remote found (expected 'origin' or 'official') for $repo_name"
            exit 1
        fi

        CURRENT_BRANCH=$(git symbolic-ref --short HEAD)
        if [ "$CURRENT_BRANCH" = "main" ]; then
            echo "ℹ️ Updating '$repo_name' on branch 'main'..."
            git pull "$REMOTE" main
        else
            echo "ℹ️ Skipping update: $repo_name is on branch '$CURRENT_BRANCH'"
        fi
    else
        echo "📦 Cloning $repo_name..."
        git clone "$repo_url" "$repo_name"
    fi
}

# Install RMG-Py
install_repo "$RMG_PY_REPO" "RMG-Py"

# Install RMG-database
install_repo "$RMG_DB_REPO" "RMG-database"

# Setup environment variables
RMG_PY_PATH="$INSTALL_DIR/RMG-Py"
RMG_DB_PATH="$INSTALL_DIR/RMG-database"

# RMG-Py Installation Steps
cd "$RMG_PY_PATH"

if conda env list | grep -q "rmg_env"; then
    echo "✔️ rmg_env already exists. Skipping environment creation."
else
    conda env create -f environment.yml
fi

eval "$(conda shell.bash hook)"
conda activate rmg_env
make

# Update PYTHONPATH and PATH for RMG-Py
export PYTHONPATH="$RMG_PY_PATH:$PYTHONPATH"
export PATH="$RMG_PY_PATH:$PATH"

# Configure Julia dependencies
julia -e 'using Pkg; Pkg.add("PyCall"); Pkg.build("PyCall"); Pkg.add(PackageSpec(name="ReactionMechanismSimulator",rev="for_rmg")); using ReactionMechanismSimulator;'

# Configure Python-Julia bridge
python -c "import julia; julia.install(); import diffeqpy; diffeqpy.install()"

# Update PYTHONPATH
for PATH_ENTRY in "$RMG_PY_PATH" "$RMG_DB_PATH"; do
    LINE="export PYTHONPATH=\${PYTHONPATH:-}:$PATH_ENTRY"
    if ! grep -Fxq "$LINE" ~/.bashrc; then
        echo "$LINE" >> ~/.bashrc
        echo "✔️ Added $PATH_ENTRY to PYTHONPATH in ~/.bashrc"
    else
        echo "ℹ️ $PATH_ENTRY already in PYTHONPATH"
    fi
    export PYTHONPATH="${PYTHONPATH:-}:$PATH_ENTRY"
done

# Set RMGDB environment variable
if ! grep -Fxq "export RMGDB=$RMG_DB_PATH" ~/.bashrc; then
    sed -i '/export RMGDB=/d' ~/.bashrc
    echo "export RMGDB=$RMG_DB_PATH" >> ~/.bashrc
    echo "✔️ RMGDB environment variable set in ~/.bashrc"
else
    echo "ℹ️ RMGDB environment variable already set in ~/.bashrc"
fi
export RMGDB="$RMG_DB_PATH"

echo "✅ RMG-Py and RMG-database installation completed successfully."
