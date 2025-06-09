#!/usr/bin/env bash
set -eo pipefail

echo ">>> Installing RMG-Py and RMG-database..."

RMG_PY_REPO="https://github.com/ReactionMechanismGenerator/RMG-Py.git"
RMG_DB_REPO="https://github.com/ReactionMechanismGenerator/RMG-database.git"
INSTALL_DIR="$(realpath "$(pwd)/..")"

install_repo() {
    local repo_url=$1
    local repo_name=$2

    cd "$INSTALL_DIR"
    if [ -d "$repo_name" ]; then
        echo "✔️ $repo_name exists. Checking for updates..."
        cd "$repo_name"

        REMOTE=$(git remote | grep -E '^(origin|official)$' | head -n1)
        if [ -z "$REMOTE" ]; then
            echo "❌ No valid remote found for $repo_name"
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

# Install repos
install_repo "$RMG_PY_REPO" "RMG-Py"
install_repo "$RMG_DB_REPO" "RMG-database"

# Absolute paths
RMG_PY_PATH="$INSTALL_DIR/RMG-Py"
RMG_DB_PATH="$INSTALL_DIR/RMG-database"

# Save RMG-Py commit hash for caching purposes
git -C "$RMG_PY_PATH" rev-parse HEAD > "$(pwd)/RMG_PY_COMMIT_HASH"

# Function to export environment variables safely to .bashrc
export_to_bashrc() {
    local var=$1
    local value=$2

    if grep -q "^export $var=" ~/.bashrc; then
        sed -i.bak "/^export $var=/c\\export $var=$value" ~/.bashrc
        echo "🔄 Updated $var in ~/.bashrc"
    else
        echo "export $var=$value" >> ~/.bashrc
        echo "➕ Added $var to ~/.bashrc"
    fi
    export $var="$value"
}

# Export necessary variables
export_to_bashrc "RMG_PY_PATH" "$RMG_PY_PATH"
export_to_bashrc "RMG_DB_PATH" "$RMG_DB_PATH"

# Configure conda to use libmamba solver
conda install -n base conda-libmamba-solver -y
conda config --set solver libmamba

# RMG-Py setup
cd "$RMG_PY_PATH"
if ! conda env list | grep -q "rmg_env"; then
    conda env create -f environment.yml
else
    echo "✔️ conda environment 'rmg_env' already exists."
fi

eval "$(conda shell.bash hook)"
conda activate rmg_env
make

# Julia dependencies
julia -e '
using Pkg
Pkg.add("PyCall")
Pkg.build("PyCall")
Pkg.add(PackageSpec(name="ReactionMechanismSimulator", rev="for_rmg"))
Pkg.instantiate()
try
    using ReactionMechanismSimulator
catch
end
'

# ensure the Python bridge runs inside the conda env
eval "$(conda shell.bash hook)"
conda activate rmg_env
python -c "import julia; julia.install(); import diffeqpy; diffeqpy.install()"

echo "✅ RMG-Py and RMG-database installation completed successfully."
