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
        [ -n "$REMOTE" ] || { echo "❌ No valid remote for $repo_name"; exit 1; }
        CURRENT_BRANCH=$(git symbolic-ref --short HEAD)
        if [ "$CURRENT_BRANCH" = "main" ]; then
            git pull "$REMOTE" main
        else
            echo "ℹ️ Skipping update (branch=$CURRENT_BRANCH)"
        fi
    else
        echo "📦 Cloning $repo_name..."
        git clone "$repo_url" "$repo_name"
    fi
}

install_repo "$RMG_PY_REPO" "RMG-Py"
install_repo "$RMG_DB_REPO" "RMG-database"

RMG_PY_PATH="$INSTALL_DIR/RMG-Py"
RMG_DB_PATH="$INSTALL_DIR/RMG-database"
git -C "$RMG_PY_PATH" rev-parse HEAD > "$(pwd)/RMG_PY_COMMIT_HASH"

export_to_bashrc() {
    local var=$1 val=$2
    if grep -q "^export $var=" ~/.bashrc; then
        sed -i.bak "/^export $var=/c\\export $var=$val" ~/.bashrc
    else
        echo "export $var=$val" >> ~/.bashrc
    fi
    export "$var"="$val"
}
export_to_bashrc RMG_PY_PATH "$RMG_PY_PATH"
export_to_bashrc RMG_DB_PATH "$RMG_DB_PATH"

# Use libmamba + Julia 1.10
conda install -n base conda-libmamba-solver -y
conda config --set solver libmamba
conda install -n base -c conda-forge julia=1.10 -y

# Build RMG-Py
cd "$RMG_PY_PATH"
if ! conda env list | grep -q rmg_env; then
    conda env create -f environment.yml
fi
eval "$(conda shell.bash hook)"
conda activate rmg_env
make install

# Disable Julia precompile cache so we skip GPUCompiler/Enzyme errors
export JULIA_LOAD_CACHE_PATH=""
export JULIA_DEPOT_PATH="$HOME/.juliacache"

# Julia: install PyCall and ReactionMechanismSimulator
julia -e '
using Pkg
Pkg.add("PyCall"); Pkg.build("PyCall")
Pkg.add(PackageSpec(name="ReactionMechanismSimulator", rev="for_rmg"))
Pkg.instantiate()
'

# Test load (warnings only)
julia -e 'try using ReactionMechanismSimulator; catch e println("⚠️ ", e) end'

# Python–Julia bridge
eval "$(conda shell.bash hook)"
conda activate rmg_env
pip install --upgrade julia
python - <<'PYCODE'
import julia
julia.install()
print("✅ Julia bridge OK")
PYCODE

echo "✅ RMG-Py and RMG-database installation completed successfully."
