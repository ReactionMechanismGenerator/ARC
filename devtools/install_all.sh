#!/bin/bash -l
set -e

if [[ ! -f environment.yml ]] || [[ ! -d devtools ]]; then
    echo "❌ This script must be run from the ARC root directory."
    exit 1
fi

function aggressive_cleanup() {
    echo ">>> Cleaning up caches and temp files to free disk space..."
    # Conda/Mamba caches
    conda clean --all --yes
    if command -v micromamba &>/dev/null; then
        micromamba clean --all --yes
    fi
    # pip cache
    rm -rf ~/.cache/pip
    # system temp
    rm -rf /tmp/* /var/tmp/*
    # apt cache (on Ubuntu CI runners)
    sudo apt-get clean
    sudo rm -rf /var/lib/apt/lists/*
    echo ">>> Done cleanup."
}

echo ">>> Configuring conda to use libmamba solver"
conda install -n base conda-libmamba-solver --yes
conda config --set solver libmamba

echo ">>> Beginning full ARC external repo installation..."
pushd . >/dev/null

# Install RMG before installing ARC and molecule + py_rdl
bash devtools/install_rmg.sh
aggressive_cleanup

# Check if running in CI environment, if so don't install ARC's env
if [[ -z "$CI" ]]; then
    bash devtools/install_arc.sh
    aggressive_cleanup
else
    echo "ℹ️ CI detected, skipping arc_env creation (handled externally)."
fi

bash devtools/install_molecule.sh
aggressive_cleanup

bash devtools/install_gcn_cpu.sh
aggressive_cleanup

bash devtools/install_autotst.sh
aggressive_cleanup

bash devtools/install_kinbot.sh
aggressive_cleanup

bash devtools/install_ob.sh
aggressive_cleanup

bash devtools/install_xtb.sh
aggressive_cleanup

bash devtools/install_sella.sh
aggressive_cleanup

bash devtools/install_torchani.sh
aggressive_cleanup

popd >/dev/null

echo "✅ Done installing all external dependencies."
