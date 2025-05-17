#!/bin/bash -l
set -e

if [[ ! -f environment.yml ]] || [[ ! -d devtools ]]; then
    echo "❌ This script must be run from the ARC root directory."
    exit 1
fi

echo ">>> Configuring conda to use libmamba solver"
conda install -n base conda-libmamba-solver --yes
conda config --set solver libmamba

echo ">>> Beginning full ARC external repo installation..."
pushd . >/dev/null

bash devtools/install_rmg.sh
bash devtools/install_molecule.sh
bash devtools/install_gcn_cpu.sh
bash devtools/install_autotst.sh
bash devtools/install_kinbot.sh
bash devtools/install_ob.sh
bash devtools/install_xtb.sh
bash devtools/install_sella.sh
bash devtools/install_torchani.sh

popd >/dev/null

echo "✅ Done installing all external dependencies."
