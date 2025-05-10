#!/bin/bash -l
set -e

echo ">>> Checking current directory…"
if [[ ! -f environment.yml ]] || [[ ! -d devtools ]]; then
    echo "❌ This script must be run from the ARC root directory."
    exit 1
fi

echo ">>> Beginning full ARC external repo installation…"
pushd . >/dev/null

bash devtools/install_rmgdb.sh
bash devtools/install_molecule.sh    # ← now includes both build+pip install of PyRDL
bash devtools/install_gcn_cpu.sh
bash devtools/install_autotst.sh
bash devtools/install_kinbot.sh
bash devtools/install_ob.sh
bash devtools/install_xtb.sh
bash devtools/install_sella.sh
bash devtools/install_torchani.sh

popd >/dev/null

echo "✅ Done installing all external dependencies."
echo "📢 To activate environment variables and PYTHONPATH, run: source ~/.bashrc"
echo "📦 PYTHONPATH is currently:"
echo "$PYTHONPATH"
