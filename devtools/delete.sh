#!/usr/bin/env bash
set -eo pipefail

REPOS_TO_REMOVE=(RMG-Py RMG-database molecule RingDecomposerLib AutoTST TS-GCN KinBot)
ENVS_TO_REMOVE=(tst_env ob_env xtb_env rmg_env arc_env gcn_env)

echo "⚠️ WARNING: This will DELETE the following repositories and environments:"
echo "Repositories: ${REPOS_TO_REMOVE[*]}"
echo "Environments: ${ENVS_TO_REMOVE[*]}"
read -rp "Do you wish to proceed? (y/[n]) " confirm

if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "Aborted by user."
    exit 1
fi

echo "🧹 Cleaning ARC build artifacts..."

# Remove Python cache files
find . -type d -name "__pycache__" -exec rm -rf {} +

# Remove specific directories
rm -rf functional ipython
rm -rf arc/testing/gcn_tst

# Remove coverage files
rm -f .coverage coverage.xml

# Remove cloned repositories
rm -rf "${REPOS_TO_REMOVE[@]}"

# Remove specified conda environments
source "$(conda info --base)/etc/profile.d/conda.sh"
for env in "${ENVS_TO_REMOVE[@]}"; do
    if conda env list | grep -qw "$env"; then
        echo "🗑️ Removing conda environment: $env"
        conda remove -n "$env" --all -y
    else
        echo "ℹ️ Conda environment '$env' not found."
    fi
done

echo "✅ Clean completed."
