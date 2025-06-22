#!/usr/bin/env bash
set -euo pipefail

###################################################################################
# Flag for cleaning
#################################################################################
SKIP_CLEAN=false
[[ "${1:-}" == --no-clean ]] && SKIP_CLEAN=true


# -----------------------------------------------------------------------------
# Helper: aggressively clean conda/micromamba caches & remove any known build
# directories in our workspace.  Ignores any permission errors.
# -----------------------------------------------------------------------------
cleanup_disk() {
    echo ">>> Cleaning package manager caches and temporary build dirs…"

    # Clean conda / micromamba
    if command -v micromamba &>/dev/null; then
        micromamba clean --all --yes || true
    elif command -v mamba &>/dev/null; then
        mamba clean --all --yes || true
    elif command -v conda &>/dev/null; then
        conda clean -afy || true
    fi

    # Remove pip cache
    rm -rf "$HOME/.cache/pip" || true

    # Remove any "build/" or "dist/" dirs left behind in our repo clones
    find . -type d \( -name build -o -name dist \) -prune -exec rm -rf {} + 2>/dev/null || true

    # Remove any __pycache__
    find . -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true

    # Prune any other big caches under home that we own
    rm -rf "$HOME/.cache"/git* "$HOME/.cache"/julia* || true

    df -h . | sed '1!p;d'   # show after-clean free space
}

# -----------------------------------------------------------------------------
# Main install sequence
# -----------------------------------------------------------------------------
echo ">>> Beginning full ARC external repo installation…"
pushd . >/dev/null

# 1) RMG
echo "=== Installing RMG ==="
bash devtools/install_rmg.sh
! $SKIP_CLEAN && cleanup_disk

# 2) PyRDL
echo "=== Installing PyRDL ==="
bash devtools/install_pyrdl.sh
! $SKIP_CLEAN && cleanup_disk

# 3) ARC itself (skip env creation in CI)
if [[ -z "${CI:-}" ]]; then
    echo "=== Installing ARC ==="
    bash devtools/install_arc.sh
    ! $SKIP_CLEAN && cleanup_disk
else
    echo "ℹ️ CI detected, skipping arc_env creation."
fi

# 4) GCN (CPU)
echo "=== Installing GCN CPU ==="
bash devtools/install_gcn_cpu.sh
! $SKIP_CLEAN && cleanup_disk

# 5) AutoTST
echo "=== Installing AutoTST ==="
bash devtools/install_autotst.sh
! $SKIP_CLEAN && cleanup_disk

# 6) KinBot
echo "=== Installing KinBot ==="
bash devtools/install_kinbot.sh
! $SKIP_CLEAN && cleanup_disk

# 7) Open Babel
echo "=== Installing OpenBabel ==="
bash devtools/install_ob.sh
! $SKIP_CLEAN && cleanup_disk

# 8) xtb
echo "=== Installing xtb ==="
bash devtools/install_xtb.sh
! $SKIP_CLEAN && cleanup_disk

# 9) Sella
echo "=== Installing Sella ==="
bash devtools/install_sella.sh
! $SKIP_CLEAN && cleanup_disk

# 10) TorchANI
echo "=== Installing TorchANI ==="
bash devtools/install_torchani.sh
! $SKIP_CLEAN && cleanup_disk

popd >/dev/null

echo "✅ Done installing all external dependencies."
