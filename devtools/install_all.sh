#!/usr/bin/env bash
set -euo pipefail

# -------------------------------------------------------------------------
# INSTALLATION OF ARC, RMG, and other external dependencies
# Options: 
# --no-clean: Skip the aggressive cleanup of conda/micromamba caches and build directories.
# --no-ext: Skip the installation of external dependencies (autotst, kinbot, etc.)
# --rmg-*: Options for RMG installation
# --arc-*: Options for ARC installation
# -------------------------------------------------------------------------

# ── locate this script and the repo root ──────────────────────────────────
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
DEVTOOLS_DIR="$SCRIPT_DIR"                   # you are already in devtools
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"    # one level up

# helper so every sub-call works no matter where we launched from
run_devtool () { bash "$DEVTOOLS_DIR/$1" "${@:2}"; }

###################################################################################
# Flag for cleaning up disk space and caches, and for installing external dependencies
# and their arguments.
#################################################################################

SKIP_CLEAN=false
SKIP_EXT=false
SKIP_ARC=false
SKIP_RMG=false
ARC_INSTALLED=false
RMG_ARGS=()
ARC_ARGS=()
EXT_ARGS=()
GENERIC_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-clean) SKIP_CLEAN=true ;;
        --no-ext)   SKIP_EXT=true  ;;
        --no-arc)   SKIP_ARC=true  ;;
        --no-rmg)   SKIP_RMG=true  ;;
        --rmg-*)    RMG_ARGS+=("--${1#--rmg-}") ;;
        --arc-*)    ARC_ARGS+=("--${1#--arc-}") ;;
        --ext-*)    EXT_ARGS+=("--${1#--ext-}") ;;
        --help|-h)
            cat <<EOF
Usage: $0 [global-flags] [--rmg-xxx] [--arc-yyy] [--ext-zzz]
  --no-clean          Skip micromamba/conda cache cleanup
  --no-ext            Skip external tools (AutoTST, KinBot, …)
  --no-rmg            Skip RMG-Py entirely
  --rmg-path          Forward '--path' to RMG installer
  --rmg-pip           Forward '--pip'  to RMG installer
  ...
EOF
            exit 0 ;;
        *) GENERIC_ARGS+=("$1") ;;
    esac
    shift
done

echo ">>> Beginning full ARC external repo installation…"
echo "    RMG sub-flags : ${RMG_ARGS[*]:-(none)}"
echo "    ARC sub-flags : ${ARC_ARGS[*]:-(none)}"
echo "    EXT sub-flags : ${EXT_ARGS[*]:-(none)}"



# -----------------------------------------------------------------------------
# Main install sequence
# -----------------------------------------------------------------------------
echo ">>> Beginning full ARC external repo installation…"
pushd . >/dev/null

# 1) RMG (optional)
if [[ $SKIP_RMG == false ]]; then
    echo "=== Installing RMG ==="
    run_devtool install_rmg.sh "${RMG_ARGS[@]}"
else
    echo "ℹ️  --no-rmg flag set. Skipping RMG installation."
fi

# 2) ARC itself (skip env creation in CI or if user requests it)
if [[ "${CI:-false}" != "true" && "${SKIP_ARC:-false}" != "true" ]]; then
    if [[ $SKIP_CLEAN == false ]]; then
        echo "=== Cleaning up old ARC build artifacts ==="
        run_devtool clean.sh --python-builds --pip
        echo ":information_source:  Disk cleanup complete."
    else
        echo ":information_source:  Skipping ARC cleanup as per --no-clean flag."
    fi

    echo "=== Installing ARC ==="
    run_devtool install_arc.sh "${ARC_ARGS[@]}"
    ARC_INSTALLED=true
else
    ARC_INSTALLED=false
    echo ":information_source:  CI detected or --no-arc flag set. Skip cleaning ARC installation."
fi

# 3) PyRDL (depends on ARC)
if [[ $ARC_INSTALLED == true ]]; then
    echo "=== Installing PyRDL ==="
    bash devtools/install_pyrdl.sh
else
    echo "ℹ️  Skipping PyRDL install because ARC installation was skipped."
fi

if [[ $SKIP_EXT == false ]]; then
    # map of friendly names → installer scripts
    declare -A EXT_INSTALLERS=(
        [GCN\ CPU]=install_gcn.sh
        [AutoTST]=install_autotst.sh
        [KinBot]=install_kinbot.sh
        [OpenBabel]=install_ob.sh
        [xtb]=install_xtb.sh
        [CREST]=install_crest.sh
        [Sella]=install_sella.sh
        [TorchANI]=install_torchani.sh
    )

        # installer-specific flag whitelists
    declare -A EXT_FLAG_WHITELIST=(
       [install_gcn.sh]="--conda"
       [install_autotst.sh]="--conda"
        # add more later, e.g.  [install_xtb.sh]="--cuda --prefix"
    )


    for name in "${!EXT_INSTALLERS[@]}"; do
        script="${EXT_INSTALLERS[$name]}"
        echo "=== Installing $name ==="

        # filter EXT_ARGS by whitelist for this script
        allowed=()
        for arg in "${EXT_ARGS[@]}"; do
            [[ " ${EXT_FLAG_WHITELIST[$script]} " == *" ${arg} "* ]] && \
                allowed+=("$arg")
        done

        run_devtool "$script" "${allowed[@]}"
    done

else
    echo "ℹ️  --no-ext flag set. Skipping external-dependency installs."
fi

# 4) Clean up disk space
run_devtool clean.sh --conda

popd >/dev/null

echo "✅ Done installing all external dependencies."
