#!/usr/bin/env bash
set -euo pipefail

# --------------------------------------------------------------------------
# helper: load micromamba’s shell integration once
# --------------------------------------------------------------------------
activate_mamba() {
    eval "$(micromamba shell hook --shell bash)"
}

# --------------------------------------------------------------------------
# helper: move to a sensible directory only when necessary
# --------------------------------------------------------------------------
fix_workdir() {
    # If $PWD is / (the Docker default) and $HOME exists & is writable,
    # switch there. Otherwise stay where the caller put us.
    if [[ $PWD == "/" && -w "${HOME:-}" ]]; then
        cd "$HOME"
    fi
}

# --------------------------------------------------------------------------
# command dispatcher
# --------------------------------------------------------------------------
main() {
    activate_mamba
    fix_workdir

    case "${1:-}" in
        rmg)
            micromamba activate rmg_env
            shift
            exec python-jl /home/mambauser/Code/RMG-Py/rmg.py "$@"
            ;;
        arc)
            micromamba activate arc_env
            shift
            exec python /home/mambauser/Code/ARC/ARC.py "$@"
            ;;
        "" )
            # No args → plain interactive shell, leave env choice to user
            exec bash --login
            ;;
        * )
            exec "$@"
            ;;
    esac
}

main "$@"
