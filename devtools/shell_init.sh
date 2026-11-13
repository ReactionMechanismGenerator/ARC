# Shell initialisation shared by the devtools installers that call "$COMMAND_PKG activate".
# Source this file, set COMMAND_PKG (micromamba | mamba | conda), then call arc_shell_init.
#
# Sourcing conda.sh alone defines only the `conda` shell function, so `mamba activate` then
# aborts with "Shell not initialized"; each frontend needs its own hook. Any failure returns 1,
# so an installer running under `set -e` stops here instead of at its first activate.

arc_shell_init() {
    local hook base err="" rc=0 had_u=0
    [[ $- == *u* ]] && had_u=1
    set +u
    [[ -n ${COMMAND_PKG:-} ]] && command -v "$COMMAND_PKG" >/dev/null || rc=1
    ((rc != 0)) || case "$COMMAND_PKG" in
        micromamba)
            if hook="$(micromamba shell hook --shell bash)" && [[ -n $hook ]]; then eval "$hook" || rc=1; else rc=1; fi
            ;;
        mamba)
            # mamba >= 2 prints its own hook; mamba 1.x ships profile.d/mamba.sh on top of conda.sh.
            local errf
            errf="$(mktemp)"
            if hook="$(mamba shell hook --shell bash 2>"$errf")" && [[ -n $hook ]]; then
                eval "$hook" || rc=1
            elif base="$(conda info --base)"; then
                source "$base/etc/profile.d/conda.sh" && source "$base/etc/profile.d/mamba.sh" || rc=1
            else
                rc=1
            fi
            ((rc == 0)) || err="$(cat "$errf")"
            rm -f "$errf"
            ;;
        conda)
            if base="$(conda info --base)"; then source "$base/etc/profile.d/conda.sh" || rc=1; else rc=1; fi
            ;;
        *)
            rc=1
            ;;
    esac
    ((had_u)) && set -u
    if ((rc == 0)) && [[ $(type -t "$COMMAND_PKG") != function ]]; then
        rc=1
    fi
    if ((rc != 0)); then
        [[ -n $err ]] && echo "$err" >&2
        echo "❌  Could not initialise the shell for COMMAND_PKG='${COMMAND_PKG:-}'" >&2
        return 1
    fi
    return 0
}
