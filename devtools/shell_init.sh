# Shell initialisation shared by the devtools installers that call "$COMMAND_PKG activate".
# Source this file, set COMMAND_PKG (micromamba | mamba | conda), then call arc_shell_init.
# Each frontend needs its own hook: conda.sh alone does not define `mamba`. Returns 1 on any failure.

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
            # mamba >= 2 prints its own hook; mamba 1.x has no `shell hook` and ships profile.d/mamba.sh.
            local errf fallback=""
            errf="$(mktemp)"
            if hook="$(mamba shell hook --shell bash 2>"$errf")" && [[ -n $hook ]]; then
                eval "$hook" || rc=1
            elif ! base="$(conda info --base)"; then
                fallback="'conda info --base' failed"
            elif ! source "$base/etc/profile.d/conda.sh" || ! source "$base/etc/profile.d/mamba.sh"; then
                fallback="sourcing $base/etc/profile.d/{conda,mamba}.sh failed"
            fi
            if [[ -n $fallback ]]; then
                rc=1
                err="mamba shell hook: $(<"$errf")"$'\n'"fallback for mamba 1.x: $fallback"
            fi
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
