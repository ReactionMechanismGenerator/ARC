# personalised shortcuts – sourced for every login shell
alias rc='source ~/.bashrc'
alias rce='nano ~/.bashrc'
alias erc='nano ~/.bashrc'

# micromamba drop-ins
alias mamba='micromamba'
alias conda='micromamba'
alias deact='micromamba deactivate'

# env activators
alias rmge='micromamba activate rmg_env'
alias arce='micromamba activate arc_env'

# code roots (set once at image build)
export rmgpy_path=/home/mambauser/Code/RMG-Py
export rmgdb_path=/home/mambauser/Code/RMG-database
export arc_path=/home/mambauser/Code/ARC

alias rmgcode='cd "$rmgpy_path"'
alias dbcode='cd "$rmgdb_path"'
alias arcode='cd "$arc_path"'


# job wrappers
# These are shell functions rather than aliases so that arguments are forwarded:
# `arc my_case/input.yml` runs my_case/input.yml, not the hard-coded default.
# With no arguments at all the historic default input file is still used.
# The work is delegated to /usr/local/bin/entrywrapper.sh, so these behave
# exactly like `docker run IMAGE arc <file>`.

_arc_job_helpers="/usr/local/bin/arc_job_helpers.sh"

if [ ! -r "$_arc_job_helpers" ]; then
    printf 'ARC shell initialization failed: required helper is missing or unreadable: %s\n' \
        "$_arc_job_helpers" >&2
    return 1 2>/dev/null || exit 1
fi

# shellcheck source=dockerfiles/job_helpers.sh
if ! . "$_arc_job_helpers"; then
    printf 'ARC shell initialization failed: could not load helper: %s\n' \
        "$_arc_job_helpers" >&2
    return 1 2>/dev/null || exit 1
fi

if ! declare -F arc_entry >/dev/null; then
    printf 'ARC shell initialization failed: helper did not define arc_entry: %s\n' \
        "$_arc_job_helpers" >&2
    return 1 2>/dev/null || exit 1
fi

unset _arc_job_helpers

rmg() {
    if [ "$#" -eq 0 ]; then set -- input.py; fi
    arc_entry rmg "$@"
}

arkane() {
    if [ "$#" -eq 0 ]; then set -- input.py; fi
    arc_entry micromamba run -n rmg_env python "$rmgpy_path/Arkane.py" "$@"
}

arc() {
    if [ "$#" -eq 0 ]; then set -- input.yml; fi
    arc_entry arc "$@"
}

arcrestart() {
    if [ "$#" -eq 0 ]; then set -- restart.yml; fi
    arc_entry arc "$@"
}
