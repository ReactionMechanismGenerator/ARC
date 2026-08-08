# shellcheck shell=bash
# Shared plumbing for the interactive `arc` / `arcrestart` / `rmg` / `arkane`
# commands that aliases.sh defines.
#
# The interactive commands dispatch through the container entry wrapper
# (/usr/local/bin/entrywrapper.sh) -- the very same code path taken by
# `docker run IMAGE arc my_case/input.yml` -- so that the interactive and the
# non-interactive form of a command cannot drift apart.
#
# This file is sourced, never executed: it must not call `exit` and must not
# set shell options that would leak into the user's interactive session.

ARC_ENTRYWRAPPER="${ARC_ENTRYWRAPPER:-/usr/local/bin/entrywrapper.sh}"

# Run the entry wrapper with the given arguments, echoing stdout/stderr to the
# terminal while appending them to stdout.log / stderr.log in the current
# directory. The exit status of the wrapper is preserved.
arc_entry() {
    if [ ! -x "$ARC_ENTRYWRAPPER" ]; then
        echo "Error: entry wrapper not found or not executable: $ARC_ENTRYWRAPPER" >&2
        return 127
    fi
    "$ARC_ENTRYWRAPPER" "$@" > >(tee -a stdout.log) 2> >(tee -a stderr.log >&2)
}
