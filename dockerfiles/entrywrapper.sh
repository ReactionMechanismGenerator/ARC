#!/usr/bin/env bash
set -euo pipefail

# 0) If root, optionally remap mambauser UID/GID for bind mounts, then drop privileges.
if [[ "$(id -u)" -eq 0 ]]; then
  if [[ -n "${PGID:-}" ]]; then
    existing_group=""
    if getent group "$PGID" >/dev/null; then
      existing_group="$(getent group "$PGID" | cut -d: -f1)"
    fi
    if [[ -n "$existing_group" && "$existing_group" != "mambauser" ]]; then
      echo "Error: requested PGID '$PGID' is already in use by group '$existing_group', cannot remap mambauser group." >&2
      exit 1
    fi
    groupmod -g "$PGID" mambauser
  fi
  if [[ -n "${PUID:-}" ]]; then
    current_uid="$(id -u mambauser)"
    if [[ "$PUID" != "$current_uid" ]]; then
      if getent passwd "$PUID" >/dev/null; then
        echo "Error: Cannot remap mambauser to UID '$PUID' because it is already in use by another user." >&2
        exit 1
      fi
      usermod -u "$PUID" mambauser
    fi
  fi

  if [[ -d /home/mambauser/Code ]]; then
    if ! chown -R mambauser:mambauser /home/mambauser/Code; then
      echo "warning: failed to change ownership of /home/mambauser/Code to mambauser:mambauser (read-only mount or permission issue?)" >&2
    fi
  fi
  if [[ -d /work ]]; then
    if ! chown -R mambauser:mambauser /work; then
      echo "warning: failed to change ownership of /work to mambauser:mambauser (read-only mount or permission issue?)" >&2
    fi
  fi

  if [[ "${ENTRYWRAPPER_AS_USER:-0}" != "1" ]]; then
    exec runuser -u mambauser -- env ENTRYWRAPPER_AS_USER=1 /usr/local/bin/entrywrapper.sh "$@"
  fi
fi

# If running non-interactively at container root and /work exists; it will go there
# This helps when users forget `-w /work`
if [[ -d /work && "${PWD:-/}" == "/" && ! -t 0 ]]; then
  cd /work || true
fi

# 1) If no args → interactive shell (when run with -it)
if [[ $# -eq 0 ]]; then
  exec /bin/bash -l
fi

usage() {
  cat >&2 <<'USAGE'
Usage:
  arc [flags] <input.yml>
  rmg [flags] <input.py>

Run with a bind mount so the container can read your input file, e.g.

  # ARC
  docker run --rm -v "$PWD:/work" -w /work IMAGE arc my_case/input.yml

  # RMG
  docker run --rm -v "$PWD:/work" -w /work IMAGE rmg my_case/input.py

Notes:
- <input.yml>/<input.py> must be a non-flag argument
- if you pass flags (e.g. -n 8), put them before the file: rmg -n 8 input.py
USAGE
}


# Return the first non-flag arg
first_nonflag() {
  local after_ddash="no"
  for a in "$@"; do
    if [[ "$after_ddash" == "no" && "$a" == "--" ]]; then
      after_ddash="yes"
      continue
    fi
    if [[ "$after_ddash" == "yes" || "$a" != -* ]]; then
      echo "$a"
      return 0
    fi
  done
  return 1
}

# Show usage on -h/--help for arc/rmg if no file was given
wants_help_no_file() {
  for a in "$@"; do
    [[ "$a" == "-h" || "$a" == "--help" ]] && return 0
  done
  return 1
}

# 2) Subcommands: rmg / arc
cmd="$1"; shift || true
case "$cmd" in
  arc|rmg)
    if wants_help_no_file "$@" && ! first_nonflag "$@" >/dev/null; then
      usage
      exit 0
    fi
    if ! file_arg="$(first_nonflag "$@")"; then
      usage
      exit 64   # EX_USAGE
    fi

    if [[ ! -f "$file_arg" ]]; then
      echo "Error: input file not found inside container: $file_arg" >&2
      echo "Tip: mount and set workdir, e.g.:  docker run -v \"\$PWD:/work\" -w /work IMAGE $cmd $file_arg" >&2
      exit 66   # EX_NOINPUT
    fi

    # no defaults: user must provide their file path
    if [[ "$cmd" == "arc" ]]; then
      exec micromamba run -n arc_env python /home/mambauser/Code/ARC/ARC.py "$@"
    else
      exec micromamba run -n rmg_env python /home/mambauser/Code/RMG-Py/rmg.py "$@"
    fi
    ;;

  *)
    # Pass-through for CI or ad-hoc shell commands
    exec "$cmd" "$@"
    ;;
esac
