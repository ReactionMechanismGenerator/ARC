#!/usr/bin/env bash
set -euo pipefail

SSH_DIR="/home/mambauser/.ssh"
ARC_SETTINGS_DIR="/home/mambauser/.arc"
ARC_PREFLIGHT="/usr/local/bin/arc_preflight.py"

# Key file names paramiko looks for under $HOME/.ssh when no explicit key is supplied.
# (paramiko >= 4 dropped DSA support, hence no id_dsa.)
SSH_DEFAULT_KEYS=(id_rsa id_ecdsa id_ed25519)

warn() { echo "entrywrapper: warning: $*" >&2; }

# True when "$1" sits on a different device than its parent directory, i.e. it is a bind
# mount from the host. Such paths must never be chown'ed or chmod'ed: they may be mounted
# read-only (the chown would simply fail), and when they are writable the change would
# leak out to the user's real files on the host.
is_bind_mount() {
  local path="$1" parent dev_path dev_parent
  parent="$(dirname "$path")"
  dev_path="$(stat -c %d "$path" 2>/dev/null || echo 'no-such-path')"
  dev_parent="$(stat -c %d "$parent" 2>/dev/null || echo 'no-such-parent')"
  [[ "$dev_path" != "$dev_parent" ]]
}

# The single quotes are deliberate: $1 must be expanded by the inner shell, not by this one.
# shellcheck disable=SC2016
readable_by_mambauser() {
  runuser -u mambauser -- bash -c '[[ -r "$1" ]]' _ "$1"
}

# shellcheck disable=SC2016
rw_by_mambauser() {
  runuser -u mambauser -- bash -c '[[ -r "$1" && -w "$1" ]]' _ "$1"
}

# Make a forwarded SSH agent socket usable after the privilege drop, or disable it.
# Sets SSH_AUTH_SOCK to "" when the socket cannot be used, so the SSH client falls back to
# key files instead of failing against a socket it cannot open.
#
# The supported way to gain access is the PUID/PGID remap above: an agent socket is mode 0600
# and owned by the host user, so a container user with that same uid can open it and nothing
# needs changing. Widening the socket's mode is NOT done by default. The bind mount shares the
# inode with the host, so `chmod o+rw` mutates the user's real, live agent socket, making it
# readable and writable by every other user on the host for as long as their agent runs. The
# entrypoint cannot undo it either: it hands off with `exec`, so no EXIT trap would ever fire.
# That contradicts the rule applied to bind-mounted paths everywhere else in this file, so it
# is available only on explicit request via ARC_WIDEN_AGENT_SOCKET=1, and says what it did.
prepare_ssh_agent_socket() {
  local sock="${SSH_AUTH_SOCK:-}"
  if [[ -z "$sock" ]]; then
    return 0
  fi
  if [[ ! -S "$sock" ]]; then
    warn "SSH_AUTH_SOCK is set to '$sock', which is not a socket inside the container; ignoring it."
    warn "forward the host agent with:  -v \"\$SSH_AUTH_SOCK:/ssh-agent\" -e SSH_AUTH_SOCK=/ssh-agent"
    SSH_AUTH_SOCK=""
    return 0
  fi
  if rw_by_mambauser "$sock"; then
    return 0
  fi
  if [[ "${ARC_WIDEN_AGENT_SOCKET:-0}" == "1" ]]; then
    if chmod o+rw "$sock" && rw_by_mambauser "$sock"; then
      warn "ARC_WIDEN_AGENT_SOCKET=1: relaxed the mode of the agent socket '$sock' to $(stat -c %a "$sock")."
      warn "this changed the socket ON THE HOST, where it is now readable and writable by any"
      warn "local user, and it is NOT restored when this container exits."
      warn "restore it yourself afterwards with:  chmod 600 \"\$SSH_AUTH_SOCK\""
      return 0
    fi
    warn "ARC_WIDEN_AGENT_SOCKET=1 was set, but relaxing the mode did not make '$sock' usable."
  fi
  warn "the forwarded SSH agent socket '$sock' is not accessible to mambauser (uid $(id -u mambauser))."
  warn "re-run with  -e PUID=\$(id -u) -e PGID=\$(id -g)  so the container user matches the socket's owner;"
  warn "that is the supported fix and it leaves the socket on the host untouched."
  warn "continuing without agent forwarding."
  SSH_AUTH_SOCK=""
}

# Prepare /home/mambauser/.ssh. A bind-mounted .ssh is left completely untouched, including
# when it is read-only: paramiko, unlike the OpenSSH CLI, does not require 0600 on key files,
# so a read-only mount owned by the host user works as long as it is readable.
prepare_ssh_dir() {
  if [[ ! -e "$SSH_DIR" ]]; then
    return 0
  fi
  if is_bind_mount "$SSH_DIR"; then
    if ! readable_by_mambauser "$SSH_DIR"; then
      warn "the mounted $SSH_DIR is not readable by mambauser (uid $(id -u mambauser))."
      warn "re-run with  -e PUID=\$(id -u) -e PGID=\$(id -g)  so the container user matches the mount owner."
    fi
    return 0
  fi
  if ! chown mambauser:mambauser "$SSH_DIR" 2>/dev/null; then
    warn "failed to change ownership of $SSH_DIR to mambauser:mambauser."
  fi
  local key
  for key in "${SSH_DEFAULT_KEYS[@]}"; do
    if [[ -e "$SSH_DIR/$key" ]] && ! readable_by_mambauser "$SSH_DIR/$key"; then
      warn "$SSH_DIR/$key exists but is not readable by mambauser."
    fi
  done
}

# Fail fast on a mis-mounted ARC settings overlay.
#
# ARC reads $HOME/.arc/{settings,submit,inputs}.py, and arc/imports.py swallows an ImportError
# from settings.py without a word: ARC then runs against the repository's *dummy* servers
# ('server1.host.edu', '<username>'). In a container a mis-typed mount path or a settings.py
# whose own imports are unavailable produces exactly that, and the only symptom is a run that
# fails hours later against a host that never existed. arc_preflight.py imports the overlay the
# same way ARC does and reports what it finds; an overlay that is present but unimportable is a
# configuration error, not something to continue past.
#
# Set ARC_SKIP_PREFLIGHT=1 to bypass this entirely.
preflight_arc_settings() {
  if [[ "${ARC_SKIP_PREFLIGHT:-0}" == "1" ]]; then
    return 0
  fi
  if [[ ! -f "$ARC_SETTINGS_DIR/settings.py" ]]; then
    warn "no settings.py under $ARC_SETTINGS_DIR, so ARC will use its dummy server settings."
    warn "to drive a remote cluster, mount your settings with  -v \"\$HOME/.arc:$ARC_SETTINGS_DIR:ro\""
    warn "if you did mount it, the source side must be the directory *containing* settings.py."
    return 0
  fi
  if [[ ! -f "$ARC_PREFLIGHT" ]]; then
    warn "$ARC_PREFLIGHT is missing from the image; skipping the settings pre-flight check."
    return 0
  fi
  local output status=0
  output="$(micromamba run -n arc_env python "$ARC_PREFLIGHT" "$ARC_SETTINGS_DIR" 2>&1)" || status=$?
  if [[ -n "$output" ]]; then
    printf '%s\n' "$output" >&2
  fi
  if [[ "$status" -ne 0 ]]; then
    echo "Error: the ARC settings overlay mounted at $ARC_SETTINGS_DIR is unusable (see above)." >&2
    exit 78   # EX_CONFIG
  fi
}

# Highest ID reserved for system accounts on Debian/Ubuntu. Sharing one of those with
# mambauser would hand it that account's file access, so such a collision stays fatal.
SYSTEM_ID_MAX=999

# True when a running process belongs to the given real UID (or GID, with field "Gid:").
# Read from /proc rather than via ps, which the image does not install.
id_has_running_process() {
  local field="$1" id="$2"
  [[ -r /proc/1/status ]] || return 1
  awk -v field="$field:" -v want="$id" \
      '$1 == field && $2 == want { found = 1 } END { exit !found }' /proc/[0-9]*/status 2>/dev/null
}

# Explain an ID collision that must stay fatal, naming the flag to drop. Always returns 1.
refuse_remap() {
  local kind="$1" id="$2" holder="$3" reason="$4" var
  var="P${kind^^}"
  echo "Error: cannot remap mambauser to $kind $id: it is already held by '$holder', $reason." >&2
  echo "Drop  -e $var=$id  to keep mambauser's built-in IDs, or pass an unused ID instead." >&2
  echo "Bind-mounted files would then be owned by uid $(id -u mambauser)/gid $(id -g mambauser)." >&2
  return 1
}

# Decide whether a requested ID that another account already holds may be shared with
# mambauser. Sharing is safe for an ordinary, idle account: file permissions are numeric, so
# mambauser gets exactly the access the bind mounts need, and nothing is deleted or renamed.
may_share_id() {
  local kind="$1" id="$2" holder="$3" field="$4"
  if [[ "$id" -eq 0 ]]; then
    refuse_remap "$kind" "$id" "$holder" "the superuser"
    return 1
  fi
  if [[ "$id" -le "$SYSTEM_ID_MAX" ]]; then
    refuse_remap "$kind" "$id" "$holder" "a system account reserved by the distribution"
    return 1
  fi
  if id_has_running_process "$field" "$id"; then
    refuse_remap "$kind" "$id" "$holder" "an account with running processes"
    return 1
  fi
  warn "$kind $id is also held by the idle account '$holder'; sharing it with mambauser."
  return 0
}

# 0) If root, optionally remap mambauser UID/GID for bind mounts, then drop privileges.
#
# A collision here used to be fatal outright. The base image ships an unused 'ubuntu' account at
# 1000:1000, which is exactly what `-e PUID=$(id -u) -e PGID=$(id -g)` asks for on a typical
# Linux desktop, so that rule aborted the container for most users. The Dockerfile now removes
# that account, but the collision can return from a rebased base image or a derived one, so it
# is also handled here instead of trusting the image alone.
if [[ "$(id -u)" -eq 0 ]]; then
  if [[ -n "${PGID:-}" ]]; then
    current_gid="$(id -g mambauser)"
    if [[ "$PGID" != "$current_gid" ]]; then
      existing_group=""
      if getent group "$PGID" >/dev/null; then
        existing_group="$(getent group "$PGID" | cut -d: -f1)"
      fi
      if [[ -n "$existing_group" && "$existing_group" != "mambauser" ]]; then
        may_share_id gid "$PGID" "$existing_group" Gid || exit 1
        groupmod -o -g "$PGID" mambauser
      else
        groupmod -g "$PGID" mambauser
      fi
    fi
  fi
  if [[ -n "${PUID:-}" ]]; then
    current_uid="$(id -u mambauser)"
    if [[ "$PUID" != "$current_uid" ]]; then
      existing_user=""
      if getent passwd "$PUID" >/dev/null; then
        existing_user="$(getent passwd "$PUID" | cut -d: -f1)"
      fi
      if [[ -n "$existing_user" && "$existing_user" != "mambauser" ]]; then
        may_share_id uid "$PUID" "$existing_user" Uid || exit 1
        usermod -o -u "$PUID" mambauser
      else
        usermod -u "$PUID" mambauser
      fi
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

  # SSH setup runs after the UID/GID remap so the checks reflect the final mambauser IDs.
  prepare_ssh_agent_socket
  prepare_ssh_dir

  if [[ "${ENTRYWRAPPER_AS_USER:-0}" != "1" ]]; then
    # runuser resets HOME/SHELL/USER/LOGNAME/PATH. SSH_AUTH_SOCK is passed on explicitly so
    # that it survives the privilege drop, and is unset explicitly when the socket was found
    # to be unusable above.
    drop_env=()
    if [[ -z "${SSH_AUTH_SOCK:-}" ]]; then
      drop_env+=(-u SSH_AUTH_SOCK)
    fi
    drop_env+=(ENTRYWRAPPER_AS_USER=1)
    if [[ -n "${SSH_AUTH_SOCK:-}" ]]; then
      drop_env+=("SSH_AUTH_SOCK=$SSH_AUTH_SOCK")
    fi
    exec runuser -u mambauser -- env "${drop_env[@]}" /usr/local/bin/entrywrapper.sh "$@"
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
- to let ARC submit ESS jobs to a remote cluster over SSH, either forward your host SSH
  agent with  -v "$SSH_AUTH_SOCK:/ssh-agent" -e SSH_AUTH_SOCK=/ssh-agent  (preferred, keys
  never enter the container), or mount your keys read-only with
  -v "$HOME/.ssh:/home/mambauser/.ssh:ro"; pass -e PUID=$(id -u) -e PGID=$(id -g) so the
  container user matches the owner of those mounts
- your server definitions live in your personal ARC settings, which must be mounted with
  -v "$HOME/.arc:/home/mambauser/.arc:ro"; without it ARC runs against its dummy servers.
  `arc` checks this before starting and refuses to run on an unimportable settings.py
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
      preflight_arc_settings
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
