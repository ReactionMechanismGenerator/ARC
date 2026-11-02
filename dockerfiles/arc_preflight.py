"""
A pre-flight check for the ARC container's ``~/.arc`` settings overlay.

ARC loads user overrides from ``$HOME/.arc/{settings,submit,inputs}.py`` (see ``arc/imports.py``).
That loader is deliberately forgiving: a ``settings.py`` which raises ``ImportError`` is skipped
without a word, and ARC then continues with the repository's dummy servers
(``server1.host.edu``, ``<username>``). Inside a container the usual cause is a mis-typed bind
mount, and the only symptom is a run that spends hours failing to reach a host that never
existed. This script turns that into an immediate, actionable message, and additionally reports
SSH identities that cannot work inside the container.

It deliberately imports ``settings`` the same way ``arc.imports`` does -- as a top-level module
found on ``sys.path`` -- rather than importing ARC itself, which is slow and would obscure the
very failure being checked for.

Usage:
    arc_preflight.py [overlay directory]

    The directory defaults to ``DEFAULT_ARC_DIR``. The entrypoint passes its own constant so
    that the two cannot drift apart, and the tests pass a temporary directory.

Exit codes:
    0: the overlay is usable, or there is no ``settings.py`` to check. Warnings may still
       have been printed.
    1: ``settings.py`` is present but raised while being imported. The caller should abort.
"""

import os
import sys
import traceback


DEFAULT_ARC_DIR = '/home/mambauser/.arc'
DUMMY_ADDRESS_SUFFIX = '.host.edu'
DUMMY_USERNAME = '<username>'
DEFAULT_KEY_NAMES = ('id_rsa', 'id_ecdsa', 'id_ed25519')


def warn(message: str) -> None:
    """
    Print an actionable warning to stderr.

    Args:
        message (str): The warning text, without a prefix.
    """
    print(f'preflight: warning: {message}', file=sys.stderr)


def load_local_settings(arc_dir: str):
    """
    Import ``settings.py`` from ``arc_dir`` exactly as ``arc.imports`` does.

    Args:
        arc_dir (str): The directory holding the personal ARC settings.

    Returns:
        module: The imported local settings module.
    """
    if arc_dir not in sys.path:
        sys.path.insert(0, arc_dir)
    import settings
    return settings


def ssh_dir() -> str:
    """
    Returns:
        str: The path of the container user's ``.ssh`` directory.
    """
    return os.path.join(os.path.expanduser('~'), '.ssh')


def check_key(name: str, cfg: dict) -> None:
    """
    Warn when a server's SSH identity cannot be used from inside the container.

    ``key`` is the path of a private key on the machine running ARC, and is optional: without it
    paramiko falls back to a running ssh-agent and then to the default key paths. Both routes are
    legitimate, so nothing here is fatal -- a server that is configured but never used must not
    abort the run.

    Args:
        name (str): The server name.
        cfg (dict): The server settings.
    """
    key = cfg.get('key')
    if key:
        if not os.path.isfile(key):
            warn(f"server '{name}' sets key '{key}', which does not exist inside the container. "
                 f"Mount it read-only, e.g.  -v \"$HOME/.ssh/id_ed25519:{key}:ro\"  , or remove "
                 f"'key' from the server entry and forward your ssh-agent instead.")
        elif not os.access(key, os.R_OK):
            warn(f"server '{name}' sets key '{key}', which exists but is not readable by uid "
                 f"{os.getuid()}. Re-run with  -e PUID=$(id -u) -e PGID=$(id -g)  so the container "
                 f"user matches the owner of the mount.")
        return
    if os.environ.get('SSH_AUTH_SOCK'):
        return
    if any(os.path.isfile(os.path.join(ssh_dir(), key_name)) for key_name in DEFAULT_KEY_NAMES):
        return
    warn(f"server '{name}' sets no 'key', and this container has neither a forwarded ssh-agent "
         f"(SSH_AUTH_SOCK is empty) nor a default key under {ssh_dir()}. Forward your agent with  "
         f"-v \"$SSH_AUTH_SOCK:/ssh-agent\" -e SSH_AUTH_SOCK=/ssh-agent  , or mount a private key "
         f"and point 'key' at it.")


def check_host_keys(name: str, cfg: dict, known_hosts: str) -> None:
    """
    Warn when strict host key checking cannot succeed for lack of a seeded ``known_hosts``.

    Args:
        name (str): The server name.
        cfg (dict): The server settings.
        known_hosts (str): The path paramiko reads known host keys from.
    """
    if not cfg.get('strict_host_key_checking'):
        return
    if os.path.isfile(known_hosts):
        return
    warn(f"server '{name}' sets strict_host_key_checking, but {known_hosts} does not exist inside "
         f"the container, so every connection will be refused. Mount your host keys at that exact "
         f"path, or seed them with  ssh-keyscan {cfg.get('address', 'HOST')} >> {known_hosts}  .")


def main(arc_dir: str) -> int:
    """
    Run the pre-flight checks.

    Args:
        arc_dir (str): The directory holding the personal ARC settings.

    Returns:
        int: 0 when the overlay is usable, 1 when it is present but unimportable.
    """
    settings_path = os.path.join(arc_dir, 'settings.py')
    if not os.path.isfile(settings_path):
        warn(f'{settings_path} does not exist, so ARC will use its dummy server settings. '
             f'Nothing to check.')
        return 0
    try:
        local_settings = load_local_settings(arc_dir)
    except ImportError:
        traceback.print_exc()
        print(f'preflight: error: {settings_path} could not be imported (traceback above).',
              file=sys.stderr)
        print('preflight: ARC catches exactly this error and carries on with its dummy server '
              'settings, without printing anything, so the run is being stopped here instead.',
              file=sys.stderr)
        return 1
    except Exception:
        traceback.print_exc()
        print(f'preflight: error: {settings_path} raised while being imported (traceback above).',
              file=sys.stderr)
        print('preflight: ARC tolerates only an ImportError here, so it would abort on this '
              'too, a moment later and with nothing but the traceback. Fix the file and re-run.',
              file=sys.stderr)
        return 1
    servers = getattr(local_settings, 'servers', None)
    if not isinstance(servers, dict) or not servers:
        warn(f'{arc_dir}/settings.py defines no non-empty "servers" dict, so ARC will use its '
             f'dummy server settings. This is expected only for runs that submit nothing.')
        return 0
    known_hosts = os.path.join(ssh_dir(), 'known_hosts')
    for name, cfg in servers.items():
        if not isinstance(cfg, dict):
            warn(f"server '{name}' is not a settings dictionary; ARC will not be able to use it.")
            continue
        address = cfg.get('address')
        if not address:
            continue
        if address.endswith(DUMMY_ADDRESS_SUFFIX) or cfg.get('un') == DUMMY_USERNAME:
            warn(f"server '{name}' still carries ARC's placeholder address or username "
                 f"('{address}', '{cfg.get('un')}'); it cannot be reached.")
            continue
        check_key(name, cfg)
        check_host_keys(name, cfg, known_hosts)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ARC_DIR))
