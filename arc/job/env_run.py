"""Invoke a script under a sibling conda/mamba env, isolated from ARC's env.

ARC runs inside ``arc_env``. Several adapters (AutoTST, GCN, TorchANI)
shell out to scripts that live in their *own* envs (``tst_env``,
``ts_gcn``, ``tani_env``). Running the target env's ``python``
binary directly leaves ARC's exported activation vars (``BABEL_LIBDIR``,
``LD_LIBRARY_PATH``, ``CONDA_PREFIX``, ...) bound to ``arc_env``'s
paths in the child, which causes ABI-mismatch crashes when shared
libraries in the child resolve plugins against the wrong env's tree.

Routing through a launcher's ``run`` subcommand makes the launcher
deactivate the caller env and re-activate the target, so the target
env's own ``activate.d`` hooks fire and bind those vars to its paths.

Three launchers are supported, in preference order:

1. ``conda`` — needs ``--no-capture-output`` to avoid buffering child
   stdio.
2. ``mamba`` — same parser as conda for ``run``; also needs
   ``--no-capture-output``.
3. ``micromamba`` — independent C++ reimplementation; streams stdio by
   default and **rejects** ``--no-capture-output``, so the flag must be
   omitted.

Buffering matters: without the right flag, conda/mamba hold the child's
stdout until exit, hiding tracebacks and progress.

The launcher is detected at call time, with the active one (per
``CONDA_EXE`` / ``MAMBA_EXE``) preferred when available.
"""

import os
import shlex
import shutil
import subprocess
from pathlib import Path

from arc.common import get_logger
from arc.imports import settings

logger = get_logger()


def env_prefix_from_python(python_executable: str) -> str:
    """Derive the env prefix from an interpreter path.

    ARC's settings expose target Python interpreters as full paths
    (``AUTOTST_PYTHON``, ``TS_GCN_PYTHON``, ``TANI_PYTHON``). The env
    prefix passed to ``<launcher> run -p <prefix>`` is the directory two
    levels above the binary (``<prefix>/bin/python``).

    Using a prefix path rather than ``-n <name>`` avoids assuming the
    env lives under a literal ``envs/`` segment — ``CONDA_ENVS_PATH``
    and bare-prefix mamba/micromamba layouts (e.g.
    ``/scratch/conda_envs/<env>/bin/python``) are both fine.

    Validation is lexical, NOT through ``Path.resolve()``: in real
    conda/mamba/micromamba envs ``<prefix>/bin/python`` is a symlink to
    ``python3.X``, so resolving first would replace the basename with
    ``python3.12`` (or similar) and trip the name check. The launcher
    follows its own interpreter, so all we need here is the prefix
    string the caller already gave us.
    """
    path = Path(python_executable)
    if path.name != "python" or path.parent.name != "bin":
        raise ValueError(
            f"Cannot derive an env prefix from {python_executable!r}; "
            "expected a path of the form '<prefix>/bin/python'."
        )
    return str(path.parent.parent)


def _run_flags_for(launcher_path: str) -> list[str]:
    """Return the per-launcher flags needed for ``run`` to stream stdio.

    Decided by the launcher's basename rather than which env var pointed
    us at it, so symlinks and odd ``MAMBA_EXE``-points-at-micromamba
    setups still get the right flag.
    """
    name = Path(launcher_path).name
    if name == "micromamba":
        return []
    return ["--no-capture-output"]


def _detect_launcher() -> tuple[str, list[str]]:
    """Return ``(launcher_path, extra_run_flags)``.

    Preference: whichever launcher is active in the current shell
    (``CONDA_EXE`` / ``MAMBA_EXE``), then conda → mamba → micromamba on
    PATH.
    """
    for env_var in ("CONDA_EXE", "MAMBA_EXE"):
        path = os.environ.get(env_var)
        if path and os.path.isfile(path):
            return path, _run_flags_for(path)
    for name in ("conda", "mamba", "micromamba"):
        found = shutil.which(name)
        if found:
            return found, _run_flags_for(found)
    raise FileNotFoundError(
        "No conda-family launcher (conda / mamba / micromamba) found on "
        "PATH. ARC's cross-env adapters (AutoTST/GCN/TorchANI) need one "
        "of these to launch their subprocess scripts in isolated envs."
    )


def run_in_conda_env(
    python_executable: str,
    script_path: str,
    *script_args: str,
    check: bool = False,
    strip_pythonpath: bool = False,
) -> subprocess.CompletedProcess:
    """Run ``python script_path *script_args`` inside the env that owns
    ``python_executable``, isolated from ARC's process env.

    stdout and stderr are captured and logged centrally — debug on
    success, warning (with both streams and the return code) on
    non-zero exit — so call sites don't each re-implement capture and
    error reporting. The captured streams are also exposed on the
    returned :class:`subprocess.CompletedProcess` (``.stdout`` /
    ``.stderr``) for callers that need to inspect them. ``check=True``
    raises ``CalledProcessError`` on non-zero exit. Args are passed as
    a list, so no shell quoting concerns.

    ``strip_pythonpath=True`` removes ``PYTHONPATH`` from the child's
    environment. The launcher's ``run`` re-fires the target env's
    activation hooks but leaves ``PYTHONPATH`` untouched, and PYTHONPATH
    entries shadow the target env's site-packages — so a stale source
    checkout on the caller's PYTHONPATH (e.g. an old KinBot clone in
    ``~/.bashrc``) would silently win over the env's installed package.
    Use it for adapters whose package must come from the target env
    itself. Leave it off for adapters that intentionally receive code
    via PYTHONPATH activation hooks set at env activation (those hooks
    still fire and re-add their paths inside the child either way).
    """
    env_prefix = env_prefix_from_python(python_executable)
    launcher, extra_flags = _detect_launcher()
    argv = [
        launcher, "run", *extra_flags,
        "-p", env_prefix,
        "python", script_path,
        *script_args,
    ]
    child_env = None
    if strip_pythonpath:
        child_env = {key: val for key, val in os.environ.items() if key != 'PYTHONPATH'}
    result = subprocess.run(argv, check=check, capture_output=True, text=True, env=child_env)
    if result.returncode:
        logger.warning(
            "env-run: %s exited with %d\ncmd: %s\nstdout:\n%s\nstderr:\n%s",
            script_path, result.returncode, " ".join(argv),
            result.stdout, result.stderr,
        )
    else:
        logger.debug(
            "env-run: %s exited 0\ncmd: %s\nstdout:\n%s\nstderr:\n%s",
            script_path, " ".join(argv), result.stdout, result.stderr,
        )
    return result


# ── RMG env invocation ──────────────────────────────────────────────────────

# Activation variables that ``arc_env`` exports into ARC's own process. A
# launcher's ``run`` deactivates the caller env and re-binds these to the
# target env, but invoking rmg_env's interpreter *directly* does not: they stay
# pointed at arc_env's tree and make the child resolve shared-library plugins
# against the wrong env, which is what silently kills Arkane's OpenBabel
# import. Only the direct-interpreter branch below has to scrub them.
# PYTHONPATH is included because the RMG helper scripts in arc/scripts/ import
# only rmgpy/arkane/rdkit plus a sibling ``common`` module (found via the
# script's own directory), so nothing there needs ARC on the path -- while a
# stale entry would shadow rmg_env's site-packages.
_ARC_ENV_ACTIVATION_VARS = (
    'CONDA_PREFIX', 'CONDA_PREFIX_1', 'CONDA_DEFAULT_ENV', 'CONDA_PROMPT_MODIFIER',
    'CONDA_SHLVL', 'LD_LIBRARY_PATH', 'BABEL_LIBDIR', 'BABEL_DATADIR',
    'PYTHONPATH', 'PYTHONHOME',
)

_NO_LAUNCHER_MSG = 'micromamba, mamba, or conda is required to run RMG helper scripts'


def rmg_env_command(py_args: str | list[str],
                    cwd: str | None = None,
                    env_vars: dict[str, str] | None = None,
                    suffix: str = '',
                    ) -> str:
    """Build a bash script that runs ``python <py_args>`` inside RMG's env.

    ARC shells out to RMG/Arkane helper scripts that must run under
    ``rmg_env``. Every call site used to carry its own copy of the
    launcher-selection ladder; they all go through this function instead.

    Resolution order, preserved from the call sites this replaced:
    ``MAMBA_EXE`` (exported by setup-micromamba in CI) → ``RMG_PYTHON`` from
    ARC's settings (needed on conda/mambaforge installs where micromamba's
    ``conda`` shim is broken) → a launcher found on PATH, hunted for under a
    login shell so that a conda initialization block in the user's profile is
    still honoured.

    Args:
        py_args (str | list[str]): Everything after ``python``. Passing a
                       ``list[str]`` is the safe, recommended form: each
                       element is one argv token, shell-quoted independently
                       via ``shlex.quote`` before joining, so caller-derived
                       values (e.g. paths from ``input.yml``) cannot break
                       out of their token or trigger command substitution.
                       Passing a plain ``str``, e.g. ``'-m arkane input.py'``
                       or ``f'{script_path} {in_path} {out_path}'``, is
                       spliced into the script verbatim as before; the caller
                       is entirely responsible for shell-quoting it.
        cwd (str, optional): A directory to change into before running.
        env_vars (dict, optional): Extra variables to export, e.g. ``RMG_DB_PATH``.
        suffix (str, optional): Appended verbatim to the python invocation, for
                                redirections such as ``' | tee -a stdout.log'``.
                                Process substitution is fine, the script is bash.
                                Never pass externally-derived/untrusted data here.

    Returns:
        str: A bash script. Run it with
             ``execute_command(command=..., shell=True, executable='/bin/bash')``.
    """
    env_name = settings.get('RMG_ENV_NAME', 'rmg_env')
    rmg_python = settings.get('RMG_PYTHON')
    if isinstance(py_args, list):
        py_args = ' '.join(shlex.quote(arg) for arg in py_args)

    preamble = ['set -euo pipefail']
    if cwd:
        preamble.append(f'cd {shlex.quote(cwd)}')
    for key, val in (env_vars or {}).items():
        preamble.append(f'export {key}={shlex.quote(val)}')

    mamba_exe = os.environ.get('MAMBA_EXE', '')
    if mamba_exe and os.path.isfile(mamba_exe):
        return '\n'.join(preamble + [
            f'{shlex.quote(mamba_exe)} run -n {env_name} python {py_args}{suffix}',
        ])

    if rmg_python and os.path.isfile(rmg_python):
        return '\n'.join(preamble + [
            f'unset {" ".join(_ARC_ENV_ACTIVATION_VARS)}',
            f'export PATH={shlex.quote(os.path.dirname(rmg_python))}:"$PATH"',
            f'{shlex.quote(rmg_python)} {py_args}{suffix}',
        ])

    # No launcher pinned by an env var and no configured interpreter: hunt for a
    # launcher on PATH. This runs under ``bash -l`` so the user's profile (where
    # conda's init block usually lives) is sourced first. The script is fed on
    # stdin via a quoted heredoc, which keeps it free of the nested shell
    # quoting the per-call-site copies of this ladder each had to get right.
    hunted = preamble + [
        'for _launcher in micromamba mamba conda; do',
        '    if command -v "$_launcher" >/dev/null 2>&1; then',
        f'        "$_launcher" run -n {env_name} python {py_args}{suffix}',
        '        exit $?',
        '    fi',
        'done',
        f'echo "{_NO_LAUNCHER_MSG}" >&2',
        'exit 1',
    ]
    body = '\n'.join(hunted)
    return f"bash -l <<'ARC_RMG_ENV_EOF'\n{body}\nARC_RMG_ENV_EOF"
