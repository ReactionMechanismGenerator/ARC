"""Invoke a script under a sibling conda/mamba env, isolated from ARC's env.

ARC runs inside ``arc_env``. Several adapters (AutoTST, GCN, TorchANI)
shell out to scripts that live in their *own* envs (``tst_env``,
``ts_gcn_env``, ``tani_env``). Running the target env's ``python``
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
import shutil
import subprocess
from pathlib import Path

from arc.common import get_logger

logger = get_logger()


def env_name_from_python(python_executable: str) -> str:
    """Derive the env name from a path like ``.../envs/<name>/bin/python``.

    ARC's settings expose target Python interpreters as full paths
    (``AUTOTST_PYTHON``, ``TS_GCN_PYTHON``, ``TANI_PYTHON``). The env
    name we need for ``<launcher> run -n <name>`` is the directory two
    levels above the binary, under ``envs/``.
    """
    parts = Path(python_executable).resolve().parts
    if "envs" in parts:
        idx = parts.index("envs")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    raise ValueError(
        f"Cannot derive an env name from {python_executable!r}; "
        "expected a path of the form '.../envs/<env>/bin/python'."
    )


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
) -> subprocess.CompletedProcess:
    """Run ``python script_path *script_args`` inside the env that owns
    ``python_executable``, isolated from ARC's process env.

    Returns the :class:`subprocess.CompletedProcess`. ``check=True``
    raises on non-zero exit. Args are passed as a list, so no shell
    quoting concerns.
    """
    env_name = env_name_from_python(python_executable)
    launcher, extra_flags = _detect_launcher()
    argv = [
        launcher, "run", *extra_flags,
        "-n", env_name,
        "python", script_path,
        *script_args,
    ]
    logger.debug("env-run: %s", " ".join(argv))
    return subprocess.run(argv, check=check)
