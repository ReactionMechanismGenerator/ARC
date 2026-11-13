"""
Utilities for locating CREST executables and activation commands.
"""

import functools
import os
import re
import shutil
import sys
from typing import Iterator, Optional, Tuple


CREST_ENV_NAME = "crest_env"
STANDALONE_DIR_ENV_VAR = "ARC_CREST_STANDALONE_DIR"


def parse_version(folder_name: str) -> Tuple[int, int, int]:
    """
    Parse a version from a folder name.

    Supports patterns such as ``3.0.2``, ``v212``, ``2.1``, ``2``.
    A three-digit run such as ``v212`` is read as ``(2, 1, 2)``.

    Args:
        folder_name (str): The folder name to parse.

    Returns:
        Tuple[int, int, int]: The major, minor and patch version numbers, ``(0, 0, 0)`` if no version was found.
    """
    version_regex = re.compile(r"(?:v?(\d+)(?:\.(\d+))?(?:\.(\d+))?)", re.IGNORECASE)
    match = version_regex.search(folder_name)
    if not match:
        return 0, 0, 0

    major = int(match.group(1)) if match.group(1) else 0
    minor = int(match.group(2)) if match.group(2) else 0
    patch = int(match.group(3)) if match.group(3) else 0

    if major >= 100 and match.group(2) is None and match.group(3) is None:
        s = str(major).rjust(3, "0")
        major, minor, patch = int(s[0]), int(s[1]), int(s[2])

    return major, minor, patch


def find_highest_version_in_directory(directory: str, name_contains: str) -> Optional[str]:
    """
    Find the ``crest`` executable under the highest-version matching subdirectory.

    Args:
        directory (str): The directory to search in.
        name_contains (str): A substring which a subdirectory name must contain (case-insensitively).

    Returns:
        Optional[str]: The path to the executable, ``None`` if the directory is missing,
        unreadable, or holds no matching executable.
    """
    if not directory or not os.path.exists(directory):
        return None

    highest_version_path = None
    highest_version = ()
    try:
        folders = os.listdir(directory)
    except OSError:
        return None
    for folder in folders:
        file_path = os.path.join(directory, folder)
        if name_contains.lower() in folder.lower() and os.path.isdir(file_path):
            crest_path = os.path.join(file_path, "crest")
            if os.path.isfile(crest_path) and os.access(crest_path, os.X_OK):
                version = parse_version(folder)
                if highest_version == () or version > highest_version:
                    highest_version = version
                    highest_version_path = crest_path
    return highest_version_path


def _iter_ancestor_dirs(directory: str) -> Iterator[str]:
    """
    Iterate over a directory and each of its ancestors up to the filesystem root.

    Args:
        directory (str): The directory to start from.

    Yields:
        str: The next directory, starting with ``directory`` itself.
    """
    current = directory
    while current:
        yield current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent


def find_env_activation_command(crest_path: str, env_name: str = CREST_ENV_NAME) -> str:
    """
    Determine a shell command which activates the environment holding a crest executable.

    A command is only generated when the executable lives under an ``envs/<env_name>/`` directory
    and an ``etc/profile.d`` activation script exists in that environment's directory or in one of
    its ancestors. Otherwise an empty string is returned, and the executable is meant to be invoked
    by its absolute path. Note that a user data directory such as ``~/.conda`` holds environments
    but no activation script, and therefore yields an empty string.

    Args:
        crest_path (str): The path to the crest executable.
        env_name (str, optional): The name of the environment to activate.

    Returns:
        str: A shell snippet activating the environment, an empty string if no activation script was found.
    """
    env_marker = os.path.join("envs", env_name) + os.path.sep
    if env_marker not in crest_path:
        return ""
    env_root = crest_path.split(env_marker)[0].rstrip(os.path.sep)
    if not env_root:
        return ""
    for directory in _iter_ancestor_dirs(env_root):
        micromamba_sh = os.path.join(directory, "etc", "profile.d", "micromamba.sh")
        if os.path.isfile(micromamba_sh):
            return f"source {micromamba_sh} && micromamba activate {env_name}"
        conda_sh = os.path.join(directory, "etc", "profile.d", "conda.sh")
        if os.path.isfile(conda_sh):
            return f"source {conda_sh} && conda activate {env_name}"
    return ""


def find_crest_executable(standalone_dir: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    """
    Locate a crest executable along with the command needed to activate its environment.

    Locations are searched in this order: the highest-version standalone build under
    ``standalone_dir``, which defaults to the directory named by the ``ARC_CREST_STANDALONE_DIR``
    environment variable and is skipped when neither is set; the active Python environment and the
    common conda, mamba and micromamba environment locations under the user's home directory;
    and finally ``PATH``.

    Args:
        standalone_dir (str, optional): A directory holding versioned standalone CREST builds.

    Returns:
        Tuple[Optional[str], Optional[str]]: The path to the crest executable and a shell snippet
        activating its environment (an empty string when no activation is required),
        or ``(None, None)`` when no executable was found.
    """
    standalone_dir = standalone_dir if standalone_dir is not None else os.getenv(STANDALONE_DIR_ENV_VAR, "")
    if standalone_dir:
        crest_path = find_highest_version_in_directory(standalone_dir, "crest")
        if crest_path and os.path.isfile(crest_path) and os.access(crest_path, os.X_OK):
            return crest_path, ""

    home = os.path.expanduser("~")
    potential_env_paths = [os.path.join(os.path.dirname(sys.executable), "crest")]
    potential_env_paths += [os.path.join(home, root, "envs", CREST_ENV_NAME, "bin", "crest")
                            for root in ("anaconda3", "miniconda3", "miniforge3", ".conda",
                                         "mambaforge", "micromamba")]

    for crest_path in potential_env_paths:
        if os.path.isfile(crest_path) and os.access(crest_path, os.X_OK):
            return crest_path, find_env_activation_command(crest_path)

    crest_in_path = shutil.which("crest")
    if crest_in_path:
        return crest_in_path, ""

    return None, None


@functools.lru_cache(maxsize=None)
def get_crest_paths(standalone_dir: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    """
    Return the cached result of :func:`find_crest_executable`.

    The filesystem is only searched on the first call with a given ``standalone_dir``.
    Call ``get_crest_paths.cache_clear()`` to force a new search.

    Args:
        standalone_dir (str, optional): A directory holding versioned standalone CREST builds.

    Returns:
        Tuple[Optional[str], Optional[str]]: The crest executable path and its environment activation command.
    """
    return find_crest_executable(standalone_dir=standalone_dir)


__all__ = [
    "STANDALONE_DIR_ENV_VAR",
    "parse_version",
    "find_highest_version_in_directory",
    "find_env_activation_command",
    "find_crest_executable",
    "get_crest_paths",
]
