"""
Utilities for locating CREST executables and activation commands.
"""

import os
import re
import shutil
import sys
from typing import Optional, Tuple


def parse_version(folder_name: str) -> Tuple[int, int, int]:
    """
    Parse a version from a folder name.

    Supports patterns such as ``3.0.2``, ``v212``, ``2.1``, ``2``.
    """
    version_regex = re.compile(r"(?:v?(\d+)(?:\.(\d+))?(?:\.(\d+))?)", re.IGNORECASE)
    match = version_regex.search(folder_name)
    if not match:
        return 0, 0, 0

    major = int(match.group(1)) if match.group(1) else 0
    minor = int(match.group(2)) if match.group(2) else 0
    patch = int(match.group(3)) if match.group(3) else 0

    # Example: v212 -> (2, 1, 2)
    if major >= 100 and match.group(2) is None and match.group(3) is None:
        s = str(major).rjust(3, "0")
        major, minor, patch = int(s[0]), int(s[1]), int(s[2])

    return major, minor, patch


def find_highest_version_in_directory(directory: str, name_contains: str) -> Optional[str]:
    """
    Find the ``crest`` executable under the highest-version matching subdirectory.
    """
    if not os.path.exists(directory):
        return None

    highest_version_path = None
    highest_version = ()
    for folder in os.listdir(directory):
        file_path = os.path.join(directory, folder)
        if name_contains.lower() in folder.lower() and os.path.isdir(file_path):
            crest_path = os.path.join(file_path, "crest")
            if os.path.isfile(crest_path) and os.access(crest_path, os.X_OK):
                version = parse_version(folder)
                if highest_version == () or version > highest_version:
                    highest_version = version
                    highest_version_path = crest_path
    return highest_version_path


def find_crest_executable() -> Tuple[Optional[str], Optional[str]]:
    """
    Return ``(crest_path, env_cmd)``.

    ``env_cmd`` is a shell snippet to activate the environment if needed, otherwise ``""``.
    """
    # Priority 1: standalone builds in a configurable directory (default: /Local/ce_dana)
    standalone_dir = os.getenv("ARC_CREST_STANDALONE_DIR", "/Local/ce_dana")
    crest_path = find_highest_version_in_directory(standalone_dir, "crest")
    if crest_path and os.path.isfile(crest_path) and os.access(crest_path, os.X_OK):
        return crest_path, ""

    # Priority 2: Conda/Mamba/Micromamba envs
    home = os.path.expanduser("~")
    potential_env_paths = [
        os.path.join(home, "anaconda3", "envs", "crest_env", "bin", "crest"),
        os.path.join(home, "miniconda3", "envs", "crest_env", "bin", "crest"),
        os.path.join(home, "miniforge3", "envs", "crest_env", "bin", "crest"),
        os.path.join(home, ".conda", "envs", "crest_env", "bin", "crest"),
        os.path.join(home, "mambaforge", "envs", "crest_env", "bin", "crest"),
        os.path.join(home, "micromamba", "envs", "crest_env", "bin", "crest"),
    ]

    current_env_bin = os.path.dirname(sys.executable)
    potential_env_paths.insert(0, os.path.join(current_env_bin, "crest"))

    for crest_path in potential_env_paths:
        if os.path.isfile(crest_path) and os.access(crest_path, os.X_OK):
            env_marker = os.path.join("envs", "crest_env") + os.path.sep
            env_root = crest_path.split(env_marker)[0]
            if "micromamba" in crest_path:
                env_cmd = (
                    f"source {env_root}/etc/profile.d/micromamba.sh && "
                    f"micromamba activate crest_env"
                )
            elif any(name in env_root for name in ("anaconda3", "miniconda3", "miniforge3", "mambaforge", ".conda")):
                env_cmd = (
                    f"source {env_root}/etc/profile.d/conda.sh && "
                    f"conda activate crest_env"
                )
            else:
                env_cmd = ""
            return crest_path, env_cmd

    # Priority 3: PATH
    crest_in_path = shutil.which("crest")
    if crest_in_path:
        return crest_in_path, ""

    return None, None


__all__ = [
    "parse_version",
    "find_highest_version_in_directory",
    "find_crest_executable",
]

