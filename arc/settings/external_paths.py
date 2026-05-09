"""
Discovery helpers for ARC's ML-based TS adapters' source checkouts and
pretrained artifacts.

This module hosts filesystem-discovery logic for sister-repo installations
of TS-search adapters (currently GoFlow and RitS). It is deliberately kept
out of ``arc/settings/settings.py``: ``settings`` holds static
configuration (dicts/constants), this module holds adapter-specific
filesystem-discovery logic.

GoFlow (Galustian et al., Digital Discovery 2025, 10.1039/D5DD00283D;
preprint doi.org/10.26434/chemrxiv-2025-bk2rh) is a flow-matching, E(3)-
equivariant TS-geometry generator. The "lean" practical fork lives at
https://github.com/heid-lab/goflow_lean . The shipped epoch_337.ckpt is a
45-byte placeholder (not a real Lightning checkpoint), so the ckpt finder
rejects anything below 1 MB. The shipped feat_dict_organic.pkl is a real
(small) pickle of about 387 bytes — feat dicts are inherently tiny — so
the feat-dict finder only guards against trivially-small (<100 B) files
and otherwise accepts whatever is in place. Real artifacts can be supplied
via env-var overrides (``ARC_GOFLOW_REPO``, ``ARC_GOFLOW_CKPT``,
``ARC_GOFLOW_FEAT_DICT``).

RitS (Right into the Saddle, Isayev lab, 10.26434/chemrxiv.15001681/v1)
is a flow-matching TS generator that handles bimolecular reactions and
charged species. The upstream repository lives at
https://github.com/isayevlab/RitS . The shipped pretrained checkpoint is
~364 MB (downloaded from Zenodo by ``devtools/install_rits.sh``); it is
located by env-var override (``ARC_RITS_REPO``, ``ARC_RITS_CKPT``) or by
filesystem convention.
"""

import os


_GOFLOW_CKPT_MIN_SIZE = 1_000_000      # any real Lightning ckpt is >> 1 MB
_GOFLOW_FEAT_DICT_MIN_SIZE = 100       # rejects only trivially-empty stubs


def _arc_root() -> str:
    """
    Return the absolute path of the ARC repo root.

    Returns:
        str: Absolute path of the ARC repo (this file lives at
        ``<arc_root>/arc/settings/external_paths.py``).
    """
    # this file: arc/settings/external_paths.py — three dirnames → repo root.
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _goflow_sibling_of_arc() -> str:
    """
    Return the conventional sibling-of-ARC location ``<parent-of-arc>/goflow_lean``.

    Returns:
        str: Absolute path candidate (no existence check).
    """
    return os.path.join(os.path.dirname(_arc_root()), 'goflow_lean')


def _rits_sibling_of_arc() -> str:
    """
    Return the conventional sibling-of-ARC location ``<parent-of-arc>/RitS``.

    Returns:
        str: Absolute path candidate (no existence check).
    """
    return os.path.join(os.path.dirname(_arc_root()), 'RitS')


def find_goflow_repo() -> str | None:
    """
    Locate a goflow_lean source checkout.

    Used by the GoFlow TS adapter to find ``src/goflow/configs/`` (Hydra
    configs) and the (validated) shipped ``data/RDB7/`` artifacts.

    Search order:
        1. ``ARC_GOFLOW_REPO`` environment variable (explicit override).
        2. ``~/Code/goflow_lean`` (default for ARC dev machines).
        3. Sibling-of-ARC location ``<parent-of-arc-repo>/goflow_lean`` —
           matches what ``devtools/install_goflow.sh`` produces.

    A directory is considered "found" only if it contains
    ``src/goflow/__init__.py`` (the package entry point).

    Returns:
        str | None: Absolute path to the checkout, or ``None`` if no
        candidate was located.
    """
    home = os.getenv('HOME') or os.path.expanduser('~')
    candidates = []
    env_override = os.getenv('ARC_GOFLOW_REPO')
    if env_override:
        candidates.append(env_override)
    candidates.append(os.path.join(home, 'Code', 'goflow_lean'))
    candidates.append(_goflow_sibling_of_arc())
    for path in candidates:
        if path and os.path.isfile(os.path.join(path, 'src', 'goflow', '__init__.py')):
            return os.path.abspath(path)
    return None


def find_goflow_ckpt(repo_path: str | None) -> str | None:
    """
    Locate a pretrained GoFlow Lightning checkpoint.

    Validates by file size to reject the LFS-pointer placeholder (45 bytes)
    shipped in goflow_lean@main.

    Search order:
        1. ``ARC_GOFLOW_CKPT`` env-var override.
        2. ``<repo_path>/data/RDB7/epoch_*.ckpt`` — any ``epoch_NNN.ckpt``
           past the size guard. Matches both the Zenodo-installed
           ``epoch_316.ckpt`` and the upstream-canonical ``epoch_337.ckpt``;
           multiple matches are sorted by epoch number descending so the
           newest wins.

    A file is considered valid iff size >= 1 MB. ``torch.load``-level
    validation is deferred to the adapter's runtime (this module stays
    torch-free).

    Args:
        repo_path (str | None): The goflow_lean checkout to search inside
            (``find_goflow_repo()`` output is the typical input).

    Returns:
        str | None: Absolute path to a real ckpt, or ``None``.
    """
    env_override = os.getenv('ARC_GOFLOW_CKPT')
    if env_override and os.path.isfile(env_override) \
            and os.path.getsize(env_override) >= _GOFLOW_CKPT_MIN_SIZE:
        return os.path.abspath(env_override)
    if repo_path:
        ckpt_dir = os.path.join(repo_path, 'data', 'RDB7')
        if os.path.isdir(ckpt_dir):
            candidates = []
            for name in os.listdir(ckpt_dir):
                if not (name.startswith('epoch_') and name.endswith('.ckpt')):
                    continue
                full = os.path.join(ckpt_dir, name)
                if os.path.isfile(full) and os.path.getsize(full) >= _GOFLOW_CKPT_MIN_SIZE:
                    try:
                        epoch_num = int(name[len('epoch_'):-len('.ckpt')])
                    except ValueError:
                        epoch_num = -1
                    candidates.append((epoch_num, full))
            if candidates:
                candidates.sort(reverse=True)
                return os.path.abspath(candidates[0][1])
    return None


def find_goflow_feat_dict(repo_path: str | None) -> str | None:
    """
    Locate the GoFlow atom-feature codebook pickle.

    Validates by file size to reject only trivially-empty (<100 B) files;
    the in-repo 387-byte file is a real (tiny) pickle and is accepted
    as-is.

    Search order:
        1. ``ARC_GOFLOW_FEAT_DICT`` env-var override.
        2. ``<repo_path>/data/RDB7/feat_dict_organic.pkl``.

    Pickle-level validation is deferred to the adapter's runtime.

    Args:
        repo_path (str | None): The goflow_lean checkout to search inside.

    Returns:
        str | None: Absolute path to a feat-dict pickle, or ``None``.
    """
    candidates = []
    env_override = os.getenv('ARC_GOFLOW_FEAT_DICT')
    if env_override:
        candidates.append(env_override)
    if repo_path:
        candidates.append(os.path.join(repo_path, 'data', 'RDB7', 'feat_dict_organic.pkl'))
    for path in candidates:
        if path and os.path.isfile(path) and os.path.getsize(path) >= _GOFLOW_FEAT_DICT_MIN_SIZE:
            return os.path.abspath(path)
    return None


def find_rits_repo() -> str | None:
    """
    Locate a RitS source checkout.

    Used by the RitS TS adapter to find ``scripts/sample_transition_state.py``
    and ``scripts/conf/rits.yaml``, which are not part of the importable
    ``megalodon`` package.

    Search order:
        1. ``ARC_RITS_REPO`` environment variable (explicit override).
        2. ``~/Code/RitS`` (default for ARC dev machines).
        3. Sibling-of-ARC location ``<parent-of-arc-repo>/RitS`` —
           matches what ``devtools/install_rits.sh`` produces.

    A directory is considered "found" only if it contains
    ``scripts/sample_transition_state.py`` (the inference entry point).

    Returns:
        str | None: Absolute path to the checkout, or ``None`` if no
        candidate was located.
    """
    home = os.getenv('HOME') or os.path.expanduser('~')
    candidates = []
    env_override = os.getenv('ARC_RITS_REPO')
    if env_override:
        candidates.append(env_override)
    candidates.append(os.path.join(home, 'Code', 'RitS'))
    candidates.append(_rits_sibling_of_arc())
    for path in candidates:
        if path and os.path.isfile(os.path.join(path, 'scripts', 'sample_transition_state.py')):
            return os.path.abspath(path)
    return None


def find_rits_ckpt(repo_path: str | None) -> str | None:
    """
    Locate the pretrained RitS checkpoint file (``rits.ckpt``).

    Search order:
        1. ``ARC_RITS_CKPT`` environment variable (explicit override).
        2. ``<repo_path>/data/rits.ckpt`` — what ``install_rits.sh`` writes.

    No size guard is applied: the upstream Zenodo-distributed checkpoint is
    a single canonical ~364 MB file and the install script verifies it via
    SHA-256 at install time. Lightning-level validation is deferred to the
    adapter's runtime (this module stays torch-free).

    Args:
        repo_path (str | None): The RitS repo path returned by
            ``find_rits_repo()``. If ``None``, only the env-var override
            is consulted.

    Returns:
        str | None: Absolute path to the checkpoint, or ``None``.
    """
    env_override = os.getenv('ARC_RITS_CKPT')
    if env_override and os.path.isfile(env_override):
        return os.path.abspath(env_override)
    if repo_path:
        candidate = os.path.join(repo_path, 'data', 'rits.ckpt')
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    return None
