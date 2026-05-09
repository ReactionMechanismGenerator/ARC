"""
Fixture loader and lookup for the mockter adapter.

Loads a per-scenario YAML (schema v1) extracted from a real ARC project
directory by ``arc/testing/scripts/build_mockter_fixture.py`` and exposes
a typed lookup over it. The fixture is the single source of truth for
mockter's replays; this module is the only place schema knowledge lives.

Schema v1 (abbreviated):

    schema_version: 1
    provenance: { ... }
    species:
      <label>:
        smiles, multiplicity, charge
        conformers: [ {xyz, e_elect, isomorphic, raise?, ...}, ... ]
        opt:        {xyz, e_elect}
        fine_opt:   {xyz, e_elect}
        freq:       {freqs, zpe, hessian_block, ...}
        sp:         {e_elect, t1_diagnostic}
        composite:  {xyz, e_elect, hessian_block}
        scans:      [ {torsions: [[i,j,k,l]], energies, xyzs, raise?}, ... ]
        irc:        {forward: {...}, reverse: {...}}
    reactions: { <rxn_label>: {...} }
    ts:        { <ts_label>: {...} }   # same shape as a species, plus 'guesses'

A fixture entry may carry a ``raise:`` clause (``crash`` | ``oom`` | ``timeout``
| ``scf_nonconvergence`` | ``sigterm``) — when mockter looks up such an entry
it must abort instead of rendering, so that interrupt scenarios can be
encoded in the fixture without touching scheduler code.
"""

import os

from arc.common import read_yaml_file


SUPPORTED_SCHEMA_VERSIONS: set[int] = {1}

VALID_RAISE_KINDS: set[str] = {
    'crash', 'oom', 'timeout', 'scf_nonconvergence', 'sigterm',
}


class FixtureError(Exception):
    """Raised when a fixture file fails schema validation or a lookup is malformed."""


class FixtureEntry:
    """
    A single entry returned by ``Fixture.lookup`` — wraps a sub-dict
    from the fixture YAML and exposes ``raise:`` semantics.
    """

    def __init__(self, payload: dict, source_path: tuple[str, ...]):
        """
        Args:
            payload (dict): The leaf dict from the fixture (e.g. the ``opt`` block,
                            or one element of ``conformers``, or a scan/irc subentry).
            source_path (tuple[str, ...]): Tuple of keys describing where this
                                           entry came from (for error messages).
        """
        self._payload = payload
        self._source_path = source_path

    @property
    def payload(self) -> dict:
        """The raw leaf dict."""
        return self._payload

    @property
    def source_path(self) -> tuple[str, ...]:
        """Tuple of keys identifying this entry's location in the fixture."""
        return self._source_path

    def is_raise(self) -> bool:
        """
        Whether this entry encodes an interrupt instead of payload data.

        Returns:
            bool: True if the entry carries a ``raise:`` clause.
        """
        return isinstance(self._payload, dict) and 'raise' in self._payload

    def raise_kind(self) -> str:
        """
        The interrupt kind for a raise entry.

        Returns:
            str: One of the values in ``VALID_RAISE_KINDS``.

        Raises:
            FixtureError: If this isn't a raise entry, or its kind is unknown.
        """
        if not self.is_raise():
            raise FixtureError(f'Entry at {self._source_path} is not a raise entry')
        kind = self._payload['raise'].get('type')
        if kind not in VALID_RAISE_KINDS:
            raise FixtureError(
                f'Invalid raise kind {kind!r} at {self._source_path}; '
                f'expected one of {sorted(VALID_RAISE_KINDS)}'
            )
        return kind

    def raise_message(self) -> str | None:
        """The optional human-readable message attached to a raise entry."""
        if not self.is_raise():
            return None
        return self._payload['raise'].get('message')


class Fixture:
    """
    A loaded mockter fixture YAML, with lookup helpers.
    """

    def __init__(self, path: str, data: dict):
        """
        Args:
            path (str): Source path the fixture was loaded from.
            data (dict): Parsed YAML content.
        """
        self.path = path
        self.data = data
        self.provenance: dict = data.get('provenance', {})
        self.species: dict = data.get('species', {}) or {}
        self.reactions: dict = data.get('reactions', {}) or {}
        self.ts: dict = data.get('ts', {}) or {}

    @classmethod
    def load(cls, path: str) -> 'Fixture':
        """
        Read a fixture YAML and validate its schema.

        Args:
            path (str): Absolute path to the YAML file.

        Returns:
            Fixture: A validated fixture instance.

        Raises:
            FixtureError: If the file is missing, malformed, or has an unsupported schema version.
        """
        if not os.path.isfile(path):
            raise FixtureError(f'Fixture not found: {path}')
        data = read_yaml_file(path)
        if not isinstance(data, dict):
            raise FixtureError(f'Fixture {path} is not a YAML mapping')
        version = data.get('schema_version')
        if version not in SUPPORTED_SCHEMA_VERSIONS:
            raise FixtureError(
                f'Fixture {path} has schema_version={version!r}; '
                f'supported versions: {sorted(SUPPORTED_SCHEMA_VERSIONS)}'
            )
        for required_key in ('species',):
            if required_key not in data:
                raise FixtureError(f'Fixture {path} missing required top-level key: {required_key!r}')
        return cls(path, data)

    def lookup(
        self,
        label: str,
        job_type: str,
        conformer: int | None = None,
        tsg: int | None = None,
        torsions: list[list[int]] | None = None,
        irc_direction: str | None = None,
        fine: bool = False,
        is_ts: bool = False,
    ) -> FixtureEntry | None:
        """
        Resolve a fixture entry from job parameters.

        Args:
            label (str): Species or TS label.
            job_type (str): ARC job_type string ('conf_opt', 'opt', 'freq', 'sp',
                            'composite', 'scan', 'directed_scan', 'irc', 'tsg', 'optfreq').
            conformer (int | None): Conformer index for conf_opt jobs.
            tsg (int | None): TS-guess index for tsg jobs.
            torsions (list[list[int]] | None): 0-indexed dihedral atoms for scan jobs.
                                               Compared against fixture entries' torsions
                                               after normalizing to the same indexing.
            irc_direction (str | None): 'forward' or 'reverse' for IRC jobs.
            fine (bool): Whether this is a fine-grid opt (selects 'fine_opt' over 'opt').
            is_ts (bool): If True, look up under self.ts instead of self.species.

        Returns:
            FixtureEntry | None: The matched entry, or None if no entry matches.
        """
        bucket = self.ts if is_ts else self.species
        spc = bucket.get(label)
        if spc is None:
            return None

        if job_type == 'conf_opt':
            confs = spc.get('conformers') or []
            if conformer is None or conformer >= len(confs):
                return None
            return FixtureEntry(confs[conformer], (label, 'conformers', conformer))

        if job_type == 'tsg':
            guesses = spc.get('guesses') or []
            if tsg is None or tsg >= len(guesses):
                return None
            return FixtureEntry(guesses[tsg], (label, 'guesses', tsg))

        if job_type == 'opt':
            key = 'fine_opt' if fine else 'opt'
            block = spc.get(key)
            return FixtureEntry(block, (label, key)) if block is not None else None

        if job_type == 'optfreq':
            block = spc.get('fine_opt') or spc.get('opt')
            return FixtureEntry(block, (label, 'fine_opt' if spc.get('fine_opt') else 'opt')) if block is not None else None

        if job_type == 'freq':
            block = spc.get('freq')
            return FixtureEntry(block, (label, 'freq')) if block is not None else None

        if job_type == 'sp':
            block = spc.get('sp')
            return FixtureEntry(block, (label, 'sp')) if block is not None else None

        if job_type == 'composite':
            block = spc.get('composite')
            return FixtureEntry(block, (label, 'composite')) if block is not None else None

        if job_type in ('scan', 'directed_scan'):
            scans = spc.get('scans') or []
            if torsions is None:
                if scans:
                    return FixtureEntry(scans[0], (label, 'scans', 0))
                return None
            normalized_query = _normalize_torsions(torsions)
            for idx, scan in enumerate(scans):
                if _normalize_torsions(scan.get('torsions') or []) == normalized_query:
                    return FixtureEntry(scan, (label, 'scans', idx))
            return None

        if job_type == 'irc':
            irc = spc.get('irc') or {}
            direction = irc_direction or 'forward'
            block = irc.get(direction)
            return FixtureEntry(block, (label, 'irc', direction)) if block is not None else None

        return None


def _normalize_torsions(torsions: list[list[int]]) -> tuple[tuple[int, ...], ...]:
    """
    Normalize a torsion list for comparison: convert each dihedral to a tuple
    and sort across torsions so the order doesn't matter.

    Args:
        torsions (list[list[int]]): A list of dihedrals, each a 4-tuple of atom indices.

    Returns:
        tuple[tuple[int, ...], ...]: A canonical form for equality comparison.
    """
    return tuple(sorted(tuple(t) for t in torsions))
