"""Held-fixed coordinate constraints for TCKDB calculation payloads.

Internal representation + serializer for the ``constraints`` field on
``CalculationWithResultsPayload`` (and the bundle-aware variants used
by computed-species / computed-reaction).

Producers (Gaussian/ORCA parser code) build a list of
:class:`TCKDBCalculationConstraint` instances; the TCKDB adapter calls
:func:`serialize_constraints` to emit the final TCKDB-shaped dict list
with deterministic 1-based ``constraint_index`` values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from arc.common import get_logger


logger = get_logger()


_VALID_KINDS: frozenset[str] = frozenset({
    'cartesian_atom',
    'bond',
    'angle',
    'dihedral',
    'improper',
})

# Number of atom indices required for each TCKDB constraint kind.
_ATOMS_PER_KIND: dict[str, int] = {
    'cartesian_atom': 1,
    'bond': 2,
    'angle': 3,
    'dihedral': 4,
    'improper': 4,
}


@dataclass(frozen=True)
class TCKDBCalculationConstraint:
    """A held-fixed coordinate constraint to emit into a TCKDB calculation payload.

    Atom indices are 1-based per TCKDB's convention. ``target_value`` is
    optional — emitted only when the producer has a reliable parsed value
    from the ESS input deck or log.
    """

    constraint_kind: str
    atom1_index: int
    atom2_index: int | None = None
    atom3_index: int | None = None
    atom4_index: int | None = None
    target_value: float | None = None

    @classmethod
    def from_atoms(
        cls,
        constraint_kind: str,
        atoms: list[int] | tuple[int, ...],
        target_value: float | None = None,
    ) -> "TCKDBCalculationConstraint":
        """Build from a flat ``atoms`` list, padding unused slots with None."""
        padded: list[int | None] = list(atoms) + [None] * (4 - len(atoms))
        return cls(
            constraint_kind=constraint_kind,
            atom1_index=padded[0],  # type: ignore[arg-type]
            atom2_index=padded[1],
            atom3_index=padded[2],
            atom4_index=padded[3],
            target_value=target_value,
        )


def _validate(c: TCKDBCalculationConstraint) -> bool:
    """Return True if ``c`` is internally consistent for TCKDB emission.

    Validates that the kind is recognised, that the right number of atom
    slots are filled, and that all filled atom indices are 1-based ints.
    Logs a warning and returns False when invalid — the caller drops the
    constraint and continues.
    """
    if c.constraint_kind not in _VALID_KINDS:
        logger.warning("TCKDB constraint: unknown kind %r; dropping",
                       c.constraint_kind)
        return False
    expected = _ATOMS_PER_KIND[c.constraint_kind]
    indices = [c.atom1_index, c.atom2_index, c.atom3_index, c.atom4_index]
    filled = [i for i in indices if i is not None]
    if len(filled) != expected:
        logger.warning("TCKDB constraint: kind %s expects %d atom indices, "
                       "got %d; dropping", c.constraint_kind, expected, len(filled))
        return False
    for idx in filled:
        if not isinstance(idx, int) or idx < 1:
            logger.warning("TCKDB constraint: non-positive or non-integer "
                           "atom index %r; dropping", idx)
            return False
    return True


def serialize_constraints(
    constraints: Iterable[TCKDBCalculationConstraint | Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Serialize an iterable of constraints into TCKDB payload shape.

    Accepts either :class:`TCKDBCalculationConstraint` instances or the
    parser-shaped dicts ``{'constraint_kind', 'atoms', 'target_value'}``
    that the Gaussian/ORCA parsers emit. Mixed input is fine.

    Output shape per element::

        {
            'constraint_index': int,           # 1-based, deterministic
            'constraint_kind': str,
            'atom1_index': int,
            'atom2_index': int | omitted,
            'atom3_index': int | omitted,
            'atom4_index': int | omitted,
            'target_value': float | omitted,
        }

    Returns ``[]`` when the input is empty or every entry is invalid.
    """
    out: list[dict[str, Any]] = []
    next_index = 1
    for raw in constraints:
        c = _coerce(raw)
        if c is None:
            continue
        if not _validate(c):
            continue
        entry: dict[str, Any] = {
            'constraint_index': next_index,
            'constraint_kind': c.constraint_kind,
            'atom1_index': c.atom1_index,
        }
        if c.atom2_index is not None:
            entry['atom2_index'] = c.atom2_index
        if c.atom3_index is not None:
            entry['atom3_index'] = c.atom3_index
        if c.atom4_index is not None:
            entry['atom4_index'] = c.atom4_index
        if c.target_value is not None:
            entry['target_value'] = float(c.target_value)
        out.append(entry)
        next_index += 1
    return out


def _coerce(
    raw: TCKDBCalculationConstraint | Mapping[str, Any],
) -> TCKDBCalculationConstraint | None:
    """Coerce a parser-dict OR existing dataclass instance into the dataclass.

    Returns None and logs a warning when the input is shaped wrong (e.g.,
    parser dict missing 'atoms' or 'constraint_kind'). The caller skips.
    """
    if isinstance(raw, TCKDBCalculationConstraint):
        return raw
    if not isinstance(raw, Mapping):
        logger.warning("TCKDB constraint: expected dataclass or mapping, "
                       "got %s; dropping", type(raw).__name__)
        return None
    kind = raw.get('constraint_kind')
    atoms = raw.get('atoms')
    if not isinstance(kind, str):
        logger.warning("TCKDB constraint: missing or non-string "
                       "'constraint_kind' in %r; dropping", raw)
        return None
    if not isinstance(atoms, (list, tuple)) or not atoms:
        logger.warning("TCKDB constraint: missing or empty 'atoms' in %r; "
                       "dropping", raw)
        return None
    try:
        atom_ints = [int(a) for a in atoms]
    except (TypeError, ValueError):
        logger.warning("TCKDB constraint: non-integer atom index in %r; "
                       "dropping", raw)
        return None
    target_value = raw.get('target_value')
    if target_value is not None:
        try:
            target_value = float(target_value)
        except (TypeError, ValueError):
            logger.warning("TCKDB constraint: non-numeric target_value %r; "
                           "treating as absent", target_value)
            target_value = None
    return TCKDBCalculationConstraint.from_atoms(
        constraint_kind=kind,
        atoms=atom_ints,
        target_value=target_value,
    )
