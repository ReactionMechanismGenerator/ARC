"""
``arc.level.species_state`` — per-species ``sp_composite`` state model.

A species can be in one of three states with respect to ``sp_composite``:

* ``"inherit"``  — no ``sp_composite`` key on the species; the project-wide
  protocol (if any) applies.
* ``"opt_out"``  — species wrote ``sp_composite: null``; the project-wide
  protocol is explicitly bypassed and plain ``sp_level`` is used.
* ``"explicit"`` — species wrote a preset name or a recipe dict; that species
  uses the explicit protocol regardless of the project-wide one.

These three must be distinguishable at parse time and must round-trip through
``as_dict`` / ``from_dict`` / restart. That requires a sentinel that is NOT
``None`` (since ``None`` is the valid explicit user input for ``"opt_out"``),
hence :data:`INHERIT` below.

The sentinel is used only at constructor time as a default argument value; on a
constructed :class:`~arc.species.species.ARCSpecies` instance you'll see the
explicit string state in ``sp_composite_state`` and either ``None`` (for
``"inherit"`` / ``"opt_out"``) or a :class:`~arc.level.protocol.CompositeProtocol`
(for ``"explicit"``) in ``sp_composite``.
"""

from typing import TYPE_CHECKING

from arc.exceptions import InputError

if TYPE_CHECKING:
    # Imported only for static type checkers. Avoids forcing
    # ``arc.level.protocol`` (and its preset/cbs dependencies) to load just
    # because a caller did ``from arc.level import INHERIT``.
    from arc.level.protocol import CompositeProtocol


SP_COMPOSITE_STATES: tuple[str, ...] = ("inherit", "opt_out", "explicit")


class _InheritSentinel:
    """Singleton marker used as a ctor default for ``sp_composite``.

    Distinct from :data:`None` so the constructor can tell "user passed nothing"
    (→ inherit the project default) from "user passed ``None``" (→ opt out).
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "arc.level.INHERIT"

    def __reduce__(self):
        # Pickle as a stable reference so restart/roundtrip preserves identity.
        return (_resolve_inherit, ())


def _resolve_inherit():
    """Unpickle helper; returns the module-level singleton."""
    return INHERIT


INHERIT = _InheritSentinel()


def active_composite_for(
    species_state: str,
    species_protocol: "CompositeProtocol | None",
    global_protocol: "CompositeProtocol | None",
) -> "CompositeProtocol | None":
    """Return the composite protocol to apply to a single species, or ``None``.

    Parameters
    ----------
    species_state : str
        One of :data:`SP_COMPOSITE_STATES`.
    species_protocol : CompositeProtocol or None
        The species-local protocol. Must be non-``None`` iff
        ``species_state == "explicit"``.
    global_protocol : CompositeProtocol or None
        The project-wide default protocol (or ``None`` if the project didn't
        set one).

    Returns
    -------
    CompositeProtocol or None
        * ``species_protocol`` if ``species_state == "explicit"``.
        * ``None`` if ``species_state == "opt_out"``.
        * ``global_protocol`` (possibly ``None``) if ``species_state == "inherit"``.

    Raises
    ------
    arc.exceptions.InputError
        If ``species_state`` is not a valid state, or if ``"explicit"`` is
        paired with a ``None`` ``species_protocol``.
    """
    if species_state == "explicit":
        if species_protocol is None:
            raise InputError(
                "active_composite_for: state 'explicit' requires a non-None "
                "species_protocol; got None."
            )
        return species_protocol
    if species_state == "opt_out":
        return None
    if species_state == "inherit":
        return global_protocol
    raise InputError(
        f"active_composite_for: unknown species_state {species_state!r}; "
        f"expected one of {SP_COMPOSITE_STATES}."
    )
