"""
``arc.level`` — level-of-theory abstractions for ARC.

This package groups everything related to specifying *how* an electronic-structure
calculation is performed:

* The legacy :class:`~arc.level.level.Level` class, which represents a single QM level
  (method, basis, dispersion, solvation, ESS-specific options) and is unchanged from
  ``arc/level.py`` prior to its relocation into this package.
* New composite single-point abstractions added in Phase 1 of the ``sp_composite`` work:
  protocols, terms, presets, CBS extrapolation, and reporting helpers. These let a
  user define the final electronic energy of a stationary point as a sum of multiple
  SP corrections — a HEAT-style focal-point analysis (Tajti et al.,
  *J. Chem. Phys.* **121**, 11599 (2004); DOI: 10.1063/1.1811608).

Backwards compatibility
-----------------------

All public symbols that historically lived in ``arc/level.py`` are re-exported here so
that existing call sites (``from arc.level import Level`` etc.) continue to work
without modification. New code should prefer the qualified imports
``from arc.level.protocol import CompositeProtocol`` etc. when reaching for the new
machinery.

References
----------

* Allen, East, Császár, *Structures and Conformations of Non-Rigid Molecules* — review
  of focal-point analysis methodology.
* Tajti, Szalay, Császár, Kállay, Gauss, Valeev, Flowers, Vázquez, Stanton,
  *J. Chem. Phys.* **121**, 11599 (2004). DOI: 10.1063/1.1811608 — HEAT protocol.
* Helgaker, Klopper, Koch, Noga, *J. Chem. Phys.* **106**, 9639 (1997).
  DOI: 10.1063/1.473863 — two-point correlation-energy CBS extrapolation.
* Halkier, Helgaker, Jørgensen, Klopper, Koch, Olsen, Wilson,
  *Chem. Phys. Lett.* **286**, 243-252 (1998). DOI: 10.1016/S0009-2614(98)00111-0 —
  extends the two-point correlation-energy CBS extrapolation to Ne, N₂, H₂O.
* Halkier, Helgaker, Jørgensen, Klopper, Olsen,
  *Chem. Phys. Lett.* **302**, 437-446 (1999). DOI: 10.1016/S0009-2614(99)00179-7 —
  two-point HF-energy CBS extrapolation; source of the fitted ``α = 1.63``.
* Martin, *Chem. Phys. Lett.* **259**, 669-678 (1996).
  DOI: 10.1016/0009-2614(96)00898-6 — three-point Schwartz-style extrapolation.
* Dunning, *J. Chem. Phys.* **90**, 1007 (1989). DOI: 10.1063/1.456153 — correlation-
  consistent basis-set families used by the cardinal-number deduction logic.
"""

from arc.level.level import (
    Level,
    assign_frequency_scale_factor,
    levels_ess,
    logger,
    supported_ess,
)
from arc.level.species_state import (
    INHERIT,
    SP_COMPOSITE_STATES,
    active_composite_for,
)

__all__ = [
    "Level",
    "assign_frequency_scale_factor",
    "levels_ess",
    "logger",
    "supported_ess",
    "INHERIT",
    "SP_COMPOSITE_STATES",
    "active_composite_for",
]
