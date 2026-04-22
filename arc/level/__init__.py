"""
``arc.level`` — level-of-theory abstractions for ARC.

This package hosts the :class:`~arc.level.level.Level` class (representing a
single QM level: method, basis, dispersion, solvation, ESS-specific options).
Historically ``Level`` lived in ``arc/level.py``; it is now ``arc/level/level.py``.
All public symbols are re-exported here so existing call sites
(``from arc.level import Level`` etc.) keep working without modification.
"""

from arc.level.level import (
    Level,
    assign_frequency_scale_factor,
    levels_ess,
    logger,
    supported_ess,
)

__all__ = [
    "Level",
    "assign_frequency_scale_factor",
    "levels_ess",
    "logger",
    "supported_ess",
]
