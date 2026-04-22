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
    get_freq_level_for_composite_method,
    levels_ess,
    logger,
    supported_ess,
)

# Not public API, and deliberately absent from __all__: this is the binding that
# ``mock.patch.dict('arc.level.default_levels_of_theory', ...)`` resolves against in
# arc/main_test.py and arc/level/level_test.py. It must stay the *same dict object* as
# arc.level.level's own global, or patch.dict mutates a copy and every such test passes
# while exercising the unpatched settings. legacy_imports_test.py asserts that identity.
from arc.level.level import default_levels_of_theory

__all__ = [
    "Level",
    "assign_frequency_scale_factor",
    "get_freq_level_for_composite_method",
    "levels_ess",
    "logger",
    "supported_ess",
]
