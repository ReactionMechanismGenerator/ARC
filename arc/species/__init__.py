"""ARC species package.

This module historically imported all submodules on package import.  Some of
those (e.g., :mod:`arc.species.conformers`) depend on optional third-party
libraries such as OpenBabel.  The testing environment used within Codex does not
provide these heavy dependencies, so importing them unconditionally would raise
``ImportError`` and prevent the lightweight modules from being used.

To keep the public API largely intact while remaining importable without the
optional dependencies, we attempt to import the heavy modules lazily.  If the
import fails, it is silently skipped; users that actually require the optional
functionality will need to ensure the relevant dependencies are installed.
"""

from arc.common import get_logger

logger = get_logger()

try:  # optional, requires OpenBabel
    import arc.species.conformers
except Exception as e:  # pragma: no cover - best effort for optional deps
    logger.debug(f"Skipping arc.species.conformers import: {e}")

import arc.species.converter
import arc.species.species
import arc.species.xyz_to_2d
import arc.species.xyz_to_smiles
from arc.species.species import ARCSpecies, TSGuess
