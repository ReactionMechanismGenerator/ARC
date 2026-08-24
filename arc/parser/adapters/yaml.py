"""
An adapter for parsing YAML log files created by ARC for other packages.
"""

from abc import ABC

import numpy as np
import pandas as pd
from arc.common import get_element_mass, is_str_float, logger, read_yaml_file
from arc.constants import E_h_kJmol, bohr_to_angstrom
from arc.parser.adapter import ESSAdapter
from arc.parser.factory import register_ess_adapter
from arc.species.converter import str_to_xyz

CARTESIAN_NORMAL_MODE_SCHEMA_VERSION = 2
CARTESIAN_CONVENTION = 'cartesian_unit_norm'
LEGACY_CONVENTION = 'legacy_mass_deweighted_unnormalized'
MASS_WEIGHTED_MARGIN = 0.5
CENTER_OF_MASS_RESIDUAL_FLOOR = 0.05


def get_weighted_residual(displacements: np.ndarray,
                          weights: np.ndarray,
                          ) -> np.ndarray:
    """
    Compute the normalized weighted sum of a set of normal mode displacement vectors.

    The residual of mode ``k`` is ``|sum_a w_a * d_ka| / sqrt(sum_a w_a * sum_a w_a * |d_ka|^2)``,
    the Cauchy-Schwarz bound of the numerator being the denominator, so that every residual lies in
    ``[0, 1]`` regardless of the scale of the displacements. A residual of one is reported for a
    mode of zero norm, which satisfies no weighting.

    Args:
        displacements (np.ndarray): The normal mode displacements, shaped (modes, atoms, 3).
        weights (np.ndarray): The per atom weights.

    Returns: np.ndarray
        The residual of each mode.
    """
    numerator = np.linalg.norm(np.einsum('a,kax->kx', weights, displacements), axis=1)
    denominator = np.sqrt(weights.sum() * np.einsum('a,kax,kax->k', weights, displacements, displacements))
    return np.divide(numerator, denominator, out=np.ones_like(numerator), where=denominator > 0)


def are_modes_mass_weighted(displacements: np.ndarray,
                            symbols: tuple | list,
                            ) -> bool:
    """
    Determine whether normal mode displacements are mass weighted rather than Cartesian.

    A Cartesian mode leaves the center of mass in place, ``sum_a m_a * d_a = 0``, while a mass
    weighted mode instead satisfies ``sum_a sqrt(m_a) * d_a = 0``. Both residuals are normalized
    onto a common ``[0, 1]`` scale by :func:`get_weighted_residual` and compared per mode. A mode
    counts as mass weighted only when its mass weighted residual is smaller than its Cartesian one
    by a clear margin and its Cartesian residual rises well above the rounding noise of a file
    written at print precision, so that neither test alone can carry the verdict. ``True`` is
    returned when most of the modes count as mass weighted.

    Args:
        displacements (np.ndarray): The normal mode displacements, shaped (modes, atoms, 3).
        symbols (tuple, list): The chemical element symbols of the atoms, in the geometry order.

    Returns: bool
        Whether the displacements are mass weighted.
    """
    if displacements.ndim != 3 or not len(symbols) or displacements.shape[1] != len(symbols):
        return False
    masses = np.array([get_element_mass(symbol)[0] for symbol in symbols], dtype=np.float64)
    cartesian_residual = get_weighted_residual(displacements, masses)
    mass_weighted_residual = get_weighted_residual(displacements, np.sqrt(masses))
    votes = (cartesian_residual > CENTER_OF_MASS_RESIDUAL_FLOOR) \
        & (mass_weighted_residual < MASS_WEIGHTED_MARGIN * cartesian_residual)
    return bool(votes.mean() > 0.5)


class YAMLParser(ESSAdapter, ABC):
    """
    A parser adapter for YAML files containing internal calculation results.

    Args:
        log_file_path (str): The path to the YAML file to be parsed.
    """
    def __init__(self, log_file_path: str):
        super().__init__(log_file_path=log_file_path)
        self.data = read_yaml_file(log_file_path) or dict()
        self.normal_mode_convention = None

    def get_schema_version(self) -> float:
        """
        Determine the schema version the file was written under.

        Returns ``0`` for a file that carries no ``schema_version`` key or carries one that is not
        a number, which is how every producer wrote before the key was introduced.

        Returns: float
            The schema version.
        """
        schema_version = self.data.get('schema_version')
        if isinstance(schema_version, bool):
            return 0.0
        if isinstance(schema_version, (int, float)):
            return float(schema_version)
        if isinstance(schema_version, str) and is_str_float(schema_version):
            return float(schema_version)
        return 0.0

    def logfile_contains_errors(self) -> str | None:
        """
        Check if the YAML output file reports a failed in-core ESS job.

        In-core adapters (e.g. PySCF) set ``success: false`` and populate ``error`` when the
        local job crashes, so a failed run is not silently treated as done. Producers that do
        not emit ``success`` (e.g. legacy ASE/TorchANI output) are unaffected.

        Returns: str | None
            None if no errors, else the error message string.
        """
        if self.data.get('success') is False:
            return self.data.get('error') or 'The in-core ESS job reported a failure.'
        return None

    def parse_geometry(self) -> dict[str, tuple] | None:
        """
        Parse the xyz geometry from an ESS log file.

        Returns: dict[str, tuple] | None
            The cartesian geometry.
        """
        for key in ['opt_xyz', 'xyz']:
            if key in self.data.keys():
                return self.data[key] if isinstance(self.data[key], dict) else str_to_xyz(self.data[key])
        return None

    def parse_frequencies(self) -> np.ndarray | None:
        """
        Parse the frequencies from a freq job output file.

        Returns: np.ndarray | None
            The parsed frequencies (in cm^-1).
        """
        freqs = self.data.get('freqs')
        if freqs is None:
            freqs = self.data.get('frequencies')
        return np.array(freqs, dtype=np.float64) if freqs is not None else None

    def parse_normal_mode_displacement(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        Parse frequencies and normal mode displacement.

        Each mode is rescaled to a unit Euclidean norm, ``sum(|d|^2) = 1``, which is the Cartesian
        displacement convention Gaussian reports and the one used elsewhere in ARC. Any Cartesian
        scale is accepted: rescaling maps it onto that convention and leaves the direction of the
        mode, and hence the physical reduced mass ``sum(m_a * |d_a|^2)``, unchanged. The rescaling is
        required because files written under a schema version below
        ``CARTESIAN_NORMAL_MODE_SCHEMA_VERSION`` carry mass deweighted unnormalized modes ``l_a``,
        normalized so that ``sum(m_a * |l_a|^2) = 1`` and therefore carrying a norm of
        ``1 / sqrt(mu)``. Which of the two conventions the file was read under is recorded on
        ``self.normal_mode_convention``. A mode of zero norm is returned as zeros.

        Rescaling does not correct a mass weighted mode, which already carries a unit norm, so the
        modes are also tested against the file's own geometry and a warning is logged when they look
        mass weighted. The modes are expected to be shaped (modes, atoms, 3); a leading axis of
        length one is dropped, and any other shape yields ``None`` rather than a set of modes
        normalized over the wrong axes.

        Returns: tuple[np.ndarray | None, np.ndarray | None]
            The frequencies (in cm^-1) and the normal mode displacements.
        """
        freqs = self.data.get('freqs') or self.data.get('frequencies')
        modes = self.data.get('modes') or self.data.get('normal_modes')
        if not freqs or not modes:
            return None, None
        displacements = np.array(modes, dtype=np.float64)
        if displacements.ndim == 4 and displacements.shape[0] == 1:
            displacements = displacements[0]
        if displacements.ndim != 3 or displacements.shape[2] != 3:
            logger.warning(f'Expected the normal mode displacements in {self.log_file_path} to be shaped '
                           f'(modes, atoms, 3), got {displacements.shape}. Not returning normal modes.')
            return None, None
        self.normal_mode_convention = CARTESIAN_CONVENTION \
            if self.get_schema_version() >= CARTESIAN_NORMAL_MODE_SCHEMA_VERSION else LEGACY_CONVENTION
        norms = np.linalg.norm(displacements.reshape(displacements.shape[0], -1), axis=1)[:, None, None]
        displacements = np.divide(displacements, norms,
                                  out=np.zeros_like(displacements), where=norms > 0)
        xyz = self.parse_geometry()
        if xyz is not None and are_modes_mass_weighted(displacements, xyz.get('symbols') or tuple()):
            logger.warning(f'The normal mode displacements in {self.log_file_path} look mass weighted rather than '
                           f'Cartesian. Rescaling them to a unit norm does not convert them, so any analysis that '
                           f'follows describes a mass weighted mode and is tilted towards the heavy atoms.')
        return np.array(freqs, dtype=np.float64), displacements

    def parse_t1(self) -> float | None:
        """
        Parse the T1 parameter from a CFOUR coupled cluster calculation.

        Returns: float | None
            The T1 parameter.
        """
        t1 = self.data.get('T1')
        return t1

    def parse_e_elect(self) -> float | None:
        """
        Parse the electronic energy from the YAML file.

        Returns: float | None
            The electronic energy in kJ/mol.
        """
        energy = self.data.get('sp') or self.data.get('energy')
        return energy

    def parse_zpe_correction(self) -> float | None:
        """
        Determine the calculated ZPE correction (E0 - e_elect) from a frequency output file.

        Returns: float | None
            The calculated zero point energy in kJ/mol.
        """
        zpe = self.data.get('zpe')
        return zpe

    def parse_1d_scan_energies(self) -> tuple[list[float] | None, list[float] | None]:
        """
        Parse the 1D torsion scan energies from an ESS log file.

        Returns: tuple[list[float] | None, list[float] | None]
            The electronic energy in kJ/mol and the dihedral scan angle in degrees.
        """
        energies = self.data.get('energies')
        angles = self.data.get('angles')
        if energies and angles and len(energies) == len(angles):
            min_energy = min(energies)
            rel_energies = [(e - min_energy) * E_h_kJmol for e in energies]
            return rel_energies, angles
        return None, None

    def parse_1d_scan_coords(self) -> list[dict[str, tuple]] | None:
        """
        Parse 1D scan coordinates from YAML data.

        Returns: Optional[List[Dict[str, tuple]]]
            The Cartesian coordinates (xyz dicts) for each scan point.
        """
        scan_coords = self.data.get('scan_coords')
        if scan_coords:
            return [xyz if isinstance(xyz, dict) else str_to_xyz(xyz) for xyz in scan_coords]
        return None

    def parse_scan_conformers(self) -> 'pd.DataFrame' | None:
        """
        Parse all internal coordinates of scan conformers into a DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing internal coordinates for all conformers
        """
        # Not implemented.
        return None

    def parse_irc_traj(self) -> list[dict[str, tuple]] | None:
        """
        Parse the IRC trajectory coordinates from an ESS log file.

        Returns: list[dict[str, tuple]]
            The Cartesian coordinates for each scan point.
        """
        irc_traj = self.data.get('irc_traj')
        if irc_traj:
            return [xyz if isinstance(xyz, dict) else str_to_xyz(xyz) for xyz in irc_traj]
        return None

    def parse_nd_scan_energies(self) -> dict | None:
        """
        Parse the ND torsion scan energies from an ESS log file.

        Returns: dict | None
            The "results" dictionary, which has the following structure::

                  results = {'directed_scan_type': <str, used for the fig name>,
                             'scans': <list, entries are lists of torsion indices>,
                             'directed_scan': <dict, keys are tuples of '{0:.2f}' formatted dihedrals,
                                               values are dictionaries with the following keys and values:
                                               {'energy': <float, energy in kJ/mol>,  * only this is used here
                                                'xyz': <dict>,
                                                'is_isomorphic': <bool>,
                                                'trsh': <list, job.ess_trsh_methods>}>
                             }
        """
        # Not implemented.
        return None

    def parse_dipole_moment(self) -> float | None:
        """
        Parse the dipole moment in Debye from an opt job output file.

        Returns: float | None
            The dipole moment in Debye.
        """
        dipole = self.data.get('dipole')
        if isinstance(dipole, (int, float)):
            return float(dipole)
        if isinstance(dipole, (list, tuple)):
            return float(np.linalg.norm(dipole))
        if isinstance(dipole, dict):
            return float(np.linalg.norm(list(dipole.values())))
        return None

    def parse_polarizability(self) -> float | None:
        """
        Parse the polarizability from a freq job output file, returns the value in Angstrom^3.

        Returns: float | None
            The polarizability in Angstrom^3.
        """
        polarizability = self.data.get('polarizability')
        if polarizability is not None:
            # Convert from Bohr^3 to A^3 if needed
            return polarizability * (bohr_to_angstrom ** 3)
        return None


register_ess_adapter('yaml', YAMLParser)
