"""
An adapter for parsing YAML log files created by ARC for other packages.
"""

from abc import ABC

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from arc.common import read_yaml_file
from arc.constants import E_h_kJmol, bohr_to_angstrom
from arc.parser.adapter import ESSAdapter
from arc.parser.factory import register_ess_adapter
from arc.species.converter import str_to_xyz


class YAMLParser(ESSAdapter, ABC):
    """
    A parser adapter for YAML files containing internal calculation results.

    Args:
        log_file_path (str): The path to the YAML file to be parsed.
    """
    def __init__(self, log_file_path: str):
        super().__init__(log_file_path=log_file_path)
        self.data = read_yaml_file(log_file_path) or dict()

    def logfile_contains_errors(self) -> Optional[str]:
        """
        Check if the TeraChem log file contains any errors.

        Returns: Optional[str]
            None if no errors, else error message string.
        """
        # YAML files don't contain runtime errors like ESS logs.
        return None

    def parse_geometry(self) -> Optional[Dict[str, tuple]]:
        """
        Parse the xyz geometry from an ESS log file.

        Returns: Optional[Dict[str, tuple]]
            The cartesian geometry.
        """
        for key in ['xyz', 'opt_xyz']:
            if key in self.data.keys():
                return self.data[key] if isinstance(self.data[key], dict) else str_to_xyz(self.data[key])
        return None

    def parse_frequencies(self) -> Optional['np.ndarray']:
        """
        Parse the frequencies from a freq job output file.

        Returns: Optional[np.ndarray]
            The parsed frequencies (in cm^-1).
        """
        freqs = self.data.get('freqs')
        return np.array(freqs, dtype=np.float64) if freqs else None

    def parse_normal_mode_displacement(self) -> Tuple[Optional['np.ndarray'], Optional['np.ndarray']]:
        """
        Parse frequencies and normal mode displacement.

        Returns: Tuple[Optional['np.ndarray'], Optional['np.ndarray']]
            The frequencies (in cm^-1) and the normal mode displacements.
        """
        freqs = self.data.get('freqs')
        modes = self.data.get('modes')
        if freqs and modes:
            return (
                np.array(freqs, dtype=np.float64) if freqs is not None else None,
                np.array(modes, dtype=np.float64) if modes is not None else None
            )
        return None, None

    def parse_t1(self) -> Optional[float]:
        """
        Parse the T1 parameter from a CFOUR coupled cluster calculation.

        Returns: Optional[float]
            The T1 parameter.
        """
        t1 = self.data.get('T1')
        return t1

    def parse_e_elect(self) -> Optional[float]:
        """
        Parse the electronic energy from an sp job output file.

        Returns: Optional[float]
            The electronic energy in kJ/mol.
        """
        energy = self.data.get('sp')
        return energy

    def parse_zpe_correction(self) -> Optional[float]:
        """
        Determine the calculated ZPE correction (E0 - e_elect) from a frequency output file.

        Returns: Optional[float]
            The calculated zero point energy in kJ/mol.
        """
        zpe = self.data.get('zpe')
        return zpe

    def parse_1d_scan_energies(self) -> Tuple[Optional[List[float]], Optional[List[float]]]:
        """
        Parse the 1D torsion scan energies from an ESS log file.

        Returns: Tuple[Optional[List[float]], Optional[List[float]]]
            The electronic energy in kJ/mol and the dihedral scan angle in degrees.
        """
        energies = self.data.get('energies')
        angles = self.data.get('angles')
        if energies and angles and len(energies) == len(angles):
            min_energy = min(energies)
            rel_energies = [(e - min_energy) * E_h_kJmol for e in energies]
            return rel_energies, angles
        return None, None

    def parse_1d_scan_coords(self) -> Optional[List[Dict[str, tuple]]]:
        """Parse 1D scan coordinates from YAML data."""
        # Not implemented.
        return None

    def parse_scan_conformers(self) -> Optional['pd.DataFrame']:
        """
        Parse all internal coordinates of scan conformers into a DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing internal coordinates for all conformers
        """
        # Not implemented.
        return None

    def parse_irc_traj(self) -> Optional[List[Dict[str, tuple]]]:
        """
        Parse the IRC trajectory coordinates from an ESS log file.

        Returns: List[Dict[str, tuple]]
            The Cartesian coordinates for each scan point.
        """
        # Not implemented.
        return None

    def parse_nd_scan_energies(self) -> Optional[Dict]:
        """
        Parse the ND torsion scan energies from an ESS log file.

        Returns: Optional[Dict]
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

    def parse_dipole_moment(self) -> Optional[float]:
        """
        Parse the dipole moment in Debye from an opt job output file.

        Returns: Optional[float]
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

    def parse_polarizability(self) -> Optional[float]:
        """
        Parse the polarizability from a freq job output file, returns the value in Angstrom^3.

        Returns: Optional[float]
            The polarizability in Angstrom^3.
        """
        polarizability = self.data.get('polarizability')
        if polarizability is not None:
            # Convert from Bohr^3 to A^3 if needed
            return polarizability * (bohr_to_angstrom ** 3)
        return None


register_ess_adapter('yaml', YAMLParser)
