"""
An adapter for parsing CFOUR log files.
"""

from abc import ABC

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from arc.common import SYMBOL_BY_NUMBER
from arc.constants import E_h_kJmol
from arc.species.converter import xyz_from_data
from arc.parser.adapter import ESSAdapter
from arc.parser.factory import register_ess_adapter
from arc.parser.parser import _get_lines_from_file


class CfourParser(ESSAdapter, ABC):
    """
    A class for parsing CFOUR log files.

    Args:
        log_file_path (str): The path to the log file to be parsed.
    """
    def __init__(self, log_file_path: str):
        super().__init__(log_file_path=log_file_path)

    def logfile_contains_errors(self) -> Optional[str]:
        """
        Check if the ESS log file contains any errors.

        Returns: Optional[str]
            ``None`` if the log file is free of errors, otherwise the error is returned as a string.
        """
        with open(self.log_file_path, 'r') as f:
            lines = f.readlines()[-100:]

        for line in reversed(lines):
            if 'error' in line.lower():
                return line.strip()
            if 'CFOUR ERROR CODE' in line and not line.strip().endswith('0.000000000000'):
                return f'Non-zero CFOUR error code: {line.strip()}'
        return None

    def parse_geometry(self) -> Optional[Dict[str, tuple]]:
        """
        Parse the xyz geometry from an ESS log file.

        Returns: Optional[Dict[str, tuple]]
            The cartesian geometry.
        """
        lines = _get_lines_from_file(self.log_file_path)
        coords = []
        numbers = []
        for i in range(len(lines)-1, -1, -1):
            if 'CARTESIAN COORDINATES (ANGSTROEM)' in lines[i]:
                j = i + 2  # skip header
                while j < len(lines) and lines[j].strip():
                    parts = lines[j].split()
                    if len(parts) >= 4:
                        try:
                            atom_symbol = parts[0].capitalize()
                            atomic_number = next(k for k, v in SYMBOL_BY_NUMBER.items() if v == atom_symbol)
                            numbers.append(atomic_number)
                            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
                        except (ValueError, StopIteration):
                            pass
                    j += 1
                if coords:
                    return xyz_from_data(coords=np.array(coords), numbers=np.array(numbers))
        return None

    def parse_frequencies(self) -> Optional[np.ndarray]:
        """
        Parse the frequencies from a freq job output file.

        Returns: Optional[np.ndarray]
            The parsed frequencies (in cm^-1).
        """
        frequencies = []
        with open(self.log_file_path, 'r') as f:
            for line in f:
                if 'FREQUENCIES (CM**-1)' in line:
                    # Example: FREQUENCIES (CM**-1)     3810.9  3810.9  3810.9
                    numbers = [float(x) for x in line.split()[2:] if x.replace('.', '', 1).replace('-', '', 1).isdigit()]
                    frequencies.extend(numbers)
        if frequencies:
            return np.array(frequencies, dtype=np.float64)
        return None

    def parse_normal_mode_displacement(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Parse frequencies and normal mode displacement.

        Returns: Tuple[Optional['np.ndarray'], Optional['np.ndarray']]
            The frequencies (in cm^-1) and the normal mode displacements.
        """
        # Not implemented for CFOUR.
        return None, None

    def parse_t1(self) -> Optional[float]:
        """
        Parse the T1 parameter from a CC calculation.

        Returns: Optional[float]
            The T1 parameter.
        """
        with open(self.log_file_path, 'r') as f:
            for line in f:
                if 'T1 DIAGNOSTIC' in line:
                    try:
                        return float(line.split()[-1])
                    except (ValueError, IndexError):
                        continue
        return None

    def parse_e_elect(self) -> Optional[float]:
        """
        Parse the electronic energy from an sp job output file.

        Returns: Optional[float]
            The electronic energy in kJ/mol.
        """
        lines = _get_lines_from_file(self.log_file_path)
        energy = None
        # Look for the most correlated energy available
        for line in reversed(lines):
            if 'CCSD(T) TOTAL ENERGY' in line:
                try:
                    energy = float(line.split()[-1])
                    break
                except (ValueError, IndexError):
                    continue
            if 'CCSD TOTAL ENERGY' in line:
                try:
                    energy = float(line.split()[-1])
                    break
                except (ValueError, IndexError):
                    continue
            if 'MP2 TOTAL ENERGY' in line:
                try:
                    energy = float(line.split()[-1])
                    break
                except (ValueError, IndexError):
                    continue
            if 'SCF TOTAL ENERGY' in line:
                try:
                    energy = float(line.split()[-1])
                    break
                except (ValueError, IndexError):
                    continue
            if 'CURRENT ENERGY' in line:
                try:
                    energy = float(line.split()[-1])
                    break
                except (ValueError, IndexError):
                    continue
        if energy is not None:
            return energy * E_h_kJmol
        return None

    def parse_zpe_correction(self) -> Optional[float]:
        """
        Determine the calculated ZPE correction (E0 - e_elect) from a frequency output file.

        Returns: Optional[float]
            The calculated zero point energy in kJ/mol.
        """
        zpe = None
        with open(self.log_file_path, 'r') as f:
            for line in f:
                if 'ZERO-POINT VIBRATIONAL ENERGY' in line:
                    # Example: ZERO-POINT VIBRATIONAL ENERGY     0.025410
                    try:
                        zpe = float(line.split()[-1])
                        break
                    except (ValueError, IndexError):
                        continue
        if zpe is not None:
            return zpe * E_h_kJmol
        return None

    def parse_1d_scan_energies(self) -> Tuple[Optional[List[float]], Optional[List[float]]]:
        """
        Parse the 1D torsion scan energies from an ESS log file.

        Returns: Tuple[Optional[List[float]], Optional[List[float]]]
            The electronic energy in kJ/mol and the dihedral scan angle in degrees.
        """
        # Not implemented for CFOUR.
        return None, None

    def parse_1d_scan_coords(self) -> Optional[List[Dict[str, tuple]]]:
        """
        Parse the 1D torsion scan coordinates from an ESS log file.

        Returns: List[Dict[str, tuple]]
            The Cartesian coordinates for each scan point.
        """
        # Not implemented for CFOUR.
        return None

    def parse_irc_traj(self) -> Optional[List[Dict[str, tuple]]]:
        """
        Parse the IRC trajectory coordinates from an ESS log file.

        Returns: List[Dict[str, tuple]]
            The Cartesian coordinates for each scan point.
        """
        # Not implemented for TeraChem.
        return None

    def parse_scan_conformers(self) -> Optional[pd.DataFrame]:
        """
        Parse all internal coordinates of scan conformers into a DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing internal coordinates for all conformers
        """
        # Not implemented for CFOUR.
        return None

    def parse_nd_scan_energies(self) -> Optional[Dict]:
        """
        Parse the ND torsion scan energies from an ESS log file.

        Returns: dict
            The "results" dictionary
        """
        # Not implemented for CFOUR.
        return None

    def parse_dipole_moment(self) -> Optional[float]:
        """
        Parse the dipole moment in Debye from an opt job output file.

        Returns: Optional[float]
            The dipole moment in Debye.
        """
        with open(self.log_file_path, 'r') as f:
            for line in f:
                if 'DIPOLE MOMENT' in line and 'DEBYE' in line:
                    # Example: DIPOLE MOMENT (DEBYE)    0.0000   0.0000   1.8600
                    try:
                        parts = [float(x) for x in line.split() if x.replace('.', '', 1).replace('-', '', 1).isdigit()]
                        if len(parts) >= 3:
                            # Magnitude is typically the last value, or sqrt(x^2 + y^2 + z^2)
                            return (parts[-3] ** 2 + parts[-2] ** 2 + parts[-1] ** 2) ** 0.5
                    except (ValueError, IndexError):
                        continue
        return None

    def parse_polarizability(self) -> Optional[float]:
        """
        Parse the polarizability from a freq job output file, returns the value in Angstrom^3.

        Returns: Optional[float]
            The polarizability in Angstrom^3.
        """
        # Not implemented for CFOUR.
        return None


register_ess_adapter('cfour', CfourParser)
