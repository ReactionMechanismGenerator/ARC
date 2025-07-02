"""
An adapter for parsing Q-Chem log files.
"""

from abc import ABC

import numpy as np
from typing import TYPE_CHECKING, Optional, Tuple, List, Dict

from arc.constants import E_h_kJmol, bohr_to_angstrom
from arc.species.converter import xyz_from_data
from arc.parser.adapter import ESSAdapter
from arc.parser.factory import register_ess_adapter
from arc.parser.parser import _get_lines_from_file


if TYPE_CHECKING:
    import pandas as pd


class QChemParser(ESSAdapter, ABC):
    """
    A class for parsing Q-Chem log files.

    Args:
        log_file_path (str): The path to the log file to be parsed.
    """
    def __init__(self, log_file_path: str):
        super().__init__(log_file_path=log_file_path)

    def logfile_contains_errors(self) -> Optional[str]:
        """
        Check if the Q-Chem log file contains any errors.

        Returns: Optional[str]
            None if no errors, else error message string.
        """
        lines = _get_lines_from_file(self.log_file_path)[-500:]
        for line in reversed(lines):
            if 'SCF failed' in line:
                return 'SCF failed'
            if 'Invalid charge/multiplicity combination' in line:
                return 'Invalid charge/multiplicity combination'
            if 'MAXIMUM OPTIMIZATION CYCLES REACHED' in line:
                return 'Maximum optimization cycles reached'
            if 'error' in line.lower():
                ignore_conditions = [
                    'DIIS' in line,
                    'gprntSymmMtrx' in line,
                    'Relative error' in line,
                    'zonesort' in line
                ]
                if not any(ignore_conditions):
                    return line.strip()
        return None

    def parse_geometry(self) -> Optional[Dict[str, tuple]]:
        """
        Parse the xyz geometry from an ESS log file.

        Returns: Optional[Dict[str, tuple]]
            The cartesian geometry.
        """
        lines = _get_lines_from_file(self.log_file_path)
        coords , symbols = list(), list()
        for i in range(len(lines)):
            if 'Standard Nuclear Orientation' in lines[i]:
                j = i + 3  # Skip to first data line after header
                while j < len(lines) and lines[j].strip() and '---' not in lines[j]:
                    parts = lines[j].split()
                    if len(parts) >= 5:  # Require at least 5 columns
                        try:
                            symbol = parts[1]
                            x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                            symbols.append(symbol)
                            coords.append([x, y, z])
                        except (ValueError, IndexError):
                            pass
                    j += 1
                if coords:
                    return xyz_from_data(coords=np.array(coords), symbols=symbols)
        return None

    def parse_frequencies(self) -> Optional[np.ndarray]:
        """
        Parse the frequencies from a freq job output file.

        Returns: Optional[np.ndarray]
            The parsed frequencies (in cm^-1).
        """
        frequencies = []
        lines = _get_lines_from_file(self.log_file_path)
        for line in lines:
            if 'Frequency:' in line:
                # Example: Frequency:  1234.56  -567.89  3456.78
                parts = line.replace('Frequency:', '').split()
                for freq_str in parts:
                    try:
                        freq = float(freq_str)
                        frequencies.append(freq)
                    except ValueError:
                        continue
        if frequencies:
            return np.array(frequencies, dtype=np.float64)
        return None

    def parse_normal_mode_displacement(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Parse frequencies and normal mode displacement.

        Returns: Tuple[Optional['np.ndarray'], Optional['np.ndarray']]
            The frequencies (in cm^-1) and the normal mode displacements.
        """
        # Not implemented for Q-Chem.
        return None, None

    def parse_t1(self) -> Optional[float]:
        """
        Parse the T1 parameter from a CC calculation.

        Returns: Optional[float]
            The T1 parameter.
        """
        # Not implemented for Q-Chem.
        return None

    def parse_e_elect(self) -> Optional[float]:
        """
        Parse the electronic energy from an sp job output file.

        Returns: Optional[float]
            The electronic energy in kJ/mol.
        """
        preferred_energy, alternative_energy = None, None
        with open(self.log_file_path, 'r') as f:
            for line in f:
                # Preferred source: optimization final energy
                if 'Final energy is' in line:
                    try:
                        preferred_energy = float(line.split()[-1])
                    except (ValueError, IndexError):
                        continue

                # Alternative source: single-point energy
                if 'Total energy in the final basis set' in line:
                    try:
                        alternative_energy = float(line.split()[-1])
                    except (ValueError, IndexError):
                        continue
        # Prioritize optimized energy, fallback to single-point energy
        energy = preferred_energy if preferred_energy is not None else alternative_energy
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
                if 'zero point vibrational energy' in line.lower():
                    # Zero point vibrational energy:       47.343 kcal/mol
                    try:
                        zpe = float(line.split()[-2])
                        break
                    except (ValueError, IndexError):
                        continue
        if zpe is not None:
            return zpe * 4.184 # kcal/mol to kj/mol
        return None

    def parse_1d_scan_energies(self) -> Tuple[Optional[List[float]], Optional[List[float]]]:
        """
        Parse the 1D torsion scan energies from an ESS log file.

        Returns: Tuple[Optional[List[float]], Optional[List[float]]]
            The electronic energy in kJ/mol and the dihedral scan angle in degrees.
        """
        # Not implemented for Q-Chem.
        return None, None

    def parse_1d_scan_coords(self) -> Optional[List[Dict[str, tuple]]]:
        """
        Parse the 1D torsion scan coordinates from an ESS log file.

        Returns: List[Dict[str, tuple]]
            The Cartesian coordinates for each scan point.
        """
        # Not implemented for Q-Chem.
        return None

    def parse_irc_traj(self) -> Optional[List[Dict[str, tuple]]]:
        """
        Parse the IRC trajectory coordinates from an ESS log file.

        Returns: List[Dict[str, tuple]]
            The Cartesian coordinates for each scan point.
        """
        # Not implemented for Q-Chem.
        return None

    def parse_scan_conformers(self) -> Optional['pd.DataFrame']:
        """
        Parse all internal coordinates of scan conformers into a DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing internal coordinates for all conformers
        """
        # Not implemented for Q-Chem.
        return None

    def parse_nd_scan_energies(self) -> Optional[Dict]:
        """
        Parse the ND torsion scan energies from an ESS log file.

        Returns: dict
            The "results" dictionary
        """
        # Not implemented for Q-Chem.
        return None

    def parse_dipole_moment(self) -> Optional[float]:
        """
        Parse the dipole moment in Debye from an opt job output file.

        Returns: Optional[float]
            The dipole moment in Debye.
        """
        skip_next, read = False, False
        dipole_moment = None
        with open(self.log_file_path, 'r') as f:
            for line in f:
                # Detect dipole moment section
                if 'dipole moment' in line.lower() and 'debye' in line.lower():
                    skip_next = True
                    read = True
                    continue
                # Skip next line (contains X, Y, Z components)
                if skip_next:
                    skip_next = False
                    continue
                # Parse total dipole moment from "Tot" line
                if 'Tot' in line and read:
                    read = False
                    try:
                        dipole_moment = float(line.split()[-1])
                    except (ValueError, IndexError):
                        continue
        return dipole_moment

    def parse_polarizability(self) -> Optional[float]:
        """
        Parse the polarizability from a freq job output file, returns the value in Angstrom^3.

        Returns: Optional[float]
            The polarizability in Angstrom^3.
        """
        polarizability = None
        with open(self.log_file_path, 'r') as f:
            for line in f:
                if 'Isotropic polarizability for W=' in line:
                    try:
                        parts = line.split()
                        # The value is the second last element before 'Bohr**3.'
                        value_bohr3 = float(parts[-2])
                        polarizability = value_bohr3 * bohr_to_angstrom ** 3
                        break
                    except (ValueError, IndexError):
                        continue
        return polarizability


register_ess_adapter('qchem', QChemParser)
