"""
An adapter for parsing Molpro log files.
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


class MolproParser(ESSAdapter, ABC):
    """
    A class for parsing Molpro log files.

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
            if 'Error' in line or 'error' in line.lower():
                if 'dimension' in line.lower():
                    return 'Memory error: dimension too large'
                elif 'convergence' in line.lower():
                    return 'Convergence failure'
                elif 'syntax' in line.lower():
                    return 'Syntax error in input'
                elif 'integral' in line.lower():
                    return 'Integral calculation error'
                return line.strip()
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

        # Look for geometry in optimization output
        for i in range(len(lines)-1, -1, -1):
            if 'Current geometry' in lines[i]:
                j = i + 4  # Skip header lines
                while j < len(lines) and lines[j].strip():
                    parts = lines[j].split()
                    if len(parts) >= 5:
                        try:
                            atom_symbol = parts[0].capitalize()
                            # Convert symbol to atomic number
                            atomic_number = next(k for k, v in SYMBOL_BY_NUMBER.items() if v == atom_symbol)
                            numbers.append(atomic_number)
                            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
                        except (ValueError, StopIteration):
                            continue
                    j += 1
                if coords:
                    return xyz_from_data(coords=np.array(coords), numbers=np.array(numbers))

        # Fallback to input geometry
        for line in lines:
            if 'ATOMIC COORDINATES' in line:
                coords = []
                numbers = []
                for _ in range(3):  # Skip header lines
                    line = next(lines)
                while line.strip() and 'Bond' not in line:
                    parts = line.split()
                    if len(parts) >= 5:
                        try:
                            atom_symbol = parts[1].capitalize()
                            atomic_number = next(k for k, v in SYMBOL_BY_NUMBER.items() if v == atom_symbol)
                            numbers.append(atomic_number)
                            coords.append([float(parts[2]), float(parts[3]), float(parts[4])])
                        except (ValueError, StopIteration):
                            continue
                    line = next(lines)
                if coords:
                    return xyz_from_data(coords=np.array(coords), numbers=np.array(numbers))
        return None

    def parse_frequencies(self) -> Optional[np.ndarray]:
        """
        Parse the frequencies from a freq job output file.

        Returns: Optional[np.ndarray]
            The parsed frequencies (in cm^-1).
        """
        frequencies = list()
        lines = _get_lines_from_file(self.log_file_path)
        read = False
        for line in lines:
            if 'Nr' in line and '[1/cm]' in line:
                continue
            if 'Vibration' in line and 'Wavenumber' in line and 'Low' not in line:
                read = True
                continue
            if read:
                if not line.strip() or '----' in line:
                    read = False
                    break
                try:
                    freq = float(line.split()[-1])
                    frequencies.append(freq)
                except (ValueError, IndexError):
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
        # Not implemented for Molpro.
        return None, None

    def parse_t1(self) -> Optional[float]:
        """
        Parse the T1 parameter from a CC calculation.

        Returns: Optional[float]
            The T1 parameter.
        """
        with open(self.log_file_path, 'r') as f:
            for line in f:
                if 'T1 diagnostic' in line:
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

        # Determine calculation type
        f12, f12a, f12b, mrci = False, False, False, False
        for line in lines:
            if 'basis' in line.lower():
                if 'vtz' in line.lower() or 'vdz' in line.lower():
                    f12a = True
                elif any(high_basis in line.lower() for high_basis in ['vqz', 'v5z', 'v6z', 'v7z', 'v8z']):
                    f12b = True
            if 'ccsd' in line.lower() and 'f12' in line.lower():
                f12 = True
            if 'mrci' in line.lower():
                mrci = True
                f12a, f12b = False, False  # MRCI takes precedence

        # Search for energy
        energy = None
        for line in reversed(lines):  # Search from end to get final energy
            if f12 and f12a and ('CCSD(T)-F12a' in line or 'CCSD(T)-F12/' in line) and 'energy' in line:
                try:
                    energy = float(line.split()[-1])
                    break
                except (ValueError, IndexError):
                    continue

            elif f12 and f12b and 'CCSD(T)-F12b' in line and 'energy' in line:
                try:
                    energy = float(line.split()[-1])
                    break
                except (ValueError, IndexError):
                    continue

            elif mrci:
                # Prioritize Davidson-corrected energy
                if '(Davidson, relaxed reference)' in line:
                    try:
                        energy = float(line.split()[3])
                        break
                    except (ValueError, IndexError):
                        continue
                # Fallback to regular MRCI energy
                if 'MRCI' in line and 'MULTI' in line and 'HF-SCF' in line:
                    try:
                        # The energy is typically on the next line
                        idx = lines.index(line)
                        if idx + 1 < len(lines):
                            energy_line = lines[idx + 1].split()
                            if energy_line:
                                energy = float(energy_line[0])
                                break
                    except (ValueError, IndexError):
                        continue

            # Fallback to common energy patterns
            if '!RHF STATE  1.1 Energy' in line:
                try:
                    energy = float(line.split()[-1])
                    break
                except (ValueError, IndexError):
                    continue

            if 'Total energy:' in line:
                try:
                    energy = float(line.split()[-1])
                    break
                except (ValueError, IndexError):
                    continue

            if 'CCSD' in line and 'energy=' in line:
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
            lines = f.readlines()

        for i, line in enumerate(lines):
            if 'Electronic Energy at 0 [K]:' in line:
                try:
                    electronic_energy = float(line.split()[5])
                    next_line = lines[i + 1]
                    ee_plus_zpe = float(next_line.split()[5])
                    zpe = (ee_plus_zpe - electronic_energy) * E_h_kJmol
                    break
                except (IndexError, ValueError):
                    continue

        return zpe

    def parse_1d_scan_energies(self) -> Tuple[Optional[List[float]], Optional[List[float]]]:
        """
        Parse the 1D torsion scan energies from an ESS log file.

        Returns: Tuple[Optional[List[float]], Optional[List[float]]]
            The electronic energy in kJ/mol and the dihedral scan angle in degrees.
        """
        # Not implemented for Molpro.
        return None, None

    def parse_1d_scan_coords(self) -> Optional[List[Dict[str, tuple]]]:
        """
        Parse the 1D torsion scan coordinates from an ESS log file.

        Returns: List[Dict[str, tuple]]
            The Cartesian coordinates for each scan point.
        """
        # Not implemented for Molpro.
        return None

    def parse_irc_traj(self) -> Optional[List[Dict[str, tuple]]]:
        """
        Parse the IRC trajectory coordinates from an ESS log file.

        Returns: List[Dict[str, tuple]]
            The Cartesian coordinates for each scan point.
        """
        # Not implemented for Molpro.
        return None

    def parse_scan_conformers(self) -> Optional[pd.DataFrame]:
        """
        Parse all internal coordinates of scan conformers into a DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing internal coordinates for all conformers
        """
        # Not implemented for Molpro.
        return None

    def parse_nd_scan_energies(self) -> Optional[Dict]:
        """
        Parse the ND torsion scan energies from an ESS log file.

        Returns: dict
            The "results" dictionary
        """
        # Not implemented for Molpro.
        return None

    def parse_dipole_moment(self) -> Optional[float]:
        """
        Parse the dipole moment in Debye from an opt job output file.

        Returns: Optional[float]
            The dipole moment in Debye.
        """
        with open(self.log_file_path, 'r') as f:
            for line in f:
                if 'dipole moment' in line.lower() and '/debye' in line.lower():
                    # Example: Dipole moment /Debye                   2.96069859     0.00000000     0.00000000
                    try:
                        parts = line.split()
                        dm_x, dm_y, dm_z = float(parts[-3]), float(parts[-2]), float(parts[-1])
                        return (dm_x**2 + dm_y**2 + dm_z**2)**0.5
                    except (ValueError, IndexError):
                        continue
        return None

    def parse_polarizability(self) -> Optional[float]:
        """
        Parse the polarizability from a freq job output file, returns the value in Angstrom^3.

        Returns: Optional[float]
            The polarizability in Angstrom^3.
        """
        # Not implemented for Molpro.
        return None


register_ess_adapter('molpro', MolproParser)
