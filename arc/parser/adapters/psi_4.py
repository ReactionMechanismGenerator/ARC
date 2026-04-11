"""
An adapter for parsing Psi4 log files.
"""

from abc import ABC

import numpy as np
import pandas as pd

from arc.common import SYMBOL_BY_NUMBER
from arc.constants import E_h_kJmol
from arc.species.converter import xyz_from_data
from arc.parser.adapter import ESSAdapter
from arc.parser.factory import register_ess_adapter
from arc.parser.parser import _get_lines_from_file

class Psi4Parser(ESSAdapter, ABC):
    """
    A class for parsing Psi4 log files.

    Args:
        log_file_path (str): The path to the log file to be parsed.
    """
    def __init__(self, log_file_path: str):
        super().__init__(log_file_path=log_file_path)

    def logfile_contains_errors(self) -> str | None:
        """
        Check if the ESS log file contains any errors.

        Returns: str | None
            ``None`` if the log file is free of errors, otherwise the error is returned as a string.
        """
        error = None
        terminated = False
        with open(self.log_file_path, 'r') as f:
            lines = f.readlines()[-500:]  # Read more lines for safety

        for line in reversed(lines):
            if 'Psi4 exiting successfully' in line:
                terminated = True
            elif 'PSIO Error' in line:
                error = 'I/O error'
            elif 'Fatal Error' in line:
                error = 'Fatal Error'
            elif 'RuntimeError' in line:
                error = 'runtime'
            if error is not None:
                return f'There was an error ({error}) with the Psi4 output file {self.log_file_path} due to line:\n{line}'
        if not terminated:
            return f'Psi4 run in output file {self.log_file_path} did not successfully converge.'
        return None

    def parse_geometry(self) -> dict[str, tuple] | None:
        """
        Parse the xyz geometry from an ESS log file.

        Returns: dict[str, tuple] | None
            The cartesian geometry.
        """
        lines = _get_lines_from_file(self.log_file_path)
        coords, numbers = list(), list()
        for i in range(len(lines) - 1, -1, -1):
            if 'Center              X                  Y                   Z               Mass' in lines[i]:
                j = i + 2  # Skip header and separator line
                while j < len(lines) and lines[j].strip():
                    parts = lines[j].split()
                    if len(parts) >= 5:  # Atom symbol + XYZ coordinates + mass
                        try:
                            atom_symbol = parts[0]
                            # Handle Psi4's atom labeling (e.g., "H1" -> "H")
                            clean_symbol = ''.join(filter(str.isalpha, atom_symbol))
                            atomic_number = next(
                                k for k, v in SYMBOL_BY_NUMBER.items() if v == clean_symbol.capitalize())
                            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                            coords.append([x, y, z])
                            numbers.append(atomic_number)
                        except (ValueError, StopIteration, IndexError):
                            # Skip malformed lines but continue parsing
                            pass
                    j += 1
                if coords:
                    return xyz_from_data(coords=np.array(coords), numbers=np.array(numbers))
        return None

    def parse_frequencies(self) -> np.ndarray | None:
        """
        Parse the frequencies from a freq job output file.

        Returns: np.ndarray | None
            The parsed frequencies (in cm^-1).
        """
        frequencies = []
        lines = _get_lines_from_file(self.log_file_path)
        in_freq_block = False

        for i, line in enumerate(lines):
            if 'Harmonic Vibrational Analysis' in line:
                in_freq_block = True
            elif in_freq_block and 'Thermochemistry Components' in line:
                break
            elif in_freq_block and 'Freq [cm^-1]' in line:
                parts = line.split()
                # Psi4 prints 3 frequencies per line, but may print fewer at the end
                # They can be positive or negative (imaginary)
                for freq_str in parts[2:]:
                    try:
                        freq = float(freq_str)
                        frequencies.append(freq)
                    except ValueError:
                        continue
        if frequencies:
            return np.array(frequencies, dtype=np.float64)
        return None

    def parse_normal_mode_displacement(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        Parse frequencies and normal mode displacement.

        Returns: tuple[np.ndarray | None, np.ndarray | None]
            The frequencies (in cm^-1) and the normal mode displacements.
        """
        # Not implemented for Psi4.
        return None, None

    def parse_t1(self) -> float | None:
        """
        Parse the T1 parameter from a CC calculation.

        Returns: float | None
            The T1 parameter.
        """
        # Not implemented for Psi4.
        return None

    def parse_e_elect(self) -> float | None:
        """
        Parse the electronic energy from an sp job output file.

        Returns: float | None
            The electronic energy in kJ/mol.
        """
        lines = _get_lines_from_file(self.log_file_path)
        energy = None
        for line in reversed(lines):
            if 'Total Energy =' in line:
                try:
                    energy = float(line.split('=')[-1].strip())
                    break
                except (ValueError, IndexError):
                    continue
            if 'Final energy:' in line:
                try:
                    energy = float(line.split()[-1])
                    break
                except (ValueError, IndexError):
                    continue
        if energy is not None:
            return energy * E_h_kJmol
        return None

    def parse_zpe_correction(self) -> float | None:
        """
        Determine the calculated ZPE correction (E0 - e_elect) from a frequency output file.

        Returns: float | None
            The calculated zero point energy in kJ/mol.
        """
        zpe = None
        with open(self.log_file_path, 'r') as f:
            for line in f:
                if 'Zero-point vibrational energy' in line or 'Zero point energy' in line:
                    try:
                        zpe = float(line.split()[-2])
                        break
                    except (ValueError, IndexError):
                        continue
        if zpe is not None:
            return zpe * E_h_kJmol
        return None

    def parse_1d_scan_energies(self) -> tuple[list[float] | None, list[float] | None]:
        """
        Parse the 1D torsion scan energies from an ESS log file.

        Returns: tuple[list[float] | None, list[float] | None]
            The electronic energy in kJ/mol and the dihedral scan angle in degrees.
        """
        # Not implemented for Psi4.
        return None, None

    def parse_1d_scan_coords(self) -> list[dict[str, tuple]] | None:
        """
        Parse the 1D torsion scan coordinates from an ESS log file.

        Returns: list[dict[str, tuple]]
            The Cartesian coordinates for each scan point.
        """
        # Not implemented for Psi4.
        return None

    def parse_irc_traj(self) -> list[dict[str, tuple]] | None:
        """
        Parse the IRC trajectory coordinates from an ESS log file.

        Returns: list[dict[str, tuple]]
            The Cartesian coordinates for each scan point.
        """
        # Not implemented for Psi4.
        return None

    def parse_scan_conformers(self) -> pd.DataFrame | None:
        """
        Parse all internal coordinates of scan conformers into a DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing internal coordinates for all conformers
        """
        # Not implemented for Psi4.
        return None

    def parse_nd_scan_energies(self) -> dict | None:
        """
        Parse the ND torsion scan energies from an ESS log file.

        Returns: dict
            The "results" dictionary
        """
        # Not implemented for Psi4.
        return None

    def parse_dipole_moment(self) -> float | None:
        """
        Parse the dipole moment in Debye from an opt job output file.

        Returns: float | None
            The dipole moment in Debye.
        """
        with open(self.log_file_path, 'r') as f:
            for line in f:
                if 'Dipole Moment:' in line:
                    # Example: Dipole Moment:  0.0000  0.0000  1.8600  | 1.8600
                    try:
                        parts = line.split()
                        if len(parts) >= 5:
                            # The last value is the magnitude
                            return float(parts[-1])
                    except (ValueError, IndexError):
                        continue
        return None

    def parse_polarizability(self) -> float | None:
        """
        Parse the polarizability from a freq job output file, returns the value in Angstrom^3.

        Returns: float | None
            The polarizability in Angstrom^3.
        """
        # Not implemented for Psi4.
        return None

register_ess_adapter('psi4', Psi4Parser)
