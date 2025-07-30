"""
An adapter for parsing CFOUR log files.
"""

from abc import ABC

import numpy as np
from typing import TYPE_CHECKING, Optional, Tuple, List, Dict

from arc.constants import E_h_kJmol
from arc.species.converter import xyz_from_data
from arc.parser.adapter import ESSAdapter
from arc.parser.factory import register_ess_adapter
from arc.parser.parser import _get_lines_from_file

if TYPE_CHECKING:
    import pandas as pd


class TeraChemParser(ESSAdapter, ABC):
    """
    A class for parsing TeraChem log files.

    Args:
        log_file_path (str): The path to the log file to be parsed.
    """
    def __init__(self, log_file_path: str):
        super().__init__(log_file_path=log_file_path)

    def logfile_contains_errors(self) -> Optional[str]:
        """
        Check if the TeraChem log file contains any errors.

        Returns: Optional[str]
            None if no errors, else error message string.
        """
        lines = _get_lines_from_file(self.log_file_path)[-500:]
        for line in reversed(lines):
            l = line.lower()
            if 'incorrect method' in l:
                return 'Incorrect method specified'
            if 'error:' in l:
                # Extract the specific error message after "ERROR:"
                error_msg = line.split('ERROR:', 1)[-1].strip()
                return f'TeraChem error: {error_msg}'
            if 'scf failed to converge' in l:
                return 'SCF convergence failure'
            if 'geometry optimization failed' in l:
                return 'Geometry optimization failed to converge'
            if 'not enough memory' in l:
                return 'Insufficient memory'
            if 'license error' in l:
                return 'License validation failed'
        return None

    def parse_geometry(self) -> Optional[Dict[str, tuple]]:
        """
        Parse the xyz geometry from an ESS log file.

        Returns: Optional[Dict[str, tuple]]
            The cartesian geometry.
        """
        lines = _get_lines_from_file(self.log_file_path)
        coords, numbers = [], []
        found_qm_coords = False

        for i, line in enumerate(lines):
            if '****** QM coordinates ******' in line:
                found_qm_coords = True
                j = i + 1
                while j < len(lines):
                    parts = lines[j].split()
                    if len(parts) == 4:
                        symbol = parts[0]
                        try:
                            x, y, z = map(float, parts[1:])
                            from arc.common import SYMBOL_BY_NUMBER
                            atomic_number = next((num for num, sym in SYMBOL_BY_NUMBER.items() if sym == symbol), None)
                            if atomic_number is not None:
                                numbers.append(atomic_number)
                                coords.append([x, y, z])
                        except ValueError:
                            break  # hit end of coordinate block
                    else:
                        break
                    j += 1
                break

        if found_qm_coords and coords:
            return xyz_from_data(coords=np.array(coords), numbers=np.array(numbers))
        return None

    def parse_frequencies(self) -> Optional[np.ndarray]:
        """
        Parse the frequencies from a freq job output file.

        Returns: Optional[np.ndarray]
            The parsed frequencies (in cm^-1).
        """
        lines = _get_lines_from_file(self.log_file_path)
        frequencies = list()
        in_modes = False
        for line in lines:
            if 'Mode      Eigenvalue(AU)      Frequency(cm-1)' in line:
                in_modes = True
                continue
            if in_modes:
                if not line.strip():
                    break
                parts = line.split()
                if len(parts) >= 3:
                    freq_str = parts[2]
                    # Handle imaginary frequencies (e.g., 170.5666870932i)
                    if freq_str.endswith('i'):
                        freq = -float(freq_str[:-1])
                    else:
                        try:
                            freq = float(freq_str)
                        except ValueError:
                            continue
                    frequencies.append(freq)
            if '=== Mode' in line:
                # === Mode 2: 1271.913 cm^-1 ===
                freq = float(line.split()[3])
                frequencies.append(freq)
        if frequencies:
            return np.array(frequencies, dtype=np.float64)
        return None

    def parse_normal_mode_displacement(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Parse frequencies and normal mode displacement.

        Returns: Tuple[Optional['np.ndarray'], Optional['np.ndarray']]
            The frequencies (in cm^-1) and the normal mode displacements.
        """
        # Not implemented for TeraChem.
        return None, None

    def parse_t1(self) -> Optional[float]:
        """
        Parse the T1 parameter from a CC calculation.

        Returns: Optional[float]
            The T1 parameter.
        """
        # Not implemented for TeraChem.
        return None

    def parse_e_elect(self) -> Optional[float]:
        """
        Parse the electronic energy from an sp job output file.

        Returns: Optional[float]
            The electronic energy in kJ/mol.
        """
        energy = None
        with open(self.log_file_path, 'r') as f:
            lines = f.readlines()
        return_first = False
        for i, line in enumerate(lines):
            if 'FREQUENCY ANALYSIS' in line:
                return_first = True
            if 'Ground state energy (a.u.):' in line:
                try:
                    energy = float(lines[i + 1].strip())
                    if return_first:
                        break
                except (ValueError, IndexError):
                    continue
            if 'FINAL ENERGY:' in line:
                try:
                    energy = float(line.split()[2])
                    if return_first:
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
                if 'Vibrational zero-point energy (ZPE)' in line:
                    # Example: 'Vibrational zero-point energy (ZPE) = 243113.467652369843563065 J/mol =     0.09259703 AU'
                    try:
                        zpe = float(line.split('J/mol')[0].split()[-1])
                        break
                    except (ValueError, IndexError):
                        continue
        return zpe

    def parse_1d_scan_energies(self) -> Tuple[Optional[List[float]], Optional[List[float]]]:
        """
        Parse the 1D torsion scan energies from an ESS log file.

        Returns: Tuple[Optional[List[float]], Optional[List[float]]]
            The electronic energy in kJ/mol and the dihedral scan angle in degrees.
        """
        energies = []
        angles = []
        with open(self.log_file_path, 'r') as f:
            lines = f.readlines()
        v_index, expected_num_of_points = 0, 0
        for line in lines:
            if 'Scan Cycle' in line:
                v_index += 1
                if not expected_num_of_points:
                    expected_num_of_points = int(line.split()[3].split('/')[1])
            if 'Optimized Energy:' in line:
                try:
                    v = float(line.split()[3])
                    if len(energies) == v_index - 1:
                        energies.append(v)
                    elif len(energies) < v_index - 1:
                        energies.extend([None] * (v_index - 1 - len(energies)))
                    else:
                        # More points than expected, something is wrong
                        return None, None
                except (ValueError, IndexError):
                    continue
        if not energies or not expected_num_of_points:
            return None, None
        # Remove None's (missing points)
        energies = [e for e in energies if e is not None]
        # Convert to kJ/mol and normalize
        energies = np.array(energies, float)
        energies -= np.min(energies)
        energies *= E_h_kJmol
        # Angles: evenly spaced over 0-360 degrees
        if len(energies) > 1:
            angles = np.linspace(0, 360, len(energies), endpoint=True)
        else:
            angles = [0.0]
        return list(energies), list(angles)

    def parse_1d_scan_coords(self) -> Optional[List[Dict[str, tuple]]]:
        """
        Parse the 1D torsion scan coordinates from an ESS log file.

        Returns: List[Dict[str, tuple]]
            The Cartesian coordinates for each scan point.
        """
        # Not implemented for TeraChem.
        return None

    def parse_irc_traj(self) -> Optional[List[Dict[str, tuple]]]:
        """
        Parse the IRC trajectory coordinates from an ESS log file.

        Returns: List[Dict[str, tuple]]
            The Cartesian coordinates for each scan point.
        """
        # Not implemented for TeraChem.
        return None

    def parse_scan_conformers(self) -> Optional['pd.DataFrame']:
        """
        Parse all internal coordinates of scan conformers into a DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing internal coordinates for all conformers
        """
        # Not implemented for TeraChem.
        return None

    def parse_nd_scan_energies(self) -> Optional[Dict]:
        """
        Parse the ND torsion scan energies from an ESS log file.

        Returns: dict
            The "results" dictionary
        """
        # Not implemented for TeraChem.
        return None

    def parse_dipole_moment(self) -> Optional[float]:
        """
        Parse the dipole moment in Debye from an opt job output file.

        Returns: Optional[float]
            The dipole moment in Debye.
        """
        dipole_moment = None
        with open(self.log_file_path, 'r') as f:
            for line in f:
                if 'DIPOLE MOMENT:' in line and 'DEBYE' in line:
                    # Example: DIPOLE MOMENT: {-0.000178, -0.000003, -0.000019} (|D| = 0.000179) DEBYE
                    try:
                        # Extract value between (|D| = ... ) DEBYE
                        if '(|D| =' in line:
                            value = line.split('(|D| =')[1].split(')')[0].strip()
                            dipole_moment = float(value)
                    except (ValueError, IndexError):
                        continue
        return dipole_moment

    def parse_polarizability(self) -> Optional[float]:
        """
        Parse the polarizability from a freq job output file, returns the value in Angstrom^3.

        Returns: Optional[float]
            The polarizability in Angstrom^3.
        """
        # Not implemented for TeraChem.
        return None


register_ess_adapter('terachem', TeraChemParser)
