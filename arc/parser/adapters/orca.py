"""
An adapter for parsing Orca log files.
"""

from abc import ABC
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import re

from arc.common import SYMBOL_BY_NUMBER
from arc.constants import E_h_kJmol, bohr_to_angstrom
from arc.species.converter import str_to_xyz, xyz_from_data
from arc.parser.adapter import ESSAdapter
from arc.parser.factory import register_ess_adapter
from arc.parser.parser import _get_lines_from_file


class OrcaParser(ESSAdapter, ABC):
    """
    A class for parsing Orca log files.

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
            lines = f.readlines()
        # Check last 100 lines first (most likely location for errors)
        for line in reversed(lines[-100:]):
            if 'ORCA TERMINATED NORMALLY' in line:
                return None
            if 'ORCA finished by error termination in SCF' in line:
                return 'SCF convergence failure'
            if 'ORCA finished by error termination in MDCI' in line:
                return 'MDCI calculation error'
            if 'Error : multiplicity' in line:
                return 'Invalid multiplicity/charge combination'
            if 'ORCA TERMINATED ABNORMALLY' in line:
                return 'ORCA terminated abnormally'

        # If nothing in last 100 lines, check entire file for specific errors
        for line in reversed(lines):
            if 'ORCA finished by error termination in SCF' in line:
                return 'SCF convergence failure'
            if 'ORCA finished by error termination in MDCI' in line:
                return 'MDCI calculation error'
            if 'Error : multiplicity' in line:
                return 'Invalid multiplicity/charge combination'
            if 'ORCA TERMINATED ABNORMALLY' in line:
                return 'ORCA terminated abnormally'
            if 'ORCA ran out of memory' in line:
                return 'Insufficient memory'
            if 'Geometry optimization failed' in line:
                return 'Geometry optimization failed to converge'

        # Check for common warning patterns that indicate errors
        for line in reversed(lines):
            if 'This wavefunction IS NOT CONVERGED!' in line:
                return 'SCF wavefunction not converged'
            if 'Convergence failure' in line:
                return 'Convergence failure'
            if 'Error' in line and 'termination' in line:
                return line.strip()

        return None

    def parse_geometry(self) -> Optional[Dict[str, tuple]]:
        """
        Parse the xyz geometry from an ESS log file.

        Returns: Optional[Dict[str, tuple]]
            The cartesian geometry.
        """
        lines = _get_lines_from_file(self.log_file_path)
        coords, numbers = list(), list()
        for i in range(len(lines) - 1, -1, -1):
            if 'CARTESIAN COORDINATES (A.U.)' in lines[i] or 'CARTESIAN COORDINATES (ANGSTROEM)' in lines[i]:
                unit = 'bohr' if 'A.U.)' in lines[i] else 'angstrom'
                j = i + 2  # Skip header lines
                # Parse atom lines until separator or empty line
                while j < len(lines) and lines[j].strip() and '----' not in lines[j]:
                    parts = lines[j].split()
                    if len(parts) < 4:
                        j += 1
                        continue
                    try:
                        atom_symbol = parts[0].capitalize()
                        atomic_number = next(k for k, v in SYMBOL_BY_NUMBER.items() if v == atom_symbol)
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        if unit == 'bohr':
                            x *= bohr_to_angstrom
                            y *= bohr_to_angstrom
                            z *= bohr_to_angstrom
                        coords.append([x, y, z])
                        numbers.append(atomic_number)
                    except (ValueError, StopIteration, IndexError):
                        # Skip malformed lines but continue parsing
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
        frequencies = list()
        found_freqs = False

        with open(self.log_file_path, 'r') as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i]
            if 'VIBRATIONAL FREQUENCIES' in line:
                i += 4
                while i < len(lines):
                    freq_line = lines[i].strip()
                    if not freq_line:
                        i += 1
                        continue
                    parts = freq_line.split()
                    if len(parts) >= 2 and parts[0].rstrip(':').isdigit():
                        try:
                            freq = float(parts[1])
                            # Keep negative freqs (imaginary modes), drop exact zeros (translations/rotations).
                            if abs(freq) > 0.0:
                                frequencies.append(freq)
                            found_freqs = True
                        except ValueError:
                            pass
                    else:
                        if found_freqs:
                            break
                    i += 1
                break
            i += 1

        return np.array(frequencies, dtype=np.float64) if frequencies else None

    def parse_normal_mode_displacement(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Parse frequencies and normal mode displacement.

        Returns: Tuple[Optional['np.ndarray'], Optional['np.ndarray']]
            The frequencies (in cm^-1) and the normal mode displacements.
        """
        freqs = self.parse_frequencies()
        if freqs is None:
            return None, None

        mode_data = {}

        with open(self.log_file_path, 'r') as f:
            lines = f.readlines()

        in_normal_modes = False
        i = 0
        while i < len(lines):
            line = lines[i]
            if 'NORMAL MODES' in line:
                in_normal_modes = True
                i += 1
                continue
            if not in_normal_modes:
                i += 1
                continue
            if 'IR SPECTRUM' in line or 'THERMOCHEMISTRY' in line:
                break

            stripped = line.strip()
            if not stripped:
                i += 1
                continue

            tokens = stripped.split()
            if tokens and all(tok.isdigit() for tok in tokens):
                mode_ids = [int(tok) for tok in tokens]
                for mode_id in mode_ids:
                    mode_data.setdefault(mode_id, [])
                i += 1
                while i < len(lines):
                    row = lines[i].strip()
                    if not row:
                        break
                    row_tokens = row.split()
                    if row_tokens and all(tok.isdigit() for tok in row_tokens):
                        # Next header starts here; let the outer loop process it.
                        break
                    if not row_tokens or not row_tokens[0].isdigit():
                        break
                    if len(row_tokens) < 1 + len(mode_ids):
                        i += 1
                        continue
                    try:
                        values = [float(val) for val in row_tokens[1:1 + len(mode_ids)]]
                    except ValueError:
                        i += 1
                        continue
                    for mode_id, val in zip(mode_ids, values):
                        mode_data[mode_id].append(val)
                    i += 1
                continue
            i += 1

        if not mode_data:
            return None, None

        mode_ids = sorted(mode_data.keys())
        total_modes = len(mode_ids)
        n_zero_modes = total_modes - len(freqs)
        if n_zero_modes < 0:
            return None, None

        first_mode = mode_ids[0]
        coord_count = len(mode_data[first_mode])
        if coord_count == 0 or coord_count % 3 != 0:
            return None, None
        n_atoms = coord_count // 3

        displacements = []
        for mode_id in mode_ids[n_zero_modes:]:
            coords = mode_data[mode_id]
            if len(coords) != n_atoms * 3:
                return None, None
            displacements.append(np.array(coords, dtype=np.float64).reshape((n_atoms, 3)))

        if not displacements:
            return None, None

        return freqs, np.array(displacements, dtype=np.float64)

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
        energy = None
        for line in reversed(lines):
            if 'FINAL SINGLE POINT ENERGY' in line:
                try:
                    energy = float(line.split()[-1])
                    break
                except (ValueError, IndexError):
                    continue
            if 'Total Energy       :' in line:
                try:
                    energy = float(line.split()[-1])
                    break
                except (ValueError, IndexError):
                    continue
            if 'E' in line and 'HF' in line and 'FINAL' in line:
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
                if 'Zero point energy' in line:
                    # Example: Zero point energy      ...    0.025410 Eh
                    try:
                        parts = line.split()
                        if 'Eh' in parts:
                            zpe = float(parts[parts.index('Eh') - 1])
                        else:
                            zpe = float(parts[-2])
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
        cs, es = [], []
        with open(self.log_file_path, "r") as f:
            flag_actual = False
            for line in f.readlines():
                if "The Calculated Surface using the 'Actual Energy'" in line:
                    flag_actual = True
                elif flag_actual:
                    if not line.strip():
                        break
                    else:
                        c, e = line.split()
                        cs.append(float(c))
                        es.append(float(e))
        if len(cs) != len(es) or not cs:
            raise ValueError("Failed to parse 1D scan energies from Orca log file.")
        return np.array(es), np.array(cs)

    def parse_1d_scan_coords(self) -> Optional[List[Dict[str, tuple]]]:
        """
        Parse the 1D torsion scan coordinates from an ESS log file.

        Returns: List[Dict[str, tuple]]
            The Cartesian coordinates for each scan point.
        """
        coords_list = []
        with open(self.log_file_path, "r") as f:
            flag_hurray, flag_coords = False, False
            pat = re.compile(
                            r'^\s*([A-Z][a-z]?)\s+'
                            r'([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s+'
                            r'([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s+'
                            r'([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$',
                            re.MULTILINE
                            )
            for line in f.readlines():
                if "HURRAY" in line:
                    coords = """"""
                    flag_hurray = True
                if flag_hurray and "CARTESIAN COORDINATES (ANGSTROEM)" in line:
                    flag_coords = True
                if flag_hurray and flag_coords:
                    if not line.strip():
                        coords_list.append(str_to_xyz(coords))
                        flag_hurray, flag_coords = False, False
                    if bool(pat.match(line)):
                        coords += line
            if not coords_list:
                raise ValueError("Failed to parse 1D scan coordinates from Orca log file.")
        return coords_list

    def parse_irc_traj(self) -> Optional[List[Dict[str, tuple]]]:
        """
        Parse the IRC trajectory coordinates from an ESS log file.

        Returns: List[Dict[str, tuple]]
            The Cartesian coordinates for each scan point.
        """
        # Not implemented for Orca.
        return None

    def parse_scan_conformers(self) -> Optional[pd.DataFrame]:
        """
        Parse all internal coordinates of scan conformers into a DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing internal coordinates for all conformers
        """
        # Not implemented for Orca.
        return None

    def parse_nd_scan_energies(self) -> Optional[Dict]:
        """
        Parse the ND torsion scan energies from an ESS log file.

        Returns: dict
            The "results" dictionary
        """
        # Not implemented for Orca.
        return None

    def parse_dipole_moment(self) -> Optional[float]:
        """
        Parse the dipole moment in Debye from an opt job output file.

        Returns: Optional[float]
            The dipole moment in Debye.
        """
        with open(self.log_file_path, 'r') as f:
            for line in f:
                if 'Magnitude (Debye)' in line:
                    try:
                        return float(line.split()[-1])
                    except (ValueError, IndexError):
                        continue
        return None

    def parse_polarizability(self) -> Optional[float]:
        """
        Parse the polarizability from a freq job output file, returns the value in Angstrom^3.

        Returns: Optional[float]
            The polarizability in Angstrom^3.
        """
        # Not implemented for Orca.
        return None


register_ess_adapter('orca', OrcaParser)
