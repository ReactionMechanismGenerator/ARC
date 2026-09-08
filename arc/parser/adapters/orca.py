"""
An adapter for parsing Orca log files.
"""

from abc import ABC

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

    def logfile_contains_errors(self) -> str | None:
        """
        Check if the ESS log file contains any errors.

        Returns: str | None
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

    def parse_geometry(self) -> dict[str, tuple] | None:
        """
        Parse the xyz geometry from an ESS log file.

        Returns: dict[str, tuple] | None
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

    def _parse_frequencies(self, include_zeros: bool = False) -> np.ndarray | None:
        """
        Parse the frequencies from a freq job output file.

        Args:
            include_zeros (bool, optional): Whether to retain exact-zero translation and rotation modes.

        Returns: np.ndarray | None
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
                            # Keep negative freqs (imaginary modes), optionally drop exact-zero translations/rotations.
                            if include_zeros or abs(freq) > 0.0:
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

    def parse_frequencies(self) -> np.ndarray | None:
        """
        Parse the nonzero frequencies from a freq job output file.

        Returns: np.ndarray | None
            The parsed nonzero frequencies (in cm^-1).
        """
        return self._parse_frequencies(include_zeros=False)

    def parse_normal_mode_displacement(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        Parse the frequencies and normal mode displacements from an Orca frequency job output file.

        Returns:
            tuple[np.ndarray | None, np.ndarray | None]:
                - frequencies (in cm^-1), exact-zero translation/rotation modes excluded.
                - normal mode displacements, shape (num_modes, num_atoms, 3), same mode order as frequencies.
        """
        with open(self.log_file_path, 'r') as f:
            lines = f.readlines()

        all_freqs = self._parse_frequencies(include_zeros=True)
        if all_freqs is None:
            return None, None
        n_dof = len(all_freqs)

        start = None
        for i, line in enumerate(lines):
            if line.strip() == 'NORMAL MODES':
                start = i
                break
        if start is None:
            return None, None

        full_matrix = [[0.0] * n_dof for _ in range(n_dof)]
        n_cols_parsed = 0
        i = start
        while n_cols_parsed < n_dof and i < len(lines):
            stripped = lines[i].strip()
            if stripped and '.' not in stripped:
                try:
                    col_indices = [int(tok) for tok in stripped.split()]
                except ValueError:
                    i += 1
                    continue
                i += 1
                for row in range(n_dof):
                    if i >= len(lines):
                        return None, None
                    vals = lines[i].split()[1:]
                    if len(vals) < len(col_indices):
                        return None, None
                    for k, col in enumerate(col_indices):
                        try:
                            full_matrix[row][col] = float(vals[k])
                        except (IndexError, ValueError):
                            return None, None
                    i += 1
                n_cols_parsed += len(col_indices)
            else:
                i += 1
        if n_cols_parsed < n_dof:
            return None, None

        keep = [idx for idx, freq in enumerate(all_freqs) if freq != 0.0]
        freqs = np.array([all_freqs[idx] for idx in keep], dtype=np.float64)
        n_atoms = n_dof // 3
        full_matrix_np = np.array(full_matrix, dtype=np.float64)
        normal_modes_disp = np.array(
            [full_matrix_np[:, idx].reshape(n_atoms, 3) for idx in keep],
            dtype=np.float64,
        )
        return freqs, normal_modes_disp

    def parse_t1(self) -> float | None:
        """
        Parse the T1 parameter from a CC calculation.

        Returns: float | None
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

    def parse_e_elect(self) -> float | None:
        """
        Parse the electronic energy from an sp job output file.

        Returns: float | None
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

    def parse_zpe_correction(self) -> float | None:
        """
        Determine the calculated ZPE correction (E0 - e_elect) from a frequency output file.

        Returns: float | None
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

    def parse_1d_scan_energies(self) -> tuple[list[float] | None, list[float] | None]:
        """
        Parse the 1D torsion scan energies from an ESS log file.

        Returns: tuple[list[float] | None, list[float] | None]
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

    def parse_1d_scan_coords(self) -> list[dict[str, tuple]] | None:
        """
        Parse the 1D torsion scan coordinates from an ESS log file.

        Returns: list[dict[str, tuple]]
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

    def parse_irc_traj(self) -> list[dict[str, tuple]] | None:
        """
        Parse the IRC trajectory coordinates from an ESS log file.

        Returns: list[dict[str, tuple]]
            The Cartesian coordinates for each scan point.
        """
        # Not implemented for Orca.
        return None

    def parse_scan_conformers(self) -> pd.DataFrame | None:
        """
        Parse all internal coordinates of scan conformers into a DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing internal coordinates for all conformers
        """
        # Not implemented for Orca.
        return None

    def parse_nd_scan_energies(self) -> dict | None:
        """
        Parse the ND torsion scan energies from an ESS log file.

        Returns: dict
            The "results" dictionary
        """
        # Not implemented for Orca.
        return None

    def parse_dipole_moment(self) -> float | None:
        """
        Parse the dipole moment in Debye from an opt job output file.

        Returns: float | None
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

    def parse_polarizability(self) -> float | None:
        """
        Parse the polarizability from a freq job output file, returns the value in Angstrom^3.

        Returns: float | None
            The polarizability in Angstrom^3.
        """
        # Not implemented for Orca.
        return None

    def parse_ess_version(self) -> str | None:
        """
        Parse the ORCA version string, e.g. ``'ORCA 5.0.4'``.
        """
        with open(self.log_file_path, 'r') as f:
            for line in f:
                # "Program Version 5.0.4 -  RELEASE  -"
                m = re.search(r'Program Version\s+([\d.]+)', line)
                if m:
                    return f'ORCA {m.group(1)}'
        return None

    def parse_bond_orders(self) -> np.ndarray | None:
        """
        Parse the Mayer bond order matrix from an Orca log file.

        Orca prints a "MAYER POPULATION ANALYSIS" section by default. It consists of a per-atom
        table (from which the number of atoms and the Mayer total valence ``VA`` are taken) followed
        by a sparse list of atom pairs, e.g. ``B(  0-O ,  1-C ) :   1.0172``. Note that Orca only
        prints pairs above a threshold (0.1 by default), so bond orders below that threshold are
        reported here as zero. The last such section in the file is used.

        Returns: np.ndarray | None
            A symmetric NxN matrix of Mayer bond orders in the log file's atom order.
            The diagonal holds the Mayer atomic valence of each atom.
            ``None`` if the log file does not contain a Mayer population analysis.
        """
        lines = _get_lines_from_file(self.log_file_path)
        start = None
        for i in reversed(range(len(lines))):
            if 'MAYER POPULATION ANALYSIS' in lines[i]:
                start = i + 1
                break
        if start is None:
            return None
        atom_pattern = re.compile(r'^\s*(\d+)\s+[A-Za-z]{1,3}\s+(-?\d+\.\d+(?:\s+-?\d+\.\d+){5})\s*$')
        bond_pattern = re.compile(r'B\(\s*(\d+)-\w+\s*,\s*(\d+)-\w+\s*\)\s*:\s*(-?\d+\.\d+)')
        valences, bonds = dict(), dict()
        for line in lines[start:]:
            if 'TIMINGS' in line:
                break
            match = atom_pattern.match(line)
            if match is not None:
                # The six columns are NA, ZA, QA, VA, BVA and FA; VA is the Mayer total valence.
                valences[int(match.group(1))] = float(match.group(2).split()[3])
                continue
            for i, j, value in bond_pattern.findall(line):
                bonds[(int(i), int(j))] = float(value)
        if not valences:
            return None
        n = max(valences.keys()) + 1
        bond_orders = np.zeros((n, n), np.float64)
        for i, valence in valences.items():
            bond_orders[i, i] = valence
        for (i, j), value in bonds.items():
            if i >= n or j >= n:
                return None
            bond_orders[i, j] = bond_orders[j, i] = value
        return bond_orders


register_ess_adapter('orca', OrcaParser)
