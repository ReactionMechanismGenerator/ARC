"""
An adapter for parsing xTB log files.
"""

from abc import ABC
from typing import Dict, List, Optional, Tuple

import numpy as np
import os
import pandas as pd

from arc.common import SYMBOL_BY_NUMBER
from arc.constants import E_h_kJmol
from arc.species.converter import str_to_xyz
from arc.parser.adapter import ESSAdapter
from arc.parser.factory import register_ess_adapter
from arc.parser import _get_lines_from_file

class XTBParser(ESSAdapter, ABC):
    """
    A class for parsing xTB log files.

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
        # Not implemented for xTB.
        return None

    def parse_geometry(self) -> Optional[Dict[str, tuple]]:
        """
        Parse the latest xyz geometry from an xTB log file.

        Returns: Optional[Dict[str, tuple]]
            The cartesian geometry.
        """
        lines = _get_lines_from_file(self.log_file_path)
        xyz_str = ""
        final_structure, in_coord_block, first_line = False, False, True
        for line in lines:
            if 'final structure:' in line:
                final_structure = True
            if final_structure and '$coord' in line:
                in_coord_block = True
                continue
            if in_coord_block and ('$' in line or 'end' in line.lower() or len(line.split()) < 4):
                in_coord_block = False
                continue
            if not in_coord_block:
                continue
            parts = line.split()
            if len(parts) < 4 and not (first_line and len(parts) > 4):
                continue
            try:
                atomic_num = int(float(parts[3]))
                element = SYMBOL_BY_NUMBER.get(atomic_num, 'X')
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                xyz_str += f"{element} {x} {y} {z}\n"
                if first_line:
                    first_line = False
            except (ValueError, IndexError):
                pass
        return str_to_xyz(xyz_str) if xyz_str else None

    def parse_frequencies(self) -> Optional['np.ndarray']:
        """
        Parse the frequencies from an xTB freq job output file.

        Returns: Optional[np.ndarray]
            The parsed frequencies (in cm^-1), including negative (imaginary) ones.
        """
        freqs = []
        lines = _get_lines_from_file(self.log_file_path)
        read_output = False

        for line in lines:
            if read_output:
                if 'eigval :' in line:
                    splits = line.split()
                    for split in splits[2:]:
                        try:
                            freq = float(split)
                            if freq != 0.0:
                                freqs.append(freq)
                        except ValueError:
                            continue
                elif line.strip() == "" or "projected vibrational frequencies" in line.lower():
                    continue
                else:
                    break
            if 'vibrational frequencies' in line.lower():
                read_output = True

        # Fallback: try vibspectrum file if no frequencies found in output
        if not freqs:
            vibspectrum_path = os.path.join(os.path.dirname(self.log_file_path), 'vibspectrum')
            if os.path.isfile(vibspectrum_path):
                with open(vibspectrum_path, 'r') as f:
                    for line in f:
                        if '$' in line or '#' in line:
                            continue
                        splits = line.split()
                        if len(splits) < 5:
                            continue
                        try:
                            freq = float(splits[-4])
                            if freq != 0.0:
                                freqs.append(freq)
                        except ValueError:
                            continue

        return np.array(freqs, dtype=np.float64) if freqs else None

    def parse_normal_mode_displacement(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Parse frequencies and normal mode displacements from xTB frequency job.

        Returns: Tuple[Optional[np.ndarray], Optional[np.ndarray]]
            The frequencies (in cm^-1) and the normal mode displacements.
            Displacement array shape: (n_modes, n_atoms, 3)
        """
        # Locate the g98.out file in the same directory as the log file
        g98_path = os.path.join(os.path.dirname(self.log_file_path), 'g98.out')
        if not os.path.isfile(g98_path):
            return None, None

        freqs = list()
        normal_mode_disp = list()  # Will be list of modes, each mode is list of atomic displacements
        current_mode_entries = list()  # Collects displacement entries for current frequency block
        num_modes_in_block = 0

        with open(g98_path, 'r') as f:
            lines = f.readlines()

        parse_freqs, parse_disp = False, False

        for line in lines:
            # Start of frequency block
            if 'Harmonic frequencies (cm**-1)' in line:
                parse_freqs, parse_disp = True, False

            # End of frequency block
            if not line.strip() or '-------------------' in line:
                if parse_disp and current_mode_entries:
                    # Process collected displacement entries
                    n_atoms = len(current_mode_entries)
                    # Transpose: from [atom][mode] to [mode][atom]
                    for mode_idx in range(num_modes_in_block):
                        mode_disp = []
                        for atom_idx in range(n_atoms):
                            start = 3 * mode_idx
                            mode_disp.append(current_mode_entries[atom_idx][start:start + 3])
                        normal_mode_disp.append(mode_disp)
                    current_mode_entries = []
                parse_freqs, parse_disp = False, False

            # Parse frequencies
            if parse_freqs and 'Frequencies --' in line:
                parts = line.split()[2:]
                freqs.extend(float(f) for f in parts)
                num_modes_in_block = len(parts)

            # Start of displacement section
            if parse_freqs and ('Atom  AN      X      Y      Z' in line or 'Atom AN      X      Y      Z' in line):
                parse_disp = True
                current_mode_entries = list()
                continue

            # Parse displacement lines
            if parse_disp:
                parts = line.split()
                if len(parts) < 2:
                    continue
                try:
                    # Skip atom index and atomic number, convert displacement values
                    disp_values = [float(x) for x in parts[2:2 + 3 * num_modes_in_block]]
                    current_mode_entries.append(disp_values)
                except (ValueError, IndexError):
                    continue

        if not freqs or not normal_mode_disp:
            return None, None

        return (
            np.array(freqs, dtype=np.float64),
            np.array(normal_mode_disp, dtype=np.float64)
        )

    def parse_t1(self) -> Optional[float]:
        """
        Parse the T1 parameter from a CC calculation.

        Returns: Optional[float]
            The T1 parameter.
        """
        # Not implemented for xTB.
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

    def parse_zpe_correction(self) -> Optional[float]:
        """
        Determine the calculated ZPE correction from a frequency output file.

        Returns: Optional[float]
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

    def parse_1d_scan_energies(self) -> Tuple[Optional[List[float]], Optional[List[float]]]:
        """
        Parse the 1D torsion scan energies from an xTB scan log file.

        Returns: Tuple[Optional[List[float]], Optional[List[float]]]
            The electronic energy in kJ/mol and the dihedral scan angle in degrees.
        """
        scan_path = os.path.join(os.path.dirname(self.log_file_path), 'xtbscan.log')
        if not os.path.isfile(scan_path):
            return None, None

        energies = list()
        lines = _get_lines_from_file(scan_path)
        for line in lines:
            if 'SCF done' in line:
                try:
                    # Example: SCF done -7.33636977
                    energy = float(line.split()[-1])
                    # xTB energies are in Hartree, convert to kJ/mol
                    energies.append(energy * 2625.49962)
                except (ValueError, IndexError):
                    continue

        if not energies:
            return None, None

        min_e = min(energies)
        rel_energies = [e - min_e for e in energies]
        n_points = len(rel_energies)
        if n_points == 0:
            return None, None

        # Angles: evenly spaced from 0 to <360, step = 360/n_points
        resolution = 360.0 / n_points
        angles = [i * resolution for i in range(n_points)]

        return rel_energies, angles

    def parse_1d_scan_coords(self) -> Optional[List[Dict[str, tuple]]]:
        """
        Parse the 1D torsion scan coordinates from an xTB scan log file.

        Returns: Optional[List[Dict[str, tuple]]]
            The Cartesian coordinates for each scan point.
        """
        scan_path = os.path.join(os.path.dirname(self.log_file_path), 'xtbscan.log')
        if not os.path.isfile(scan_path):
            return None

        lines = _get_lines_from_file(scan_path)
        traj = list()
        xyz_str = ''
        in_structure = False
        atom_count = 0
        atoms_parsed = 0

        for line in lines:
            stripped = line.strip()

            # Start of new structure
            if stripped.isdigit():
                if xyz_str:
                    traj.append(str_to_xyz(xyz_str))
                    xyz_str = ''
                atom_count = int(stripped)
                atoms_parsed = 0
                in_structure = True
                continue

            # Skip comment/energy lines
            if in_structure and 'energy:' in stripped.lower():
                continue

            # Parse atom lines
            if in_structure and atoms_parsed < atom_count:
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        # Format: <element> <x> <y> <z>
                        element = parts[0]
                        # Validate element symbol
                        if element.isalpha() and len(element) < 3:
                            xyz_str += f"{element} {parts[1]} {parts[2]} {parts[3]}\n"
                            atoms_parsed += 1
                    except (IndexError, ValueError):
                        continue

            # Finalize structure after last atom
            if in_structure and atoms_parsed >= atom_count:
                if xyz_str:
                    traj.append(str_to_xyz(xyz_str))
                    xyz_str = ''
                in_structure = False

        # Handle last structure in file
        if xyz_str:
            traj.append(str_to_xyz(xyz_str))

        return traj if traj else None

    def parse_scan_conformers(self) -> Optional['pd.DataFrame']:
        """
        Parse all internal coordinates of scan conformers into a DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing internal coordinates for all conformers
        """
        # Not implemented for xTB.
        return None

    def parse_nd_scan_energies(self) -> Optional[Dict]:
        """
        Parse the ND torsion scan energies from an ESS log file.

        Returns: dict
            The "results" dictionary
        """
        # Not implemented for xTB.
        return None

    def parse_dipole_moment(self) -> Optional[float]:
        """
        Parse the dipole moment in Debye from an opt job output file.

        Returns: Optional[float]
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

    def parse_polarizability(self) -> Optional[float]:
        """
        Parse the polarizability from a freq job output file, returns the value in Angstrom^3.

        Returns: Optional[float]
            The polarizability in Angstrom^3.
        """
        # Not implemented for xTB.
        return None


register_ess_adapter('xtb', XTBParser)
