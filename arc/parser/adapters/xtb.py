"""
An adapter for parsing xTB log files.
"""

from abc import ABC

import math
import numpy as np
import os
import pandas as pd

from arc.common import is_str_float
from arc.constants import E_h_kJmol, bohr_to_angstrom
from arc.species.converter import str_to_xyz, xyz_from_data, logger
from arc.parser.adapter import ESSAdapter
from arc.parser.factory import register_ess_adapter
from arc.parser.parser import _get_lines_from_file


class XTBParser(ESSAdapter, ABC):
    """
    A class for parsing xTB log files.

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
            for line in f:
                if 'abnormal termination' in line.lower():
                    return line.strip()
                if '[ERROR]' in line or '#ERROR' in line:
                    return line.strip()
        return None

    def parse_geometry(self) -> dict[str, tuple] | None:
        """
        Parse the xyz geometry from an ESS log file.

        Supports both Turbomol ``$coord`` (Bohr) and Molfile V2000 (Angstrom) formats.
        If the file contains multiple geometry blocks (e.g., from multiple optimization
        cycles), only the last one is returned.

        Returns: dict[str, tuple] | None
            The cartesian geometry.
        """
        lines = _get_lines_from_file(self.log_file_path)
        coords, symbols = list(), list()
        in_coord_block, final_structure, molfile_mode = False, False, False
        molfile_line_counter, atom_count = 0, 0

        for i, line in enumerate(lines):
            line = line.strip()

            # Detect start of either format
            if 'final structure:' in line.lower():
                final_structure = True
                continue
            if final_structure and '$coord' in line:
                # Reset on each new $coord block so we keep only the last one
                coords, symbols = list(), list()
                in_coord_block = True
                continue
            if final_structure and 'V2000' in line and not in_coord_block:
                molfile_mode = True
                parts = line.strip().split()
                if parts and parts[0].isdigit():
                    atom_count = int(parts[0])
                    molfile_line_counter = 0
                # Reset on each new V2000 block
                coords, symbols = list(), list()
                continue

            # Parse $coord format (Turbomole $coord coordinates are in Bohr, convert to Angstrom)
            if in_coord_block:
                if '$' in line or 'end' in line.lower() or len(line.split()) < 4:
                    in_coord_block = False
                    continue
                parts = line.split()
                try:
                    x, y, z = (float(v) * bohr_to_angstrom for v in parts[:3])
                    symbol = parts[3].capitalize() if len(parts[3]) == 1 else parts[3][0].upper() + parts[3][1:].lower()
                    coords.append([x, y, z])
                    symbols.append(symbol)
                except ValueError:
                    continue

            # Parse Molfile atom block
            elif molfile_mode and molfile_line_counter < atom_count:
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        x, y, z = map(float, parts[:3])
                        symbol = parts[3].capitalize() if len(parts[3]) == 1 else parts[3][0].upper() + parts[3][1:].lower()
                        coords.append([x, y, z])
                        symbols.append(symbol)
                        molfile_line_counter += 1
                    except ValueError:
                        continue
                else:
                    continue
            elif molfile_mode and molfile_line_counter >= atom_count:
                molfile_mode = False  # done reading mol block

        return xyz_from_data(coords=np.array(coords), symbols=symbols) if coords else None

    def parse_frequencies(self) -> np.ndarray | None:
        """
        Parse the frequencies from a freq job output file.

<<<<<<< HEAD
        xTB prints frequencies twice (once after the Hessian and once in the
        Frequency Printout section). This method reads ALL eigval blocks and
        returns only the last complete one to ensure we get the final values.

        Returns: Optional[np.ndarray]
=======
        Returns: np.ndarray | None
>>>>>>> c8fe67a3 (Modernize typing imports for Python 3.14+)
            The parsed frequencies (in cm^-1).
        """
        # Collect all eigval blocks; use the last one
        all_blocks = list()
        current_block = list()
        lines = _get_lines_from_file(self.log_file_path)

        for line in lines:
            if 'eigval :' in line:
                splits = line.split()
                for split in splits[2:]:
                    try:
                        current_block.append(float(split))
                    except ValueError:
                        continue
            elif current_block:
                # End of an eigval run
                all_blocks.append(current_block)
                current_block = list()
        if current_block:
            all_blocks.append(current_block)

        freqs = [f for f in all_blocks[-1] if f != 0.0] if all_blocks else list()

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

    def parse_normal_mode_displacement(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        Parse frequencies and normal mode displacement.

        Returns: tuple[np.ndarray | None, np.ndarray | None]
            The frequencies (in cm^-1) and the normal mode displacements.
        """
        # Locate the g98.out file in the same directory as the log file
        g98_path = os.path.join(os.path.dirname(self.log_file_path), 'g98.out')
        if not os.path.isfile(g98_path):
            return None, None

        freqs, displacements = list(), list()

        with open(g98_path, 'r') as f:
            lines = f.readlines()

        i, n_modes_in_block = 0, 0
        while i < len(lines):
            line = lines[i]

            # Frequencies
            if 'Frequencies --' in line:
                freq_block = [float(x) for x in line.split()[2:]]
                freqs.extend(freq_block)
                n_modes_in_block = len(freq_block)
                i += 1

            # Atom AN ... displacement values
            elif line.strip().startswith('Atom AN'):
                i += 1
                floats_list = list()
                while i < len(lines):
                    next_line = lines[i].strip()
                    if not next_line or next_line.startswith('----'):
                        break
                    parts = next_line.split()[2:]  # skip Atom index and Atomic Number
                    if not parts:
                        i += 1
                        continue
                    if not any(is_str_float(x) for x in parts):
                        break
                    if len(parts) != 3 * n_modes_in_block:
                        i += 1
                        continue
                    floats_list.append([float(x) for x in parts])
                    i += 1

                n_atoms = len(floats_list)
                current_block = np.zeros((n_modes_in_block, n_atoms, 3), dtype=np.float64)

                for atom_idx, floats in enumerate(floats_list):
                    if len(floats) != 3 * n_modes_in_block:
                        continue
                    for mode in range(n_modes_in_block):
                        current_block[mode, atom_idx] = floats[3 * mode: 3 * mode + 3]

                displacements.append(current_block)
            else:
                i += 1

        if not freqs or not displacements:
            return None, None

        full_displacements = np.concatenate(displacements, axis=0)

        return np.array(freqs, dtype=np.float64), full_displacements

    def parse_t1(self) -> float | None:
        """
        Parse the T1 parameter from a CC calculation.

        Returns: float | None
            The T1 parameter.
        """
        # Not implemented for xTB.
        return None

    def parse_e_elect(self) -> float | None:
        """
        Parse the electronic energy from an sp job output file.

<<<<<<< HEAD
        Looks for ``:: total energy ... Eh`` (SUMMARY block) or
        ``| TOTAL ENERGY ... Eh |`` (TOTAL section).
        Avoids false matches against ``total energy gain`` (optimization deltas).

        Returns: Optional[float]
=======
        Returns: float | None
>>>>>>> c8fe67a3 (Modernize typing imports for Python 3.14+)
            The electronic energy in kJ/mol.
        """
        import re
        lines = _get_lines_from_file(self.log_file_path)
        energy = None
        # Iterate forward and keep the LAST hit (final result)
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(':: total energy') or 'TOTAL ENERGY' in line:
                m = re.search(r'(-?\d+\.\d+)\s+Eh', line)
                if m:
                    energy = float(m.group(1))
        if energy is None:
            # Fallback: 'final energy' lines (rare)
            for line in reversed(lines):
                if 'final energy' in line.lower():
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
        import re
        zpe = None
        with open(self.log_file_path, 'r') as f:
            for line in f:
                if 'zero-point vibrational energy' in line.lower() or 'zero point energy' in line.lower():
                    #          :: zero point energy           0.056690417480 Eh   ::
                    m = re.search(r'(\d+\.\d+(?:[eE][+-]?\d+)?)\s+Eh', line)
                    if m:
                        zpe = float(m.group(1))
                        break
        if zpe is not None:
            return zpe * E_h_kJmol
        return None

    def parse_1d_scan_energies(self) -> tuple[list[float] | None, list[float] | None]:
        """
        Parse the 1D torsion scan energies from an xTB scan log file.

        Returns: tuple[list[float] | None, list[float] | None]
            The electronic energy in kJ/mol and the dihedral scan angle in degrees.
        """
        scan_path = os.path.join(os.path.dirname(self.log_file_path), 'xtbscan.log')
        if not os.path.isfile(scan_path):
            logger.warning(f'xTB scan log file {scan_path} not found.')
            return None, None

        lines = _get_lines_from_file(scan_path)
        energies = []

        for line in lines:
            if 'energy:' in line.lower():
                try:
                    parts = line.lower().split()
                    idx = parts.index('energy:')
                    energy = float(parts[idx + 1])
                    energies.append(energy * 2625.49962)  # Convert Hartree to kJ/mol
                except (ValueError, IndexError):
                    continue

        if not energies:
            logger.warning(f'No energies found in xTB scan log file {scan_path}.')
            return None, None

        # Remove duplicate energies due to format (if any)
        deduped = []
        for i, e in enumerate(energies):
            if i == 0 or not math.isclose(e, energies[i - 1], abs_tol=1e-7):
                deduped.append(e)

        energies = deduped
        min_e = min(energies)
        rel_energies = [e - min_e for e in energies]

        n_points = len(rel_energies)
        if n_points == 0:
            logger.warning(f'No valid scan points found in xTB scan log file {scan_path}.')
            return None, None

        # Angles: evenly spaced from 0 deg with one angle per energy point.
        # For 44 energies, the dihedral was scanned in steps of 360/45 = 8 deg.
        # We return n_points angles (matching the energies length): [0, 8, 16, ..., 344].
        # Note: callers expecting (n_points+1) angles for n_points energies should
        # add the closing 360 deg themselves.
        angles = [i * 360.0 / (n_points + 1) for i in range(n_points)]

        return rel_energies, angles

    def parse_1d_scan_coords(self) -> list[dict[str, tuple]] | None:
        """
        Parse the 1D torsion scan coordinates from an xTB scan log file.

        Returns: list[dict[str, tuple]] | None
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
                        # Capitalize properly: e.g., 'c' → 'C', 'cl' → 'Cl'
                        symbol = element.capitalize() if len(element) == 1 else element[0].upper() + element[1:].lower()
                        x, y, z = parts[1:4]
                        xyz_str += f"{symbol} {x} {y} {z}\n"
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

    def parse_irc_traj(self) -> list[dict[str, tuple]] | None:
        """
        Parse the IRC trajectory coordinates from an ESS log file.

        Returns: list[dict[str, tuple]]
            The Cartesian coordinates for each scan point.
        """
        # Not implemented for xTB.
        return None

    def parse_scan_conformers(self) -> pd.DataFrame | None:
        """
        Parse all internal coordinates of scan conformers into a DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing internal coordinates for all conformers
        """
        # Not implemented for xTB.
        return None

    def parse_nd_scan_energies(self) -> dict | None:
        """
        Parse the ND torsion scan energies from an ESS log file.

        Returns: dict
            The "results" dictionary
        """
        # Not implemented for xTB.
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
        # Not implemented for xTB.
        return None


register_ess_adapter('xtb', XTBParser)
