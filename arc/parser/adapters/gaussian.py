"""
An adapter for parsing Gaussian log files.
"""

from abc import ABC

import numpy as np
import pandas as pd
import re
from typing import Dict, List, Optional, Tuple

from arc.common import SYMBOL_BY_NUMBER
from arc.constants import E_h_kJmol, bohr_to_angstrom
from arc.species.converter import str_to_xyz, xyz_from_data
from arc.parser.adapter import ESSAdapter
from arc.parser.factory import register_ess_adapter
from arc.parser.parser import _get_lines_from_file


class GaussianParser(ESSAdapter, ABC):
    """
    A class for parsing Gaussian log files.

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
        terminated = False
        error = None
        for line in reversed(lines):
            if 'termination' in line:
                terminated = True
                if 'l9999.exe' in line or 'link 9999' in line:
                    error = 'Unconverged'
                elif 'l101.exe' in line:
                    error = ('Coordinate section formatting error. The blank line after the coordinate section is '
                             'missing, or charge/multiplicity was not specified correctly')
                elif 'l103.exe' in line:
                    error = 'Internal coordinate error'
                elif 'l108.exe' in line:
                    error = 'Z-matrix formatting error. There are two blank lines between z-matrix and the variables, expected only one'
                elif 'l202.exe' in line:
                    error = 'Molecular orientation/point group changed during optimization'
                elif 'l301.exe' in line:
                    # This is not exactly correct - l301 can have multiple reasons
                    error = 'No data on chk file.'
                elif 'l401.exe' in line:
                    error = 'Basis set data is not on the checkpoint file.'
                elif 'l502.exe' in line:
                    error = 'Unconverged SCF.'
                elif 'l716.exe' in line:
                    error = 'Invalid z-matrix angle (outside 0-180 range)'
                elif 'l906.exe' in line:
                    error = 'MP2 failure (pseudopotential/basis set issue)'
                elif 'l913.exe' in line:
                    error = 'Maximum optimization cycles reached'
                break
        if not terminated:
            return "Job did not terminate normally"
        if error:
            return error
        return None

    def parse_geometry(self) -> Optional[Dict[str, tuple]]:
        """
        Parse the latest xyz geometry from an ESS log file by searching from the end.
        Try to find the 'Standard orientation:' first, and if not found, fall back to the 'Input orientation:'.

        Returns: Optional[Dict[str, tuple]]
            The cartesian geometry.
        """
        lines = _get_lines_from_file(self.log_file_path)
        for i in range(len(lines) - 1, -1, -1):  # Search for the latest 'Standard orientation:' from the end
            if 'Standard orientation:' in lines[i]:
                j = i + 5  # Usually 5 lines after 'Standard orientation:' is the table start
                xyz_str = ''
                while j < len(lines) and lines[j].strip() and not lines[j].startswith(' ---'):
                    splits = lines[j].split()
                    if splits and splits[0].isdigit():
                        xyz_str += f'{SYMBOL_BY_NUMBER[int(splits[1])]}  {splits[3]}  {splits[4]}  {splits[5]}\n'
                    j += 1
                if xyz_str:
                    return str_to_xyz(xyz_str)
                break

        numbers, coords, = list(), list()
        with open(self.log_file_path, 'r') as f:
            line = f.readline()
            while line != '':
                if 'Input orientation:' in line:
                    numbers, coords = list(), list()
                    for i in range(5):
                        line = f.readline()
                    while '---------------------------------------------------------------------' not in line:
                        data = line.split()
                        numbers.append(int(data[1]))
                        coords.append([float(data[3]), float(data[4]), float(data[5])])
                        line = f.readline()
                line = f.readline()
        coords = np.array(coords, float)
        numbers = np.array(numbers, int)
        if len(numbers) != 0 and len(coords) != 0:
            return xyz_from_data(coords=coords, numbers=numbers)
        return None

    def parse_frequencies(self) -> Optional['np.ndarray']:
        """
        Parse the frequencies from a freq job output file.

        Returns: Optional[np.ndarray]
            The parsed frequencies (in cm^-1), or None if not found.
        """
        frequencies = list()
        found_block = False
        with open(self.log_file_path, 'r') as f:
            for line in f:
                if 'and normal coordinates' in line:
                    # Start a new block for the last occurrence
                    frequencies = []
                    found_block = True
                elif found_block and 'Frequencies --' in line:
                    # Extract frequency values after 'Frequencies --'
                    frequencies.extend(float(frq) for frq in line.split()[2:])
        if frequencies:
            return np.array(frequencies, dtype=np.float64)
        return None

    def parse_normal_mode_displacement(self) -> Optional[Tuple['np.ndarray', 'np.ndarray']]:
        """
        Parse frequencies and normal mode displacement from Gaussian log file.

        Returns: Tuple[np.ndarray, np.ndarray]
            The frequencies (in cm^-1) and the normal mode displacements.
            Displacement array shape: (n_modes, n_atoms, 3)
        """
        freqs = list()
        displacements = list()  # Will store modes as [mode1, mode2, ...] where mode = [atom1_vec, atom2_vec, ...]
        current_block_disps = list()  # Displacements for current frequency block
        in_freq_block = False
        in_displacement_block = False
        num_modes_in_block = 0
        with open(self.log_file_path, 'r') as f:
            for line in f:
                if 'Harmonic frequencies (cm**-1)' in line:
                    in_freq_block = True
                if in_freq_block and 'Frequencies --' in line:
                    freqs.extend(float(f) for f in line.split()[2:])
                    num_modes_in_block = len(line.split()) - 2
                    current_block_disps = [[] for _ in range(num_modes_in_block)]
                if in_freq_block and ('Atom  AN      X      Y      Z' in line or
                                      'Atom AN      X      Y      Z' in line):
                    in_displacement_block = True
                    continue
                if in_displacement_block:
                    if not line.strip() or '------' in line:  # End of displacement block
                        in_displacement_block = False
                        displacements.extend(current_block_disps)
                        current_block_disps = []
                    else:
                        parts = line.split()
                        if len(parts) < 2:
                            continue
                        coords = parts[2:2 + 3 * num_modes_in_block]
                        for i in range(num_modes_in_block):
                            try:
                                # Extract XYZ components for this mode
                                vec = [float(x) for x in coords[3 * i:3 * i + 3]]
                                current_block_disps[i].append(vec)
                            except (ValueError, IndexError):
                                continue
                # End of frequency block
                if in_freq_block and ('----------' in line or not line.strip()):
                    in_freq_block = False
                    in_displacement_block = False
                    # Save any remaining displacements
                    if current_block_disps:
                        displacements.extend(current_block_disps)
                        current_block_disps = []
        freq_array = np.array(freqs, dtype=np.float64)
        # Convert displacements to 3D array: (n_modes, n_atoms, 3)
        if displacements:
            n_modes = len(displacements)
            n_atoms = len(displacements[0])
            disp_array = np.zeros((n_modes, n_atoms, 3), dtype=np.float64)
            for i, mode in enumerate(displacements):
                for j, atom_disp in enumerate(mode):
                    disp_array[i, j] = atom_disp
        else:
            disp_array = np.array([], dtype=np.float64).reshape(0, 0, 0)
        return freq_array, disp_array

    def parse_t1(self) -> Optional[float]:
        """
        Parse the T1 parameter from a CC calculation.

        Returns: Optional[float]
            The T1 parameter.
        """
        # Not implemented for Gaussian.
        return None

    def parse_e_elect(self) -> Optional[float]:
        """
        Parse the electronic energy from an sp job output file.

        Returns: Optional[float]
            The electronic energy in kJ/mol.
        """
        lines = _get_lines_from_file(self.log_file_path)
        e_elect = None
        e0_composite = None
        scaled_zpe = None

        for i, line in enumerate(lines):
            if 'SCF Done:' in line:
                value = extract_scf_done(line)
                if value is not None:
                    e_elect = value
            elif ' E2(' in line and ' E(' in line:
                value = extract_last_float(line)
                if value is not None:
                    e_elect = value
            elif 'MP2 =' in line:
                value = extract_last_float(line)
                if value is not None:
                    e_elect = value
            elif 'E(CORR)=' in line:
                value = extract_float_at_index(line, 3)
                if value is not None:
                    e_elect = value
            elif 'CCSD(T)=' in line:
                value = extract_float_at_index(line, 1)
                if value is not None:
                    e_elect = value
            elif 'CBS-QB3 (0 K)' in line:
                value = extract_float_at_index(line, 3)
                if value is not None:
                    e0_composite = value
            elif 'E(CBS-QB3)=' in line:
                value = extract_float_at_index(line, 1)
                if value is not None:
                    e_elect = value
            elif 'CBS-4 (0 K)=' in line:
                value = extract_float_at_index(line, 3)
                if value is not None:
                    e0_composite = value
            elif 'G3(0 K)' in line:
                value = extract_float_at_index(line, 2)
                if value is not None:
                    e0_composite = value
            elif 'G3 Energy=' in line:
                value = extract_float_at_index(line, 2)
                if value is not None:
                    e_elect = value
            elif 'G4(0 K)' in line:
                value = extract_float_at_index(line, 2)
                if value is not None:
                    e0_composite = value
            elif 'G4 Energy=' in line:
                value = extract_float_at_index(line, 2)
                if value is not None:
                    e_elect = value
            elif 'G4MP2(0 K)' in line:
                value = extract_float_at_index(line, 2)
                if value is not None:
                    e0_composite = value
            elif 'G4MP2 Energy=' in line:
                value = extract_float_at_index(line, 2)
                if value is not None:
                    e_elect = value
            elif 'E(ZPE)' in line:
                value = extract_float_at_index(line, 1)
                if value is not None:
                    scaled_zpe = value
            elif '\\ZeroPoint=' in line:
                value = extract_zero_point(line, lines, i)
                if value is not None:
                    scaled_zpe = value
            elif 'HF=' in line and e_elect is None:
                value = extract_hf(line, lines, i)
                if value is not None:
                    e_elect = value

        if e0_composite is not None and scaled_zpe is not None:
            return (e0_composite - scaled_zpe) * E_h_kJmol
        elif e0_composite is not None:
            return e0_composite * E_h_kJmol
        elif e_elect is not None:
            return e_elect * E_h_kJmol
        return None

    def parse_zpe_correction(self) -> Optional[float]:
        """
        Determine the calculated ZPE correction from a frequency output file.

        Returns: Optional[float]
            The calculated zero point energy in kJ/mol.
        """
        zpe_hartree = None
        with open(self.log_file_path, 'r') as f:
            for line in f:
                # Handle "Zero-point correction=" lines
                if 'Zero-point correction=' in line:
                    parts = line.split()
                    if len(parts) > 2:
                        try:
                            zpe_hartree = float(parts[2])
                        except ValueError:
                            continue
                # Handle "\ZeroPoint=" in archive entries
                elif '\\ZeroPoint=' in line:
                    # Combine with next line if needed
                    full_line = line.strip()
                    next_line = f.readline().strip() if '\\' not in line[-2:] else ''
                    full_line += next_line
                    # Extract value between \ZeroPoint= and next \
                    start = full_line.find('\\ZeroPoint=') + 11
                    end = full_line.find('\\', start)
                    if start > 10 and end > start:
                        val_str = full_line[start:end]
                        try:
                            zpe_hartree = float(val_str)
                        except ValueError:
                            continue
        if zpe_hartree is not None:
            return zpe_hartree * E_h_kJmol
        return None

    def parse_1d_scan_energies(self) -> Tuple[Optional[List[float]], Optional[List[float]]]:
        """
        Parse the 1D torsion scan energies from a Gaussian log file.

        Returns: Tuple[Optional[List[float]], Optional[List[float]]]
            The electronic energy in kJ/mol and the dihedral scan angle in degrees.
        """
        energies, angles = list(), list()
        rigid_scan = False
        scan_resolution, current_energy, current_angle = None, None, None
        lines = _get_lines_from_file(self.log_file_path)

        # Detect scan type and resolution
        for line in lines:
            if '# scan' in line:
                rigid_scan = True
                # Extract scan resolution (e.g., "S 10 5.0")
                parts = line.split()
                for i, part in enumerate(parts):
                    if part.lower() == 's' and i + 2 < len(parts):
                        try:
                            scan_resolution = float(parts[i + 2])
                        except ValueError:
                            pass
                break  # Only need to check once

        # Parse energies and angles
        for i, line in enumerate(lines):
            # Capture SCF energies
            if 'SCF Done:' in line:
                energy = _extract_scf_energy(line)
                if energy is not None:
                    current_energy = energy
                    if rigid_scan:
                        energies.append(current_energy)

            # Capture optimized energies for relaxed scans
            if not rigid_scan and 'Optimization completed' in line and current_energy is not None:
                energies.append(current_energy)

            # Capture dihedral angles for relaxed scans
            if not rigid_scan and '! ' in line and 'D(' in line and 'Scan' not in line:
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        current_angle = float(parts[3])
                        angles.append(current_angle)
                    except ValueError:
                        pass

        # Handle rigid scans
        if rigid_scan and scan_resolution is not None and energies:
            angles = [i * scan_resolution for i in range(len(energies))]

        # Handle relaxed scans angle normalization
        if not rigid_scan and angles:
            # Normalize to start at zero
            start_angle = angles[0]
            angles = [(a - start_angle) % 360 for a in angles]
            # Adjust last angle if needed
            if len(angles) > 1 and angles[-1] < 2 * (angles[1] - angles[0]):
                angles[-1] += 360

        # Return None if no valid data
        if not energies or not angles or len(energies) != len(angles):
            return None, None

        # Normalize energies to minimum and convert to kJ/mol
        min_energy = min(energies)
        rel_energies = [(e - min_energy) * E_h_kJmol for e in energies]

        return rel_energies, angles

    def parse_1d_scan_coords(self) -> Optional[List[Dict[str, tuple]]]:
        """
        Parse the 1D torsion scan coordinates from an ESS log file.

        Returns: List[Dict[str, tuple]]
            The Cartesian coordinates for each scan point.
        """
        lines = _get_lines_from_file(self.log_file_path)
        traj = list()
        i = 0
        while i < len(lines):
            if "Optimization completed" in lines[i]:
                # Find next coordinate block
                coord_start = None
                for j in range(i, min(i + 200, len(lines))):
                    if "Standard orientation:" in lines[j] or "Input orientation:" in lines[j]:
                        coord_start = j
                        break
                if coord_start is None:
                    i += 1
                    continue
                # Skip header lines (5 lines after orientation marker)
                k = coord_start + 5
                xyz_str = ""
                valid_block = True
                # Parse coordinate block
                while k < len(lines) and "---" not in lines[k]:
                    if "Error termination" in lines[k]:
                        valid_block = False
                        break
                    parts = lines[k].split()
                    if len(parts) >= 6 and parts[0].isdigit():
                        try:
                            atomic_num = int(parts[1])
                            element = SYMBOL_BY_NUMBER.get(atomic_num, 'X')
                            x, y, z = parts[3], parts[4], parts[5]
                            xyz_str += f"{element} {x} {y} {z}\n"
                        except (ValueError, IndexError):
                            valid_block = False
                            break
                    k += 1
                if valid_block and xyz_str:
                    traj.append(str_to_xyz(xyz_str))
                i = k  # Jump to end of coordinate block
            else:
                i += 1
        return traj

    def parse_scan_conformers(self) -> Optional['pd.DataFrame']:
        """
        Parse all internal coordinates of scan conformers into a DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing internal coordinates for all conformers
        """

        # Helper function to parse IC blocks
        def parse_ic_block(lines: List[str]) -> List[Dict]:
            ics = []
            for line in lines:
                if line.strip().startswith('!'):
                    parts = line.split()
                    if len(parts) >= 4:
                        label = parts[1]
                        value = float(parts[3])
                        ics.append({'label': label, 'value': value})
            return ics

        # Get all lines from log file
        lines = _get_lines_from_file(self.log_file_path)

        # Find initial parameters block
        initial_params = []
        in_initial_block = False
        for line in lines:
            if 'Initial Parameters' in line:
                in_initial_block = True
                continue
            if in_initial_block and '------------' in line:
                break
            if in_initial_block and line.strip().startswith('!'):
                initial_params.append(line)

        # Parse initial IC info
        ic_info = []
        for line in initial_params:
            parts = line.split()
            if len(parts) < 4:
                continue

            label = parts[1]
            ic_type = label[0]  # R, A, D
            atoms = [int(x) for x in re.findall(r'\d+', parts[2])]

            # Determine redundancy (simplified logic)
            redundant = False
            if (ic_type == 'A' and len(atoms) > 3) or \
                    (ic_type == 'R' and len(atoms) > 2) or \
                    (ic_type == 'D' and len(atoms) > 4):
                redundant = True

            # Determine if this is a scan coordinate (simplified)
            scan = False
            if ic_type == 'D' and len(atoms) == 4:
                scan = True  # Assume all dihedrals are scan coordinates

            ic_info.append({
                'label': label,
                'type': ic_type,
                'atoms': atoms,
                'redundant': redundant,
                'scan': scan
            })

        if not ic_info:
            return None

        # Create initial DataFrame
        ic_info_df = pd.DataFrame(ic_info).set_index('label')

        # Find all optimized parameters blocks
        optimized_blocks = []
        current_block = []
        in_block = False

        for line in lines:
            if 'Optimized Parameters' in line:
                in_block = True
                current_block = []
                continue
            if in_block and '------------' in line:
                in_block = False
                if current_block:
                    optimized_blocks.append(current_block)
                continue
            if in_block:
                current_block.append(line)

        # Parse IC values for each conformer
        conformers = []
        for i, block in enumerate(optimized_blocks):
            ics = parse_ic_block(block)
            if not ics:
                continue
            df = pd.DataFrame(ics).set_index('label')
            df = df.rename(columns={'value': f'conformer_{i}'})
            conformers.append(df)

        if not conformers:
            return None

        # Combine all conformers
        scan_conformers = ic_info_df
        for df in conformers:
            scan_conformers = scan_conformers.join(df, how='left')

        # Remove redundant coordinates
        scan_conformers = scan_conformers[~scan_conformers['redundant']]
        return scan_conformers

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
        # internal variables:
        # - scan_d_dict (dict): keys are scanning dihedral names (e.g., 'D2', or 'D4'), values are the corresponding
        #                       torsion indices tuples (e.g., (4, 1, 2, 5), or (4, 1, 3, 6)).
        # - dihedrals_dict (dict): keys are torsion tuples (e.g., (4, 1, 2, 5), or (4, 1, 3, 6)),
        #                          values are lists of dihedral angles in degrees corresponding to the torsion
        #                          (e.g., [-159.99700, -149.99690, -139.99694, -129.99691, -119.99693]).
        # - torsions (list): entries are torsion indices that are scanned, e.g.: [(4, 1, 2, 5), (4, 1, 3, 6)]
        path = self.log_file_path
        results = {'directed_scan_type': 'ess_gaussian',
                   'scans': [],
                   'directed_scan': {},
                   }
        with open(path, 'r', buffering=8192) as f:
            line = f.readline()
            symbols, torsions, shape, resolution, original_dihedrals = list(), list(), list(), list(), list()
            scan_d_dict = dict()
            min_e = None
            while line:
                line = f.readline()
                if 'The following ModRedundant input section has been read:' in line:
                    # ' The following ModRedundant input section has been read:'
                    # ' D       4       1       2       5 S  36 10.000'
                    # ' D       4       1       3       6 S  36 10.000'
                    line = f.readline()
                    while True:
                        splits = line.split()
                        if len(splits) == 8:
                            torsions.append(tuple([int(index) for index in splits[1:5]]))
                            shape.append(int(splits[6]) + 1)  # the last point is repeated
                            resolution.append(float(splits[7]))
                        else:
                            break
                        line = f.readline()
                    results['scans'] = torsions
                    if 'Symbolic Z-matrix:' in line:
                        #  ---------------------
                        #  HIR calculation by AI
                        #  ---------------------
                        #  Symbolic Z-matrix:
                        #  Charge =  0 Multiplicity = 1
                        #  c
                        #  o                    1    oc2
                        #  o                    1    oc3      2    oco3
                        #  o                    1    oc4      2    oco4     3    dih4     0
                        #  h                    2    ho5      1    hoc5     3    dih5     0
                        #  h                    3    ho6      1    hoc6     4    dih6     0
                        #        Variables:
                        #   oc2                   1.36119
                        #   oc3                   1.36119
                        #   oco3                114.896
                        #   oc4                   1.18581
                        #   oco4                122.552
                        #   dih4                180.
                        #   ho5                   0.9637
                        #   hoc5                111.746
                        #   dih5                 20.003
                        #   ho6                   0.9637
                        #   hoc6                111.746
                        #   dih6               -160.
                        for i in range(2):
                            f.readline()
                        while 'Variables' not in line:
                            symbols.append(line.split()[0].upper())
                            line = f.readline()
                if 'Initial Parameters' in line:
                    #                            ----------------------------
                    #                            !    Initial Parameters    !
                    #                            ! (Angstroms and Degrees)  !
                    #  --------------------------                            --------------------------
                    #  ! Name  Definition              Value          Derivative Info.                !
                    #  --------------------------------------------------------------------------------
                    #  ! R1    R(1,2)                  1.3612         calculate D2E/DX2 analytically  !
                    #  ! R2    R(1,3)                  1.3612         calculate D2E/DX2 analytically  !
                    #  ! R3    R(1,4)                  1.1858         calculate D2E/DX2 analytically  !
                    #  ! R4    R(2,5)                  0.9637         calculate D2E/DX2 analytically  !
                    #  ! R5    R(3,6)                  0.9637         calculate D2E/DX2 analytically  !
                    #  ! A1    A(2,1,3)              114.896          calculate D2E/DX2 analytically  !
                    #  ! A2    A(2,1,4)              122.552          calculate D2E/DX2 analytically  !
                    #  ! A3    A(3,1,4)              122.552          calculate D2E/DX2 analytically  !
                    #  ! A4    A(1,2,5)              111.746          calculate D2E/DX2 analytically  !
                    #  ! A5    A(1,3,6)              111.746          calculate D2E/DX2 analytically  !
                    #  ! D1    D(3,1,2,5)             20.003          calculate D2E/DX2 analytically  !
                    #  ! D2    D(4,1,2,5)           -159.997          Scan                            !
                    #  ! D3    D(2,1,3,6)             20.0            calculate D2E/DX2 analytically  !
                    #  ! D4    D(4,1,3,6)           -160.0            Scan                            !
                    #  --------------------------------------------------------------------------------
                    for i in range(5):
                        line = f.readline()
                    while '--------------------------' not in line:
                        splits = line.split()
                        if 'Scan' in line:
                            scan_d_dict[splits[1]] = \
                                tuple([int(index) for index in splits[2][2:].replace(')', '').split(',')])
                            original_dihedrals.append(float(splits[3]))
                        line = f.readline()
                elif 'Summary of Optimized Potential Surface Scan' in line:
                    # ' Summary of Optimized Potential Surface Scan (add -264.0 to energies):'
                    base_e = float(
                        line.split('(add ')[1].split()[0]) if '(add' in line and 'to energies):' in line else 0.0
                    energies, dihedrals_dict = list(), dict()
                    dihedral_num = 0
                    while 'Grad' not in line and 'Largest change from initial coordinates' not in line:
                        line = f.readline()
                        splits = line.split()
                        numbers = re.findall(r'-?\d+\.\d+', line)
                        if 'Eigenvalues --' in line:
                            # convert Hartree energy to kJ/mol
                            energies = [(base_e + float(e)) * 4.3597447222071e-18 * 6.02214179e23 * 1e-3
                                        for e in numbers]
                            min_es = min(energies)
                            min_e = min_es if min_e is None else min(min_e, min_es)
                            dihedral_num = 0
                        if splits and splits[0] in scan_d_dict and scan_d_dict[splits[0]] not in dihedrals_dict:
                            # parse the dihedral information
                            # '           D1          20.00308  30.00361  40.05829  50.36777  61.07341'
                            # '           D2        -159.99700-149.99690-139.99694-129.99691-119.99693'
                            # '           D3          19.99992  19.99959  19.94509  19.63805  18.93967'
                            # '           D4        -160.00000-159.99990-159.99994-159.99991-159.99993'
                            dihedrals = [float(dihedral) for dihedral in line.replace('-', ' -').split()[1:]]
                            for i in range(len(dihedrals)):
                                if 0 > dihedrals[i] >= -0.0049999:
                                    dihedrals[i] = 0.0
                            dihedrals_dict[scan_d_dict[splits[0]]] = dihedrals
                            dihedral_num += 1
                        if len(dihedrals_dict) == len(scan_d_dict):
                            # we have all the data for this block, pass to ``results`` and initialize ``dihedrals_dict``
                            for i, energy in enumerate(energies):
                                dihedral_list = [dihedrals_dict[torsion][i] for torsion in torsions]  # ordered
                                key = tuple(f'{dihedral:.2f}' for dihedral in dihedral_list)
                                # overwrite previous values for a close key if found:
                                # key = get_close_tuple(key, results['directed_scan'].keys()) or key
                                results['directed_scan'][key] = {'energy': energy}
                            dihedrals_dict = dict()  # keys are torsion tuples, values are dihedral angles
                    break
        for key in results['directed_scan'].keys():
            results['directed_scan'][key] = {'energy': results['directed_scan'][key]['energy'] - min_e}
        return results

    def parse_dipole_moment(self) -> Optional[float]:
        """
        Parse the dipole moment in Debye from an opt job output file.

        Returns: Optional[float]
            The dipole moment in Debye.
        """
        lines = _get_lines_from_file(self.log_file_path)
        dipole_moment = None
        # Gaussian outputs dipole moments in two-line format:
        # Line 1: "Dipole moment (field-independent basis, Debye):"
        # Line 2: "X= [value] Y= [value] Z= [value] Tot= [value]"
        for i, line in enumerate(lines[:-1]):  # Skip last line since we need i+1
            line_lower = line.lower()
            if 'dipole moment' in line_lower and 'debye' in line_lower:
                next_line = lines[i + 1]
                if next_line.strip():
                    # Extract total dipole moment (last value in the line)
                    parts = next_line.split()
                    if parts:
                        try:
                            # Last value is total dipole moment
                            dipole_moment = float(parts[-1])
                        except (ValueError, IndexError):
                            # Try alternative pattern: " Tot= 1.2345"
                            if 'tot=' in next_line.lower():
                                tot_part = next_line.lower().split('tot=')[-1].split()[0]
                                try:
                                    dipole_moment = float(tot_part)
                                except ValueError:
                                    continue
        return dipole_moment

    def parse_polarizability(self) -> Optional[float]:
        """
        Parse the polarizability from a freq job output file, returns the value in Angstrom^3.

        Returns: Optional[float]
            The polarizability in Angstrom^3.
        """
        lines = _get_lines_from_file(self.log_file_path)
        polarizability = None
        for line in lines:
            if 'Isotropic polarizability for W' in line:
                # Example: Isotropic polarizability for W=    0.000000       11.49 Bohr**3.
                try:
                    value_bohr3 = float(line.split()[-2])
                    polarizability = value_bohr3 * bohr_to_angstrom ** 3
                except (ValueError, IndexError):
                    continue
        return polarizability


def is_float(s: str) -> bool:
    """Check if a string can be converted to float (handles scientific notation with D/E)."""
    try:
        float(s.replace('D', 'E'))
        return True
    except (ValueError, AttributeError):
        return False


def extract_last_float(line: str) -> Optional[float]:
    """Extract the last float (possibly in D notation) from a line."""
    parts = line.split()
    if parts and is_float(parts[-1]):
        return float(parts[-1].replace('D', 'E'))
    return None


def extract_float_at_index(line: str, idx: int) -> Optional[float]:
    """Extract float at a given index from a split line."""
    parts = line.split()
    if len(parts) > idx and is_float(parts[idx]):
        return float(parts[idx].replace('D', 'E'))
    return None


def extract_scf_done(line: str) -> Optional[float]:
    """Extract SCF Done energy from a line."""
    match = re.search(r'E\(.+\)\s+=\s+([-]?\d+\.\d+)', line)
    if match and is_float(match.group(1)):
        return float(match.group(1))
    return None


def extract_zero_point(line: str, lines: list, idx: int) -> Optional[float]:
    """Extract ZeroPoint energy from a line and possibly the next line."""
    next_line = lines[idx + 1].strip() if idx + 1 < len(lines) else ''
    joined = line.strip() + next_line
    start = joined.find('\\ZeroPoint=') + 11
    end = joined.find('\\', start)
    if start > 10 and end > start:
        val = joined[start:end]
        if is_float(val):
            return float(val)
    return None


def extract_hf(line: str, lines: list, idx: int) -> Optional[float]:
    """Extract HF energy from a line and possibly the next line."""
    next_line = lines[idx + 1].strip() if idx + 1 < len(lines) else ''
    joined = line.strip() + next_line
    start = joined.find('HF=') + 3
    end = joined.find('\\', start)
    if start > 2 and end > start:
        val = joined[start:end]
        if is_float(val):
            return float(val)
    return None


def _extract_scf_energy(line: str) -> Optional[float]:
    """Extract SCF energy from Gaussian output line."""
    match = re.search(r'E\([^)]+\)\s+=\s+([-]?\d+\.\d+)', line)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


register_ess_adapter('gaussian', GaussianParser)
