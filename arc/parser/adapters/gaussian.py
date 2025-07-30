"""
An adapter for parsing Gaussian log files.
"""

from abc import ABC

import numpy as np
import pandas as pd
import re
from typing import Dict, List, Match, Optional, Tuple, Union

from arc.common import SYMBOL_BY_NUMBER, is_same_pivot
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
        Parse the xyz geometry from an ESS log file.
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

    def parse_frequencies(self) -> Optional[np.ndarray]:
        """
        Parse the frequencies from a freq job output file.

        Returns: Optional[np.ndarray]
            The parsed frequencies (in cm^-1).
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

    def parse_normal_mode_displacement(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Parse frequencies and normal mode displacement.

        Returns: Tuple[Optional['np.ndarray'], Optional['np.ndarray']]
            The frequencies (in cm^-1) and the normal mode displacements.
        """
        freqs, displacements = list(), list()

        with open(self.log_file_path, 'r') as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i]

            # Start of a normal mode block
            if 'Frequencies --' in line:
                current_freqs = [float(val) for val in line.split()[2:]]
                freqs.extend(current_freqs)
                n_modes_in_block = len(current_freqs)
                mode_disps = [[] for _ in range(n_modes_in_block)]  # list of atoms for each mode

                # Skip to displacement block header
                while i < len(lines) and 'Atom  AN' not in lines[i]:
                    i += 1
                i += 1  # Skip header line

                # Read displacement rows
                while i < len(lines) and lines[i].strip() and not lines[i].startswith(' Frequencies --'):
                    parts = lines[i].split()
                    if len(parts) >= 2 + 3 * n_modes_in_block:
                        coords = parts[2:]
                        for j in range(n_modes_in_block):
                            vec = [float(coords[3*j]), float(coords[3*j+1]), float(coords[3*j+2])]
                            mode_disps[j].append(vec)
                    i += 1

                displacements.extend(mode_disps)
            else:
                i += 1

        if not freqs or not displacements:
            return None, None

        freq_array = np.array(freqs, dtype=np.float64)
        disp_array = np.array(displacements, dtype=np.float64)  # (n_modes, n_atoms, 3)

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
        Determine the calculated ZPE correction (E0 - e_elect) from a frequency output file.

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
        Parse the 1D torsion scan energies from an ESS log file.

        Returns: Tuple[Optional[List[float]], Optional[List[float]]]
            The electronic energy in kJ/mol and the dihedral scan angle in degrees.
        """
        opt_freq = False
        rigid_scan = False
        energy = None
        vlist = []
        non_optimized = []
        angle = []

        scan_pivot_atoms = self.load_scan_pivot_atoms()
        internal_coord = f"D({','.join(str(i) for i in scan_pivot_atoms)})"

        with open(self.log_file_path, 'r') as f:
            for line in f:
                if ' Freq' in line and ' Geom=' in line:
                    opt_freq = True
                if '# scan' in line.lower():
                    rigid_scan = True
                if 'SCF Done:' in line:
                    try:
                        energy = float(line.split()[4])
                    except (IndexError, ValueError):
                        energy = None
                    if rigid_scan and energy is not None:
                        vlist.append(energy)
                if 'Optimization completed' in line and energy is not None:
                    vlist.append(energy)
                if 'Optimization stopped' in line and energy is not None:
                    non_optimized.append(len(vlist))
                    vlist.append(energy)
                if internal_coord in line and 'Scan' not in line:
                    try:
                        angle.append(float(line.strip().split()[3]))
                    except (IndexError, ValueError):
                        continue

        if not vlist:
            return None, None

        if rigid_scan:
            try:
                scan_angle_resolution_deg = self.load_scan_angle()
            except AttributeError:
                return None, None
            angle = [i * scan_angle_resolution_deg for i in range(len(vlist))]
        else:
            angle = np.array(angle, float)
            if len(angle) != len(vlist):
                return None, None
            angle -= angle[0]
            angle[angle < 0] += 360.0
            if len(angle) > 1 and angle[-1] < 2 * (angle[1] - angle[0]):
                angle[-1] += 360.0
            angle = angle.tolist()

        vlist = np.array(vlist, float)
        vlist -= np.min(vlist)
        vlist *= E_h_kJmol

        if opt_freq:
            vlist = vlist[:-1]
            angle = angle[:-1]

        if non_optimized:
            vlist = np.delete(vlist, non_optimized)
            angle = np.delete(angle, non_optimized)

        return vlist.tolist(), angle

    def parse_1d_scan_coords(self) -> Optional[List[Dict[str, tuple]]]:
        """
        Parse the 1D torsion scan coordinates from an ESS log file.

        Returns: List[Dict[str, tuple]]
            The Cartesian coordinates for each scan point.
        """

        lines = _get_lines_from_file(self.log_file_path)
        traj = []
        energy = None
        i = 0

        while i < len(lines):
            line = lines[i]

            if 'SCF Done:' in line:
                try:
                    energy = float(line.split()[4])
                except (IndexError, ValueError):
                    pass

            if 'Optimization completed.' in line:
                # Find the last geometry block before this point
                back = i
                orientation_index = None
                while back > 0:
                    if any(kw in lines[back].lower() for kw in
                           ['standard orientation', 'input orientation', 'z-matrix orientation']):
                        orientation_index = back
                        break
                    back -= 1

                if orientation_index is not None and orientation_index + 5 < len(lines):
                    j = orientation_index + 5  # skip orientation header
                    coords, atomic_nums = [], []
                    while j < len(lines):
                        if not lines[j].strip() or '---' in lines[j]:
                            break
                        parts = lines[j].split()
                        if len(parts) >= 6 and parts[0].isdigit():
                            try:
                                atomic_num = int(parts[1])
                                x, y, z = map(float, parts[3:6])
                                coords.append([x, y, z])
                                atomic_nums.append(atomic_num)
                            except ValueError:
                                break
                        j += 1

                    if coords and atomic_nums:
                        symbols = [SYMBOL_BY_NUMBER[num] for num in atomic_nums]
                        traj.append(xyz_from_data(coords=np.array(coords), symbols=symbols))

            i += 1

        return traj if traj else None

    def parse_irc_traj(self) -> Optional[List[Dict[str, tuple]]]:
        """
        Parse the IRC trajectory coordinates from an ESS log file.

        Returns: List[Dict[str, tuple]]
            The Cartesian coordinates for each scan point.
        """
        lines = _get_lines_from_file(self.log_file_path)
        traj = list()
        i = 0
        while i < len(lines):
            line = lines[i]
            if 'Point Number:' in line and 'Path Number:' in line:
                dashed_line_counter = 0
                while dashed_line_counter < 2:
                    if i >= len(lines):
                        break
                    if '----' in lines[i]:
                        dashed_line_counter += 1
                    i += 1
                coords, numbers = list(), list()
                while '----' not in lines[i]:
                    parts = lines[i].split()
                    if len(parts) >= 5:
                        atomic_num = int(parts[1])
                        x, y, z = map(float, parts[2:5])
                        coords.append([x, y, z])
                        numbers.append(atomic_num)
                    i += 1
                    if i >= len(lines):
                        break
                if coords and numbers:
                    traj.append(xyz_from_data(coords=np.array(coords), numbers=numbers))
                continue
            i += 1
        return traj if traj else None

    def parse_scan_conformers(self) -> Optional[pd.DataFrame]:
        """
        Parse all internal coordinates of scan conformers into a DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing internal coordinates for all conformers
        """
        scan_args = parse_scan_args(self.log_file_path)
        if not scan_args or 'step' not in scan_args:
            return None

        ic_info_df = parse_ic_info(self.log_file_path)
        if ic_info_df is None or ic_info_df.empty:
            return None

        ic_blocks = parse_str_blocks(
            file_path=self.log_file_path,
            head_pat='Optimized Parameters',
            tail_pat='-----------',
            regex=False,
            tail_count=3,
            block_count=scan_args['step'] + 1,
        )
        if not ic_blocks:
            return None

        conformer_dfs = []
        for i, block in enumerate(ic_blocks):
            lines = block[5:-1]  # skip headers and footers
            if not lines:
                continue
            try:
                df = parse_ic_values(lines)
            except (ValueError, IndexError):
                continue
            df.rename(columns={'value': f'conformer_{i}'}, inplace=True)
            conformer_dfs.append(df)

        if not conformer_dfs:
            return None

        result = pd.concat([ic_info_df] + conformer_dfs, axis=1)
        if 'redundant' in result.columns:
            result = result[~result['redundant']]

        return result if not result.empty else None

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




    def _load_scan_specs(self, letter_spec, get_after_letter_spec=False):
        """
        This method reads the ouptput file for optional parameters
        sent to gaussian, and returns the list of optional parameters
        as a list of tuples.

        `letter_spec` is a character used to identify whether a specification
        defines pivot atoms ('S'), frozen atoms ('F') or other attributes.

        `get_after_letter_spec` is a boolean that, if True, will return the
        parameters after letter_spec is found. If not specified or False, it will
        return the preceeding letters, which are typically the atom numbers.

        More information about the syntax can be found http://gaussian.com/opt/
        """
        output = []
        reached_input_spec_section = False
        with open(self.log_file_path, 'r') as f:
            line = f.readline()
            while line != '':
                if reached_input_spec_section:
                    terms = line.split()
                    if len(terms) == 0:
                        # finished reading specs
                        break
                    if terms[0] == 'D':
                        action_index = 5  # dihedral angle with four terms
                    elif terms[0] == 'A':
                        action_index = 4  # valance angle with three terms
                    elif terms[0] == 'B':
                        action_index = 3  # bond length with 2 terms
                    elif terms[0] == 'L':
                        # Can be either L 1 2 3 B or L 1 2 3 -1 B
                        # It defines a linear bend which is helpful in calculating
                        # molecules with ~180 degree bond angles. As no other module
                        # now depends on this information, simply skipping this line.
                        line = f.readline()
                        continue
                    else:
                        raise ValueError(f'This file has an unsupported option, Unable to read scan specs for line: {line}')
                    if len(terms) > action_index:
                        # specified type explicitly
                        if terms[action_index] == letter_spec:
                            if get_after_letter_spec:
                                output.append(terms[action_index+1:])
                            else:
                                output.append(terms[1:action_index])
                    else:
                        # no specific specification, assume freezing
                        if letter_spec == 'F':
                            output.append(terms[1:action_index])
                if " The following ModRedundant input section has been read:" in line:
                    reached_input_spec_section = True
                line = f.readline()
        return output

    def load_scan_pivot_atoms(self):
        """
        Extract the atom numbers which the rotor scan pivots around
        Return a list of atom numbers starting with the first atom as 1
        """
        output = self._load_scan_specs('S')
        return output[0] if len(output) > 0 else []



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


def parse_str_blocks(file_path: str,
                     head_pat: Union[Match, str],
                     tail_pat: Union[Match, str],
                     regex: bool = True,
                     tail_count: int = 1,
                     block_count: int = 1,
                     ) -> List[str]:
    """
    Return a list of blocks defined by the head pattern and the tail pattern.

    Args:
        file_path (str): The path to the readable file.
        head_pat (str/regex): Str pattern or regular expression of the head of the block.
        tail_pat (str/regex): Str pattern or regular expresion of the tail of the block.
        regex (bool, optional): Use regex (True) or str pattern (False) to search.
        tail_count (int, optional): The number of times that the tail repeats.
        block_count (int, optional): The max number of blocks to search. -1 for any number.

    Raises:
        InputError: If the file could not be found.

    Returns: List[str]
        List of str blocks.
    """
    with open(file_path, 'r') as f:
        blks = []
        # Different search mode
        if regex:
            def search(x, y):
                return re.search(x, y)
        else:
            def search(x, y):
                return x in y
        # 'search' for the head or 'read' until the tail
        mode = 'search'
        line = f.readline()
        while line != '':
            if mode == 'search':
                # Stop searching if enough blocks were found.
                if (len(blks)) == block_count:
                    break
                # Check if matching the head pattern.
                else:
                    match = search(head_pat, line)
                    # Switch to 'read' mode.
                    if match:
                        tail_repeat = 0
                        mode = 'read'
                        blks.append([])
                        blks[-1].append(line)
            elif mode == 'read':
                blks[-1].append(line)
                match = search(tail_pat, line)
                if match:
                    tail_repeat += 1
                    # If there are enough tail patterns, switch to 'search' mode.
                    if tail_repeat == tail_count:
                        mode = 'search'
            line = f.readline()
        # Remove the last incomplete search
        if len(blks) > 0 and (tail_repeat != tail_count):
            blks.pop()
        return blks


def parse_scan_args(file_path: str) -> dict:
    """
    Get the scan arguments, including which internal coordinates (IC) are being scanned, which are frozen,
    what is the step size and the number of atoms, etc.

    Args:
        file_path (str): The path to a readable output file.

    Raises:
        NotImplementedError: If files other than Gaussian log is input

    Returns: dict
        A dictionary that contains the scan arguments as well as step number, step size, number of atom::

              {'scan': <list, atom indexes of the torsion to be scanned>,
               'freeze': <list, list of internal coordinates identified by atom indexes>,
               'step': <int, number of steps to scan>,
               'step_size': <float, the size of each step>,
               'n_atom': <int, the number of atoms of the molecule>,
               }
    """
    scan_args = {'scan': None, 'freeze': [],
                 'step': 0, 'step_size': 0, 'n_atom': 0}
    try:
        # g09, g16
        scan_blk = parse_str_blocks(file_path, 'The following ModRedundant input section has been read:',
                                    'Isotopes and Nuclear Properties', regex=False)[0][1:-1]
    except IndexError:  # Cannot find any block
        # g03
        scan_blk_1 = parse_str_blocks(file_path, 'The following ModRedundant input section has been read:',
                                      'GradGradGradGrad', regex=False)[0][1:-2]
        scan_blk_2 = parse_str_blocks(file_path, 'NAtoms=',
                                      'One-electron integrals computed', regex=False)[0][:1]
        scan_blk = scan_blk_1 + scan_blk_2
    scan_pat = r'[DBA]?(\s+\d+){2,4}\s+S\s+\d+[\s\d.]+'
    frz_pat = r'[DBA]?(\s+\d+){2,4}\s+F'
    value_pat = r'[\d.]+'
    for line in scan_blk:
        if re.search(scan_pat, line.strip()):
            values = re.findall(value_pat, line)
            scan_len = len(values) - 2  # atom indexes + step + stepsize
            scan_args['scan'] = [int(values[i]) for i in range(scan_len)]
            scan_args['step'] = int(values[-2])
            scan_args['step_size'] = float(values[-1])
        if re.search(frz_pat, line.strip()):
            values = re.findall(value_pat, line)
            scan_args['freeze'].append([int(values[i]) for i in range(len(values))])
        if 'NAtoms' in line:
            scan_args['n_atom'] = int(line.split()[1])
    return scan_args


def parse_ic_info(file_path: str) -> pd.DataFrame:
    """
    Get the information of internal coordinates (ic) of an intermediate scan conformer.

    Args:
        file_path (str): The path to a readable output file.

    Raises:
        NotImplementedError: If files other than Gaussian log is input

    Returns: pd.DataFrame
        A DataFrame containing the information of the internal coordinates
    """
    ic_dict = {item: []
               for item in ['label', 'type', 'atoms', 'redundant', 'scan']}
    scan_args = parse_scan_args(file_path)
    max_atom_ind = scan_args['n_atom']
    ic_info_block = parse_str_blocks(file_path, 'Initial Parameters', '-----------', regex=False,
                                     tail_count=3)[0][5:-1]
    for line in ic_info_block:
        # Line example with split() indices:
        # 0 1     2                        3              4         5       6            7
        # ! R1    R(1, 2)                  1.3581         calculate D2E/DX2 analytically !
        terms = line.split()
        ic_dict['label'].append(terms[1])
        ic_dict['type'].append(terms[1][0])  # 'R: bond, A: angle, D: dihedral
        atom_inds = re.split(r'[(),]', terms[2])[1:-1]
        ic_dict['atoms'].append([int(atom_ind) for atom_ind in atom_inds])

        # Identify redundant, cases like 5 atom angles or redundant atoms
        if (ic_dict['type'][-1] == 'A' and len(atom_inds) > 3) \
                or (ic_dict['type'][-1] == 'R' and len(atom_inds) > 2) \
                or (ic_dict['type'][-1] == 'D' and len(atom_inds) > 4):
            ic_dict['redundant'].append(True)
        else:
            # Sometimes, redundant atoms with weird indices are added.
            # Reason unclear. Maybe to better define the molecule, or to
            # solve equations more easily.
            weird_indices = [index for index in ic_dict['atoms'][-1]
                             if index <= 0 or index > max_atom_ind]
            if weird_indices:
                ic_dict['redundant'].append(True)
            else:
                ic_dict['redundant'].append(False)

        # Identify ics being scanned
        if len(scan_args['scan']) == len(atom_inds) == 4 \
                and is_same_pivot(scan_args['scan'], ic_dict['atoms'][-1]):
            ic_dict['scan'].append(True)
        elif len(scan_args['scan']) == len(atom_inds) == 2 \
                and set(scan_args['scan']) == set(ic_dict['atoms'][-1]):
            ic_dict['scan'].append(True)
        else:
            # Currently doesn't support scan of angles.
            ic_dict['scan'].append(False)
    ic_info = pd.DataFrame.from_dict(ic_dict)
    ic_info = ic_info.set_index('label')
    return ic_info


def parse_ic_values(ic_block: List[str],
                    ) -> pd.DataFrame:
    """
    Get the internal coordinates (ic) for an intermediate scan conformer

    Args:
        ic_block (list): A list of strings containing the optimized internal coordinates of
        an intermediate scan conformer

    Raises:
        NotImplementedError: If the software is not supported

    Returns:
        pd.DataFrame: A DataFrame containing the values of the internal coordinates
    """
    ic_dict = {item: [] for item in ['label', 'value']}
    for line in ic_block:
        # Line example with split() indices:
        # 0 1     2                       3              4      5    6                   7
        # ! R1    R(1,2)                  1.3602         -DE/DX =    0.0                 !
        terms = line.split()
        ic_dict['label'].append(terms[1])
        ic_dict['value'].append(float(terms[3]))
    ics = pd.DataFrame.from_dict(ic_dict)
    ics = ics.set_index('label')
    return ics


register_ess_adapter('gaussian', GaussianParser)
