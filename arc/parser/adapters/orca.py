"""
An adapter for parsing Orca log files.
"""

from abc import ABC

import numpy as np
import pandas as pd
import re

from arc.common import SYMBOL_BY_NUMBER, get_logger
from arc.constants import E_h_kJmol, bohr_to_angstrom
from arc.species.converter import str_to_xyz, xyz_from_data
from arc.parser.adapter import ESSAdapter
from arc.parser.factory import register_ess_adapter
from arc.parser.parser import _get_lines_from_file


logger = get_logger()


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

    def parse_frequencies(self) -> np.ndarray | None:
        """
        Parse the frequencies from a freq job output file.

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

    def parse_normal_mode_displacement(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        Parse frequencies and normal mode displacement.

        Returns: tuple[np.ndarray | None, np.ndarray | None]
            The frequencies (in cm^-1) and the normal mode displacements.
        """
        # Not implemented for Orca.
        return None, None

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


_ORCA_LETTER_TO_TCKDB_KIND: dict[str, tuple[str, int]] = {
    'C': ('cartesian_atom', 1),
    'B': ('bond', 2),
    'A': ('angle', 3),
    'D': ('dihedral', 4),
}


def parse_orca_constraints(file_path: str) -> list[dict]:
    """Parse held-fixed constraints from an ORCA input deck (best-effort).

    Recognises the standard ORCA ``%geom Constraints`` block::

        %geom Constraints
          { B 0 1 1.4 C }
          { A 0 1 2 90.0 C }
          { D 0 1 2 3 180.0 C }
          { C 0 C }
        end

    Notes / known limitations:
        - ORCA atom indices in the input deck are 0-based; this parser
          converts them to TCKDB's 1-based convention at the boundary.
        - ARC's ORCA adapter does not currently emit ``%geom Constraints``
          blocks (only ``%geom Scan``). This parser is therefore mainly
          defensive — it handles user-supplied decks and any future ARC
          emission. Scan blocks are *not* parsed as constraints.
        - Variants like ``optimize { B i j C }`` (single-coordinate form)
          and ``Constraints` blocks scattered across multiple ``%geom``
          sections are recognised; everything else is ignored with a
          debug log rather than failing the whole parse.

    Returns ``[]`` on file read errors or when no recognised
    ``Constraints`` block is found.
    """
    try:
        with open(file_path, 'r') as f:
            text = f.read()
    except (OSError, IOError) as exc:
        logger.warning("parse_orca_constraints: cannot read %s: %s",
                       file_path, exc)
        return []

    constraints: list[dict] = []
    # Find every Constraints block: from 'Constraints' up to the matching
    # 'end' (case-insensitive). ORCA blocks are not nested.
    pattern = re.compile(r'Constraints(.*?)end', re.IGNORECASE | re.DOTALL)
    for match in pattern.finditer(text):
        block = match.group(1)
        for raw in block.splitlines():
            record = _parse_orca_constraint_line(raw)
            if record is not None:
                constraints.append(record)
    return constraints


def _parse_orca_constraint_line(line: str) -> dict | None:
    """Parse one ``{ B i j v C }``-style ORCA constraint line into a record.

    ORCA constraint syntax inside ``%geom Constraints``::

        { <letter> <atom indices...> [<value>] C }

    The trailing ``C`` flags the coordinate as constrained. ``value`` is
    optional. Atom indices are converted from 0-based (ORCA) to 1-based
    (TCKDB). Unparseable lines return None and are skipped silently at
    debug level so the rest of the block still parses.
    """
    stripped = line.strip()
    if not stripped or stripped.startswith('#'):
        return None
    # Tolerate either '{' or no braces (rare hand-written variants).
    inner = stripped.strip('{}').strip()
    if not inner:
        return None
    tokens = inner.split()
    if len(tokens) < 2:
        return None
    letter = tokens[0].upper()
    entry = _ORCA_LETTER_TO_TCKDB_KIND.get(letter)
    if entry is None:
        logger.debug("parse_orca_constraints: skipping unrecognised letter "
                     "%s in line: %s", letter, line)
        return None
    kind, expected_n = entry

    if len(tokens) < 1 + expected_n:
        logger.debug("parse_orca_constraints: too few atom tokens for letter "
                     "%s (need %d): %s", letter, expected_n, line)
        return None

    try:
        zero_based = [int(tok) for tok in tokens[1:1 + expected_n]]
    except ValueError:
        logger.debug("parse_orca_constraints: non-integer atom index in: %s",
                     line)
        return None
    atoms = [a + 1 for a in zero_based]

    target_value: float | None = None
    rest = tokens[1 + expected_n:]
    for tok in rest:
        if tok.upper() == 'C':
            break
        try:
            target_value = float(tok)
        except ValueError:
            continue

    return {
        'constraint_kind': kind,
        'atoms': atoms,
        'target_value': target_value,
    }


register_ess_adapter('orca', OrcaParser)
