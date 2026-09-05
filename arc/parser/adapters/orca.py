"""
An adapter for parsing Orca log files.
"""

from abc import ABC

import math
import numpy as np
import pandas as pd
import re

from arc.common import SYMBOL_BY_NUMBER, is_str_int
from arc.constants import E_h_kJmol, bohr_to_angstrom
from arc.species.converter import str_to_xyz, xyz_from_data
from arc.parser.adapter import ESSAdapter
from arc.parser.factory import register_ess_adapter
from arc.parser.parser import _get_lines_from_file, s_squared_expected_from_multiplicity


SPIN_SYMMETRY_BREAKING_S_SQUARED = 0.01


def _root_is_negative(eigenvalue: float) -> bool:
    """
    Check whether a stability-matrix root is a negative one.

    Negative zero counts as negative: ORCA prints a marginal root as ``-0.00000000``,
    which parses to ``-0.0``, for which the ordinary ``< 0`` comparison is ``False``.
    A wavefunction ORCA reports unstable on such a root would otherwise be recorded
    with no negative roots at all.

    Args:
        eigenvalue (float): The root of the stability matrix.

    Returns:
        bool: Whether the root is negative.
    """
    return eigenvalue < 0 or (eigenvalue == 0 and math.copysign(1.0, eigenvalue) < 0)


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

    def parse_wavefunction_stability(self) -> dict | None:
        """
        Parse the verdict of an ORCA ``STABPerform`` wavefunction stability analysis.

        ORCA opens each analysis with a ``WAVEFUNCTION STABILITY ANALYSIS`` banner,
        lists the lowest roots of the stability matrix as::

            The eigenvalues of the stability matrix:
               E( 0) =  -0.06466151 Eh

        and closes with one of::

            The stability analysis shows that the wavefunction is stable
            The stability analysis indicates that the wavefunction is unstable

        Unlike Gaussian, ORCA neither labels a root by the perturbation it came from
        nor reports its spin expectation value, so every entry of
        ``negative_eigenvectors`` carries ``'label': None``. A root printed as ``-0.00000000``
        parses to negative zero and counts as negative, so a wavefunction reported unstable
        on a marginal root is not reported with an empty root list.

        TWO ANALYSES PER LOG. ``STABRestartUHFifUnstable true``, which ARC always sets
        because ORCA 6.0.0 aborts in LEANSCF when it is false, rotates the orbitals of
        an unstable wavefunction, re-converges the SCF and analyses the result again.
        Such a log holds two analyses with opposite verdicts. ``verdict`` is read from
        the FIRST one, which is the wavefunction under test, i.e. the one the frequency
        job built its Hessian from; ``lowest_eigenvalue`` and ``negative_eigenvectors``
        likewise come from that block. ``n_analyses`` counts the blocks and
        ``followed_to_stable`` reports whether an analysis that opened UNSTABLE ended
        stable, i.e. whether ORCA reached a stable solution after following the
        instability. A log opening on a stable analysis reports ``False`` however many
        blocks follow it, so a concatenation of stable analyses is not read as a follow
        and no ``<S**2>`` of a wavefunction the log never relaxed into is reported.
        ``restricted`` is read from the ``HFTyp`` line preceding the first analysis, so
        a restart to an unrestricted solution does not overwrite the reference tested;
        an ``RO`` reference is reported as ``None`` rather than as restricted, since its
        instabilities relax neither of the two constraints the flags name.

        WHICH SECTORS ARE TESTED, and hence which of the two instability flags a verdict
        can set. ORCA analyses an RHF/RKS reference in UHF/UKS space and a UHF/UKS
        reference in UHF/UKS space, both of which are Ms-conserving; the spin-flip
        (UHF -> GHF) sector is analysed in neither, and Gaussian's ``Stable=RExt`` uses
        the same Ms-conserving ``<AA,BB:AA,BB>`` singles matrix for both references, so
        the two codes span the same space and neither reaches the GHF sector. Measured on
        four systems the verdicts agreed in every case, and at matched functional (ORCA's
        ``B3LYP/G`` is Gaussian's VWN3 parameterisation, while plain ORCA ``B3LYP`` uses
        VWN-5) the lowest roots agreed to under 0.4% on the three systems where both codes
        converged to the same SCF solution.

        * An unrestricted reference is tested against spin-conserving rotations, which is
          Gaussian's internal sector, so an instability is recorded as
          ``internal_instability`` with ``relaxations`` empty. ``external_instability``
          stays ``None``, since a spin-flip root would be the evidence for it and no root
          of that kind is computed.
        * For a restricted reference the single unlabelled matrix spans both the
          spin-conserving (internal) and the spin-symmetry-breaking (RHF -> UHF, external)
          sectors, and ORCA does not say which root it found. The sector is therefore
          MEASURED rather than assumed, from the solution ORCA relaxes into: a nominal
          singlet that reaches a stable solution whose ``<S**2>`` exceeds
          ``SPIN_SYMMETRY_BREAKING_S_SQUARED`` broke the spin symmetry, which is an
          external instability, while one that reaches a stable solution still at
          ``<S**2>`` of zero relaxed within the spin-conserving sector, which is an
          internal instability. That value is reported as ``s_squared_after_follow``, and
          the threshold sits far above the ``1e-5`` a spin-symmetric UHF solution's
          numerical noise reaches and far below the few tenths a broken-symmetry singlet
          carries, so nothing realistic falls near it.
          THE SECTOR IS READ OFF EVERY FOLLOWED SOLUTION, whether or not the last analysis
          ended stable. ORCA re-converges the SCF before each analysis it runs, so the
          ``<S**2>`` of the solution it relaxed into is that of a converged determinant
          whichever try it stopped on, and a solution that reached ``<S**2>`` of a few
          tenths broke the spin symmetry whether or not a further root remains. ORCA allows
          five follow attempts, so a biradicaloid singlet reaching the last of them is
          ordinary, and the question the sector answers is whether a lower solution exists
          outside the spin symmetry rather than whether the one ORCA stopped on is itself
          the bottom.
          An instability ORCA never followed at all, which is a log holding one analysis,
          leaves nothing to measure the sector from: the verdict is
          ``'unattributed_instability'`` with both flags ``None``.

        A log that ran an analysis but whose verdict line could not be read yields
        ``'unknown'``. An instability whose reference could not be read yields
        ``'unattributed_instability'``, since the reference decides which of the two flags
        an instability sets. The roots of the first block are reported either way.

        ``invalidates_analytic_freq`` follows the same rule the Gaussian reader applies, so
        the two ESSs report the same value for the same physical situation: an internal
        instability invalidates the analytic frequencies of either reference, an external
        one invalidates only an unrestricted reference's, and an instability whose sector or
        reference is undetermined leaves the question open as ``None``.

        WHICH WAVEFUNCTION EACH FIELD DESCRIBES. ``verdict``, ``lowest_eigenvalue``,
        ``negative_eigenvectors`` and ``restricted`` describe the wavefunction under TEST.
        The rest of a restart log, its ``FINAL SINGLE POINT ENERGY`` and its final
        ``<S**2>`` among them, describes the FOLLOWED solution ORCA relaxed into, which is a
        different wavefunction; ``s_squared_after_follow`` is reported under a name that says
        so. A consumer reading a quantity off the log this verdict came from is reading the
        followed solution unless it is one of the four fields named here.

        Returns: dict | None
            ``{'verdict': str, 'internal_instability': bool | None,
               'external_instability': bool | None, 'relaxations': list[str],
               'negative_eigenvectors': list[dict], 'lowest_eigenvalue': float | None,
               'restricted': bool | None, 'invalidates_analytic_freq': bool | None,
               'n_analyses': int, 'followed_to_stable': bool,
               's_squared_after_follow': float | None}``,
            or ``None`` when the log holds no stability analysis. ``verdict`` is one of
            ``'stable'``, ``'internal_instability'``, ``'external_instability'``,
            ``'unattributed_instability'`` or ``'unknown'``.
        """
        blocks, restricted = list(), None
        for line in _get_lines_from_file(self.log_file_path):
            if 'WAVEFUNCTION STABILITY ANALYSIS' in line:
                blocks.append({'eigenvalues': list(), 'verdict': None})
                continue
            if not blocks:
                if 'HFTyp' in line:
                    match = re.search(r'HFTyp\s*\.+\s*(\S+)', line)
                    if match is not None:
                        hf_type = match.group(1).upper()
                        restricted = None if hf_type.startswith('RO') else not hf_type.startswith('U')
                continue
            match = re.match(r'\s*E\(\s*\d+\)\s*=\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?)\s*Eh', line)
            if match is not None:
                blocks[-1]['eigenvalues'].append(float(re.sub(r'[Dd]', 'e', match.group(1))))
                continue
            if 'stability analysis' in line and 'wavefunction is' in line:
                if 'wavefunction is unstable' in line:
                    blocks[-1]['verdict'] = 'unstable'
                elif 'wavefunction is stable' in line:
                    blocks[-1]['verdict'] = 'stable'
        if not blocks:
            return None
        eigenvalues = blocks[0]['eigenvalues']
        negative_eigenvectors = [{'label': None, 'eigenvalue': eigenvalue}
                                 for eigenvalue in eigenvalues if _root_is_negative(eigenvalue)]
        lowest_eigenvalue = min(eigenvalues) if eigenvalues else None
        followed = len(blocks) > 1 and blocks[0]['verdict'] == 'unstable'
        followed_to_stable = followed and blocks[-1]['verdict'] == 'stable'
        s_squared_after_follow = None
        if followed:
            s_squared = self.parse_s_squared()
            s_squared_after_follow = s_squared['s_squared'] if s_squared is not None else None
        internal_instability, external_instability, relaxations = None, None, list()
        if blocks[0]['verdict'] == 'stable':
            verdict = 'stable'
            internal_instability = False
            external_instability = False if restricted else None
        elif blocks[0]['verdict'] == 'unstable' and restricted is False:
            verdict, internal_instability = 'internal_instability', True
        elif blocks[0]['verdict'] == 'unstable' and restricted is True \
                and s_squared_after_follow is not None:
            if s_squared_after_follow > SPIN_SYMMETRY_BREAKING_S_SQUARED:
                verdict, external_instability = 'external_instability', True
                relaxations.append('RHF -> UHF')
            else:
                verdict, internal_instability = 'internal_instability', True
        elif blocks[0]['verdict'] == 'unstable':
            verdict = 'unattributed_instability'
        else:
            verdict = 'unknown'
        if verdict == 'internal_instability':
            invalidates_analytic_freq = True
        elif verdict == 'stable':
            invalidates_analytic_freq = False
        elif verdict == 'external_instability' and restricted is not None:
            invalidates_analytic_freq = not restricted
        else:
            invalidates_analytic_freq = None
        return {'verdict': verdict,
                'internal_instability': internal_instability,
                'external_instability': external_instability,
                'relaxations': relaxations,
                'negative_eigenvectors': negative_eigenvectors,
                'lowest_eigenvalue': lowest_eigenvalue,
                'restricted': restricted,
                'invalidates_analytic_freq': invalidates_analytic_freq,
                'n_analyses': len(blocks),
                'followed_to_stable': followed_to_stable,
                's_squared_after_follow': s_squared_after_follow,
                }

    def parse_s_squared(self) -> dict[str, float | None] | None:
        """
        Parse the S**2 spin-contamination diagnostic from an ORCA UHF/UKS log.

        ORCA prints, for an unrestricted reference::

            Expectation value of <S**2>     :     0.754185
            Ideal value S*(S+1) for S=0.5   :     0.750000

        The value of record is the *last* (converged / final-SCF) pair on the
        log; on a multi-image or multi-step log every SCF prints its own block
        and the final one is the calculation's. On a wavefunction-stability log
        that followed an instability, that is the SCF ORCA relaxed into and not
        the one the analysis tested. Unlike Gaussian's ``<S**2>=``,
        this anchor string occurs nowhere in an ORCA log but in that block, so
        it needs no further anchoring. Restricted (closed-shell) references
        don't print these lines, so this returns ``None`` for them. ORCA has no
        spin-contaminant annihilation step, so ``s_squared_annihilated`` is
        always ``None``. The ideal value is taken from the
        ``Ideal value S*(S+1)`` line of the same block as the expectation value
        of record (that is exactly the expected ``S(S+1)``), else from the
        parsed ``Multiplicity`` line.

        Returns: dict[str, float | None] | None
            ``{'s_squared': float, 's_squared_expected': float | None,
               's_squared_annihilated': None}`` or ``None``.
        """
        s_squared, s_squared_expected, multiplicity = None, None, None
        for line in _get_lines_from_file(self.log_file_path):
            if 'Expectation value of <S**2>' in line:
                match = re.search(r':\s*([-+]?\d*\.?\d+)', line)
                if match:
                    try:
                        s_squared, s_squared_expected = float(match.group(1)), None
                    except ValueError:
                        continue
            elif 'Ideal value S*(S+1)' in line:
                match = re.search(r':\s*([-+]?\d*\.?\d+)', line)
                if match:
                    try:
                        s_squared_expected = float(match.group(1))
                    except ValueError:
                        continue
            elif 'Multiplicity' in line:
                match = re.search(r'\.\.\.\.\s*(\d+)', line)
                if match and is_str_int(match.group(1)):
                    multiplicity = int(match.group(1))
        if s_squared is None:
            return None
        if s_squared_expected is None:
            s_squared_expected = s_squared_expected_from_multiplicity(multiplicity)
        return {
            's_squared': s_squared,
            's_squared_expected': s_squared_expected,
            's_squared_annihilated': None,
        }

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


register_ess_adapter('orca', OrcaParser)
