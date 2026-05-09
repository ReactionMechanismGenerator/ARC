"""
Forge skeletal Gaussian 16-format log files from mockter fixture data.

This module is a pure-function module: input is a structured dict (geometry,
electronic energy, frequencies, ZPE, optional Hessian text block), output is
a Gaussian log string parsable by both ARC's ``arc.parser.adapters.gaussian``
and Arkane's ``arkane.ess.gaussian.GaussianLog``.

Only scientific content is emitted: route section, orientation tables, SCF /
CCSD(T) energy lines, the "Harmonic frequencies" + "Thermochemistry" block,
the verbatim "Force constants in Cartesian coordinates" splice (when
provided), and a "Normal termination" line. No banner, citations, runtime
or machine info.
"""

import datetime
import math
from typing import Iterable

import numpy as np

from arc.species.converter import str_to_xyz


KJ_PER_HARTREE = 2625.4996394798
KB_CM1_PER_K = 0.6950356                # k_B in cm^-1 per K (vibrational temperatures)
H_PLANCK_SI = 6.62607015e-34            # J·s
NA_AVO = 6.02214076e23                  # 1/mol
AMU_KG = 1.66053906660e-27              # kg per amu

ELEMENT_MASS_AMU: dict[str, float] = {
    'H': 1.00782503207, 'D': 2.01410177812, 'T': 3.01604927791,
    'He': 4.002602,
    'Li': 7.016003436, 'Be': 9.012183, 'B': 11.00930536, 'C': 12.0,
    'N': 14.0030740048, 'O': 15.99491461956, 'F': 18.998403163, 'Ne': 19.9924401762,
    'Na': 22.989769282, 'Mg': 23.985041697, 'Al': 26.98153853,
    'Si': 27.9769265325, 'P': 30.97376163, 'S': 31.97207100, 'Cl': 34.96885268,
    'Ar': 39.9623831225, 'K': 38.96370668, 'Ca': 39.96259098,
    'Br': 78.9183371, 'I': 126.904473,
}

ATOMIC_NUMBER: dict[str, int] = {
    'H': 1, 'D': 1, 'T': 1, 'He': 2,
    'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17,
    'Ar': 18, 'K': 19, 'Ca': 20,
    'Br': 35, 'I': 53,
}


def render_gaussian_log(
    job_type: str,
    xyz: dict | str | None = None,
    e_elect_hartree: float | None = None,
    method: str = 'wB97XD',
    basis: str = 'def2TZVP',
    multiplicity: int = 1,
    charge: int = 0,
    freqs_cm1: list[float] | None = None,
    zpe_hartree: float | None = None,
    hessian_block: str | None = None,
    composite_method: str | None = None,
    e_composite_hartree: float | None = None,
    is_t1_capable: bool = False,
    t1_diagnostic: float | None = None,
    title: str = 'mockter',
) -> str:
    """
    Forge a Gaussian log string from fixture data.

    Args:
        job_type (str): One of 'opt', 'fine_opt', 'freq', 'sp', 'composite',
                        'conformer', 'scan'.
        xyz (dict | str | None): Geometry as ARC xyz dict or string. Required
                                 for any job_type that emits an orientation block.
        e_elect_hartree (float | None): Electronic energy in Hartree. Emitted as
                                        "SCF Done:" for DFT or "CCSD(T)=" if
                                        ``is_t1_capable`` is True.
        method (str): Method name for the route section / "SCF Done:" tag.
        basis (str): Basis set name for the route section.
        multiplicity (int): Spin multiplicity (used for the ``Multiplicity =`` line).
        charge (int): Charge (used for the symbolic Z-matrix line).
        freqs_cm1 (list[float] | None): Harmonic frequencies in cm^-1. Negative
                                        values become imaginary modes. Required
                                        for 'freq' and 'composite' jobs.
        zpe_hartree (float | None): ZPE in Hartree. Required for 'freq' and 'composite'.
        hessian_block (str | None): Verbatim Gaussian "Force constants in Cartesian
                                    coordinates:" text block, including its header
                                    line. Spliced as-is when provided.
        composite_method (str | None): Composite method label (e.g. 'CBS-QB3').
                                       When set, emits a composite energy line.
        e_composite_hartree (float | None): Composite energy in Hartree.
        is_t1_capable (bool): If True, emit electronic energy as "CCSD(T)=" instead of "SCF Done:".
        t1_diagnostic (float | None): T1 diagnostic value (informational only;
                                      ARC's Gaussian parser does not extract it).
        title (str): Title block string.

    Returns:
        str: Forged Gaussian log text.
    """
    parts: list[str] = []
    parts.append(_render_header(method, basis, job_type, charge, multiplicity, title, composite_method))

    if xyz is not None:
        xyz_dict = xyz if isinstance(xyz, dict) else str_to_xyz(xyz)
        parts.append(_render_orientation_block('Input orientation:', xyz_dict))
        parts.append(_render_orientation_block('Standard orientation:', xyz_dict))
        parts.append(_render_rotational_constants(xyz_dict))

    if e_elect_hartree is not None:
        if is_t1_capable:
            parts.append(_render_ccsdt_energy(e_elect_hartree, t1_diagnostic))
        else:
            parts.append(_render_scf_done(method, e_elect_hartree))

    if freqs_cm1 is not None:
        parts.append(_render_frequencies_block(freqs_cm1))
        if zpe_hartree is None:
            raise ValueError('zpe_hartree is required when freqs_cm1 is provided.')
        if xyz is None:
            raise ValueError('xyz is required when freqs_cm1 is provided (for thermo block).')
        parts.append(_render_thermochemistry_block(xyz_dict, freqs_cm1, multiplicity, zpe_hartree, e_elect_hartree))

    if composite_method is not None and e_composite_hartree is not None:
        if zpe_hartree is None:
            raise ValueError('zpe_hartree is required for composite logs.')
        parts.append(_render_composite_energy(composite_method, e_composite_hartree, zpe_hartree))

    if hessian_block is not None:
        parts.append(hessian_block.rstrip() + '\n')

    parts.append(_render_normal_termination())
    return ''.join(parts)


def _render_header(
    method: str,
    basis: str,
    job_type: str,
    charge: int,
    multiplicity: int,
    title: str,
    composite_method: str | None,
) -> str:
    """
    Emit the route section, title, and symbolic Z-matrix charge/multiplicity line.

    Args:
        method (str): Method name (e.g. 'wB97XD').
        basis (str): Basis set name (e.g. 'def2TZVP').
        job_type (str): One of 'opt', 'fine_opt', 'freq', 'sp', 'composite',
                        'conformer', 'scan'.
        charge (int): Net charge.
        multiplicity (int): Spin multiplicity.
        title (str): Job title for the title block.
        composite_method (str | None): Composite method (e.g. 'CBS-QB3') that
                                       overrides ``method``/``basis`` if set.

    Returns:
        str: Header text including the route, title, and charge/multiplicity lines.
    """
    if composite_method is not None:
        route_body = composite_method
    elif job_type in ('opt', 'fine_opt', 'conformer'):
        route_body = f'{method}/{basis} opt'
    elif job_type == 'freq':
        route_body = f'{method}/{basis} freq IOp(7/33=1)'
    elif job_type == 'sp':
        route_body = f'{method}/{basis}'
    elif job_type == 'scan':
        route_body = f'{method}/{basis} opt=modredundant'
    else:
        route_body = f'{method}/{basis}'

    lines: list[str] = []
    lines.append(' ' + '-' * 70 + '\n')
    lines.append(f' #P {route_body}\n')
    lines.append(' ' + '-' * 70 + '\n')
    lines.append(' ---\n')
    lines.append(f' {title}\n')
    lines.append(' ---\n')
    lines.append(' Symbolic Z-matrix:\n')
    lines.append(f' Charge =  {charge} Multiplicity = {multiplicity}\n')
    return ''.join(lines)


def _render_orientation_block(header: str, xyz: dict) -> str:
    """
    Emit a Gaussian orientation block ('Input orientation:' or 'Standard orientation:').

    Args:
        header (str): Either 'Input orientation:' or 'Standard orientation:'.
        xyz (dict): ARC xyz dict with 'symbols' and 'coords'.

    Returns:
        str: The orientation block including both dashed separators.
    """
    lines: list[str] = []
    lines.append(f'                          {header}                          \n')
    lines.append(' ' + '-' * 69 + '\n')
    lines.append(' Center     Atomic      Atomic             Coordinates (Angstroms)\n')
    lines.append(' Number     Number       Type             X           Y           Z\n')
    lines.append(' ' + '-' * 69 + '\n')
    for i, (sym, (x, y, z)) in enumerate(zip(xyz['symbols'], xyz['coords']), start=1):
        an = ATOMIC_NUMBER[sym]
        lines.append(
            f'   {i:>4d}        {an:>4d}           0     {x:>11.6f} {y:>11.6f} {z:>11.6f}\n'
        )
    lines.append(' ' + '-' * 69 + '\n')
    return ''.join(lines)


def _render_scf_done(method: str, e_hartree: float) -> str:
    """
    Emit one ``SCF Done:`` line in Gaussian's format.

    Args:
        method (str): Method tag (e.g. 'RwB97XD').
        e_hartree (float): Electronic energy in Hartree.

    Returns:
        str: A single ``SCF Done:`` line.
    """
    method_tag = f'R{method.upper()}' if not method.upper().startswith(('R', 'U')) else method.upper()
    return f' SCF Done:  E({method_tag}) =  {e_hartree:.10f}     A.U. after    1 cycles\n'


def _render_ccsdt_energy(e_hartree: float, t1: float | None) -> str:
    """
    Emit a ``CCSD(T)=`` energy line plus an optional T1 diagnostic comment.

    Args:
        e_hartree (float): CCSD(T) energy in Hartree.
        t1 (float | None): T1 diagnostic, or None to skip the line.

    Returns:
        str: One or two lines.
    """
    out = f' CCSD(T)= {e_hartree:.10f}\n'
    if t1 is not None:
        out += f' T1 Diagnostic    {t1:.6f}\n'
    return out


def _render_frequencies_block(freqs: Iterable[float]) -> str:
    """
    Emit the Harmonic frequencies block (3 freqs per chunk, no displacements).

    Args:
        freqs (Iterable[float]): Frequencies in cm^-1.

    Returns:
        str: The full frequencies block, prefixed by the
             ``Harmonic frequencies (cm**-1) ... and normal coordinates:`` header.
    """
    freqs = list(freqs)
    lines: list[str] = []
    lines.append(' Harmonic frequencies (cm**-1), IR intensities (KM/Mole), Raman scattering\n')
    lines.append(' activities (A**4/AMU), depolarization ratios for plane and unpolarized\n')
    lines.append(' incident light, reduced masses (AMU), force constants (mDyne/A),\n')
    lines.append(' and normal coordinates:\n')
    for chunk_start in range(0, len(freqs), 3):
        chunk = freqs[chunk_start:chunk_start + 3]
        idx_line = '          ' + ''.join(f'{chunk_start + j + 1:>23d}' for j in range(len(chunk))) + '\n'
        sym_line = '          ' + ''.join(f'{"A":>23s}' for _ in chunk) + '\n'
        freq_line = ' Frequencies --   ' + '  '.join(f'{f:>9.4f}' for f in chunk) + '\n'
        rmass_line = ' Red. masses --   ' + '  '.join(f'{1.0:>9.4f}' for _ in chunk) + '\n'
        fcons_line = ' Frc consts  --   ' + '  '.join(f'{1.0:>9.4f}' for _ in chunk) + '\n'
        irint_line = ' IR Inten    --   ' + '  '.join(f'{0.0:>9.4f}' for _ in chunk) + '\n'
        lines.extend([idx_line, sym_line, freq_line, rmass_line, fcons_line, irint_line])
    return ''.join(lines)


def _render_rotational_constants(xyz: dict) -> str:
    """
    Compute and emit the ``Rotational constants (GHZ):`` line from the geometry.

    Linear molecules emit ``Rotational constant (GHZ):`` (singular) per Arkane's
    ``LinearRotor`` parsing branch.

    Args:
        xyz (dict): ARC xyz dict.

    Returns:
        str: The rotational-constants line(s).
    """
    coords = np.asarray(xyz['coords'], dtype=float)
    masses = np.asarray([ELEMENT_MASS_AMU[s] for s in xyz['symbols']], dtype=float)

    com = (coords.T * masses).sum(axis=1) / masses.sum()
    centered = coords - com

    inertia = np.zeros((3, 3))
    for m, (x, y, z) in zip(masses, centered):
        inertia[0, 0] += m * (y * y + z * z)
        inertia[1, 1] += m * (x * x + z * z)
        inertia[2, 2] += m * (x * x + y * y)
        inertia[0, 1] -= m * x * y
        inertia[0, 2] -= m * x * z
        inertia[1, 2] -= m * y * z
    inertia[1, 0] = inertia[0, 1]
    inertia[2, 0] = inertia[0, 2]
    inertia[2, 1] = inertia[1, 2]

    eigvals = np.sort(np.linalg.eigvalsh(inertia))
    is_linear = eigvals[0] < 1e-3 * max(eigvals[-1], 1e-12)

    moments_si = eigvals * AMU_KG * (1e-10) ** 2
    rot_constants_ghz = []
    for I in moments_si:
        if I < 1e-46:
            continue
        rot_constants_ghz.append(H_PLANCK_SI / (8 * math.pi ** 2 * I) / 1e9)

    if is_linear and rot_constants_ghz:
        b = rot_constants_ghz[-1]
        return f' Rotational constant (GHZ):  {b:.7f}\n'
    while len(rot_constants_ghz) < 3:
        rot_constants_ghz.append(0.0)
    a, b, c = sorted(rot_constants_ghz, reverse=True)[:3]
    return f' Rotational constants (GHZ):       {a:>11.7f}    {b:>11.7f}    {c:>11.7f}\n'


def _render_thermochemistry_block(
    xyz: dict,
    freqs_cm1: Iterable[float],
    multiplicity: int,
    zpe_hartree: float,
    e_elect_hartree: float | None,
) -> str:
    """
    Emit the Multiplicity, ``- Thermochemistry -`` and ``Zero-point correction=`` blocks.

    Includes ``Molecular mass:`` and ``Vibrational temperatures:`` (Arkane's
    ``load_conformer`` reads these). The ``Sum of electronic and zero-point
    Energies=`` line is emitted when ``e_elect_hartree`` is provided.

    Args:
        xyz (dict): ARC xyz dict.
        freqs_cm1 (Iterable[float]): Harmonic frequencies in cm^-1.
        multiplicity (int): Spin multiplicity.
        zpe_hartree (float): ZPE in Hartree.
        e_elect_hartree (float | None): Electronic energy for the
                                        ``Sum of electronic and zero-point Energies=`` line.

    Returns:
        str: The thermochemistry block.
    """
    masses = [ELEMENT_MASS_AMU[s] for s in xyz['symbols']]
    total_mass = sum(masses)
    pos_freqs = [f for f in freqs_cm1 if f > 0]
    vib_temps = [f / KB_CM1_PER_K for f in pos_freqs]

    lines: list[str] = []
    lines.append(f' Multiplicity = {multiplicity}\n')
    lines.append(' -------------------\n')
    lines.append(' - Thermochemistry -\n')
    lines.append(' -------------------\n')
    lines.append(' Temperature   298.150 Kelvin.  Pressure   1.00000 Atm.\n')
    for i, (sym, m) in enumerate(zip(xyz['symbols'], masses), start=1):
        lines.append(f' Atom     {i} has atomic number  {ATOMIC_NUMBER[sym]} and mass  {m:>9.5f}\n')
    lines.append(f' Molecular mass:    {total_mass:>9.5f} amu.\n')

    if vib_temps:
        formatted = [f'{t:>8.2f}' for t in vib_temps]
        # Match Gaussian's layout: up to 5 temps per row.
        # Row 1 starts with "Vibrational temperatures:" — Arkane reads [2:].
        # Row 2 starts with "(Kelvin)" — Arkane reads [1:]; emit it even when
        # there are no continuation temps so Arkane's mandatory second readline
        # consumes a parseable line, not the next block.
        # Rows 3+ are unprefixed continuations until a blank line.
        lines.append(' Vibrational temperatures:   ' + '  '.join(formatted[:5]) + '\n')
        if len(formatted) > 5:
            lines.append('          (Kelvin)           ' + '  '.join(formatted[5:10]) + '\n')
        else:
            lines.append('          (Kelvin)\n')
        for i in range(10, len(formatted), 5):
            lines.append('                             ' + '  '.join(formatted[i:i + 5]) + '\n')
        lines.append('\n')

    lines.append(f' Zero-point correction=                           {zpe_hartree:.6f} (Hartree/Particle)\n')
    if e_elect_hartree is not None:
        lines.append(
            f' Sum of electronic and zero-point Energies=            {e_elect_hartree + zpe_hartree:.6f}\n'
        )
    return ''.join(lines)


def _render_composite_energy(composite_method: str, e_composite_hartree: float, zpe_hartree: float) -> str:
    """
    Emit a composite-method energy line in the format Arkane recognizes.

    Args:
        composite_method (str): Composite method (e.g. 'CBS-QB3', 'G4', 'G3', 'G4MP2').
        e_composite_hartree (float): Composite E0 (= electronic + ZPE) in Hartree.
        zpe_hartree (float): Scaled ZPE in Hartree (already part of E0).

    Returns:
        str: A line like ``CBS-QB3 (0 K)= -115.59054`` plus an ``E(ZPE)=`` line.
    """
    name = composite_method.upper()
    if name == 'CBS-QB3':
        line = f' CBS-QB3 (0 K)=             {e_composite_hartree:>13.6f}  E(CBS-QB3)= {e_composite_hartree:>13.6f}\n'
    elif name == 'G3':
        line = f' G3(0 K)=             {e_composite_hartree:>13.6f}  G3 Energy=             {e_composite_hartree:>13.6f}\n'
    elif name == 'G4':
        line = f' G4(0 K)=             {e_composite_hartree:>13.6f}  G4 Energy=             {e_composite_hartree:>13.6f}\n'
    elif name == 'G4MP2':
        line = f' G4MP2(0 K)=          {e_composite_hartree:>13.6f}  G4MP2 Energy=          {e_composite_hartree:>13.6f}\n'
    elif name == 'CBS-4':
        line = f' CBS-4 (0 K)=             {e_composite_hartree:>13.6f}\n'
    else:
        line = f' {name} (0 K)=             {e_composite_hartree:>13.6f}\n'
    line += f' E(ZPE)= {zpe_hartree:.6f}\n'
    return line


def _render_normal_termination() -> str:
    """
    Emit the final ``Normal termination of Gaussian`` line.

    Returns:
        str: The termination line.
    """
    now = datetime.datetime.now().strftime('%a %b %e %H:%M:%S %Y')
    return f' Normal termination of Gaussian 16 at {now}.\n'
