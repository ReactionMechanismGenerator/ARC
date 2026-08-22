#!/usr/bin/env python3
# encoding: utf-8

"""
A standalone script to run ASE (Atomic Simulation Environment) jobs.
Standardizes interaction with various calculators.
"""

import argparse
import math
import os
import sys
import yaml
import numpy as np

from ase import Atoms
from ase.constraints import FixInternals
from ase.data import covalent_radii
from ase.neighborlist import build_neighbor_list, natural_cutoffs
from ase.optimize import BFGS, LBFGS, GPMin
from ase.optimize.sciopt import SciPyFminBFGS, SciPyFminCG
from ase.vibrations import Vibrations

# Constants matched to ASE internal units (3.23.0+) for exact numerical matching
c = 299792458.0
e = 1.602176565e-19
amu = 1.660538921e-27
pi = math.pi
h = 6.62606896e-34
E_h = 4.35974434e-18  # Hartree in Joules
SCHEMA_VERSION = 2
N_A = 6.02214179e23


def to_kJmol(energy_ev: float) -> float:
    """
    Convert ASE default (eV) to kJ/mol.
    """
    return energy_ev * e * N_A / 1000.0


def read_yaml_file(path: str):
    """
    Read a YAML file.
    """
    with open(path, 'r') as f:
        return yaml.load(stream=f, Loader=yaml.FullLoader)


def save_yaml_file(path: str, content: dict):
    """
    Save a YAML file.
    """
    def string_representer(dumper, data):
        if len(data.splitlines()) > 1:
            return dumper.represent_scalar(tag='tag:yaml.org,2002:str', value=data, style='|')
        return dumper.represent_scalar(tag='tag:yaml.org,2002:str', value=data)
    yaml.add_representer(str, string_representer)
    with open(path, 'w') as f:
        f.write(yaml.dump(data=content))


def get_calculator(calc_config: dict, charge: int = 0, multiplicity: int = 1):
    """
    Initialize the ASE calculator based on settings.
    """
    name = calc_config.get('calculator', '').lower()
    kwargs = calc_config.get('calculator_kwargs', {})
    
    if name == 'torchani':
        import torch
        import torchani
        model_name = calc_config.get('model', 'ANI2x')
        device = torch.device(calc_config.get('device', 'cpu'))
        if model_name.lower() == 'ani1ccx':
            model = torchani.models.ANI1ccx(periodic_table_index=True).to(device)
        elif model_name.lower() == 'ani1x':
            model = torchani.models.ANI1x(periodic_table_index=True).to(device)
        else:
            model = torchani.models.ANI2x(periodic_table_index=True).to(device)
        return model.ase()
    
    elif name == 'xtb':
        from xtb.ase.calculator import XTB
        if 'charge' not in kwargs:
            kwargs['charge'] = charge
        if 'uhf' not in kwargs:
            kwargs['uhf'] = multiplicity - 1
        return XTB(**kwargs)
    
    elif name == 'mopac':
        from ase.calculators.mopac import MOPAC
        if 'charge' not in kwargs:
            kwargs['charge'] = charge
        if multiplicity > 1:
            raise ValueError("ARC's integration with MOPAC vua the ASE calculator does not support multiplicity > 1.")
        return MOPAC(**kwargs)

    elif name in ('uma', 'fairchem'):
        # UMA (Meta FAIR fairchem-core). Total charge and spin (= multiplicity) are conditioned on
        # the ase.Atoms via atoms.info in main(); they are not calculator kwargs.
        from fairchem.core import FAIRChemCalculator, pretrained_mlip
        model = calc_config.get('model', 'uma-s-1p2')
        device = calc_config.get('device', 'cpu')
        task = calc_config.get('task', 'omol')
        predictor = pretrained_mlip.get_predict_unit(model, device=device)
        return FAIRChemCalculator(predictor, task_name=task)


    from ase.calculators.calculator import get_calculator_class
    try:
        calc_class = get_calculator_class(name)
        return calc_class(**kwargs)
    except Exception as exc:
        print(f"Could not load ASE calculator '{name}': {exc}")
        sys.exit(1)


def apply_constraints(atoms: Atoms, constraints_data: list):
    """
    Apply internal constraints to the Atoms object.
    """
    if not constraints_data:
        return
    bonds, angles, dihedrals = list(), list(), list()
    for constraint in constraints_data:
        indices = constraint[0]
        if len(indices) == 2:
            bonds.append([constraint[1], indices])
        elif len(indices) == 3:
            angles.append([constraint[1], indices])
        elif len(indices) == 4:
            dihedrals.append([constraint[1], indices])
    atoms.set_constraint(FixInternals(bonds=bonds, angles_deg=angles, dihedrals_deg=dihedrals))


def rotor_top(atoms: Atoms, pivot_1: int, pivot_2: int, mult: float = 1.2,
              pivot_mult: float = 1.8) -> list:
    """
    Determine the rotating group (the "top") on the ``pivot_2`` side of the ``pivot_1``-``pivot_2`` bond.

    The two pivots are the central atoms of the scanned torsion; breaking their bond splits the
    molecule into two fragments, and this returns the atom indices reachable from ``pivot_2``.

    This is the same "top" that ``arc.common.determine_top_group_indices`` computes, but kept as a
    standalone reimplementation on purpose: this script runs in the calculator's own environment
    (e.g. ``uma_env``), which has neither ARC nor RMG-Py, so it can import neither that function nor
    the RMG ``Molecule`` it operates on. Connectivity is instead rebuilt from the geometry via ASE's
    neighbor list. Keep the two traversals in agreement if either changes.

    Because connectivity here is geometric rather than a real bond graph, a bond stretched well past
    its equilibrium length - the forming or breaking bond of a TS being the case that matters - can
    fall outside the default cutoff and look non-bonded. Only the pivot pair is given the looser
    ``pivot_mult`` allowance; the rest of the graph stays at ``mult``. Loosening the whole graph
    instead would let the substituents on the two pivots reach each other across the widened gap and
    fuse into a spurious ring, which the ring check below would then report.

    Args:
        atoms (Atoms): The molecule.
        pivot_1 (int): The 0-indexed atom on the fixed side of the rotation axis.
        pivot_2 (int): The 0-indexed atom on the rotating side of the rotation axis.
        mult (float, optional): The natural-cutoff multiplier defining the bond graph.
        pivot_mult (float, optional): The covalent-radii multiplier allowed for the pivot bond
                                      alone, accommodating a stretched TS bond.

    Raises:
        ValueError: If the pivots are too far apart to be bonded even under ``pivot_mult``, or if
                    the pivot bond is part of a ring (a 1D rotor is then ill-defined).

    Returns:
        list: The sorted 0-indexed atoms of the rotating group.
    """
    nl = build_neighbor_list(atoms, natural_cutoffs(atoms, mult=mult),
                             self_interaction=False, bothways=True)
    adj = {i: set(nl.get_neighbors(i)[0]) for i in range(len(atoms))}
    if pivot_2 not in adj[pivot_1]:
        numbers = atoms.get_atomic_numbers()
        radii_sum = covalent_radii[numbers[pivot_1]] + covalent_radii[numbers[pivot_2]]
        distance = float(atoms.get_distance(pivot_1, pivot_2))
        if distance > pivot_mult * radii_sum:
            raise ValueError(f'The pivot atoms {pivot_1} and {pivot_2} are {distance:.2f} Angstrom '
                             f'apart, too far to be bonded; cannot determine the rotating group.')
    adj[pivot_1].discard(pivot_2)
    adj[pivot_2].discard(pivot_1)
    seen, stack = {pivot_2}, [pivot_2]
    while stack:
        cur = stack.pop()
        for nb in adj[cur]:
            if nb not in seen:
                seen.add(nb)
                stack.append(nb)
    if pivot_1 in seen:
        raise ValueError(f'The pivot bond {pivot_1}-{pivot_2} is part of a ring; '
                         f'a 1D rotor scan is ill-defined.')
    return sorted(int(x) for x in seen)


def _scan_walk(atoms: Atoms, torsion: list, step_deg: float, nsteps: int, top: list,
               fmax: float, steps: int, direction: int, optimizer) -> tuple:
    """
    Perform one directional sequential relaxed torsional walk.

    Each grid point starts from the previous point's relaxed geometry (a hysteretic walk), so a
    coupled coordinate can carry the walk into a different conformer and never return - which is
    why a full scan runs this in both directions and keeps the lower branch at each point.

    Args:
        atoms (Atoms): The molecule with a calculator attached (not modified; a copy is walked).
        torsion (list): The 0-indexed four atoms (i, j, k, l) defining the scanned dihedral.
        step_deg (float): The dihedral increment in degrees.
        nsteps (int): The number of increments (the walk visits ``nsteps`` + 1 points).
        top (list): The 0-indexed rotating group.
        fmax (float): The force convergence criterion in eV/Angstrom.
        steps (int): The maximum optimizer steps per point.
        direction (int): +1 for a 0->360 walk, -1 for a 360->0 walk.
        optimizer: The ASE optimizer class to relax each point.

    Returns:
        tuple: (energies in eV, residual max-forces in eV/Angstrom, relaxed Cartesian coordinates),
        each of length ``nsteps`` + 1.
    """
    i, j, k, l = torsion
    work = atoms.copy()
    work.calc = atoms.calc
    energies, residual_forces, coords = list(), list(), list()
    for n in range(nsteps + 1):
        if n:
            work.rotate_dihedral(i, j, k, l, angle=direction * step_deg, indices=top)
        target = work.get_dihedral(i, j, k, l)
        work.set_constraint(FixInternals(dihedrals_deg=[[target, [i, j, k, l]]]))
        opt = optimizer(work, logfile=None)
        opt.run(fmax=fmax, steps=steps)
        # Residual force WITH the dihedral constraint still applied is the convergence criterion:
        # reading opt.converged() after clearing the constraint re-evaluates the free torsional
        # force, which is nonzero everywhere except the stationary points.
        residual = float(np.sqrt((work.get_forces() ** 2).sum(axis=1)).max())
        work.set_constraint()
        energies.append(float(work.get_potential_energy()))
        residual_forces.append(residual)
        coords.append(tuple(map(tuple, work.get_positions().tolist())))
    return energies, residual_forces, coords


def merge_scan_branches(e_f: list, e_b: list, nsteps: int, coords_f: list = None,
                        coords_b: list = None) -> tuple:
    """
    Fold a forward and a backward torsional walk onto a common grid and keep the lower branch.

    The forward walk visits phi = +n * step and the backward walk phi = -n * step, so index ``n``
    of the backward walk lands on grid point ``(-n) % nsteps``. Both walks also revisit their
    starting point at index ``nsteps``, which folds back onto grid point 0. At each grid point the
    lower of the two energies wins, which is what recovers the lowest torsional path when one walk
    has been carried into a different conformer by a coupled coordinate.

    Args:
        e_f (list): The forward walk energies, of length ``nsteps`` + 1.
        e_b (list): The backward walk energies, of length ``nsteps`` + 1.
        nsteps (int): The number of increments; there are ``nsteps`` unique grid points.
        coords_f (list, optional): The forward walk Cartesian coordinates, aligned with ``e_f``.
        coords_b (list, optional): The backward walk Cartesian coordinates, aligned with ``e_b``.

    Returns:
        tuple: (the merged per-grid-point energies as a numpy array of length ``nsteps``, the
        forward grid, the backward grid, and the winning coordinates per grid point - the last
        being None if either coordinate list was not supplied).
    """
    grid_f, grid_b = np.full(nsteps, np.inf), np.full(nsteps, np.inf)
    xyz_f, xyz_b = [None] * nsteps, [None] * nsteps
    for n in range(nsteps + 1):
        p = n % nsteps  # forward:  phi = +n*step
        if e_f[n] < grid_f[p]:
            grid_f[p] = e_f[n]
            if coords_f is not None:
                xyz_f[p] = coords_f[n]
        q = (-n) % nsteps  # backward: phi = -n*step
        if e_b[n] < grid_b[q]:
            grid_b[q] = e_b[n]
            if coords_b is not None:
                xyz_b[q] = coords_b[n]
    merged = np.minimum(grid_f, grid_b)
    coords = None
    if coords_f is not None and coords_b is not None:
        coords = [xyz_f[p] if grid_f[p] <= grid_b[p] else xyz_b[p] for p in range(nsteps)]
    return merged, grid_f, grid_b, coords


def relaxed_torsion_scan(atoms: Atoms, torsion: list, step_deg: float, nsteps: int,
                         top: list = None, fmax: float = 0.005, steps: int = 500,
                         optimizer=LBFGS) -> dict:
    """
    Perform a full 1D relaxed torsional scan in a single process.

    The torsion is swept 0->360 and 360->0 from the same starting geometry, and the lower energy
    of the two walks is kept at each grid point. A single directional walk is path-dependent: if a
    coupled coordinate flips partway round, that walk finishes in a different conformer and its
    curve is not a torsional potential. Walking both ways and taking the pointwise minimum recovers
    the lowest torsional path, which is the one the downstream statistical mechanics wants.

    Points that do not fully converge are kept (never dropped), so the periodic grid the downstream
    Fourier fit reads is always complete; ``fmax_worst`` reports the largest residual force.

    Args:
        atoms (Atoms): The molecule with a calculator attached.
        torsion (list): The 0-indexed four atoms (i, j, k, l) defining the scanned dihedral.
        step_deg (float): The dihedral increment in degrees.
        nsteps (int): The number of increments; the returned grid has ``nsteps`` + 1 points.
        top (list, optional): The 0-indexed rotating group; determined from the graph if not given.
        fmax (float): The force convergence criterion in eV/Angstrom.
        steps (int): The maximum optimizer steps per point.
        optimizer: The ASE optimizer class to relax each point.

    Raises:
        ValueError: If ``nsteps`` is less than 1, ``step_deg`` is not positive, or the two do not
                    together span a full 360 degree revolution.

    Returns:
        dict: ``energies`` (absolute, in eV, ``nsteps`` + 1 points with the endpoint duplicating the
        start), ``angles`` (degrees, 0..360), ``scan_coords`` (the relaxed geometry at each point),
        ``top``, and convergence diagnostics.
    """
    if nsteps < 1:
        raise ValueError(f'A relaxed torsion scan needs at least one increment, but got nsteps={nsteps}.')
    if step_deg <= 0:
        raise ValueError(f'The dihedral increment must be positive, but got step_deg={step_deg}.')
    # The grid folding below (n % nsteps) and the duplicated endpoint both assume the walk closes on
    # itself. ARC guarantees this via divmod(360, scan_res) in check_argument_consistency, but this
    # function is public and runs in a separate environment where that guard does not apply.
    if abs(nsteps * step_deg - 360.0) > 1e-6:
        raise ValueError(f'A relaxed torsion scan must span a full revolution, but nsteps={nsteps} '
                         f'increments of step_deg={step_deg} span {nsteps * step_deg} degrees.')
    i, j, k, l = torsion
    if top is None:
        top = rotor_top(atoms, j, k)

    e_f, r_f, c_f = _scan_walk(atoms, torsion, step_deg, nsteps, top, fmax, steps, +1, optimizer)
    e_b, r_b, c_b = _scan_walk(atoms, torsion, step_deg, nsteps, top, fmax, steps, -1, optimizer)

    merged, grid_f, grid_b, coords = merge_scan_branches(e_f, e_b, nsteps, c_f, c_b)
    energies = merged.tolist()
    energies.append(energies[0])  # duplicate endpoint: 46-point protocol
    scan_coords = list(coords)
    scan_coords.append(scan_coords[0])
    angles = [n * step_deg for n in range(nsteps + 1)]
    return {
        'energies': energies,
        'angles': angles,
        'scan_coords': scan_coords,
        'top': top,
        'branch_gap_max': float(np.abs(grid_f - grid_b).max()),
        'fmax_worst': float(max(r_f + r_b)),
        'converged': bool(max(r_f + r_b) <= fmax),
    }


def run_torsion_scan(atoms: Atoms, input_dict: dict, settings: dict) -> dict:
    """
    Run a 1D relaxed torsional scan for an ARC ``scan`` job and shape it for ARC's YAML parser.

    Args:
        atoms (Atoms): The molecule with a calculator attached.
        input_dict (dict): The job input; ``torsions`` (a list holding one 0-indexed torsion) and
                           ``scan_res`` (the dihedral increment in degrees) are read here.
        settings (dict): ASE run settings; optional ``fmax``, ``steps``, ``optimizer`` override the
                         relaxed-scan defaults (LBFGS, fmax 0.005 eV/Angstrom, 500 steps per point).

    Raises:
        ValueError: If no torsion is supplied or more than one is (a 1D scan sweeps a single
                    torsion), or if ``scan_res`` is not positive or does not divide 360 evenly.

    Returns:
        dict: ``energies`` (Hartree), ``angles`` (degrees) and ``scan_coords`` (an xyz dict per
        point) for ARC's YAML scan parser, plus the rotating ``top`` and convergence diagnostics.
    """
    torsions = input_dict.get('torsions')
    if not torsions:
        raise ValueError("A 'scan' job requires a torsion, but none was supplied.")
    if len(torsions) > 1:
        raise ValueError(f"A 1D 'scan' job sweeps a single torsion, but got {len(torsions)}: {torsions}.")
    torsion = list(torsions[0])
    scan_res = float(input_dict.get('scan_res', 8.0))
    if scan_res <= 0:
        raise ValueError(f"The scan resolution must be positive, but got scan_res={scan_res}.")
    if divmod(360.0, scan_res)[1]:
        raise ValueError(f"The scan resolution must divide 360 evenly, but got scan_res={scan_res}.")
    nsteps = int(round(360.0 / scan_res))
    fmax = float(settings.get('fmax', 0.005))
    steps = int(settings.get('steps', 500))
    engine_dict = {'bfgs': BFGS, 'lbfgs': LBFGS, 'gpmin': GPMin,
                   'scipyfminbfgs': SciPyFminBFGS, 'scipyfmincg': SciPyFminCG}
    optimizer = engine_dict.get(str(settings.get('optimizer', 'LBFGS')).lower(), LBFGS)

    result = relaxed_torsion_scan(atoms, torsion, scan_res, nsteps, top=None,
                                  fmax=fmax, steps=steps, optimizer=optimizer)
    # ARC's YAML scan parser reads absolute energies and subtracts the minimum before converting
    # Hartree->kJ/mol, so the energies must be in Hartree.
    result['energies'] = [energy * e / E_h for energy in result['energies']]
    # Shape the relaxed geometries as ARC xyz dicts so parse_1d_scan_coords can read them back;
    # trsh.scan_quality_check needs them to emit a 'change conformer' action, and the scheduler
    # passes them in to check that a TS survived the rotation.
    xyz = input_dict.get('xyz') or dict()
    symbols = xyz.get('symbols') or tuple(atoms.get_chemical_symbols())
    isotopes = xyz.get('isotopes') or tuple([None] * len(symbols))
    result['scan_coords'] = [{'symbols': symbols, 'isotopes': isotopes, 'coords': coords}
                             for coords in result['scan_coords']]
    return result


def scan_convergence_warning(result: dict, fmax: float, branch_gap_tolerance: float = 0.05) -> str | None:
    """
    Describe what is wrong with a completed torsional scan, if anything.

    Two things make a scan untrustworthy without making it fail: points that never reached the
    force criterion, and a large disagreement between the forward and backward branches (which
    means the two walks explored genuinely different conformers rather than one torsional path).

    Args:
        result (dict): A completed scan result, carrying ``converged``, ``fmax_worst`` and
                       ``branch_gap_max``.
        fmax (float): The force convergence criterion that was requested, in eV/Angstrom.
        branch_gap_tolerance (float): The forward/backward branch disagreement, in eV, above which
                                      the scan is called out.

    Returns:
        str | None: A human-readable warning, or None if the scan is clean.
    """
    issues = list()
    if not result.get('converged', True):
        issues.append(f"the worst point relaxed only to {result.get('fmax_worst'):.4g} eV/Angstrom "
                      f"against a requested fmax of {fmax:.4g}")
    branch_gap = result.get('branch_gap_max')
    if branch_gap is not None and branch_gap > branch_gap_tolerance:
        issues.append(f"the forward and backward branches disagree by up to {branch_gap:.4g} eV, "
                      f"so they likely relaxed into different conformers")
    if not issues:
        return None
    return f"The torsional scan completed but {' and '.join(issues)}."


def is_linear(atoms: Atoms) -> bool:
    """
    Determine whether an Atoms object represents a linear molecule.
    """
    coordinates = atoms.get_positions()
    n_atoms = len(coordinates)
    if n_atoms <= 1:
        return False
    if n_atoms == 2:
        return True

    for i in range(1, n_atoms - 1):
        v1 = coordinates[i - 1] - coordinates[i]
        v2 = coordinates[i + 1] - coordinates[i]
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            continue
        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = math.degrees(math.acos(cos_angle))
        if not ((180.0 - 0.1 < angle <= 180.0) or (0.0 <= angle < 0.1)):
            return False
    return True


def numpy_vibrational_analysis(masses: np.ndarray, hessian: np.ndarray, is_linear: bool = False):
    """
    Computing vibrational wavenumbers, modes, reduced masses, and force constants from Hessian.
    NumPy implementation following physical constants and ASE units.
    Logic follows TorchANI and ASE VibrationsData standards.
    
    Args:
        masses: (n_atoms,) array of atomic masses in AMU.
        hessian: (3*n_atoms, 3*n_atoms) array in eV/A^2.
        
    Returns:
        dict: Containing freqs, modes, force_constants, reduced_masses.
    """
    # 1. Mass-weighted Hessian
    # inv_sqrt_mass: (3*n_atoms,)
    inv_sqrt_mass = (1.0 / np.sqrt(masses)).repeat(3)
    # H_mw = M^-1/2 * H * M^-1/2
    mass_scaled_hessian = hessian * inv_sqrt_mass[:, np.newaxis] * inv_sqrt_mass[np.newaxis, :]
    
    # 2. Diagonalize
    eigenvalues, eigenvectors = np.linalg.eigh(mass_scaled_hessian)
    
    # 3. Frequencies (cm^-1)
    # Factor to convert sqrt(eV / (A^2 * AMU)) to cm^-1
    # nu = 1/(2*pi*c*100) * sqrt(e * 10^20 / amu)
    freq_factor = (1.0 / (2.0 * pi * c * 100.0)) * np.sqrt((e * 1.0e20) / amu)
    
    freqs = []
    for eig in eigenvalues:
        if eig >= 0:
            f = freq_factor * np.sqrt(eig)
        else:
            # ARC convention: imaginary frequencies are represented as negative real numbers
            f = -freq_factor * np.sqrt(-eig)
        freqs.append(float(f))
    
    # 4. Normal Modes (MDU: Mass Deweighted Unnormalized in TorchANI / Standard in ASE)
    # These modes are normalized such that sum_i m_i * |v_i|^2 = 1
    # eigenvectors.T has modes as rows
    mw_normalized = eigenvectors.T
    md_unnormalized = mw_normalized * inv_sqrt_mass[np.newaxis, :]
    
    # 5. Reduced Masses (AMU)
    # Formula from ASE/TorchANI: mu_n = 1 / sum_i |v_{n,i}|^2
    # where v are the mass-weighted normalized modes calculated above.
    norm_sq = np.sum(np.square(np.abs(md_unnormalized)), axis=1)
    rmasses = 1.0 / norm_sq
    
    # 6. Force Constants (mDyne/A)
    # k_n = mu_n * omega_n^2
    # Conversion factor from eV/A^2 to mDyne/A is e * 10^-2 * 10^20 = e * 10^18 ?
    # 1 eV/A^2 = 16.021766 N/m = 0.16021766 mDyne/A
    # eigenvalue (eV/(A^2*AMU)) * rmass (AMU) = k (eV/A^2)
    fconst_factor = e * 1.0e18
    fconstants = eigenvalues * rmasses * fconst_factor
    
    # MDN modes (Mass Deweighted Normalized) for output
    # normalized such that sum_i |v_i|^2 = 1
    norm_factors = 1.0 / np.sqrt(norm_sq)
    md_normalized = md_unnormalized * norm_factors[:, np.newaxis]
    
    # Filter out translations and rotations (first 6 modes for non-linear, 5 for linear)
    # Most ESS only report 3N-6 / 3N-5 modes.
    # We'll filter modes with very small magnitude if they are in the first 6.
    # Sorting by magnitude ensures we catch the smallest ones.
    indices = np.argsort(np.abs(freqs))
    
    # Threshold for considering a mode as a translation/rotation (cm^-1)
    rot_trans_threshold = 50.0
    
    if len(masses) == 1:
        num_to_filter = 3
    elif len(masses) == 2:
        num_to_filter = 5
    else:
        num_to_filter = 5 if is_linear else 6
    filtered_indices = []
    for i in range(len(freqs)):
        if i < num_to_filter and abs(freqs[indices[i]]) < rot_trans_threshold:
            continue
        filtered_indices.append(indices[i])
    
    # Sort back the remaining indices by their original order (which is by eigenvalue)
    # but we'll return them sorted by frequency value (imaginary first, then increasing real)
    final_indices = sorted(filtered_indices, key=lambda i: freqs[i])

    return {
        'schema_version': SCHEMA_VERSION,
        'freqs': [freqs[i] for i in final_indices],
        'modes': md_normalized[final_indices].reshape(len(final_indices), -1, 3).tolist(),
        'force_constants': [fconstants[i].tolist() for i in final_indices],
        'reduced_masses': [rmasses[i].tolist() for i in final_indices],
        'hessian': hessian.tolist()
    }


def run_vibrational_analysis(atoms: Atoms, settings: dict):
    """
    Perform vibrational analysis and return frequencies, modes, and other properties.
    """
    if settings.get('calculator', '').lower() == 'torchani':
        try:
            import torch
            import torchani
            device = torch.device(settings.get('device', 'cpu'))
            model_name = settings.get('model', 'ANI2x')
            if model_name.lower() == 'ani1ccx':
                model = torchani.models.ANI1ccx(periodic_table_index=True).to(device)
            elif model_name.lower() == 'ani1x':
                model = torchani.models.ANI1x(periodic_table_index=True).to(device)
            else:
                model = torchani.models.ANI2x(periodic_table_index=True).to(device)
            
            species = torch.tensor(atoms.get_atomic_numbers(), device=device, dtype=torch.long).unsqueeze(0)
            coordinates = torch.from_numpy(atoms.get_positions()).unsqueeze(0).requires_grad_(True)
            masses = torchani.utils.get_atomic_masses(species)
            energies = model.double()((species, coordinates)).energies
            hessian = torchani.utils.hessian(coordinates, energies=energies)
            freqs, modes, force_constants, reduced_masses = torchani.utils.vibrational_analysis(masses, hessian, mode_type='MDN')
            
            return {
                'schema_version': SCHEMA_VERSION,
                'freqs': (freqs.cpu().numpy().tolist() if hasattr(freqs, 'cpu') else freqs.numpy().tolist()),
                'hessian': hessian.cpu().numpy().tolist() if hasattr(hessian, 'cpu') else hessian.tolist(),
                'modes': modes.cpu().numpy().tolist() if hasattr(modes, 'cpu') else modes.tolist(),
                'force_constants': force_constants.cpu().numpy().tolist() if hasattr(force_constants, 'cpu') else force_constants.tolist(),
                'reduced_masses': reduced_masses.cpu().numpy().tolist() if hasattr(reduced_masses, 'cpu') else reduced_masses.tolist()
            }
        except Exception as exc:
            print(f'TorchANI vibrational analysis failed, falling back to ASE: {exc}')

    vib = Vibrations(atoms, name='vib_tmp', nfree=4)
    vib.run()
    vib_data = vib.get_vibrations()
    try:
        hessian = vib_data.get_hessian_2d()
    except AttributeError:
        hessian = vib_data.get_hessian()
        if len(hessian.shape) == 4:
            n_atoms = hessian.shape[0]
            hessian = hessian.reshape(3 * n_atoms, 3 * n_atoms)
    masses = atoms.get_masses()
    vib.clean()
    is_lin = is_linear(atoms)
    return numpy_vibrational_analysis(masses, hessian, is_linear=is_lin)


def main():
    """
    Main execution logic.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--yml_path', type=str, default='input.yml')
    args = parser.parse_args()
    
    input_path = os.path.abspath(args.yml_path)
    if os.path.isdir(input_path):
        input_path = os.path.join(input_path, 'input.yml')
    
    try:
        input_dict = read_yaml_file(input_path)
    except Exception as exc:
        print(f"Error reading input file: {exc}")
        return

    job_type = input_dict.get('job_type')
    xyz = input_dict.get('xyz')
    settings = input_dict.get('settings', {})
    charge = input_dict.get('charge', 0)
    multiplicity = input_dict.get('multiplicity', 1)
    is_ts = input_dict.get('is_ts', False)

    atoms = Atoms(symbols=xyz['symbols'], positions=xyz['coords'])
    atoms.info.update({'charge': charge, 'spin': multiplicity})  # UMA (omol) conditions on these
    calc = get_calculator(settings, charge, multiplicity)
    atoms.calc = calc
    
    apply_constraints(atoms, input_dict.get('constraints'))
    
    output = {'schema_version': SCHEMA_VERSION}
    
    def save_current_geometry(out_dict, atoms_obj, input_xyz):
        out_dict['opt_xyz'] = {
            'coords': tuple(map(tuple, atoms_obj.get_positions().tolist())),
            'symbols': input_xyz['symbols'],
            'isotopes': input_xyz.get('isotopes') or tuple([None] * len(input_xyz['symbols']))
        }

    if job_type == 'sp':
        output['sp'] = to_kJmol(atoms.get_potential_energy())

    if job_type == 'scan':
        try:
            output.update(run_torsion_scan(atoms, input_dict, settings))
            warning = scan_convergence_warning(output, float(settings.get('fmax', 0.005)))
            if warning is not None:
                # The scan is kept and reported: a loose point still carries a usable potential,
                # but a silent 'clean success' would hide a scan where every point hit the step
                # limit. The adapter logs this without failing the job.
                output['warnings'] = [warning]
        except Exception as exc:
            output['success'] = False
            output['error'] = f"Torsion scan failed: {exc}"

    if job_type in ['opt', 'conf_opt', 'optfreq', 'directed_scan']:
        fmax = float(settings.get('fmax', 0.001))
        steps = int(settings.get('steps', 1000))
        engine_name = settings.get('optimizer', 'BFGS').lower()
        
        engine_dict = {
            'bfgs': BFGS, 'lbfgs': LBFGS, 'gpmin': GPMin,
            'scipyfminbfgs': SciPyFminBFGS, 'scipyfmincg': SciPyFminCG,
            'sella': None,
        }
        logfile = os.path.join(os.path.dirname(input_path), 'opt.log')
        if is_ts or engine_name == 'sella':
            # A TS search needs a saddle-point optimizer; UMA ships none, so use Sella.
            from sella import Sella
            opt_class = Sella
            opt = opt_class(atoms, order=1 if is_ts else 0, logfile=logfile)
        else:
            opt_class = engine_dict.get(engine_name, BFGS)
            opt = opt_class(atoms, logfile=logfile)

        try:
            opt.run(fmax=fmax, steps=steps)
            save_current_geometry(output, atoms, xyz)
            output['sp'] = to_kJmol(atoms.get_potential_energy())
        except Exception as exc:
            output['error'] = f"Optimization failed: {exc}"
            save_current_geometry(output, atoms, xyz)
    else:
        # For non-optimization jobs, still save the geometry
        save_current_geometry(output, atoms, xyz)

    if job_type == 'irc':
        from sella import IRC
        from ase.io import read
        fmax = float(settings.get('fmax', 0.001))
        steps = int(settings.get('steps', 1000))
        direction = input_dict.get('irc_direction', 'forward')
        traj_path = os.path.join(os.path.dirname(input_path), 'irc.traj')
        try:
            irc = IRC(atoms, logfile=os.path.join(os.path.dirname(input_path), 'irc.log'),
                      trajectory=traj_path)
            irc.run(fmax=fmax, steps=steps, direction=direction)
            images = read(traj_path, index=':')
            output['irc_traj'] = [
                {'coords': tuple(map(tuple, image.get_positions().tolist())),
                 'symbols': xyz['symbols'],
                 'isotopes': xyz.get('isotopes') or tuple([None] * len(xyz['symbols']))}
                for image in images]
        except Exception as exc:
            output['error'] = f"IRC failed: {exc}"

    if job_type in ['freq', 'optfreq']:
        try:
            freq_results = run_vibrational_analysis(atoms, settings)
            output.update(freq_results)
        except Exception as exc:
            output['error'] = output.get('error', '') + f" Frequency calculation failed: {exc}"

    output_path = os.path.join(os.path.dirname(input_path), 'output.yml')
    save_yaml_file(output_path, output)


if __name__ == '__main__':
    main()
