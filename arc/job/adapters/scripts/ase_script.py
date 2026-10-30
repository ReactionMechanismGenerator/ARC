#!/usr/bin/env python3
# encoding: utf-8

"""
A standalone script to run ASE (Atomic Simulation Environment) jobs.
Standardizes interaction with various calculators.
"""

import argparse
import json
import math
import os
import signal
import socket
import sys
import threading
import time
import traceback
import uuid
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
    A plain float is returned, since a NumPy scalar (as some calculators return) is written to
    output.yml with a Python tag that ARC cannot load.
    """
    return float(energy_ev * e * N_A / 1000.0)


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


def load_torchani_model(model_name: str, device, cache: dict | None = None, double: bool = False):
    """
    Load a TorchANI model, reusing it from ``cache`` if given (the persistent worker's model cache).
    A double-precision copy is cached separately, so a frequency job cannot change the precision
    of a later job's model.
    """
    key = ('torchani', model_name.lower(), str(device), double)
    if cache is not None and key in cache:
        return cache[key]
    import torchani
    if model_name.lower() == 'ani1ccx':
        model = torchani.models.ANI1ccx(periodic_table_index=True).to(device)
    elif model_name.lower() == 'ani1x':
        model = torchani.models.ANI1x(periodic_table_index=True).to(device)
    else:
        model = torchani.models.ANI2x(periodic_table_index=True).to(device)
    if double:
        model = model.double()
    if cache is not None:
        cache[key] = model
    return model


def uma_inference_settings(calc_config: dict, hessian: bool = False) -> tuple:
    """
    Resolve the fairchem inference settings of a UMA job.

    ``inference_settings`` is a fairchem preset name or a dict of ``InferenceSettings`` fields. fairchem's
    default preset differs between versions, and some presets are reduced precision on a GPU.

    A Hessian predictor always runs at full precision: no TF32, unmerged, uncompiled and without activation
    checkpointing. Of ``inference_settings`` it keeps only ``base_precision_dtype``. fairchem warns that
    reduced precision degrades second derivatives, and a merged or compiled model would be rebuilt for every
    new species anyway. FP32 is enough: its frequencies match FP64 to within 0.01 cm^-1.
    ``hessian_vmap`` (default False) vectorizes the backward passes. That is faster for small molecules on
    a CPU, but its memory grows steeply (about 13 GB in FP32 for 32 atoms).

    Args:
        calc_config (dict): The run settings.
        hessian (bool): Whether to resolve the settings of the Hessian predictor.

    Returns:
        tuple: The settings (a preset name, a dict of ``InferenceSettings`` fields, or ``None`` for fairchem's
               default) and a hashable key for the model cache.
    """
    spec = calc_config.get('inference_settings')
    if hessian:
        dtype = spec.get('base_precision_dtype', 'float32') if isinstance(spec, dict) else 'float32'
        vmap = bool(calc_config.get('hessian_vmap', False))
        task = calc_config.get('task', 'omol')
        settings = {'tf32': False, 'activation_checkpointing': False, 'merge_mole': False, 'compile': False,
                    'base_precision_dtype': dtype, 'hessian_vmap': vmap, 'predict_untrained_hessian': [task]}
        return settings, ('hessian', dtype, vmap, task)
    if isinstance(spec, dict):
        return dict(spec), json.dumps(spec, sort_keys=True)
    return spec, spec


def load_uma_predictor(calc_config: dict, cache: dict | None = None, hessian: bool = False):
    """
    Load a UMA predictor, reusing it from ``cache`` if given (the persistent worker's model cache).

    Args:
        calc_config (dict): The run settings.
        cache (dict, optional): Loaded models, kept across jobs by a persistent worker.
        hessian (bool): Whether to load a predictor that also computes the Hessian (see ``uma_hessian()``).

    Raises:
        NotImplementedError: If a Hessian is requested from a fairchem without autograd Hessians (before 2.15).
    """
    from fairchem.core import pretrained_mlip
    model = calc_config.get('model', 'uma-s-1p2')
    device = calc_config.get('device', 'cpu')
    settings, settings_key = uma_inference_settings(calc_config, hessian=hessian)
    key = ('uma', model, device, settings_key)
    predictor = cache.get(key) if cache is not None else None
    if predictor is None:
        predict_kwargs = {'device': device}
        if isinstance(settings, dict):
            from fairchem.core.units.mlip_unit.api.inference import InferenceSettings
            if hessian and 'predict_untrained_hessian' not in getattr(InferenceSettings, '__dataclass_fields__', {}):
                raise NotImplementedError('The installed fairchem cannot compute autograd Hessians')
            settings = InferenceSettings(**{k: set(v) if k.startswith('predict_untrained_') else v
                                            for k, v in settings.items()})
        if settings:
            predict_kwargs['inference_settings'] = settings
        predictor = pretrained_mlip.get_predict_unit(model, **predict_kwargs)
        if cache is not None:
            cache[key] = predictor
    return predictor


def uma_hessian(atoms: Atoms, settings: dict, cache: dict | None = None) -> np.ndarray | None:
    """
    Compute a UMA model's Hessian by automatic differentiation.

    It uses a predictor of its own. With the Hessian enabled, fairchem computes it on every call, and
    builds the double-backward graph for every force call too, which would slow optimizations and scans.

    Args:
        atoms (Atoms): The structure, with its charge and spin in ``atoms.info``.
        settings (dict): The run settings.
        cache (dict, optional): Loaded models, kept across jobs by a persistent worker.

    Returns:
        np.ndarray | None: The (3N, 3N) Hessian in eV/A^2, or ``None`` if the installed fairchem cannot
                           compute one.
    """
    from fairchem.core import FAIRChemCalculator
    try:
        predictor = load_uma_predictor(settings, cache=cache, hessian=True)
    except NotImplementedError as exc:
        print(f'{exc}; using finite differences instead.')
        return None
    hessian_atoms = atoms.copy()
    hessian_atoms.calc = FAIRChemCalculator(predictor, task_name=settings.get('task', 'omol'))
    n_coords = 3 * len(atoms)
    return np.asarray(hessian_atoms.calc.get_property('hessian', hessian_atoms), dtype=float).reshape(n_coords, n_coords)


def sella_hessian_function(settings: dict, cache: dict | None = None):
    """
    The exact Hessian for a Sella saddle-point search, if the calculator gives one cheaply (UMA's autograd Hessian).

    Without it, Sella estimates the lowest-curvature mode iteratively from a random start. At a TS guess with a
    second negative mode (e.g., a torsion), the estimate can lock onto the wrong one, and Sella then climbs to the
    wrong saddle. From ARC's heuristic guess for OH + C3H8, plain Sella reached a saddle of the OH...propane complex
    (one 33i cm^-1 torsion) in both of ARC's attempts and in 1 of 10 seeded runs; with the exact Hessian, all 10
    runs reached the H-abstraction TS.

    Args:
        settings (dict): The run settings.
        cache (dict, optional): Loaded models, kept across jobs by a persistent worker.

    Returns:
        Callable | None: A function of the atoms returning the (3N, 3N) Hessian in eV/A^2, or ``None`` if the
                         calculator has no autograd Hessian (Sella then estimates the Hessian as before).
    """
    if settings.get('calculator', '').lower() not in ('uma', 'fairchem') \
            or settings.get('hessian', 'autograd') != 'autograd':
        return None
    try:
        load_uma_predictor(settings, cache=cache, hessian=True)
    except NotImplementedError:
        return None

    def hessian_function(atoms: Atoms) -> np.ndarray:
        return uma_hessian(atoms, settings, cache)

    return hessian_function


def get_calculator(calc_config: dict, charge: int = 0, multiplicity: int = 1, cache: dict | None = None,
                   hessian: bool = False):
    """
    Initialize the ASE calculator based on settings.

    Args:
        calc_config (dict): The run settings.
        charge (int): The total charge.
        multiplicity (int): The spin multiplicity.
        cache (dict, optional): Loaded models, kept across jobs by a persistent worker. A new calculator
                                is still built for every job, so no results are carried over between jobs.
        hessian (bool): Whether the calculator only serves an autograd Hessian. UMA then loads its Hessian
                        predictor (see uma_hessian()) instead of the ordinary one, if fairchem supports it.
    """
    name = calc_config.get('calculator', '').lower()
    kwargs = calc_config.get('calculator_kwargs', {})
    
    if name == 'torchani':
        import torch
        model_name = calc_config.get('model', 'ANI2x')
        device = torch.device(calc_config.get('device', 'cpu'))
        return load_torchani_model(model_name, device, cache).ase()
    
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
        # ``inference_settings`` optionally sets fairchem's precision (see uma_inference_settings()).
        from fairchem.core import FAIRChemCalculator
        try:
            predictor = load_uma_predictor(calc_config, cache=cache, hessian=hessian)
        except NotImplementedError:
            predictor = load_uma_predictor(calc_config, cache=cache)
        return FAIRChemCalculator(predictor, task_name=calc_config.get('task', 'omol'))


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

    ARC's constraint atom indices are 1-indexed (that is what Gaussian's modredundant section and
    xTB's $constrain block consume); ASE's FixInternals is 0-indexed.
    """
    if not constraints_data:
        return
    bonds, angles, dihedrals = list(), list(), list()
    for constraint in constraints_data:
        indices = [index - 1 for index in constraint[0]]
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


def internal_basis(masses: np.ndarray, positions: np.ndarray, is_linear: bool = False) -> np.ndarray:
    """
    An orthonormal basis of the internal coordinates, in mass-weighted Cartesian coordinates: the complement
    of the translations and the (Eckart) rotations about the center of mass.

    Args:
        masses: (n_atoms,) array of atomic masses in AMU.
        positions: (n_atoms, 3) array of Cartesian coordinates in Angstrom.
        is_linear: Whether the molecule is linear (it then has two rotations, not three).

    Returns:
        np.ndarray: A (3*n_atoms, 3*n_atoms - k) array, with k = 3 for an atom, 5 for a linear molecule,
                    and 6 otherwise.
    """
    n_atoms = len(masses)
    n_rigid = 3 if n_atoms == 1 else 5 if is_linear or n_atoms == 2 else 6
    sqrt_mass = np.sqrt(masses)[:, np.newaxis]
    centered = positions - (masses[:, np.newaxis] * positions).sum(axis=0) / masses.sum()
    rigid = []
    for axis in np.eye(3):
        rigid.append((np.tile(axis, (n_atoms, 1)) * sqrt_mass).ravel())
        rigid.append((np.cross(axis, centered) * sqrt_mass).ravel())
    # The first n_rigid left singular vectors span the rigid motions, so the rest span their complement.
    # Taking n_rigid from is_linear rather than from the singular values keeps a nearly linear molecule
    # (within is_linear()'s tolerance) from losing a real bend to a near-zero rotation.
    u, _, _ = np.linalg.svd(np.array(rigid).T, full_matrices=True)
    return u[:, n_rigid:]


def numpy_vibrational_analysis(masses: np.ndarray, hessian: np.ndarray, is_linear: bool = False,
                               positions: np.ndarray | None = None):
    """
    Computing vibrational wavenumbers, modes, reduced masses, and force constants from Hessian.
    NumPy implementation following physical constants and ASE units.
    Logic follows TorchANI and ASE VibrationsData standards.

    Args:
        masses: (n_atoms,) array of atomic masses in AMU.
        hessian: (3*n_atoms, 3*n_atoms) array in eV/A^2.
        is_linear: Whether the molecule is linear.
        positions: (n_atoms, 3) array of Cartesian coordinates in Angstrom. If given, translations and
                   rotations are projected out exactly, and exactly 3N-6 (3N-5) modes are returned.
                   Otherwise, the up to 6 (5) smallest modes are dropped if they are below 50 cm^-1, which
                   keeps a spurious rotation above 50 cm^-1, such as a 2-point finite-difference Hessian
                   gives, as a vibration.

    Returns:
        dict: Containing freqs, modes, force_constants, reduced_masses.
    """
    # 1. Mass-weighted Hessian
    # inv_sqrt_mass: (3*n_atoms,)
    inv_sqrt_mass = (1.0 / np.sqrt(masses)).repeat(3)
    # H_mw = M^-1/2 * H * M^-1/2
    mass_scaled_hessian = hessian * inv_sqrt_mass[:, np.newaxis] * inv_sqrt_mass[np.newaxis, :]

    # 2. Diagonalize, within the internal coordinates if the geometry is known
    if positions is None:
        eigenvalues, eigenvectors = np.linalg.eigh(mass_scaled_hessian)
    else:
        basis = internal_basis(np.asarray(masses, dtype=float), np.asarray(positions, dtype=float), is_linear)
        eigenvalues, internal_vectors = np.linalg.eigh(basis.T @ mass_scaled_hessian @ basis)
        eigenvectors = basis @ internal_vectors
    
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
    
    if positions is not None:
        # Translations and rotations were projected out, so every mode left is a vibration.
        filtered_indices = list(range(len(freqs)))
    else:
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
        'modes': md_normalized[final_indices].reshape(len(final_indices), len(masses), 3).tolist(),
        'force_constants': [fconstants[i].tolist() for i in final_indices],
        'reduced_masses': [rmasses[i].tolist() for i in final_indices],
        'hessian': hessian.tolist()
    }


def run_vibrational_analysis(atoms: Atoms, settings: dict, work_dir: str | None = None, cache: dict | None = None):
    """
    Perform vibrational analysis and return frequencies, modes, and other properties.

    TorchANI and UMA Hessians are computed by automatic differentiation, and all others by finite differences
    of the forces. For UMA, ``settings['hessian']`` can be ``'autograd'`` (the default) or
    ``'finite_difference'``. An older fairchem without autograd Hessians falls back to finite differences.

    Args:
        atoms (Atoms): The structure, with its calculator attached.
        settings (dict): The run settings.
        work_dir (str, optional): The job's directory, which holds the finite-difference cache.
                                  Defaults to the current working directory.
        cache (dict, optional): Loaded models, kept across jobs by a persistent worker.
    """
    hessian_choice = settings.get('hessian', 'autograd')
    if hessian_choice not in ('autograd', 'finite_difference'):
        raise ValueError(f"settings['hessian'] must be 'autograd' or 'finite_difference', got {hessian_choice!r}")
    atom_masses, positions, is_lin = atoms.get_masses(), atoms.get_positions(), is_linear(atoms)

    if settings.get('calculator', '').lower() in ('uma', 'fairchem') and hessian_choice == 'autograd':
        hessian = uma_hessian(atoms, settings, cache)
        if hessian is not None:
            results = numpy_vibrational_analysis(atom_masses, hessian, is_linear=is_lin, positions=positions)
            results['hessian_method'] = 'analytic'
            return results

    if settings.get('calculator', '').lower() == 'torchani':
        try:
            import torch
            import torchani
            device = torch.device(settings.get('device', 'cpu'))
            model_name = settings.get('model', 'ANI2x')
            model = load_torchani_model(model_name, device, cache, double=True)
            
            species = torch.tensor(atoms.get_atomic_numbers(), device=device, dtype=torch.long).unsqueeze(0)
            coordinates = torch.from_numpy(atoms.get_positions()).unsqueeze(0).requires_grad_(True)
            masses = torchani.utils.get_atomic_masses(species)
            energies = model((species, coordinates)).energies
            hessian = torchani.utils.hessian(coordinates, energies=energies)
            freqs, modes, force_constants, reduced_masses = torchani.utils.vibrational_analysis(masses, hessian, mode_type='MDN')
            
            return {
                'schema_version': SCHEMA_VERSION,
                'freqs': (freqs.cpu().numpy().tolist() if hasattr(freqs, 'cpu') else freqs.numpy().tolist()),
                'hessian': hessian.cpu().numpy().tolist() if hasattr(hessian, 'cpu') else hessian.tolist(),
                'modes': modes.cpu().numpy().tolist() if hasattr(modes, 'cpu') else modes.tolist(),
                'force_constants': force_constants.cpu().numpy().tolist() if hasattr(force_constants, 'cpu') else force_constants.tolist(),
                'reduced_masses': reduced_masses.cpu().numpy().tolist() if hasattr(reduced_masses, 'cpu') else reduced_masses.tolist(),
                'hessian_method': 'analytic',
            }
        except Exception as exc:
            print(f'TorchANI vibrational analysis failed, falling back to ASE: {exc}')

    # Vibrations.run() skips every displacement already in its cache, and the cache is keyed only by
    # displacement name. A cache shared with another job (incore runs in ARC's working directory, or a
    # persistent worker) would silently mix forces from different molecules, so keep it in the job's
    # own directory and clear what an interrupted run left behind.
    vib = Vibrations(atoms, name=os.path.join(work_dir or os.getcwd(), 'vib_tmp'), nfree=4)
    vib.clean()
    try:
        vib.run()
        vib_data = vib.get_vibrations()
    finally:
        vib.clean()
    try:
        hessian = vib_data.get_hessian_2d()
    except AttributeError:
        hessian = vib_data.get_hessian()
        if len(hessian.shape) == 4:
            n_atoms = hessian.shape[0]
            hessian = hessian.reshape(3 * n_atoms, 3 * n_atoms)
    results = numpy_vibrational_analysis(atom_masses, hessian, is_linear=is_lin, positions=positions)
    results['hessian_method'] = 'finite_difference_gradient'
    return results


def resolve_input_path(yml_path: str) -> str:
    """
    Resolve the path of a job's input.yml, given it or its directory.
    """
    input_path = os.path.abspath(yml_path)
    if os.path.isdir(input_path):
        input_path = os.path.join(input_path, 'input.yml')
    return input_path


def run_job(input_path: str, input_dict: dict | None = None, cache: dict | None = None) -> dict:
    """
    Run one ASE job and write its output.yml next to its input.yml.

    Args:
        input_path (str): The path to the job's input.yml.
        input_dict (dict, optional): The parsed input.yml. Read from ``input_path`` if not given.
        cache (dict, optional): Loaded models, kept across jobs by a persistent worker.

    Returns:
        dict: The output, as written to output.yml.
    """
    if input_dict is None:
        input_dict = read_yaml_file(input_path)
    job_dir = os.path.dirname(input_path)

    job_type = input_dict.get('job_type')
    xyz = input_dict.get('xyz')
    settings = input_dict.get('settings', {})
    charge = input_dict.get('charge', 0)
    multiplicity = input_dict.get('multiplicity', 1)
    is_ts = input_dict.get('is_ts', False)

    atoms = Atoms(symbols=xyz['symbols'], positions=xyz['coords'])
    atoms.info.update({'charge': charge, 'spin': multiplicity})  # UMA (omol) conditions on these
    # A UMA freq job needs only the Hessian predictor (see uma_hessian()). Loading the ordinary one too would
    # double a one-shot run's model loading, so load just that one, which uma_hessian() then takes from the cache.
    if cache is None:
        cache = dict()
    hessian_only = job_type == 'freq' and settings.get('calculator', '').lower() in ('uma', 'fairchem') \
        and settings.get('hessian', 'autograd') == 'autograd'
    calc = get_calculator(settings, charge, multiplicity, cache=cache, hessian=hessian_only)
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
            hessian_function = sella_hessian_function(settings, cache) if is_ts else None
            opt = opt_class(atoms, order=1 if is_ts else 0, logfile=logfile,
                            **({'hessian_function': hessian_function} if hessian_function else dict()))
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
            freq_results = run_vibrational_analysis(atoms, settings, work_dir=job_dir, cache=cache)
            output.update(freq_results)
        except Exception as exc:
            output['error'] = output.get('error', '') + f" Frequency calculation failed: {exc}"

    output_path = os.path.join(job_dir, 'output.yml')
    save_yaml_file(output_path, output)
    return output


# Persistent worker.
# A worker (``ase_script.py --serve <spool>``) loads each model once and then runs ARC's ASE jobs from
# a spool directory, instead of a new process (and a model reload) per job. ARC submits a job by
# dropping a ticket that names the job's input.yml (see ASEAdapter.execute_through_worker()); the worker
# writes output.yml exactly as a one-shot run does. Spool layout:
#   queue/<ticket>.json    submitted, not yet claimed
#   running/<ticket>.json  claimed by a worker, by an atomic rename, so several workers can share a spool
#   done/<ticket>.json     finished: {'status': 'done' or 'errored', ...}
#   workers/<worker>.json  a worker's heartbeat: its mtime is refreshed while the worker process lives
#   tmp/                   files being written, renamed into place once complete
# The functions below up to serve() use only the standard library, so ARC can import them as a client.
SPOOL_SUBDIRS = ('queue', 'running', 'done', 'workers', 'tmp')
HEARTBEAT_INTERVAL = 5.0  # seconds between heartbeats
HEARTBEAT_STALE = 30.0  # seconds without a heartbeat after which a worker is presumed dead


def spool_dirs(spool_dir: str) -> dict:
    """
    Return the spool's subdirectories by name, creating any that are missing.
    """
    dirs = {name: os.path.join(spool_dir, name) for name in SPOOL_SUBDIRS}
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs


def write_json_atomic(path: str, content: dict, tmp_dir: str) -> None:
    """
    Write a JSON file so that readers never see it partially written.
    """
    tmp_path = os.path.join(tmp_dir, f'{uuid.uuid4().hex}.json')
    with open(tmp_path, 'w') as f:
        json.dump(content, f)
    os.replace(tmp_path, path)


def read_json(path: str) -> dict | None:
    """
    Read a JSON file, returning ``None`` if it does not exist (any more).
    """
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def live_workers(spool_dir: str, stale_after: float = HEARTBEAT_STALE) -> list:
    """
    Return the IDs of the workers on a spool whose heartbeat is recent.
    """
    workers_dir = os.path.join(spool_dir, 'workers')
    if not os.path.isdir(workers_dir):
        return list()
    now, alive = time.time(), list()
    for file_name in os.listdir(workers_dir):
        try:
            if now - os.path.getmtime(os.path.join(workers_dir, file_name)) < stale_after:
                alive.append(file_name[:-len('.json')])
        except FileNotFoundError:
            continue
    return alive


def submit_ticket(spool_dir: str, input_path: str, label: str = '') -> str:
    """
    Submit a job to a spool.

    Args:
        spool_dir (str): The spool directory.
        input_path (str): The absolute path to the job's input.yml.
        label (str, optional): A readable label (e.g., the job name), used in the ticket ID.

    Returns:
        str: The ticket ID.
    """
    dirs = spool_dirs(spool_dir)
    ticket = f'{label}_{uuid.uuid4().hex[:12]}' if label else uuid.uuid4().hex
    write_json_atomic(os.path.join(dirs['queue'], f'{ticket}.json'),
                      {'input_path': input_path, 'submitted': time.time()}, dirs['tmp'])
    return ticket


def withdraw_ticket(spool_dir: str, ticket: str) -> bool:
    """
    Withdraw a ticket that no worker has claimed yet.

    Returns:
        bool: Whether the ticket was withdrawn (``False`` if a worker had already claimed it).
    """
    try:
        os.remove(os.path.join(spool_dir, 'queue', f'{ticket}.json'))
        return True
    except FileNotFoundError:
        return False


def get_ticket_status(spool_dir: str, ticket: str) -> tuple:
    """
    Get the state of a ticket: 'queued', 'running', 'done', 'errored', or 'missing'.
    The directories are checked in the order a ticket moves through them, so a ticket in transit is
    never reported missing.

    Returns:
        tuple: The state, and the ticket's content (``None`` if missing).
    """
    for state in ('queue', 'running', 'done'):
        content = read_json(os.path.join(spool_dir, state, f'{ticket}.json'))
        if content is not None:
            if state == 'done':
                return content.get('status', 'done'), content
            return {'queue': 'queued', 'running': 'running'}[state], content
    return 'missing', None


def finish_ticket(spool_dir: str, ticket: str, content: dict) -> None:
    """
    Mark a ticket as finished. The ``done`` marker is written before the ``running`` one is removed.
    """
    dirs = spool_dirs(spool_dir)
    write_json_atomic(os.path.join(dirs['done'], f'{ticket}.json'), content, dirs['tmp'])
    try:
        os.remove(os.path.join(dirs['running'], f'{ticket}.json'))
    except FileNotFoundError:
        pass


def claim_ticket(spool_dir: str, worker_id: str) -> tuple | None:
    """
    Claim the oldest queued ticket for a worker.

    Returns:
        tuple: The ticket ID and its content, or ``None`` if nothing is queued.
    """
    dirs = spool_dirs(spool_dir)
    queued = list()
    for file_name in os.listdir(dirs['queue']):
        try:
            queued.append((os.path.getmtime(os.path.join(dirs['queue'], file_name)), file_name))
        except FileNotFoundError:
            continue
    for _, file_name in sorted(queued):
        running_path = os.path.join(dirs['running'], file_name)
        try:
            os.rename(os.path.join(dirs['queue'], file_name), running_path)
        except FileNotFoundError:
            continue  # another worker claimed it first
        content = read_json(running_path) or dict()
        content.update({'worker': worker_id, 'started': time.time()})
        write_json_atomic(running_path, content, dirs['tmp'])
        return file_name[:-len('.json')], content
    return None


def start_heartbeat(path: str, info: dict, interval: float = HEARTBEAT_INTERVAL) -> threading.Event:
    """
    Keep a worker's heartbeat file fresh from a background thread, also while a job runs.

    Returns:
        threading.Event: Set it to stop the heartbeat.
    """
    stop = threading.Event()
    tmp_dir = os.path.join(os.path.dirname(os.path.dirname(path)), 'tmp')

    def beat():
        while True:
            try:
                os.utime(path)
            except FileNotFoundError:
                write_json_atomic(path, info, tmp_dir)
            if stop.wait(interval):
                break

    write_json_atomic(path, info, tmp_dir)
    threading.Thread(target=beat, daemon=True).start()
    return stop


def run_ticket(spool_dir: str, ticket: str, content: dict, worker_id: str, cache: dict) -> dict:
    """
    Run a claimed ticket's job in its own directory and mark the ticket as finished.
    Any failure, including a ``sys.exit()`` in the job, is recorded in the ticket instead of ending the worker.

    Returns:
        dict: The ticket's final content.
    """
    t0, cwd = time.time(), os.getcwd()
    result = {'input_path': content.get('input_path'), 'worker': worker_id, 'status': 'done'}
    try:
        os.chdir(os.path.dirname(content['input_path']))  # as a queue job runs in its job directory
        run_job(content['input_path'], cache=cache)
    except (Exception, SystemExit):
        result.update({'status': 'errored', 'error': traceback.format_exc()})
    finally:
        os.chdir(cwd)
    finished = time.time()
    result.update({'started': t0, 'finished': finished, 'run_time': finished - t0})
    finish_ticket(spool_dir, ticket, result)
    return result


def serve(spool_dir: str, max_jobs: int | None = None, idle_timeout: float | None = None,
          poll_interval: float = 0.5) -> int:
    """
    Run as a persistent worker: claim and run the spool's tickets one at a time, keeping loaded models.
    SIGTERM or SIGINT stops the worker after its current job; a second signal acts as usual.

    Args:
        spool_dir (str): The spool directory.
        max_jobs (int, optional): Stop after this many jobs.
        idle_timeout (float, optional): Stop after this many seconds without a ticket.
        poll_interval (float, optional): Seconds between checks of an empty queue.

    Returns:
        int: The number of jobs run.
    """
    dirs = spool_dirs(os.path.abspath(spool_dir))
    spool_dir = os.path.dirname(dirs['queue'])
    worker_id = f'{socket.gethostname()}_{os.getpid()}'
    heartbeat_path = os.path.join(dirs['workers'], f'{worker_id}.json')
    stop_heartbeat = start_heartbeat(heartbeat_path, {'pid': os.getpid(), 'host': socket.gethostname(),
                                                      'started': time.time()})
    stopping = threading.Event()
    previous_handlers = dict()

    def request_stop(signum, frame):
        stopping.set()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)

    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous_handlers[sig] = signal.signal(sig, request_stop)
    print(f'ASE worker {worker_id} serving {spool_dir}', flush=True)

    cache, n_jobs, idle_since = dict(), 0, time.time()
    try:
        while not stopping.is_set() and (max_jobs is None or n_jobs < max_jobs):
            claimed = claim_ticket(spool_dir, worker_id)
            if claimed is None:
                if idle_timeout is not None and time.time() - idle_since > idle_timeout:
                    break
                stopping.wait(poll_interval)
                continue
            ticket, content = claimed
            result = run_ticket(spool_dir, ticket, content, worker_id, cache)
            print(f'{ticket}: {result["status"]} in {result["run_time"]:.1f} s', flush=True)
            n_jobs, idle_since = n_jobs + 1, time.time()
    finally:
        stop_heartbeat.set()
        try:
            os.remove(heartbeat_path)
        except FileNotFoundError:
            pass
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
    return n_jobs


def main():
    """
    Main execution logic.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--yml_path', type=str, default='input.yml')
    parser.add_argument('--serve', type=str, default=None, metavar='SPOOL_DIR',
                        help='Run as a persistent worker on this spool directory instead of running one job.')
    parser.add_argument('--max_jobs', type=int, default=None, help='Worker: stop after this many jobs.')
    parser.add_argument('--idle_timeout', type=float, default=None,
                        help='Worker: stop after this many seconds without a job.')
    args = parser.parse_args()

    if args.serve:
        serve(args.serve, max_jobs=args.max_jobs, idle_timeout=args.idle_timeout)
        return

    input_path = resolve_input_path(args.yml_path)
    try:
        input_dict = read_yaml_file(input_path)
    except Exception as exc:
        print(f"Error reading input file: {exc}")
        return
    run_job(input_path, input_dict)


if __name__ == '__main__':
    main()
