#!/usr/bin/env python3
# encoding: utf-8

"""
A standalone script to run ASE (Atomic Simulation Environment) jobs.
Standardizes interaction with various calculators.
"""

import argparse
import contextlib
import datetime
import json
import math
import os
import re
import signal
import socket
import sys
import threading
import time
import traceback
import uuid
import warnings
import yaml
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import PackageNotFoundError, version as package_version
from typing import Callable

from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
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
    Convert ASE default (eV) to kJ/mol, as a plain Python float.
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
    Single- and double-precision models are cached under separate keys.
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


def package_release(name: str) -> tuple | None:
    """
    The (major, minor) release of an installed package, or ``None`` if it is not installed.
    """
    try:
        return tuple(int(part) for part in re.findall(r'\d+', package_version(name))[:2])
    except PackageNotFoundError:
        return None


def fairchem_hessian_support(calc_config: dict) -> str | None:
    """
    Why the installed fairchem-core cannot give a UMA job its autograd Hessian, or ``None`` if it can.
    Autograd Hessians need fairchem-core 2.17 or newer. fairchem-core 2.18 to 2.21 vectorize the Hessian's
    backward passes (vmap) whatever ``InferenceSettings.hessian_vmap`` says, so there the Hessian is used
    only if the job sets ``hessian_vmap: True``.

    Args:
        calc_config (dict): The run settings.

    Returns:
        str | None: The reason, or ``None``.
    """
    release = package_release('fairchem-core')
    if release is None or release < (2, 17):
        return f'fairchem-core {".".join(map(str, release)) if release else "(not found)"} computes no autograd ' \
               f'Hessians (2.17 or newer does)'
    if (2, 18) <= release <= (2, 21) and not calc_config.get('hessian_vmap', False):
        return f'fairchem-core {release[0]}.{release[1]} computes autograd Hessians only with vmap; ' \
               f'set hessian_vmap: True to use it'
    return None


def uma_inference_settings(calc_config: dict, hessian: bool = False) -> tuple:
    """
    Resolve the fairchem inference settings of a UMA job.

    ``inference_settings`` is a fairchem preset name or a dict of ``InferenceSettings`` fields.

    A Hessian predictor runs without TF32, merging, compilation or activation checkpointing, at the
    ``base_precision_dtype`` of ``inference_settings`` (FP32 by default), and with ``hessian_vmap`` as the job
    sets it (False by default), predicting the Hessian for the job's task.

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
        NotImplementedError: If a Hessian is requested and ``fairchem_hessian_support()`` gives a reason it cannot be.
    """
    from fairchem.core import pretrained_mlip
    model = calc_config.get('model', 'uma-s-1p2')
    device = calc_config.get('device', 'cpu')
    settings, settings_key = uma_inference_settings(calc_config, hessian=hessian)
    key = ('uma', model, device, settings_key)
    predictor = cache.get(key) if cache is not None else None
    if predictor is None:
        predict_kwargs = {'device': device}
        if hessian:
            reason = fairchem_hessian_support(calc_config)
            if reason is not None:
                raise NotImplementedError(reason)
        if isinstance(settings, dict):
            from fairchem.core.units.mlip_unit.api.inference import InferenceSettings
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
    Compute a UMA model's Hessian by automatic differentiation, with the Hessian predictor of
    ``load_uma_predictor(..., hessian=True)``, separate from the predictor that computes energies and forces.

    If a Hessian with ``hessian_vmap`` runs out of GPU memory, it is computed without vmap instead, and with
    ``cache``, so is every later Hessian of as many atoms or more.

    Args:
        atoms (Atoms): The structure, with its charge and spin in ``atoms.info``.
        settings (dict): The run settings.
        cache (dict, optional): Loaded models, kept across jobs by a persistent worker.

    Returns:
        np.ndarray | None: The (3N, 3N) Hessian in eV/A^2, or ``None`` if the installed fairchem cannot
                           compute one.
    """
    from fairchem.core import FAIRChemCalculator
    too_large = ('uma_hessian_vmap_too_large', settings.get('model'), settings.get('device'))
    if settings.get('hessian_vmap') and cache is not None and len(atoms) >= cache.get(too_large, np.inf):
        return uma_hessian(atoms, dict(settings, hessian_vmap=False), cache)
    try:
        predictor = load_uma_predictor(settings, cache=cache, hessian=True)
    except NotImplementedError as exc:
        print(f'{exc}; using finite differences instead.')
        return None
    hessian_atoms = atoms.copy()
    hessian_atoms.calc = FAIRChemCalculator(predictor, task_name=settings.get('task', 'omol'))
    n_coords = 3 * len(atoms)
    try:
        hessian = hessian_atoms.calc.get_property('hessian', hessian_atoms)
    except Exception as exc:
        if not settings.get('hessian_vmap') or not is_out_of_gpu_memory(exc):
            raise
        release_gpu_memory()
        print(f'The vectorized Hessian of {len(atoms)} atoms ran out of GPU memory; computing it without vmap.')
        if cache is not None:
            cache[too_large] = min(len(atoms), cache.get(too_large, np.inf))
        return uma_hessian(atoms, dict(settings, hessian_vmap=False), cache)
    return np.asarray(hessian, dtype=float).reshape(n_coords, n_coords)


def sella_hessian_function(settings: dict, cache: dict | None = None):
    """
    The exact Hessian for a Sella saddle-point search: UMA's autograd Hessian (see ``uma_hessian()``), unless the
    job sets ``hessian`` to something other than ``'autograd'`` or the installed fairchem cannot compute it.

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


MACE_OFF_SIZES = ('small', 'medium', 'large')


def require_neutral_singlet(model: str, charge: int, multiplicity: int) -> None:
    """
    Raise ``ValueError`` if a species is charged or open-shell, for a model that reads neither.
    """
    if charge != 0 or multiplicity != 1:
        raise ValueError(f'{model} takes neither charge nor multiplicity, so it cannot compute a species '
                         f'of charge {charge} and multiplicity {multiplicity}.')


def mace_model(calc_config: dict) -> tuple:
    """
    The MACE model family ('omol' or 'off') and the model passed to it. ``model`` is ``omol`` (the default),
    ``omol-<size>``, ``off-<size>`` or a MACE-OFF size (small, medium, large), or a checkpoint file, whose family
    is given by ``family``.

    Raises:
        ValueError: If the model is not recognized, or a checkpoint is given without its family.
    """
    model = str(calc_config.get('model', 'omol'))
    family = calc_config.get('family')
    if os.path.isfile(model):
        if family not in ('omol', 'off'):
            raise ValueError(f"Set 'family' to 'omol' or 'off' for the MACE checkpoint {model}.")
        return family, model
    if model == 'omol':
        return 'omol', None
    if model.startswith('omol-'):
        return 'omol', model[len('omol-'):]
    size = model[len('off-'):] if model.startswith('off-') else model
    if size in MACE_OFF_SIZES:
        return 'off', size
    raise ValueError(f"Unknown MACE model '{model}': use omol, omol-<size>, off-<size> ({', '.join(MACE_OFF_SIZES)}), "
                     f"or a checkpoint file with 'family'.")


def build_mace_calculator(family: str, model: str | None, device: str, dtype: str):
    """
    Build a MACE-OMol or MACE-OFF calculator (mace-torch 0.3.14 or newer for MACE-OMol).
    """
    if family == 'omol':
        try:
            from mace.calculators import mace_omol
        except ImportError as exc:
            raise ImportError('MACE-OMol needs mace-torch 0.3.14 or newer.') from exc
        return mace_omol(model=model, device=device, default_dtype=dtype)
    from mace.calculators import mace_off
    return mace_off(model=model, device=device, default_dtype=dtype)


def build_orb_calculator(model: str, device: str, precision: str):
    """
    Build an Orb calculator for a pretrained model of orb-models 0.6.0 or newer.
    """
    try:
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.inference.calculator import ORBCalculator
    except ImportError as exc:
        raise ImportError('The Orb calculator needs orb-models 0.6.0 or newer.') from exc
    orbff, adapter = getattr(pretrained, model.replace('-', '_'))(device=device, precision=precision)
    return ORBCalculator(orbff, atoms_adapter=adapter, device=device)


def cached_calculator(cache: dict | None, key: tuple, build):
    """
    A calculator from ``cache`` under ``key``, reset so that it keeps no results, or a new one from ``build()``,
    which is cached if a cache is given.
    """
    if cache is None:
        return build()
    if key not in cache:
        cache[key] = build()
    calc = cache[key]
    calc.reset()
    return calc


def get_calculator(calc_config: dict, charge: int = 0, multiplicity: int = 1, cache: dict | None = None,
                   hessian: bool = False):
    """
    Initialize the ASE calculator based on settings.

    UMA, MACE-OMol (``mace`` with an ``omol`` model), the Orb ``omol`` models and the AIMNet2 NSE models are given
    the charge and multiplicity; the other AIMNet2 models take the charge only, and MACE-OFF and the other Orb
    models take neither, so a charged or open-shell species raises ``ValueError`` on them, as one does on MOPAC.
    Minimum versions: mace-torch 0.3.14, orb-models 0.6.0, aimnet 0.1.

    Args:
        calc_config (dict): The run settings.
        charge (int): The total charge.
        multiplicity (int): The spin multiplicity.
        cache (dict, optional): Loaded models and calculators, kept across jobs by a persistent worker. A cached
                                calculator is reset before it is returned, so no results carry over between jobs.
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
        from fairchem.core import FAIRChemCalculator
        try:
            predictor = load_uma_predictor(calc_config, cache=cache, hessian=hessian)
        except NotImplementedError:
            predictor = load_uma_predictor(calc_config, cache=cache)
        return FAIRChemCalculator(predictor, task_name=calc_config.get('task', 'omol'))

    elif name == 'mace':
        family, model = mace_model(calc_config)
        if family == 'off':
            require_neutral_singlet(f'MACE-OFF model {model}', charge, multiplicity)
        device, dtype = calc_config.get('device', 'cpu'), calc_config.get('default_dtype', 'float64')
        return cached_calculator(cache, ('mace', family, model, device, dtype),
                                 lambda: build_mace_calculator(family, model, device, dtype))

    elif name == 'orb':
        model = calc_config.get('model', 'orb-v3-conservative-omol')
        if 'omol' not in model:
            require_neutral_singlet(f'Orb model {model}', charge, multiplicity)
        device, precision = calc_config.get('device', 'cpu'), calc_config.get('precision', 'float64')
        return cached_calculator(cache, ('orb', model, device, precision),
                                 lambda: build_orb_calculator(model, device, precision))

    elif name == 'aimnet2':
        model = calc_config.get('model', 'aimnet2nse')
        if 'nse' not in model.lower() and multiplicity != 1:
            raise ValueError(f'AIMNet2 model {model} does not take the multiplicity ({multiplicity}); '
                             f'use an NSE model (e.g., aimnet2nse).')
        from aimnet.calculators.aimnet2ase import AIMNet2ASE
        device = calc_config.get('device', 'cpu')
        key = ('aimnet2', model, device)
        base = cache.get(key) if cache is not None else None
        if base is None:
            from aimnet.calculators import AIMNet2Calculator
            base = AIMNet2Calculator(model, device=device)
            if cache is not None:
                cache[key] = base
        return AIMNet2ASE(base, charge=charge, mult=multiplicity)

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
              pivot_mult: float = 1.8, extra_bonds: list | None = None) -> list:
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
    ``pivot_mult`` allowance; the rest of the graph stays at ``mult``. ``extra_bonds`` (e.g., a TS's forming
    and breaking bonds) are added to the graph, so a torsion inside a cyclic TS is part of a ring.

    Args:
        atoms (Atoms): The molecule.
        pivot_1 (int): The 0-indexed atom on the fixed side of the rotation axis.
        pivot_2 (int): The 0-indexed atom on the rotating side of the rotation axis.
        mult (float, optional): The natural-cutoff multiplier defining the bond graph.
        pivot_mult (float, optional): The covalent-radii multiplier allowed for the pivot bond
                                      alone, accommodating a stretched TS bond.
        extra_bonds (list, optional): 0-indexed atom pairs to bond whatever their distance (a TS's partial bonds).

    Raises:
        ValueError: If the pivots are too far apart to be bonded even under ``pivot_mult``, or if
                    the pivot bond is part of a ring (a 1D rotor is then ill-defined).

    Returns:
        list: The sorted 0-indexed atoms of the rotating group.
    """
    nl = build_neighbor_list(atoms, natural_cutoffs(atoms, mult=mult),
                             self_interaction=False, bothways=True, skin=0.0)
    adj = {i: set(nl.get_neighbors(i)[0]) for i in range(len(atoms))}
    for atom_1, atom_2 in extra_bonds or list():
        adj[atom_1].add(atom_2)
        adj[atom_2].add(atom_1)
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


def well_defined_torsion(atoms: Atoms, torsion: list, max_angle: float = 165.0, mult: float = 1.2) -> list:
    """
    The scanned dihedral, with an end atom moved outward if it is collinear with the pivot bond.

    An end atom whose angle with the pivot bond exceeds ``max_angle`` is replaced by the nearest atom beyond it in
    the bond graph whose angle does not, e.g., the far C of H-O-C#C in HOC#COH is replaced by the far H.

    Args:
        atoms (Atoms): The molecule.
        torsion (list): The 0-indexed four atoms (i, j, k, l); j-k is the pivot bond.
        max_angle (float, optional): The largest i-j-k or j-k-l angle (degrees) taken as well defined.
        mult (float, optional): The natural-cutoff multiplier defining the bond graph, as in rotor_top().

    Raises:
        ValueError: If one side of the pivot bond is linear to its end, so that no torsion exists about it.

    Returns:
        list: The 0-indexed four atoms of a well-defined dihedral about the same pivot bond.
    """
    i, j, k, l = torsion
    nl = build_neighbor_list(atoms, natural_cutoffs(atoms, mult=mult), self_interaction=False, bothways=True, skin=0.0)

    def outward(end: int, pivot: int, other_pivot: int) -> int:
        """The nearest atom beyond ``end`` (away from the pivot bond) off the pivot axis."""
        seen, frontier = {end, pivot, other_pivot}, [end]
        while frontier:
            for atom in frontier:
                if atoms.get_angle(other_pivot, pivot, atom) <= max_angle:
                    return atom
            frontier = [nb for atom in frontier for nb in nl.get_neighbors(atom)[0] if nb not in seen]
            seen.update(frontier)
        raise ValueError(f'The atoms beyond pivot {pivot} are collinear with the pivot bond {j}-{k}, '
                         f'so no torsion about it is defined.')

    return [int(outward(i, j, k)), j, k, int(outward(l, k, j))]


def is_out_of_gpu_memory(exc: BaseException) -> bool:
    """Whether an error means the GPU ran out of memory (cuBLAS reports it as a failed allocation of its own)."""
    message = str(exc).lower()
    return 'out of memory' in message or 'alloc_failed' in message


def is_transient_gpu_error(exc: BaseException) -> bool:
    """
    Whether an error is a cuBLAS failure (``CUBLAS_STATUS_...``) other than a failed allocation, which is an
    out-of-memory error instead (see ``is_out_of_gpu_memory()``).
    """
    return 'cublas_status' in str(exc).lower() and not is_out_of_gpu_memory(exc)


def release_gpu_memory() -> None:
    """Return PyTorch's cached GPU memory after an out-of-memory error, if PyTorch is loaded."""
    torch = sys.modules.get('torch')
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def bare_copy(atoms: Atoms) -> Atoms:
    """A copy of a structure with its atoms, positions and info (charge and spin) only: no calculator or constraints."""
    return Atoms(numbers=atoms.get_atomic_numbers(), positions=atoms.get_positions(), info=dict(atoms.info))


def batch_evaluation_function(calc: Calculator) -> Callable:
    """
    A function that evaluates a list of structures with ``calc`` and returns their (energy in eV, forces in eV/A).

    For UMA (a fairchem calculator), the structures go to the model in one batched forward pass, converted exactly as
    the calculator converts a single structure; each structure's charge and spin come from its own ``info``. Any other
    calculator evaluates them one after another.

    Args:
        calc (Calculator): The calculator.

    Returns:
        Callable: A function of a list of structures, returning a list of (energy, forces) tuples in the same order.
    """
    predictor, a2g = getattr(calc, 'predictor', None), getattr(calc, 'a2g', None)
    if predictor is not None and a2g is not None:
        from fairchem.core.datasets.atomic_data import atomicdata_list_to_batch

        def evaluate_batch(structures: list) -> list:
            for structure in structures:
                predictor.validate_atoms_data(structure, calc.task_name)
            prediction = predictor.predict(atomicdata_list_to_batch([a2g(structure) for structure in structures]))
            energies = prediction['energy'].detach().cpu().numpy().astype(float)
            forces = prediction['forces'].detach().cpu().numpy().astype(float)
            forces = np.split(forces, np.cumsum([len(structure) for structure in structures])[:-1])
            return [(float(energy), force) for energy, force in zip(energies, forces)]

        return evaluate_batch

    def evaluate_one_by_one(structures: list) -> list:
        results = list()
        for structure in structures:
            structure = bare_copy(structure)
            structure.calc = calc
            results.append((float(structure.get_potential_energy()), np.array(structure.get_forces())))
        return results

    return evaluate_one_by_one


class BatchedEvaluator:
    """
    Evaluates the force calls of optimizations that run in threads together, in one call to the model.

    A torsional scan walks each torsion in both directions, and a worker can run several scans at once. Each walk runs
    in its own thread, with an ``EvaluatorCalculator`` that hands its structure to this evaluator and waits. Once every
    registered walk is waiting, the last one to arrive evaluates all of them: for UMA in one batched forward pass,
    which keeps a GPU busy where a single small molecule leaves it mostly idle. Hessian requests (from a TS walk's
    Sella searches) are evaluated one at a time in the same round. The models are thus only ever called from one
    thread at a time.

    A walk registers with ``join()`` before it starts, and it must call ``leave()`` when it ends, also after an
    error; the other walks would otherwise wait for it forever.

    A batch that runs out of GPU memory is evaluated in halves, and ``max_batch`` stays at the size that fit. A batch or
    Hessian that fails with a transient cuBLAS error (``is_transient_gpu_error()``) is evaluated again, up to
    ``max_retries`` times; only then do its walks get the error. Any other error raised while a round is evaluated
    is delivered to every walk of the round that has no result yet.

    Args:
        evaluate (Callable): Evaluates a list of structures (see ``batch_evaluation_function()``).
        hessian (Callable, optional): The (3N, 3N) Hessian of a structure, in eV/A^2.
        max_batch (int, optional): The most structures per call of ``evaluate``.
        max_retries (int, optional): How often to repeat a call that failed with a transient cuBLAS error.
    """

    def __init__(self, evaluate: Callable, hessian: Callable | None = None, max_batch: int = 256,
                 max_retries: int = 3):
        self.evaluate = evaluate
        self.hessian = hessian
        self.max_batch = max_batch
        self.max_retries = max_retries
        self.n_rounds, self.n_structures, self.n_retries = 0, 0, 0
        self._condition = threading.Condition()
        self._clients = 0
        self._pending, self._results = dict(), dict()
        self._next_key = 0

    def join(self, n: int = 1) -> None:
        """Register ``n`` walks."""
        with self._condition:
            self._clients += n

    def leave(self) -> None:
        """Unregister a walk that has ended. The walks still waiting may now be complete, so evaluate them."""
        with self._condition:
            self._clients -= 1
            self._evaluate_if_all_wait()

    def request(self, kind: str, structure: Atoms):
        """
        Evaluate a structure together with the other walks' structures, and wait for the result.

        Args:
            kind (str): 'forces' for (energy, forces), or 'hessian'.
            structure (Atoms): The structure (see ``bare_copy()``), which must not change until this returns.
        """
        with self._condition:
            key = self._next_key
            self._next_key += 1
            self._pending[key] = (kind, structure)
            self._evaluate_if_all_wait()
            while key not in self._results:
                self._condition.wait()
            result = self._results.pop(key)
        if isinstance(result, BaseException):
            raise result
        return result

    def _evaluate_if_all_wait(self) -> None:
        """Evaluate the pending requests if no registered walk is still running. Called with the lock held."""
        if not self._pending or len(self._pending) < self._clients:
            return
        pending, self._pending = self._pending, dict()
        try:
            self._evaluate(pending)
        except BaseException as exc:
            for key in pending:
                self._results.setdefault(key, exc)
        finally:
            self._condition.notify_all()

    def _evaluate(self, pending: dict) -> None:
        """Evaluate pending requests into ``self._results``. Called with the lock held."""
        forces = [key for key, (kind, _) in pending.items() if kind == 'forces']
        start, retries = 0, 0
        while start < len(forces):
            keys = forces[start:start + self.max_batch]
            try:
                results = self.evaluate([pending[key][1] for key in keys])
            except Exception as exc:
                if is_out_of_gpu_memory(exc) and len(keys) > 1:
                    self.max_batch = len(keys) // 2
                    release_gpu_memory()
                    continue
                if is_transient_gpu_error(exc) and retries < self.max_retries:
                    retries += 1
                    self.n_retries += 1
                    release_gpu_memory()
                    continue
                results = [exc] * len(keys)
            self._results.update(zip(keys, results))
            start += len(keys)
            retries = 0
        for key, (kind, structure) in pending.items():
            if kind == 'hessian':
                self._results[key] = self._hessian_with_retries(structure)
        self.n_rounds += 1
        self.n_structures += len(forces)

    def _hessian_with_retries(self, structure: Atoms):
        """The Hessian of a structure, or the error that remained after ``max_retries`` retries of a transient one."""
        for attempt in range(self.max_retries + 1):
            try:
                return self.hessian(structure)
            except Exception as exc:
                if not is_transient_gpu_error(exc) or attempt == self.max_retries:
                    return exc
                self.n_retries += 1
                release_gpu_memory()


class EvaluatorCalculator(Calculator):
    """An ASE calculator that gets its energy and forces from a ``BatchedEvaluator``. Each walk needs its own."""
    implemented_properties = ['energy', 'free_energy', 'forces']

    def __init__(self, evaluator: BatchedEvaluator):
        super().__init__()
        self.evaluator = evaluator

    def calculate(self, atoms=None, properties=('energy',), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        energy, forces = self.evaluator.request('forces', bare_copy(self.atoms))
        self.results = {'energy': energy, 'free_energy': energy, 'forces': forces}


def _scan_walk(atoms: Atoms, torsion: list, step_deg: float, nsteps: int, top: list,
               fmax: float, steps: int, direction: int, optimizer, calc: Calculator | None = None,
               saddle: bool = False, hessian_function: Callable | None = None) -> tuple:
    """
    Perform one directional sequential relaxed torsional walk.

    Each grid point starts from the previous point's relaxed geometry (a hysteretic walk), so a
    coupled coordinate can carry the walk into a different conformer and never return - which is
    why a full scan runs this in both directions and keeps the lower branch at each point.

    A TS walk (``saddle``) searches each point for a first-order saddle point with the dihedral fixed (Sella,
    order 1), as Gaussian's TS scans do (opt=(ts, modredundant)).

    Args:
        atoms (Atoms): The molecule with a calculator attached (not modified; a copy is walked).
        torsion (list): The 0-indexed four atoms (i, j, k, l) defining the scanned dihedral.
        step_deg (float): The dihedral increment in degrees.
        nsteps (int): The number of increments (the walk visits ``nsteps`` + 1 points).
        top (list): The 0-indexed rotating group.
        fmax (float): The force convergence criterion in eV/Angstrom.
        steps (int): The maximum optimizer steps per point.
        direction (int): +1 for a 0->360 walk, -1 for a 360->0 walk.
        optimizer: The ASE optimizer class to relax each point of a minimum.
        calc (Calculator, optional): The calculator of the walk, if not ``atoms.calc``.
        saddle (bool, optional): Whether ``atoms`` is a TS whose walk must stay on the saddle.
        hessian_function (Callable, optional): The exact Hessian for the Sella searches of a TS walk.

    Returns:
        tuple: (energies in eV, residual max-forces in eV/Angstrom, relaxed Cartesian coordinates),
        each of length ``nsteps`` + 1.
    """
    i, j, k, l = torsion
    work = atoms.copy()
    work.calc = calc or atoms.calc
    energies, residual_forces, coords = list(), list(), list()
    for n in range(nsteps + 1):
        if n:
            work.rotate_dihedral(i, j, k, l, angle=direction * step_deg, indices=top)
        target = work.get_dihedral(i, j, k, l)
        work.set_constraint(FixInternals(dihedrals_deg=[[target, [i, j, k, l]]]))
        if saddle:
            from sella import Sella
            opt = Sella(work, order=1, logfile=None,
                        **({'hessian_function': hessian_function} if hessian_function else dict()))
        else:
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
                        coords_b: list = None, converged_f: list | None = None, converged_b: list | None = None) -> tuple:
    """
    Fold a forward and a backward torsional walk onto a common grid and keep the lower branch.

    The forward walk visits phi = +n * step and the backward walk phi = -n * step, so index ``n``
    of the backward walk lands on grid point ``(-n) % nsteps``. Both walks also revisit their
    starting point at index ``nsteps``, which folds back onto grid point 0. At each grid point the
    lower of the two energies wins, which is what recovers the lowest torsional path when one walk
    has been carried into a different conformer by a coupled coordinate. Given which points converged
    (``converged_f``, ``converged_b``, as for a TS walk, whose unconverged point can lie below its constrained
    saddle), a converged point wins over an unconverged one, and the lower energy only decides between two
    points that both converged or both did not.

    Args:
        e_f (list): The forward walk energies, of length ``nsteps`` + 1.
        e_b (list): The backward walk energies, of length ``nsteps`` + 1.
        nsteps (int): The number of increments; there are ``nsteps`` unique grid points.
        coords_f (list, optional): The forward walk Cartesian coordinates, aligned with ``e_f``.
        coords_b (list, optional): The backward walk Cartesian coordinates, aligned with ``e_b``.
        converged_f (list, optional): Whether each forward point converged, aligned with ``e_f``.
        converged_b (list, optional): Whether each backward point converged, aligned with ``e_b``.

    Returns:
        tuple: (the merged per-grid-point energies as a numpy array of length ``nsteps``, the
        forward grid, the backward grid, and the winning coordinates per grid point - the last
        being None if either coordinate list was not supplied).
    """
    converged_f = converged_f if converged_f is not None else [True] * len(e_f)
    converged_b = converged_b if converged_b is not None else [True] * len(e_b)
    grid_f, grid_b = np.full(nsteps, np.inf), np.full(nsteps, np.inf)
    ok_f, ok_b = [False] * nsteps, [False] * nsteps
    xyz_f, xyz_b = [None] * nsteps, [None] * nsteps
    for n in range(nsteps + 1):
        p = n % nsteps
        if (converged_f[n], -e_f[n]) > (ok_f[p], -grid_f[p]):
            grid_f[p], ok_f[p] = e_f[n], converged_f[n]
            if coords_f is not None:
                xyz_f[p] = coords_f[n]
        q = (-n) % nsteps
        if (converged_b[n], -e_b[n]) > (ok_b[q], -grid_b[q]):
            grid_b[q], ok_b[q] = e_b[n], converged_b[n]
            if coords_b is not None:
                xyz_b[q] = coords_b[n]
    forward_wins = [(ok_f[p], -grid_f[p]) >= (ok_b[p], -grid_b[p]) for p in range(nsteps)]
    merged = np.where(forward_wins, grid_f, grid_b)
    coords = None
    if coords_f is not None and coords_b is not None:
        coords = [xyz_f[p] if forward_wins[p] else xyz_b[p] for p in range(nsteps)]
    return merged, grid_f, grid_b, coords


def relaxed_torsion_scan(atoms: Atoms, torsion: list, step_deg: float, nsteps: int,
                         top: list = None, fmax: float = 0.005, steps: int = 500,
                         optimizer=LBFGS, saddle: bool = False, evaluator: BatchedEvaluator | None = None,
                         hessian_function: Callable | None = None) -> dict:
    """
    Perform a full 1D relaxed torsional scan in a single process.

    The torsion is swept 0->360 and 360->0 from the same starting geometry, and the lower energy
    of the two walks is kept at each grid point. A single directional walk is path-dependent: if a
    coupled coordinate flips partway round, that walk finishes in a different conformer and its
    curve is not a torsional potential. Walking both ways and taking the pointwise minimum recovers
    the lowest torsional path, which is the one the downstream statistical mechanics wants.

    The two walks run in two threads whose force calls a ``BatchedEvaluator`` evaluates together. A worker that
    runs several scans at once passes one evaluator to all of them, so that every walk shares each call.

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
        optimizer: The ASE optimizer class to relax each point of a minimum.
        saddle (bool, optional): Whether ``atoms`` is a TS, whose walks search each point for a saddle point.
        evaluator (BatchedEvaluator, optional): The evaluator shared with other scans. By default, one that
                                                evaluates ``atoms.calc``, with ``hessian_function``.
        hessian_function (Callable, optional): The exact Hessian for a TS walk's Sella searches, if no
                                               ``evaluator`` is given.

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
    torsion = well_defined_torsion(atoms, torsion)
    i, j, k, l = torsion
    if top is None:
        top = rotor_top(atoms, j, k)

    if evaluator is None:
        evaluator = BatchedEvaluator(batch_evaluation_function(atoms.calc), hessian=hessian_function)
    walk_hessian = None
    if saddle and evaluator.hessian is not None:
        def walk_hessian(structure: Atoms) -> np.ndarray:
            return evaluator.request('hessian', bare_copy(structure))

    def walk(direction: int) -> tuple:
        try:
            return _scan_walk(atoms, torsion, step_deg, nsteps, top, fmax, steps, direction, optimizer,
                              calc=EvaluatorCalculator(evaluator), saddle=saddle, hessian_function=walk_hessian)
        finally:
            evaluator.leave()

    evaluator.join(2)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = list()
        try:
            for direction in (+1, -1):
                futures.append(pool.submit(walk, direction))
        except BaseException:
            for _ in range(2 - len(futures)):
                evaluator.leave()
            raise
        e_f, r_f, c_f = futures[0].result()
        e_b, r_b, c_b = futures[1].result()

    converged_f, converged_b = ([r <= fmax for r in r_f], [r <= fmax for r in r_b]) if saddle else (None, None)
    merged, grid_f, grid_b, coords = merge_scan_branches(e_f, e_b, nsteps, c_f, c_b,
                                                         converged_f=converged_f, converged_b=converged_b)
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
        'saddle': saddle,
        'branch_gap_max': float(np.abs(grid_f - grid_b).max()),
        'fmax_worst': float(max(r_f + r_b)),
        'converged': bool(max(r_f + r_b) <= fmax),
    }


def run_torsion_scan(atoms: Atoms, input_dict: dict, settings: dict, evaluator: BatchedEvaluator | None = None,
                     cache: dict | None = None) -> dict:
    """
    Run a 1D relaxed torsional scan for an ARC ``scan`` job and shape it for ARC's YAML parser.
    The scan of a TS (``is_ts``) stays on the saddle (see ``_scan_walk()``).

    Args:
        atoms (Atoms): The molecule with a calculator attached.
        input_dict (dict): The job input; ``torsions`` (a list holding one 0-indexed torsion),
                           ``scan_res`` (the dihedral increment in degrees), ``is_ts`` and, optionally,
                           ``partial_bonds`` (a TS's 0-indexed forming and breaking bonds) and ``top`` (ARC's
                           0-indexed rotating group, on either side of the pivots) are read here. The rotating group
                           is rebuilt from distances plus ``partial_bonds`` if given, else ``top`` is used, else
                           distances alone (see ``rotor_top()``).
        settings (dict): ASE run settings; optional ``fmax``, ``steps``, ``optimizer`` override the
                         relaxed-scan defaults (LBFGS, fmax 0.005 eV/Angstrom, 500 steps per point).
        evaluator (BatchedEvaluator, optional): An evaluator shared with other scans (see ``run_scan_group()``).
        cache (dict, optional): Loaded models, kept across jobs by a persistent worker.

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
    saddle = bool(input_dict.get('is_ts', False))
    hessian_function = sella_hessian_function(settings, cache) if saddle and evaluator is None else None
    top = input_dict.get('top')
    if input_dict.get('partial_bonds'):
        top = rotor_top(atoms, *well_defined_torsion(atoms, torsion)[1:3],
                        extra_bonds=[tuple(bond) for bond in input_dict['partial_bonds']])
    if top is not None and torsion[1] in top and torsion[2] not in top:
        top = sorted(set(range(len(atoms))) - set(top))
    if top is not None and (torsion[2] not in top or torsion[1] in top):
        print(f'Ignoring the rotating group {top}, which does not separate the pivots {torsion[1:3]}.')
        top = None

    result = relaxed_torsion_scan(atoms, torsion, scan_res, nsteps, top=top, fmax=fmax, steps=steps,
                                  optimizer=optimizer, saddle=saddle, evaluator=evaluator,
                                  hessian_function=hessian_function)
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
                    and 6 otherwise. The k rigid motions removed are the first k left singular vectors of
                    the rigid-motion vectors, with k taken from ``is_linear``.
    """
    n_atoms = len(masses)
    n_rigid = 3 if n_atoms == 1 else 5 if is_linear or n_atoms == 2 else 6
    sqrt_mass = np.sqrt(masses)[:, np.newaxis]
    centered = positions - (masses[:, np.newaxis] * positions).sum(axis=0) / masses.sum()
    rigid = []
    for axis in np.eye(3):
        rigid.append((np.tile(axis, (n_atoms, 1)) * sqrt_mass).ravel())
        rigid.append((np.cross(axis, centered) * sqrt_mass).ravel())
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
        work_dir (str, optional): The job's directory, which holds the finite-difference cache (``vib_tmp``),
                                  cleared before and after the run. Defaults to the current working directory.
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


def read_trajectory(path: str) -> list:
    """
    The images of an ASE trajectory, or an empty list if the file does not exist.
    """
    from ase.io import read
    return read(path, index=':') if os.path.isfile(path) else list()


class ImageCalculator(Calculator):
    """
    A per-image calculator that evaluates one shared calculator on a copy of its image, with the image's ``info``
    (charge and spin), and counts the calls in ``counter['force_calls']``.
    """
    implemented_properties = ['energy', 'forces']

    def __init__(self, shared, counter: dict, **kwargs):
        super().__init__(**kwargs)
        self.shared, self.counter = shared, counter

    def calculate(self, atoms=None, properties=('energy',), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        image = self.atoms.copy()
        image.info.update(self.atoms.info)
        image.calc = self.shared
        self.results = {'energy': image.get_potential_energy(), 'forces': image.get_forces()}
        self.counter['force_calls'] = self.counter.get('force_calls', 0) + 1


def xyz_of(atoms: Atoms, xyz: dict) -> dict:
    """The ARC xyz dict of ``atoms``, with the symbols and isotopes of ``xyz``."""
    return {'coords': tuple(map(tuple, atoms.get_positions().tolist())),
            'symbols': tuple(xyz['symbols']),
            'isotopes': tuple(xyz.get('isotopes') or [None] * len(xyz['symbols']))}


def relax_endpoint(image: Atoms, attach, log_path: str, bonds: list | None = None, fragments: list | None = None,
                   fmax: float = 0.05, steps: int = 300) -> tuple[Atoms, str | None]:
    """
    Relax a band endpoint on the band's calculator (BFGS, to ``fmax`` eV/A within ``steps`` steps).

    A one-molecule endpoint is relaxed as a whole, and the relaxed geometry is kept only if every bond of ``bonds``
    (pairs of atom indices) stays shorter than 1.25 times its length before. A multi-molecule endpoint is relaxed
    fragment by fragment: each entry of ``fragments`` (``atoms``, ``charge``, ``multiplicity``) is relaxed on its own,
    with its own charge and spin, and then moved rigidly back onto its atoms' places. An endpoint with neither is not
    relaxed.

    Args:
        image (Atoms): The endpoint.
        attach (Callable): Attaches the band's calculator to a list of images.
        log_path (str): The optimizer's log file.
        bonds (list, optional): The bonds of a one-molecule endpoint.
        fragments (list, optional): The fragments of a multi-molecule endpoint.
        fmax (float, optional): The force criterion.
        steps (int, optional): The most optimizer steps.

    Returns:
        tuple[Atoms, str | None]: The endpoint, and a warning if it was kept unrelaxed.
    """
    from ase.build import minimize_rotation_and_translation
    if bonds:
        relaxed = image.copy()
        attach([relaxed])
        BFGS(relaxed, logfile=log_path).run(fmax=fmax, steps=steps)
        broken = [(i, j) for i, j in bonds if relaxed.get_distance(i, j) > 1.25 * image.get_distance(i, j)]
        if broken:
            return image, f'relaxing the endpoint broke its bond(s) {broken}; it was kept unrelaxed'
        return relaxed, None
    if fragments:
        relaxed = image.copy()
        attach([relaxed])
        for fragment in fragments:
            atoms = list(fragment['atoms'])
            if len(atoms) < 2:
                continue
            piece = image[atoms]
            attach([piece])
            piece.info.update({'charge': fragment.get('charge', 0), 'spin': fragment.get('multiplicity', 1)})
            BFGS(piece, logfile=log_path).run(fmax=fmax, steps=steps)
            minimize_rotation_and_translation(image[atoms], piece)
            positions = relaxed.get_positions()
            positions[atoms] = piece.get_positions()
            relaxed.set_positions(positions)
        return relaxed, None
    return image, None


def climbing_image_force(images: list, k: float, method: str) -> float:
    """
    The largest atomic force (eV/A) on the climbing image of a band, its highest interior image, under ASE's
    climbing-image NEB projection (independent of the springs).
    """
    from ase.mep import NEB
    band = NEB(images, k=k, climb=True, method=method, remove_rotation_and_translation=False)
    forces = band.get_forces().reshape(len(images) - 2, len(images[0]), 3)
    highest = int(np.argmax([image.get_potential_energy() for image in images[1:-1]]))
    return float(np.max(np.linalg.norm(forces[highest], axis=1)))


def run_band(atoms: Atoms, xyz: dict, product_xyz: dict, settings: dict, job_dir: str,
             reactant_structure: dict | None = None, product_structure: dict | None = None) -> dict:
    """
    One nudged elastic band from ``xyz`` (the reactant) to ``product_xyz``, with ASE's NEB, DyNEB or AutoNEB on the
    calculator of ``atoms`` (whose positions are not used; its ``info`` carries the charge and spin).

    The endpoints must list the same atoms in the same order. Unless ``neb_relax_endpoints`` is False, each is first
    relaxed on the band's calculator (see ``relax_endpoint()``, given ``reactant_structure`` and ``product_structure``,
    each a dict with the endpoint's ``bonds`` or ``fragments``), and the product is then aligned onto the reactant.
    The endpoints then stay fixed. The interior images start from an image-dependent pair potential (IDPP)
    interpolation. NEB and DyNEB relax the band first without a climbing image (to ``neb_fmax_pre``), then with one
    (to ``neb_fmax``). AutoNEB starts from the endpoints and adds images where the band needs them, up to
    ``neb_images``; it is converged if the force on its climbing image is within ``neb_fmax`` at its end. If AutoNEB fails an
    assertion, the band is relaxed as a NEB instead. The highest interior image is the TS guess, which is successful
    only if its energy is above both endpoints.

    Settings (optional): ``neb_method`` ('neb', 'dyneb' or 'autoneb'; default 'neb'), ``neb_images`` (interior images;
    default 10), ``neb_fmax`` (default 0.05 eV/A), ``neb_fmax_pre`` (default 0.1 eV/A), ``neb_steps`` (per stage;
    default 500), ``neb_k`` (spring constant; default 0.1 eV/A^2), ``neb_tangent`` (ASE's ``method``; default
    'improvedtangent', and 'eb' for AutoNEB), ``neb_relax_endpoints`` (default True).

    Returns:
        dict: ``ts_guesses`` (the highest image as an ARC TSGuess dict), ``neb_energies`` (kJ/mol relative to the
              reactant), ``neb_path`` (the final images), ``neb_highest_image``, ``neb_converged``,
              ``neb_force_calls``, ``opt_xyz`` (the highest image), and ``warnings`` or ``error``.
    """
    from ase.build import minimize_rotation_and_translation
    from ase.mep import NEB, DyNEB
    from ase.optimize import FIRE

    if not product_xyz or tuple(product_xyz['symbols']) != tuple(xyz['symbols']):
        return {'error': 'A NEB job needs a product_xyz with the same atoms in the same order as its xyz.'}
    method = str(settings.get('neb_method', 'neb')).lower()
    n_images = int(settings.get('neb_images', 10))
    fmax, fmax_pre = float(settings.get('neb_fmax', 0.05)), float(settings.get('neb_fmax_pre', 0.1))
    steps, k = int(settings.get('neb_steps', 500)), float(settings.get('neb_k', 0.1))
    shared, counter = atoms.calc, dict()

    def attach(images):
        for image in images:
            image.info.update(atoms.info)
            image.calc = ImageCalculator(shared, counter)

    reactant = Atoms(symbols=xyz['symbols'], positions=xyz['coords'])
    product = Atoms(symbols=product_xyz['symbols'], positions=product_xyz['coords'])
    attach([reactant, product])
    warnings, t0 = list(), time.time()
    if settings.get('neb_relax_endpoints', True):
        for name, structure in (('reactant', reactant_structure), ('product', product_structure)):
            image = reactant if name == 'reactant' else product
            relaxed, warning = relax_endpoint(image, attach, os.path.join(job_dir, f'relax_{name}.log'),
                                              **(structure or dict()))
            if warning:
                warnings.append(f'The {name} endpoint: {warning}.')
            if name == 'reactant':
                reactant = relaxed
            else:
                product = relaxed
        minimize_rotation_and_translation(reactant, product)
    tangent = settings.get('neb_tangent', 'eb' if method == 'autoneb' else 'improvedtangent')
    if method == 'autoneb':
        try:
            from ase.io import write
            from ase.mep.autoneb import AutoNEB
            prefix = os.path.join(job_dir, 'neb')
            for index, image in enumerate((reactant, product)):
                image.get_forces()
                write(f'{prefix}{index:03d}.traj', image)
            autoneb = AutoNEB(attach, prefix=prefix, n_simul=1, n_max=n_images + 2, fmax=fmax, maxsteps=steps, k=k,
                              climb=True, method=tangent, parallel=False, remove_rotation_and_translation=False,
                              interpolate_method='idpp', iter_folder=os.path.join(job_dir, 'AutoNEB_iter'))
            autoneb.run()
            images = autoneb.all_images
            attach(images)
            converged = climbing_image_force(images, k, tangent) <= fmax
            if not converged:
                warnings.append(f'AutoNEB ended above fmax {fmax} eV/A; its highest image is still reported.')
        except AssertionError as exc:
            warnings.append(f'AutoNEB stopped ({exc}); the band was relaxed again as a NEB.')
            method, tangent = 'neb', settings.get('neb_tangent', 'improvedtangent')
        except Exception as exc:
            return {'error': f'NEB failed: {type(exc).__name__}: {exc}'}
    if method != 'autoneb':
        try:
            neb_kwargs = dict(k=k, method=tangent, remove_rotation_and_translation=True)
            images = [reactant] + [reactant.copy() for _ in range(n_images)] + [product]
            attach(images[1:-1])
            NEB(images, **neb_kwargs).interpolate(method='idpp')
            band = DyNEB(images, fmax=fmax, dynamic_relaxation=True, **neb_kwargs) if method == 'dyneb' \
                else NEB(images, **neb_kwargs)
            log = os.path.join(job_dir, 'neb.log')
            FIRE(band, logfile=log).run(fmax=fmax_pre, steps=steps)
            band.climb = True
            converged = FIRE(band, logfile=log).run(fmax=fmax, steps=steps)
            if not converged:
                warnings.append(f'The NEB did not converge to fmax {fmax} eV/A in {steps} climbing-image steps; '
                                f'its highest image is still reported as a TS guess.')
        except Exception as exc:
            return {'error': f'NEB failed: {type(exc).__name__}: {exc}'}

    energies = [image.get_potential_energy() for image in images]
    energies = [to_kJmol(energy - energies[0]) for energy in energies]
    highest = 1 + int(np.argmax(energies[1:-1]))
    has_barrier = energies[highest] > max(energies[0], energies[-1])
    if not has_barrier:
        warnings.append('The NEB path has no interior maximum above both endpoints (a barrierless or monotonic path); '
                        'its highest interior image is reported as an unsuccessful TS guess.')
    guess = {'method': 'ase_neb', 'success': has_barrier, 'energy': energies[highest],
             'initial_xyz': xyz_of(images[highest], xyz),
             'execution_time': str(datetime.timedelta(seconds=round(time.time() - t0)))}
    out = {'ts_guesses': [guess],
           'neb_energies': energies,
           'neb_path': [xyz_of(image, xyz) for image in images],
           'neb_highest_image': highest,
           'neb_converged': bool(converged),
           'neb_force_calls': counter.get('force_calls', 0),
           'opt_xyz': xyz_of(images[highest], xyz)}
    if warnings:
        out['warnings'] = warnings
    return out


def run_neb(atoms: Atoms, input_dict: dict, settings: dict, job_dir: str) -> dict:
    """
    Nudged elastic bands for a 'neb' job: one per entry of ``bands`` (each an ``xyz`` and a ``product_xyz``, e.g. from
    different atom maps of the reaction, and optionally the ``reactant_structure`` and ``product_structure`` that
    relax the endpoints), or one between the job's ``xyz`` and ``product_xyz``. See run_band().

    Each band's highest interior image is a TS guess (``method_index`` = the band's index). Each band keeps its files in
    ``band_<i>``. The top-level ``neb_*`` keys and ``opt_xyz`` are those of the band with the lowest highest image.

    Returns:
        dict: ``ts_guesses``, ``neb_bands`` (a summary per band), ``neb_energies``, ``neb_path``, ``neb_highest_image``,
              ``neb_converged``, ``neb_force_calls`` (all bands), ``opt_xyz``, and ``warnings`` or ``error``.
    """
    bands = input_dict.get('bands') or [{'xyz': input_dict['xyz'], 'product_xyz': input_dict.get('product_xyz')}]
    results, warnings = list(), list()
    for index, band in enumerate(bands):
        band_dir = os.path.join(job_dir, f'band_{index}')
        os.makedirs(band_dir, exist_ok=True)
        result = run_band(atoms, band['xyz'], band.get('product_xyz'), settings, band_dir,
                          reactant_structure=band.get('reactant_structure'),
                          product_structure=band.get('product_structure'))
        results.append(result)
        if 'error' in result:
            warnings.append(f"Band {index}: {result['error']}")
        warnings.extend(f'Band {index}: {w}' for w in result.get('warnings', list()))
    done = [(i, r) for i, r in enumerate(results) if 'error' not in r]
    if not done:
        return {'error': '; '.join(warnings) or 'No NEB band ran.'}
    for index, result in done:
        result['ts_guesses'][0]['method_index'] = index
    best_index, best = min(done, key=lambda item: item[1]['neb_energies'][item[1]['neb_highest_image']])
    out = {'ts_guesses': [r['ts_guesses'][0] for _, r in done],
           'neb_bands': [{'band': i, 'energies': r.get('neb_energies'), 'highest_image': r.get('neb_highest_image'),
                          'converged': r.get('neb_converged'), 'force_calls': r.get('neb_force_calls'),
                          'error': r.get('error')} for i, r in enumerate(results)],
           'neb_best_band': best_index,
           'neb_force_calls': sum(r.get('neb_force_calls', 0) for _, r in done)}
    for key in ('neb_energies', 'neb_path', 'neb_highest_image', 'neb_converged', 'opt_xyz'):
        out[key] = best[key]
    if warnings:
        out['warnings'] = warnings
    return out


def resolve_input_path(yml_path: str) -> str:
    """
    Resolve the path of a job's input.yml, given it or its directory.
    """
    input_path = os.path.abspath(yml_path)
    if os.path.isdir(input_path):
        input_path = os.path.join(input_path, 'input.yml')
    return input_path


def run_job(input_path: str, input_dict: dict | None = None, cache: dict | None = None,
            evaluator: BatchedEvaluator | None = None) -> dict:
    """
    Run one ASE job and write its output.yml next to its input.yml.

    Args:
        input_path (str): The path to the job's input.yml.
        input_dict (dict, optional): The parsed input.yml. Read from ``input_path`` if not given.
        cache (dict, optional): Loaded models, kept across jobs by a persistent worker.
        evaluator (BatchedEvaluator, optional): For a scan, an evaluator shared with other scans.

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
    if input_dict.get('masses'):
        atoms.set_masses(input_dict['masses'])
    atoms.info.update({'charge': charge, 'spin': multiplicity})  # UMA (omol) conditions on these
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
            output.update(run_torsion_scan(atoms, input_dict, settings, evaluator=evaluator, cache=cache))
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
        fmax = float(settings.get('irc_fmax', 0.01))
        steps = int(settings.get('steps', 1000))
        direction = input_dict.get('irc_direction', 'forward')
        traj_path = os.path.join(os.path.dirname(input_path), 'irc.traj')
        irc_issues, run_error = list(), None
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                irc = IRC(atoms, logfile=os.path.join(os.path.dirname(input_path), 'irc.log'),
                          trajectory=traj_path, keep_going=True)
                irc.run(fmax=fmax, steps=steps, direction=direction)
            except Exception as exc:
                run_error = f'{type(exc).__name__}: {exc}'
        n_inner_failures = sum('trustworthy IRC' in str(warning.message) for warning in caught)
        if n_inner_failures:
            irc_issues.append(f'the IRC inner loop failed to converge on {n_inner_failures} step(s), '
                              f'so the path past them is not a trustworthy IRC')
        try:
            images = read_trajectory(traj_path)
        except Exception as exc:
            images = list()
            output['error'] = f'IRC trajectory {traj_path} could not be read: {type(exc).__name__}: {exc}'
        if run_error is not None:
            if len(images) > 1:
                irc_issues.append(f'the IRC stopped after {len(images)} points: {run_error}')
            elif 'error' not in output:
                output['error'] = f'IRC failed: {run_error}'
        if irc_issues:
            output['irc_complete'] = False
            output.setdefault('warnings', list()).extend(irc_issues)
        if len(images) > 1:
            save_current_geometry(output, images[-1], xyz)
            output['irc_traj'] = [
                {'coords': tuple(map(tuple, image.get_positions().tolist())),
                 'symbols': xyz['symbols'],
                 'isotopes': xyz.get('isotopes') or tuple([None] * len(xyz['symbols']))}
                for image in images]

    if job_type == 'neb':
        output.update(run_neb(atoms, input_dict, settings, job_dir))

    if job_type in ['freq', 'optfreq']:
        try:
            freq_results = run_vibrational_analysis(atoms, settings, work_dir=job_dir, cache=cache)
            output.update(freq_results)
        except Exception as exc:
            output['error'] = output.get('error', '') + f" Frequency calculation failed: {exc}"

    output_path = os.path.join(job_dir, 'output.yml')
    save_yaml_file(output_path, output)
    return output


SPOOL_SUBDIRS = ('queue', 'running', 'done', 'workers', 'tmp')
HEARTBEAT_INTERVAL = 5.0
HEARTBEAT_STALE = 30.0


def spool_dirs(spool_dir: str) -> dict:
    """
    Return the subdirectories of a persistent worker's spool by name, creating any that are missing.

    A ticket is a JSON file naming a job's input.yml. It is submitted to ``queue/``, moved to ``running/``
    by the worker that claims it (an atomic rename, so several workers can share a spool), and finished
    as ``done/<ticket>.json`` with ``'status'`` ``'done'`` or ``'errored'``. A worker keeps the mtime of
    ``workers/<worker>.json`` fresh every ``HEARTBEAT_INTERVAL`` seconds while it lives, and is presumed
    dead after ``HEARTBEAT_STALE`` seconds without a heartbeat. Files are written in ``tmp/`` and renamed
    into place.
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


def submit_ticket(spool_dir: str, input_path: str, label: str = '', max_run_time: float | None = None) -> str:
    """
    Submit a job to a spool.

    Args:
        spool_dir (str): The spool directory.
        input_path (str): The absolute path to the job's input.yml.
        label (str, optional): A readable label (e.g., the job name), used in the ticket ID.
        max_run_time (float, optional): Seconds after submission at which the ticket's ``deadline`` passes.

    Returns:
        str: The ticket ID.
    """
    dirs = spool_dirs(spool_dir)
    ticket = f'{label}_{uuid.uuid4().hex[:12]}' if label else uuid.uuid4().hex
    submitted = time.time()
    content = {'input_path': input_path, 'submitted': submitted}
    if max_run_time is not None:
        content['deadline'] = submitted + max_run_time
    write_json_atomic(os.path.join(dirs['queue'], f'{ticket}.json'), content, dirs['tmp'])
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
    The directories are read in the order a ticket moves through them: queue, running, done.

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
    with contextlib.suppress(FileNotFoundError):
        os.remove(os.path.join(dirs['running'], f'{ticket}.json'))


def ticket_is_live(state: str, content: dict | None, workers: list, now: float | None = None) -> bool:
    """
    Whether a queued or running ticket can still finish on its spool: its deadline has not passed, and a worker
    can claim it (queued: any live worker) or runs it (running: its worker is live; any live worker while the
    claiming worker has not yet recorded itself in the ticket). A ticket whose content is ``None`` is live.

    Args:
        state (str): The ticket's state, 'queued' or 'running'.
        content (dict): The ticket's content.
        workers (list): The IDs of the spool's live workers.
        now (float, optional): The current time.

    Returns:
        bool: Whether the ticket is live.
    """
    if content is None:
        return True
    if content.get('deadline') is not None and (now or time.time()) > content['deadline']:
        return False
    if state == 'queued' or content.get('worker') is None:
        return bool(workers)
    return content['worker'] in workers


def live_tickets(spool_dir: str) -> list:
    """
    The IDs of a spool's queued and running tickets that are live (see ``ticket_is_live()``).
    A ticket that leaves its directory while the spool is read is included.
    """
    workers, now, tickets = live_workers(spool_dir), time.time(), list()
    for directory, state in (('queue', 'queued'), ('running', 'running')):
        path = os.path.join(spool_dir, directory)
        if not os.path.isdir(path):
            continue
        for file_name in os.listdir(path):
            if file_name.endswith('.json') \
                    and ticket_is_live(state, read_json(os.path.join(path, file_name)), workers, now):
                tickets.append(file_name[:-len('.json')])
    return tickets


def claim_ticket(spool_dir: str, worker_id: str, accept: Callable | None = None) -> tuple | None:
    """
    Claim the oldest queued ticket for a worker.

    Args:
        spool_dir (str): The spool directory.
        worker_id (str): The worker's ID.
        accept (Callable, optional): Claim only a ticket whose content this function of the content accepts.

    Returns:
        tuple: The ticket ID and its content, or ``None`` if nothing (acceptable) is queued.
    """
    dirs = spool_dirs(spool_dir)
    queued = list()
    for file_name in os.listdir(dirs['queue']):
        try:
            queued.append((os.path.getmtime(os.path.join(dirs['queue'], file_name)), file_name))
        except FileNotFoundError:
            continue
    for _, file_name in sorted(queued):
        if accept is not None and not accept(read_json(os.path.join(dirs['queue'], file_name)) or dict()):
            continue
        running_path = os.path.join(dirs['running'], file_name)
        try:
            os.rename(os.path.join(dirs['queue'], file_name), running_path)
        except FileNotFoundError:
            continue
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


def run_ticket(spool_dir: str, ticket: str, content: dict, worker_id: str, cache: dict,
               evaluator: BatchedEvaluator | None = None) -> dict:
    """
    Run a claimed ticket's job in its own directory and mark the ticket as finished.
    Any failure, including a ``sys.exit()`` in the job, is recorded in the ticket instead of ending the worker.
    A job run with a shared ``evaluator`` (a scan of a group, in a thread) does not change the working directory.

    Returns:
        dict: The ticket's final content.
    """
    t0, cwd = time.time(), os.getcwd()
    result = {'input_path': content.get('input_path'), 'worker': worker_id, 'status': 'done'}
    try:
        if evaluator is None:
            os.chdir(os.path.dirname(content['input_path']))
        run_job(content['input_path'], cache=cache, evaluator=evaluator)
    except (Exception, SystemExit):
        result.update({'status': 'errored', 'error': traceback.format_exc()})
    finally:
        if evaluator is None:
            os.chdir(cwd)
    finished = time.time()
    result.update({'started': t0, 'finished': finished, 'run_time': finished - t0})
    finish_ticket(spool_dir, ticket, result)
    return result


def scan_group_key(input_path: str) -> str | None:
    """
    The key under which a worker runs a scan job together with other scans, or ``None`` for a job it runs alone.

    Only UMA scans of the same model and settings are grouped.

    Args:
        input_path (str): The path to the job's input.yml.
    """
    try:
        input_dict = read_yaml_file(input_path)
    except Exception:
        return None
    settings = input_dict.get('settings') or dict()
    if input_dict.get('job_type') != 'scan' or str(settings.get('calculator', '')).lower() not in ('uma', 'fairchem'):
        return None
    return json.dumps({key: settings.get(key) for key in ('calculator', 'model', 'task', 'device', 'inference_settings',
                                                          'hessian', 'hessian_vmap')}, sort_keys=True, default=str)


def scan_group_evaluator(input_paths: list, cache: dict) -> BatchedEvaluator:
    """
    One evaluator for UMA scan jobs with the same settings (see ``scan_group_key()``): their walks, two per torsion,
    all go to the model in one batch. The Hessian predictor is loaded only if one of the jobs scans a TS.

    Args:
        input_paths (list): The jobs' input.yml paths.
        cache (dict): Loaded models, kept across jobs by a persistent worker.
    """
    input_dicts = [read_yaml_file(path) for path in input_paths]
    settings = input_dicts[0].get('settings') or dict()
    calc = get_calculator(settings, cache=cache)
    hessian = sella_hessian_function(settings, cache) if any(d.get('is_ts') for d in input_dicts) else None
    return BatchedEvaluator(batch_evaluation_function(calc), hessian=hessian)


def run_scan_group(spool_dir: str, group: list, worker_id: str, cache: dict) -> list:
    """
    Run claimed scan tickets at the same time, one thread per job, sharing one ``BatchedEvaluator``.

    Args:
        spool_dir (str): The spool directory.
        group (list): The (ticket, content) of each claimed ticket, all with the same ``scan_group_key()``.
        worker_id (str): The worker's ID.
        cache (dict): Loaded models, kept across jobs by a persistent worker.

    Returns:
        list: The tickets' final contents, in the order of ``group``.
    """
    try:
        evaluator = scan_group_evaluator([content['input_path'] for _, content in group], cache)
    except (Exception, SystemExit):
        return [run_ticket(spool_dir, ticket, content, worker_id, cache) for ticket, content in group]
    with ThreadPoolExecutor(max_workers=len(group)) as pool:
        futures = [pool.submit(run_ticket, spool_dir, ticket, content, worker_id, cache, evaluator)
                   for ticket, content in group]
        return [future.result() for future in futures]


def serve(spool_dir: str, max_jobs: int | None = None, idle_timeout: float | None = None,
          poll_interval: float = 0.5, max_scan_group: int = 64, scan_group_wait: float = 2.0) -> int:
    """
    Run as a persistent worker: claim and run the spool's tickets one at a time, keeping loaded models.
    SIGTERM or SIGINT stops the worker after its current job; a second signal acts as usual.

    UMA scans are the exception: having claimed one, the worker also claims the other queued scans with the same
    settings, waiting ``scan_group_wait`` seconds for more (ARC submits a species' rotor scans together), and runs
    them at the same time, all their walks sharing each batched force call (see ``run_scan_group()``).

    Args:
        spool_dir (str): The spool directory.
        max_jobs (int, optional): Stop after this many jobs.
        idle_timeout (float, optional): Stop after this many seconds without a ticket.
        poll_interval (float, optional): Seconds between checks of an empty queue.
        max_scan_group (int, optional): The most scans run at the same time (1 runs every scan alone).
        scan_group_wait (float, optional): Seconds to wait for more scans to run with a claimed one.

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
            group = [claimed]
            key = scan_group_key(claimed[1].get('input_path', '')) if max_scan_group > 1 else None
            if key is not None:
                limit = max_scan_group if max_jobs is None else min(max_scan_group, max_jobs - n_jobs)
                deadline = time.time() + scan_group_wait
                while len(group) < limit and not stopping.is_set():
                    more = claim_ticket(spool_dir, worker_id,
                                        accept=lambda c: scan_group_key(c.get('input_path', '')) == key)
                    if more is not None:
                        group.append(more)
                    elif time.time() < deadline:
                        stopping.wait(min(poll_interval, 0.2))
                    else:
                        break
            if len(group) == 1:
                results = [run_ticket(spool_dir, *claimed, worker_id, cache)]
            else:
                results = run_scan_group(spool_dir, group, worker_id, cache)
            for (ticket, _), result in zip(group, results):
                print(f'{ticket}: {result["status"]} in {result["run_time"]:.1f} s', flush=True)
            n_jobs, idle_since = n_jobs + len(group), time.time()
    finally:
        stop_heartbeat.set()
        with contextlib.suppress(FileNotFoundError):
            os.remove(heartbeat_path)
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
    parser.add_argument('--max_scan_group', type=int, default=64,
                        help='Worker: the most UMA scans run at the same time, sharing batched force calls.')
    args = parser.parse_args()

    if args.serve:
        serve(args.serve, max_jobs=args.max_jobs, idle_timeout=args.idle_timeout, max_scan_group=args.max_scan_group)
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
