#!/usr/bin/env python3
# encoding: utf-8

"""
A standalone script to run PySCF electronic-structure jobs (sp, opt, freq, optfreq).

Executed inside the dedicated ``pyscf_env`` conda environment (isolated from ARC's env).
Reads an ``input.yml`` in ARC's internal format and writes an ``output.yml`` in the
in-core ESS schema (v1) that ARC's ``YAMLParser`` re-parses.

Level of theory is taken from the input (ARC routes wb97m-v/def2-tzvp here). Closed-shell
species use RKS with an analytic Hessian for frequencies; open-shell species use UKS with a
numerical Hessian (central finite difference of analytic gradients), because PySCF has no
analytic Hessian for NLC (VV10) functionals in the unrestricted case.
"""

import argparse
import os

import numpy as np
import yaml

from pyscf import dft, gto, lib
from pyscf.geomopt.geometric_solver import optimize
from pyscf.hessian import thermo

# Physical constants (CODATA 2018), hard-coded so this script needs no ARC imports.
HARTREE2KJMOL = 2625.4996394798254
BOHR2ANG = 0.52917721090380
CM2KJMOL = 0.0119626565  # kJ/mol per cm^-1 (= h*c*N_A*100/1000)

# Numerical Hessian finite-difference step (Angstrom), used for open-shell (UKS) freqs.
NUM_HESS_DISP_ANG = 0.005

# XC grid level for a "fine" job: the closest PySCF match to Gaussian's UltraFine grid.
FINE_GRIDS_LEVEL = 4


def read_yaml_file(path):
    """
    Read a YAML file.

    Args:
        path (str): The file path.

    Returns:
        The parsed content.
    """
    with open(path, 'r') as f:
        return yaml.load(stream=f, Loader=yaml.FullLoader)


def save_yaml_file(path, content):
    """
    Save a dictionary as a YAML file.

    Args:
        path (str): The file path.
        content (dict): The content to dump.
    """
    with open(path, 'w') as f:
        f.write(yaml.dump(data=content, default_flow_style=False))


def normalize_basis(basis):
    """
    Normalize an ARC basis-set string to a name PySCF recognizes.

    ARC stores def2 basis sets without a hyphen (e.g. ``def2tzvp``) while PySCF expects
    ``def2-tzvp``. Unknown names are passed through lower-cased.

    Args:
        basis (str): The ARC basis-set string.

    Returns:
        str: A PySCF-compatible basis-set string.
    """
    if not basis:
        return 'def2-tzvp'
    b = basis.lower().strip()
    mapping = {'def2svp': 'def2-svp', 'def2svpd': 'def2-svpd',
               'def2tzvp': 'def2-tzvp', 'def2tzvpp': 'def2-tzvpp', 'def2tzvpd': 'def2-tzvpd',
               'def2qzvp': 'def2-qzvp', 'def2qzvpp': 'def2-qzvpp'}
    return mapping.get(b, b)


def build_mol(xyz, charge, multiplicity, basis, memory_mb=None):
    """
    Build a PySCF ``Mole`` from an ARC xyz dictionary.

    Args:
        xyz (dict): ARC xyz dict with ``symbols`` and ``coords`` (Angstrom).
        charge (int): The net molecular charge.
        multiplicity (int): The spin multiplicity (2S+1).
        basis (str): The basis-set name.
        memory_mb (int, optional): Max memory in MB.

    Returns:
        gto.Mole: The built molecule.
    """
    atoms = [[sym, tuple(coord)] for sym, coord in zip(xyz['symbols'], xyz['coords'])]
    mol = gto.Mole()
    mol.atom = atoms
    mol.unit = 'Angstrom'
    mol.charge = int(charge)
    mol.spin = int(multiplicity) - 1  # number of unpaired electrons (2S)
    mol.basis = normalize_basis(basis)
    if memory_mb:
        mol.max_memory = float(memory_mb)
    mol.verbose = 0
    mol.build()
    return mol


def make_mf(mol, xc, multiplicity, settings=None):
    """
    Construct a DFT mean-field object (RKS for closed-shell, UKS for open-shell).

    Numerical knobs (integration-grid levels, SCF convergence) default to PySCF's own defaults
    and are only overridden when explicitly requested, so results stay reproducible.

    A ``fine`` job raises the XC grid to level 4, the closest PySCF analogue of the
    ``integral=(grid=ultrafine)`` grid ARC's Gaussian adapter requests for fine jobs:
    level 4 is (90, 590) radial x angular points for a second-row atom vs. Gaussian
    UltraFine's (99, 590), while PySCF's own default level 3 is (75, 302), matching
    Gaussian's plain FineGrid. Level 5 would overshoot UltraFine by ~1.5x in grid points.
    The NLC (VV10) grid is left at PySCF's default level 3, since the nonlocal correlation
    kernel is smooth and conventionally integrated on a coarser grid than the XC term.

    Args:
        mol (gto.Mole): The molecule.
        xc (str): The exchange-correlation functional.
        multiplicity (int): The spin multiplicity.
        settings (dict, optional): Optional numerical settings (``fine``, ``grids_level``,
                                   ``nlcgrids_level``, ``conv_tol``, ``max_cycle``).

    Returns:
        The configured PySCF mean-field object.
    """
    settings = settings or dict()
    mf = dft.UKS(mol) if int(multiplicity) > 1 else dft.RKS(mol)
    mf.xc = xc.lower()
    mf.verbose = 0
    fine = bool(settings.get('fine', False))
    grids_level = settings.get('grids_level', FINE_GRIDS_LEVEL if fine else None)
    nlcgrids_level = settings.get('nlcgrids_level')
    if grids_level is not None:
        mf.grids.level = int(grids_level)
    if nlcgrids_level is not None and mf.do_nlc():
        mf.nlcgrids.level = int(nlcgrids_level)
    if settings.get('conv_tol') is not None:
        mf.conv_tol = float(settings['conv_tol'])
    if settings.get('max_cycle') is not None:
        mf.max_cycle = int(settings['max_cycle'])
    return mf


def mol_to_xyz_dict(mol, xyz_in):
    """
    Convert an optimized ``Mole`` geometry to an ARC xyz dict, preserving isotopes.

    Args:
        mol (gto.Mole): The molecule (geometry in Bohr internally).
        xyz_in (dict): The input ARC xyz dict (for symbols and isotopes).

    Returns:
        dict: An ARC xyz dict with Angstrom coordinates.
    """
    coords = mol.atom_coords(unit='ANG')
    symbols = tuple(xyz_in['symbols'])
    isotopes = tuple(xyz_in.get('isotopes')) if xyz_in.get('isotopes') else tuple([None] * len(symbols))
    return {'symbols': symbols,
            'isotopes': isotopes,
            'coords': tuple(tuple(float(x) for x in row) for row in coords)}


def get_dipole_debye(mf):
    """
    Compute the dipole-moment magnitude in Debye.

    Args:
        mf: A converged mean-field object.

    Returns:
        float: The dipole magnitude in Debye.
    """
    dip = mf.dip_moment(unit='Debye', verbose=0)
    return float(np.linalg.norm(np.asarray(dip, dtype=float)))


def write_geometric_constraints(constraints, path):
    """
    Write ARC internal-coordinate constraints to a geomeTRIC ``$set`` constraints file.

    ARC constraints are (indices, value) tuples with 1-indexed atoms; a 2/3/4-atom index list
    denotes a bond (Angstrom) / angle (degrees) / dihedral (degrees). geomeTRIC atom indices
    are 1-indexed, matching ARC.

    Args:
        constraints (list): The ARC constraints.
        path (str): The output constraints-file path.

    Returns:
        str: The constraints-file path.
    """
    lines = ['$set']
    for indices, value in constraints:
        atoms = ' '.join(str(int(i)) for i in indices)
        kind = {2: 'distance', 3: 'angle', 4: 'dihedral'}[len(indices)]
        lines.append(f'{kind} {atoms} {float(value)}')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    return path


def run_opt(mf, xyz_in, multiplicity, is_ts=False, constraints=None, work_dir='.', settings=None):
    """
    Run a geomeTRIC geometry optimization and return a fresh converged mean-field.

    Transition-state searches use tightened trust radii, which are more robust for saddle-point
    optimization than geomeTRIC's minimization defaults.

    Args:
        mf: The starting mean-field object.
        xyz_in (dict): The input ARC xyz dict.
        multiplicity (int): The spin multiplicity.
        is_ts (bool): Whether to search for a first-order saddle point (TS).
        constraints (list, optional): ARC internal-coordinate constraints.
        work_dir (str): Directory for the geomeTRIC constraints file.
        settings (dict, optional): Numerical settings forwarded to the mean-field builder.

    Returns:
        tuple: (mf_opt, opt_xyz_dict) where mf_opt is converged at the optimized geometry.
    """
    settings = settings or dict()
    params = {}
    if is_ts:
        params.update({'transition': True, 'trust': 0.02, 'tmax': 0.06})
    if constraints:
        params['constraints'] = write_geometric_constraints(
            constraints, os.path.join(work_dir, 'constraints.txt'))
    mol_eq = optimize(mf, maxsteps=int(settings.get('maxsteps', 250)), **params)
    mf_opt = make_mf(mol_eq, mf.xc, multiplicity, settings=settings)
    mf_opt.kernel()
    return mf_opt, mol_to_xyz_dict(mol_eq, xyz_in)


def numerical_hessian(mf, disp_ang=NUM_HESS_DISP_ANG, settings=None):
    """
    Compute a numerical Hessian by central finite difference of analytic nuclear gradients.

    Used for open-shell (UKS) NLC functionals, for which PySCF has no analytic Hessian.

    Args:
        mf: A converged mean-field object.
        disp_ang (float): The displacement step in Angstrom.
        settings (dict, optional): Numerical settings forwarded to the mean-field builder.

    Returns:
        np.ndarray: The Hessian, shape (natm, natm, 3, 3), in Hartree/Bohr^2.
    """
    settings = settings or dict()
    mol = mf.mol
    natm = mol.natm
    coords0 = mol.atom_coords()  # Bohr
    d = disp_ang / BOHR2ANG
    hess = np.zeros((natm, natm, 3, 3))
    for a in range(natm):
        for x in range(3):
            grads = {}
            for sign in (1, -1):
                c = coords0.copy()
                c[a, x] += sign * d
                m = mol.copy()
                m.set_geom_(c, unit='Bohr')
                m.build(False, False)
                mf_d = make_mf(m, mf.xc, mol.spin + 1, settings=settings)
                mf_d.kernel()
                grads[sign] = mf_d.nuc_grad_method().kernel()
            hess[a, :, x, :] = (grads[1] - grads[-1]) / (2 * d)
    return 0.5 * (hess + hess.transpose(1, 0, 3, 2))


def run_freq(mf, multiplicity, settings=None):
    """
    Compute harmonic frequencies, normal modes, and ZPE at the current geometry.

    Closed-shell species use the analytic Hessian; open-shell species use a numerical Hessian.
    Imaginary frequencies are returned as negative reals (ARC convention).

    Args:
        mf: A converged mean-field object.
        multiplicity (int): The spin multiplicity.
        settings (dict, optional): Numerical settings forwarded to the mean-field builder.

    Returns:
        dict: Keys ``freqs`` (cm^-1), ``modes``, and ``zpe`` (kJ/mol).
    """
    if int(multiplicity) > 1:
        hess = numerical_hessian(mf, settings=settings)
    else:
        hess = mf.Hessian().kernel()
    results = thermo.harmonic_analysis(mf.mol, hess)
    raw = np.asarray(results['freq_wavenumber'], dtype=complex)
    freqs = [float(f.real) if abs(f.imag) < 1e-6 else -float(abs(f.imag)) for f in raw]
    modes = np.asarray(results['norm_mode'], dtype=float).tolist()
    zpe = 0.5 * sum(f for f in freqs if f > 0) * CM2KJMOL
    return {'freqs': freqs, 'modes': modes, 'zpe': float(zpe)}


def run_job(input_dict, work_dir):
    """
    Run the requested PySCF job and assemble the output dictionary.

    Args:
        input_dict (dict): The parsed ``input.yml`` content.
        work_dir (str): The working directory (for scratch files).

    Returns:
        dict: The schema-v1 output dictionary.
    """
    job_type = input_dict.get('job_type')
    xyz = input_dict.get('xyz')
    charge = input_dict.get('charge', 0)
    multiplicity = input_dict.get('multiplicity', 1)
    is_ts = input_dict.get('is_ts', False)
    constraints = input_dict.get('constraints') or list()
    settings = input_dict.get('settings') or dict()
    xc = settings.get('method', 'wb97m-v')
    basis = settings.get('basis', 'def2-tzvp')
    memory_mb = settings.get('memory_mb')

    output = {'schema_version': 1, 'adapter': 'pyscf', 'success': True, 'error': None}

    # Pin the thread count so concurrent local PySCF jobs cannot each grab every core.
    if settings.get('cpu_cores'):
        lib.num_threads(int(settings['cpu_cores']))

    mol = build_mol(xyz, charge, multiplicity, basis, memory_mb=memory_mb)
    mf = make_mf(mol, xc, multiplicity, settings=settings)
    mf.kernel()
    if not mf.converged:
        # Stop here: an optimization or Hessian built on an unconverged wavefunction costs as much as
        # a valid one and yields numbers that only look like results.
        output['success'] = False
        output['error'] = 'SCF did not converge'
        output['xyz'] = mol_to_xyz_dict(mol, xyz)
        return output

    do_opt = job_type in ('opt', 'conf_opt', 'optfreq')
    do_freq = job_type in ('freq', 'optfreq')

    if do_opt:
        mf, opt_xyz = run_opt(mf, xyz, multiplicity, is_ts=is_ts,
                              constraints=constraints, work_dir=work_dir, settings=settings)
        output['opt_xyz'] = opt_xyz
    else:
        output['xyz'] = mol_to_xyz_dict(mol, xyz)

    output['sp'] = float(mf.e_tot) * HARTREE2KJMOL
    output['dipole'] = get_dipole_debye(mf)

    if do_freq:
        output.update(run_freq(mf, multiplicity, settings=settings))

    return output


def main():
    """
    Parse the input path, run the job, and write ``output.yml`` (always, even on failure).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--yml_path', type=str, default='input.yml')
    args = parser.parse_args()

    input_path = os.path.abspath(args.yml_path)
    if os.path.isdir(input_path):
        input_path = os.path.join(input_path, 'input.yml')
    work_dir = os.path.dirname(input_path)

    try:
        input_dict = read_yaml_file(input_path)
        output = run_job(input_dict, work_dir)
    except Exception as exc:
        output = {'schema_version': 1, 'adapter': 'pyscf', 'success': False,
                  'error': f'{type(exc).__name__}: {exc}'}

    save_yaml_file(os.path.join(work_dir, 'output.yml'), output)


if __name__ == '__main__':
    main()
