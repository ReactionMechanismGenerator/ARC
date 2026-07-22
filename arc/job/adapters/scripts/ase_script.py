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
        model = calc_config.get('model', 'uma-m-1p1')
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
                'freqs': (freqs.cpu().numpy().tolist() if hasattr(freqs, 'cpu') else freqs.numpy().tolist()),
                'hessian': hessian.cpu().numpy().tolist() if hasattr(hessian, 'cpu') else hessian.tolist(),
                'modes': modes.cpu().numpy().tolist() if hasattr(modes, 'cpu') else modes.tolist(),
                'force_constants': force_constants.cpu().numpy().tolist() if hasattr(force_constants, 'cpu') else force_constants.tolist(),
                'reduced_masses': reduced_masses.cpu().numpy().tolist() if hasattr(reduced_masses, 'cpu') else reduced_masses.tolist()
            }
        except Exception:
            pass

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
    
    output = {}
    
    def save_current_geometry(out_dict, atoms_obj, input_xyz):
        out_dict['opt_xyz'] = {
            'coords': tuple(map(tuple, atoms_obj.get_positions().tolist())),
            'symbols': input_xyz['symbols'],
            'isotopes': input_xyz.get('isotopes') or tuple([None] * len(input_xyz['symbols']))
        }

    if job_type == 'sp':
        output['sp'] = to_kJmol(atoms.get_potential_energy())

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
