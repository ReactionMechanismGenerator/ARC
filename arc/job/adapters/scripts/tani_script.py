#!/usr/bin/env python3
# encoding: utf-8

"""
A standalone script to run TorhANI,
should be run under the tani environment.
"""

import argparse
import os
import yaml

from ase import Atoms
from ase.constraints import FixInternals
from ase.optimize import BFGS
from ase.optimize.sciopt import Converged, OptimizerConvergenceError, SciPyFminBFGS, SciPyFminCG
import torch
import torchani
from torchani.ase import Calculator

elements = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
 'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17,
 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22,'V': 23,'Cr': 24, 'Mn': 25, 'Fe': 26,
 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35,
 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44,
 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50, 'Sb': 51, 'Te': 52, 'I': 53,
 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62,
 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71,
 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80, 
 'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84,'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89,
 'Th': 90, 'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98,
 'Es': 99,'Fm': 100, 'Md': 101, 'No': 102,'Lr': 103, 'Rf': 104, 'Db': 105, 'Sg': 106,
 'Bh': 107, 'Hs': 108, 'Mt': 109, 'Ds': 110, 'Rg': 111, 'Cn': 112, 'Nh': 113, 'Fl': 114,
 'Mc': 115, 'Lv': 116, 'Ts': 117,'Og': 118}


def run_sp(xyz, device, model):
    """
    Run a single-point energy calculation.
    """
    coords, z_list = xyz_to_coords_and_element_numbers(xyz)
    coordinates = torch.tensor([coords], requires_grad=True, device=device)
    species = torch.tensor([z_list], device=device)
    energy = model((species, coordinates)).energies
    sp = hartree_to_si(energy.item(), kilo=True)
    return sp


def to_Z(element: str) -> int:
    """
    Return the element number of an element.
    
    Args:
        element (str): the element symbol. For example: "H", "Br"
    
    Returns:
        The element number. For the example: 1, 35
    """
    return elements.get(element.capitalize(), None)


def run_force(xyz, device, model):
    """
    Compute the force matrix.
    """
    coords, z_list = xyz_to_coords_and_element_numbers(xyz)
    coordinates = torch.tensor([coords], requires_grad=True, device=device)
    species = torch.tensor([z_list], device=device)
    energy = model((species, coordinates)).energies
    derivative = torch.autograd.grad(energy.sum(), coordinates)[0]
    force = -derivative
    force = force.squeeze().numpy().tolist()
    return force


def run_opt(xyz,
            model,
            constraints: dict = None,
            fmax: float = 0.001,
            steps: int = None,
            engine: str = 'SciPyFminBFGS',
            ):
    """
    Run a geometry optimization calculation with optional constraints.
    The convergence criteria satisfied when the forces on all individual
    atoms are less than ``fmax`` or when the number of ``steps`` exceeds.

    Args:
        fmax (float, optional): The maximal force for convergence.
        steps (int, optional): The maximal number of steps for the optimization.
        engine (str, optional): The optimizer to use.
                                'BFGS': Broyden–Fletcher–Goldfarb–Shanno. This algorithm chooses each step from
                                        the current atomic forces and an approximation of the Hessian matrix.
                                            The Hessian is established from an initial guess which is gradually
                                            improved as more forces are evaluated. Implemented in ASE.
                                'SciPyFminBFGS': A Quasi-Newton method (Broydon-Fletcher-Goldfarb-Shanno).
                                                    An ASE interface to SciPy.
                                'SciPyFminCG': A non-linear (Polak-Ribiere) conjugate gradient algorithm.
                                                An ASE interface to SciPy.
    """
    calculator = model.ase()
    atoms = Atoms(xyz['symbols'], xyz['coords'])
    atoms.set_calculator(calculator)

    if constraints is not None:
        bonds, angles, dihedrals = list(), list(), list()
        for constraint in constraints:
            atom_indices = convert_list_index_0_to_1(constraint[0], direction=-1)
            if len(atom_indices) == 2:
                bonds.append([constraint[1], atom_indices])
            if len(atom_indices) == 3:
                angles.append([constraint[1], atom_indices])
            if len(atom_indices) == 4:
                dihedrals.append([constraint[1], atom_indices])
        constraints = FixInternals(bonds=bonds, angles_deg=angles, dihedrals_deg=dihedrals)
        atoms.set_constraint(constraints)

    engine_list = ['bfgs', 'scipyfminbfgs', 'scipyfmincg']
    engine_set = set([engine] + engine_list)
    engine_dict = {'bfgs': BFGS, 'scipyfminbfgs': SciPyFminBFGS, 'scipyfmincg': SciPyFminCG}
    for opt_engine_name in engine_set:
        opt_engine = engine_dict[opt_engine_name]
        opt = opt_engine(atoms, logfile=None)
        try:
            opt.run(fmax=fmax, steps=steps)
        except (Converged, NotImplementedError, OptimizerConvergenceError):
            pass
        else:
            print(f"optimization converged with engine {opt_engine_name}!")
            break
    else:
        return None
    opt_xyz = {'coords': tuple(map(tuple, atoms.get_positions().tolist())), 'isotopes': xyz['isotopes'], 'symbols': xyz['symbols'] }
    return opt_xyz


def run_vibrational_analysis(xyz: dict = None,
                             opt_xyz: dict = None,
                             device: str = None,
                             model: Calculator = None):
    """
    Compute the Hessian matrix along with vibrational frequencies (cm^-1),
    normal mode displacements, force constants (mDyne/A), and reduced masses (AMU).
    """
    if all([i is None for i in [xyz, opt_xyz]]):
        raise ValueError("Must recive at least one geometry to run vibrational analysis.")
    xyz = opt_xyz or xyz
    atoms = Atoms(xyz['symbols'], xyz['coords'])
    species = torch.tensor(atoms.get_atomic_numbers(), device=device, dtype=torch.long).unsqueeze(0)
    coordinates = torch.from_numpy(atoms.get_positions()).unsqueeze(0).requires_grad_(True)
    masses = torchani.utils.get_atomic_masses(species)
    energies = model.double()((species, coordinates)).energies
    hessian = torchani.utils.hessian(coordinates, energies=energies)
    freqs, modes, force_constants, reduced_masses = torchani.utils.vibrational_analysis(masses, hessian, mode_type='MDU')
    freqs = freqs.numpy()
    results = {'hessian': hessian.tolist(),
               'freqs': freqs.tolist(),
               'modes': modes.tolist(),
               'force_constants': force_constants.tolist(),
               'reduced_masses': reduced_masses.tolist(),
               }
    return results


def hartree_to_si(e: float,
                  kilo: bool = True,
                  ) -> float:
    """
    Convert Hartree units into J/mol or into kJ/mol.

    Args:
        e (float): Energy in Hartree.
        kilo (bool, optional): Whether to return kJ/mol units. ``True`` by default.
    """
    if not isinstance(e, (int, float)):
        raise ValueError(f'Expected a float, got {e} which is a {type(e)}.')
    factor = 0.001 if kilo else 1
    Na = 6.02214179e23 
    E_h = 4.35974434e-18
    return e * Na * E_h * factor


def read_yaml_file(path: str):
    """
    Read a YAML file (usually an input / restart file, but also conformers file)
    and return the parameters as python variables.

    Args:
        path (str): The YAML file path to read.

    Returns: Union[dict, list]
        The content read from the file.
    """
    with open(path, 'r') as f:
        content = yaml.load(stream=f, Loader=yaml.FullLoader)
    return content


def save_yaml_file(path: str,
                   content: list,
                   ) -> None:
    """
    Save a YAML file.

    Args:
        path (str): The YAML file path to save.
        content (list): The content to save.
    """
    yaml.add_representer(str, string_representer)
    yaml_str = yaml.dump(data=content)
    with open(path, 'w') as f:
        f.write(yaml_str)


def save_output_file(path,
                     key = None,
                     val = None,
                     content_dict = None,
                     ):
    """
    Save the output of a job to the YAML output file.

    Args:
        key (str, optional): The key for the YAML output file.
        val (Union[float, dict, np.ndarray], optional): The value to be stored.
        content_dict (dict, optional): A dictionary to store.

    """
    yml_out_path = os.path.join(path, 'output.yml')
    content = read_yaml_file(yml_out_path) if os.path.isfile(yml_out_path) else dict()
    if content_dict is not None:
        content.update(content_dict)
    if key is not None:
        content[key] = val
    save_yaml_file(path=yml_out_path, content=content)


def convert_list_index_0_to_1(_list: list, direction: int = 1):
    """
    Convert a list from 0-indexed to 1-indexed, or vice versa.
    Ensures positive values in the resulting list.

    Args:
        _list (list): The list to be converted.
        direction (int, optional): Either 1 or -1 to convert 0-indexed to 1-indexed or vice versa, respectively.

    Raises:
        ValueError: If the new list contains negative values.

    Returns:
        Union[list, tuple]: The converted indices.
    """
    new_list = [item + direction for item in _list]
    if any(val < 0 for val in new_list):
        raise ValueError(f'The resulting list from converting {_list} has negative values:\n{new_list}')
    if isinstance(_list, tuple):
        new_list = tuple(new_list)
    return new_list


def string_representer(dumper, data):
    """
    Add a custom string representer to use block literals for multiline strings.
    """
    if len(data.splitlines()) > 1:
        return dumper.represent_scalar(tag='tag:yaml.org,2002:str', value=data, style='|')
    return dumper.represent_scalar(tag='tag:yaml.org,2002:str', value=data)


def parse_command_line_arguments(command_line_args=None):
    """
    Parse command-line arguments.
    This uses the :mod:`argparse` module, which ensures that the command-line arguments are
    sensible, parses them, and returns them.
    """
    parser = argparse.ArgumentParser(description='torchani')
    parser.add_argument('--yml_path', metavar='input', type=str, default='input.yml',
                        help='A path to the YAML input file')
    args = parser.parse_args(command_line_args)
    return args


def xyz_to_coords_and_element_numbers(xyz: dict):
    """
    Convert xyz to a coords list and an atomic number list.

    Args:
        xyz (dict): The coordinates.

    Returns:
        Tuple[list, list]: Coords and atomic numbers.
    """
    coords = xyz_to_coords_list(xyz)
    z_list = list(map(to_Z, list(xyz["symbols"])))
    return coords, z_list


def xyz_to_coords_list(xyz_dict: dict):
    """
    Get the coords part of an xyz dict as a (mutable) list of lists (rather than a tuple of tuples).

    Args:
        xyz_dict (dict): The ARC xyz format.

    Returns: Optional[List[List[float]]]
        The coordinates.
    """
    if xyz_dict is None:
        return None
    coords_tuple = xyz_dict['coords']
    coords_list = list()
    for coords_tup in coords_tuple:
        coords_list.append([coords_tup[0], coords_tup[1], coords_tup[2]])
    return coords_list


def main():
    """
    Run a job with torchani guesses.
    """
    args = parse_command_line_arguments()
    try:
        input_dict = read_yaml_file(os.path.join(str(args.yml_path), "input.yml"))
    except FileNotFoundError:
        output = "file not found"
        save_output_file(path = str(args.yml_path), content_dict = {"error" : output})
        return
    job_type = input_dict["job_type"]
    device = input_dict["device"]
    model = None
    if input_dict["model"].lower() == 'ani1ccx':
        model = torchani.models.ANI1ccx(periodic_table_index=True).to(device)
    elif input_dict["model"].lower() =='ani1x':
        model = torchani.models.ANI1x(periodic_table_index=True).to(device)
    elif input_dict["model"].lower() == 'ani2x' or model is None:
        model = torchani.models.ANI2x(periodic_table_index=True).to(device)

    xyz = input_dict["xyz"]
    opt_xyz = None
    if job_type == 'sp':
        sp = run_sp(xyz=xyz, device=device, model=model)
        save_output_file(path = str(args.yml_path), key="sp", val=sp)

    elif job_type == 'force':
        forces = run_force(xyz=xyz, device=device, model=model)
        save_output_file(path = str(args.yml_path), key="force", val=forces)

    elif job_type in ['opt', 'conf_opt', 'directed_scan', 'optfreq']:
        constraints = input_dict["constraints"] if "constraints" in input_dict.keys() else None
        opt_xyz = run_opt(xyz=xyz, constraints=constraints, fmax=input_dict["fmax"], model=model,
                          steps=input_dict["steps"] if "steps" in input_dict.keys() else None, engine=input_dict["engine"])
        save_output_file(path = str(args.yml_path), key="opt_xyz", val=opt_xyz)
        sp = run_sp(xyz=opt_xyz, device=device, model=model)
        save_output_file(path = str(args.yml_path), key="sp", val=sp)

    if job_type in ['freq', 'optfreq']:
        freqs = run_vibrational_analysis(xyz=xyz, opt_xyz=opt_xyz, device=device, model=model)
        save_output_file(path = str(args.yml_path), content_dict=freqs)
    
    if job_type == 'scan':
        raise NotImplementedError("Scan job type is not implemented for TorchANI. Use ARC's directed scan instead")
    return None


if __name__ == '__main__':
    main()
