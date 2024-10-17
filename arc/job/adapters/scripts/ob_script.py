#!/usr/bin/env python3
# encoding: utf-8

from openbabel import openbabel
import argparse
import os
import yaml


def xyz_to_OBMol(xyz: str) ->  openbabel.OBMol:
    """
    Turns an xyz to OBMol object.
    Args:
        xyz (str): xyz in xyz file format
    Returns:
        mol (OBMol): OBMol object that corresponds to the xyz.
    """
    ob_conv = openbabel.OBConversion()
    ob_conv.SetInAndOutFormats('xyz', 'xyz')
    ob_mol = openbabel.OBMol()
    ob_conv.ReadString(ob_mol, xyz)
    return ob_mol


def run_sp(mol : openbabel.OBMol, ff_method : str) -> int:
    ff = openbabel.OBForceField.FindForceField(ff_method)
    ff.Setup(mol)
    return ff.Energy()*4.184 # kcal/mol to kj/mol


def constraint_opt(mol: openbabel.OBMol,
                   constraints_dict: dict = None,
                   steps: int = 2000,
                   econv: float = 1.0e-6,
                   ff_method: str = "MMFF94s"):
    """
    perform constraint optimization.
    Args:
        mol (openbabel.OBMol): the molecule required optimization.
        constraints_dict (dict): the degrees of freedom that are required to be preserved.
        steps (int): the number of steps taken.
        econv (float): the convergence parameter.
        ff_method (str): the force field method required for the optimization.
    Returns:
        optimized xyz string, energy in kj/mol
    """
    # Setup the force field with the constraints
    ff = openbabel.OBForceField.FindForceField(ff_method)

    if constraints_dict is not None:
        # Define constraints
        constraints = openbabel.OBFFConstraints()
        bonds, angles, dihedrals = list(), list(), list()
        for constraint in constraints_dict:
                atom_indices = [i-1 for i in constraint[0]]

                if len(atom_indices) == 2:
                    bonds.append([atom_indices, constraint[1]])
                if len(atom_indices) == 3:
                    angles.append([atom_indices, constraint[1]])
                if len(atom_indices) == 4:
                    dihedrals.append([atom_indices, constraint[1]])

        for bond in bonds:
            constraints.AddDistanceConstraint(*bond[0], bond[1])

        for angle in angles:
            constraints.AddAngleConstraint(*angle[0], angle[1])

        for dihedral in dihedrals:
            constraints.AddTorsionConstraint(*dihedral[0], dihedral[1])
        ff.SetConstraints(constraints)
        ff.Setup(mol, constraints)
    else:
        ff.Setup(mol)
    # Optimize the molecule
    ff.SteepestDescent(steps, econv)
    ff.GetCoordinates(mol)
    energy = ff.Energy()*4.184 # kcal/mol to kj/mol
    # Export the coordinates:
    obConversion = openbabel.OBConversion()
    obConversion.SetOutFormat("xyz")
    xyz_string = obConversion.WriteString(mol)
    return xyz_string[3:-1], energy


def parse_command_line_arguments(command_line_args=None):
    """
    Parse command-line arguments.
    This uses the :mod:`argparse` module, which ensures that the command-line arguments are
    sensible, parses them, and returns them.
    """
    parser = argparse.ArgumentParser(description='openbabel')
    parser.add_argument('--yml_path', metavar='input', type=str, default='input.yml',
                        help='A path to the YAML input file')
    args = parser.parse_args(command_line_args)
    return args


def string_representer(dumper, data):
    """
    Add a custom string representer to use block literals for multiline strings.
    """
    if len(data.splitlines()) > 1:
        return dumper.represent_scalar(tag='tag:yaml.org,2002:str', value=data, style='|')
    return dumper.represent_scalar(tag='tag:yaml.org,2002:str', value=data)


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
                   content: dict,
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


def main():
    """
    Run a job with torchani guesses.
    """
    args = parse_command_line_arguments()
    try:
        input_dict = read_yaml_file(os.path.join(str(args.yml_path), "input.yml"))
    except FileNotFoundError:
        print(f"could not find input file at {os.path.join(str(args.yml_path), 'input.yml')}")
        return None
    job_type = input_dict["job_type"]

    xyz = input_dict["xyz"]
    ff_method = input_dict["FF"]
    opt_xyz = None
    mol = xyz_to_OBMol(xyz)
    if job_type == 'sp':
        sp = run_sp(mol=mol, ff_method=ff_method)
        save_output_file(path = str(args.yml_path), key="sp", val=sp)

    if job_type in ['opt', 'conf_opt', 'directed_scan']:
        constraints = input_dict["constraints"] if "constraints" in input_dict.keys() else None
        opt_xyz, sp = constraint_opt(mol=mol, constraints_dict=constraints, ff_method=ff_method)
        save_output_file(path = str(args.yml_path), content_dict = {"opt_xyz" : opt_xyz, "sp" : sp})
    
    if job_type == 'scan':
        raise NotImplementedError("Scan job type is not implemented for TorchANI. Use ARC's directed scan instead")
    return None


if __name__ == '__main__':
    main()
