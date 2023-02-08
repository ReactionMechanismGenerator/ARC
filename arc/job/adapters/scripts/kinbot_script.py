# General Imports
import yaml
import os
import argparse
import datetime
from typing import TYPE_CHECKING, List, Optional, Tuple, Union


# Kinbot Imports
from kinbot.modify_geom import modify_coordinates
from kinbot.reaction_finder import ReactionFinder
from kinbot.reaction_generator import ReactionGenerator
from kinbot.parameters import Parameters
from kinbot.qc import QuantumChemistry
from kinbot.stationary_pt import StationaryPoint



def read_yaml_file(path: str):
    """
    Read a YAML file (usually an input / restart file, but also conformers file)
    and return the parameters as python variables.

    Args:
        path (str): The YAML file path to read.
        project_directory (str, optional): The current project directory to rebase upon.

    Returns: Union[dict, list]
        The content read from the file.
    """
    # if project_directory is not None:
    #     path = globalize_paths(path, project_directory)
    if not isinstance(path, str):
        raise Exception(f'path must be a string, got {path} which is a {type(path)}')
    if not os.path.isfile(path):
        raise Exception(f'Could not find the YAML file {path}')
    with open(path, 'r') as f:
        content = yaml.load(stream=f, Loader=yaml.FullLoader)
    return content

def set_up_kinbot(mol: 'Smiles',
                  families: List[str],
                  kinbot_xyz: List[Union[str, float]],
                  multiplicity: int,
                  charge: int,
                  ) -> ReactionGenerator:
    """
    This will set up KinBot to run for a unimolecular reaction starting from the single reactant side.

    Args:
        mol (Molecule): The RMG Molecule instance representing the unimolecular well to react.
        families (List[str]): The specific KinBot families to try.
        kinbot_xyz (list): The cartesian coordinates of the well in the KinBot list format.
        multiplicity (int): The well/reaction multiplicity.
        charge (int): The well/reaction charge.

    Returns:
        ReactionGenerator: The KinBot ReactionGenerator instance.
    """
    params = Parameters()
    params.par['title'] = 'ARC'
    # molecule information
    params.par['smiles'] = mol
    params.par['structure'] = kinbot_xyz
    params.par['charge'] = charge
    params.par['mult'] = multiplicity
    params.par['dimer'] = 0
    # steps
    params.par['reaction_search'] = 1
    params.par['families'] = families
    params.par['homolytic_scissions'] = 0
    params.par['pes'] = 0
    params.par['high_level'] = 0
    params.par['conformer_search'] = 0
    params.par['me'] = 0
    #     params.par['one_reaction_fam'] = 1
    params.par['ringrange'] = [3, 9]

    well = StationaryPoint(name='well0',
                        charge=charge,
                        mult=multiplicity,
                        structure=kinbot_xyz,
                        )

    well.calc_chemid()
    well.bond_mx()
    well.find_cycle()
    well.find_atom_eqv()
    well.find_conf_dihedral()

    qc = QuantumChemistry(params.par)
    rxn_finder = ReactionFinder(well, params.par, qc)
    rxn_finder.find_reactions()

    reaction_generator = ReactionGenerator(species=well,
                                        par=params.par,
                                        qc=qc,
                                        input_file=None,
                                        )

    return reaction_generator

def generate_coords(reaction_generator = None):
    dict_files = {}
    for r, kinbot_rxn in enumerate(reaction_generator.species.reac_obj):
        step, fix, change, release = kinbot_rxn.get_constraints(step=20,
                                                                geom=kinbot_rxn.species.geom)
        

        dict_files[str(kinbot_rxn.instance_name)] = {}

        change_starting_zero = list()
        for c in change:
            c_new = [ci - 1 for ci in c[:-1]]
            c_new.append(c[-1])
            change_starting_zero.append(c_new)

        success, coords = modify_coordinates(species=kinbot_rxn.species,
                                            name=kinbot_rxn.instance_name,
                                            geom=kinbot_rxn.species.geom,
                                            changes=change_starting_zero,
                                            bond=kinbot_rxn.species.bond,
                                            )

        dict_files[str(kinbot_rxn.instance_name)]['success'] = success
        dict_files[str(kinbot_rxn.instance_name)]['coords'] = coords.tolist()
    to_yaml("output", dict_files)
    
def to_yaml(file_name: str, py_content: dict) -> str:
    """
    Convert a Python list or dictionary to a YAML string format.

    Args:
        py_content (list, dict): The Python content to save.

    Returns: str
        The corresponding YAML representation.
    """
    with open(file_name + '.yaml', 'w') as file:
        yaml.dump(py_content, file)

        file.close()

def parse_command_line_arguments(command_line_args=None):
    """
    Parse command-line arguments.

    Args:
        command_line_args: The command line arguments.

    Returns:
        The parsed command-line arguments by key words.
    """

    parser = argparse.ArgumentParser(description='KinBot')
    parser.add_argument('--yaml_path', metavar='FILE', type=str, nargs=1,
                        default='input.yml'
                        required=True,
                        help='a file containing KinBot input requirements')

    args = parser.parse_args(command_line_args)
    args.file = args.file[0]

    return args

def main():
    """
    Run a job with Kinbot
    """
    args = parse_command_line_arguments()
    input_file = args.file
    #project_directory = os.path.abspath(os.path.dirname(args.file))
    file = read_yaml_file(path=input_file)
    input_dict = dict(file[0])
    #if 'project' not in list(input_dict.keys()):
    #    raise ValueError('A project name must be provided!')
    kinbot_object = set_up_kinbot(**input_dict)

    generate_coords(kinbot_object)

if __name__ == '__main__':
    main()
