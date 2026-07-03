#!/usr/bin/env python3
# encoding: utf-8

"""
A standalone script to generate TS guess geometries with KinBot,
should be run under the kinbot_env environment.

This script must only import KinBot, PyYAML, and the standard library
(it runs in an environment where ARC is not installed).

Input (YAML file passed via ``--yml_in_path``), keys:
    charge (int): The well/reaction charge.
    multiplicity (int): The well/reaction multiplicity.
    families (list[str]): The KinBot reaction families to search.
    step (int, optional): The constraint step passed to KinBot's get_constraints() (default: 20).
    wells (list[dict]): Entries are dicts with keys:
        direction (str): 'F' or 'R', the direction this well corresponds to.
        smiles (str): A SMILES representation of the well (resonance structure specific).
        structure (list): The cartesian coordinates in the KinBot flat list format,
                          [symbol_1, x_1, y_1, z_1, symbol_2, ...].
    yml_out_path (str): The path to which the results are saved.

Output (YAML file written to ``yml_out_path``): a list of TS guess dicts with keys:
    direction (str): 'F' or 'R' (copied from the corresponding well).
    success (bool): Whether KinBot successfully modified the geometry.
    coords (list[list[float]] or None): The TS guess cartesian coordinates (N x 3).
    execution_time (str): The time it took to generate this guess.
    error (str, optional): An error message if the well processing failed.
"""

import argparse
import datetime
import json
import os
import sys
import tempfile
import traceback

import yaml

from kinbot.modify_geom import modify_coordinates
from kinbot.parameters import Parameters
from kinbot.qc import QuantumChemistry
from kinbot.reaction_finder import ReactionFinder
from kinbot.reaction_generator import ReactionGenerator
from kinbot.stationary_pt import StationaryPoint


def set_up_kinbot(smiles: str,
                  structure: list,
                  families: list,
                  multiplicity: int,
                  charge: int,
                  ) -> 'ReactionGenerator':
    """
    Set up KinBot to run for a unimolecular reaction starting from the single well side.

    Args:
        smiles (str): The SMILES representation of the unimolecular well to react.
        structure (list): The cartesian coordinates of the well in the KinBot list format.
        families (list): The specific KinBot families to try.
        multiplicity (int): The well/reaction multiplicity.
        charge (int): The well/reaction charge.

    Returns:
        ReactionGenerator: The KinBot ReactionGenerator instance.
    """
    par_dict = {'title': 'ARC',
                # molecule information
                'smiles': smiles,
                'structure': structure,
                'charge': charge,
                'mult': multiplicity,
                # steps
                'reaction_search': 1,
                'families': families,
                'pes': 0,
                'high_level': 0,
                'conformer_search': 0,
                'me': 0,
                'ringrange': [3, 9],
                # Not used in ARC's flow (no KinBot QC jobs are ever spawned), but
                # Parameters refuses to initialize without a barrier threshold (kcal/mol).
                'barrier_threshold': 200.0,
                }
    # Parameters validates its values at construction time (and calls sys.exit() on
    # invalid ones), so the overrides must be supplied via a JSON input file rather
    # than by mutating params.par after the fact.
    fd, par_path = tempfile.mkstemp(suffix='.json', prefix='kinbot_params_', dir=os.getcwd(), text=True)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(par_dict, f)
        params = Parameters(par_path)
    finally:
        os.remove(par_path)

    well = StationaryPoint(name='well0',
                           charge=charge,
                           mult=multiplicity,
                           structure=structure,
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


def get_ts_guesses_for_well(well: dict,
                            families: list,
                            multiplicity: int,
                            charge: int,
                            step: int = 20,
                            ) -> list:
    """
    Generate TS guesses for a single well.

    Args:
        well (dict): The well dictionary with 'direction', 'smiles', and 'structure' keys.
        families (list): The specific KinBot families to try.
        multiplicity (int): The well/reaction multiplicity.
        charge (int): The well/reaction charge.
        step (int, optional): The constraint step passed to KinBot's get_constraints().

    Returns:
        list: TS guess dictionaries.
    """
    results = list()
    reaction_generator = set_up_kinbot(smiles=well['smiles'],
                                       structure=well['structure'],
                                       families=families,
                                       multiplicity=multiplicity,
                                       charge=charge,
                                       )
    for kinbot_rxn in reaction_generator.species.reac_obj:
        t0 = datetime.datetime.now()
        step_, fix, change, release = kinbot_rxn.get_constraints(step=step,
                                                                 geom=kinbot_rxn.species.geom)

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

        results.append({'direction': well['direction'],
                        'success': bool(success),
                        'coords': [[float(ci) for ci in coord] for coord in coords] if success else None,
                        'execution_time': str(datetime.datetime.now() - t0),
                        })
    return results


def run_kinbot(input_dict: dict) -> None:
    """
    Run KinBot for all wells in the input dictionary and save the results.

    Args:
        input_dict (dict): The input dictionary.
    """
    results = list()
    for well in input_dict['wells']:
        try:
            results.extend(get_ts_guesses_for_well(well=well,
                                                   families=input_dict['families'],
                                                   multiplicity=input_dict['multiplicity'],
                                                   charge=input_dict['charge'],
                                                   step=input_dict.get('step', 20),
                                                   ))
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            results.append({'direction': well.get('direction'),
                            'success': False,
                            'coords': None,
                            'execution_time': None,
                            'error': f'{e.__class__.__name__}: {e}',
                            })
    save_yaml_file(path=input_dict['yml_out_path'], content=results)


def read_yaml_file(path: str):
    """
    Read a YAML file and return its content.

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
    parser = argparse.ArgumentParser(description='KinBot')
    parser.add_argument('--yml_in_path', metavar='input', type=str, default='input.yml',
                        help='A path to the YAML input file')
    args = parser.parse_args(command_line_args)
    return args


def main():
    """
    Run KinBot to generate TS guesses.
    """
    args = parse_command_line_arguments()
    yml_in_path = str(args.yml_in_path)
    input_dict = read_yaml_file(yml_in_path)
    # KinBot's QuantumChemistry writes a 'kinbot.db' file (and possibly other
    # scratch files) to the current working directory, direct these to the job folder.
    os.chdir(os.path.dirname(os.path.abspath(input_dict['yml_out_path'])))
    run_kinbot(input_dict=input_dict)


if __name__ == '__main__':
    main()
