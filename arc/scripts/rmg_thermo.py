#!/usr/bin/env python3
# encoding: utf-8

"""
A standalone script to run RMG
and get thermodynamic properties for species
"""

import os
from typing import List

from common import parse_command_line_arguments, read_yaml_file, save_yaml_file

from rmgpy.data.rmg import RMGDatabase
from rmgpy import settings as rmg_settings
from rmgpy.species import Species


DB_PATH = rmg_settings['database.directory']


def main():
    """
    Get species thermodynamic properties from RMG.

    The input YAML file should be a list of species dictionaries.
    Each species dictionary should have the following keys:
        - label: The species label.
        - adjlist: The adjacency list of the species.
    The returned YAML file will have the same species dictionaries with the following additional keys:
        - h298: The enthalpy at 298 K in kJ/mol.
        - s298: The entropy at 298 K in J/mol*K.
        - comment: The thermo comment (source).
    """
    args = parse_command_line_arguments()
    input_file = args.file
    species_list = read_yaml_file(path=input_file)
    if not isinstance(species_list, list):
        raise ValueError(f'The content of {input_file} must be a list, got {species_list} which is a {type(species_list)}')
    result = get_thermo(species_list)
    save_yaml_file(path=input_file, content=result)


def get_thermo(species_list: List[dict]) -> List[dict]:
    """
    Get thermo properties for a list of species.

    Args:
        species_list (List[dict]): A list of species dictionaries.

    Returns:
        List[dict]: A list of species dictionaries with thermo properties.
    """
    rmgdb = load_rmg_database()
    for i in range(len(species_list)):
        spc = Species(label=species_list[i]['label'])
        spc.from_adjacency_list(species_list[i]['adjlist'])
        spc.thermo = rmgdb.thermo.get_thermo_data(spc)
        species_list[i]['h298'] = spc.thermo.get_enthalpy(298) * 0.001  # converted to kJ/mol
        species_list[i]['s298'] = spc.thermo.get_entropy(298)  # in J/mol*K
        species_list[i]['comment'] = spc.thermo.comment
    return species_list


def load_rmg_database() -> RMGDatabase:
    """
    Load the RMG database.

    Returns:
        RMGDatabase: The loaded RMG database.
    """
    rmgdb = RMGDatabase()
    thermo_libraries = read_yaml_file(path=os.path.join(os.path.dirname(__file__), 'libraries.yaml'))['thermo']
    rmgdb.load_thermo(path=os.path.join(DB_PATH, 'thermo'), thermo_libraries=thermo_libraries, depository=True, surface=False)
    return rmgdb


if __name__ == '__main__':
    main()
