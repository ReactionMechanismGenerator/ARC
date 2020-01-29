#!/usr/bin/env python3
# encoding: utf-8

"""
A utility script to train TS heuristics on existing TS structures
"""

import os

from arc.common import save_yaml_file
from arc.settings import arc_path
from arc.species.converter import str_to_xyz


def read_data(family, save_result=False):
    """
    Read the TS training data.
    The .xyz files should be in the form of::

        3
        reaction: F+H2=HF+H, family: H_Abstraction, labels: 2 0 1, stretch: 1.5919 1.0366, level: QCISD/MG3, source: xx
        H          0.14657       -1.12839        0.00000
        F          0.00000        0.33042        0.00000
        H         -0.14657       -1.84541        0.00000

    Args:
        family (str): The family to load.

    Returns:
        dict: The loaded data.

    Raises:
        ValueError: If the family folder cannot be found
    """
    base_path = os.path.join(arc_path, 'arc', 'ts', 'data', family)
    if not os.path.isdir(base_path):
        raise ValueError(f'Could not find the {base_path} folder')

    file_names = list()
    for (_, _, files) in os.walk(base_path):
        file_names.extend(files)
        break  # don't continue to explore subdirectories

    content = dict()
    i = 0
    for file_name in file_names:
        if file_name.split('.')[-1] == 'xyz':
            with open(os.path.join(base_path, file_name), 'r') as f:
                lines = f.read().splitlines()  # different than .(f.readlines() sine here it also removes the '\n's

            entry = dict()
            entry['name'] = file_name.rpartition('.')[0]
            xyz_lines = [line for line in lines[2:] if line]
            entry['xyz'] = str_to_xyz('\n'.join(xyz_lines))
            if int(lines[0]) != len(entry['xyz']['symbols']):
                raise ValueError(f'Expected to fined {int(lines[0])} atoms in the {file_name} TS '
                                 f'of family {family}, but got {len(xyz_lines)}')

            tokens = lines[1].split(', ')
            for token in tokens:
                key, val = token.split(': ')
                if key in ['labels']:
                    # make it a list of integers
                    entry[key] = [int(v) for v in val.split()]
                elif key in ['stretch']:
                    # make it a list of floats
                    entry[key] = [float(v) for v in val.split()]
                else:
                    entry[key] = val

            content[i] = entry
            i += 1

    if save_result:
        save_yaml_file(path=os.path.join(base_path, f'{family}_data.yaml'), content=content)

    return content
