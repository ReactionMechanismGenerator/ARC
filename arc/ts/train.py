#!/usr/bin/env python3
# encoding: utf-8

"""
A utility script to train TS heuristics on existing TS structures
"""

import matplotlib.pyplot as plt
import numpy as np
import os

from arc.common import save_yaml_file, sort_two_lists_by_the_first
from arc.settings import arc_path
from arc.species import ARCSpecies
from arc.species.converter import str_to_xyz, xyz_from_data, xyz_to_coords_list
from arc.species.vectors import calculate_angle


def read_data(family, save_result=False):
    """
    Read the TS training data.
    The .xyz files should be in the form of::

        3
        reaction: F+H2=HF+H, family: H_Abstraction, labels: 1 0 2, stretch: 1.5919 1.0366, level: QCISD/MG3, source: xx
        H          0.14657       -1.12839        0.00000
        F          0.00000        0.33042        0.00000
        H         -0.14657       -1.84541        0.00000

    Notes::

        - Indices are 0-indexed
        - Bimolecular reactions should have the atoms aggregated by the separate molecules,
          e.g., the abstraced H in an H abstraction reaction must separate the two fragments in the coordinates
        - The order of the labels must correspond to the order of the additional data such as stretches

    Args:
        family (str): The family to load.
        save_result (bool, optional): Whether to save the output as a file, ``True`` to save, default is ``False``.

    Returns:
        list: The loaded data.

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

    data = list()
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

            data.append(entry)

    if save_result:
        save_yaml_file(path=os.path.join(base_path, f'{family}_raw_data.yaml'), content=data)

    print(f'loaded {len(data)} entries for {family}')

    return data


def process(family, data, plot=False):
    """
    Process the training data for a specific family.

    Args:
        family (str): The reaction family to process.
        data (list): The training data
        plot (bool, optional): Whether to generate plots describing training data statistics.
                               ``True`` generate,``False``by default..

    Returns:
        tuple:
            - dict: The trained angles.
            - dict: The trained stretches.
    """
    if family == 'H_Abstraction':
        angle_dict, stretch_dict = dict(), dict()
        for i, entry in enumerate(data):
            h_index = entry['labels'][1]
            coords_list = xyz_to_coords_list(entry['xyz'])

            coords_1 = coords_list[0: h_index + 1]
            symbols_1 = entry['xyz']['symbols'][0: h_index + 1]
            isotopes_1 = entry['xyz']['isotopes'][0: h_index + 1]
            xyz1 = xyz_from_data(coords=coords_1, symbols=symbols_1, isotopes=isotopes_1)
            spc1 = ARCSpecies(label='RH1', xyz=xyz1)

            coords_2 = coords_list[h_index:]
            symbols_2 = entry['xyz']['symbols'][h_index:]
            isotopes_2 = entry['xyz']['isotopes'][h_index:]
            xyz2 = xyz_from_data(coords=coords_2, symbols=symbols_2, isotopes=isotopes_2)
            spc2 = ARCSpecies(label='RH2', xyz=xyz2)

            a_atom_symbol = spc1.get_xyz()['symbols'][entry['labels'][0]]
            a_atom_type = spc1.mol.atoms[entry['labels'][0]].atomtype.label if spc1.mol is not None \
                else a_atom_symbol

            b_atom_symbol = spc2.get_xyz()['symbols'][entry['labels'][2] - len(spc1.get_xyz()['symbols']) + 1]
            b_atom_type = spc2.mol.atoms[entry['labels'][2] - len(spc1.get_xyz()['symbols']) + 1].atomtype.label \
                if spc2.mol is not None else b_atom_symbol

            # train angles
            angle = calculate_angle(coords=entry['xyz']['coords'], atoms=entry['labels'], index=0, units='degs')

            if a_atom_type != a_atom_symbol and b_atom_type != b_atom_symbol:
                # atoms for which there's no specific atom type, such as H and P, will only be considered as symbols
                key = tuple(sorted([a_atom_type, b_atom_type]))
                if key in angle_dict:
                    angle_dict[key]['angle'] = (angle_dict[key]['angle'] * len(angle_dict[key]['entries']) + angle) \
                                               / (len(angle_dict[key]['entries']) + 1)
                    angle_dict[key]['entries'].append(i)
                else:
                    angle_dict[key] = dict()
                    angle_dict[key]['angle'] = angle
                    angle_dict[key]['entries'] = [i]

            # also store just by symbol, mimicking a more generalized node in this simple tree
            key = tuple(sorted([a_atom_symbol, b_atom_symbol]))
            if key in angle_dict:
                angle_dict[key]['angle'] = (angle_dict[key]['angle'] * len(angle_dict[key]['entries']) + angle) \
                                           / (len(angle_dict[key]['entries']) + 1)
                angle_dict[key]['entries'].append(i)
            else:
                angle_dict[key] = dict()
                angle_dict[key]['angle'] = angle
                angle_dict[key]['entries'] = [i]

            # also store as the default value, the top node in this tree
            if 'default' in angle_dict:
                angle_dict['default']['angle'] = (angle_dict['default']['angle'] *
                                                  len(angle_dict['default']['entries']) + angle) \
                                                 / (len(angle_dict['default']['entries']) + 1)
                angle_dict['default']['entries'].append(i)
            else:
                angle_dict['default'] = dict()
                angle_dict['default']['angle'] = angle
                angle_dict['default']['entries'] = [i]

            # train stretches:
            if a_atom_type != a_atom_symbol and b_atom_type != b_atom_symbol:
                # atoms for which there's no specific atom type, such as H and P, will only be considered as symbols
                key_1 = tuple([a_atom_type, b_atom_type])
                key_2 = tuple([b_atom_type, a_atom_type])
                for j, key in enumerate([key_1, key_2]):
                    if key in stretch_dict:
                        stretch_dict[key]['stretch'] = (stretch_dict[key]['stretch'] * len(stretch_dict[key]['entries'])
                                                        + entry['stretch'][j]) / (len(stretch_dict[key]['entries']) + 1)
                        stretch_dict[key]['entries'].append(i)
                        if plot:
                            stretch_dict[key]['stretches'].append(entry['stretch'][j])
                    else:
                        stretch_dict[key] = dict()
                        stretch_dict[key]['stretch'] = entry['stretch'][j]
                        stretch_dict[key]['entries'] = [i]
                        if plot:
                            stretch_dict[key]['stretches'] = [entry['stretch'][j]]

            # also store just by symbol, mimicking a more generalized node in this simple tree
            key_1 = tuple([a_atom_symbol, b_atom_symbol])
            key_2 = tuple([b_atom_symbol, a_atom_symbol])
            for j, key in enumerate([key_1, key_2]):
                if key in stretch_dict:
                    stretch_dict[key]['stretch'] = (stretch_dict[key]['stretch'] * len(stretch_dict[key]['entries'])
                                                    + entry['stretch'][j]) / (len(stretch_dict[key]['entries']) + 1)
                    stretch_dict[key]['entries'].append(i)
                    if plot:
                        stretch_dict[key]['stretches'].append(entry['stretch'][j])
                else:
                    stretch_dict[key] = dict()
                    stretch_dict[key]['stretch'] = entry['stretch'][j]
                    stretch_dict[key]['entries'] = [i]
                    if plot:
                        stretch_dict[key]['stretches'] = [entry['stretch'][j]]

            # also store as the default value, the top node in this tree
            if 'default' in stretch_dict:
                stretch_dict['default']['stretch'] = (stretch_dict['default']['stretch'] * 2 *
                                                      len(stretch_dict['default']['entries'])
                                                      + sum(entry['stretch'])) / \
                                                     (2 * len(stretch_dict['default']['entries']) + 2)
                stretch_dict['default']['entries'].append(i)
            else:
                stretch_dict['default'] = dict()
                stretch_dict['default']['stretch'] = entry['stretch'][j]
                stretch_dict['default']['entries'] = [i]
            if plot:
                if 'stretches' not in stretch_dict['default']:
                    stretch_dict['default']['stretches'] = list()
                stretch_dict['default']['stretches'].extend(entry['stretch'])

        if plot:
            plt.rcdefaults()
            for key, val in angle_dict.items():
                if len(val['entries']) > 1:
                    fig, ax = plt.subplots()

                    reactions = [data[i]['reaction'] for i in val['entries']]
                    y_pos = np.arange(len(reactions))
                    angles = [calculate_angle(coords=data[i]['xyz']['coords'], atoms=data[i]['labels'],
                                              index=0, units='degs') for i in val['entries']]
                    angles, reactions = sort_two_lists_by_the_first(angles, reactions)

                    ax.barh(y_pos, angles, align='center', color='b')
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(reactions)
                    ax.invert_yaxis()  # labels read top-to-bottom
                    ax.set_xlabel('Angle (degrees)')
                    ax.set_title(f'Angles for {key}')
                    ax.set_xlim(min(angles) - 10, 180)
                    fig.tight_layout()
                    plt.show()

            for key, val in stretch_dict.items():
                if len(val['entries']) > 1:
                    fig, ax = plt.subplots()

                    stretches = val['stretches']
                    reactions = [data[i]['reaction'] for i in val['entries']] if key != 'default' \
                        else [i for i in range(len(stretches))]
                    y_pos = np.arange(len(reactions))
                    stretches, reactions = sort_two_lists_by_the_first(stretches, reactions)

                    ax.barh(y_pos, stretches, align='center', color='g')
                    ax.set_yticks(y_pos)
                    if key != 'default':
                        ax.set_yticklabels(reactions)
                    ax.invert_yaxis()  # labels read top-to-bottom
                    ax.set_xlabel('Bond stretch')
                    ax.set_title(f'Bond stretches for {key}')
                    ax.set_xlim(min(stretches) * .9, max(stretches) * 1.1)
                    fig.tight_layout()
                    plt.show()

    else:
        raise NotImplementedError(f'TS heuristics training is not implemented for family {family}')

    return angle_dict, stretch_dict




"""

add:

get params

add training data


"""

























