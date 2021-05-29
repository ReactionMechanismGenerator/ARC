#!/usr/bin/env python3
# encoding: utf-8

"""
AutoTST could not be run directly from ARC since it cases a `Segmentation Fault (core dump)` error
The error was traced to be related to a compiled C file of RDKit:
in autotst.reaction.py L 356:      bm = rdkit.Chem.rdDistGeom.GetMoleculeBoundsMatrix(combined)
This is likely caused by import orders and might be solved by organizing import differently in both ARC and AutoTST
A different solution is to independently call AutoTST via os.system('python ...')
or even running it on the server as an independent job.
This file is meant to accept command line arguments of an AutoTST reaction string
and save the TS guess output as auto_tst.xyz
"""

import argparse
import os

has_auto_tst = True
try:
    # old format
    from autotst.reaction import AutoTST_Reaction
except ImportError:
    try:
        # new format
        from autotst.reaction import Reaction as AutoTST_Reaction
    except ImportError:
        has_auto_tst = False

from arc.common import ARC_PATH, get_logger
from arc.exceptions import TSError
from arc.species.converter import xyz_from_data, xyz_to_str


logger = get_logger()


def parse_command_line_arguments(command_line_args=None):
    """
    Parse command-line arguments.
    This uses the :mod:`argparse` module, which ensures that the command-line arguments are
    sensible, parses them, and returns them.
    """
    parser = argparse.ArgumentParser(description='AutoTST')
    parser.add_argument('reaction_label', metavar='Reaction', type=str, nargs=1,
                        help='a string representation of a reaction, e.g., CCCC+[O]O_[CH2]CCC+OO')
    parser.add_argument('reaction_family', metavar='Reaction Family', type=str, nargs=1,
                        help='a string representation of a reaction family, e.g., H_Abstraction')

    args = parser.parse_args(command_line_args)

    # Process args to set correct default values and format
    args.reaction_label = args.reaction_label[0]
    args.reaction_family = args.reaction_family[0]

    return args


def main(reaction_label=None, reaction_family=None):
    """
    Run AutoTST to generate a TS guess
    Currently only works for H Abstraction
    """
    if has_auto_tst:
        if reaction_label is None:
            # Parse the command-line arguments (requires the argparse module)
            args = parse_command_line_arguments()
            reaction_label = str(args.reaction_label)
            reaction_family = str(args.reaction_family)
        reaction_family = str('H_Abstraction') if reaction_family is None else reaction_family

        try:
            reaction = AutoTST_Reaction(label=reaction_label, reaction_family=reaction_family)
        except AssertionError as e:
            logger.error('Could not generate a TS guess using AutoTST for reaction {0}'.format(reaction_label))
            raise TSError('Could not generate AutoTST guess:\n{0}'.format(e))
        else:
            positions = reaction.ts.ase_ts.get_positions()
            numbers = reaction.ts.ase_ts.get_atomic_numbers()
            xyz_guess = xyz_to_str(xyz_dict=xyz_from_data(coords=positions, numbers=numbers))

            xyz_path = os.path.join(ARC_PATH, 'arc', 'ts', 'auto_tst.xyz')

            with open(xyz_path, 'w') as f:
                f.write(xyz_guess)


if __name__ == '__main__':
    main()
