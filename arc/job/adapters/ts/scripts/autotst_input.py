#!/usr/bin/env python3
# encoding: utf-8

"""
A standalone script to run AutoTST.
"""

import argparse
import numpy as np
import os
import yaml
from typing import Optional

HAS_AUTOTST = True
try:
    # new format
    from autotst.reaction import Reaction as AutoTST_Reaction
except (ImportError, ModuleNotFoundError):
    try:
        # old format
        from autotst.reaction import AutoTST_Reaction
    except (ImportError, ModuleNotFoundError):
        HAS_AUTOTST = False

if not HAS_AUTOTST:
    raise ModuleNotFoundError(f'Could not import AutoTST, make sure it is properly installed.\n'
                              f'See {{url}} for more information, or use the MAKEFILE provided with ARC.')


def parse_command_line_arguments(command_line_args=None):
    """
    Parse command-line arguments.
    This uses the :mod:`argparse` module, which ensures that the command-line arguments are
    sensible, parses them, and returns them.
    """
    parser = argparse.ArgumentParser(description='AutoTST')
    parser.add_argument('reaction_label', metavar='Reaction', type=str, nargs=1,
                        help='a string representation of a reaction, e.g., CCCC+[O]O_[CH2]CCC+OO')
    parser.add_argument('output_path', metavar='The output path', type=str, nargs=1,
                        help='a string representation of the output path to store the results')
    parser.add_argument('mkl_num_threads', metavar='The number of threads path', type=str, nargs=1,
                        help='a string representation of the number of threads to use', default='1')

    args = parser.parse_args(command_line_args)

    # Process args to set correct default values and format
    args.reaction_label = args.reaction_label[0]
    args.output_path = args.output_path[0]

    return args


def main(reaction_label: Optional[str] = None,
         output_path: Optional[str] = None,
         ) -> None:
    """
    Run AutoTST to generate TS guesses.

    Args:
        reaction_label (str, optional): The AutoTST reaction label in a r1+r1_p1+p2 format.
        output_path (str, optional): The path for storing the output file.
    """
    # Parse command-line arguments
    args = parse_command_line_arguments()

    if reaction_label is None:
        reaction_label = str(args.reaction_label)
        output_path = str(args.output_path)
    print(f'AutoTST reaction label: {reaction_label}')

    os.environ['MKL_NUM_THREADS'] = str(args.mkl_num_threads)
    os.environ['MKL_DYNAMIC'] = 'FALSE'

    try:
        autotst_reaction = AutoTST_Reaction(label=reaction_label)
        autotst_reaction.get_labeled_reaction()
    except Exception:
        return None

    ts_list = autotst_reaction.ts['forward'] + autotst_reaction.ts['reverse']
    results = list()
    for ts in ts_list:
        coords = np.ndarray.tolist(ts.ase_molecule.get_positions())
        numbers = np.ndarray.tolist(ts.ase_molecule.get_atomic_numbers())
        results.append({'coords': coords, 'numbers': numbers})

    if '/' in output_path and os.path.dirname(output_path) and not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    if '.yml' not in output_path and '.yaml' not in output_path:
        output_path = os.path.join(output_path, 'ts_results.yml')
    with open(output_path, 'w') as f:
        f.write(yaml.dump(data=results))


if __name__ == '__main__':
    main()
