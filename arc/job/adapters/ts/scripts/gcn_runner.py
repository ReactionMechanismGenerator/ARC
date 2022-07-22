#!/usr/bin/env python3
# encoding: utf-8

"""
A runner to execute gcn_script.
Can be imported and used locally or ran via a submit script.
should be run under the arc_env environment.
"""

import argparse
import os

from arc.common import read_yaml_file, save_yaml_file
from arc.job.adapters.ts.gcn_ts import run_subprocess_locally
from arc.species.species import ARCSpecies


def run_gcn(input_dict: dict):
    """
    Run GCN by calling the gcn_script.py

    Args:
        input_dict (dict): The input dictionary.
    """
    ts_species = ARCSpecies(species_dict=input_dict['ts_species'])
    ts_fwd_path = os.path.join(input_dict['local_path'], "TS_fwd.xyz")
    ts_rev_path = os.path.join(input_dict['local_path'], "TS_rev.xyz")
    for i in range(input_dict['repetitions']):
        print(f'Running GCN, iteration {i}')
        run_subprocess_locally(direction='F',
                               reactant_path=input_dict['reactant_path'],
                               product_path=input_dict['product_path'],
                               ts_path=ts_fwd_path,
                               local_path=input_dict['local_path'],
                               ts_species=ts_species,
                               )
        run_subprocess_locally(direction='R',
                               reactant_path=input_dict['product_path'],
                               product_path=input_dict['reactant_path'],
                               ts_path=ts_rev_path,
                               local_path=input_dict['local_path'],
                               ts_species=ts_species,
                               )
    tsgs = [tsg.as_dict() for tsg in ts_species.ts_guesses if 'gcn' in tsg.method.lower()]
    save_yaml_file(path=input_dict['yml_out_path'], content=tsgs)


def parse_command_line_arguments(command_line_args=None):
    """
    Parse command-line arguments.
    This uses the :mod:`argparse` module, which ensures that the command-line arguments are
    sensible, parses them, and returns them.
    """
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument('--yml_in_path', metavar='input', type=str, default='input.yml',
                        help='A path to the YAML input file')
    args = parser.parse_args(command_line_args)
    return args


def main():
    """
    Run GCN to generate TS guesses.
    """
    args = parse_command_line_arguments()
    yml_in_path = str(args.yml_in_path)
    run_gcn(input_dict=read_yaml_file(yml_in_path))


if __name__ == '__main__':
    main()
