#!/usr/bin/env python3
# encoding: utf-8

"""
A standalone script to retrieve xyz data from GCN,
should be run under the ts_gcn environment.
"""

import argparse
import datetime
import os

import yaml
from rdkit import Chem

from inference import inference


def initialize_gcn_run(input_dict: dict):
    """
    Initialize a GCN run.

    Args:
        input_dict (dict): The input dictionary.
    """
    ts_fwd_path = os.path.join(input_dict['local_path'], "TS_fwd.xyz")
    ts_rev_path = os.path.join(input_dict['local_path'], "TS_rev.xyz")
    tsgs = list()
    for i in range(input_dict['repetitions']):
        tsg_f = run_gcn_locally(direction='F',
                                reactant_path=input_dict['reactant_path'],
                                product_path=input_dict['product_path'],
                                ts_path=ts_fwd_path,
                                )
        tsg_r = run_gcn_locally(direction='R',
                                reactant_path=input_dict['product_path'],
                                product_path=input_dict['reactant_path'],
                                ts_path=ts_rev_path,
                                )
        tsgs.extend([tsg_f, tsg_r])
    save_yaml_file(path=input_dict['yml_out_path'], content=tsgs)


def run_gcn_locally(direction: str,
                    reactant_path: str,
                    product_path: str,
                    ts_path: str,
                    ) -> dict:
    """
    Run GCN incore.

    Args:
        direction (str): Either 'F' or 'R' for forward ort reverse directions, respectively.
        reactant_path (str): The path to the reactant SDF file.
        product_path (str): The path to the product SDF file.
        ts_path (str): The path to the resulting TS guess file.

    Returns:
        dict: The TS guess dictionary.
    """
    tsg = {'method': 'GCN',
           'method_direction': direction,
           'initial_xyz': None,
           }
    t0 = datetime.datetime.now()
    run_gcn(r_sdf_path=reactant_path,
            p_sdf_path=product_path,
            ts_xyz_path=ts_path,
            )
    if os.path.isfile(ts_path):
        with open(ts_path, 'r') as f:
            tsg['initial_xyz'] = ''.join(f.readlines()[2:])
            tsg['success'] = True
    else:
        tsg['success'] = False
    tsg['execution_time'] = str(datetime.datetime.now() - t0)
    return tsg


def run_gcn(r_sdf_path: str,
            p_sdf_path: str,
            ts_xyz_path: str,
            ):
    r_mols = Chem.SDMolSupplier(r_sdf_path, removeHs=False, sanitize=True)
    p_mols = Chem.SDMolSupplier(p_sdf_path, removeHs=False, sanitize=True)
    try:
        inference(r_mols, p_mols, ts_xyz_path)
    except:
        print('\nGCN Failed\n')


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
    initialize_gcn_run(input_dict=read_yaml_file(yml_in_path))


if __name__ == '__main__':
    main()
