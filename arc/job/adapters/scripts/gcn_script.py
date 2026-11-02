#!/usr/bin/env python3
# encoding: utf-8

"""
A standalone script to retrieve xyz TS guesses from GCN (the TS-GCN project),
should be run under the ts_gcn environment.

This script must NOT import any ``arc`` modules: it is executed with the
``ts_gcn`` environment's Python interpreter (see ``arc.job.env_run``),
where ARC and its dependencies are not installed.

Two modes are supported:

1. Direct (used by the GCN adapter's incore execution, one direction per call)::

       python gcn_script.py --r_sdf_path reactant.sdf --p_sdf_path product.sdf --ts_xyz_path TS.xyz

2. Batch (used by queue execution via a YAML input file)::

       python gcn_script.py --yml_in_path input.yml

   where the YAML file contains the keys ``reactant_path``, ``product_path``,
   ``local_path``, ``yml_out_path``, and ``repetitions``.
"""

import argparse
import datetime
import os
import sys
import traceback

import yaml


def import_inference():
    """
    Import and return the ``inference`` function of the TS-GCN repository.

    The TS-GCN clone must be importable as the top-level ``inference`` module.
    ARC's installers (devtools/install_gcn.sh, devtools/install_gcn_cpu.sh)
    clone TS-GCN next to the ARC clone and export its path via ``~/.bashrc``
    or a conda activation hook, but launchers such as ``conda run`` source
    neither, so fall back to well-known locations here.

    Returns:
        callable: The ``inference.inference`` function.

    Raises:
        ImportError: If the TS-GCN repository could not be located.
    """
    candidate_dirs = list()
    tsgcn_root = os.environ.get('TSGCN_ROOT')
    if tsgcn_root:
        candidate_dirs.append(tsgcn_root)
    # This script lives at <ARC>/arc/job/adapters/scripts/gcn_script.py,
    # and the installers clone TS-GCN as a sibling of the ARC clone.
    arc_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    candidate_dirs.append(os.path.join(os.path.dirname(arc_root), 'TS-GCN'))
    try:
        from inference import inference
    except ImportError:
        for candidate in candidate_dirs:
            if os.path.isfile(os.path.join(candidate, 'inference.py')) and candidate not in sys.path:
                sys.path.insert(0, candidate)
        try:
            from inference import inference
        except ImportError as e:
            raise ImportError(f'Could not import the TS-GCN "inference" module. '
                              f'Searched sys.path and the candidate TS-GCN locations {candidate_dirs}. '
                              f'Make sure TS-GCN is cloned and importable in the ts_gcn environment '
                              f'(see devtools/install_gcn.sh). Got: {e}') from e
    return inference


def initialize_gcn_run(input_dict: dict):
    """
    Initialize a batch GCN run.

    Args:
        input_dict (dict): The input dictionary with keys ``reactant_path``, ``product_path``,
                           ``local_path``, ``yml_out_path``, and ``repetitions``.
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
    Run GCN in a single direction and package the result as a TS guess dictionary.

    Args:
        direction (str): Either 'F' or 'R' for forward or reverse directions, respectively.
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
    if os.path.isfile(ts_path):
        os.remove(ts_path)
    success = run_gcn(r_sdf_path=reactant_path,
                      p_sdf_path=product_path,
                      ts_xyz_path=ts_path,
                      )
    if success and os.path.isfile(ts_path):
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
            ) -> bool:
    """
    Run a single GCN inference: read the reactant and product SDF files
    and write the TS guess to ``ts_xyz_path``.

    Args:
        r_sdf_path (str): The path to the reactant SDF file.
        p_sdf_path (str): The path to the product SDF file.
        ts_xyz_path (str): The path to write the TS guess to.

    Returns:
        bool: Whether the inference completed without raising.
    """
    from rdkit import Chem
    inference = import_inference()
    r_mols = Chem.SDMolSupplier(r_sdf_path, removeHs=False, sanitize=True)
    p_mols = Chem.SDMolSupplier(p_sdf_path, removeHs=False, sanitize=True)
    try:
        inference(r_mols, p_mols, ts_xyz_path)
    except Exception as e:
        print(f'GCN inference failed with: {e}\n{traceback.format_exc()}', file=sys.stderr)
        return False
    return True


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
    parser = argparse.ArgumentParser(description='GCN TS guess generation (TS-GCN)')
    parser.add_argument('--yml_in_path', metavar='input', type=str, default=None,
                        help='A path to a YAML input file (batch mode)')
    parser.add_argument('--r_sdf_path', metavar='reactant', type=str, default=None,
                        help='A path to the reactant SDF file (direct mode)')
    parser.add_argument('--p_sdf_path', metavar='product', type=str, default=None,
                        help='A path to the product SDF file (direct mode)')
    parser.add_argument('--ts_xyz_path', metavar='ts', type=str, default=None,
                        help='A path to write the TS guess XYZ file to (direct mode)')
    args = parser.parse_args(command_line_args)
    return args


def main():
    """
    Run GCN to generate TS guesses.
    """
    args = parse_command_line_arguments()
    if args.yml_in_path is not None:
        initialize_gcn_run(input_dict=read_yaml_file(str(args.yml_in_path)))
    elif args.r_sdf_path is not None and args.p_sdf_path is not None and args.ts_xyz_path is not None:
        success = run_gcn(r_sdf_path=str(args.r_sdf_path),
                          p_sdf_path=str(args.p_sdf_path),
                          ts_xyz_path=str(args.ts_xyz_path),
                          )
        if not success or not os.path.isfile(str(args.ts_xyz_path)):
            sys.exit(1)
    else:
        print('Either --yml_in_path, or all of --r_sdf_path, --p_sdf_path, and --ts_xyz_path must be given.',
              file=sys.stderr)
        sys.exit(2)


if __name__ == '__main__':
    main()
