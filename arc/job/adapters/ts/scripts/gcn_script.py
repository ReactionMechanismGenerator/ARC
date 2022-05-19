#!/usr/bin/env python3
# encoding: utf-8

"""
A standalone script to retrieve xyz data from GCN,
should be run under the ts_gcn environment.
"""

import argparse
from typing import Optional

from rdkit import Chem

from inference import inference


def parse_command_line_arguments(command_line_args=None):
    """
    Parse command-line arguments.
    This uses the :mod:`argparse` module, which ensures that the command-line arguments are
    sensible, parses them, and returns them.
    """
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument('--r_sdf_path', metavar='Reactant', type=str, default='reactant.sdf',
                        help='a path to the reactant .sdf file')
    parser.add_argument('--p_sdf_path', metavar='Product', type=str, default='product.sdf',
                        help='a path to the product .sdf file')
    parser.add_argument('--ts_xyz_path', metavar='TS', type=str, default='TS.xyz',
                        help='a path to the output TS .xyz file')

    args = parser.parse_args(command_line_args)

    return args


def main(r_sdf_path: Optional[str] = None,
         p_sdf_path: Optional[str] = None,
         ts_xyz_path: Optional[str] = None,
         ) -> None:
    """
    Run GCN to generate TS guesses.

    Args:
        r_sdf_path (str, optional): A path to the reactant .sdf file.
        p_sdf_path (str, optional): A path to the product .sdf file.
        ts_xyz_path (str, optional): A path to the output TS .xyz file.
    """
    if any(arg is None for arg in [r_sdf_path, p_sdf_path, ts_xyz_path]):
        # Parse command-line arguments
        args = parse_command_line_arguments()
        r_sdf_path = str(args.r_sdf_path)
        p_sdf_path = str(args.p_sdf_path)
        ts_xyz_path = str(args.ts_xyz_path)

    # read in sdf files for reactant and product of the atom-mapped reaction
    r_mols = Chem.SDMolSupplier(r_sdf_path, removeHs=False, sanitize=True)
    p_mols = Chem.SDMolSupplier(p_sdf_path, removeHs=False, sanitize=True)

    try:
        inference(r_mols, p_mols, ts_xyz_path)
    except Exception:
        pass


if __name__ == '__main__':
    main()
