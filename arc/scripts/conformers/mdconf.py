#!/usr/bin/env python3
# encoding: utf-8

"""
ARC - Automatic Rate Calculator
Conformational MD FF Optimization
written by Alon Grinberg Dana

To run this module, the server has to have AmberTools and Gromacs installed.
Refer to the installation guides for more details:
AmberTools19: http://ambermd.org/Manuals.php
Gromacs 2019: http://manual.gromacs.org/documentation/2019/install-guide/index.html
"""

import argparse
import os
import re
import subprocess
import time
import yaml
from acpype import MolTopol

# try:
#     import ambertools
# except ImportError:
#     subprocess.call('conda install -c ambermd ambertools -y', shell=True)

cwd = os.getcwd()

# Parse the Gaussian output file and create Amber .ac, .mol2, and .esp files
ANTECHAMBER = """#!/bin/bash

antechamber -i {gaussian} -fi gout -o M00.ac -fo ac -rn M00
antechamber -i {gaussian} -fi gout -o M00.mol2 -fo mol2 -rn M00
espgen -i {gaussian} -o M00.esp

"""

# Process with Ambertools (after coords were modified for the specific conformer)
AMBERTOOLS = """#!/bin/bash

respgen -i M00.ac -o M00.respin1 -f resp1
respgen -i M00.ac -o M00.respin2 -f resp2
resp -O -i M00.respin1 -o M00.respout1 -e M00.esp -t qout_stage1
resp -O -i M00.respin2 -o M00.respout2 -e M00.esp -q qout_stage1 -t qout_stage2
antechamber -i M00.mol2 -fi mol2 -o M00_resp.mol2 -fo mol2 -c rc -cf qout_stage2
antechamber -i M00.ac -fi ac -o M00_resp.mol2 -fo mol2 -c rc -cf qout_stage2
parmchk2 -i M00_resp.mol2 -f mol2 -o M00.frcmod
antechamber -i M00_resp.mol2 -fi mol2 -o M00_resp.pdb -fo pdb
tleap -s -f M00.tleap

"""

# Change box size, run the GROMACS pre processor, run GROMACS
# We use here a costom mdp file rather than the md.mdp file generated by Ambertools
GROMACS = """#!/bin/bash

gmx editconf -f M00_GMX.gro -o M00_GMX.gro -c -box {size} {size} {size}
gmx grompp -c M00_GMX.gro -f {mdp} -p M00_GMX.top
gmx mdrun -s topol.tpr

"""


def read_yaml(path):
    """
    Read a YAML file.

    Args:
        (str): The YAML file path.

    Returns:
        list: The list saved to file.
    """
    if not os.path.isfile(path):
        raise IOError('The file {0} was not found and cannot be read.'.format(path))
    with open(path, 'r') as f:
        content = yaml.load(stream=f, Loader=yaml.FullLoader)
    return content


def purge(pattern=r'\.\d+#', directory=None):
    """
    Delete files that match a pattern.

    Args:
        pattern (str): The file name patters to search.
        directory (str, optional): The directory path to search is (default: current working directory).
    """
    if directory is None:
        directory = os.getcwd()
    for f in os.listdir(directory):
        if re.search(pattern, f):
            os.remove(os.path.join(directory, f))


def write_mol_files(coord, ac_path=None, mol2_path=None):
    """
    Modify an existing .mol2 file for the same molecule with updated coordinates.

    Args:
        coord (list): The coordinates of a single conformer in array form to be updated in the .mol2 file.
        ac_path (str, optional): The path to the .mol2 file to be modified.
        mol2_path (str, optional): The path to the .mol2 file to be modified.
    """
    ac_path = ac_path if ac_path is not None else 'M00.ac'
    mol2_path = mol2_path if mol2_path is not None else 'M00.mol2'

    with open(ac_path, 'r') as f:
        lines = f.readlines()
    content = ''
    change_xyz = False
    i = 0
    for line in lines:
        if 'BOND' in line:
            # e.g., `BOND    1    1    3    1     O1   C1`
            change_xyz = False
        if not change_xyz:
            content += line
        elif line.strip():
            # e.g., `ATOM      4  C2  M00     1      -2.759   0.222  -0.777  0.000000        c3`
            s0 = ' ' if coord[i][0] >= 0 else ''
            s1 = ' ' if coord[i][0] >= 0 else ''
            s2 = ' ' if coord[i][0] >= 0 else ''
            content += line[:32] + s0 + str(coord[i][0]) + '  ' + s1 + str(coord[i][1]) + '  ' + s2 + str(coord[i][2]) \
                + '  ' + line[56:] + '\n'
            i += 1
        if 'Formula' in line:
            # e.g., `Formula: H18 C13 O2 `
            change_xyz = True
    with open(ac_path, 'w') as f:
        f.write(content)

    with open(mol2_path, 'r') as f:
        lines = f.readlines()
    content = ''
    change_xyz = False
    i = 0
    for line in lines:
        if '<TRIPOS>' in line and 'ATOM' not in line:
            # e.g., `@<TRIPOS>BOND`
            change_xyz = False
        if not change_xyz:
            content += line
        else:
            # e.g., `      4 C2          -2.7590     0.2220    -0.7770 c3         1 M00       0.000000`
            splits = line.split()
            splits[2] = str(coord[i][0])
            splits[3] = str(coord[i][1])
            splits[4] = str(coord[i][2])
            content += '     ' + '   '.join(splits) + '\n'
            i += 1
        if '<TRIPOS>' in line and 'ATOM' in line:
            # i.e., `@<TRIPOS>ATOM`
            change_xyz = True
    with open(mol2_path, 'w') as f:
        f.write(str(content))


def amber_to_gromacs(g_path, coord=None, size=25, mdp_filename='mdp.mdp', first_iteration=False):
    """
    Use Amber Tools to train a force field and prepare all input files for a Gromacs minimization.

    Args:
        g_path (str): A path to the Gaussian output file.
        coord (list, optional): The 3D coordinates for a single conformer (ordered as in the Gaussian output file).
        size (float, optional): The box size used in MD simulations.
        mdp_filename (str, optional): The MD properties file path to use.
        first_iteration (bool, optional): Whether this is the first conformer. True if it is.

    Returns:
        str: The xyz coordinates of the optimized conformer in string format.
    Returns:
        float: The energy in kJ/mol of the optimized conformer.
    """
    if not first_iteration and not os.path.isfile(os.path.join(cwd, 'M00.mol2')):
        first_iteration = True
    if first_iteration:
        # creates the M00.ac, M00.mol2, and M00.esp files from the Gaussian output
        subprocess.call(ANTECHAMBER.format(gaussian=g_path), shell=True)
    if coord is not None:
        write_mol_files(coord)
    subprocess.call(AMBERTOOLS, shell=True)

    system = MolTopol(acFileXyz='M00.crd7', acFileTop='M00.parm7', basename='M00', verbose=False)
    system.writeGromacsTopolFiles(amb2gmx=True)

    subprocess.call(GROMACS.format(size=size, mdp=mdp_filename), shell=True)

    opt_xyz, e = '', None
    if os.path.isfile('md.log'):
        with open('md.log', 'r') as f:
            lines = f.readlines()
            for line in reversed(lines):
                if 'Potential Energy' in line:
                    e = float(line.split()[-1])  # kJ/mol
                    break
    if os.path.isfile('confout.gro'):
        with open('confout.gro', 'r') as f:
            lines = f.readlines()
            for line in lines:
                splits = line.split()
                if len(splits) > 4:
                    # e.g., `    1M00     C2    4   5.189   5.207   4.806`
                    symbol = ''.join([char for char in splits[1] if not char.isdigit()])
                    opt_xyz += symbol + ' ' * (4 - len(symbol))
                    for c in splits[3:6]:
                        opt_xyz += '' if '-' in c else ' '
                        opt_xyz += '   {0}'.format(c)
                    opt_xyz += '\n'
    return opt_xyz, e


def main():
    """
    The main function for the Conformational FF optimization.
    Note: it is crucial that the attom mapping is conserved between the representation in the Gaussian file
    and the YAML coordinates file.

    Command line argument:
        '-f': The ESS output file (default: gaussian.out).
        '-s': The FF box size in Angstroms (default: 10). Thumb-rule: 4 * radius (or 2 * diameter).
        '-m': The custom Molecular Dynamics parameter .mdp filename (default: mdp.mdp).

    Returns:
        list: Entries are lists of coordinates (in array form) and energies (in kJ/mol).
    """
    # Parse the command-line arguments (requires the argparse module)
    args = parse_command_line_arguments()
    t0 = time.time()
    path = args.file[0]
    size = args.size[0]
    mdp_filename = args.mdp[0]
    with open('coords.yml', 'r') as f:
        coords = yaml.load(stream=f, Loader=yaml.FullLoader)

    output = list()
    for i, coord in enumerate(coords):
        opt_xyz, e = amber_to_gromacs(g_path=path, coord=coord, size=size, mdp_filename=mdp_filename)
        output.append([opt_xyz, e])
        if i % 10 == 0:
            purge()
    purge()

    # save YAML output
    yaml.add_representer(str, string_representer)
    content = yaml.dump(data=output, encoding='utf-8')
    with open('output.yml', 'w') as f:
        f.write(content)
    dt = time.time() - t0
    print(dt)


def string_representer(dumper, data):
    """Add a custom string representer to use block literals for multiline strings"""
    if len(data.splitlines()) > 1:
        return dumper.represent_scalar(tag='tag:yaml.org,2002:str', value=data, style='|')
    return dumper.represent_scalar(tag='tag:yaml.org,2002:str', value=data)


def parse_command_line_arguments(command_line_args=None):
    """
    Parse the command-line arguments.
    """
    parser = argparse.ArgumentParser(description='ARC, Conformational FF optimization')
    parser.add_argument('-f', '--file', type=str, nargs=1, default=['gaussian.out'],
                        metavar='ess', help='The ESS output file')
    parser.add_argument('-s', '--size', type=int, nargs=1, default=[25],
                        metavar='size', help='The FF box size in Angstroms')
    parser.add_argument('-m', '--mdp', type=str, nargs=1, default=['mdp.mdp'],
                        metavar='ess', help='The Molecular Dynamics parameter file name')
    arguments = parser.parse_args(command_line_args)
    return arguments


if __name__ == '__main__':
    main()
