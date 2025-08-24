#!/usr/bin/env python3

"""
This module contains utilities related to compilation and code maintenance.
"""

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys


def check_dependencies():
    """
    Checks for and locates major dependencies that RMG requires.
    """
    print('\nChecking vital dependencies...\n')
    print('{0:<15}{1:<15}{2}'.format('Package', 'Version', 'Location'))

    missing = {
        'openbabel': _check_openbabel(),
        'rdkit': _check_rdkit(),
    }

    if any(missing.values()):
        print("""
There are missing dependencies as listed above. Please install them before proceeding.

Using Anaconda, these dependencies can be individually installed from the RMG channel as follows:

    conda install -c rmg [package name]
{0}
You can alternatively update your environment and install all missing dependencies as follows:

    conda env update -f environment.yml

Be sure to activate your conda environment (rmg_env by default) before installing or updating.
""".format("""
RDKit should be installed from the RDKit channel instead:

    conda install -c rdkit rdkit
""" if missing['rdkit'] else ''))
    else:
        print("""
Everything was found :)
""")


def _check_openbabel():
    """Check for OpenBabel"""
    missing = False

    try:
        from openbabel import openbabel
    except ImportError:
        print('{0:<30}{1}'.format('OpenBabel',
                                  'Not found. Necessary for SMILES/InChI functionality for nitrogen compounds.'))
        missing = True
    else:
        version = openbabel.OBReleaseVersion()
        location = openbabel.__file__
        print('{0:<15}{1:<15}{2}'.format('OpenBabel', version, location))

    return missing


def _check_rdkit():
    """Check for RDKit"""
    missing = False

    try:
        import rdkit
        from rdkit import Chem
    except ImportError:
        print('{0:<30}{1}'.format('RDKit',
                                  'Not found. Please install RDKit version 2015.03.1 or later with InChI support.'))
        missing = True
    else:
        try:
            version = rdkit.__version__
        except AttributeError:
            version = False
        location = rdkit.__file__
        inchi = Chem.inchi.INCHI_AVAILABLE

        if version:
            print('{0:<15}{1:<15}{2}'.format('RDKit', version, location))
            if not inchi:
                print('    !!! RDKit installed without InChI Support. Please install with InChI.')
                missing = True
        else:
            print('    !!! RDKit version out of date, please install RDKit version 2015.03.1 or later with InChI support.')
            missing = True

    return missing


def check_python():
    """
    Check that Python 3 is in the environment.
    """
    major = sys.version_info.major
    minor = sys.version_info.minor
    if not (major == 3 and minor >= 7):
        sys.exit('\nRMG-Py requires Python 3.7 or higher. You are using Python {0}.{1}.\n\n'
                 'If you are using Anaconda, you should create a new environment using\n\n'
                 '    conda env create -f environment.yml\n\n'
                 'If you have an existing rmg_env, you can remove it using\n\n'
                 '    conda remove --name rmg_env --all\n'.format(major, minor))


def clean(subdirectory=''):
    """
    Removes files generated during compilation.

    For *nix systems, remove `.so` files and `.pyc` files.
    For Windows, remove `.pyd` files and `.pyc` files.
    """
    if platform.system() == 'Windows':
        extensions = ['.pyd', '.pyc', '.c']
    else:
        extensions = ['.so', '.pyc', '.c']
    # Remove temporary build files
    print('Removing build directory...')
    directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'build', subdirectory)
    try:
        shutil.rmtree(directory)
    except FileNotFoundError:
        print(f'{directory} not found. Unable to clean up build directory.')
    # Remove C shared object files and compiled Python files
    print('Removing compiled files...')
    directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), subdirectory)
    for root, dirs, files in os.walk(directory):
        for f in files:
            ext = os.path.splitext(f)[1]
            if ext in extensions:
                os.remove(os.path.join(root, f))
    print('Cleanup completed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ARC Code Utilities')
    parser.add_argument('command', metavar='COMMAND', type=str,
                        choices=['check-dependencies',
                                 'check-python',
                                 'clean',
                                 'clean-solver',
                                #  'update-headers'],
                        ],
                        help='command to execute')
    args = parser.parse_args()
    if args.command == 'check-dependencies':
        check_dependencies()
    elif args.command == 'check-python':
        check_python()
    elif args.command == 'clean':
        clean()
