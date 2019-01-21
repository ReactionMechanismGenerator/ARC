#!/usr/bin/env python
# encoding: utf-8

"""
ARC - Automatic Rate Calculator
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)
import argparse
import os
import logging

import yaml
try:
    from yaml import CDumper as Dumper, CLoader as Loader, CSafeLoader as SafeLoader
except ImportError:
    from yaml import Dumper, Loader, SafeLoader

from arc.main import ARC

################################################################################


def parse_command_line_arguments(command_line_args=None):
    """
    Parse the command-line arguments being passed to ARC. This uses the
    :mod:`argparse` module, which ensures that the command-line arguments are
    sensible, parses them, and returns them.
    """

    parser = argparse.ArgumentParser(description='Automatic Rate Calculator (ARC)')
    parser.add_argument('file', metavar='FILE', type=str, nargs=1,
                        help='a file describing the job to execute')

    # Options for controlling the amount of information printed to the console
    # By default a moderate level of information is printed; you can either
    # ask for less (quiet), more (verbose), or much more (debug)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-d', '--debug', action='store_true', help='print debug information')
    group.add_argument('-q', '--quiet', action='store_true', help='only print warnings and errors')

    args = parser.parse_args(command_line_args)

    # Process args to set correct default values and format

    # For output and scratch directories, if they are empty strings, set them
    # to match the input file location
    args.file = args.file[0]

    # Set directories
    # input_directory = os.path.abspath(os.path.dirname(args.file))

    return args


def main():
    # Parse the command-line arguments (requires the argparse module)
    args = parse_command_line_arguments()
    input_file = args.file
    input_dict = read_file(input_file)
    project_directory = os.path.abspath(os.path.dirname(args.file))
    try:
        input_dict['project']
    except KeyError:
        print('A project name must be provided!')

    verbose = logging.INFO
    if args.debug:
        verbose = logging.DEBUG
    elif args.quiet:
        verbose = logging.WARNING
    try:
        input_dict['verbose']
    except KeyError:
        input_dict['verbose'] = verbose
    arc_object = ARC(input_dict=input_dict, project_directory=project_directory)
    arc_object.execute()


def read_file(path):
    """
    Read the ARC YAML input file and return the parameters in a dictionary
    """
    if not os.path.isfile(path):
        raise ValueError('Could not find the input file {0}'.format(path))
    with open(path, 'r') as f:
        input_dict = yaml.load(stream=f)
    return input_dict


################################################################################

if __name__ == '__main__':
    main()
