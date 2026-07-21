#!/usr/bin/env python3
# encoding: utf-8

"""
ARC - Automatic Rate Calculator
"""

import argparse
import logging
import os

from arc.common import read_yaml_file
from arc.main import ARC

try:
    from tckdb_arc.config import TCKDBConfig
    from tckdb_arc.sweep import run_upload_sweep
except ImportError:  # in-tree fallback until Phase 4 removes arc/tckdb/
    from arc.tckdb.config import TCKDBConfig
    from arc.tckdb.sweep import run_upload_sweep


def parse_command_line_arguments(command_line_args=None):
    """
    Parse command-line arguments.

    Args:
        command_line_args: The command line arguments.

    Returns:
        The parsed command-line arguments by keywords.
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
    args.file = args.file[0]

    return args


def main():
    """
    The main ARC executable function
    """
    args = parse_command_line_arguments()
    input_file = args.file
    project_directory = os.path.abspath(os.path.dirname(args.file))
    input_dict = read_yaml_file(path=input_file, project_directory=project_directory)
    if 'project' not in list(input_dict.keys()):
        raise ValueError('A project name must be provided!')

    verbose = logging.INFO
    if args.debug:
        verbose = logging.DEBUG
    elif args.quiet:
        verbose = logging.WARNING
    input_dict['verbose'] = input_dict['verbose'] if 'verbose' in input_dict else verbose
    if 'project_directory' not in input_dict or not input_dict['project_directory']:
        input_dict['project_directory'] = project_directory

    tckdb_config = TCKDBConfig.from_dict(input_dict.pop('tckdb', None))

    arc_object = ARC(**input_dict)
    arc_object.tckdb_config = tckdb_config
    if tckdb_config is not None:
        print(f'TCKDB integration enabled: {tckdb_config.base_url}')

    # Persistent SSH pool lives for the duration of the run; close it
    # explicitly on every exit path (success, error, ctrl-C) so we don't
    # leave paramiko Transports orphaned. Lazily instantiated on first
    # remote-queue job, so this is a no-op for fully-local runs.
    try:
        arc_object.execute()

        if tckdb_config is not None:
            try:
                from tckdb_arc.adapter import TCKDBAdapter
            except ImportError:
                from arc.tckdb.adapter import TCKDBAdapter
            adapter = TCKDBAdapter(tckdb_config, project_directory=arc_object.project_directory)
            run_upload_sweep(
                adapter=adapter,
                project_directory=arc_object.project_directory,
                tckdb_config=tckdb_config,
            )
    finally:
        from arc.job.ssh_pool import reset_default_pool
        reset_default_pool()


if __name__ == '__main__':
    main()
