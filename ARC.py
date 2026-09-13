#!/usr/bin/env python3
# encoding: utf-8

"""
ARC - Automatic Rate Calculator
"""

import argparse
from functools import lru_cache
import logging
import os

from arc.common import read_yaml_file
from arc.main import ARC


logger = logging.getLogger('arc')


@lru_cache(maxsize=1)
def _warn_missing_tckdb_package() -> None:
    """Log the optional-adapter warning at most once per ARC process."""
    logger.warning(
        'TCKDB upload requested, but the optional standalone tckdb-arc package is not installed; '
        'continuing without upload.'
    )


def run_tckdb_upload(tckdb_settings, project_directory: str) -> None:
    """Run the optional standalone TCKDB adapter after ARC completes.

    ``tckdb_settings`` is whatever the input file put under its ``tckdb`` key,
    so it is validated here rather than assumed to be a mapping.

    Uploading publishes the run's full scientific record to a remote endpoint,
    so it requires ``enabled: true`` stated explicitly in the ``tckdb`` block.
    Defaulting to "upload unless told otherwise" would mean a ``tckdb`` block
    written to configure anything else — a URL, a dry run — silently shipped
    the data. The resolved destination is logged before anything leaves the
    machine so the target is visible in the run log.
    """
    if not isinstance(tckdb_settings, dict):
        logger.warning("TCKDB upload skipped: the 'tckdb' entry in the input file is %s, "
                       "not a settings block. Write it as a mapping with 'enabled: true'.",
                       type(tckdb_settings).__name__)
        return
    if tckdb_settings.get('enabled') is not True:
        logger.info("TCKDB upload skipped: the 'tckdb' block does not set 'enabled: true'.")
        return
    try:
        import tckdb_arc
    except ModuleNotFoundError as exc:
        if exc.name == 'tckdb_arc':
            _warn_missing_tckdb_package()
            return
        raise
    from tckdb_arc.adapter import TCKDBAdapter
    from tckdb_arc.config import TCKDBConfig
    from tckdb_arc.sweep import run_upload_sweep

    config = TCKDBConfig.from_dict(tckdb_settings)
    if config is None:
        return
    logger.info(f'Uploading ARC results to TCKDB at '
                f'{tckdb_settings.get("url") or tckdb_settings.get("host") or "the configured endpoint"}.')
    adapter = TCKDBAdapter(config, project_directory=project_directory)
    run_upload_sweep(
        adapter=adapter,
        project_directory=project_directory,
        tckdb_config=config,
    )


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
    tckdb_settings = input_dict.pop('tckdb', None)
    arc_object = ARC(**input_dict)
    arc_object.execute()
    if tckdb_settings is not None:
        try:
            run_tckdb_upload(tckdb_settings, arc_object.project_directory)
        except Exception as exc:
            logger.error('The TCKDB upload failed: %s. The ARC run itself completed and its '
                         'results are on disk under %s.',
                         exc, arc_object.project_directory, exc_info=True)


if __name__ == '__main__':
    main()
