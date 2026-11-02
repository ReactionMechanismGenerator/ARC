"""
A common module for subprocess scripts with importable functions outside the environment of ARC
"""
from __future__ import annotations

import argparse
import os
import yaml


def parse_command_line_arguments(command_line_args=None):
    """
    Parse command-line arguments.

    Args:
        command_line_args: The command line arguments.

    Returns:
        The parsed command-line arguments by keywords.
        ``args.file`` is the input YAML path (positional).
        ``args.output`` is an optional output path (``-o``/``--output``); when omitted,
        callers should default to overwriting ``args.file`` to preserve historical behavior.
    """
    parser = argparse.ArgumentParser(description='Automatic Rate Calculator (ARC)')
    parser.add_argument('file', metavar='FILE', type=str, nargs=1, help='a file with input information')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='optional output YAML path; if omitted, the input file is overwritten')
    args = parser.parse_args(command_line_args)
    args.file = args.file[0]
    return args


def read_yaml_file(path: str) -> dict | list:
    """
    Read a YAML file (usually an input / restart file, but also conformers file)
    and return the parameters as python variables.

    Args:
        path (str): The YAML file path to read.

    Returns: dict | list
        The content read from the file.
    """
    if not isinstance(path, str):
        raise ValueError(f'path must be a string, got {path} which is a {type(path)}')
    if not os.path.isfile(path):
        raise ValueError(f'Could not find the YAML file {path}')
    with open(path, 'r') as f:
        content = yaml.load(stream=f, Loader=yaml.FullLoader)
    return content


def save_yaml_file(path: str, content: list | dict) -> None:
    """
    Save a YAML file (usually an input / restart file, but also conformers file).

    Args:
        path (str): The YAML file path to save.
        content (list, dict): The content to save.
    """
    if not isinstance(path, str):
        raise ValueError(f'path must be a string, got {path} which is a {type(path)}')
    yaml_str = to_yaml(py_content=content)
    if '/' in path and os.path.dirname(path) and not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'w') as f:
        f.write(yaml_str)


def to_yaml(py_content: list | dict) -> str:
    """
    Convert a Python list or dictionary to a YAML string format.

    Args:
        py_content (list, dict): The Python content to save.

    Returns: str
        The corresponding YAML representation.
    """
    yaml.add_representer(str, string_representer)
    yaml_str = yaml.dump(data=py_content)
    return yaml_str


def string_representer(dumper, data):
    """
    Add a custom string representer to use block literals for multiline strings.
    """
    if len(data.splitlines()) > 1:
        return dumper.represent_scalar(tag='tag:yaml.org,2002:str', value=data, style='|')
    return dumper.represent_scalar(tag='tag:yaml.org,2002:str', value=data)
