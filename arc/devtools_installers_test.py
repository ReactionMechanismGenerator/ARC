#!/usr/bin/env python3
# encoding: utf-8

"""
Static and shell-level checks for the devtools installer scripts.

Any installer that calls ``activate``/``deactivate`` must initialise the shell for the selected
package manager through the shared ``devtools/shell_init.sh`` helper first: sourcing only
``conda.sh`` defines the ``conda`` shell function but not ``mamba``'s, so ``mamba activate`` aborts
with "Shell not initialized" on a shell where neither was initialised.
"""

import os
import re
import shutil
import subprocess

import pytest


DEVTOOLS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'devtools')
SHELL_INIT = os.path.join(DEVTOOLS, 'shell_init.sh')
INSTALLERS = sorted(f for f in os.listdir(DEVTOOLS) if f.startswith('install_') and f.endswith('.sh'))

# The frontend token may be quoted -- "$COMMAND_PKG" activate is the spelling shellcheck asks for --
# so the quotes are optional on both sides. Under-matching here is silent: an installer this misses
# drops out of ACTIVATING_INSTALLERS and both shared-init checks below skip it rather than fail it.
# The trailing (?![/.]) keeps paths such as etc/conda/activate.d, and prose such as
# "conda activate/deactivate", out. test_activate_re_matches_the_spellings_installers_use pins both
# halves of that contract.
ACTIVATE_RE = re.compile(r'''["']?(?:\$COMMAND_PKG|\$\{COMMAND_PKG\}|\bconda|\bmamba|\bmicromamba)["']?'''
                         r'\s+(?:de)?activate\b(?![/.])')
SOURCE_INIT_RE = re.compile(r'(?:^|[;&]\s*)(?:source|\.)\s+.*shell_init\.sh')
CALL_INIT_RE = re.compile(r'(?:^|[;&|]\s*)arc_shell_init\b')
LOCAL_INIT_RE = re.compile(r'shell[ .]hook|shell\.bash\s+hook|profile\.d/(?:conda|mamba)\.sh')


def code_lines(name: str) -> list[tuple[int, str]]:
    """
    Read an installer and return its non-comment lines.

    Args:
        name (str): The script file name inside ``devtools/``.

    Returns:
        list[tuple[int, str]]: ``(line number, stripped line)`` for every line that is not blank or a comment.
    """
    with open(os.path.join(DEVTOOLS, name)) as f:
        lines = [(i, line.strip()) for i, line in enumerate(f, start=1)]
    return [(i, line) for i, line in lines if line and not line.startswith('#')]


def first_line(name: str, pattern: re.Pattern) -> int | None:
    """
    Find the first non-comment line of an installer that matches a pattern.

    Args:
        name (str): The script file name inside ``devtools/``.
        pattern (re.Pattern): The pattern to search for.

    Returns:
        int | None: The 1-based line number, or ``None`` if no line matches.
    """
    return next((i for i, line in code_lines(name) if pattern.search(line)), None)


ACTIVATING_INSTALLERS = [name for name in INSTALLERS if first_line(name, ACTIVATE_RE) is not None]


def run_shell_init(command_pkg: str, path: str | None = None, check: str = '') -> subprocess.CompletedProcess:
    """
    Source the helper in a fresh, non-interactive bash and call ``arc_shell_init``.

    Args:
        command_pkg (str): The value to give ``COMMAND_PKG``.
        path (str, optional): A PATH to run under. Defaults to the caller's PATH.
        check (str): Shell code to run after a successful ``arc_shell_init``.

    Returns:
        subprocess.CompletedProcess: The finished bash process, with text stdout and stderr.
    """
    env = {'PATH': path or os.environ.get('PATH', ''), 'HOME': os.environ.get('HOME', '/tmp'), 'COMMAND_PKG': command_pkg}
    if 'MAMBA_ROOT_PREFIX' in os.environ:
        env['MAMBA_ROOT_PREFIX'] = os.environ['MAMBA_ROOT_PREFIX']
    script = f'set -euo pipefail; source "{SHELL_INIT}"; arc_shell_init; {check}'
    return subprocess.run(['bash', '--norc', '--noprofile', '-c', script], env=env, capture_output=True, text=True, timeout=120)


ACTIVATE_SPELLINGS = [
    # (line, should the detector flag it)
    ('set +u; $COMMAND_PKG activate "$GOFLOW_ENV_NAME"; set -u', True),
    ('${COMMAND_PKG} activate xtb_env', True),
    ('"$COMMAND_PKG" activate xtb_env', True),       # the spelling shellcheck asks for
    ('"${COMMAND_PKG}" deactivate', True),
    ("'$COMMAND_PKG' activate env", True),
    ('conda activate arc_env', True),
    ('mamba deactivate', True),
    ('micromamba activate ts_gcn', True),
    # Not an activation: a path, a different subcommand, or prose.
    ('act_dir="$prefix/etc/conda/activate.d"', False),
    ('DEACTIVATE_DIR="$ENV_PREFIX/etc/conda/deactivate.d"', False),
    ('--conda  install hooks into conda activate/deactivate', False),
    ('"$COMMAND_PKG" run -n arc_env python -c "import arc"', False),
    ('echo "To run a UMA job, activate arc_env and set method to uma"', False),
]


@pytest.mark.parametrize('line, flagged', ACTIVATE_SPELLINGS)
def test_activate_re_matches_the_spellings_installers_use(line: str, flagged: bool):
    """The detector flags every way an installer can activate, and nothing else.

    This guards the guard. A miss is silent in the direction that matters: an installer
    ACTIVATE_RE does not match is dropped from ACTIVATING_INSTALLERS, so the two shared-init
    checks below never run against it and the suite stays green while that installer has no
    initialisation at all. Over-matching is the safe direction -- it only adds a script to the
    checked set, where a wrong answer fails loudly -- which is why the quotes are optional rather
    than paired.
    """
    assert bool(ACTIVATE_RE.search(line)) is flagged, line


def test_installers_found():
    """The glob sees the installers, and every installer known to activate an env is detected."""
    assert len(INSTALLERS) >= 10
    # Named explicitly, not counted: `assert ACTIVATING_INSTALLERS` would stay green if the
    # detector silently stopped recognising three of the four.
    for name in ['install_gcn.sh', 'install_goflow.sh', 'install_rits.sh', 'install_xtb.sh']:
        assert name in ACTIVATING_INSTALLERS, f'{name} activates an env but the detector missed it'


@pytest.mark.parametrize('name', ACTIVATING_INSTALLERS)
def test_activating_installers_use_shared_shell_init(name: str):
    """Every installer that activates sources the helper and calls it before its first activate line."""
    activate = first_line(name, ACTIVATE_RE)
    source = first_line(name, SOURCE_INIT_RE)
    call = first_line(name, CALL_INIT_RE)
    assert source is not None, f'{name} activates (line {activate}) but never sources devtools/shell_init.sh'
    assert call is not None, f'{name} activates (line {activate}) but never calls arc_shell_init'
    assert source <= call < activate, f'{name}: source (line {source}) and arc_shell_init (line {call}) must precede activate (line {activate})'


@pytest.mark.parametrize('name', ACTIVATING_INSTALLERS)
def test_activating_installers_carry_no_local_init(name: str):
    """The shell init rule lives only in the helper, so no activating installer keeps its own copy."""
    local = [i for i, line in code_lines(name) if LOCAL_INIT_RE.search(line)]
    assert not local, f'{name} carries its own shell init on lines {local}; use arc_shell_init from devtools/shell_init.sh'


@pytest.mark.parametrize('command_pkg', ['', 'bogus'])
def test_shell_init_fails_closed_on_unknown_frontend(command_pkg: str):
    """An empty or unknown COMMAND_PKG is an error, not a silent no-op."""
    result = run_shell_init(command_pkg, check='echo reached')
    assert result.returncode != 0
    assert 'reached' not in result.stdout


@pytest.mark.parametrize('command_pkg', ['micromamba', 'mamba', 'conda'])
def test_shell_init_fails_closed_when_frontend_missing(command_pkg: str):
    """A frontend that is not on PATH makes arc_shell_init fail instead of evaluating an empty hook."""
    if shutil.which(command_pkg, path='/usr/bin:/bin'):
        pytest.skip(f'{command_pkg} is installed system-wide')
    result = run_shell_init(command_pkg, path='/usr/bin:/bin', check='echo reached')
    assert result.returncode != 0
    assert 'reached' not in result.stdout


@pytest.mark.parametrize('command_pkg', ['conda', 'mamba', 'micromamba'])
def test_shell_init_defines_activate_function(command_pkg: str):
    """With the frontend on PATH, arc_shell_init turns it into a shell function that can activate."""
    if shutil.which(command_pkg) is None:
        pytest.skip(f'{command_pkg} is not on PATH')
    result = run_shell_init(command_pkg, check=f'type -t {command_pkg}; set +u; {command_pkg} activate base; echo "prefix=$CONDA_PREFIX"')
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[0] == 'function'
    prefixes = [line.removeprefix('prefix=') for line in lines if line.startswith('prefix=')]
    assert len(prefixes) == 1 and os.path.isdir(prefixes[0]), result.stdout
