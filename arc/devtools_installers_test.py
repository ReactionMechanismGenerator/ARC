#!/usr/bin/env python3
# encoding: utf-8

"""
Checks that every devtools script which activates an env initialises the shell through
``devtools/shell_init.sh``, and tests that helper against stub frontends.
"""

import os
import re
import shutil
import subprocess

import stat
import tempfile
import textwrap

import pytest


DEVTOOLS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'devtools')
SHELL_INIT = os.path.join(DEVTOOLS, 'shell_init.sh')
INSTALLERS = sorted(f for f in os.listdir(DEVTOOLS) if f.endswith('.sh') and f != 'shell_init.sh')

# A frontend token, optionally quoted, then activate/deactivate; not a path such as activate.d.
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
    ('set +u; $COMMAND_PKG activate "$GOFLOW_ENV_NAME"; set -u', True),
    ('${COMMAND_PKG} activate xtb_env', True),
    ('"$COMMAND_PKG" activate xtb_env', True),
    ('"${COMMAND_PKG}" deactivate', True),
    ("'$COMMAND_PKG' activate env", True),
    ('conda activate arc_env', True),
    ('mamba deactivate', True),
    ('micromamba activate ts_gcn', True),
    ('act_dir="$prefix/etc/conda/activate.d"', False),
    ('DEACTIVATE_DIR="$ENV_PREFIX/etc/conda/deactivate.d"', False),
    ('--conda  install hooks into conda activate/deactivate', False),
    ('"$COMMAND_PKG" run -n arc_env python -c "import arc"', False),
    ('echo "To run a UMA job, activate arc_env and set method to uma"', False),
]


@pytest.mark.parametrize('line, flagged', ACTIVATE_SPELLINGS)
def test_activate_re_matches_the_spellings_installers_use(line: str, flagged: bool):
    """ACTIVATE_RE flags every activation spelling and nothing else."""
    assert bool(ACTIVATE_RE.search(line)) is flagged, line


def test_installers_found():
    """The scan sees the devtools scripts and detects every one known to activate an env."""
    assert len(INSTALLERS) >= 10
    for name in ['compile.sh', 'install_gcn.sh', 'install_goflow.sh', 'install_rits.sh', 'install_xtb.sh']:
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


def write_stub(directory: str, name: str, body: str) -> None:
    """
    Write an executable bash stub.

    Args:
        directory (str): The directory to write it into.
        name (str): The file name.
        body (str): The script body, after the shebang.
    """
    path = os.path.join(directory, name)
    with open(path, 'w') as f:
        f.write('#!/usr/bin/env bash\n' + textwrap.dedent(body))
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)


MAMBA_1 = """\
    if [[ "$1" == shell ]]; then
        echo "mamba: error: argument command: invalid choice: 'shell'" >&2
        exit 2
    fi
    """


@pytest.fixture
def stub_dir():
    """A temporary directory for stub frontends, removed afterwards."""
    with tempfile.TemporaryDirectory() as tmp:
        yield tmp


def stub_conda_base(stub_dir: str, mamba_sh: bool = True, info_fails: bool = False) -> None:
    """
    Write a stub ``conda`` whose base holds ``conda.sh`` and, optionally, ``mamba.sh``.

    Args:
        stub_dir (str): The stub directory on PATH.
        mamba_sh (bool): Whether the base ships ``etc/profile.d/mamba.sh``.
        info_fails (bool): Whether ``conda info --base`` fails.
    """
    base = os.path.join(stub_dir, 'base')
    profile = os.path.join(base, 'etc', 'profile.d')
    os.makedirs(profile)
    with open(os.path.join(profile, 'conda.sh'), 'w') as f:
        f.write('conda () { echo "conda-fn $*"; }\n')
    if mamba_sh:
        with open(os.path.join(profile, 'mamba.sh'), 'w') as f:
            f.write('mamba () { echo "mamba-fn $*"; }\n')
    if info_fails:
        write_stub(stub_dir, 'conda', 'echo "CondaError: stub info failure" >&2\nexit 1\n')
    else:
        write_stub(stub_dir, 'conda', f'echo "{base}"\n')


def test_shell_init_mamba_2_uses_its_own_hook(stub_dir: str):
    """mamba >= 2: the hook mamba prints defines the mamba function."""
    write_stub(stub_dir, 'mamba', """\
        if [[ "$1 $2" == "shell hook" ]]; then echo 'mamba () { echo "mamba-fn $*"; }'; exit 0; fi
        exit 1
        """)
    result = run_shell_init('mamba', path=f'{stub_dir}:/usr/bin:/bin', check='mamba activate x')
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'mamba-fn activate x'


def test_shell_init_mamba_1_falls_back_to_profile_scripts(stub_dir: str):
    """mamba 1.x: no shell hook, so conda.sh and mamba.sh from the conda base define the functions."""
    write_stub(stub_dir, 'mamba', MAMBA_1)
    stub_conda_base(stub_dir)
    result = run_shell_init('mamba', path=f'{stub_dir}:/usr/bin:/bin', check='mamba activate x; conda activate y')
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ['mamba-fn activate x', 'conda-fn activate y']
    assert result.stderr == ''


@pytest.mark.parametrize('info_fails, mamba_sh, reason', [
    (True, True, "fallback for mamba 1.x: 'conda info --base' failed"),
    (False, False, 'fallback for mamba 1.x: sourcing'),
])
def test_shell_init_mamba_1_fallback_failure_is_reported(stub_dir: str, info_fails: bool, mamba_sh: bool, reason: str):
    """A failed mamba 1.x fallback names the fallback step, after mamba's own error."""
    write_stub(stub_dir, 'mamba', MAMBA_1)
    stub_conda_base(stub_dir, mamba_sh=mamba_sh, info_fails=info_fails)
    result = run_shell_init('mamba', path=f'{stub_dir}:/usr/bin:/bin', check='echo reached')
    assert result.returncode != 0
    assert 'reached' not in result.stdout
    lines = result.stderr.splitlines()
    assert lines[-1].startswith('❌'), result.stderr
    assert lines[-2].startswith(reason), result.stderr
    assert lines[-3] == "mamba shell hook: mamba: error: argument command: invalid choice: 'shell'", result.stderr
    if info_fails:
        assert 'CondaError: stub info failure' in result.stderr


@pytest.mark.parametrize('command_pkg', ['mamba', 'micromamba'])
def test_shell_init_fails_when_hook_defines_no_function(stub_dir: str, command_pkg: str):
    """A hook that runs but defines no shell function is a failure, not a silent success."""
    write_stub(stub_dir, command_pkg, 'echo true\n')
    result = run_shell_init(command_pkg, path=f'{stub_dir}:/usr/bin:/bin', check='echo reached')
    assert result.returncode != 0
    assert 'reached' not in result.stdout
    assert 'Could not initialise' in result.stderr
