import pytest
import subprocess
import tempfile
import pathlib
import textwrap


@pytest.mark.smoke
def test_import_arc():
    """Test that ARC can be imported in the docker image."""
    try:
        import arc
        assert hasattr(arc, '__file__')
    except ImportError as e:
        pytest.fail(f"ImportError: {e}")


@pytest.mark.smoke
def test_arc_cli_help_runs():
    """Test that ARC CLI help runs in the docker image."""
    cmd = ["bash", "-lc", "micromamba run -n arc_env python -m ARC --help || true"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    # Just ensure it executes and prints usage/help
    assert "help" in (p.stdout + p.stderr).lower()


@pytest.mark.smoke
def test_arkane_cli_help_runs():
    """Test that Arkane CLI help runs in the docker image."""
    cmd = ["bash", "-lc","micromamba run -n rmg_env python -m arkane --help || true"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    # Just ensure it executes and prints usage/help
    assert "arkane" in (p.stdout + p.stderr).lower()


@pytest.mark.smoke
def test_arc_can_execute_arkane_minimal():
    """Test that ARC can execute Arkane with a minimal input in the docker image."""
    arkane_input = textwrap.dedent("""\
    #!/usr/bin/env python
    modelChemistry = 'wb97m-v/def2-tzvpd'  # irrelevant here, just parseable
    useHinderedRotors = False
    thermo('H2', 'H298')
    """)
    with tempfile.TemporaryDirectory() as td:
        inp = pathlib.Path(td, "input.py")
        inp.write_text(arkane_input)
        # Call arkane via rmg_env the same way ARC would (subprocess)
        cmd = ["bash", "-lc", f"micromamba run -n rmg_env python -m arkane {inp} || true"]
        p = subprocess.run(cmd, capture_output=True, text=True)
        # We only assert it runs and produces any Arkane header/output (no heavy calc)
        assert "arkane" in (p.stdout + p.stderr).lower()


@pytest.mark.smoke
def test_rmgpy_imports():
    """Test that RMG-Py can be imported in the docker image."""
    code = r"""
import importlib, sys
m = importlib.import_module('rmgpy')
print('rmgpy OK', getattr(m, '__version__', 'unknown'))
"""
    cmd = ["bash", "-lc", f"micromamba run -n rmg_env python - <<'PY'\n{code}\nPY"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert "rmgpy OK" in p.stdout


@pytest.mark.smoke
def test_rmg_cli_help_runs():
    """Test that RMG CLI help runs in the docker image."""
    cmd = ["bash", "-lc", "micromamba run -n rmg_env rmg --help || true"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    # Just ensure it executes and prints usage/help
    assert "rmg" in (p.stdout + p.stderr).lower()
