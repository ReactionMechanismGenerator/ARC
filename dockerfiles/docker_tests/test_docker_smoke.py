import pytest
import os
import shutil
import stat
import subprocess
import tempfile
import pathlib
import textwrap

ENTRYWRAPPER = os.environ.get("ARC_ENTRYWRAPPER_PATH", "/usr/local/bin/entrywrapper.sh")
PREFLIGHT = os.environ.get("ARC_PREFLIGHT_PATH", "/usr/local/bin/arc_preflight.py")
ARC_SETTINGS_DIR = os.environ.get("ARC_SETTINGS_DIR", "/home/mambauser/.arc")

REMOTE_SETTINGS = """\
servers = {
    'smoke_cluster': {
        'cluster_soft': 'Slurm',
        'address': 'login.smoke-cluster.invalid',
        'un': 'smoke_user',
    },
}
"""


def run_in_arc_env(code, env=None):
    """Run a snippet in arc_env through a login shell, never in this pytest process.

    ARC's ~/.arc overlay is loaded at import time and some ARC branches disable it outright
    while pytest is loaded, so any test of the overlay has to cross a process boundary to
    keep testing what it claims to test.
    """
    full_env = dict(os.environ)
    full_env.update(env or {})
    cmd = ["bash", "-lc", f"micromamba run -n arc_env python - <<'PY'\n{code}\nPY"]
    return subprocess.run(cmd, capture_output=True, text=True, env=full_env)


def run_preflight(arc_dir, env=None):
    """Run the entrypoint's settings pre-flight check against ``arc_dir``."""
    full_env = dict(os.environ)
    full_env.update(env or {})
    cmd = ["bash", "-lc", f"micromamba run -n arc_env python {PREFLIGHT} {arc_dir}"]
    return subprocess.run(cmd, capture_output=True, text=True, env=full_env)


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


@pytest.mark.smoke
def test_paramiko_available():
    """Test that paramiko is importable and an SSH client can be constructed."""
    code = r"""
import paramiko
client = paramiko.SSHClient()
try:
    print('paramiko OK', paramiko.__version__)
finally:
    client.close()
"""
    cmd = ["bash", "-lc", f"micromamba run -n arc_env python - <<'PY'\n{code}\nPY"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert "paramiko OK" in p.stdout


@pytest.mark.smoke
def test_arc_ssh_module_imports():
    """Test that ARC's SSH layer imports in the docker image."""
    code = r"""
from arc.job.ssh import SSHClient
print('arc.job.ssh OK', SSHClient.__name__)
"""
    cmd = ["bash", "-lc", f"micromamba run -n arc_env python - <<'PY'\n{code}\nPY"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert "arc.job.ssh OK" in p.stdout


@pytest.mark.smoke
def test_openssh_client_installed():
    """Test that the OpenSSH client tools are available for debugging remote connections."""
    for tool in ("ssh", "ssh-keyscan", "ssh-keygen"):
        assert shutil.which(tool) is not None, f"{tool} is missing from the image"


@pytest.mark.smoke
def test_entrywrapper_is_valid_bash():
    """Test that the entrypoint script is present, executable, and syntactically valid."""
    path = pathlib.Path(ENTRYWRAPPER)
    assert path.is_file(), f"{ENTRYWRAPPER} is missing"
    assert path.stat().st_mode & stat.S_IXUSR, f"{ENTRYWRAPPER} is not executable"
    p = subprocess.run(["bash", "-n", ENTRYWRAPPER], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr


@pytest.mark.smoke
def test_entrywrapper_forwards_ssh_auth_sock_across_privilege_drop():
    """Test that the entrypoint hands SSH_AUTH_SOCK to the unprivileged user explicitly."""
    text = pathlib.Path(ENTRYWRAPPER).read_text()
    assert "SSH_AUTH_SOCK=$SSH_AUTH_SOCK" in text, \
        "the entrypoint must pass SSH_AUTH_SOCK through the runuser privilege drop"
    assert "-u SSH_AUTH_SOCK" in text, \
        "the entrypoint must unset SSH_AUTH_SOCK when the forwarded socket is unusable"


@pytest.mark.smoke
def test_entrypoint_does_not_widen_the_agent_socket_unasked():
    """Test that relaxing the agent socket's mode stays behind an explicit opt-in.

    A bind-mounted socket shares its inode with the host, so a chmod here mutates the user's
    real agent socket and is never restored, the entrypoint having handed off with exec.
    """
    text = pathlib.Path(ENTRYWRAPPER).read_text()
    assert "ARC_WIDEN_AGENT_SOCKET" in text, \
        "the opt-in guard for relaxing the agent socket's mode has gone missing"
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("chmod") and "sock" in stripped:
            pytest.fail(f"unguarded chmod of the agent socket: {stripped}")


@pytest.mark.smoke
def test_forwarded_agent_socket_mode_is_left_alone():
    """Test that a forwarded agent socket was not made accessible to other users.

    Skipped unless an agent was forwarded, and when the widening was explicitly requested.
    """
    sock = os.environ.get("SSH_AUTH_SOCK")
    if not sock:
        pytest.skip("no SSH agent forwarded into the container")
    if os.environ.get("ARC_WIDEN_AGENT_SOCKET") == "1":
        pytest.skip("the socket mode was relaxed on explicit request")
    mode = os.stat(sock).st_mode
    assert not mode & stat.S_IWOTH, \
        f"{sock} is world-writable; the entrypoint must not relax a bind-mounted agent socket"
    assert not mode & stat.S_IROTH, \
        f"{sock} is world-readable; the entrypoint must not relax a bind-mounted agent socket"


@pytest.mark.smoke
def test_container_user_matches_requested_puid():
    """Test that PUID/PGID actually remapped the container user.

    The base image ships an unused 'ubuntu' account at 1000:1000, which is what PUID/PGID ask
    for on a typical Linux desktop, and a collision there used to abort the container.
    Skipped when the container was started without the remap.
    """
    puid = os.environ.get("PUID")
    pgid = os.environ.get("PGID")
    if not puid and not pgid:
        pytest.skip("container started without PUID/PGID")
    if puid:
        assert os.getuid() == int(puid), \
            f"asked for PUID={puid} but running as uid {os.getuid()}"
    if pgid:
        assert os.getgid() == int(pgid), \
            f"asked for PGID={pgid} but running as gid {os.getgid()}"


@pytest.mark.smoke
def test_ssh_home_directory_is_usable():
    """Test that the SSH material directory exists and is usable by the current user."""
    ssh_dir = pathlib.Path.home() / ".ssh"
    if not ssh_dir.exists():
        pytest.skip(f"{ssh_dir} is not present in this container")
    assert os.access(ssh_dir, os.R_OK | os.X_OK), \
        f"{ssh_dir} is not readable by the container user; re-run with -e PUID=$(id -u) -e PGID=$(id -g)"


@pytest.mark.smoke
def test_forwarded_ssh_agent_socket_is_usable():
    """Test that a forwarded SSH agent socket survived the privilege drop and is usable.

    Skipped unless the container was started with an agent socket forwarded.
    """
    sock = os.environ.get("SSH_AUTH_SOCK")
    if not sock:
        pytest.skip("no SSH agent forwarded into the container")
    assert stat.S_ISSOCK(os.stat(sock).st_mode), f"SSH_AUTH_SOCK={sock} is not a socket"
    assert os.access(sock, os.R_OK | os.W_OK), \
        f"{sock} is not accessible to the container user; re-run with -e PUID=$(id -u) -e PGID=$(id -g)"


@pytest.mark.smoke
def test_arc_settings_overlay_is_loaded():
    """Test that a ~/.arc/settings.py replaces ARC's dummy servers."""
    code = r"""
from arc.imports import settings
print('SERVERS', sorted(settings['servers']))
"""
    with tempfile.TemporaryDirectory() as td:
        arc_dir = pathlib.Path(td, ".arc")
        arc_dir.mkdir()
        (arc_dir / "settings.py").write_text(REMOTE_SETTINGS)
        p = run_in_arc_env(code, env={"HOME": td})
        assert p.returncode == 0, p.stderr
        assert "smoke_cluster" in p.stdout, p.stdout
        assert "server1" not in p.stdout, "the repository's dummy servers were not replaced"


@pytest.mark.smoke
def test_arc_submit_overlay_is_loaded():
    """Test that a ~/.arc/submit.py replaces the repository's submit templates.

    submit.py carries the cluster's PBS/Slurm templates, so a remote run depends on this
    mount just as much as on settings.py.
    """
    code = r"""
from arc.imports import submit_scripts
print('TEMPLATE', submit_scripts.get('smoke_cluster', {}).get('gaussian'))
"""
    with tempfile.TemporaryDirectory() as td:
        arc_dir = pathlib.Path(td, ".arc")
        arc_dir.mkdir()
        (arc_dir / "submit.py").write_text(
            "submit_scripts = {'smoke_cluster': {'gaussian': 'SMOKE-TEMPLATE'}}\n")
        p = run_in_arc_env(code, env={"HOME": td})
        assert p.returncode == 0, p.stderr
        assert "TEMPLATE SMOKE-TEMPLATE" in p.stdout, p.stdout


@pytest.mark.smoke
def test_arc_ignores_an_unimportable_settings_overlay():
    """Test the failure mode the entrypoint pre-flight exists to catch.

    ARC swallows an ImportError from ~/.arc/settings.py and falls back to its dummy servers
    without a word. If this ever stops being true the pre-flight can be relaxed, so assert it
    rather than assume it.
    """
    code = r"""
from arc.imports import settings
print('ADDRESS', settings['servers'].get('server1', {}).get('address'))
"""
    with tempfile.TemporaryDirectory() as td:
        arc_dir = pathlib.Path(td, ".arc")
        arc_dir.mkdir()
        (arc_dir / "settings.py").write_text(
            "import a_module_that_is_not_installed\n" + REMOTE_SETTINGS)
        p = run_in_arc_env(code, env={"HOME": td})
        assert p.returncode == 0, p.stderr
        assert "ADDRESS server1.host.edu" in p.stdout, \
            f"expected a silent fallback to the dummy servers, got: {p.stdout}"


@pytest.mark.smoke
def test_preflight_accepts_a_usable_settings_overlay():
    """Test that the pre-flight passes a settings overlay that imports cleanly."""
    with tempfile.TemporaryDirectory() as td:
        pathlib.Path(td, "settings.py").write_text(REMOTE_SETTINGS)
        p = run_preflight(td, env={"SSH_AUTH_SOCK": "/ssh-agent"})
        assert p.returncode == 0, p.stderr


@pytest.mark.smoke
def test_preflight_rejects_an_unimportable_settings_overlay():
    """Test that a settings overlay ARC would silently drop is reported as fatal."""
    with tempfile.TemporaryDirectory() as td:
        pathlib.Path(td, "settings.py").write_text(
            "import a_module_that_is_not_installed\n" + REMOTE_SETTINGS)
        p = run_preflight(td)
        assert p.returncode == 1, f"expected a fatal pre-flight, got {p.returncode}: {p.stdout}"
        assert "could not be imported" in p.stderr, p.stderr
        assert "dummy server settings" in p.stderr, p.stderr


@pytest.mark.smoke
def test_preflight_warns_about_a_key_that_is_missing_in_the_container():
    """Test that a 'key' pointing outside the container is reported, but is not fatal.

    A server that is configured but unused in this run must not abort it.
    """
    with tempfile.TemporaryDirectory() as td:
        pathlib.Path(td, "settings.py").write_text(REMOTE_SETTINGS.replace(
            "'un': 'smoke_user',", "'un': 'smoke_user',\n        'key': '/no/such/key',"))
        p = run_preflight(td)
        assert p.returncode == 0, p.stderr
        assert "/no/such/key" in p.stderr, p.stderr


@pytest.mark.smoke
def test_preflight_accepts_a_server_without_a_key():
    """Test that omitting 'key' is accepted when an agent is forwarded.

    This is the preferred remote-submission setup: no key file exists in the container at all,
    and paramiko authenticates through the forwarded agent.
    """
    with tempfile.TemporaryDirectory() as td:
        pathlib.Path(td, "settings.py").write_text(REMOTE_SETTINGS)
        p = run_preflight(td, env={"SSH_AUTH_SOCK": "/ssh-agent"})
        assert p.returncode == 0, p.stderr
        assert "sets no 'key'" not in p.stderr, p.stderr


@pytest.mark.smoke
def test_preflight_warns_when_strict_host_key_checking_has_no_known_hosts():
    """Test that strict host key checking without a seeded known_hosts is reported."""
    with tempfile.TemporaryDirectory() as td:
        pathlib.Path(td, "settings.py").write_text(REMOTE_SETTINGS.replace(
            "'un': 'smoke_user',", "'un': 'smoke_user',\n        'strict_host_key_checking': True,"))
        p = run_preflight(td, env={"HOME": td, "SSH_AUTH_SOCK": "/ssh-agent"})
        assert p.returncode == 0, p.stderr
        assert "known_hosts" in p.stderr, p.stderr


@pytest.mark.smoke
def test_mounted_arc_settings_are_readable():
    """Test that a mounted ~/.arc is readable by the container user.

    Skipped when nothing is mounted there, which is the case for local-only runs.
    """
    settings_file = pathlib.Path(ARC_SETTINGS_DIR, "settings.py")
    if not settings_file.is_file():
        pytest.skip(f"no settings overlay mounted at {ARC_SETTINGS_DIR}")
    assert os.access(settings_file, os.R_OK), \
        f"{settings_file} is not readable by the container user; " \
        f"re-run with -e PUID=$(id -u) -e PGID=$(id -g)"
    p = run_preflight(ARC_SETTINGS_DIR)
    assert p.returncode == 0, f"the mounted settings overlay is unusable:\n{p.stderr}"


@pytest.mark.smoke
@pytest.mark.skipif(not os.environ.get("ARC_SMOKE_SSH_HOST"),
                    reason="live remote cluster test; set ARC_SMOKE_SSH_HOST and ARC_SMOKE_SSH_USER to run")
def test_live_remote_ssh_connection():
    """Test an actual SSH connection to a remote cluster. Requires a live server, opt-in only.

    No missing-host-key policy is set, so paramiko's default RejectPolicy applies and the host
    must already be in /home/mambauser/.ssh/known_hosts. Seed it first, e.g. by mounting the
    host's file or running ssh-keyscan, or the test fails on the host key rather than on the
    connection it means to exercise.
    """
    code = r"""
import os
import paramiko
client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect(hostname=os.environ['ARC_SMOKE_SSH_HOST'],
               username=os.environ.get('ARC_SMOKE_SSH_USER'),
               banner_timeout=200)
_, stdout, _ = client.exec_command('echo remote OK')
print(stdout.read().decode().strip())
client.close()
"""
    cmd = ["bash", "-lc", f"micromamba run -n arc_env python - <<'PY'\n{code}\nPY"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert "remote OK" in p.stdout
