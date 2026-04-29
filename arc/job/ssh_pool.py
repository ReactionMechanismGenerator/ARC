"""Persistent per-server SSHClient pool for the lifetime of an ARC run.

Without this, each remote-queue job opens its own TCP+auth handshake
for upload, then another for qsub. Option 1 (in :mod:`arc.job.adapter`)
collapsed those two into one (per-job sharing). This module is Option
2: extend the share across ALL jobs run during this Python process,
so 100 TS guess opts end up sharing one paramiko Transport instead of
opening 100 of them. The closest equivalent to OpenSSH's
``ControlMaster``, applied at the library level for paramiko.

Concurrency: ARC's scheduler is single-threaded (verified — no
``Thread`` / ``asyncio`` / ``concurrent.futures`` imports across
``scheduler.py`` / ``main.py`` / ``adapter.py``), so the pool does no
locking. A future async/parallel scheduler would need per-server
locks; flagged in :meth:`SSHConnectionPool.borrow`.

Lifecycle: the default process-global pool is opened lazily on first
borrow and closed via :func:`reset_default_pool`. ARC.py's ``main()``
calls that on exit so pooled connections close cleanly even on
ctrl-C / crash; tests call it in ``tearDown`` to start fresh.
"""

from contextlib import contextmanager
from typing import Callable

from arc.common import get_logger
from arc.job.ssh import SSHClient

logger = get_logger()


SSHClientFactory = Callable[[str], SSHClient]


def _default_factory(server: str) -> SSHClient:
    """Open and connect a real SSHClient. Override for tests."""
    client = SSHClient(server)
    client.connect()
    return client


class SSHConnectionPool:
    """Process-lifetime cache of SSHClient instances keyed by server name.

    One client per server, opened lazily on first borrow, kept alive
    until :meth:`close_all` is called (or the process exits). Health
    is re-checked on every operation by the existing
    ``check_connections`` decorator on SSHClient methods, so a stale
    Transport is silently re-established mid-run.
    """

    def __init__(self, factory: SSHClientFactory = _default_factory):
        self._factory = factory
        self._clients: dict[str, SSHClient] = {}
        # Counters expose pool behavior to tests/observability without
        # forcing them to peek at internals or hook the factory.
        self.opens = 0
        self.borrows = 0

    @contextmanager
    def borrow(self, server: str):
        """Lease the pool's SSHClient for ``server``.

        Returns a context manager yielding an :class:`SSHClient`.
        Exiting the context does NOT close the client — the pool
        retains ownership. The borrowed client is transient by
        contract; do not stash it past the ``with`` block.

        Concurrent borrows of the same server are not safe today.
        ARC's scheduler is single-threaded, so this hasn't bitten;
        a parallel scheduler would need a per-server lock around the
        yield (or a small "free clients" stack instead of a single).
        """
        self.borrows += 1
        client = self._clients.get(server)
        if client is None or not _is_alive(client):
            if client is not None:
                _close_quietly(client, f"reaping dead {server} SSHClient before reopen")
            client = self._factory(server)
            self._clients[server] = client
            self.opens += 1
            logger.debug("ssh_pool: opened SSHClient for %s (total opens=%d)", server, self.opens)
        else:
            logger.debug("ssh_pool: reusing SSHClient for %s", server)
        yield client
        # No close on exit — pool keeps the connection.

    def close_all(self) -> None:
        """Close every pooled client. Safe to call multiple times."""
        for server, client in list(self._clients.items()):
            _close_quietly(client, f"closing pooled {server} SSHClient")
        self._clients.clear()


def _is_alive(client: SSHClient) -> bool:
    """Cheap liveness check: does the paramiko Transport report active?

    Doesn't roundtrip to the server — the SSHClient method's own
    ``check_connections`` decorator does that on the next call. This is
    just enough to skip the obvious "connection got reset between
    jobs" case so we don't hand out a known-dead handle.
    """
    underlying = getattr(client, "_ssh", None)
    if underlying is None:
        return False
    transport_getter = getattr(underlying, "get_transport", None)
    if transport_getter is None:
        return False
    transport = transport_getter()
    return bool(transport and transport.is_active())


def _close_quietly(client: SSHClient, context: str) -> None:
    try:
        client.close()
    except Exception:
        # Pool teardown should never propagate a close error; ARC's
        # main path is past the work that needed the connection.
        logger.debug("ssh_pool: close errored %s", context, exc_info=True)


# Process-global default pool. Lazily instantiated. Reset between ARC
# runs (and between tests) via reset_default_pool().
_default_pool: SSHConnectionPool | None = None


def get_default_pool() -> SSHConnectionPool:
    """Return the process-global pool, creating it on first call."""
    global _default_pool
    if _default_pool is None:
        _default_pool = SSHConnectionPool()
    return _default_pool


def set_default_pool(pool: SSHConnectionPool | None) -> None:
    """Replace the process-global pool. Mainly for tests that want to
    inject a stub-factory pool without monkeypatching the module."""
    global _default_pool
    _default_pool = pool


def reset_default_pool() -> None:
    """Close and discard the default pool. Idempotent."""
    global _default_pool
    if _default_pool is not None:
        _default_pool.close_all()
        _default_pool = None


__all__ = [
    "SSHClientFactory",
    "SSHConnectionPool",
    "get_default_pool",
    "reset_default_pool",
    "set_default_pool",
]
