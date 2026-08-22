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
borrow and closed via :func:`reset_default_pool`.
``arc.main.ARC.execute()`` calls that in a ``finally``, so pooled
connections close cleanly on a clean run, on ctrl-C and on a crash
alike, and for an in-process consumer as much as for ``ARC.py``;
tests call it in ``tearDown`` to start fresh.

Entry point: :func:`borrow_ssh_client` is the single borrow path every
SSH caller in ARC goes through, so the scheduler's status poll, the
job adapters, server troubleshooting and the ESS survey all share one
connection per server rather than one each.
"""

from contextlib import ExitStack, contextmanager
from collections.abc import Iterator
from typing import Callable

from arc.common import get_logger
from arc.job.ssh import SSHClient

logger = get_logger()


SSHClientFactory = Callable[[str], SSHClient]

KEEPALIVE_INTERVAL = 30


def _default_factory(server: str) -> SSHClient:
    """Open and connect a real SSHClient, sending keepalives. Override for tests.

    Args:
        server (str): The server name.

    Returns: SSHClient
        A connected client.
    """
    client = SSHClient(server)
    client.connect()
    set_keepalive(client, KEEPALIVE_INTERVAL)
    return client


def set_keepalive(client: SSHClient, interval: int = KEEPALIVE_INTERVAL) -> bool:
    """
    Ask a client's paramiko Transport to send a keepalive every ``interval`` seconds.

    A pooled connection is held for the lifetime of the run and sits idle between polls, which
    is long enough for an SSH daemon's ``ClientAliveInterval`` or an intermediate firewall's
    idle-connection timeout to drop it. Without a keepalive that drop is silent: the socket
    stays half-open, ``Transport.is_active()`` keeps reporting ``True``, and the next borrow
    hands out a handle whose first command hangs until TCP gives up. A periodic global request
    both keeps the session in the middle boxes' tables and lets paramiko mark the transport dead
    when the peer stops answering.

    Args:
        client (SSHClient): The client whose transport should send keepalives.
        interval (int, optional): Seconds between keepalives.

    Returns: bool
        Whether a keepalive was set, ``False`` when the client has no live transport.
    """
    underlying = getattr(client, '_ssh', None)
    transport_getter = getattr(underlying, 'get_transport', None)
    transport = transport_getter() if transport_getter is not None else None
    if transport is None:
        logger.debug('ssh_pool: no transport to set a keepalive on for %s',
                     getattr(client, 'server', client))
        return False
    transport.set_keepalive(interval)
    return True


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


@contextmanager
def borrow_ssh_client(server: str) -> Iterator[SSHClient]:
    """
    Lease an :class:`SSHClient` for ``server`` from the process-global pool.

    This is the single borrow path every SSH caller in ARC goes through, so that all of them
    share one connection per server for the lifetime of the run. Leasing from the pool is tried
    first; if the lease itself fails -- the default pool was replaced by an object that cannot
    lease, or the factory could not connect -- a one-shot client is opened and closed inline
    instead, so a caller never loses its connection because the pool is unavailable.

    The fallback covers leasing only. Once a client has been yielded, an exception raised by the
    body of the ``with`` block propagates to the caller untouched.

    Exiting the context does not close a pooled client, which the pool keeps and hands to the
    next borrower; a one-shot client opened by the fallback is closed on exit.

    Args:
        server (str): The server name, as a key of the ``servers`` settings dictionary.

    Yields: SSHClient
        A connected client for ``server``.
    """
    with ExitStack() as stack:
        try:
            client = stack.enter_context(get_default_pool().borrow(server))
        except Exception as e:
            logger.debug(f'Could not lease an SSHClient for {server} from the pool, opening a '
                         f'one-shot client instead. Got {type(e).__name__}: {e}', exc_info=True)
            client = stack.enter_context(SSHClient(server))
        yield client


__all__ = [
    "KEEPALIVE_INTERVAL",
    "SSHClientFactory",
    "SSHConnectionPool",
    "borrow_ssh_client",
    "get_default_pool",
    "reset_default_pool",
    "set_default_pool",
    "set_keepalive",
]
