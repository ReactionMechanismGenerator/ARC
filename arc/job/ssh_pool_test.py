#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests for the arc.job.ssh_pool module.

These exercise the pool directly, without routing through a JobAdapter.
The adapter-driven integration tests (one client shared across many jobs,
status polling and downloads reusing the pooled client) live alongside the
adapter in arc/job/adapter_test.py.
"""

import unittest
from unittest.mock import patch

from arc.job.ssh_pool import (
    KEEPALIVE_INTERVAL,
    SSHConnectionPool,
    borrow_ssh_client,
    get_default_pool,
    reset_default_pool,
    set_default_pool,
    set_keepalive,
    _default_factory,
)


class _FakeTransport(object):
    """Stands in for a paramiko Transport."""

    def __init__(self, active=True):
        self._active = active
        self.keepalive_interval = None

    def is_active(self):
        return self._active

    def set_keepalive(self, interval):
        """Record the keepalive interval paramiko was asked for."""
        self.keepalive_interval = interval


class _FakeParamikoSSH(object):
    """Stands in for paramiko.SSHClient as held by SSHClient._ssh."""

    def __init__(self, active=True):
        self._transport = _FakeTransport(active=active)

    def get_transport(self):
        return self._transport


class _SSHClientStub(object):
    """Minimal stand-in for arc.job.ssh.SSHClient. Records connect() and close() calls."""

    def __init__(self, server, active=True):
        self.server = server
        self._ssh = _FakeParamikoSSH(active=active)
        self._closed = False
        self.connects = 0

    def connect(self):
        """Count the connection the real client would open."""
        self.connects += 1

    def close(self):
        self._closed = True

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.close()
        return False


class TestSSHConnectionPool(unittest.TestCase):
    """Test SSHConnectionPool.borrow() and close_all()."""

    def setUp(self):
        self.opened = []

        def factory(server):
            client = _SSHClientStub(server)
            self.opened.append(client)
            return client

        self.pool = SSHConnectionPool(factory=factory)

    def test_borrow_opens_once_per_server(self):
        """Repeated borrows of one server reuse a single client."""
        for _ in range(10):
            with self.pool.borrow('server2') as client:
                self.assertIsInstance(client, _SSHClientStub)
        self.assertEqual(self.pool.opens, 1)
        self.assertEqual(self.pool.borrows, 10)
        self.assertEqual(len(self.opened), 1)

    def test_borrow_yields_the_same_object(self):
        """The identical client instance comes back on each borrow."""
        with self.pool.borrow('server2') as first:
            pass
        with self.pool.borrow('server2') as second:
            pass
        self.assertIs(first, second)

    def test_distinct_servers_get_distinct_clients(self):
        with self.pool.borrow('server2'):
            pass
        with self.pool.borrow('server3'):
            pass
        self.assertEqual(self.pool.opens, 2)
        self.assertEqual(sorted(self.pool._clients.keys()), ['server2', 'server3'])

    def test_borrow_does_not_close_on_exit(self):
        """The pool retains ownership; leaving the context must not close."""
        with self.pool.borrow('server2') as client:
            pass
        self.assertFalse(client._closed)

    def test_dead_client_is_reaped_and_reopened(self):
        """An inactive Transport causes the client to be closed and replaced."""
        with self.pool.borrow('server2') as first:
            pass
        first._ssh = None  # simulate a dropped connection
        with self.pool.borrow('server2') as second:
            pass
        self.assertTrue(first._closed, 'the dead client should be closed before reopening')
        self.assertIsNot(first, second)
        self.assertEqual(self.pool.opens, 2)

    def test_inactive_transport_is_treated_as_dead(self):
        """A Transport reporting is_active() False is not handed out."""
        with self.pool.borrow('server2') as first:
            pass
        first._ssh = _FakeParamikoSSH(active=False)
        with self.pool.borrow('server2') as second:
            pass
        self.assertIsNot(first, second)
        self.assertEqual(self.pool.opens, 2)

    def test_exception_in_body_leaves_pool_usable(self):
        """A raising with-body must propagate and not corrupt pool state."""
        def borrow_and_raise():
            with self.pool.borrow('server2'):
                raise ValueError('boom')

        self.assertRaises(ValueError, borrow_and_raise)
        with self.pool.borrow('server2') as client:
            self.assertIsInstance(client, _SSHClientStub)
        self.assertEqual(self.pool.opens, 1, 'the client should have been reused, not reopened')

    def test_close_all_closes_and_empties(self):
        with self.pool.borrow('server2'):
            pass
        with self.pool.borrow('server3'):
            pass
        clients = list(self.pool._clients.values())
        self.pool.close_all()
        self.assertEqual(self.pool._clients, {})
        for client in clients:
            self.assertTrue(client._closed)

    def test_close_all_is_idempotent(self):
        with self.pool.borrow('server2'):
            pass
        self.pool.close_all()
        self.pool.close_all()
        self.assertEqual(self.pool._clients, {})


class TestSSHPoolDefaultLifecycle(unittest.TestCase):
    """The module-level default pool is lazy and resettable."""

    def setUp(self):
        reset_default_pool()

    def tearDown(self):
        reset_default_pool()

    def test_get_default_pool_is_idempotent(self):
        p1 = get_default_pool()
        p2 = get_default_pool()
        self.assertIs(p1, p2)

    def test_reset_default_pool_drops_the_instance(self):
        p1 = get_default_pool()
        reset_default_pool()
        p2 = get_default_pool()
        self.assertIsNot(p1, p2)

    def test_set_default_pool_replaces_instance(self):
        replacement = SSHConnectionPool(factory=_SSHClientStub)
        set_default_pool(replacement)
        self.assertIs(get_default_pool(), replacement)

    def test_reset_default_pool_closes_pooled_clients(self):
        """Resetting must release connections, not just drop the reference."""
        pool = SSHConnectionPool(factory=_SSHClientStub)
        set_default_pool(pool)
        with pool.borrow('server2') as client:
            pass
        reset_default_pool()
        self.assertTrue(client._closed)


class TestSetKeepalive(unittest.TestCase):
    """A pooled connection sits idle between polls, so it must send keepalives."""

    def test_the_transport_is_asked_to_send_keepalives(self):
        """Without this, an idle pooled connection is dropped by the server or a firewall."""
        client = _SSHClientStub('server2')
        self.assertTrue(set_keepalive(client))
        self.assertEqual(client._ssh.get_transport().keepalive_interval, KEEPALIVE_INTERVAL)

    def test_the_interval_is_the_one_asked_for(self):
        client = _SSHClientStub('server2')
        set_keepalive(client, 7)
        self.assertEqual(client._ssh.get_transport().keepalive_interval, 7)

    def test_a_client_without_a_transport_does_not_raise(self):
        """A client that never connected has nothing to keep alive, and that is not an error."""
        client = _SSHClientStub('server2')
        client._ssh = None
        self.assertFalse(set_keepalive(client))

    def test_the_default_factory_sets_a_keepalive(self):
        """The factory is where a pooled connection is born, so it is where this must happen."""
        client = _SSHClientStub('server2')
        with patch('arc.job.ssh_pool.SSHClient', return_value=client):
            built = _default_factory('server2')
        self.assertIs(built, client)
        self.assertEqual(client._ssh.get_transport().keepalive_interval, KEEPALIVE_INTERVAL)


class TestBorrowSSHClientFallback(unittest.TestCase):
    """The shared borrow path every SSH caller in ARC goes through."""

    def setUp(self):
        reset_default_pool()
        self.addCleanup(reset_default_pool)
        self.one_shot = list()

        def _one_shot(server):
            client = _SSHClientStub(server)
            self.one_shot.append(client)
            return client
        patcher = patch('arc.job.ssh_pool.SSHClient', side_effect=_one_shot)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_a_pooled_client_is_yielded(self):
        """The whole point: the caller gets the pool's client, not a new connection."""
        set_default_pool(SSHConnectionPool(factory=_SSHClientStub))
        with borrow_ssh_client('server2') as client:
            self.assertIsInstance(client, _SSHClientStub)
            self.assertEqual(client.server, 'server2')
        self.assertEqual(self.one_shot, list())

    def test_the_pooled_client_is_reused_across_borrows(self):
        set_default_pool(SSHConnectionPool(factory=_SSHClientStub))
        with borrow_ssh_client('server2') as first:
            pass
        with borrow_ssh_client('server2') as second:
            pass
        self.assertIs(first, second)
        self.assertEqual(get_default_pool().opens, 1)

    def test_a_pooled_client_is_not_closed_on_exit(self):
        set_default_pool(SSHConnectionPool(factory=_SSHClientStub))
        with borrow_ssh_client('server2') as client:
            pass
        self.assertFalse(client._closed)

    def test_a_factory_failure_falls_back_to_a_one_shot_client(self):
        """A caller must not lose its connection because the pool could not open one."""
        def _factory(server):
            raise RuntimeError(f'no route to {server}')
        set_default_pool(SSHConnectionPool(factory=_factory))
        with borrow_ssh_client('server2') as client:
            self.assertIs(client, self.one_shot[0])
        self.assertEqual(len(self.one_shot), 1)

    def test_a_pool_that_cannot_lease_falls_back_too(self):
        """Any failure to lease is a failure to lease, whatever the pool object is."""
        class _PoolWithoutBorrow:
            """A process-global pool object that cannot lease a client."""

            def close_all(self):
                """Tear down nothing, since nothing was ever leased."""
        set_default_pool(_PoolWithoutBorrow())
        with borrow_ssh_client('server2') as client:
            self.assertIs(client, self.one_shot[0])

    def test_a_one_shot_client_is_closed_on_exit(self):
        """The fallback owns what it opened, so it must not leak it."""
        def _factory(server):
            raise RuntimeError(f'no route to {server}')
        set_default_pool(SSHConnectionPool(factory=_factory))
        with borrow_ssh_client('server2') as client:
            pass
        self.assertTrue(client._closed)

    def test_an_error_raised_by_the_caller_is_not_treated_as_a_lease_failure(self):
        """The fallback covers leasing only: the caller's own failure must reach the caller."""
        set_default_pool(SSHConnectionPool(factory=_SSHClientStub))

        def borrow_and_raise():
            with borrow_ssh_client('server2'):
                raise ValueError('the job, not the connection, went wrong')

        self.assertRaises(ValueError, borrow_and_raise)
        self.assertEqual(self.one_shot, list())


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
