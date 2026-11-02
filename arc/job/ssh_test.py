#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.job.ssh module
"""

import base64
import hashlib
import os
import shlex
import shutil
import subprocess
import tempfile
import unittest
import warnings
from unittest.mock import MagicMock, patch

import paramiko

import arc.job.ssh as ssh
from arc.exceptions import ServerError


class FakeHostKey(object):
    """A stand-in for a paramiko host key, carrying only what the host key policies use."""

    def __init__(self, blob: bytes = b'fake-host-key-blob'):
        self.blob = blob

    def asbytes(self) -> bytes:
        """Return the key blob."""
        return self.blob

    def get_name(self) -> str:
        """Return the key type."""
        return 'ssh-ed25519'

    def get_base64(self) -> str:
        """Return the base64-encoded key blob."""
        return base64.b64encode(self.blob).decode()

    def get_fingerprint(self) -> bytes:
        """Return the MD5 digest of the key blob, which is what paramiko fingerprints with."""
        return hashlib.md5(self.blob).digest()


class TestSSH(unittest.TestCase):
    """
    Contains unit tests for the SSH module
    """

    def test_check_job_status_in_stdout(self):
        """Test checking the job status in stdout"""
        # OGE
        stdout_1 = """job-ID  prior   name       user         state submit/start at     queue                          slots ja-task-ID 
-----------------------------------------------------------------------------------------------------------------
 582682 0.45451 a9654      alongd       e     04/17/2019 16:22:14 long5@node93.cluster              48
 588334 0.45451 pf1005a    alongd       r     05/07/2019 16:24:31 long3@node67.cluster              48
 588345 0.45451 a14121     alongd       r     05/08/2019 02:11:42 long3@node69.cluster              48    """
        status1 = ssh.check_job_status_in_stdout(job_id=588345, stdout=stdout_1, server='server1')
        self.assertEqual(status1, 'running')
        status2 = ssh.check_job_status_in_stdout(job_id=582682, stdout=stdout_1, server='server1')
        self.assertEqual(status2, 'errored')
        status3 = ssh.check_job_status_in_stdout(job_id=582600, stdout=stdout_1, server='server1')
        self.assertEqual(status3, 'done')

        # HTCondor
        stdout_2 = ['5231.0 R 10 7885 a20596 130',
                    '5232.0 R 10 7885 a20597 130',
                    '5233.0 R 10 7885 a20598 130',
                    '5241.0 P 10 7885 a20616 0']
        status1 = ssh.check_job_status_in_stdout(job_id=5231, stdout=stdout_2, server='local')
        self.assertEqual(status1, 'running')
        status1 = ssh.check_job_status_in_stdout(job_id=5241, stdout=stdout_2, server='local')
        self.assertEqual(status1, 'running')
        status1 = ssh.check_job_status_in_stdout(job_id=4000, stdout=stdout_2, server='local')
        self.assertEqual(status1, 'done')



class TestSSHConnectHardening(unittest.TestCase):
    """Host-key policy selection and retry scoping in SSHClient._connect()."""

    SERVER = {'address': 'host.example.edu', 'un': 'user', 'key': '/dev/null',
              'cluster_soft': 'PBS'}

    def _connect_with(self, server_cfg, connect_side_effect=None):
        """Run _connect() against a fake paramiko, returning (fake_client, exception).

        Only the exception types these tests assert on are caught, so any other exception
        escaping _connect() fails the test rather than being reported as an expected one.
        """
        fake = MagicMock()
        if connect_side_effect is not None:
            fake.connect.side_effect = connect_side_effect
        raised = None
        with patch.object(ssh, 'servers', {'srv': server_cfg}), \
                patch.object(ssh.paramiko, 'SSHClient', return_value=fake):
            client = ssh.SSHClient('srv')
            try:
                client._connect()
            except (paramiko.SSHException, OSError, KeyboardInterrupt) as exc:
                raised = exc
        return fake, raised

    def _policy_used(self, fake):
        return type(fake.set_missing_host_key_policy.call_args[0][0])

    def test_default_policy_warns_rather_than_adding_silently(self):
        """Unknown host keys must not be added silently by default."""
        fake, _ = self._connect_with(dict(self.SERVER))
        self.assertIs(self._policy_used(fake), ssh.LogAndAcceptHostKeyPolicy)

    def test_strict_host_key_checking_rejects_unknown_hosts(self):
        """strict_host_key_checking opts into refusing unknown hosts."""
        fake, _ = self._connect_with(dict(self.SERVER, strict_host_key_checking=True))
        self.assertIs(self._policy_used(fake), ssh.RejectUnknownHostKeyPolicy)

    def test_configured_key_is_offered_as_the_identity(self):
        """The key must reach connect() as key_filename, not only host keys."""
        fake, _ = self._connect_with(dict(self.SERVER))
        self.assertEqual(fake.connect.call_args.kwargs['key_filename'], '/dev/null')

    def test_transport_error_is_retried_once(self):
        """A banner/reset SSHException retries, matching the documented flake."""
        fake, raised = self._connect_with(
            dict(self.SERVER),
            connect_side_effect=[paramiko.SSHException('Error reading SSH protocol banner'), None])
        self.assertIsNone(raised)
        self.assertEqual(fake.connect.call_count, 2)

    def test_keyboard_interrupt_is_not_swallowed(self):
        """The retry must not catch KeyboardInterrupt, as a bare except did."""
        fake, raised = self._connect_with(dict(self.SERVER),
                                          connect_side_effect=KeyboardInterrupt())
        self.assertIsInstance(raised, KeyboardInterrupt)
        self.assertEqual(fake.connect.call_count, 1, 'must not retry after an interrupt')

    def test_second_transport_failure_propagates(self):
        """If the retry also fails, the error surfaces rather than being hidden."""
        fake, raised = self._connect_with(
            dict(self.SERVER),
            connect_side_effect=[paramiko.SSHException('first'), paramiko.SSHException('second')])
        self.assertIsInstance(raised, paramiko.SSHException)
        self.assertEqual(fake.connect.call_count, 2)


class TestSSHOptionalKey(unittest.TestCase):
    """``servers[...]['key']`` is optional, for ssh-agent and default-key-path authentication."""

    SERVER = {'address': 'host.example.edu', 'un': 'user', 'cluster_soft': 'PBS'}

    def _client(self, server_cfg):
        """Instantiate an SSHClient against ``server_cfg`` without touching the network."""
        with patch.object(ssh, 'servers', {'srv': server_cfg}):
            return ssh.SSHClient('srv')

    def _key_filename_used(self, server_cfg):
        """Return the ``key_filename`` that _connect() hands to paramiko."""
        fake = MagicMock()
        with patch.object(ssh, 'servers', {'srv': server_cfg}), \
                patch.object(ssh.paramiko, 'SSHClient', return_value=fake):
            ssh.SSHClient('srv')._connect()
        return fake.connect.call_args.kwargs['key_filename']

    def test_configured_key_is_stored(self):
        """A configured key is still read off the server settings."""
        self.assertEqual(self._client(dict(self.SERVER, key='/dev/null')).key, '/dev/null')

    def test_configured_key_is_forwarded_as_key_filename(self):
        """A configured key is offered to paramiko as the connection identity."""
        self.assertEqual(self._key_filename_used(dict(self.SERVER, key='/dev/null')), '/dev/null')

    def test_missing_key_does_not_raise(self):
        """A server entry without a key must not raise a KeyError on instantiation."""
        self.assertIsNone(self._client(dict(self.SERVER)).key)

    def test_missing_key_is_not_offered_as_an_identity(self):
        """Without a key, paramiko must be free to fall back to the agent and default keys."""
        self.assertIsNone(self._key_filename_used(dict(self.SERVER)))

    def test_empty_key_is_treated_as_unset(self):
        """An empty key path is not a usable identity, and must not be handed to paramiko."""
        self.assertIsNone(self._client(dict(self.SERVER, key='')).key)
        self.assertIsNone(self._key_filename_used(dict(self.SERVER, key='')))


class TestMissingKeyFileIsPermanent(unittest.TestCase):
    """A ``key`` path that does not exist cannot be resolved by retrying it for 24 hours."""

    SERVER = {'address': 'host.example.edu', 'un': 'user', 'key': '/no/such/key',
              'cluster_soft': 'PBS'}

    def _client(self, **overrides):
        with patch.object(ssh, 'servers', {'srv': dict(self.SERVER, **overrides)}):
            return ssh.SSHClient('srv', connection_attempts=5)

    def _connect_with(self, error, **overrides):
        """Call connect() with _connect() failing on ``error``, returning (raised, calls, sleeps)."""
        client = self._client(**overrides)
        inner = MagicMock(side_effect=error)
        raised = None
        with patch.object(ssh, 'servers', {'srv': dict(self.SERVER, **overrides)}), \
                patch.object(client, '_connect', inner), \
                patch.object(ssh.time, 'sleep') as sleep:
            try:
                client.connect()
            except ServerError as exc:
                raised = exc
        return raised, inner.call_count, sleep.call_count

    def test_a_missing_key_file_is_not_retried(self):
        """paramiko reads the identity outside the SSHException it guards that read with."""
        error = FileNotFoundError(2, 'No such file or directory', '/no/such/key')
        raised, calls, sleeps = self._connect_with(error)
        self.assertIsInstance(raised, ServerError)
        self.assertEqual(calls, 1)
        self.assertEqual(sleeps, 0)

    def test_the_error_names_the_key_and_the_way_out(self):
        """A 24 hour stall used to be the only report that the path was wrong."""
        error = FileNotFoundError(2, 'No such file or directory', '/no/such/key')
        message = str(self._connect_with(error)[0])
        self.assertIn('/no/such/key', message)
        self.assertIn('ssh-agent', message)

    def test_a_network_failure_is_still_retried(self):
        """A refused or reset connection is an OSError too, and is exactly what the retry is for."""
        raised, calls, sleeps = self._connect_with(ConnectionRefusedError(111, 'Connection refused'))
        self.assertIsInstance(raised, ServerError)
        self.assertEqual(calls, 5)
        self.assertEqual(sleeps, 4)

    def test_a_missing_file_that_is_not_the_key_is_still_retried(self):
        """Only the configured key path is permanent; another absent file is not classified here."""
        error = FileNotFoundError(2, 'No such file or directory', '/some/other/file')
        raised, calls, sleeps = self._connect_with(error)
        self.assertIsInstance(raised, ServerError)
        self.assertEqual(calls, 5)
        self.assertEqual(sleeps, 4)

    def test_a_missing_file_without_a_configured_key_is_still_retried(self):
        """With no key configured there is no key file to be missing."""
        error = FileNotFoundError(2, 'No such file or directory', '/no/such/key')
        raised, calls, sleeps = self._connect_with(error, key=None)
        self.assertIsInstance(raised, ServerError)
        self.assertEqual(calls, 5)
        self.assertEqual(sleeps, 4)

    def test_the_inner_retry_does_not_reattempt_a_missing_key(self):
        """_connect() retries a transport failure once, which a missing key file is not."""
        fake = MagicMock()
        fake.connect.side_effect = FileNotFoundError(2, 'No such file or directory', '/no/such/key')
        with patch.object(ssh, 'servers', {'srv': dict(self.SERVER)}), \
                patch.object(ssh.paramiko, 'SSHClient', return_value=fake):
            self.assertRaises(FileNotFoundError, ssh.SSHClient('srv')._connect)
        self.assertEqual(fake.connect.call_count, 1)


class TestKeepaliveSurvivesReconnects(unittest.TestCase):
    """A keepalive lives on a paramiko Transport, and a reconnect replaces the transport."""

    SERVER = {'address': 'host.example.edu', 'un': 'user', 'key': '/dev/null',
              'cluster_soft': 'PBS'}

    def _client(self):
        with patch.object(ssh, 'servers', {'srv': dict(self.SERVER)}):
            return ssh.SSHClient('srv')

    def _connect_onto(self, client, transport):
        """Connect ``client`` so that its paramiko client reports ``transport``."""
        paramiko_client = MagicMock()
        paramiko_client.get_transport.return_value = transport
        with patch.object(ssh, 'servers', {'srv': dict(self.SERVER)}), \
                patch.object(client, '_connect', MagicMock(return_value=(MagicMock(), paramiko_client))):
            client.connect()

    def test_a_client_never_asked_for_a_keepalive_does_not_set_one(self):
        """A one-shot client is not held open long enough to be dropped while idle."""
        transport = MagicMock()
        client = self._client()
        self._connect_onto(client, transport)
        transport.set_keepalive.assert_not_called()

    def test_the_keepalive_is_reapplied_to_a_new_transport(self):
        """check_connections reconnects a pooled client in place, replacing its transport."""
        client = self._client()
        client._keepalive_interval = 30
        second = MagicMock()
        self._connect_onto(client, second)
        second.set_keepalive.assert_called_once_with(30)

    def test_a_reconnect_without_a_transport_does_not_raise(self):
        """A client whose paramiko client has no transport has nothing to keep alive."""
        client = self._client()
        client._keepalive_interval = 30
        self._connect_onto(client, None)
        self.assertFalse(client._apply_keepalive())


class TestKnownHostsCheck(unittest.TestCase):
    """The startup report of servers that have no host key on this machine."""

    @classmethod
    def setUpClass(cls):
        """Generate one host key, reused by every test as the key of a 'known' host."""
        key = paramiko.ECDSAKey.generate()
        cls.key_type, cls.key_b64 = key.get_name(), key.get_base64()

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp_dir, True)
        self.known_hosts = os.path.join(self.tmp_dir, 'known_hosts')

    def _write_known_hosts(self, *host_patterns):
        """Write a known_hosts file listing ``host_patterns``, and return its path."""
        with open(self.known_hosts, 'w') as f:
            for host_pattern in host_patterns:
                f.write(f'{host_pattern} {self.key_type} {self.key_b64}\n')
        return self.known_hosts

    def test_a_known_host_is_not_reported(self):
        """A server whose address is in known_hosts must not be reported."""
        self._write_known_hosts('login.cluster.edu')
        missing = ssh.get_servers_missing_host_keys(
            server_dict={'srv': {'address': 'login.cluster.edu', 'un': 'me', 'cluster_soft': 'PBS'}},
            known_hosts_path=self.known_hosts)
        self.assertEqual(missing, {})

    def test_an_unknown_host_is_reported_with_its_address(self):
        """A server absent from known_hosts is reported, keyed by its server name."""
        self._write_known_hosts('other.cluster.edu')
        missing = ssh.get_servers_missing_host_keys(
            server_dict={'srv': {'address': 'login.cluster.edu', 'un': 'me', 'cluster_soft': 'PBS'}},
            known_hosts_path=self.known_hosts)
        self.assertEqual(missing, {'srv': 'login.cluster.edu'})

    def test_a_hashed_entry_is_recognized(self):
        """``ssh-keyscan -H`` writes hashed host names; paramiko's lookup must resolve them."""
        self._write_known_hosts(paramiko.HostKeys.hash_host('login.cluster.edu'))
        missing = ssh.get_servers_missing_host_keys(
            server_dict={'srv': {'address': 'login.cluster.edu', 'un': 'me', 'cluster_soft': 'PBS'}},
            known_hosts_path=self.known_hosts)
        self.assertEqual(missing, {})

    def test_an_absent_known_hosts_file_reports_every_server(self):
        """No known_hosts file at all must report the servers, not raise."""
        missing = ssh.get_servers_missing_host_keys(
            server_dict={'srv': {'address': 'login.cluster.edu', 'un': 'me', 'cluster_soft': 'PBS'}},
            known_hosts_path=os.path.join(self.tmp_dir, 'does_not_exist'))
        self.assertEqual(missing, {'srv': 'login.cluster.edu'})

    def test_local_and_addressless_servers_are_skipped(self):
        """A 'local' server and a server without an address are not reachable over SSH."""
        missing = ssh.get_servers_missing_host_keys(
            server_dict={'local': {'cluster_soft': 'PBS', 'un': 'me'},
                         'no_address': {'cluster_soft': 'PBS', 'un': 'me'}},
            known_hosts_path=self.known_hosts)
        self.assertEqual(missing, {})

    def test_placeholder_servers_are_skipped(self):
        """The repository's shipped placeholders must not warn on every default install."""
        missing = ssh.get_servers_missing_host_keys(
            server_dict={'server1': {'address': 'server1.host.edu', 'un': '<username>',
                                     'cluster_soft': 'OGE'},
                         'named_user': {'address': 'real.cluster.edu', 'un': '<username>',
                                        'cluster_soft': 'PBS'}},
            known_hosts_path=self.known_hosts)
        self.assertEqual(missing, {})

    def test_check_warns_naming_the_server_and_the_fix(self):
        """The warning must name the server, the address and the ssh-keyscan command."""
        server_dict = {'srv': {'address': 'login.cluster.edu', 'un': 'me', 'cluster_soft': 'PBS'}}
        with self.assertLogs(ssh.logger, level='WARNING') as captured:
            missing = ssh.check_servers_known_hosts(server_dict=server_dict,
                                                    known_hosts_path=self.known_hosts)
        self.assertEqual(missing, {'srv': 'login.cluster.edu'})
        logged = '\n'.join(captured.output)
        self.assertIn('srv', logged)
        self.assertIn('login.cluster.edu', logged)
        self.assertIn(f'ssh-keyscan -H login.cluster.edu >> {self.known_hosts}', logged)

    def test_check_reports_refusal_for_a_strict_server(self):
        """With strict_host_key_checking the consequence is a refusal, not a warning."""
        server_dict = {'srv': {'address': 'login.cluster.edu', 'un': 'me', 'cluster_soft': 'PBS',
                               'strict_host_key_checking': True}}
        with self.assertLogs(ssh.logger, level='WARNING') as captured:
            ssh.check_servers_known_hosts(server_dict=server_dict,
                                          known_hosts_path=self.known_hosts)
        self.assertIn('will be refused', '\n'.join(captured.output))

    def test_check_is_silent_when_every_host_is_known(self):
        """No warning may be emitted when nothing is missing."""
        self._write_known_hosts('login.cluster.edu')
        server_dict = {'srv': {'address': 'login.cluster.edu', 'un': 'me', 'cluster_soft': 'PBS'}}
        with patch.object(ssh.logger, 'warning') as warning:
            missing = ssh.check_servers_known_hosts(server_dict=server_dict,
                                                    known_hosts_path=self.known_hosts)
        self.assertEqual(missing, {})
        warning.assert_not_called()


class TestTransfersCheckTheConnection(unittest.TestCase):
    """A pooled client is long-lived, so a transfer must re-establish a dropped connection."""

    SERVER = {'address': 'host.example.edu', 'un': 'user', 'key': '/dev/null',
              'cluster_soft': 'PBS'}

    def setUp(self):
        """Build an unconnected client whose connect() only records that it was called."""
        self.tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp_dir, ignore_errors=True)
        with patch.object(ssh, 'servers', {'srv': dict(self.SERVER)}):
            self.client = ssh.SSHClient('srv')

        def _connect():
            self.client._sftp = MagicMock()
            self.client._ssh = MagicMock()
        self.connect = MagicMock(side_effect=_connect)
        patcher = patch.object(self.client, 'connect', self.connect)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_upload_file_connects_an_unconnected_client(self):
        """Without this, the SFTP handle is None and the upload raises AttributeError."""
        with patch.object(self.client, '_check_dir_exists', return_value=True):
            self.client.upload_file(remote_file_path='/remote/in.gjf', file_string='#p opt\n')
        self.connect.assert_called_once()
        self.client._sftp.open.assert_called_once()

    def test_download_file_connects_an_unconnected_client(self):
        with patch.object(self.client, '_check_file_exists', return_value=True):
            self.client.download_file(remote_file_path='/remote/out.txt',
                                      local_file_path=os.path.join(self.tmp_dir, 'out.txt'))
        self.connect.assert_called_once()
        self.client._sftp.get.assert_called_once()

    def test_upload_file_reconnects_a_dead_connection(self):
        """The case the pool makes routine: the transport died between two jobs."""
        self.client._sftp, self.client._ssh = MagicMock(), MagicMock()
        self.client._ssh.exec_command.side_effect = OSError('Socket is closed')
        with patch.object(self.client, '_check_dir_exists', return_value=True):
            self.client.upload_file(remote_file_path='/remote/in.gjf', file_string='#p opt\n')
        self.connect.assert_called_once()

    def test_download_file_reconnects_a_dead_connection(self):
        self.client._sftp, self.client._ssh = MagicMock(), MagicMock()
        self.client._ssh.exec_command.side_effect = OSError('Socket is closed')
        with patch.object(self.client, '_check_file_exists', return_value=True):
            self.client.download_file(remote_file_path='/remote/out.txt',
                                      local_file_path=os.path.join(self.tmp_dir, 'out.txt'))
        self.connect.assert_called_once()

    def test_an_unconnected_client_does_not_unpack_the_connect_result(self):
        """connect() assigns the handles and returns None, so unpacking it raised TypeError."""
        self.client.read_remote_file(remote_file_path='/remote/out.txt')
        self.connect.assert_called_once()
        self.client._sftp.open.assert_called_once()


class TestConflictingHostKeys(unittest.TestCase):
    """known_hosts entries that contradict each other are reported without connecting."""

    @classmethod
    def setUpClass(cls):
        """Generate two distinct keys of one type, which is what a contradiction is made of."""
        cls.key_1 = paramiko.ECDSAKey.generate()
        cls.key_2 = paramiko.ECDSAKey.generate()
        cls.rsa_key = paramiko.RSAKey.generate(2048)

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp_dir, True)
        self.known_hosts = os.path.join(self.tmp_dir, 'known_hosts')

    SERVER = {'address': 'login.cluster.edu', 'un': 'me', 'cluster_soft': 'PBS'}

    def _write(self, *entries):
        """Write ``(host pattern, key)`` pairs to known_hosts, and return its path."""
        with open(self.known_hosts, 'w') as f:
            for host_pattern, key in entries:
                f.write(f'{host_pattern} {key.get_name()} {key.get_base64()}\n')
        return self.known_hosts

    def _conflicting(self, server_dict=None):
        """Return the conflict report for ``server_dict``, defaulting to one ordinary server."""
        return ssh.get_servers_with_conflicting_host_keys(
            server_dict=server_dict if server_dict is not None else {'srv': dict(self.SERVER)},
            known_hosts_path=self.known_hosts)

    def test_two_keys_of_one_type_for_one_host_are_reported(self):
        """This is the shape of a stale entry, and equally of one placed there to shadow."""
        self._write(('login.cluster.edu', self.key_1), ('login.cluster.edu', self.key_2))
        self.assertEqual(self._conflicting(), {'srv': [self.key_1.get_name()]})

    def test_one_key_per_type_is_not_a_conflict(self):
        """A host legitimately has one key of each type, which must not be reported."""
        self._write(('login.cluster.edu', self.key_1), ('login.cluster.edu', self.rsa_key))
        self.assertEqual(self._conflicting(), {})

    def test_a_single_entry_is_not_a_conflict(self):
        self._write(('login.cluster.edu', self.key_1))
        self.assertEqual(self._conflicting(), {})

    def test_a_hashed_entry_contradicting_a_plain_one_is_reported(self):
        """ssh-keyscan -H writes hashed names, so a stale pair may not look like a pair."""
        self._write((paramiko.HostKeys.hash_host('login.cluster.edu'), self.key_1),
                    ('login.cluster.edu', self.key_2))
        self.assertEqual(self._conflicting(), {'srv': [self.key_1.get_name()]})

    def test_an_unknown_host_is_not_reported_as_conflicting(self):
        """Absence is the sibling check's business, not this one's."""
        self._write(('other.cluster.edu', self.key_1))
        self.assertEqual(self._conflicting(), {})

    def test_an_absent_known_hosts_file_reports_nothing(self):
        self.assertEqual(
            ssh.get_servers_with_conflicting_host_keys(
                server_dict={'srv': dict(self.SERVER)},
                known_hosts_path=os.path.join(self.tmp_dir, 'does_not_exist')),
            {})

    def test_local_and_placeholder_servers_are_skipped(self):
        """The same servers the absence check skips, for the same reasons."""
        self._write(('server1.host.edu', self.key_1), ('server1.host.edu', self.key_2))
        self.assertEqual(
            self._conflicting({'local': {'cluster_soft': 'PBS', 'un': 'me'},
                               'server1': {'address': 'server1.host.edu', 'un': '<username>',
                                           'cluster_soft': 'OGE'}}),
            {})

    def test_the_check_reports_a_conflict_at_the_error_level(self):
        """A file that disagrees with itself about a server's identity is not a warning."""
        self._write(('login.cluster.edu', self.key_1), ('login.cluster.edu', self.key_2))
        with self.assertLogs(ssh.logger, level='ERROR') as captured:
            ssh.check_servers_known_hosts(server_dict={'srv': dict(self.SERVER)},
                                          known_hosts_path=self.known_hosts)
        logged = '\n'.join(captured.output)
        self.assertIn('srv', logged)
        self.assertIn('login.cluster.edu', logged)
        self.assertIn('ssh-keygen -R login.cluster.edu', logged)

    def test_the_check_is_silent_when_the_single_key_is_known(self):
        """One key, no absence and no contradiction, must produce no report at all."""
        self._write(('login.cluster.edu', self.key_1))
        with patch.object(ssh.logger, 'warning') as warning, \
                patch.object(ssh.logger, 'error') as error:
            ssh.check_servers_known_hosts(server_dict={'srv': dict(self.SERVER)},
                                          known_hosts_path=self.known_hosts)
        warning.assert_not_called()
        error.assert_not_called()


class TestHostKeyMismatch(unittest.TestCase):
    """A stored key contradicted by the server is the security-relevant case."""

    SERVER = {'address': 'host.example.edu', 'un': 'user', 'key': '/dev/null',
              'cluster_soft': 'PBS'}

    def setUp(self):
        with patch.object(ssh, 'servers', {'srv': dict(self.SERVER)}):
            self.client = ssh.SSHClient('srv')
        self.stored = FakeHostKey(b'the-key-known_hosts-has')
        self.presented = FakeHostKey(b'the-key-the-server-sent')
        self.error = paramiko.BadHostKeyException(self.SERVER['address'],
                                                  self.presented, self.stored)

    def _connect(self):
        """Call connect() with _connect() failing on the mismatch, and return what was raised."""
        with patch.object(ssh, 'servers', {'srv': dict(self.SERVER)}), \
                patch.object(self.client, '_connect', MagicMock(side_effect=self.error)), \
                patch.object(ssh.time, 'sleep'):
            try:
                self.client.connect()
            except ServerError as exc:
                return exc
        return None

    def test_a_mismatch_raises_its_own_error_type(self):
        """Told apart from an absent key and from a wrong password, which mean other things."""
        raised = self._connect()
        self.assertIsInstance(raised, ssh.HostKeyMismatchError)
        self.assertIsInstance(raised, ServerError)
        self.assertIs(raised.__cause__, self.error)

    def test_the_error_names_both_fingerprints(self):
        """Which of the two the reader recognises is what decides re-keyed from intercepted."""
        message = str(self._connect())
        self.assertIn(ssh.get_host_key_fingerprint(self.stored), message)
        self.assertIn(ssh.get_host_key_fingerprint(self.presented), message)

    def test_the_error_gives_the_command_that_replaces_the_stale_key(self):
        message = str(self._connect())
        self.assertIn(f'ssh-keygen -R {self.SERVER["address"]}', message)

    def test_the_mismatch_is_reported_at_the_error_level(self):
        """A mismatch buried at the warning level reads as one more failed connection."""
        with self.assertLogs(ssh.logger, level='ERROR') as captured:
            self._connect()
        self.assertIn('does not match', '\n'.join(captured.output))

    def test_a_mismatch_is_not_retried(self):
        """Retrying an intercepted or re-keyed server cannot resolve it."""
        inner = MagicMock(side_effect=self.error)
        with patch.object(ssh, 'servers', {'srv': dict(self.SERVER)}), \
                patch.object(self.client, '_connect', inner), \
                patch.object(ssh.time, 'sleep') as sleep:
            self.assertRaises(ssh.HostKeyMismatchError, self.client.connect)
        self.assertEqual(inner.call_count, 1)
        self.assertEqual(sleep.call_count, 0)


class TestRemoteCommandQuoting(unittest.TestCase):
    """Caller-derived values must reach the remote shell as single arguments."""

    SERVER = {'address': 'host.example.edu', 'un': 'user', 'key': '/dev/null',
              'cluster_soft': 'PBS'}

    INJECTING = '/home/u/runs/a"; touch /tmp/pwned; "b'
    SPACED = '/home/u/my runs/proj 1'
    DASHED = '-rf'
    PLAIN = '/home/u/runs/ARC/proj/calcs'

    HOSTILE = (INJECTING, SPACED, DASHED)

    def _client(self):
        """Build an SSHClient without connecting to anything."""
        with patch.object(ssh, 'servers', {'srv': dict(self.SERVER)}):
            return ssh.SSHClient('srv')

    def _command_from(self, method_name, *args, **kwargs):
        """Return the command string the named method hands to the transport."""
        client = self._client()
        sender = MagicMock(return_value=([], []))
        with patch.object(ssh, 'servers', {'srv': dict(self.SERVER)}), \
                patch.object(client, '_send_command_to_server', sender):
            getattr(client, method_name)(*args, **kwargs)
        return sender.call_args[0][0]

    def _tokens(self, command):
        """Split a single-command string the way a POSIX shell would."""
        return shlex.split(command)

    def test_remove_dir_passes_the_path_as_one_argument(self):
        """A path with a quote, a space or a leading dash must not become shell syntax."""
        for path in self.HOSTILE:
            with self.subTest(path=path):
                tokens = self._tokens(self._command_from('remove_dir', path))
                self.assertEqual(tokens, ['rm', '-rf', '--', path])

    def test_remove_dir_does_not_leak_an_injected_command(self):
        """The injected payload must survive as data inside the path operand."""
        command = self._command_from('remove_dir', self.INJECTING)
        self.assertNotIn('; touch /tmp/pwned; ', command.replace(shlex.quote(self.INJECTING), ''))

    def test_create_dir_passes_the_path_as_one_argument(self):
        """mkdir must receive the path as a single operand after an end-of-options marker."""
        for path in self.HOSTILE:
            with self.subTest(path=path):
                tokens = self._tokens(self._command_from('_create_dir', path))
                self.assertEqual(tokens, ['mkdir', '-p', '--', path])

    def test_check_file_exists_keeps_a_valid_test_expression(self):
        """The path must be one word inside [ -f ... ] and the && echo must be intact."""
        for path in self.HOSTILE:
            with self.subTest(path=path):
                tokens = self._tokens(self._command_from('_check_file_exists', path))
                self.assertEqual(tokens, ['[', '-f', path, ']', '&&', 'echo', 'File exists'])

    def test_check_dir_exists_keeps_a_valid_test_expression(self):
        """The path must be one word inside [ -d ... ] and the && echo must be intact."""
        for path in self.HOSTILE:
            with self.subTest(path=path):
                tokens = self._tokens(self._command_from('_check_dir_exists', path))
                self.assertEqual(tokens, ['[', '-d', path, ']', '&&', 'echo', 'Dir exists'])

    def test_change_mode_quotes_only_the_file_name(self):
        """The mode stays literal shell while the file name becomes one operand."""
        for name in self.HOSTILE:
            with self.subTest(name=name):
                tokens = self._tokens(
                    self._command_from('change_mode', '+x', name, remote_path=''))
                self.assertEqual(tokens, ['chmod', '--', '+x', name])

    def test_change_mode_keeps_the_recursive_flag_before_the_marker(self):
        """Recursion is an ARC-controlled option and must precede the end-of-options marker."""
        tokens = self._tokens(
            self._command_from('change_mode', '+x', self.SPACED, recursive=True, remote_path=''))
        self.assertEqual(tokens, ['chmod', '-R', '--', '+x', self.SPACED])

    def test_find_package_passes_the_name_as_one_argument(self):
        """A package name reaches which() as a single argument."""
        tokens = self._tokens(self._command_from('find_package', 'g16'))
        self.assertEqual(tokens, ['.', '~/.bashrc;', 'which', 'g16'])

    def test_find_package_does_not_let_a_name_become_shell_syntax(self):
        """A hostile package name must not close the command and start another."""
        command = self._command_from('find_package', 'g16; touch /tmp/pwned')
        self.assertTrue(command.endswith(shlex.quote('g16; touch /tmp/pwned')))

    def test_ordinary_paths_are_unchanged(self):
        """A shell-safe path must be interpolated verbatim, with no quoting added."""
        self.assertEqual(self._command_from('remove_dir', self.PLAIN),
                         f'rm -rf -- {self.PLAIN}')
        self.assertEqual(self._command_from('_create_dir', self.PLAIN),
                         f'mkdir -p -- {self.PLAIN}')
        self.assertEqual(self._command_from('_check_file_exists', self.PLAIN),
                         f'[ -f {self.PLAIN} ] && echo "File exists"')


class TestRemotePathCdQuoting(unittest.TestCase):
    """The remote_path a command is executed in is quoted, the command itself is not."""

    SERVER = {'address': 'host.example.edu', 'un': 'user', 'key': '/dev/null',
              'cluster_soft': 'PBS'}

    def _command_sent(self, inner_command, remote_path):
        """Return the string handed to exec_command for a command run inside remote_path."""
        fake_ssh = MagicMock()
        fake_ssh.exec_command.return_value = (MagicMock(), MagicMock(), MagicMock())
        with patch.object(ssh, 'servers', {'srv': dict(self.SERVER)}):
            client = ssh.SSHClient('srv')
        client._ssh = fake_ssh
        with patch.object(client, '_check_dir_exists', return_value=True):
            client._send_command_to_server(inner_command, remote_path)
        return fake_ssh.exec_command.call_args_list[-1][0][0]

    def test_the_directory_is_quoted_and_the_command_is_not(self):
        """Only the path is a value; the assembled command stays shell."""
        command = self._command_sent('ls -alF', '/home/u/a"; touch /tmp/pwned; "b')
        self.assertEqual(
            command,
            "cd -- '/home/u/a\"; touch /tmp/pwned; \"b'; ls -alF; cd ")

    def test_a_path_with_a_space_stays_one_argument(self):
        """A spaced directory must not split into two cd operands."""
        command = self._command_sent('ls -alF', '/home/u/my runs')
        self.assertTrue(command.startswith("cd -- '/home/u/my runs'; "))

    def test_a_path_with_a_leading_dash_is_not_read_as_an_option(self):
        """The end-of-options marker keeps a dashed path an operand."""
        command = self._command_sent('ls -alF', '-P')
        self.assertTrue(command.startswith('cd -- -P; '))

    def test_an_ordinary_path_is_interpolated_verbatim(self):
        """A shell-safe path gains no quoting."""
        command = self._command_sent('ls -alF', '/home/u/runs/proj')
        self.assertEqual(command, 'cd -- /home/u/runs/proj; ls -alF; cd ')


class TestHostKeyPolicies(unittest.TestCase):
    """The unknown-host-key policies must reach the user and be classifiable."""

    HOST = 'host.example.edu'

    def setUp(self):
        """Build a host key and its expected fingerprint, derived independently of ssh.py."""
        self.key = FakeHostKey()
        digest = hashlib.sha256(self.key.asbytes()).digest()
        self.fingerprint = 'SHA256:' + base64.b64encode(digest).decode().rstrip('=')

    def _accept(self):
        """Run the default policy against the fake key, returning the logged records."""
        with self.assertLogs('arc', level='WARNING') as logged:
            result = ssh.LogAndAcceptHostKeyPolicy().missing_host_key(
                client=None, hostname=self.HOST, key=self.key)
        return result, logged.output

    def test_the_fingerprint_is_the_openssh_sha256_form(self):
        """The fingerprint must be comparable to what ssh-keygen -lf prints."""
        self.assertEqual(ssh.get_host_key_fingerprint(self.key), self.fingerprint)
        self.assertNotIn('=', ssh.get_host_key_fingerprint(self.key))

    def test_the_default_policy_logs_the_host_and_the_fingerprint(self):
        """Both are needed to verify the key against a trusted source."""
        _, output = self._accept()
        self.assertEqual(len(output), 1)
        self.assertIn(self.HOST, output[0])
        self.assertIn(self.fingerprint, output[0])

    def test_the_default_policy_still_accepts_the_key(self):
        """Logging must not change the connect-anyway behavior."""
        result, _ = self._accept()
        self.assertIsNone(result)

    def test_the_log_survives_the_paramiko_warnings_filter(self):
        """initialize_log() ignores paramiko warnings; ARC's own report must not be ignored."""
        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore', module='.*paramiko.*')
            with self.assertLogs('arc', level='WARNING') as logged:
                ssh.LogAndAcceptHostKeyPolicy().missing_host_key(
                    client=None, hostname=self.HOST, key=self.key)
        self.assertEqual(len(logged.output), 1)

    def test_paramikos_own_policy_is_silenced_by_that_filter(self):
        """The reason ARC cannot rely on paramiko.WarningPolicy: one warning becomes none."""
        with warnings.catch_warnings(record=True) as unfiltered:
            warnings.simplefilter('always')
            paramiko.WarningPolicy().missing_host_key(None, self.HOST, self.key)
        with warnings.catch_warnings(record=True) as filtered:
            warnings.simplefilter('always')
            warnings.filterwarnings(action='ignore', module='.*paramiko.*')
            paramiko.WarningPolicy().missing_host_key(None, self.HOST, self.key)
        self.assertEqual(len(unfiltered), 1)
        self.assertEqual(len(filtered), 0)

    def test_the_strict_policy_raises_a_distinct_error(self):
        """A refused host key must be identifiable by type, not by message text."""
        with self.assertRaises(ssh.UnknownHostKeyError) as raised:
            ssh.RejectUnknownHostKeyPolicy().missing_host_key(
                client=None, hostname=self.HOST, key=self.key)
        self.assertIsInstance(raised.exception, ServerError)
        self.assertIsInstance(raised.exception, paramiko.SSHException)
        self.assertIn(self.HOST, str(raised.exception))
        self.assertIn(self.fingerprint, str(raised.exception))


class TestConnectRetryClassification(unittest.TestCase):
    """connect() retries for 24 hours; only failures that retrying can resolve may enter it."""

    SERVER = {'address': 'host.example.edu', 'un': 'user', 'key': '/dev/null',
              'cluster_soft': 'PBS'}

    def _connect(self, side_effect):
        """Call connect() with _connect() and the retry interval faked out.

        Returns: tuple
            The raised exception (or None), the number of _connect() calls,
            and the number of sleeps between retries.
        """
        with patch.object(ssh, 'servers', {'srv': dict(self.SERVER)}):
            client = ssh.SSHClient('srv')
        inner = MagicMock(side_effect=side_effect)
        raised = None
        with patch.object(ssh, 'servers', {'srv': dict(self.SERVER)}), \
                patch.object(client, '_connect', inner), \
                patch.object(ssh.time, 'sleep') as sleep:
            try:
                client.connect()
            except ServerError as exc:
                raised = exc
        return raised, inner.call_count, sleep.call_count

    def _assert_not_retried(self, error):
        """A permanent failure surfaces as a ServerError on the first attempt."""
        raised, calls, sleeps = self._connect(error)
        self.assertIsInstance(raised, ServerError)
        self.assertIs(raised.__cause__, error)
        self.assertEqual(calls, 1)
        self.assertEqual(sleeps, 0)

    def test_a_rejected_authentication_is_not_retried(self):
        """A wrong key or username is not going to be accepted an hour later."""
        self._assert_not_retried(paramiko.AuthenticationException('Authentication failed.'))

    def test_a_required_password_is_not_retried(self):
        """PasswordRequiredException is an authentication failure, and equally permanent."""
        self._assert_not_retried(paramiko.PasswordRequiredException('Private key file is encrypted'))

    def test_a_changed_host_key_is_not_retried(self):
        """A key that contradicts known_hosts needs a human, not another attempt."""
        key = FakeHostKey()
        self._assert_not_retried(paramiko.BadHostKeyException(self.SERVER['address'], key, key))

    def test_a_refused_host_key_is_not_retried(self):
        """strict_host_key_checking refuses statelessly, so every retry refuses too."""
        self._assert_not_retried(ssh.UnknownHostKeyError('not in known_hosts'))

    def test_a_transport_failure_is_still_retried(self):
        """The banner/reset flake the retry was written for must keep being retried."""
        raised, calls, sleeps = self._connect(
            [paramiko.SSHException('Error reading SSH protocol banner'), ('sftp', 'ssh')])
        self.assertIsNone(raised)
        self.assertEqual(calls, 2)
        self.assertEqual(sleeps, 1)

    def test_an_unreachable_server_is_still_retried(self):
        """A refused socket may well be a server that is rebooting."""
        raised, calls, sleeps = self._connect([ConnectionRefusedError(111, 'Connection refused'),
                                               ('sftp', 'ssh')])
        self.assertIsNone(raised)
        self.assertEqual(calls, 2)
        self.assertEqual(sleeps, 1)


class TestDownloadFileWithoutARemoteFile(unittest.TestCase):
    """A local file must never survive a missing remote file as if it were this job's output."""

    SERVER = {'address': 'host.example.edu', 'un': 'user', 'key': '/dev/null',
              'cluster_soft': 'PBS'}

    def setUp(self):
        """Create a connected client whose remote files are all absent, and a temporary directory."""
        self.tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp_dir, ignore_errors=True)
        with patch.object(ssh, 'servers', {'srv': dict(self.SERVER)}):
            self.client = ssh.SSHClient('srv')
        self.client._sftp = MagicMock()
        self.client._ssh = MagicMock()
        self.local_path = os.path.join(self.tmp_dir, 'out.txt')

    def _download(self, exists=False):
        """Download a remote file whose existence check answers ``exists``.

        Returns: tuple
            The existence check and the sleep between attempts, as mocks.
        """
        with patch.object(self.client, '_check_file_exists',
                               side_effect=exists if isinstance(exists, list) else None,
                               return_value=None if isinstance(exists, list) else exists) as checked, \
                patch.object(ssh.time, 'sleep') as slept:
            self.client.download_file(remote_file_path='/remote/out.txt',
                                      local_file_path=self.local_path)
        return checked, slept

    def test_a_stale_local_file_is_emptied(self):
        """Otherwise a previous job's out.txt is read back as this job's server output."""
        with open(self.local_path, 'w') as f:
            f.write('slurmstepd: *** JOB 1 CANCELLED AT 2019-03-27 DUE TO TIME LIMIT ***\n')
        self._download()
        self.assertTrue(os.path.isfile(self.local_path))
        self.assertEqual(os.path.getsize(self.local_path), 0)

    def test_an_absent_local_file_is_created_empty(self):
        """The base behavior, which ESS troubleshooting reads: an empty file, not no file."""
        self._download()
        self.assertTrue(os.path.isfile(self.local_path))
        self.assertEqual(os.path.getsize(self.local_path), 0)

    def test_no_download_is_attempted(self):
        """Emptying the local file must not cost a pointless SFTP round trip."""
        self._download()
        self.client._sftp.get.assert_not_called()

    def test_the_miss_is_reported_at_the_warning_level(self):
        """A job that produced no stdout at all leaves this log line as its only trace."""
        with self.assertLogs('arc', level='WARNING') as logged:
            self._download()
        self.assertEqual(len(logged.output), 1)
        self.assertIn('/remote/out.txt', logged.output[0])
        self.assertIn('srv', logged.output[0])
        self.assertIn(self.local_path, logged.output[0])

    def test_the_existence_check_is_retried_three_times(self):
        """Scheduler epilogues can flush stdout a second or two after the job leaves the queue."""
        checked, slept = self._download()
        self.assertEqual(checked.call_count, 3)
        self.assertEqual(slept.call_count, 2)
        self.assertEqual([call.args[0] for call in slept.call_args_list], [1.0, 1.0])

    def test_a_file_that_appears_on_the_second_attempt_is_downloaded(self):
        """The retry exists to download that file, not merely to delay the warning."""
        with open(self.local_path, 'w') as f:
            f.write('an earlier download\n')
        checked, slept = self._download(exists=[False, True])
        self.assertEqual(checked.call_count, 2)
        self.assertEqual(slept.call_count, 1)
        self.client._sftp.get.assert_called_once_with(remotepath='/remote/out.txt',
                                                      localpath=self.local_path)
        self.assertEqual(os.path.getsize(self.local_path), len('an earlier download\n'))

    def test_an_unwritable_local_path_does_not_raise(self):
        """Downloads are best-effort; a directory that is gone must not abort the job."""
        self.client._empty_local_file(os.path.join(self.tmp_dir, 'no_such_dir', 'out.txt'))


class TestSSHClientConnect(unittest.TestCase):
    """Connection trial counting and interval scoping in SSHClient.connect()."""

    SERVER = {'address': 'host.example.edu', 'un': 'user', 'key': '/dev/null', 'cluster_soft': 'PBS'}

    def setUp(self):
        """
        A method that is run before each unit test in this class.
        Count the connection trials instead of attempting to reach a server.
        """
        self.trials, self.intervals = list(), list()
        for patcher in [patch.object(ssh, 'servers', {'srv': self.SERVER}),
                        patch.object(ssh.SSHClient, '_connect', lambda ssh_client: self.fail_to_connect()),
                        patch.object(ssh.time, 'sleep', lambda interval: self.intervals.append(interval))]:
            patcher.start()
            self.addCleanup(patcher.stop)

    def fail_to_connect(self):
        """
        Record a connection trial and fail it, as an unreachable server would.

        Raises:
            ServerError: Always.
        """
        self.trials.append(len(self.trials) + 1)
        raise ServerError('Could not connect.')

    def test_connect_gives_up_after_a_single_requested_trial(self):
        """Test that a single connection trial is not followed by an interval, so teardown cannot block"""
        ssh_client = ssh.SSHClient('srv', connection_attempts=1)
        with self.assertRaises(ServerError):
            ssh_client.connect()
        self.assertEqual(len(self.trials), 1)
        self.assertEqual(self.intervals, list())

    def test_connect_does_not_wait_after_its_last_trial(self):
        """Test that connecting waits between trials, but not after the last one"""
        ssh_client = ssh.SSHClient('srv', connection_attempts=3)
        with self.assertRaises(ServerError):
            ssh_client.connect()
        self.assertEqual(len(self.trials), 3)
        self.assertEqual(len(self.intervals), 2)

    def test_connect_defaults_to_the_long_haul(self):
        """Test that the default number of connection trials, used while jobs are running, is unchanged"""
        self.assertEqual(ssh.SSHClient('srv').connection_attempts, 1440)

    def test_a_permanent_failure_raises_on_the_first_trial_however_many_are_allowed(self):
        """Test that the retry budget does not resurrect retrying of a failure that retrying cannot fix"""
        with patch.object(ssh.SSHClient, '_connect',
                          side_effect=paramiko.AuthenticationException('nope')) as connect:
            ssh_client = ssh.SSHClient('srv', connection_attempts=1440)
            with self.assertRaises(ServerError):
                ssh_client.connect()
        self.assertEqual(connect.call_count, 1)
        self.assertEqual(self.intervals, list())


class TestDeleteCheckFilesOnServers(unittest.TestCase):
    """Deleting ESS checkfiles on the servers a project ran on."""

    SERVER = {'address': 'host.example.edu', 'un': 'user', 'key': '/dev/null', 'cluster_soft': 'PBS'}

    def setUp(self):
        """
        A method that is run before each unit test in this class.
        Set up a fake remote server: a temporary directory in which the commands ARC would have sent
        to a server are actually executed, so that the real cleanup code path is exercised.
        """
        self.remote_root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.remote_root, ignore_errors=True)
        self.commands = list()
        self.server = 'srv'
        self.project_path = os.path.join('runs', 'ARC_Projects', 'a_project')
        self.other_project_path = os.path.join('runs', 'ARC_Projects', 'an_unrelated_project')
        for patcher in [patch.object(ssh, 'servers', {self.server: self.SERVER, 'local': self.SERVER}),
                        patch.object(ssh.SSHClient, 'connect', lambda ssh_client: None),
                        patch.object(ssh.SSHClient, '_send_command_to_server',
                                     lambda ssh_client, command, remote_path='':
                                     self.execute_on_fake_server(command, remote_path))]:
            patcher.start()
            self.addCleanup(patcher.stop)
        self.check_file = self.write_remote_file(self.project_path, 'spc1', 'opt_a1', 'check.chk')
        self.output_file = self.write_remote_file(self.project_path, 'spc1', 'opt_a1', 'input.log')
        self.other_check_file = self.write_remote_file(self.other_project_path, 'spc2', 'opt_a1', 'check.chk')

    def execute_on_fake_server(self, command: str, remote_path: str = '') -> tuple:
        """
        Execute a command in the fake remote server directory instead of sending it to a server.

        Args:
            command (str): The command to execute.
            remote_path (str, optional): The directory path at which the command will be executed.

        Returns: tuple[list, list]
            The lines of the standard output and of the standard error streams.
        """
        self.commands.append(command)
        result = subprocess.run(command, shell=True, capture_output=True, text=True,
                                cwd=os.path.join(self.remote_root, remote_path))
        return result.stdout.splitlines(True), result.stderr.splitlines(True)

    def write_remote_file(self, *args) -> str:
        """
        Write a file in the fake remote server directory.

        Args:
            args: The path of the file relative to the fake remote server directory.

        Returns: str
            The path of the written file.
        """
        path = os.path.join(self.remote_root, *args)
        if not os.path.isdir(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        with open(path, 'w') as f:
            f.write('dummy file content')
        return path

    def test_only_check_files_of_the_given_project_are_deleted(self):
        """Test that check files are deleted, and that nothing else on the server is"""
        ssh.delete_check_files_on_servers({self.server: self.project_path})
        self.assertFalse(os.path.isfile(self.check_file))
        self.assertTrue(os.path.isfile(self.output_file))
        self.assertTrue(os.path.isfile(self.other_check_file))

    def test_a_local_server_is_skipped(self):
        """Test that a server named 'local' is skipped, its check files are deleted by the local cleanup"""
        ssh.delete_check_files_on_servers({'local': self.project_path})
        self.assertEqual(self.commands, list())
        self.assertTrue(os.path.isfile(self.check_file))

    def test_a_missing_remote_directory_is_a_no_op(self):
        """Test that a project directory which does not exist on the server is silently skipped"""
        ssh.delete_check_files_on_servers({self.server: os.path.join('runs', 'ARC_Projects', 'no_such_project')})
        self.assertEqual([command for command in self.commands if command.startswith('find')], list())
        self.assertTrue(os.path.isfile(self.check_file))
        self.assertTrue(os.path.isfile(self.other_check_file))

    def test_a_path_with_shell_characters_is_not_interpreted(self):
        """Test that the remote path is quoted, so that its content cannot become part of the command"""
        wild_path = os.path.join('runs', 'ARC_Projects', 'a project $(touch injected.txt)')
        wild_check_file = self.write_remote_file(wild_path, 'spc1', 'opt_a1', 'check.chk')
        with patch.object(ssh.SSHClient, '_check_dir_exists', lambda ssh_client, remote_dir_path: True):
            ssh.delete_check_files_on_servers({self.server: wild_path})
        self.assertEqual(len(self.commands), 1)
        self.assertEqual(shlex.split(self.commands[0])[1], wild_path)
        self.assertFalse(os.path.isfile(wild_check_file))
        self.assertFalse(os.path.isfile(os.path.join(self.remote_root, 'injected.txt')))

    def test_an_unreachable_server_is_only_logged(self):
        """Test that a server which cannot be reached does not raise, ARC is done with the science by now"""
        def raise_server_error(ssh_client):
            raise ServerError(f'Could not connect to server {ssh_client.server}.')

        with patch.object(ssh.SSHClient, 'connect', raise_server_error):
            ssh.delete_check_files_on_servers({self.server: self.project_path})
        self.assertTrue(os.path.isfile(self.check_file))

    def test_the_cleanup_connects_with_a_single_attempt(self):
        """Test that the cleanup does not inherit the 24 hour retry, which would stall ARC's teardown"""
        attempts = list()
        with patch.object(ssh.SSHClient, 'connect',
                          lambda ssh_client: attempts.append(ssh_client.connection_attempts)):
            ssh.delete_check_files_on_servers({self.server: self.project_path})
        self.assertEqual(attempts, [1])


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
