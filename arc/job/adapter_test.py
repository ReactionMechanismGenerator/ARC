#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.job.adapter module
"""

import datetime
import glob
import math
import os
import tempfile
import time
import shutil
import unittest
from unittest.mock import patch

from arc.common import ARC_TESTING_PATH
from arc.imports import settings
from arc.job.adapter import JobAdapter, JobEnum, JobTypeEnum, JobExecutionTypeEnum
from arc.job.adapters.gaussian import GaussianAdapter
from arc.level import Level
from arc.species import ARCSpecies

servers, submit_filenames = settings['servers'], settings['submit_filenames']


class TestEnumerationClasses(unittest.TestCase):
    """
    Contains unit tests for various enumeration classes.
    """

    def test_job_enum(self):
        """Test the JobEnum class"""
        self.assertEqual(JobEnum('gaussian').value, 'gaussian')
        self.assertEqual(JobEnum('molpro').value, 'molpro')
        self.assertEqual(JobEnum('orca').value, 'orca')
        self.assertEqual(JobEnum('psi4').value, 'psi4')
        self.assertEqual(JobEnum('qchem').value, 'qchem')
        self.assertEqual(JobEnum('terachem').value, 'terachem')
        self.assertEqual(JobEnum('torchani').value, 'torchani')
        self.assertEqual(JobEnum('xtb').value, 'xtb')
        self.assertEqual(JobEnum('autotst').value, 'autotst')
        self.assertEqual(JobEnum('heuristics').value, 'heuristics')
        self.assertEqual(JobEnum('kinbot').value, 'kinbot')
        self.assertEqual(JobEnum('gcn').value, 'gcn')
        self.assertEqual(JobEnum('user').value, 'user')
        self.assertEqual(JobEnum('xtb_gsm').value, 'xtb_gsm')
        with self.assertRaises(ValueError):
            JobEnum('wrong')

    def test_job_type_enum(self):
        """Test the JobTypeEnum class"""
        self.assertEqual(JobTypeEnum('composite').value, 'composite')
        self.assertEqual(JobTypeEnum('conf_opt').value, 'conf_opt')
        self.assertEqual(JobTypeEnum('conf_sp').value, 'conf_sp')
        self.assertEqual(JobTypeEnum('freq').value, 'freq')
        self.assertEqual(JobTypeEnum('gen_confs').value, 'gen_confs')
        self.assertEqual(JobTypeEnum('irc').value, 'irc')
        self.assertEqual(JobTypeEnum('onedmin').value, 'onedmin')
        self.assertEqual(JobTypeEnum('opt').value, 'opt')
        self.assertEqual(JobTypeEnum('optfreq').value, 'optfreq')
        self.assertEqual(JobTypeEnum('orbitals').value, 'orbitals')
        self.assertEqual(JobTypeEnum('scan').value, 'scan')
        self.assertEqual(JobTypeEnum('directed_scan').value, 'directed_scan')
        self.assertEqual(JobTypeEnum('sp').value, 'sp')
        self.assertEqual(JobTypeEnum('tsg').value, 'tsg')
        with self.assertRaises(ValueError):
            JobTypeEnum('wrong')

    def test_job_execution_type_enum(self):
        """Test the JobExecutionTypeEnum class"""
        self.assertEqual(JobExecutionTypeEnum('incore').value, 'incore')
        self.assertEqual(JobExecutionTypeEnum('queue').value, 'queue')
        self.assertEqual(JobExecutionTypeEnum('pipe').value, 'pipe')
        with self.assertRaises(ValueError):
            JobExecutionTypeEnum('wrong')


class TestJobAdapter(unittest.TestCase):
    """
    Contains unit tests for the JobAdapter class.

    Here we use the GaussianAdapter class, but only test methods defined under the parent JobAdapter abstract class.
    """
    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None
        cls.job_1 = GaussianAdapter(execution_type='queue',
                                    job_type='conf_opt',
                                    level=Level(method='cbs-qb3'),
                                    project='test',
                                    project_directory=os.path.join(ARC_TESTING_PATH, 'test_JobAdapter'),
                                    species=[ARCSpecies(label='spc1',
                                                        xyz=['O 0 0 1',
                                                             'O 0 0 2',
                                                             'O 0 0 3',
                                                             'O 0 0 4',
                                                             'O 0 0 5',
                                                             'O 0 0 6']),
                                             ARCSpecies(label='spc2',
                                                        xyz=['N 0 0 1',
                                                             'N 0 0 2',
                                                             'N 0 0 3',
                                                             'N 0 0 4',
                                                             'N 0 0 5',
                                                             'N 0 0 6']),
                                             ARCSpecies(label='spc3',
                                                        xyz=['S 0 0 1',
                                                             'S 0 0 2',
                                                             'S 0 0 3',
                                                             'S 0 0 4',
                                                             'S 0 0 5',
                                                             'S 0 0 6']),
                                             ],
                                    testing=True,
                                    )
        cls.job_2 = GaussianAdapter(execution_type='incore',
                                    job_type='opt',
                                    level=Level(method='cbs-qb3'),
                                    project='test',
                                    project_directory=os.path.join(ARC_TESTING_PATH, 'test_JobAdapter'),
                                    species=[ARCSpecies(label='spc1', xyz=['O 0 0 1'])],
                                    testing=True,
                                    )
        cls.spc_3a = ARCSpecies(label='methanol',
                                smiles='CO',
                                directed_rotors={'brute_force_sp': ['all']})
        cls.spc_3b = ARCSpecies(label='methanol',
                                smiles='CO',
                                directed_rotors={'brute_force_opt': ['all']})
        cls.spc_3c = ARCSpecies(label='methanol',
                                smiles='CO',
                                directed_rotors={'cont_opt': ['all']})
        cls.spc_3d = ARCSpecies(label='methanol',
                                smiles='CO',
                                directed_rotors={'brute_force_sp_diagonal': ['all']})
        cls.spc_3e = ARCSpecies(label='methanol',
                                smiles='CO',
                                directed_rotors={'brute_force_opt_diagonal': ['all']})
        cls.spc_3f = ARCSpecies(label='methanol',
                                smiles='CO',
                                directed_rotors={'cont_opt_diagonal': ['all']})
        cls.job_3 = GaussianAdapter(execution_type='incore',
                                    job_type='scan',
                                    torsions=[[1, 2, 3, 4]],
                                    level=Level(method='wb97xd', basis='def2-tzvp'),
                                    project='test_scans',
                                    project_directory=os.path.join(ARC_TESTING_PATH, 'test_JobAdapter_scan'),
                                    species=[cls.spc_3a, cls.spc_3b, cls.spc_3c, cls.spc_3d, cls.spc_3e, cls.spc_3f],
                                    testing=True,
                                    )
        cls.job_4 = GaussianAdapter(execution_type='queue',
                                    job_type='opt',
                                    level=Level(method='cbs-qb3'),
                                    project='test',
                                    project_directory=os.path.join(ARC_TESTING_PATH, 'test_JobAdapter'),
                                    species=[ARCSpecies(label='spc1', xyz=['O 0 0 1'])],
                                    testing=True,
                                    )
        # Copy the PBS time limit fixture into the directory structure the adapter expects.
        stl_dir = os.path.join(ARC_TESTING_PATH, 'test_JobAdapter_ServerTimeLimit')
        err_dest = os.path.join(stl_dir, 'calcs', 'Species', 'spc1', 'opt_101')
        os.makedirs(err_dest, exist_ok=True)
        shutil.copy(os.path.join(ARC_TESTING_PATH, 'server', 'pbs', 'timelimit', 'err.txt'),
                    os.path.join(err_dest, 'err.txt'))
        cls.job_5 = GaussianAdapter(execution_type='queue',
                                    job_name='opt_101',
                                    job_type='opt',
                                    job_id='123456',
                                    job_num=101,
                                    job_server_name='server3',
                                    level=Level(method='cbs-qb3'),
                                    project='test',
                                    project_directory=stl_dir,
                                    species=[ARCSpecies(label='spc1', xyz=['O 0 0 1'])],
                                    server='server3',
                                    testing=True,
                                    )
        cls.job_6 = GaussianAdapter(execution_type='queue',
                                    job_name='opt_101',
                                    job_type='opt',
                                    job_id='123456',
                                    job_num=101,
                                    job_server_name='server1',
                                    level=Level(method='cbs-qb3'),
                                    project='test',
                                    project_directory=stl_dir,
                                    species=[ARCSpecies(label='spc1', xyz=['O 0 0 1'])],
                                    testing=True,
                                    queue='short_queue',
                                    attempted_queues=['short_queue']
                                    )

    def test_write_queue_submit_script(self):
        """Test writing a queue submit script"""
        self.job_4.number_of_processes, self.job_4.workers = 1, None
        self.job_4.write_submit_script()
        with open(os.path.join(self.job_4.local_path, submit_filenames[servers[self.job_4.server]['cluster_soft']]),
                  'r') as f:
            lines = f.readlines()
        array, hdf5, g16 = False, False, False
        for line in lines:
            if '#SBATCH --array=1-5' in line:
                array = True
            if 'job/scripts/pipe.py' in line and 'data.hdf5' in line:
                hdf5 = True
            if 'g16 < input.gjf > input.log' in line:
                g16 = True
        self.assertFalse(array)
        self.assertFalse(hdf5)
        self.assertTrue(g16)

    def test_determine_run_time(self):
        """Test determining the job run time"""
        self.job_2.initial_time = datetime.datetime.now()
        time.sleep(1)
        self.job_2.final_time = datetime.datetime.now()
        self.job_2.determine_run_time()
        self.assertEqual(self.job_2.run_time, datetime.timedelta(seconds=1))

    def test_set_cpu_and_mem(self):
        """Test assigning number of cpu's and memory"""
        # HTCondor
        self.job_4.cpu_cores = None
        self.job_4.set_cpu_and_mem()
        self.assertEqual(self.job_4.cpu_cores, 8)
        expected_memory = math.ceil((14 * 1024 * 1.1))
        self.assertEqual(self.job_4.submit_script_memory, expected_memory)

        # Slurm
        self.job_4.server = 'server2'
        self.job_4.cpu_cores = None
        self.job_4.set_cpu_and_mem()
        self.assertEqual(self.job_4.cpu_cores, 8)
        expected_memory = math.ceil((14 * 1024 * 1.1) / self.job_4.cpu_cores)
        self.assertEqual(self.job_4.submit_script_memory, expected_memory)
        self.job_4.server = 'local'

        # PBS
        self.job_4.server = 'server3'
        self.job_4.cpu_cores = None
        self.job_4.set_cpu_and_mem()
        expected_memory = math.ceil(14 * 1024 * 1.1) * 1E6
        self.assertEqual(self.job_4.submit_script_memory, expected_memory)
        self.job_4.server = 'local'

    def test_set_file_paths(self):
        """Test setting up the job's paths"""
        self.assertEqual(self.job_1.local_path, os.path.join(self.job_1.project_directory, 'calcs', 'Species',
                                                             self.job_1.species_label, self.job_1.job_name))
        self.assertEqual(self.job_1.remote_path, os.path.join('runs', 'ARC_Projects', self.job_1.project,
                                                              self.job_1.species_label, self.job_1.job_name))

    def test_format_max_job_time(self):
        """Test that the maximum job time can be formatted properly, including days, minutes, and seconds"""
        test_job = GaussianAdapter.__new__(GaussianAdapter)
        test_job.max_job_time = 59.888
        self.assertEqual(test_job.format_max_job_time('days'), '2-11:53:16')
        self.assertEqual(test_job.format_max_job_time('hours'), '59:53:16')

    def test_add_to_args(self):
        """Test adding parameters to self.args"""
        self.assertEqual(self.job_1.args, {'block': {}, 'keyword': {}, 'trsh': {}})
        self.job_1.add_to_args(val='val_tst_1')
        self.job_1.add_to_args(val='val_tst_2')
        self.job_1.add_to_args(val='val_tst_3', separator='     ')
        self.job_1.add_to_args(val="""val_tst_4\nval_tst_5""", key1='block', key2='specific_key_2')
        expected_args = {'keyword': {'general': 'val_tst_1 val_tst_2     val_tst_3'},
                         'block': {'specific_key_2': 'val_tst_4\nval_tst_5'},
                         'trsh': {}}
        self.assertEqual(self.job_1.args, expected_args)
        self.job_1.write_input_file()
        new_expected_args = {'keyword': {'general': 'val_tst_1 val_tst_2     val_tst_3'}, 
                             'block': {'specific_key_2': 'val_tst_4\nval_tst_5'},
                             'trsh': {}}
        self.assertEqual(self.job_1.args, new_expected_args)

        job_with_args = GaussianAdapter(execution_type='queue',
                                        job_type='opt',
                                        level=Level(method='cbs-qb3'),
                                        project='test',
                                        project_directory=os.path.join(ARC_TESTING_PATH, 'test_JobAdapter'),
                                        species=[ARCSpecies(label='spc1', xyz=['O 0 0 1'])],
                                        testing=True,
                                        args={'keyword': {'general': 'val_tst_1 val_tst_2     val_tst_3'},
                                              'block': {'specific_key_2': 'val_tst_4\nval_tst_5'},
                                              'trsh': {'no_xqc': 'no_xqc'},
                                              },
                                        )
        expected_args = {'keyword': {'general': 'val_tst_1 val_tst_2     val_tst_3'},
                         'block': {'specific_key_2': 'val_tst_4\nval_tst_5'},
                         'trsh': {'no_xqc': 'no_xqc'}}
        self.assertEqual(job_with_args.args, expected_args)

    def test_get_file_property_dictionary(self):
        """Test getting the file property dictionary"""
        file_dict_1 = self.job_1.get_file_property_dictionary(file_name='file.name')
        self.assertEqual(file_dict_1, {'file_name': 'file.name',
                                       'local': os.path.join(self.job_1.local_path, 'file.name'),
                                       'remote': os.path.join(self.job_1.remote_path, 'file.name'),
                                       'source': 'path',
                                       'make_x': False})
        file_dict_2 = self.job_1.get_file_property_dictionary(file_name='m.x',
                                                              local='onedmin.molpro.x',
                                                              source='input_files',
                                                              make_x=True)
        self.assertEqual(file_dict_2, {'file_name': 'm.x',
                                       'local': 'onedmin.molpro.x',
                                       'remote': os.path.join(self.job_1.remote_path, 'm.x'),
                                       'source': 'input_files',
                                       'make_x': True})

    def test_determine_job_status(self):
        """Test determining the job status"""
        self.job_5.determine_job_status()
        self.assertEqual(self.job_5.job_status[0], 'done')
        self.assertEqual(self.job_5.job_status[1]['status'], 'errored')
        self.assertEqual(self.job_5.job_status[1]['keywords'], ['ServerTimeLimit'])

    @patch(
        "arc.job.trsh.servers",
        {
            "local": {
                "cluster_soft": "PBS",
                "un": "test_user",
                "queues": {"short_queue": "24:00:0","middle_queue": "48:00:00", "long_queue": "3600:00:00"},
            }
        },
    ) 
    def test_troubleshoot_queue(self):
        """Test troubleshooting a queue job"""
        self.job_6.troubleshoot_queue()
        self.assertEqual(self.job_6.queue, 'middle_queue')
        # Assert that 'middle_queue' and 'short_queue' were attempted
        # We do not do assert equal because a user may have different queues from the settings.py originally during cls
        self.assertIn('short_queue', self.job_6.attempted_queues)
        self.assertIn('middle_queue', self.job_6.attempted_queues)



    @classmethod
    def tearDownClass(cls):
        """
        A function that is run ONCE after all unit tests in this class.
        Delete all project directories created during these unit tests
        """
        shutil.rmtree(os.path.join(ARC_TESTING_PATH, 'test_JobAdapter'), ignore_errors=True)
        shutil.rmtree(os.path.join(ARC_TESTING_PATH, 'test_JobAdapter_scan'), ignore_errors=True)
        shutil.rmtree(os.path.join(ARC_TESTING_PATH, 'test_JobAdapter_ServerTimeLimit'), ignore_errors=True)


class TestRotateCSV(unittest.TestCase):
    """
    Contains unit tests for the CSV rotation logic.
    """

    def _make_csv(self, path, num_lines):
        """Helper to create a CSV file with a header and ``num_lines - 1`` data rows."""
        with open(path, 'w') as f:
            f.write('col1,col2\n')
            for i in range(num_lines - 1):
                f.write(f'{i},data\n')

    def test_no_rotation_below_threshold(self):
        """Test that no rotation occurs when the file is below the threshold."""
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, 'jobs.csv')
            self._make_csv(csv_path, 10)
            JobAdapter._rotate_csv_if_needed(csv_path, max_lines=50)
            self.assertTrue(os.path.isfile(csv_path))
            self.assertEqual(glob.glob(os.path.join(tmp, 'jobs.old.*.csv')), [])

    def test_rotation_at_threshold(self):
        """Test that the file is rotated when it reaches the threshold."""
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, 'jobs.csv')
            self._make_csv(csv_path, 50)
            JobAdapter._rotate_csv_if_needed(csv_path, max_lines=50)
            self.assertFalse(os.path.isfile(csv_path))
            archives = glob.glob(os.path.join(tmp, 'jobs.old.*.csv'))
            self.assertEqual(len(archives), 1)

    def test_no_error_for_missing_file(self):
        """Test that rotation is a no-op when the file does not exist."""
        JobAdapter._rotate_csv_if_needed('/tmp/nonexistent_arc_test.csv')

    def test_multiple_rotations(self):
        """Test that multiple rotations produce distinct archive files."""
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, 'jobs.csv')
            # First rotation on "day 1"
            self._make_csv(csv_path, 50)
            with patch('arc.job.adapter.datetime') as mock_dt:
                mock_dt.datetime.now.return_value = datetime.datetime(2026, 1, 15)
                mock_dt.timedelta = datetime.timedelta
                JobAdapter._rotate_csv_if_needed(csv_path, max_lines=50)
            self.assertFalse(os.path.isfile(csv_path))
            # Second rotation on "day 2"
            self._make_csv(csv_path, 50)
            with patch('arc.job.adapter.datetime') as mock_dt:
                mock_dt.datetime.now.return_value = datetime.datetime(2026, 2, 20)
                mock_dt.timedelta = datetime.timedelta
                JobAdapter._rotate_csv_if_needed(csv_path, max_lines=50)
            self.assertFalse(os.path.isfile(csv_path))
            archives = glob.glob(os.path.join(tmp, 'jobs.old.*.csv'))
            self.assertEqual(len(archives), 2)


# ---------------------------------------------------------------------------
# SSH connection sharing & pooling (Options 1 + 2).
#
# Option 1 (per-job share): one SSHClient covers both upload and submit
# inside a single execute() call — collapses 2N connections to N.
# Option 2 (process-lifetime pool): the SSHClient for a given server is
# kept alive across jobs — collapses N to a small constant.
# ---------------------------------------------------------------------------


class _SSHClientStub:
    """In-memory SSHClient lookalike for the pool to hand out.

    Records every upload/submit so tests can assert which calls landed
    on which (shared) client. The pool calls ``connect()`` after
    instantiation; we no-op that since there's no real socket.
    """

    def __init__(self, server):
        self.server = server
        self.uploaded = []
        self.submits = []
        self.downloaded = []
        self._closed = False
        # Mimic SSHClient's ``_ssh`` attribute so ssh_pool._is_alive()
        # finds an active fake-Transport.
        self._ssh = _FakeParamikoSSH()

    def connect(self):
        pass  # the real one opens TCP+auth; we no-op for tests

    def close(self):
        self._closed = True
        self._ssh = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def upload_file(self, *, remote_file_path, local_file_path=None, file_string=None):
        self.uploaded.append(remote_file_path)

    def submit_job(self, remote_path, recursion=False):
        self.submits.append(remote_path)
        return 'initializing', 12345

    def change_mode(self, *, mode, file_name, remote_path):
        pass

    # Methods that the post-submit lifecycle paths exercise.
    def check_job_status(self, job_id):
        return 'running'

    def download_file(self, *, remote_file_path, local_file_path):
        self.downloaded.append(remote_file_path)

    def remove_dir(self, *, remote_path):
        pass

    def delete_job(self, job_id):
        pass


class _FakeParamikoSSH:
    """Stand-in for paramiko.SSHClient — _is_alive checks Transport.is_active()."""
    def get_transport(self):
        return _FakeTransport()


class _FakeTransport:
    def is_active(self):
        return True


class _StubFactoryPool:
    """A pool whose factory builds _SSHClientStub instead of real SSHClient.

    Wraps the production ``SSHConnectionPool`` so reuse + lifecycle
    semantics are exactly the production behavior — only the
    underlying object is faked.
    """

    def __init__(self):
        from arc.job.ssh_pool import SSHConnectionPool
        self.created = []  # log of every server name we built a client for
        def factory(server):
            client = _SSHClientStub(server)
            self.created.append(server)
            return client
        self._inner = SSHConnectionPool(factory=factory)

    def borrow(self, server):
        return self._inner.borrow(server)

    def close_all(self):
        self._inner.close_all()

    @property
    def opens(self):
        return self._inner.opens

    @property
    def borrows(self):
        return self._inner.borrows


class _MinimalAdapter(JobAdapter):
    """Concrete JobAdapter with just enough state to exercise execute().

    Skips the heavyweight construction the GaussianAdapter does — we
    only need ``server``, ``execution_type``, ``files_to_upload``,
    ``remote_path``, and ``testing=False`` for the SSH-share path.
    """

    job_adapter = 'mockter'

    def __init__(self, *, server, execution_type='queue'):
        # Bypass JobAdapter.__init__ entirely — all of its real work
        # (file paths, settings, csv setup) is unrelated to the SSH
        # share contract we're testing here.
        self.server = server
        self.execution_type = execution_type
        self.testing = False
        self.restarted = True   # skip _write_initiated_job_to_csv_file
        self.files_to_upload = [
            {'file_name': 'input.gjf', 'source': 'path',
             'local': '/local/input.gjf', 'remote': '/remote/input.gjf', 'make_x': False},
            {'file_name': 'submit.sh', 'source': 'path',
             'local': '/local/submit.sh', 'remote': '/remote/submit.sh', 'make_x': True},
        ]
        self.remote_path = '/remote'
        self.local_path = '/local'
        self.job_status = ['initializing', {'status': 'initializing'}]
        self.job_id = 0
        self.initial_time = None
        self.final_time = None
        self.job_name = 'job_test'
        self.species_label = 'spc_test'

    # JobAdapter requires these abstracts; trivial bodies are fine.
    def execute_incore(self): pass
    def execute_queue(self): self.legacy_queue_execution()
    def write_input_file(self): pass
    def set_files(self): pass
    def set_additional_file_paths(self): pass
    def set_input_file_memory(self): pass
    def upload_during_execution(self): pass
    def _log_job_execution(self): pass


class TestSSHConnectionSharing(unittest.TestCase):
    """``execute()`` shares one SSHClient per remote-queue job, and the
    pool reuses it across jobs."""

    def setUp(self):
        # Inject a pool whose factory builds stubs, so the test never
        # tries to open a real SSH connection to a server that isn't
        # in this user's settings (e.g., 'server2').
        import arc.job.ssh_pool as _pool
        self._stub_pool = _StubFactoryPool()
        _pool.set_default_pool(self._stub_pool)
        # Also stub the legacy-direct path: bare
        # ``legacy_queue_execution()`` (called outside execute()) uses
        # the SSHClient class in ``arc.job.adapter`` directly, so patch
        # that name with a context-manager wrapper around our stub.
        self._direct_patch = patch(
            'arc.job.adapter.SSHClient',
            lambda server: _SSHClientStub(server),
        )
        self._direct_patch.start()

    def tearDown(self):
        import arc.job.ssh_pool as _pool
        _pool.set_default_pool(None)
        self._direct_patch.stop()

    def test_remote_queue_opens_one_ssh_per_job(self):
        """Upload + submit share a single SSHClient inside one execute()."""
        adapter = _MinimalAdapter(server='server2', execution_type='queue')
        adapter.execute()
        # One SSHClient created (the pool's first borrow), one borrow.
        self.assertEqual(self._stub_pool.opens, 1)
        self.assertEqual(self._stub_pool.borrows, 1)

    def test_remote_queue_clears_shared_ssh_after_dispatch(self):
        """``self._shared_ssh`` is None after execute() returns."""
        adapter = _MinimalAdapter(server='server2', execution_type='queue')
        adapter.execute()
        self.assertIsNone(getattr(adapter, '_shared_ssh', None))

    def test_local_server_opens_no_ssh(self):
        """local-server queue jobs use the host's queue, no SSH at all."""
        adapter = _MinimalAdapter(server='local', execution_type='queue')
        with patch('arc.job.adapter.submit_job', return_value=('initializing', 99)):
            adapter.execute()
        self.assertEqual(self._stub_pool.opens, 0)
        self.assertEqual(self._stub_pool.borrows, 0)

    def test_incore_opens_no_ssh(self):
        """incore execution runs in-process — never touches SSH."""
        adapter = _MinimalAdapter(server='server2', execution_type='incore')
        adapter.execute()
        self.assertEqual(self._stub_pool.opens, 0)

    def test_legacy_queue_execution_routes_through_pool_when_called_directly(self):
        """Even when called bare (outside execute()), legacy_queue_execution
        now reuses the pool — that's Option 2's payoff for adapter
        ``execute_queue`` overrides that call ``self.legacy_queue_execution()``
        from inside their own custom flow.
        """
        adapter = _MinimalAdapter(server='server2', execution_type='queue')
        adapter.legacy_queue_execution()  # bare — no execute() wrapper
        self.assertEqual(self._stub_pool.opens, 1)
        self.assertEqual(self._stub_pool.borrows, 1)

    def test_shared_ssh_carries_uploads_and_submit(self):
        """The pooled SSHClient sees both upload calls AND the submit call."""
        adapter = _MinimalAdapter(server='server2', execution_type='queue')
        adapter.execute()
        # Inspect the stub the pool kept.
        self.assertEqual(self._stub_pool.opens, 1)
        client = self._stub_pool._inner._clients['server2']
        self.assertEqual(len(client.uploaded), 2)
        self.assertEqual(len(client.submits), 1)


class TestSSHConnectionPoolReuse(unittest.TestCase):
    """The process-lifetime pool reuses one SSHClient across many jobs."""

    def setUp(self):
        import arc.job.ssh_pool as _pool
        self._stub_pool = _StubFactoryPool()
        _pool.set_default_pool(self._stub_pool)

    def tearDown(self):
        import arc.job.ssh_pool as _pool
        _pool.set_default_pool(None)

    def test_one_open_for_many_jobs_same_server(self):
        """100 jobs against one server → 1 SSHClient, 100 borrows."""
        for _ in range(100):
            adapter = _MinimalAdapter(server='server2', execution_type='queue')
            adapter.execute()
        self.assertEqual(self._stub_pool.opens, 1, "should reuse the same client")
        self.assertEqual(self._stub_pool.borrows, 100)

    def test_separate_clients_per_distinct_server(self):
        """Different servers → different clients, each opened once."""
        for _ in range(5):
            _MinimalAdapter(server='server2', execution_type='queue').execute()
        for _ in range(3):
            _MinimalAdapter(server='server3', execution_type='queue').execute()
        self.assertEqual(self._stub_pool.opens, 2)
        self.assertEqual(self._stub_pool.borrows, 8)
        self.assertEqual(sorted(self._stub_pool._inner._clients.keys()),
                         ['server2', 'server3'])

    def test_dead_client_is_reaped_and_reopened(self):
        """If the underlying Transport reports inactive, pool reopens."""
        # First borrow → opens stub #1.
        _MinimalAdapter(server='server2', execution_type='queue').execute()
        client1 = self._stub_pool._inner._clients['server2']
        # Simulate a dead Transport (remote rebooted, etc.).
        client1._ssh = None
        # Next borrow should detect the dead client and open a fresh one.
        _MinimalAdapter(server='server2', execution_type='queue').execute()
        client2 = self._stub_pool._inner._clients['server2']
        self.assertIs(client1._closed, True, "stale client should be closed before reopen")
        self.assertIsNot(client1, client2)
        self.assertEqual(self._stub_pool.opens, 2)

    def test_close_all_closes_every_pooled_client(self):
        for srv in ('server2', 'server3'):
            _MinimalAdapter(server=srv, execution_type='queue').execute()
        clients = list(self._stub_pool._inner._clients.values())
        self._stub_pool.close_all()
        self.assertEqual(self._stub_pool._inner._clients, {})
        for c in clients:
            self.assertTrue(c._closed)

    def test_close_all_is_idempotent(self):
        _MinimalAdapter(server='server2', execution_type='queue').execute()
        self._stub_pool.close_all()
        # Second call must not raise or mutate state.
        self._stub_pool.close_all()
        self.assertEqual(self._stub_pool._inner._clients, {})

    def test_status_poll_reuses_pooled_client(self):
        """The hot path: hundreds of status checks open exactly one client.

        ARC polls a job's queue status every poll cycle for the entire
        duration of the job. Pre-pool, each call opened a fresh
        SSHClient. After Option 2, all polls reuse the pool's client
        for that server — the dominant SSH-cost reducer in a real run.
        """
        adapter = _MinimalAdapter(server='server2', execution_type='queue')
        # Simulate 200 poll cycles (~1.5 hour run at 30s polling).
        for _ in range(200):
            adapter._check_job_server_status()
        self.assertEqual(self._stub_pool.opens, 1, "pool should reuse one client")
        self.assertEqual(self._stub_pool.borrows, 200)

    def test_download_files_reuses_pooled_client(self):
        """download_files (called once per finished job) uses the pool too."""
        adapter = _MinimalAdapter(server='server2', execution_type='queue')
        adapter.files_to_download = [
            {'remote': '/r/output.log', 'local': '/l/output.log'},
        ]
        # set_initial_and_final_times reads file mtimes — stub it.
        adapter.set_initial_and_final_times = lambda ssh=None: None
        adapter.download_files()
        client = self._stub_pool._inner._clients['server2']
        self.assertIn('/r/output.log', client.downloaded)
        self.assertEqual(self._stub_pool.opens, 1)

    def test_full_lifecycle_one_open_per_server(self):
        """Submit + many polls + download + cleanup all share one pooled client.

        End-to-end view of one job's life: this collapses what was
        previously ~(2 + N_polls + 1 + 1) ≈ N+4 individual SSH
        connections into a single reused client.
        """
        adapter = _MinimalAdapter(server='server2', execution_type='queue')
        adapter.files_to_download = [{'remote': '/r/o.log', 'local': '/l/o.log'}]
        adapter.set_initial_and_final_times = lambda ssh=None: None

        adapter.execute()                  # upload + submit (1 borrow)
        for _ in range(50):                # 50 status polls
            adapter._check_job_server_status()
        adapter.download_files()           # 1 download borrow
        adapter.remove_remote_files()      # 1 cleanup borrow
        adapter.delete()                   # 1 delete borrow

        # All phases share the same pooled client.
        self.assertEqual(self._stub_pool.opens, 1)
        # 1 execute + 50 polls + 1 download + 1 cleanup + 1 delete = 54 borrows.
        self.assertEqual(self._stub_pool.borrows, 54)


class TestSSHPoolDefaultLifecycle(unittest.TestCase):
    """The module-level default pool is lazy and resettable."""

    def setUp(self):
        import arc.job.ssh_pool as _pool
        _pool.reset_default_pool()
        self._pool_module = _pool

    def tearDown(self):
        self._pool_module.reset_default_pool()

    def test_get_default_pool_is_idempotent(self):
        p1 = self._pool_module.get_default_pool()
        p2 = self._pool_module.get_default_pool()
        self.assertIs(p1, p2)

    def test_reset_default_pool_drops_the_instance(self):
        p1 = self._pool_module.get_default_pool()
        self._pool_module.reset_default_pool()
        p2 = self._pool_module.get_default_pool()
        self.assertIsNot(p1, p2)

    def test_set_default_pool_replaces_instance(self):
        replacement = _StubFactoryPool()
        self._pool_module.set_default_pool(replacement)
        self.assertIs(self._pool_module.get_default_pool(), replacement)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
