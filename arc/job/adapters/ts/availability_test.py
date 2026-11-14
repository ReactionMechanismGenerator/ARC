#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.job.adapters.ts.availability module
"""

import os
import pathlib
import shutil
import tempfile
import unittest
from unittest.mock import patch

import arc.job.adapters.ts.autotst_ts as autotst_ts
import arc.job.adapters.ts.gcn_ts as gcn_ts
import arc.job.adapters.ts.goflow_ts as goflow_ts
import arc.job.adapters.ts.kinbot_ts as kinbot_ts
import arc.job.adapters.ts.rits_ts as rits_ts
import arc.job.adapters.ts.xtb_gsm as xtb_gsm
from arc.imports import settings
from arc.job.adapters.common import ts_adapters_by_rmg_family
from arc.job.adapters.ts.availability import (ENGINE_FREE_TS_ADAPTERS,
                                              ENGINE_REASON_FUNCTIONS,
                                              SERVER_REASON_FUNCTIONS,
                                              ts_adapter_unavailable_reason,
                                              )
from arc.job.factory import _registered_job_adapters


class TestTSAdapterAvailability(unittest.TestCase):
    """
    Contains unit tests for ts_adapter_unavailable_reason().
    """

    @classmethod
    def setUpClass(cls):
        """A method that is run before all unit tests in this class."""
        cls.tmp_dir = tempfile.mkdtemp(prefix='arc_availability_test_')
        cls.exe = os.path.join(cls.tmp_dir, 'python')
        with open(cls.exe, 'w') as f:
            f.write('')
        cls.big_file = os.path.join(cls.tmp_dir, 'big.ckpt')
        with open(cls.big_file, 'wb') as f:
            f.truncate(2_000_000)

    @classmethod
    def tearDownClass(cls):
        """A method that is run after all unit tests in this class."""
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def test_engine_free_adapters_are_available(self):
        """Test that engine-free adapters are always available."""
        for adapter in ENGINE_FREE_TS_ADAPTERS:
            self.assertIsNone(ts_adapter_unavailable_reason(adapter, ess_settings=dict()))
        self.assertIsNone(ts_adapter_unavailable_reason('Linear', ess_settings=None))
        self.assertIsNone(ts_adapter_unavailable_reason('mockter', ess_settings=None))

    def test_python_env_adapters(self):
        """Test the adapters whose engine is the python executable of a conda env."""
        for adapter, module, attr, env in [('autotst', autotst_ts, 'AUTOTST_PYTHON', 'tst_env'),
                                           ('kinbot', kinbot_ts, 'KINBOT_PYTHON', 'kinbot_env'),
                                           ('gcn', gcn_ts, 'TS_GCN_PYTHON', 'ts_gcn'),
                                           ('xtb_gsm', xtb_gsm, 'XTB', 'xtb_env')]:
            with self.subTest(adapter=adapter):
                with patch.object(module, attr, self.exe):
                    self.assertIsNone(ts_adapter_unavailable_reason(adapter, ess_settings=None))
                with patch.object(module, attr, None):
                    reason = ts_adapter_unavailable_reason(adapter, ess_settings=None)
                    self.assertIn(attr, reason)
                    self.assertIn(env, reason)
                with patch.object(module, attr, os.path.join(self.tmp_dir, 'missing')):
                    self.assertIn(attr, ts_adapter_unavailable_reason(adapter, ess_settings=None))
                with patch.object(module, attr, pathlib.Path(self.exe)):  # A PathLike setting is a path.
                    self.assertIsNone(ts_adapter_unavailable_reason(adapter, ess_settings=None))
                with patch.object(module, attr, 42):  # A wrong type refuses rather than raising.
                    self.assertIn(attr, ts_adapter_unavailable_reason(adapter, ess_settings=None))

    def test_goflow(self):
        """Test the GoFlow availability reason, which joins every missing piece into one sentence."""
        with patch.object(goflow_ts, 'GOFLOW_PYTHON', self.exe), \
                patch.object(goflow_ts, 'GOFLOW_REPO_PATH', self.tmp_dir), \
                patch.object(goflow_ts, 'GOFLOW_CKPT_PATH', self.big_file), \
                patch.object(goflow_ts, 'GOFLOW_FEAT_DICT_PATH', self.big_file):
            self.assertIsNone(ts_adapter_unavailable_reason('goflow', ess_settings=None))
        with patch.object(goflow_ts, 'GOFLOW_PYTHON', pathlib.Path(self.exe)), \
                patch.object(goflow_ts, 'GOFLOW_REPO_PATH', pathlib.Path(self.tmp_dir)), \
                patch.object(goflow_ts, 'GOFLOW_CKPT_PATH', self.big_file), \
                patch.object(goflow_ts, 'GOFLOW_FEAT_DICT_PATH', self.big_file), \
                patch('arc.job.adapters.ts.goflow_ts.os.path.getsize', side_effect=FileNotFoundError):
            # A file removed between the existence and the size checks is reported, not raised.
            reason = ts_adapter_unavailable_reason('goflow', ess_settings=None)
        self.assertIn('ARC_GOFLOW_CKPT', reason)
        self.assertNotIn('GOFLOW_PYTHON', reason)
        with patch.object(goflow_ts, 'GOFLOW_PYTHON', None), \
                patch.object(goflow_ts, 'GOFLOW_REPO_PATH', None), \
                patch.object(goflow_ts, 'GOFLOW_CKPT_PATH', self.exe), \
                patch.object(goflow_ts, 'GOFLOW_FEAT_DICT_PATH', 7):
            reason = ts_adapter_unavailable_reason('goflow', ess_settings=None)
        for token in ['GOFLOW_PYTHON', 'goflow_env', 'ARC_GOFLOW_REPO', 'ARC_GOFLOW_CKPT', 'ARC_GOFLOW_FEAT_DICT']:
            self.assertIn(token, reason)
        self.assertNotIn('\n', reason)

    def test_rits(self):
        """Test the RitS availability reason."""
        with patch.object(rits_ts, 'RITS_PYTHON', self.exe), \
                patch.object(rits_ts, 'RITS_REPO_PATH', self.tmp_dir), \
                patch.object(rits_ts, 'RITS_CKPT_PATH', self.exe):
            self.assertIsNone(ts_adapter_unavailable_reason('rits', ess_settings=None))
        with patch.object(rits_ts, 'RITS_PYTHON', pathlib.Path(self.exe)), \
                patch.object(rits_ts, 'RITS_REPO_PATH', pathlib.Path(self.tmp_dir)), \
                patch.object(rits_ts, 'RITS_CKPT_PATH', pathlib.Path(self.exe)):
            self.assertIsNone(ts_adapter_unavailable_reason('rits', ess_settings=None))
        with patch.object(rits_ts, 'RITS_PYTHON', None), \
                patch.object(rits_ts, 'RITS_REPO_PATH', self.tmp_dir), \
                patch.object(rits_ts, 'RITS_CKPT_PATH', None):
            reason = ts_adapter_unavailable_reason('rits', ess_settings=None)
        self.assertIn('RITS_PYTHON', reason)
        self.assertIn('rits_env', reason)
        self.assertIn('ARC_RITS_CKPT', reason)
        self.assertNotIn('ARC_RITS_REPO', reason)

    @patch('arc.job.adapters.ts.availability.servers', {'orca_server': {'cluster_soft': 'Slurm'}})
    def test_orca_neb(self):
        """Test that Orca NEB needs a configured Orca server in the ESS settings, which is where its job finds one."""
        server = 'orca_server'
        self.assertIsNone(ts_adapter_unavailable_reason('orca_neb', ess_settings={'orca': [server]}))
        self.assertIsNone(ts_adapter_unavailable_reason('orca_neb', ess_settings={'orca': server}))
        with patch('arc.job.adapters.ts.availability.global_ess_settings', {'orca': [server]}):
            # Empty ESS settings fall back to the global ones, as the job does.
            self.assertIsNone(ts_adapter_unavailable_reason('orca_neb', ess_settings=dict()))
            self.assertIsNotNone(ts_adapter_unavailable_reason('orca_neb', ess_settings={'gaussian': [server]}))
        with patch('arc.job.adapters.ts.availability.global_ess_settings', {'gaussian': [server]}):
            self.assertIsNotNone(ts_adapter_unavailable_reason('orca_neb', ess_settings=None))
        for ess_settings in [{'gaussian': [server]}, {'orca_neb': [server]}, {'orca': list()}, {'orca': None},
                             {'orca': ['no_such_server_xyz']}, {'orca': [server, 'no_such_server_xyz']},
                             {'orca': [3]}]:
            with self.subTest(ess_settings=ess_settings):
                reason = ts_adapter_unavailable_reason('orca_neb', ess_settings=ess_settings)
                self.assertIsNotNone(reason)
                self.assertIn("'orca'", reason)

    def test_unregistered_adapter(self):
        """Test that an adapter with no registered module is unavailable, and named."""
        reason = ts_adapter_unavailable_reason('no_such_ts_adapter_xyz', ess_settings=None)
        self.assertIn("'no_such_ts_adapter_xyz'", reason)
        self.assertIn('registered', reason)

    def test_every_mapped_adapter_is_classified(self):
        """Test that every TS adapter ARC may pick is engine-free, has an engine check, or is unregistered.

        A new engine-backed adapter added to the family map (or to the default ``ts_adapters``)
        without an entry in ENGINE_REASON_FUNCTIONS or SERVER_REASON_FUNCTIONS would otherwise skip the check and fail open.
        """
        adapters = {adapter.lower() for adapters in ts_adapters_by_rmg_family.values() for adapter in adapters}
        adapters |= {adapter.lower() for adapter in settings['ts_adapters']}
        for adapter in sorted(adapters):
            with self.subTest(adapter=adapter):
                self.assertTrue(adapter in ENGINE_FREE_TS_ADAPTERS
                                or adapter in ENGINE_REASON_FUNCTIONS
                                or adapter in SERVER_REASON_FUNCTIONS
                                or adapter not in _registered_job_adapters,
                                msg=f'{adapter!r} has no engine availability check')


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
