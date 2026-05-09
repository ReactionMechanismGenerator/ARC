#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.job.adapters.mockter module
"""

import os
import shutil
import tempfile
import unittest
from unittest import mock

from arc.common import ARC_TESTING_PATH, read_yaml_file, save_yaml_file
from arc.job.adapters.mockter import MockAdapter, MockterAbort, _parse_mockter_index
from arc.level import Level
from arc.reaction.reaction import ARCReaction
from arc.settings.settings import input_filenames, output_filenames
from arc.species import ARCSpecies


class TestMockAdapter(unittest.TestCase):
    """
    Contains unit tests for the MockAdapter class.
    """
    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.job_1 = MockAdapter(execution_type='incore',
                                job_type='sp',
                                level=Level(method='CCMockSD(T)', basis='cc-pVmockZ'),
                                project='test',
                                project_directory=os.path.join(ARC_TESTING_PATH, 'test_MockAdapter_1'),
                                species=[ARCSpecies(label='spc1', xyz=['O 0 0 1'], multiplicity=3)],
                                testing=True,
                                )
        cls.job_2 = MockAdapter(job_type='opt',
                                level=Level(method='CCMockSD(T)', basis='cc-pVmockZ'),
                                project='test',
                                project_directory=os.path.join(ARC_TESTING_PATH, 'test_MockAdapter_2'),
                                species=[ARCSpecies(label='spc2', xyz=['O 0 0 1'], multiplicity=3)],
                                testing=True,
                                )
        cls.job_3 = MockAdapter(job_type='freq',
                                level=Level(method='CCMockSD(T)', basis='cc-pVmockZ'),
                                project='test',
                                project_directory=os.path.join(ARC_TESTING_PATH, 'test_MockAdapter_3'),
                                species=[ARCSpecies(label='spc3', xyz=['O 0 0 1\nH 0 0 0\nH 1 0 0'], is_ts=True)],
                                testing=True,
                                )
        cls.job_4 = MockAdapter(job_type='tsg',
                                level=Level(method='mock)', basis='cc-pVmockZ'),
                                project='test',
                                project_directory=os.path.join(ARC_TESTING_PATH, 'test_MockAdapter_4'),
                                reactions=[ARCReaction(r_species=[ARCSpecies(label='O', smiles='[O]'),
                                                                  ARCSpecies(label='CCC', smiles='CCC')],
                                                       p_species=[ARCSpecies(label='OH', smiles='[OH]'),
                                                                  ARCSpecies(label='CCCyl', smiles='[CH2]CC')])],
                                testing=True,
                                )

    def test_set_cpu_and_mem(self):
        """Test assigning number of cpu's and memory"""
        self.job_1.cpu_cores = 48
        self.job_1.input_file_memory = None
        self.job_1.submit_script_memory = 14
        self.job_1.set_cpu_and_mem()
        self.assertEqual(self.job_1.cpu_cores, 48)

    def test_set_input_file_memory(self):
        """Test setting the input_file_memory argument"""
        self.job_1.input_file_memory = None
        self.job_1.cpu_cores = 48
        self.job_1.set_input_file_memory()
        self.assertEqual(self.job_1.input_file_memory, 14)

    def test_write_input_file(self):
        """Test writing Gaussian input files"""
        self.job_1.cpu_cores = 48
        self.job_1.set_input_file_memory()
        self.job_1.write_input_file()
        content_1 = read_yaml_file(os.path.join(self.job_1.local_path, input_filenames[self.job_1.job_adapter]))
        job_1_expected_input_file = {'basis': 'cc-pvmockz',
                                     'charge': 0,
                                     'job_type': 'sp',
                                     'label': 'spc1',
                                     'memory': 14.0,
                                     'method': 'ccmocksd(t)',
                                     'multiplicity': 3,
                                     'xyz': 'O       0.00000000    0.00000000    1.00000000'}
        self.assertEqual(content_1, job_1_expected_input_file)

        self.job_2.cpu_cores = 48
        self.job_2.set_input_file_memory()
        self.job_2.write_input_file()
        content_2 = read_yaml_file(os.path.join(self.job_2.local_path, input_filenames[self.job_2.job_adapter]))
        job_2_expected_input_file = {'basis': 'cc-pvmockz',
                                     'charge': 0,
                                     'job_type': 'opt',
                                     'label': 'spc2',
                                     'memory': 14.0,
                                     'method': 'ccmocksd(t)',
                                     'multiplicity': 3,
                                     'xyz': 'O       0.00000000    0.00000000    1.00000000'}
        self.assertEqual(content_2, job_2_expected_input_file)

        self.job_3.cpu_cores = 48
        self.job_3.set_input_file_memory()
        self.job_3.write_input_file()
        content_3 = read_yaml_file(os.path.join(self.job_3.local_path, input_filenames[self.job_3.job_adapter]))
        job_3_expected_input_file = {'basis': 'cc-pvmockz',
                                     'charge': 0,
                                     'job_type': 'freq',
                                     'label': 'spc3',
                                     'memory': 14.0,
                                     'method': 'ccmocksd(t)',
                                     'multiplicity': 1,
                                     'xyz': 'O       0.00000000    0.00000000    1.00000000\n'
                                            'H       0.00000000    0.00000000    0.00000000\n'
                                            'H       1.00000000    0.00000000    0.00000000'}
        self.assertEqual(content_3, job_3_expected_input_file)

    def test_get_mock_xyz(self):
        """Test getting the xyz coordinates from the mock input file"""
        self.job_1.write_input_file()
        xyz = self.job_1.get_mock_xyz()
        expected_xyz = ((0.0, 0.0, 1.0),)
        self.assertEqual(xyz['coords'], expected_xyz)

        self.job_4.write_input_file()
        xyz = self.job_4.get_mock_xyz()
        expected_xyz = {'coords': ((0.0, 0.0, 0.0),
                                   (-1.23638885, -0.31576286, 0.08035856),
                                   (-0.01951084, 0.58603214, 0.20011967),
                                   (1.25111292, -0.12649245, -0.23138091),
                                   (-1.1299088, -1.20421894, 0.71100631),
                                   (-1.38117893, -0.64405306, -0.95389019),
                                   (-2.13745897, 0.21915243, 0.39632612),
                                   (-0.16491381, 1.47886657, -0.41803363),
                                   (0.08498971, 0.92174781, 1.23780143),
                                   (1.43621378, -1.00896698, 0.38941334),
                                   (1.1849428, -0.44879975, -1.27548267),
                                   (2.11210099, 0.5424951, -0.13623803)),
                        'isotopes': (16, 12, 12, 12, 1, 1, 1, 1, 1, 1, 1, 1),
                        'symbols': ('O', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H')}
        self.assertEqual(xyz, expected_xyz)

    def test_executing_mockter(self):
        """
        Test executing mockter and reading back the debug ``output.yml`` mirror.

        Canonical mockter output is now the forged ``output.log`` Gaussian
        log (consumed by ARC's parser and Arkane); a ``output.yml`` mirror
        is written alongside for human/test inspection of the structured
        values.
        """
        self.job_1.execute()
        output = read_yaml_file(os.path.join(self.job_1.local_path, 'output.yml'))
        self.assertEqual(output['sp'], 0.0)
        self.assertEqual(output['T1'], 0.0002)

        self.job_2.execute()
        output = read_yaml_file(os.path.join(self.job_2.local_path, 'output.yml'))
        self.assertEqual(output['xyz'], {'coords': ((0.0, 0.0, 1.0),), 'isotopes': (16,), 'symbols': ('O',)})

        self.job_3.execute()
        output = read_yaml_file(os.path.join(self.job_3.local_path, 'output.yml'))
        self.assertEqual(output['freqs'], [-500, 520, 540])
        self.assertEqual(output['adapter'], 'mockter')

    @classmethod
    def tearDownClass(cls):
        """
        A function that is run ONCE after all unit tests in this class.
        Delete all project directories created during these unit tests
        """
        for folder in ['test_MockAdapter_1', 'test_MockAdapter_2', 'test_MockAdapter_3', 'test_MockAdapter_4']:
            shutil.rmtree(os.path.join(ARC_TESTING_PATH, folder), ignore_errors=True)


class TestMockAdapterFixtureRouting(unittest.TestCase):
    """Tests for the level.method → fixture-index parsing."""

    def test_simple_mockter_method(self):
        """``mockterN`` parses to N."""
        self.assertEqual(_parse_mockter_index('mockter3'), 3)
        self.assertEqual(_parse_mockter_index('Mockter12'), 12)

    def test_composite_mockter_method(self):
        """``mockter_<METHOD>_N`` parses to N (composite scenarios)."""
        self.assertEqual(_parse_mockter_index('mockter_CBS-QB3_3'), 3)
        self.assertEqual(_parse_mockter_index('mockter_G4_7'), 7)

    def test_non_mockter_method_returns_none(self):
        """Anything else returns None."""
        self.assertIsNone(_parse_mockter_index('cbs-qb3'))
        self.assertIsNone(_parse_mockter_index('wb97xd'))
        self.assertIsNone(_parse_mockter_index(None))
        self.assertIsNone(_parse_mockter_index(''))


class TestMockAdapterFixtureExecution(unittest.TestCase):
    """End-to-end execution against the committed mockter4 fixture."""

    def setUp(self):
        """Build a per-test project dir under ARC_TESTING_PATH."""
        self.project_dir = tempfile.mkdtemp(
            dir=ARC_TESTING_PATH, prefix='test_MockFixture_'
        )

    def tearDown(self):
        """Clean up the per-test project dir."""
        shutil.rmtree(self.project_dir, ignore_errors=True)

    def test_sp_lookup_against_mockter4_emits_real_energy(self):
        """``mockter4/...`` for H2O sp emits the real CCSD(T) energy from the fixture."""
        h2o = ARCSpecies(label='H2O', smiles='O')
        h2o.final_xyz = h2o.get_xyz()
        job = MockAdapter(
            execution_type='incore', job_type='sp',
            level=Level(method='mockter4', basis='def2tzvp'),
            project='test', project_directory=self.project_dir,
            species=[h2o], testing=True,
        )
        job.execute()
        log_path = os.path.join(job.local_path, 'output.log')
        self.assertTrue(os.path.isfile(log_path), 'output.log not produced')
        with open(log_path) as f:
            content = f.read()
        self.assertIn('CCSD(T)=', content)
        # Real H2O sp at CCSD(T)/cc-pVTZ from mockter4 is ~-76.33 Hartree.
        self.assertIn('-76.', content)
        self.assertIn('Normal termination', content)
        self.assertFalse(
            os.path.isfile(os.path.join(job.local_path, 'mockter_fallback.flag')),
            'fallback flag set despite a fixture hit'
        )

    def test_freq_lookup_against_mockter4_emits_freqs_and_hessian(self):
        """``mockter4/...`` for H2O freq emits the fixture's freqs, ZPE, and Hessian."""
        h2o = ARCSpecies(label='H2O', smiles='O')
        h2o.final_xyz = h2o.get_xyz()
        job = MockAdapter(
            execution_type='incore', job_type='freq',
            level=Level(method='mockter4', basis='def2tzvp'),
            project='test', project_directory=self.project_dir,
            species=[h2o], testing=True,
        )
        job.execute()
        with open(os.path.join(job.local_path, 'output.log')) as f:
            content = f.read()
        self.assertIn('Frequencies --', content)
        self.assertIn('Zero-point correction=', content)
        self.assertIn('Force constants in Cartesian coordinates:', content)

    def test_unknown_fixture_index_falls_back_with_flag(self):
        """``mockter99`` (no fixture file) writes a flag and uses fallback values."""
        spc = ARCSpecies(label='X', xyz=['O 0 0 1'])
        job = MockAdapter(
            execution_type='incore', job_type='sp',
            level=Level(method='mockter99', basis='def2tzvp'),
            project='test', project_directory=self.project_dir,
            species=[spc], testing=True,
        )
        job.execute()
        self.assertTrue(
            os.path.isfile(os.path.join(job.local_path, 'mockter_fallback.flag')),
            'fallback flag missing despite no fixture'
        )

    def test_non_mockter_level_uses_fallback(self):
        """A level that doesn't parse as mockterN falls back without a flag."""
        spc = ARCSpecies(label='X', xyz=['O 0 0 1'])
        job = MockAdapter(
            execution_type='incore', job_type='sp',
            level=Level(method='wb97xd', basis='def2tzvp'),
            project='test', project_directory=self.project_dir,
            species=[spc], testing=True,
        )
        job.execute()
        with open(os.path.join(job.local_path, 'output.log')) as f:
            content = f.read()
        self.assertIn('Normal termination', content)
        self.assertFalse(
            os.path.isfile(os.path.join(job.local_path, 'mockter_fallback.flag')),
            'flag set for non-mockter level (we only flag fixture-misses)'
        )


class TestMockAdapterRaiseClause(unittest.TestCase):
    """A fixture entry's ``raise:`` clause must abort mockter."""

    def setUp(self):
        """Build a tiny in-memory fixture with one raise entry per kind."""
        self.project_dir = tempfile.mkdtemp(
            dir=ARC_TESTING_PATH, prefix='test_MockRaise_'
        )
        # Write a temp fixture file at the path mockter expects.
        from arc.job.adapters import mockter as mockter_module
        self.fixture_path = os.path.join(mockter_module.MOCKTER_FIXTURES_DIR, 'mockter77.yml')
        save_yaml_file(path=self.fixture_path, content={
            'schema_version': 1,
            'species': {
                'X': {
                    'sp': {'raise': {'type': 'crash', 'message': 'simulated SCF crash'}},
                    'freq': {'raise': {'type': 'oom'}},
                },
            },
        })
        # Bust the cache so the fresh fixture is loaded.
        mockter_module._FIXTURE_CACHE.pop(77, None)

    def tearDown(self):
        from arc.job.adapters import mockter as mockter_module
        mockter_module._FIXTURE_CACHE.pop(77, None)
        if os.path.isfile(self.fixture_path):
            os.unlink(self.fixture_path)
        shutil.rmtree(self.project_dir, ignore_errors=True)

    def test_crash_kind_raises_mockter_abort(self):
        """A ``crash`` raise entry → MockterAbort with kind='crash'."""
        spc = ARCSpecies(label='X', xyz=['O 0 0 1'])
        job = MockAdapter(
            execution_type='incore', job_type='sp',
            level=Level(method='mockter77', basis='def2tzvp'),
            project='test', project_directory=self.project_dir,
            species=[spc], testing=True,
        )
        with self.assertRaises(MockterAbort) as ctx:
            job.execute_incore()
        self.assertEqual(ctx.exception.kind, 'crash')
        self.assertEqual(ctx.exception.message, 'simulated SCF crash')

    def test_oom_kind_raises_mockter_abort(self):
        """An ``oom`` raise entry → MockterAbort with kind='oom'."""
        spc = ARCSpecies(label='X', xyz=['O 0 0 1'])
        job = MockAdapter(
            execution_type='incore', job_type='freq',
            level=Level(method='mockter77', basis='def2tzvp'),
            project='test', project_directory=self.project_dir,
            species=[spc], testing=True,
        )
        with self.assertRaises(MockterAbort) as ctx:
            job.execute_incore()
        self.assertEqual(ctx.exception.kind, 'oom')


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
