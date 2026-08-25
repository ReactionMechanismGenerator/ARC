#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests for the arc.job.adapters.pyscf_adapter module.

The adapter tests verify IO and logic without executing PySCF (the engine runs in the separate
``pyscf_env`` conda environment, which is absent from ARC's CI env). Tests that import the engine
script (``pyscf_script.py``) are skipped unless PySCF is importable.
"""

import importlib.util
import os
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from arc.common import read_yaml_file, save_yaml_file
from arc.job.adapters.pyscf_adapter import PySCFAdapter
from arc.level import Level
from arc.parser.adapters.yaml import CARTESIAN_CONVENTION, YAMLParser
from arc.settings.settings import PYSCF_PYTHON
from arc.species.species import ARCSpecies

HAS_PYSCF = importlib.util.find_spec('pyscf') is not None

PYSCF_IMPORT_PROBE_TIMEOUT = 120


def pyscf_env_can_import() -> bool:
    """
    Determine whether the interpreter configured as ``PYSCF_PYTHON`` can import PySCF.

    ``PYSCF_PYTHON`` is a configured path, and a path being set does not mean that the
    environment it points at has PySCF installed, so the imports are attempted once rather
    than inferred from the setting. The modules probed are those ``pyscf_script`` imports
    at module level, which excludes the geometry optimization solver.

    Returns:
        bool: Whether that interpreter imports PySCF successfully.
    """
    if PYSCF_PYTHON is None:
        return False
    try:
        completed = subprocess.run([PYSCF_PYTHON, '-c',
                                    'from pyscf import dft, gto, lib; from pyscf.hessian import thermo'],
                                   capture_output=True,
                                   timeout=PYSCF_IMPORT_PROBE_TIMEOUT)
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


PYSCF_ENV_AVAILABLE = pyscf_env_can_import()
PYSCF_SCHEMA_VERSION = 2


class TestPySCFAdapter(unittest.TestCase):
    """
    Contains unit tests for the PySCFAdapter class.
    """

    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None
        cls.project_directory = tempfile.mkdtemp(prefix='test_PySCFAdapter_')

        water_xyz = {'symbols': ('O', 'H', 'H'),
                     'isotopes': (16, 1, 1),
                     'coords': ((0.0, 0.0, 0.1173),
                                (0.0, 0.7572, -0.4692),
                                (0.0, -0.7572, -0.4692))}
        oh_xyz = {'symbols': ('O', 'H'),
                  'isotopes': (16, 1),
                  'coords': ((0.0, 0.0, 0.0),
                             (0.0, 0.0, 0.9738))}

        cls.job_1 = PySCFAdapter(execution_type='incore',
                                 job_type='sp',
                                 project='test_1',
                                 project_directory=os.path.join(cls.project_directory, 'test_1'),
                                 level=Level(repr='wb97m-v/def2tzvp'),
                                 species=[ARCSpecies(label='H2O', xyz=water_xyz)],
                                 testing=True)

        cls.job_2 = PySCFAdapter(execution_type='queue',
                                 job_type='optfreq',
                                 project='test_2',
                                 project_directory=os.path.join(cls.project_directory, 'test_2'),
                                 level=Level(repr='wb97m-v/def2tzvp'),
                                 species=[ARCSpecies(label='OH', xyz=oh_xyz, multiplicity=2)],
                                 testing=True)

        cls.job_1.local_path = os.path.join(cls.project_directory, 'test_1')
        cls.job_2.local_path = os.path.join(cls.project_directory, 'test_2')
        cls.job_2.remote_path = '/path/to/remote'
        os.makedirs(cls.job_1.local_path, exist_ok=True)
        os.makedirs(cls.job_2.local_path, exist_ok=True)

    def test_determine_settings(self):
        """Test resolving the method (xc functional) and basis from the level."""
        settings = self.job_1.determine_settings()
        self.assertEqual(settings['method'], 'wb97m-v')
        self.assertEqual(settings['basis'], 'def2tzvp')
        self.assertIn('memory_mb', settings)
        self.assertIn('device', settings)
        self.assertIn('fine', settings)

    def test_device_can_be_requested_per_job(self):
        """Test that an explicit device keyword overrides the default device."""
        job = PySCFAdapter(execution_type='incore',
                           job_type='sp',
                           project='test_gpu',
                           project_directory=os.path.join(self.project_directory, 'test_gpu'),
                           level=Level(repr='wb97m-v/def2tzvp'),
                           species=[ARCSpecies(label='H2O', xyz=self.job_1.xyz)],
                           args={'keyword': {'device': 'gpu'}},
                           testing=True)
        self.assertEqual(job.determine_settings()['device'], 'gpu')

    def test_level_takes_precedence_over_conflicting_keyword(self):
        """Test that the job's level, not a conflicting args keyword, sets method/basis.

        A user keyword must never silently override the level of theory: PySCF would then compute
        one method while ARC's level object (and restart.yml) records another. The level wins and a
        warning is logged naming the dropped keyword value.
        """
        water_xyz = {'symbols': ('O', 'H', 'H'), 'isotopes': (16, 1, 1),
                     'coords': ((0.0, 0.0, 0.1173), (0.0, 0.7572, -0.4692), (0.0, -0.7572, -0.4692))}
        job = PySCFAdapter(execution_type='incore',
                           job_type='sp',
                           project='test_conflict',
                           project_directory=os.path.join(self.project_directory, 'test_conflict'),
                           level=Level(repr='wb97m-v/def2tzvp'),
                           species=[ARCSpecies(label='H2O', xyz=water_xyz)],
                           args={'keyword': {'method': 'pbe', 'basis': 'sto-3g'}},
                           testing=True)
        with self.assertLogs('arc', level='WARNING') as cm:
            settings = job.determine_settings()
        self.assertEqual(settings['method'], 'wb97m-v')
        self.assertEqual(settings['basis'], 'def2tzvp')
        self.assertTrue(any('pbe' in msg for msg in cm.output))

    def test_matching_keyword_does_not_warn(self):
        """Test that a keyword matching the level is accepted silently (no divergence, no warning)."""
        water_xyz = {'symbols': ('O', 'H', 'H'), 'isotopes': (16, 1, 1),
                     'coords': ((0.0, 0.0, 0.1173), (0.0, 0.7572, -0.4692), (0.0, -0.7572, -0.4692))}
        job = PySCFAdapter(execution_type='incore',
                           job_type='sp',
                           project='test_match',
                           project_directory=os.path.join(self.project_directory, 'test_match'),
                           level=Level(repr='wb97m-v/def2tzvp'),
                           species=[ARCSpecies(label='H2O', xyz=water_xyz)],
                           args={'keyword': {'method': 'wb97m-v'}},
                           testing=True)
        with self.assertNoLogs('arc', level='WARNING'):
            settings = job.determine_settings()
        self.assertEqual(settings['method'], 'wb97m-v')

    def test_write_input_file(self):
        """Test writing the YAML input file for the PySCF script."""
        self.job_2.write_input_file()
        input_path = os.path.join(self.job_2.local_path, 'input.yml')
        self.assertTrue(os.path.isfile(input_path))
        data = read_yaml_file(input_path)
        self.assertEqual(data['job_type'], 'optfreq')
        self.assertEqual(data['multiplicity'], 2)
        self.assertEqual(data['settings']['method'], 'wb97m-v')
        self.assertEqual(data['settings']['basis'], 'def2tzvp')
        self.assertEqual(data['xyz']['symbols'], ('O', 'H'))

    def test_write_submit_script(self):
        """Test writing the submission script for the queue fallback path."""
        self.job_2.write_submit_script()
        submit_path = os.path.join(self.job_2.local_path, 'submit.sh')
        self.assertTrue(os.path.isfile(submit_path))
        with open(submit_path, 'r') as f:
            content = f.read()
        self.assertIn('--yml_path /path/to/remote', content)
        self.assertIn('pyscf_script.py', content)

    def test_set_files(self):
        """Test properly assigning upload and download files."""
        self.job_2.set_files()
        self.assertTrue(any('submit.sh' in f['local'] for f in self.job_2.files_to_upload))
        self.assertTrue(any('input.yml' in f['local'] for f in self.job_2.files_to_upload))
        self.assertTrue(any('pyscf_script.py' in f['local'] for f in self.job_2.files_to_upload))
        self.assertTrue(any('output.yml' in f['local'] for f in self.job_2.files_to_download))

    def test_parse_results(self):
        """Test parsing a dummy output YAML back into object attributes."""
        output_data = {
            'schema_version': 1, 'adapter': 'pyscf', 'success': True, 'error': None,
            'sp': -200665.9,
            'opt_xyz': {'symbols': ('O', 'H', 'H'),
                        'isotopes': (16, 1, 1),
                        'coords': ((0.0, 0.0, 0.116), (0.0, 0.763, -0.469), (0.0, -0.763, -0.469))},
            'freqs': [1617.5, 3823.7, 3927.5],
            'modes': [[[0.0, 0.0, 0.1]]],
            'zpe': 56.0,
        }
        save_yaml_file(os.path.join(self.job_1.local_path, 'output.yml'), output_data)
        self.job_1.parse_results()
        self.assertEqual(self.job_1.sp, -200665.9)
        self.assertEqual(self.job_1.freqs, [1617.5, 3823.7, 3927.5])
        self.assertIsNotNone(self.job_1.opt_xyz)
        self.assertAlmostEqual(self.job_1.opt_xyz['coords'][1][1], 0.763)

    def test_output_yaml_round_trips_through_yaml_parser(self):
        """Test that a PySCF output.yml round-trips through YAMLParser (energy, geom, freqs, zpe)."""
        output_data = {
            'schema_version': 1, 'adapter': 'pyscf', 'success': True, 'error': None,
            'sp': -200665.87676,
            'opt_xyz': {'symbols': ('O', 'H', 'H'),
                        'isotopes': (16, 1, 1),
                        'coords': ((0.0, 0.0, 0.116), (0.0, 0.763, -0.469), (0.0, -0.763, -0.469))},
            'freqs': [1617.5, 3823.7, 3927.5],
            'modes': [[[0.0, 0.0, 0.1], [0.0, 0.4, 0.5], [0.0, -0.4, 0.5]]],
            'zpe': 56.04,
            'dipole': 2.07,
        }
        out_path = os.path.join(self.job_1.local_path, 'output_rt.yml')
        save_yaml_file(out_path, output_data)
        parser = YAMLParser(log_file_path=out_path)
        self.assertAlmostEqual(parser.parse_e_elect(), -200665.87676, places=3)
        self.assertEqual(parser.parse_geometry()['symbols'], ('O', 'H', 'H'))
        self.assertAlmostEqual(parser.parse_frequencies()[0], 1617.5, places=1)
        self.assertAlmostEqual(parser.parse_zpe_correction(), 56.04, places=2)
        self.assertAlmostEqual(parser.parse_dipole_moment(), 2.07, places=2)
        self.assertIsNone(parser.logfile_contains_errors())

    def test_failed_job_is_surfaced_by_yaml_parser(self):
        """Test that a failed in-core job (success: false) is surfaced by logfile_contains_errors."""
        out_path = os.path.join(self.job_1.local_path, 'output_fail.yml')
        save_yaml_file(out_path, {'schema_version': 1, 'adapter': 'pyscf',
                                  'success': False, 'error': 'SCF did not converge'})
        parser = YAMLParser(log_file_path=out_path)
        self.assertEqual(parser.logfile_contains_errors(), 'SCF did not converge')

    def test_execute_queue_reports_incore_execution(self):
        """Test that a queued PySCF job runs incore and updates its execution type accordingly.

        ARC only skips the queue status check when ``execution_type == 'incore'``, so leaving it
        as 'queue' makes ARC query a queueing system for a job that never reached one.
        """
        job = PySCFAdapter(execution_type='queue',
                           job_type='sp',
                           project='test_queue_fallback',
                           project_directory=os.path.join(self.project_directory, 'test_queue_fallback'),
                           level=Level(repr='wb97m-v/def2tzvp'),
                           species=[ARCSpecies(label='OH',
                                               xyz={'symbols': ('O', 'H'),
                                                    'isotopes': (16, 1),
                                                    'coords': ((0.0, 0.0, 0.0),
                                                               (0.0, 0.0, 0.9738))},
                                               multiplicity=2)],
                           testing=True)
        with patch.object(PySCFAdapter, 'execute_incore') as mock_execute_incore:
            job.execute_queue()
        mock_execute_incore.assert_called_once()
        self.assertEqual(job.execution_type, 'incore')

    @unittest.skipUnless(HAS_PYSCF, 'PySCF is not installed in this environment')
    def test_normalize_basis(self):
        """Test the basis-set normalization helper in pyscf_script."""
        from arc.job.adapters.scripts.pyscf_script import normalize_basis
        self.assertEqual(normalize_basis('def2tzvp'), 'def2-tzvp')
        self.assertEqual(normalize_basis('def2SVP'), 'def2-svp')
        self.assertEqual(normalize_basis('cc-pvtz'), 'cc-pvtz')

    @unittest.skipUnless(PYSCF_ENV_AVAILABLE, 'The interpreter at PYSCF_PYTHON cannot import PySCF')
    def test_run_freq_executes_in_the_pyscf_environment(self):
        """Test that a freq job really runs run_freq() and writes Cartesian unit norm modes.

        PySCF is absent from the environment ARC itself runs in, so the engine is exercised through
        the in-core subprocess that production uses rather than by importing it here.
        """
        job = PySCFAdapter(execution_type='incore',
                           job_type='freq',
                           project='test_run_freq',
                           project_directory=os.path.join(self.project_directory, 'test_run_freq'),
                           level=Level(repr='b3lyp/sto-3g'),
                           cpu_cores=2,
                           species=[ARCSpecies(label='H2O',
                                               xyz={'symbols': ('O', 'H', 'H'),
                                                    'isotopes': (16, 1, 1),
                                                    'coords': ((0.0, 0.0, 0.1173),
                                                               (0.0, 0.7572, -0.4692),
                                                               (0.0, -0.7572, -0.4692))})])
        job.execute()
        output = read_yaml_file(path=job.local_path_to_output_file)
        self.assertTrue(output['success'], msg=f"PySCF reported: {output.get('error')}")
        self.assertEqual(output['schema_version'], PYSCF_SCHEMA_VERSION)
        self.assertEqual(len(output['freqs']), 3)
        modes = np.array(output['modes'], dtype=np.float64)
        self.assertEqual(modes.shape, (3, 3, 3))
        norms = np.linalg.norm(modes.reshape(modes.shape[0], -1), axis=1)
        self.assertTrue(np.allclose(norms, 1.0, rtol=0, atol=1e-8), msg=f'Got norms {norms}')
        parser = YAMLParser(log_file_path=job.local_path_to_output_file)
        freqs, displacements = parser.parse_normal_mode_displacement()
        self.assertEqual(parser.normal_mode_convention, CARTESIAN_CONVENTION)
        self.assertEqual(displacements.shape, (len(freqs), 3, 3))

    @unittest.skipUnless(PYSCF_ENV_AVAILABLE, 'The interpreter at PYSCF_PYTHON cannot import PySCF')
    def test_run_freq_of_a_monatomic_species(self):
        """Test that a freq job of a species with no vibrational modes reports none and does not fail."""
        job = PySCFAdapter(execution_type='incore',
                           job_type='freq',
                           project='test_run_freq_monatomic',
                           project_directory=os.path.join(self.project_directory, 'test_run_freq_monatomic'),
                           level=Level(repr='b3lyp/sto-3g'),
                           cpu_cores=2,
                           species=[ARCSpecies(label='Ne',
                                               xyz={'symbols': ('Ne',),
                                                    'isotopes': (20,),
                                                    'coords': ((0.0, 0.0, 0.0),)})])
        job.execute()
        output = read_yaml_file(path=job.local_path_to_output_file)
        self.assertTrue(output['success'], msg=f"PySCF reported: {output.get('error')}")
        self.assertEqual(output['freqs'], list())
        self.assertEqual(output['modes'], list())

    @classmethod
    def tearDownClass(cls):
        """
        A function that is run ONCE after all unit tests in this class.
        Delete all project directories created during these unit tests.
        """
        shutil.rmtree(cls.project_directory, ignore_errors=True)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
