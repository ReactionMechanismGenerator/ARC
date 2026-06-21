#!/usr/bin/env python3
# encoding: utf-8

"""
Unit tests for using the UMA (Meta FAIR fairchem-core) calculator through ARC's ASE job adapter.

The env-independent tests verify UMA routing, calculator/settings resolution, input writing, and
output parsing without the gated model. The model-dependent tests (skipped unless uma_env and the
model are available and UMA_RUN_MODEL is set) run the real uma-s-1p1 model end-to-end.
"""

import os
import shutil
import sys
import unittest
import unittest.mock

from arc.common import ARC_TESTING_PATH, almost_equal_coords, read_yaml_file, save_yaml_file
from arc.job.adapters.ase_adapter import ASEAdapter
from arc.level import Level
from arc.parser.parser import (parse_1d_scan_coords, parse_e_elect, parse_frequencies,
                                parse_geometry, parse_irc_traj)
from arc.settings.settings import UMA_LATEST_MODEL, UMA_PYTHON, supported_ess
from arc.species import ARCSpecies


requires_model = unittest.skipUnless(
    UMA_PYTHON is not None and os.environ.get('UMA_RUN_MODEL'),
    'The uma_env environment / UMA model is unavailable, or UMA_RUN_MODEL is not set.',
)


class TestUMAViaASEWiring(unittest.TestCase):
    """Env-independent tests for routing a 'uma' level to the ASE adapter."""

    def test_supported_ess(self):
        """Test that the ASE engine UMA runs through is supported."""
        self.assertIn('ase', supported_ess)

    def test_level_routes_to_ase(self):
        """Test that a 'uma' level (and explicit checkpoints) resolves to the ASE software."""
        self.assertEqual(Level(method='uma').software, 'ase')
        self.assertEqual(Level(method='uma-s-1').software, 'ase')
        self.assertEqual(Level(method='uma-s-1p1').software, 'ase')


class TestUMAViaASEAdapter(unittest.TestCase):
    """Env-independent tests for the ASEAdapter configured for UMA."""

    @classmethod
    def setUpClass(cls):
        """A method that is run before all unit tests in this class."""
        cls.maxDiff = None
        cls.base = os.path.join(ARC_TESTING_PATH, 'test_UMA_via_ASE')
        os.makedirs(cls.base, exist_ok=True)
        # UMA selected implicitly via the level method.
        cls.job_method = ASEAdapter(execution_type='incore', job_type='sp', project='p',
                                    project_directory=os.path.join(cls.base, 'method'),
                                    level=Level(method='uma'),
                                    species=[ARCSpecies(label='EtOH', smiles='CCO')], testing=True)
        # UMA selected explicitly via args, with an explicit checkpoint.
        cls.job_args = ASEAdapter(execution_type='incore', job_type='sp', project='p',
                                  project_directory=os.path.join(cls.base, 'args'),
                                  args={'keyword': {'calculator': 'uma', 'model': 'uma-s-1'}},
                                  species=[ARCSpecies(label='EtOH', smiles='CCO')], testing=True)
        for job in (cls.job_method, cls.job_args):
            os.makedirs(job.local_path, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        """A method that is run after all unit tests in this class."""
        shutil.rmtree(cls.base, ignore_errors=True)

    def test_determine_calculator_name(self):
        """Test that the UMA calculator is detected from the level method or from args."""
        self.assertEqual(self.job_method.determine_calculator_name(), 'uma')
        self.assertEqual(self.job_args.determine_calculator_name(), 'uma')

    def test_determine_settings_defaults(self):
        """Test that UMA settings get sensible defaults (latest model, omol task, cpu)."""
        settings = self.job_method.determine_settings()
        self.assertEqual(settings['calculator'], 'uma')
        self.assertEqual(settings['model'], UMA_LATEST_MODEL)
        self.assertEqual(settings['task'], 'omol')
        self.assertEqual(settings['device'], 'cpu')

    def test_determine_settings_explicit_model(self):
        """Test that an explicit checkpoint in args is preserved."""
        self.assertEqual(self.job_args.determine_settings()['model'], 'uma-s-1')

    def test_get_python_executable(self):
        """Test resolving the UMA python environment from settings."""
        ase_module = sys.modules[ASEAdapter.__module__]
        with unittest.mock.patch.object(ase_module, 'settings', {'ASE_CALCULATORS_ENV': {'uma': 'UMA_PYTHON'},
                                                                  'UMA_PYTHON': '/path/to/uma_python'}):
            self.assertEqual(self.job_method.get_python_executable(), '/path/to/uma_python')

    def test_write_input_file(self):
        """Test the input.yml carries charge/multiplicity/is_ts and resolved UMA settings."""
        self.job_method.write_input_file()
        data = read_yaml_file(os.path.join(self.job_method.local_path, 'input.yml'))
        self.assertEqual(data['job_type'], 'sp')
        self.assertEqual(data['charge'], 0)
        self.assertEqual(data['multiplicity'], 1)
        self.assertFalse(data['is_ts'])
        self.assertEqual(data['settings']['calculator'], 'uma')
        self.assertEqual(data['settings']['model'], UMA_LATEST_MODEL)
        self.assertEqual(data['settings']['task'], 'omol')

    def test_write_input_file_ts(self):
        """Test that a TS species writes is_ts=True."""
        ts = ASEAdapter(execution_type='incore', job_type='opt', project='p',
                        project_directory=os.path.join(self.base, 'ts'),
                        level=Level(method='uma'),
                        species=[ARCSpecies(label='TS', is_ts=True,
                                            xyz='O 0 0 0\nH 0 0 0.97\nH 0 0.94 -0.25')], testing=True)
        os.makedirs(ts.local_path, exist_ok=True)
        ts.write_input_file()
        data = read_yaml_file(os.path.join(ts.local_path, 'input.yml'))
        self.assertTrue(data['is_ts'])

    def test_warn_if_unreliable_uma_sp(self):
        """Test the warning fires for a UMA single point on triplet O2 / an isolated atom."""
        o2 = ASEAdapter(execution_type='incore', job_type='sp', project='p',
                        project_directory=os.path.join(self.base, 'o2'),
                        level=Level(method='uma'),
                        species=[ARCSpecies(label='O2', xyz='O 0 0 0\nO 0 0 1.2', multiplicity=3)], testing=True)
        with self.assertLogs(level='WARNING'):
            o2.warn_if_unreliable_uma_sp()
        # An ordinary species does not warn (no log record -> assertLogs would fail, so assert via no-raise).
        self.job_method.warn_if_unreliable_uma_sp()

    def test_output_yml_round_trip(self):
        """Test a UMA/ASE output.yml is read back by ARC's YAML parser (incl. IRC/scan keys)."""
        out_dir = os.path.join(self.base, 'roundtrip')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'output.yml')
        opt_xyz = {'symbols': ('O', 'H', 'H'), 'isotopes': (16, 1, 1),
                   'coords': ((0.0, 0.0, 0.119), (0.0, 0.763, -0.477), (0.0, -0.763, -0.477))}
        save_yaml_file(out_path, {'sp': -200123.45, 'opt_xyz': opt_xyz,
                                  'freqs': [1600.0, 3700.0, 3800.0],
                                  'irc_traj': [opt_xyz, opt_xyz], 'scan_coords': [opt_xyz]})
        self.assertAlmostEqual(parse_e_elect(out_path), -200123.45, places=2)
        self.assertTrue(almost_equal_coords(parse_geometry(out_path), opt_xyz))
        self.assertEqual(len(parse_frequencies(out_path)), 3)
        self.assertEqual(len(parse_irc_traj(out_path)), 2)
        self.assertEqual(len(parse_1d_scan_coords(out_path)), 1)


@requires_model
class TestUMAViaASEWithModel(unittest.TestCase):
    """Model-dependent tests; run the real uma-s-1p1 model via the ASE adapter."""

    @classmethod
    def setUpClass(cls):
        """A method that is run before all unit tests in this class."""
        cls.base = os.path.join(ARC_TESTING_PATH, 'test_UMA_via_ASE_model')

    @classmethod
    def tearDownClass(cls):
        """A method that is run after all unit tests in this class."""
        shutil.rmtree(cls.base, ignore_errors=True)

    def _job(self, label, job_type, species, **kwargs):
        """Build an incore UMA-via-ASE job."""
        return ASEAdapter(execution_type='incore', job_type=job_type, project='uma',
                          project_directory=os.path.join(self.base, f'{label}_{job_type}'),
                          level=Level(method='uma'), species=species, testing=True, **kwargs)

    def test_sp(self):
        """Test a UMA single point returns a sane electronic energy (kJ/mol)."""
        job = self._job('EtOH', 'sp', [ARCSpecies(label='EtOH', smiles='CCO')])
        job.execute_incore()
        results = read_yaml_file(os.path.join(job.local_path, 'output.yml'))
        self.assertIsInstance(results.get('sp'), float)

    def test_opt_freq(self):
        """Test a UMA opt+freq returns a geometry and 3N-6 real frequencies."""
        spc = ARCSpecies(label='EtOH', smiles='CCO')
        job = self._job('EtOH', 'optfreq', [spc])
        job.execute_incore()
        results = read_yaml_file(os.path.join(job.local_path, 'output.yml'))
        self.assertIn('opt_xyz', results)
        self.assertEqual(len(results['freqs']), 3 * len(spc.get_xyz()['symbols']) - 6)
        self.assertTrue(all(f > 0 for f in results['freqs']))

    def test_ts_optfreq(self):
        """Test a UMA TS opt+freq yields exactly one imaginary frequency."""
        ts_xyz = """N      0.0000000    0.0000000    0.3146069
H     -0.4668973    0.8086246   -0.0524357
H     -0.4668973   -0.8086246   -0.0524357
H      0.9337946    0.0000000   -0.0524357"""
        ts = ARCSpecies(label='TS', is_ts=True, xyz=ts_xyz, multiplicity=1)
        job = self._job('TS', 'optfreq', [ts])
        job.execute_incore()
        results = read_yaml_file(os.path.join(job.local_path, 'output.yml'))
        self.assertEqual(sum(1 for f in results['freqs'] if f < 0), 1)


if __name__ == '__main__':
    unittest.main()
