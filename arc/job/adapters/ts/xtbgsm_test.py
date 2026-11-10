#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.job.adapters.ts.xtb_gsm module
"""

import os
import re
import shutil
import tarfile
import tempfile
import unittest
from unittest.mock import PropertyMock, patch

from arc.common import ARC_TESTING_PATH
from arc.job.adapter import JobAdapter
from arc.job.adapters.ts.xtb_gsm import xTBGSMAdapter
from arc.level import Level
from arc.parser.parser import parse_trajectory
from arc.reaction import ARCReaction
from arc.species.species import ARCSpecies


class TestxTBGSMAdapter(unittest.TestCase):
    """
    Contains unit tests for the xTBGSMAdapter class.
    """

    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None
        worker = os.environ.get('PYTEST_XDIST_WORKER', 'local')
        for i in range(10):
            cls.addClassCleanup(shutil.rmtree, os.path.join(ARC_TESTING_PATH,
                                                           f'test_xTBAGSMdapter_{worker}_{i}'),
                                ignore_errors=True)
        cls.job_1 = xTBGSMAdapter(project='test_1',
                                  job_type='tsg',
                                  server='local',
                                  project_directory=os.path.join(ARC_TESTING_PATH,
                                                                 f'test_xTBAGSMdapter_{worker}_1'),
                                  reactions=[ARCReaction(r_species=[ARCSpecies(label='HNO', smiles='N=O')],
                                                         p_species=[ARCSpecies(label='HON', smiles='[N-]=[OH+]')])],
                                  )
        cls.job_2 = xTBGSMAdapter(execution_type='incore',
                                  project='test_2',
                                  job_type='tsg',
                                  level=Level(method='xtb',  # These settings make the test converge in <1 min instead of ~25 min
                                              args={'keyword': {'max_opt_iters': 10,
                                                                'conv_tol': 0.5,
                                                                'add_node_tol': 0.5,
                                                                'final_opt': 10,
                                                                'nnodes': 9}}),
                                  project_directory=os.path.join(ARC_TESTING_PATH,
                                                                 f'test_xTBAGSMdapter_{worker}_2'),
                                  reactions=[ARCReaction(r_species=[ARCSpecies(label='HNO', smiles='N=O')],
                                                         p_species=[ARCSpecies(label='HON', smiles='[N-]=[OH+]')])],
                                  )
        cls.job_2.reactions[0].ts_species = ARCSpecies(label='TS2', is_ts=True)

    def test_set_files(self):
        """Test setting files"""
        job_1_files_to_upload = [{'file_name': 'submit.sub',
                                  'local': os.path.join(self.job_1.local_path, 'submit.sub'),
                                  'remote': os.path.join(self.job_1.remote_path, 'submit.sub'),
                                  'source': 'path',
                                  'make_x': False},
                                 {'file_name': 'initial0000.xyz',
                                  'local': os.path.join(self.job_1.local_path, 'scratch', 'initial0000.xyz'),
                                  'remote': os.path.join(self.job_1.remote_path, 'scratch', 'initial0000.xyz'),
                                  'source': 'path',
                                  'make_x': False},
                                 {'file_name': 'gsm.orca',
                                  'local': os.path.join(self.job_1.local_path, 'gsm.orca'),
                                  'remote': os.path.join(self.job_1.remote_path, 'gsm.orca'),
                                  'source': 'path',
                                  'make_x': True},
                                 {'file_name': 'gsm.orca.bin',
                                  'local': os.path.join(self.job_1.xtb_gsm_scripts_path, 'gsm.orca'),
                                  'remote': os.path.join(self.job_1.remote_path, 'gsm.orca.bin'),
                                  'source': 'path',
                                  'make_x': True},
                                 {'file_name': 'inpfileq',
                                  'local': os.path.join(self.job_1.xtb_gsm_scripts_path, 'inpfileq'),
                                  'remote': os.path.join(self.job_1.remote_path, 'inpfileq'),
                                  'source': 'path',
                                  'make_x': False},
                                 {'file_name': 'ograd',
                                  'local': self.job_1.ograd_path,
                                  'remote': os.path.join(self.job_1.remote_path, 'ograd'),
                                  'source': 'path', 'make_x': True},
                                 {'file_name': 'tm2orca.py',
                                  'local': os.path.join(self.job_1.xtb_gsm_scripts_path, 'tm2orca.py'),
                                  'remote': os.path.join(self.job_1.remote_path, 'tm2orca.py'),
                                  'source': 'path',
                                  'make_x': True}]
        job_1_files_to_download = [{'file_name': 'gsm_evidence.tar.gz',
                                    'local': os.path.join(self.job_1.local_path, 'gsm_evidence.tar.gz'),
                                    'remote': os.path.join(self.job_1.remote_path, 'gsm_evidence.tar.gz'),
                                    'source': 'path', 'make_x': False},
                                   {'file_name': 'stringfile.xyz0000',
                                    'local': os.path.join(self.job_1.local_path, 'stringfile.xyz0000'),
                                    'remote': os.path.join(self.job_1.remote_path, 'stringfile.xyz0000'),
                                    'source': 'path', 'make_x': False}]
        self.assertEqual(self.job_1.files_to_upload, job_1_files_to_upload)
        self.assertEqual(self.job_1.files_to_download, job_1_files_to_download)

        self.assertEqual(self.job_2.files_to_upload, list())
        self.assertEqual(self.job_2.files_to_download, list())

    def test_queue_wrapper_archives_evidence_and_preserves_submit_contract(self):
        """The legacy ``./gsm.orca`` command now invokes a binary and archives on exit."""
        with open(self.job_1.gsm_orca_path) as handle:
            wrapper = handle.read()
        self.assertIn('trap collect_gsm_evidence EXIT', wrapper)
        self.assertIn('./gsm.orca.bin', wrapper)
        self.assertIn('gsm_evidence.tar.gz', wrapper)
        self.assertIn('gsm_node_outputs', wrapper)
        self.assertTrue(os.access(self.job_1.gsm_orca_path, os.X_OK))

    def test_extract_downloaded_evidence_archive(self):
        """A queue evidence archive restores the stringfile and per-node files."""
        os.makedirs(self.job_1.gsm_node_outputs_path, exist_ok=True)
        node_energy = os.path.join(self.job_1.gsm_node_outputs_path, 'node_1.energy')
        with open(self.job_1.stringfile_path, 'w') as handle:
            handle.write('string trajectory')
        with open(node_energy, 'w') as handle:
            handle.write('-1.23')
        with tarfile.open(self.job_1.gsm_evidence_archive_path, 'w:gz') as archive:
            archive.add(self.job_1.stringfile_path, arcname='stringfile.xyz0000')
            archive.add(self.job_1.gsm_node_outputs_path, arcname='gsm_node_outputs')
        os.unlink(self.job_1.stringfile_path)
        shutil.rmtree(self.job_1.gsm_node_outputs_path)

        self.assertTrue(self.job_1._extract_gsm_evidence_archive())
        self.assertTrue(os.path.isfile(self.job_1.stringfile_path))
        self.assertTrue(os.path.isfile(node_energy))

    def test_missing_archive_preserves_legacy_stringfile(self):
        """Old queued runs with only a stringfile remain consumable."""
        if os.path.isfile(self.job_1.gsm_evidence_archive_path):
            os.unlink(self.job_1.gsm_evidence_archive_path)
        with open(self.job_1.stringfile_path, 'w') as handle:
            handle.write('legacy trajectory')
        self.assertFalse(self.job_1._extract_gsm_evidence_archive())
        self.assertTrue(os.path.isfile(self.job_1.stringfile_path))

    @patch.object(xTBGSMAdapter, '_extract_gsm_evidence_archive')
    @patch.object(JobAdapter, 'download_files')
    def test_queue_download_lifecycle_extracts_declared_archive(self, base_download, extract_archive):
        """Queued completion downloads declared files before extracting node evidence."""
        self.job_1.download_files()
        base_download.assert_called_once_with()
        extract_archive.assert_called_once_with()

    def test_archive_extraction_rejects_path_traversal(self):
        """Downloaded archives cannot write outside the local job directory."""
        source = os.path.join(self.job_1.local_path, 'unsafe_source')
        with open(source, 'w') as handle:
            handle.write('unsafe')
        with tarfile.open(self.job_1.gsm_evidence_archive_path, 'w:gz') as archive:
            archive.add(source, arcname='../gsm_node_outputs/escaped.energy')
        with self.assertLogs('arc', level='WARNING'):
            self.assertFalse(self.job_1._extract_gsm_evidence_archive())
        self.assertFalse(os.path.exists(os.path.join(os.path.dirname(self.job_1.local_path),
                                                    'gsm_node_outputs', 'escaped.energy')))

    def test_write_input_file(self):
        """Test writing the initial0000.xyz file"""
        self.job_1.write_input_file()
        expected_string = """3

N      -0.51560854    0.35498613    0.00000000
O       0.53496586   -0.29425836    0.00000000
H      -1.32625080   -0.26220791    0.00000000
3

N       0.74809511   -0.26339075    0.00000000
O      -0.57624878    0.25974119    0.00000000
H      -1.24880926   -0.46263779    0.00000000
"""
        with open(self.job_1.scratch_initial0000_path, 'r') as f:
            actual_string = f.read()
        self.assertEqual(actual_string, expected_string)

    def test_incore_scripts_are_executable(self):
        """Test executability of scripts invoked directly by the GSM driver."""
        self.job_2.write_input_file()
        self.assertTrue(os.access(self.job_2.gsm_orca_path, os.X_OK))
        self.assertTrue(os.access(self.job_2.ograd_path, os.X_OK))
        self.assertTrue(os.access(self.job_2.tm2orca_path, os.X_OK))

    def test_ograd_preserves_node_artifacts(self):
        """Test that the gradient wrapper preserves raw per-node xTB results."""
        with open(os.path.join(self.job_1.xtb_gsm_scripts_path, 'ograd'), 'r') as f:
            ograd = f.read()
        self.assertIn('gsm_node_outputs', ograd)
        self.assertIn('${node_label}.energy', ograd)
        self.assertIn('${node_label}.gradient', ograd)
        self.assertIn('${node_label}.xtbout', ograd)

    def _staged_ograd(self, r_species, p_species) -> str:
        """Stage the ograd of a GSM job for the given reaction and return its text."""
        project_directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, project_directory, ignore_errors=True)
        job = xTBGSMAdapter(project='test_ograd',
                            job_type='tsg',
                            server='local',
                            project_directory=project_directory,
                            reactions=[ARCReaction(r_species=r_species, p_species=p_species)],
                            )
        job.write_input_file()
        with open(job.ograd_path, 'r') as f:
            return f.read()

    def test_an_unknown_multiplicity_or_charge_refuses_to_stage_the_job(self):
        """Test that no input file, inpfileq or ograd is staged, and no job is built, for a reaction whose spin or charge is unknown"""
        for attribute in ('multiplicity', 'charge'):
            with self.subTest(attribute=attribute):
                project_directory = tempfile.mkdtemp()
                self.addCleanup(shutil.rmtree, project_directory, ignore_errors=True)
                reaction = ARCReaction(r_species=[ARCSpecies(label='HNO', smiles='N=O')],
                                       p_species=[ARCSpecies(label='HON', smiles='[N-]=[OH+]')])
                with patch.object(ARCReaction, attribute, new_callable=PropertyMock, return_value=None), \
                        self.assertRaises(ValueError):
                    xTBGSMAdapter(project='test_unknown', job_type='tsg', server='local',
                                  project_directory=project_directory, reactions=[reaction])
                staged = [name for _, _, names in os.walk(project_directory) for name in names]
                for name in ('ograd', 'initial0000.xyz', 'inpfileq'):
                    self.assertNotIn(name, staged)

    def test_ograd_neutral_closed_shell(self):
        """A neutral singlet reaction stages charge 0, no unpaired electrons, and an explicit GFN2-xTB."""
        ograd = self._staged_ograd([ARCSpecies(label='HNO', smiles='N=O')],
                                   [ARCSpecies(label='HON', smiles='[N-]=[OH+]')])
        xtb_line = next(line for line in ograd.splitlines() if line.startswith('xtb '))
        self.assertIn('--gfn 2', xtb_line)
        self.assertIn('--chrg 0', xtb_line)
        self.assertIn('--uhf 0', xtb_line)
        self.assertNotIn('@', ograd)

    def test_ograd_anion(self):
        """An anionic reaction stages the net charge of the reactants."""
        xyz = {'symbols': ('O', 'H'), 'isotopes': (16, 1), 'coords': ((0.0, 0.0, 0.0), (0.0, 0.0, 0.97))}
        with patch.object(ARCReaction, 'get_reactants_xyz', return_value=xyz), \
                patch.object(ARCReaction, 'get_products_xyz', return_value=xyz):
            ograd = self._staged_ograd([ARCSpecies(label='OH-', smiles='[OH-]', charge=-1),
                                        ARCSpecies(label='CH3OH', smiles='CO')],
                                       [ARCSpecies(label='H2O', smiles='O'),
                                        ARCSpecies(label='CH3O-', smiles='C[O-]', charge=-1)])
        self.assertRegex(ograd, r'xtb \S+ --grad --gfn 2 --chrg -1 --uhf 0 ')

    def test_ograd_doublet(self):
        """A doublet reaction stages one unpaired electron (multiplicity minus one)."""
        ograd = self._staged_ograd([ARCSpecies(label='H', smiles='[H]'), ARCSpecies(label='CH4', smiles='C')],
                                   [ARCSpecies(label='H2', smiles='[H][H]'), ARCSpecies(label='CH3', smiles='[CH3]')])
        self.assertRegex(ograd, r'xtb \S+ --grad --gfn 2 --chrg 0 --uhf 1 ')
        self.assertIsNone(re.search(r'--uhf 0', ograd))

    def test_set_inpfileq_keywords(self):
        """Test the set_inpfileq_keywords() method."""
        keywords = self.job_1.set_inpfileq_keywords()
        self.assertEqual(keywords['conv_tol'], '0.0005')
        self.assertEqual(keywords['growth_direction'], '0')
        self.assertEqual(keywords['initial_opt'], '0')
        self.assertEqual(keywords['final_opt'], '150')
        self.assertEqual(keywords['nnodes'], '15')

        keywords = self.job_2.set_inpfileq_keywords()
        self.assertEqual(keywords['conv_tol'], '0.5')
        self.assertEqual(keywords['max_opt_iters'], '10')
        self.assertEqual(keywords['add_node_tol'], '0.5')
        self.assertEqual(keywords['growth_direction'], '0')
        self.assertEqual(keywords['initial_opt'], '0')
        self.assertEqual(keywords['int_thresh'], '2.0')
        self.assertEqual(keywords['final_opt'], '10')
        self.assertEqual(keywords['nnodes'], '9')

    def test_execute_incore(self):
        """Test executing DE-GSM via xTB"""
        self.job_2.execute()
        traj = parse_trajectory(self.job_2.stringfile_path)
        self.assertEqual(len(traj), 9)

if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
