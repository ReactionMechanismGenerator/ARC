#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.job.adapters.orca module
Compatible with Orca version 5
"""

import math
import os
import shutil
import tempfile
import unittest

from arc.job.adapter import JobTypeEnum
from arc.job.adapters.gaussian import GaussianAdapter
from arc.job.adapters.orca import (MULTIREFERENCE_METHOD_TOKENS,
                                   ORBITALS_DOWNLOAD_JOB_TYPES,
                                   ORBITALS_GUESS_JOB_TYPES,
                                   SYMMETRY_BREAKING_JOB_TYPES,
                                   OrcaAdapter,
                                   _format_orca_basis,
                                   _format_orca_basis_token,
                                   _format_orca_method,
                                   )
from arc.level import Level
from arc.settings.settings import input_filenames, output_filenames
from arc.species import ARCSpecies
from arc.species.converter import str_to_xyz
from arc.species.vectors import calculate_dihedral_angle


class TestOrcaAdapter(unittest.TestCase):
    """
    Contains unit tests for the OrcaAdapter class.
    """
    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None
        cls.scratch_dir = tempfile.mkdtemp(prefix='arc_test_orca_')
        cls.addClassCleanup(shutil.rmtree, cls.scratch_dir, ignore_errors=True)
        cls.job_1 = OrcaAdapter(execution_type='queue',
                                job_type='sp',
                                level=Level(method='DLPNO-CCSD(T)', basis='def2-tzvp', auxiliary_basis='def2-tzvp/c'),
                                project='test',
                                project_directory=os.path.join(cls.scratch_dir, 'test_OrcaAdapter'),
                                species=[ARCSpecies(label='CH3O',
                                                    xyz="""C       0.03807240    0.00035621   -0.00484242
                                                           O       1.35198769    0.01264937   -0.17195885
                                                           H      -0.33965241   -0.14992727    1.02079480
                                                           H      -0.51702680    0.90828035   -0.29592912
                                                           H      -0.53338088   -0.77135867   -0.54806440""")],
                                testing=True,
                                )
        cls.job_2 = OrcaAdapter(execution_type='queue',
                                job_type='sp',
                                level=Level(method='DLPNO-CCSD(T)', basis='def2-tzvp', auxiliary_basis='def2-tzvp/c',
                                            solvation_method='SMD', solvent='DMSO'),
                                project='test',
                                project_directory=os.path.join(cls.scratch_dir, 'test_OrcaAdapter'),
                                species=[ARCSpecies(label='CH3O',
                                                    xyz="""C       0.03807240    0.00035621   -0.00484242
                                                           O       1.35198769    0.01264937   -0.17195885
                                                           H      -0.33965241   -0.14992727    1.02079480
                                                           H      -0.51702680    0.90828035   -0.29592912
                                                           H      -0.53338088   -0.77135867   -0.54806440""")],
                                testing=True,
                                )
        cls.job_3 = OrcaAdapter(execution_type='queue',
                                job_type='sp',
                                level=Level(method='DLPNO-CCSD(T)', basis='def2-tzvp', auxiliary_basis='def2-tzvp/c',
                                            solvation_method='cpcm', solvent='water'),
                                project='test',
                                project_directory=os.path.join(cls.scratch_dir, 'test_OrcaAdapter'),
                                species=[ARCSpecies(label='CH3O',
                                                    xyz="""C       0.03807240    0.00035621   -0.00484242
                                                           O       1.35198769    0.01264937   -0.17195885
                                                           H      -0.33965241   -0.14992727    1.02079480
                                                           H      -0.51702680    0.90828035   -0.29592912
                                                           H      -0.53338088   -0.77135867   -0.54806440""")],
                                testing=True,
                                )
        cls.job_4 = OrcaAdapter(execution_type='queue',
                                job_type='sp',
                                level=Level(method='MP2_CASSCF_MRCI', basis='aug-cc-pVTZ'),
                                project='test4',
                                project_directory=os.path.join(cls.scratch_dir, 'test_OrcaAdapter'),
                                species=[ARCSpecies(label='CH3O',
                                                    active=(14, 7),
                                                    xyz="""C       0.03807240    0.00035621   -0.00484242
                                                           O       1.35198769    0.01264937   -0.17195885
                                                           H      -0.33965241   -0.14992727    1.02079480
                                                           H      -0.51702680    0.90828035   -0.29592912
                                                           H      -0.53338088   -0.77135867   -0.54806440""")],
                                testing=True,
                                )
    def test_set_cpu_and_mem(self):
        """Test assigning number of cpu's and memory"""
        self.job_1.input_file_memory = None
        self.job_1.submit_script_memory = None
        self.job_1.set_cpu_and_mem()
        self.assertEqual(self.job_1.cpu_cores, 8)

    def test_set_input_file_memory(self):
        """Test setting the input_file_memory argument"""
        expected_memory = math.ceil(14 * 1024 / 8)
        self.assertEqual(self.job_1.input_file_memory, expected_memory)

    def test_write_input_file(self):
        """Test writing Orca input files"""
        self.job_1.write_input_file()
        with open(os.path.join(self.job_1.local_path, input_filenames[self.job_1.job_adapter]), 'r') as f:
            content_1 = f.read()
        job_1_expected_input_file = """!uHF dlpno-ccsd(t) def2-tzvp def2-tzvp/c tightscf normalpno
!sp 

%maxcore 1792
%pal nprocs 8 end

* xyz 0 2
C       0.03807240    0.00035621   -0.00484242
O       1.35198769    0.01264937   -0.17195885
H      -0.33965241   -0.14992727    1.02079480
H      -0.51702680    0.90828035   -0.29592912
H      -0.53338088   -0.77135867   -0.54806440
*

%scf
MaxIter 999
end

"""
        self.assertEqual(content_1, job_1_expected_input_file)

    def test_write_input_file_with_SMD_solvation(self):
        """Test writing ORCA input files with SMD solvation"""
        self.job_2.write_input_file()
        with open(os.path.join(self.job_2.local_path, input_filenames[self.job_2.job_adapter]), 'r') as f:
            content_2 = f.read()
        job_2_expected_input_file = """!uHF dlpno-ccsd(t) def2-tzvp def2-tzvp/c tightscf normalpno
!sp 

%maxcore 1792
%pal nprocs 8 end

* xyz 0 2
C       0.03807240    0.00035621   -0.00484242
O       1.35198769    0.01264937   -0.17195885
H      -0.33965241   -0.14992727    1.02079480
H      -0.51702680    0.90828035   -0.29592912
H      -0.53338088   -0.77135867   -0.54806440
*

%scf
MaxIter 999
end



%cpcm SMD true
      SMDsolvent "dmso"
end

"""
        self.assertEqual(content_2, job_2_expected_input_file)


    def test_write_input_file_with_CPCM_solvation(self):
        """Test writing ORCA input files with CPCM solvation"""
        self.job_3.write_input_file()
        with open(os.path.join(self.job_3.local_path, input_filenames[self.job_3.job_adapter]), 'r') as f:
            content_3 = f.read()
        job_3_expected_input_file = """!uHF dlpno-ccsd(t) def2-tzvp def2-tzvp/c tightscf normalpno
!sp 

%maxcore 1792
%pal nprocs 8 end

* xyz 0 2
C       0.03807240    0.00035621   -0.00484242
O       1.35198769    0.01264937   -0.17195885
H      -0.33965241   -0.14992727    1.02079480
H      -0.51702680    0.90828035   -0.29592912
H      -0.53338088   -0.77135867   -0.54806440
*

%scf
MaxIter 999
end



!CPCM(water)

"""
        self.assertEqual(content_3, job_3_expected_input_file)

    def test_write_input_file_f12_with_cabs(self):
        """F12 sp_level with a cabs basis emits the CABS token on the ! line."""
        job_f12 = OrcaAdapter(execution_type='queue',
                              job_type='sp',
                              level=Level(method='DLPNO-CCSD(T)-F12',
                                          basis='cc-pVTZ-F12',
                                          auxiliary_basis='aug-cc-pVTZ/C',
                                          cabs='cc-pVTZ-F12-CABS'),
                              project='test_f12',
                              project_directory=os.path.join(self.scratch_dir, 'test_OrcaAdapter'),
                              species=[ARCSpecies(label='O_atom', smiles='[O]',
                                                  xyz='O 0.0 0.0 0.0')],
                              testing=True,
                              )
        job_f12.write_input_file()
        with open(os.path.join(job_f12.local_path, input_filenames[job_f12.job_adapter]), 'r') as f:
            content = f.read()
        bang_line = content.splitlines()[0]
        self.assertIn('dlpno-ccsd(t)-f12', bang_line)
        self.assertIn('cc-pvtz-f12', bang_line)
        self.assertIn('aug-cc-pvtz/c', bang_line)
        self.assertIn('cc-pvtz-f12-cabs', bang_line)

    def test_write_input_file_f12_without_cabs_raises(self):
        """F12 sp_level without a cabs basis raises at input-file generation."""
        # _initialize_adapter calls set_files() which calls write_input_file(),
        # so the guard fires during OrcaAdapter construction — wrap the whole
        # thing in assertRaises.
        with self.assertRaises(ValueError):
            OrcaAdapter(execution_type='queue',
                        job_type='sp',
                        level=Level(method='DLPNO-CCSD(T)-F12',
                                    basis='cc-pVTZ-F12',
                                    auxiliary_basis='aug-cc-pVTZ/C'),
                        project='test_f12_bad',
                        project_directory=os.path.join(self.scratch_dir, 'test_OrcaAdapter'),
                        species=[ARCSpecies(label='O_atom', smiles='[O]',
                                            xyz='O 0.0 0.0 0.0')],
                        testing=True,
                        )

    def test_format_orca_method(self):
        """Test ORCA method formatting helper."""
        self.assertEqual(_format_orca_method('wb97xd3'), 'wb97x-d3')
        self.assertEqual(_format_orca_method('wb97xd'), 'wb97xd')
        self.assertEqual(_format_orca_method('B3LYP'), 'B3LYP')

    def test_format_orca_basis_token(self):
        """Test ORCA basis token formatting helper."""
        self.assertEqual(_format_orca_basis_token('def2tzvp'), 'def2-tzvp')
        self.assertEqual(_format_orca_basis_token('def2-TZVP'), 'def2-tzvp')
        self.assertEqual(_format_orca_basis_token('def2tzvp/c'), 'def2-tzvp/c')
        self.assertEqual(_format_orca_basis_token('def2-TZVP/C'), 'def2-tzvp/c')
        self.assertEqual(_format_orca_basis_token('cc-pvtz'), 'cc-pvtz')

    def test_format_orca_basis(self):
        """Test ORCA basis formatting helper."""
        self.assertEqual(_format_orca_basis('def2tzvp'), 'def2-tzvp')
        self.assertEqual(_format_orca_basis('def2-TZVP'), 'def2-tzvp')
        self.assertEqual(_format_orca_basis('def2tzvp/c'), 'def2-tzvp/c')
        self.assertEqual(_format_orca_basis('def2tzvp def2tzvp/c'), 'def2-tzvp def2-tzvp/c')

    def test_write_input_file_mrci(self):
        """Test writing Orca input files"""
        self.job_4.write_input_file()
        with open(os.path.join(self.job_4.local_path, input_filenames[self.job_4.job_adapter]), 'r') as f:
            content_4 = f.read()
        job_4_expected_input_file = """!uHF  aug-cc-pvtz  tightscf
!sp 

%maxcore 1792
%pal nprocs 8 end

* xyz 0 2
C       0.03807240    0.00035621   -0.00484242
O       1.35198769    0.01264937   -0.17195885
H      -0.33965241   -0.14992727    1.02079480
H      -0.51702680    0.90828035   -0.29592912
H      -0.53338088   -0.77135867   -0.54806440
*

%scf
MaxIter 999
end


%mp2
    RI true
end

%casscf
    nel 14
    norb 7
    nroots 1
    maxiter 999
end

%mrci
    citype MRCI
    davidsonopt true
    maxiter 999
end

"""
        self.assertEqual(content_4, job_4_expected_input_file)

    def test_set_files(self):
        """Test setting files"""
        job_1_files_to_upload = [{'file_name': 'submit.sub',
                                  'local': os.path.join(self.job_1.local_path, 'submit.sub'),
                                  'remote': os.path.join(self.job_1.remote_path, 'submit.sub'),
                                  'source': 'path',
                                  'make_x': False},
                                 {'file_name': 'input.in',
                                  'local': os.path.join(self.job_1.local_path, input_filenames[self.job_1.job_adapter]),
                                  'remote': os.path.join(self.job_1.remote_path, input_filenames[self.job_1.job_adapter]),
                                  'source': 'path',
                                  'make_x': False},
                                 ]
        job_1_files_to_download = [{'file_name': 'input.log',
                                    'local': os.path.join(self.job_1.local_path, output_filenames[self.job_1.job_adapter]),
                                    'remote': os.path.join(self.job_1.remote_path, output_filenames[self.job_1.job_adapter]),
                                    'source': 'path',
                                    'make_x': False}]
        self.assertEqual(self.job_1.files_to_upload, job_1_files_to_upload)
        self.assertEqual(self.job_1.files_to_download, job_1_files_to_download)

    def test_dft_grid_regular_opt(self):
        """Test that regular opt job uses defgrid2 for DFT"""
        job_opt = OrcaAdapter(execution_type='queue',
                              job_type='opt',
                              level=Level(method='wb97x-d3', basis='def2-tzvp'),
                              project='test_dft_grid',
                              project_directory=os.path.join(self.scratch_dir, 'test_OrcaAdapter'),
                              species=[ARCSpecies(label='CH3O',
                                                  xyz="""C       0.03807240    0.00035621   -0.00484242
                                                         O       1.35198769    0.01264937   -0.17195885
                                                         H      -0.33965241   -0.14992727    1.02079480
                                                         H      -0.51702680    0.90828035   -0.29592912
                                                         H      -0.53338088   -0.77135867   -0.54806440""")],
                              testing=True,
                              fine=False,
                              )
        job_opt.write_input_file()
        with open(os.path.join(job_opt.local_path, input_filenames[job_opt.job_adapter]), 'r') as f:
            content = f.read()
        self.assertIn('defgrid2', content)
        self.assertNotIn('defgrid3', content)

    def test_dft_grid_fine_opt(self):
        """Test that fine opt job uses defgrid3 for DFT"""
        job_fine_opt = OrcaAdapter(execution_type='queue',
                                   job_type='opt',
                                   level=Level(method='wb97x-d3', basis='def2-tzvp'),
                                   project='test_dft_grid_fine',
                                   project_directory=os.path.join(self.scratch_dir, 'test_OrcaAdapter'),
                                   species=[ARCSpecies(label='CH3O',
                                                       xyz="""C       0.03807240    0.00035621   -0.00484242
                                                              O       1.35198769    0.01264937   -0.17195885
                                                              H      -0.33965241   -0.14992727    1.02079480
                                                              H      -0.51702680    0.90828035   -0.29592912
                                                              H      -0.53338088   -0.77135867   -0.54806440""")],
                                   testing=True,
                                   fine=True,
                                   )
        job_fine_opt.write_input_file()
        with open(os.path.join(job_fine_opt.local_path, input_filenames[job_fine_opt.job_adapter]), 'r') as f:
            content = f.read()
        self.assertIn('defgrid3', content)

    def test_dft_grid_freq(self):
        """Test that freq job uses defgrid3 for DFT"""
        job_freq = OrcaAdapter(execution_type='queue',
                               job_type='freq',
                               level=Level(method='wb97x-d3', basis='def2-tzvp'),
                               project='test_dft_grid_freq',
                               project_directory=os.path.join(self.scratch_dir, 'test_OrcaAdapter'),
                               species=[ARCSpecies(label='CH3O',
                                                   xyz="""C       0.03807240    0.00035621   -0.00484242
                                                          O       1.35198769    0.01264937   -0.17195885
                                                          H      -0.33965241   -0.14992727    1.02079480
                                                          H      -0.51702680    0.90828035   -0.29592912
                                                          H      -0.53338088   -0.77135867   -0.54806440""")],
                               testing=True,
                               fine=False,
                               )
        job_freq.write_input_file()
        with open(os.path.join(job_freq.local_path, input_filenames[job_freq.job_adapter]), 'r') as f:
            content = f.read()
        self.assertIn('defgrid3', content)

    def test_dft_grid_optfreq(self):
        """Test that optfreq job uses defgrid3 for DFT"""
        job_optfreq = OrcaAdapter(execution_type='queue',
                                  job_type='optfreq',
                                  level=Level(method='wb97x-d3', basis='def2-tzvp'),
                                  project='test_dft_grid_optfreq',
                                  project_directory=os.path.join(self.scratch_dir, 'test_OrcaAdapter'),
                                  species=[ARCSpecies(label='CH3O',
                                                      xyz="""C       0.03807240    0.00035621   -0.00484242
                                                             O       1.35198769    0.01264937   -0.17195885
                                                             H      -0.33965241   -0.14992727    1.02079480
                                                             H      -0.51702680    0.90828035   -0.29592912
                                                             H      -0.53338088   -0.77135867   -0.54806440""")],
                                  testing=True,
                                  fine=False,
                                  )
        job_optfreq.write_input_file()
        with open(os.path.join(job_optfreq.local_path, input_filenames[job_optfreq.job_adapter]), 'r') as f:
            content = f.read()
        self.assertIn('defgrid3', content)

    def test_fine_opt_convergence_tightopt(self):
        """Test that fine opt job uses TightOpt convergence for DFT"""
        job_fine_opt = OrcaAdapter(execution_type='queue',
                                   job_type='opt',
                                   level=Level(method='wb97x-d3', basis='def2-tzvp'),
                                   project='test_fine_opt_conv',
                                   project_directory=os.path.join(self.scratch_dir, 'test_OrcaAdapter'),
                                   species=[ARCSpecies(label='CH3O',
                                                       xyz="""C       0.03807240    0.00035621   -0.00484242
                                                              O       1.35198769    0.01264937   -0.17195885
                                                              H      -0.33965241   -0.14992727    1.02079480
                                                              H      -0.51702680    0.90828035   -0.29592912
                                                              H      -0.53338088   -0.77135867   -0.54806440""")],
                                   testing=True,
                                   fine=True,
                                   )
        job_fine_opt.write_input_file()
        with open(os.path.join(job_fine_opt.local_path, input_filenames[job_fine_opt.job_adapter]), 'r') as f:
            content = f.read()
        # TightOpt should be present in fine opt
        self.assertIn('tightopt', content.lower())

    def test_recalc_hess_in_optts(self):
        """Test that OptTS job includes calc_Hess true in %geom block"""
        job_optts = OrcaAdapter(execution_type='queue',
                                job_type='opt',
                                level=Level(method='wb97x-d3', basis='def2-tzvp'),
                                project='test_optts_hess',
                                project_directory=os.path.join(self.scratch_dir, 'test_OrcaAdapter'),
                                species=[ARCSpecies(label='TS_example',
                                                    xyz="""C       0.03807240    0.00035621   -0.00484242
                                                           O       1.35198769    0.01264937   -0.17195885
                                                           H      -0.33965241   -0.14992727    1.02079480
                                                           H      -0.51702680    0.90828035   -0.29592912
                                                           H      -0.53338088   -0.77135867   -0.54806440""",
                                                    is_ts=True)],
                                testing=True,
                                fine=False,
                                )
        job_optts.write_input_file()
        with open(os.path.join(job_optts.local_path, input_filenames[job_optts.job_adapter]), 'r') as f:
            content = f.read()
        # Check that the file contains the %geom block with Calc_Hess and Recalc_Hess
        self.assertIn('%geom', content)
        self.assertIn('Calc_Hess true', content)
        # Check that it's an OptTS job
        self.assertIn('OptTS', content)

    def test_recalc_hess_not_in_regular_opt(self):
        """Test that regular Opt job (non-TS) does NOT include Recalc_Hess block"""
        job_opt_regular = OrcaAdapter(execution_type='queue',
                                      job_type='opt',
                                      level=Level(method='wb97x-d3', basis='def2-tzvp'),
                                      project='test_opt_no_hess',
                                      project_directory=os.path.join(self.scratch_dir, 'test_OrcaAdapter'),
                                      species=[ARCSpecies(label='CH3O',
                                                          xyz="""C       0.03807240    0.00035621   -0.00484242
                                                                 O       1.35198769    0.01264937   -0.17195885
                                                                 H      -0.33965241   -0.14992727    1.02079480
                                                                 H      -0.51702680    0.90828035   -0.29592912
                                                                 H      -0.53338088   -0.77135867   -0.54806440""",
                                                          is_ts=False)],
                                      testing=True,
                                      fine=False,
                                      )
        job_opt_regular.write_input_file()
        with open(os.path.join(job_opt_regular.local_path, input_filenames[job_opt_regular.job_adapter]), 'r') as f:
            content = f.read()
        # Check that it's a regular Opt job, not OptTS
        self.assertIn('!Opt', content)
        self.assertNotIn('OptTS', content)
        # The %geom Calc_Hess block should NOT be present for regular opt
        self.assertNotIn('Calc_Hess true', content)
        self.assertNotIn('Recalc_Hess 5', content)

    def test_writing_input_does_not_pollute_level_args(self):
        """Test that adapter-injected keywords do not reach the Level nor its serialized dict."""
        level = Level(method='b3lyp', basis='def2tzvp', software='orca')
        self.assertNotIn('args', level.as_dict())
        job = OrcaAdapter(execution_type='queue',
                          job_type='opt',
                          level=level,
                          project='test',
                          project_directory=os.path.join(self.scratch_dir, 'test_OrcaAdapter'),
                          species=[ARCSpecies(label='CH3O',
                                              xyz="""C       0.03807240    0.00035621   -0.00484242
                                                     O       1.35198769    0.01264937   -0.17195885
                                                     H      -0.33965241   -0.14992727    1.02079480
                                                     H      -0.51702680    0.90828035   -0.29592912
                                                     H      -0.53338088   -0.77135867   -0.54806440""")],
                          testing=True,
                          )
        job.write_input_file()
        self.assertIn('defgrid2', job.args['keyword'].values())
        self.assertEqual(level.args, {'keyword': dict(), 'block': dict()})
        self.assertNotIn('args', level.as_dict())


class TestOrcaStabilityJob(unittest.TestCase):
    """
    Contains unit tests for the ORCA wavefunction stability analysis job.
    """

    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None
        cls.scratch_dir = tempfile.mkdtemp(prefix='arc_test_orca_')
        cls.addClassCleanup(shutil.rmtree, cls.scratch_dir, ignore_errors=True)
        cls.xyz = """O      -0.00032    0.39999    0.00000
H      -0.76950   -0.19750    0.00000
H       0.76982   -0.20249    0.00000"""
        cls.torsional_xyz = """H       0.86000   -0.03000    0.62000
O       0.10000    0.00000    0.00000
O      -1.10000    0.00000    0.00000
H      -1.30000    0.94000    0.00000"""
        cls.job_type_args = {'directed_scan': {'torsions': [[0, 1, 2, 3]], 'dihedrals': [120.0]},
                             'irc': {'irc_direction': 'forward'},
                             'scan': {'torsions': [[0, 1, 2, 3]]},
                             }

    def _job(self,
             job_type: str = 'stability',
             checkfile: str | None = None,
             species: list | None = None,
             **kwargs,
             ) -> OrcaAdapter:
        """Build a testing ORCA job of the requested type."""
        return OrcaAdapter(execution_type='queue',
                           job_type=job_type,
                           level=Level(method='b3lyp', basis='def2tzvp'),
                           project='test',
                           project_directory=os.path.join(self.scratch_dir, 'test_OrcaStabilityJob'),
                           checkfile=checkfile,
                           species=species if species is not None else [ARCSpecies(label='H2O', xyz=self.xyz)],
                           testing=True,
                           **kwargs,
                           )

    def _torsional_job(self, job_type: str, checkfile: str | None = None) -> OrcaAdapter:
        """Build a testing ORCA job of any job type, on a species carrying a torsion."""
        return self._job(job_type=job_type,
                         checkfile=checkfile,
                         species=[ARCSpecies(label='HOOH', xyz=self.torsional_xyz)],
                         **self.job_type_args.get(job_type, dict()),
                         )

    def _checkfile(self, file_name: str = 'input.gbw', content: str = 'orbitals') -> str:
        """Write a stand-in for a previous job's orbitals file and return its path."""
        directory = tempfile.mkdtemp(prefix='arc_test_orca_gbw_', dir=self.scratch_dir)
        path = os.path.join(directory, file_name)
        with open(path, 'w') as f:
            f.write(content)
        self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
        return path

    def _input_file(self, job: OrcaAdapter) -> str:
        """Write a job's input file and return its content."""
        job.write_input_file()
        with open(os.path.join(job.local_path, input_filenames[job.job_adapter]), 'r') as f:
            return f.read()

    def _plant_orbitals(self, job: OrcaAdapter, content: str = 'orbitals') -> str:
        """Write an orbitals file into a job's own directory and return its path."""
        path = os.path.join(job.local_path, job.check_file_name)
        with open(path, 'w') as f:
            f.write(content)
        self.addCleanup(lambda: os.path.isfile(path) and os.remove(path))
        return path

    def test_write_stability_input_file(self):
        """Test that a stability job is a single point carrying the two STAB keys"""
        job = self._job()
        expected_input_file = f"""!rKS b3lyp def2-tzvp  tightscf defgrid3
!sp 

%maxcore {job.input_file_memory}
%pal nprocs {job.cpu_cores} end

* xyz 0 1
O      -0.00032000    0.39999000    0.00000000
H      -0.76950000   -0.19750000    0.00000000
H       0.76982000   -0.20249000    0.00000000
*

%scf
MaxIter 999
STABPerform true
STABRestartUHFifUnstable true
end

"""
        self.assertEqual(self._input_file(job), expected_input_file)

    def test_the_instability_is_always_followed(self):
        """Test that the restart key is true, which ORCA 6.0.0 needs to survive an instability"""
        content = self._input_file(self._job())
        self.assertIn('STABRestartUHFifUnstable true', content)
        self.assertNotIn('STABRestartUHFifUnstable false', content)

    def test_stability_input_file_reads_the_orbitals_under_test(self):
        """Test that a stability job holding a checkfile reads it as its initial guess"""
        job = self._job(checkfile=self._checkfile())
        expected_input_file = f"""!rKS b3lyp def2-tzvp  tightscf defgrid3
!sp 
!MORead
%moinp "guess.gbw"
%maxcore {job.input_file_memory}
%pal nprocs {job.cpu_cores} end

* xyz 0 1
O      -0.00032000    0.39999000    0.00000000
H      -0.76950000   -0.19750000    0.00000000
H       0.76982000   -0.20249000    0.00000000
*

%scf
MaxIter 999
STABPerform true
STABRestartUHFifUnstable true
end

"""
        self.assertEqual(self._input_file(job), expected_input_file)

    def test_no_guess_is_read_without_a_checkfile(self):
        """Test that a stability job holding no checkfile emits no MORead"""
        content = self._input_file(self._job())
        self.assertNotIn('MORead', content)
        self.assertNotIn('moinp', content)

    def test_a_missing_checkfile_is_not_read(self):
        """Test that a checkfile path that does not exist emits no MORead"""
        content = self._input_file(self._job(checkfile=os.path.join(self.scratch_dir, 'nonexistent.gbw')))
        self.assertNotIn('MORead', content)

    def test_every_guess_reading_job_type_reads_the_guess(self):
        """Test that each job type listed as reading a guess emits MORead when a checkfile is held"""
        for job_type in ORBITALS_GUESS_JOB_TYPES:
            content = self._input_file(self._torsional_job(job_type=job_type, checkfile=self._checkfile()))
            self.assertIn('!MORead', content, msg=f'a {job_type} job emitted no MORead')
            self.assertIn('%moinp "guess.gbw"', content, msg=f'a {job_type} job emitted no moinp')

    def test_only_the_stability_job_analyses_the_wavefunction(self):
        """Test that reading a guess does not make another job type request a stability analysis"""
        for job_type in ['sp', 'opt', 'freq']:
            content = self._input_file(self._job(job_type=job_type, checkfile=self._checkfile()))
            self.assertNotIn('STABPerform', content, msg=f'a {job_type} job emitted STABPerform')

    def test_the_frequency_job_reads_the_optimization_orbitals(self):
        """Test that a freq job holding a checkfile starts its SCF from it"""
        job = self._job(job_type='freq', checkfile=self._checkfile())
        expected_input_file = f"""!rKS b3lyp def2-tzvp  tightscf defgrid3
!Freq 
!MORead
%moinp "guess.gbw"
%maxcore {job.input_file_memory}
%pal nprocs {job.cpu_cores} end

* xyz 0 1
O      -0.00032000    0.39999000    0.00000000
H      -0.76950000   -0.19750000    0.00000000
H       0.76982000   -0.20249000    0.00000000
*

%scf
MaxIter 999
end

"""
        self.assertEqual(self._input_file(job), expected_input_file)

    def test_a_frequency_job_holding_no_checkfile_reads_no_guess(self):
        """Test that a freq job with no checkfile emits neither keyword"""
        content = self._input_file(self._job(job_type='freq'))
        self.assertNotIn('MORead', content)
        self.assertNotIn('moinp', content)

    def test_the_guess_keywords_occupy_their_own_lines(self):
        """Test that MORead and moinp each begin a line rather than running into the keyword above"""
        for job_type in ORBITALS_GUESS_JOB_TYPES:
            content = self._input_file(self._torsional_job(job_type=job_type, checkfile=self._checkfile()))
            self.assertIn('\n!MORead\n%moinp "guess.gbw"\n', content,
                          msg=f'a {job_type} job ran the guess keywords into the line above')

    def test_job_types_that_read_no_guess(self):
        """Test that a job type ORCA is handed no calculation for reads no guess"""
        for job_type in ['composite', 'directed_scan', 'irc', 'orbitals']:
            self.assertNotIn(job_type, ORBITALS_GUESS_JOB_TYPES)
            content = self._input_file(self._torsional_job(job_type=job_type, checkfile=self._checkfile()))
            self.assertNotIn('MORead', content, msg=f'a {job_type} job emitted MORead')
            self.assertNotIn('moinp', content, msg=f'a {job_type} job emitted moinp')

    def test_a_monatomic_species_reads_no_guess(self):
        """Test that a species of one atom is excluded, as it is in Gaussian"""
        job = OrcaAdapter(execution_type='queue',
                          job_type='sp',
                          level=Level(method='b3lyp', basis='def2tzvp'),
                          project='test',
                          project_directory=os.path.join(self.scratch_dir, 'test_OrcaStabilityJob'),
                          checkfile=self._checkfile(),
                          species=[ARCSpecies(label='H', smiles='[H]')],
                          testing=True,
                          )
        self.assertFalse(job.reads_orbital_guess())
        self.assertNotIn('MORead', self._input_file(job))
        self.assertNotIn('guess.gbw', [up_file['file_name'] for up_file in job.files_to_upload])

    def test_the_stability_job_uses_the_frequency_job_grid(self):
        """Test that the stability single point integrates on the grid a frequency job uses"""
        self.assertIn('defgrid3', self._input_file(self._job()))
        self.assertIn('defgrid2', self._input_file(self._job(job_type='sp')))

    def test_orbital_file_names(self):
        """Test that ORCA writes its orbitals to input.gbw and reads a guess from another name"""
        job = self._job()
        self.assertEqual(job.check_file_name, 'input.gbw')
        self.assertEqual(job.guess_file_name, 'guess.gbw')
        self.assertNotEqual(job.check_file_name, job.guess_file_name)
        self.assertEqual(job.local_path_to_check_file, os.path.join(job.local_path, 'input.gbw'))

    def test_set_files_uploads_the_guess_and_downloads_the_orbitals(self):
        """Test that a stability job uploads the orbitals under test and downloads its own"""
        checkfile = self._checkfile()
        job = self._job(checkfile=checkfile)
        self.assertIn({'file_name': 'guess.gbw',
                       'local': checkfile,
                       'remote': os.path.join(job.remote_path, 'guess.gbw'),
                       'source': 'path',
                       'make_x': False},
                      job.files_to_upload)
        self.assertIn({'file_name': 'input.gbw',
                       'local': os.path.join(job.local_path, 'input.gbw'),
                       'remote': os.path.join(job.remote_path, 'input.gbw'),
                       'source': 'path',
                       'make_x': False},
                      job.files_to_download)

    def test_no_guess_is_uploaded_without_a_checkfile(self):
        """Test that a job holding no checkfile uploads no orbitals"""
        job = self._job()
        self.assertNotIn('guess.gbw', [up_file['file_name'] for up_file in job.files_to_upload])
        self.assertIn('input.gbw', [file['file_name'] for file in job.files_to_download])

    def test_every_guess_reading_job_type_uploads_the_guess(self):
        """Test that each job type listed as reading a guess uploads the orbitals it reads"""
        for job_type in ORBITALS_GUESS_JOB_TYPES:
            job = self._torsional_job(job_type=job_type, checkfile=self._checkfile())
            self.assertIn('guess.gbw', [up_file['file_name'] for up_file in job.files_to_upload],
                          msg=f'a {job_type} job uploaded no guess')

    def test_job_types_that_upload_no_guess(self):
        """Test that a job type ORCA is handed no calculation for uploads no orbitals"""
        for job_type in ['composite', 'directed_scan', 'irc', 'orbitals']:
            job = self._torsional_job(job_type=job_type, checkfile=self._checkfile())
            self.assertNotIn('guess.gbw', [up_file['file_name'] for up_file in job.files_to_upload],
                             msg=f'a {job_type} job uploaded a guess')

    def test_the_upload_set_and_the_emission_set_agree(self):
        """Test that over every job type a guess is uploaded for exactly the jobs that read one"""
        job_types = [job_type.value for job_type in JobTypeEnum]
        self.assertTrue(set(ORBITALS_GUESS_JOB_TYPES).issubset(set(job_types)))
        for job_type in job_types:
            job = self._torsional_job(job_type=job_type, checkfile=self._checkfile())
            uploaded = 'guess.gbw' in [up_file['file_name'] for up_file in job.files_to_upload]
            emitted = 'MORead' in self._input_file(job)
            self.assertEqual(uploaded, emitted,
                             msg=f'a {job_type} job uploaded a guess: {uploaded}, emitted MORead: {emitted}')
            self.assertEqual(emitted, job.reads_orbital_guess(),
                             msg=f'a {job_type} job emitted MORead: {emitted}, '
                                 f'reads_orbital_guess: {job.reads_orbital_guess()}')
            self.assertEqual(emitted, job_type in ORBITALS_GUESS_JOB_TYPES,
                             msg=f'a {job_type} job emitted MORead: {emitted}')

    def test_a_job_array_reads_no_guess(self):
        """Test that a job array, whose members share one remote path, reads no guess"""
        job = self._job(job_type='sp', checkfile=self._checkfile())
        self.assertTrue(job.reads_orbital_guess())
        job.iterate_by = ['species']
        self.assertFalse(job.reads_orbital_guess())

    def test_the_orbitals_are_downloaded_only_where_they_are_read(self):
        """Test that only the job types something later reads a .gbw from download one"""
        for job_type in ORBITALS_DOWNLOAD_JOB_TYPES:
            job = self._job(job_type=job_type)
            self.assertIn('input.gbw', [file['file_name'] for file in job.files_to_download],
                          msg=f'a {job_type} job did not download its orbitals')
        for job_type in ['sp', 'freq', 'orbitals']:
            job = self._job(job_type=job_type)
            self.assertNotIn('input.gbw', [file['file_name'] for file in job.files_to_download],
                             msg=f'a {job_type} job downloaded orbitals nothing reads')

    def test_a_checkfile_written_by_another_ess_is_refused(self):
        """Test that a Gaussian check.chk is not uploaded to ORCA as an initial guess"""
        job = self._job(checkfile=self._checkfile(file_name='check.chk'))
        self.assertIsNone(job.checkfile)
        self.assertNotIn('guess.gbw', [up_file['file_name'] for up_file in job.files_to_upload])
        self.assertNotIn('MORead', self._input_file(job))

    def test_an_empty_orbitals_file_is_refused(self):
        """Test that the zero-byte file a failed download leaves behind is not read as a guess"""
        job = self._job(checkfile=self._checkfile(content=''))
        self.assertIsNone(job.checkfile)
        self.assertFalse(job.reads_orbital_guess())
        self.assertNotIn('guess.gbw', [up_file['file_name'] for up_file in job.files_to_upload])
        self.assertNotIn('MORead', self._input_file(job))

    def test_an_orbitals_file_emptied_after_the_job_was_built_is_not_read(self):
        """Test that the guess predicate answers on the file rather than on the path"""
        checkfile = self._checkfile()
        job = self._job(checkfile=checkfile)
        self.assertTrue(job.reads_orbital_guess())
        with open(checkfile, 'w'):
            pass
        self.assertFalse(job.reads_orbital_guess())

    def test_an_empty_orbitals_file_in_a_reused_job_directory_is_not_adopted(self):
        """Test that a zero-byte orbitals file left in the job directory is not picked up"""
        first = self._job()
        planted = self._plant_orbitals(first, content='')
        second = self._job(job_name=first.job_name, job_num=first.job_num)
        self.assertEqual(second.local_path, first.local_path)
        self.assertIsNone(second.checkfile)
        self.assertTrue(os.path.isfile(planted))

    def test_an_orbitals_file_in_a_reused_job_directory_is_adopted(self):
        """Test that the job directory remains a source of orbitals for a species holding none"""
        first = self._job()
        planted = self._plant_orbitals(first)
        second = self._job(job_name=first.job_name, job_num=first.job_num)
        self.assertEqual(second.checkfile, planted)

    def test_a_directed_rotor_gbw_is_still_read(self):
        """Test that the name ARC gives a directed rotor's orbitals is not read as a foreign one"""
        directed = self._checkfile(file_name='directed_rotor_input.gbw')
        job = self._job(checkfile=directed)
        self.assertEqual(job.checkfile, directed)
        self.assertIn('MORead', self._input_file(job))


class TestOrcaBrokenSymmetry(unittest.TestCase):
    """
    Contains unit tests for the ORCA symmetry-breaking directive of an adopted unrestricted reference.
    """

    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None
        cls.scratch_dir = tempfile.mkdtemp(prefix='arc_test_orca_bs_')
        cls.addClassCleanup(shutil.rmtree, cls.scratch_dir, ignore_errors=True)
        cls.xyz = """O      -0.00032    0.39999    0.00000
H      -0.76950   -0.19750    0.00000
H       0.76982   -0.20249    0.00000"""
        cls.torsional_xyz = """H       0.86000   -0.03000    0.62000
O       0.10000    0.00000    0.00000
O      -1.10000    0.00000    0.00000
H      -1.30000    0.94000    0.00000"""
        cls.adopted_verdict = {'verdict': 'external_instability', 'restricted': True,
                               'relaxations': ['RHF -> UHF']}
        cls.job_type_args = {'directed_scan': {'torsions': [[0, 1, 2, 3]], 'dihedrals': [120.0]},
                             'irc': {'irc_direction': 'forward'},
                             'scan': {'torsions': [[0, 1, 2, 3]]},
                             }

    def _species(self,
                 verdict: dict | None = None,
                 multiplicity: int | None = None,
                 xyz: str | None = None,
                 is_ts: bool = True,
                 label: str | None = None,
                 **kwargs,
                 ) -> ARCSpecies:
        """Build a testing species carrying a wavefunction stability verdict."""
        species = ARCSpecies(label=label if label is not None else 'TS0' if is_ts else 'HOH',
                             xyz=xyz if xyz is not None else self.xyz,
                             is_ts=is_ts,
                             multiplicity=multiplicity,
                             **kwargs,
                             )
        species.derived_stability_verdict = verdict if verdict is not None else self.adopted_verdict
        return species

    def _job(self,
             job_type: str = 'sp',
             checkfile: str | None = None,
             species: ARCSpecies | None = None,
             level: Level | None = None,
             **kwargs,
             ) -> OrcaAdapter:
        """Build a testing ORCA job on a species carrying a wavefunction stability verdict."""
        return OrcaAdapter(execution_type='queue',
                           job_type=job_type,
                           level=level if level is not None else Level(method='b3lyp', basis='def2tzvp'),
                           project='test',
                           project_directory=os.path.join(self.scratch_dir, 'test_OrcaBrokenSymmetry'),
                           checkfile=checkfile,
                           species=[species if species is not None
                                    else self._species(xyz=self.torsional_xyz)
                                    if job_type in self.job_type_args else self._species()],
                           testing=True,
                           **self.job_type_args.get(job_type, dict()),
                           **kwargs,
                           )

    def _checkfile(self, file_name: str = 'input.gbw') -> str:
        """Write a stand-in for a previous job's orbitals file and return its path."""
        directory = tempfile.mkdtemp(prefix='arc_test_orca_bs_gbw_', dir=self.scratch_dir)
        path = os.path.join(directory, file_name)
        with open(path, 'w') as f:
            f.write('orbitals')
        self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
        return path

    def _input_file(self, job: OrcaAdapter) -> str:
        """Write a job's input file and return its content."""
        job.write_input_file()
        with open(os.path.join(job.local_path, input_filenames[job.job_adapter]), 'r') as f:
            return f.read()

    def _plant_orbitals(self, job: OrcaAdapter, content: str = 'orbitals') -> str:
        """Write an orbitals file into a job's own directory and return its path."""
        path = os.path.join(job.local_path, job.check_file_name)
        with open(path, 'w') as f:
            f.write(content)
        self.addCleanup(lambda: os.path.isfile(path) and os.remove(path))
        return path

    def test_the_directive_is_emitted_for_an_adopted_reference_with_no_guess(self):
        """Test that a job on an adopted unrestricted reference with no orbitals carries BrokenSym"""
        job = self._job()
        content = self._input_file(job)
        self.assertIn('\n%scf\nMaxIter 999\nBrokenSym 1,1\nend', content)
        self.assertIn('!uKS', content)
        self.assertNotIn('MORead', content)

    def test_the_operands_are_one_broken_pair(self):
        """Test that the operands are the single pair a closed-shell external instability establishes"""
        self.assertEqual(self._job().spin_symmetry_breaking_operands(), (1, 1))

    def test_no_directive_where_a_readable_guess_exists(self):
        """Test that the orbitals of the broken-symmetry solution are read instead of the directive"""
        job = self._job(checkfile=self._checkfile())
        content = self._input_file(job)
        self.assertIn('!MORead', content)
        self.assertNotIn('BrokenSym', content)
        self.assertIsNone(job.spin_symmetry_breaking_operands())

    def test_a_directive_where_the_guess_is_a_foreign_checkfile(self):
        """Test that orbitals ORCA refuses to read leave the directive as the mechanism in play"""
        job = self._job(checkfile=self._checkfile(file_name='check.chk'))
        content = self._input_file(job)
        self.assertIsNone(job.checkfile)
        self.assertNotIn('MORead', content)
        self.assertIn('BrokenSym 1,1', content)

    def test_orbitals_dropped_for_an_adopted_verdict_are_not_re_adopted(self):
        """Test that a species holding no orbitals of its adopted reference reads none from its directory"""
        first = self._job()
        self._plant_orbitals(first)
        second = self._job(job_name=first.job_name, job_num=first.job_num)
        self.assertEqual(second.local_path, first.local_path)
        self.assertIsNone(second.checkfile)
        content = self._input_file(second)
        self.assertNotIn('MORead', content)
        self.assertIn('BrokenSym 1,1', content)

    def test_no_directive_without_a_verdict(self):
        """Test that a species carrying no stability verdict is handed no directive"""
        species = self._species()
        species.derived_stability_verdict = None
        job = self._job(species=species)
        self.assertIsNone(job.spin_symmetry_breaking_operands())
        self.assertNotIn('BrokenSym', self._input_file(job))

    def test_no_directive_for_a_verdict_arc_does_not_act_on(self):
        """Test that only the verdict adoption acts on emits the directive"""
        for verdict in [{'verdict': 'stable', 'restricted': True},
                        {'verdict': 'internal_instability', 'restricted': True},
                        {'verdict': 'external_instability', 'restricted': False},
                        {'verdict': 'external_instability', 'restricted': None},
                        {'verdict': 'unknown', 'restricted': True},
                        ]:
            job = self._job(species=self._species(verdict=verdict))
            self.assertIsNone(job.spin_symmetry_breaking_operands(),
                              msg=f'a job on the verdict {verdict} was handed operands')
            self.assertNotIn('BrokenSym', self._input_file(job),
                             msg=f'a job on the verdict {verdict} emitted BrokenSym')

    def test_no_directive_for_a_species_that_is_not_a_transition_state(self):
        """Test that a verdict ARC reports without acting on it emits no directive"""
        job = self._job(species=self._species(is_ts=False))
        self.assertNotIn('BrokenSym', self._input_file(job))

    def test_no_directive_where_the_user_declared_the_reference(self):
        """Test that a declared number_of_radicals blocks the directive as it blocks the adoption"""
        job = self._job(species=self._species(multiplicity=1, number_of_radicals=2))
        content = self._input_file(job)
        self.assertIn('!uKS', content)
        self.assertNotIn('BrokenSym', content)

    def test_no_directive_for_a_restricted_job(self):
        """Test that a job composing a restricted reference is handed no directive"""
        species = self._species()
        species.derived_stability_verdict = None
        job = self._job(species=species)
        content = self._input_file(job)
        self.assertIn('!rKS', content)
        self.assertNotIn('BrokenSym', content)

    def test_no_directive_for_an_unrestricted_job_of_an_open_shell_species(self):
        """Test that a species unrestricted by its own multiplicity is handed no directive"""
        job = self._job(species=self._species(verdict=self.adopted_verdict, multiplicity=3))
        content = self._input_file(job)
        self.assertIn('!uKS', content)
        self.assertNotIn('BrokenSym', content)
        self.assertIsNone(job.spin_symmetry_breaking_operands())

    def test_no_directive_where_the_job_declared_no_single_unrestricted_reference(self):
        """Test that the directive follows the reference the input declared and not the species alone"""
        job = self._job()
        self.assertEqual(job.spin_symmetry_breaking_operands(), (1, 1))
        for restricted_used in [None, True, [False], [False, False]]:
            job.restricted_used = restricted_used
            self.assertIsNone(job.spin_symmetry_breaking_operands(),
                              msg=f'a job whose reference memo is {restricted_used} was handed operands')

    def test_no_directive_for_a_monatomic_species(self):
        """Test that a species ARC spawns no geometry chain for is handed no directive"""
        job = self._job(species=self._species(xyz='O 0.0 0.0 0.0', multiplicity=1))
        self.assertIsNone(job.spin_symmetry_breaking_operands())
        self.assertNotIn('BrokenSym', self._input_file(job))

    def test_no_directive_for_a_job_array(self):
        """Test that a job array, which writes no input file, is handed no directive"""
        job = self._job()
        self.assertEqual(job.spin_symmetry_breaking_operands(), (1, 1))
        job.iterate_by = ['species']
        self.assertIsNone(job.spin_symmetry_breaking_operands())

    def test_job_types_that_compose_no_seedable_scf(self):
        """Test that a job type ORCA is written no SCF keyword for is handed no directive"""
        for job_type in [job_type.value for job_type in JobTypeEnum
                         if job_type.value not in ORBITALS_GUESS_JOB_TYPES]:
            job = self._job(job_type=job_type)
            self.assertIsNone(job.spin_symmetry_breaking_operands(),
                              msg=f'a {job_type} job was handed operands')

    def test_every_guess_reading_job_type_takes_the_directive(self):
        """Test that each job type reading a guess takes the directive when no guess is held"""
        for job_type in SYMMETRY_BREAKING_JOB_TYPES:
            job = self._job(job_type=job_type)
            self.assertEqual(job.spin_symmetry_breaking_operands(), (1, 1),
                             msg=f'a {job_type} job was handed no operands')
            self.assertIn('BrokenSym 1,1', self._input_file(job),
                          msg=f'a {job_type} job emitted no BrokenSym')

    def test_the_guess_and_the_directive_are_alternatives(self):
        """Test that exactly one of the two mechanisms is written for every job that admits either"""
        for job_type in SYMMETRY_BREAKING_JOB_TYPES:
            for checkfile in [None, self._checkfile()]:
                job = self._job(job_type=job_type, checkfile=checkfile)
                content = self._input_file(job)
                self.assertEqual(['MORead' in content, 'BrokenSym' in content].count(True), 1,
                                 msg=f'a {job_type} job holding checkfile {checkfile} wrote '
                                     f'MORead: {"MORead" in content}, BrokenSym: {"BrokenSym" in content}')

    def test_the_symmetry_breaking_job_types_are_the_guess_reading_ones_without_the_analysis(self):
        """Test that the directive covers the guess-reading job types except the analysis itself"""
        self.assertEqual(set(ORBITALS_GUESS_JOB_TYPES) - set(SYMMETRY_BREAKING_JOB_TYPES), {'stability'})
        self.assertEqual(set(SYMMETRY_BREAKING_JOB_TYPES) - set(ORBITALS_GUESS_JOB_TYPES), set())

    def test_the_analysis_job_takes_no_directive(self):
        """Test that a job analysing a reference is not handed a directive that would replace it"""
        job = self._job(job_type='stability')
        self.assertIsNone(job.spin_symmetry_breaking_operands())
        content = self._input_file(job)
        self.assertIn('STABPerform true', content)
        self.assertNotIn('BrokenSym', content)

    def test_no_directive_without_a_spin_relaxation(self):
        """Test that an external instability of a constraint other than spin takes no directive"""
        for relaxations in [['RHF -> CRHF'], ['RHF -> CRHF', 'RKS -> CRKS'], list(), None]:
            species = self._species(verdict=dict(self.adopted_verdict, relaxations=relaxations))
            job = self._job(species=species)
            self.assertIsNone(job.spin_symmetry_breaking_operands(),
                              msg=f'the relaxations {relaxations} were handed operands')
            self.assertNotIn('BrokenSym', self._input_file(job),
                             msg=f'the relaxations {relaxations} emitted BrokenSym')

    def test_a_spin_relaxation_among_several_takes_the_directive(self):
        """Test that a verdict naming a spin relaxation alongside another one takes the directive"""
        species = self._species(verdict=dict(self.adopted_verdict, relaxations=['RHF -> CRHF', 'RHF -> UHF']))
        self.assertEqual(self._job(species=species).spin_symmetry_breaking_operands(), (1, 1))

    def test_no_directive_for_a_multireference_level(self):
        """Test that a level whose reference space is optimized is handed no directive"""
        for token in MULTIREFERENCE_METHOD_TOKENS:
            job = self._job(level=Level(method=token, basis='def2tzvp'))
            self.assertIsNone(job.spin_symmetry_breaking_operands(),
                              msg=f'a {token} job was handed operands')
            self.assertNotIn('BrokenSym', self._input_file(job),
                             msg=f'a {token} job emitted BrokenSym')

    def test_no_directive_for_a_composition_with_no_pair_to_break(self):
        """Test that a composition of fewer than two electrons is handed no directive"""
        species = self._species(xyz='H 0.00 0.00 0.00\nH 0.00 0.00 0.74')
        species.charge = 2
        job = self._job(species=species)
        self.assertEqual(job.charge, 2)
        self.assertIsNone(job.spin_symmetry_breaking_operands())
        self.assertNotIn('BrokenSym', self._input_file(job))

    def test_no_directive_for_an_electron_count_the_multiplicity_contradicts(self):
        """Test that a composition whose electron count cannot pair off is handed no directive"""
        species = self._species()
        species.charge = 1
        job = self._job(species=species)
        self.assertEqual(job.multiplicity, 1)
        self.assertIsNone(job.spin_symmetry_breaking_operands())
        self.assertNotIn('BrokenSym', self._input_file(job))

    def test_a_two_electron_species_takes_the_directive(self):
        """Test that the smallest composition holding one pair is handed the directive"""
        job = self._job(species=self._species(xyz='H 0.00 0.00 0.00\nH 0.00 0.00 0.74'))
        self.assertEqual(job.spin_symmetry_breaking_operands(), (1, 1))
        self.assertIn('BrokenSym 1,1', self._input_file(job))

    def test_a_charged_species_takes_the_directive_on_its_own_electron_count(self):
        """Test that the electron count the guards read is the charged one"""
        species = self._species()
        species.charge = 2
        job = self._job(species=species)
        self.assertEqual(job.spin_symmetry_breaking_operands(), (1, 1))
        self.assertIn('BrokenSym 1,1', self._input_file(job))

    def test_the_stability_job_of_a_restricted_species_is_unchanged(self):
        """Test that the analysis itself, which runs before any adoption, carries no directive"""
        species = self._species()
        species.derived_stability_verdict = None
        content = self._input_file(self._job(job_type='stability', species=species))
        self.assertIn('\n%scf\nMaxIter 999\nSTABPerform true\nSTABRestartUHFifUnstable true\nend', content)
        self.assertNotIn('BrokenSym', content)


class TestOrcaScanJob(unittest.TestCase):
    """
    Contains unit tests for the ORCA torsion (rotor) scan input file.
    """

    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None
        cls.scratch_dir = tempfile.mkdtemp(prefix='arc_test_orca_scan_')
        cls.addClassCleanup(shutil.rmtree, cls.scratch_dir, ignore_errors=True)
        cls.xyz = """H       0.86000   -0.03000    0.62000
O       0.10000    0.00000    0.00000
O      -1.10000    0.00000    0.00000
H      -1.30000    0.94000    0.00000"""
        cls.torsion = [0, 1, 2, 3]
        cls.dihedral = calculate_dihedral_angle(coords=str_to_xyz(cls.xyz), torsion=cls.torsion)
        cls.ethanol_xyz = """C      -1.16800000    0.27500000    0.00000000
C       0.10600000   -0.55400000    0.00000000
O       1.26300000    0.25900000    0.00000000
H       1.59740256    0.19883055   -0.90125167
H      -2.05100000   -0.37000000    0.00000000
H      -1.19300000    0.91500000    0.88500000
H      -1.19300000    0.91500000   -0.88500000
H       0.14300000   -1.20300000    0.88100000
H       0.14300000   -1.20300000   -0.88100000"""
        cls.ethanol_torsions = [[4, 0, 1, 2], [0, 1, 2, 3]]
        cls.ethanol_dihedrals = [calculate_dihedral_angle(coords=str_to_xyz(cls.ethanol_xyz), torsion=torsion)
                                 for torsion in cls.ethanol_torsions]

    def _job(self,
             torsions: list | None = None,
             scan_res: float = 8.0,
             xyz: str | None = None,
             is_ts: bool = False,
             adapter: type = OrcaAdapter,
             ):
        """Build a testing scan job of the requested adapter over the requested torsions."""
        species = ARCSpecies(label='HOOH', xyz=xyz if xyz is not None else self.xyz, is_ts=is_ts)
        return adapter(execution_type='queue',
                       job_type='scan',
                       level=Level(method='b3lyp', basis='def2tzvp'),
                       project='test',
                       project_directory=os.path.join(self.scratch_dir, 'test_OrcaScanJob'),
                       species=[species],
                       torsions=torsions if torsions is not None else [self.torsion],
                       args={'trsh': {'scan_res': scan_res}},
                       testing=True,
                       )

    def _input_file(self, job) -> str:
        """Write a job's input file and return its content."""
        job.write_input_file()
        with open(os.path.join(job.local_path, input_filenames[job.job_adapter]), 'r') as f:
            return f.read()

    def _scan_lines(self, content: str) -> list:
        """Return the ``D`` lines of the ``%geom Scan`` block of a rendered input file."""
        return [line for line in content.splitlines() if line.startswith('D ')]

    def _scan_arguments(self, line: str) -> tuple:
        """Return the start, the end and the point count of a rendered ``D`` line."""
        start, end, points = line.split('=')[1].split(',')
        return float(start), float(end), float(points)

    def test_write_scan_input_file(self):
        """Test the input file a one-torsion rotor scan writes"""
        job = self._job()
        expected_input_file = f"""!rKS b3lyp def2-tzvp  tightscf defgrid2
!Opt 

%maxcore {job.input_file_memory}
%pal nprocs {job.cpu_cores} end

* xyz 0 1
H       0.86000000   -0.03000000    0.62000000
O       0.10000000    0.00000000    0.00000000
O      -1.10000000    0.00000000    0.00000000
H      -1.30000000    0.94000000    0.00000000
*

%scf
MaxIter 999
end
%geom Scan
D 0 1 2 3 = 92.8, 452.8, 46
end
end

"""
        self.assertEqual(self._input_file(job), expected_input_file)

    def test_the_third_scan_argument_is_a_point_count_not_a_step(self):
        """Test that the third ``Scan`` argument counts points rather than sizing a step"""
        start, end, points = self._scan_arguments(self._scan_lines(self._input_file(self._job()))[0])
        self.assertEqual(points, 46)
        self.assertAlmostEqual(end - start, 360.0, places=6)

    def test_the_scan_starts_at_the_current_dihedral_and_turns_once(self):
        """Test that the scan starts at the species' own dihedral and spans a full rotation"""
        start, end, points = self._scan_arguments(self._scan_lines(self._input_file(self._job()))[0])
        self.assertAlmostEqual(start, self.dihedral, delta=0.05)
        self.assertAlmostEqual(end, self.dihedral + 360.0, delta=0.05)

    def test_the_sweep_runs_in_the_positive_direction(self):
        """Test that the scan ends above the dihedral it starts at"""
        start, end, points = self._scan_arguments(self._scan_lines(self._input_file(self._job()))[0])
        self.assertGreater(end, start)

    def test_the_implied_step_is_the_scan_resolution(self):
        """Test that the point count and the span imply a step of ``scan_res`` degrees"""
        job = self._job()
        start, end, points = self._scan_arguments(self._scan_lines(self._input_file(job))[0])
        self.assertAlmostEqual((end - start) / (points - 1), job.scan_res, places=6)

    def test_the_point_count_follows_the_scan_resolution(self):
        """Test that a troubleshooting resolution changes the point count and not the span"""
        for scan_res, expected_points in [(8.0, 46), (4.0, 91), (2.0, 181), (1.0, 361)]:
            job = self._job(scan_res=scan_res)
            self.assertAlmostEqual(job.scan_res, scan_res, places=6)
            start, end, points = self._scan_arguments(self._scan_lines(self._input_file(job))[0])
            self.assertEqual(points, expected_points)
            self.assertAlmostEqual(end - start, 360.0, places=6)
            self.assertAlmostEqual((end - start) / (points - 1), scan_res, places=6)

    def test_the_point_count_is_rendered_as_an_integer(self):
        """Test that no rendered point count carries a decimal part"""
        for scan_res in [8.0, 5.0, 4.0, 3.0, 2.0, 1.0]:
            line = self._scan_lines(self._input_file(self._job(scan_res=scan_res)))[0]
            self.assertNotIn('.', line.split(',')[-1])

    def test_every_torsion_gets_its_own_dihedral_and_a_full_rotation(self):
        """Test that each ``D`` line of a multi-torsion scan turns about that torsion's own dihedral"""
        content = self._input_file(self._job(torsions=self.ethanol_torsions, xyz=self.ethanol_xyz))
        lines = self._scan_lines(content)
        self.assertEqual(len(lines), 2)
        self.assertNotAlmostEqual(self.ethanol_dihedrals[0], self.ethanol_dihedrals[1], delta=1.0)
        self.assertEqual(lines, ['D 4 0 1 2 = 180.0, 540.0, 46',
                                 'D 0 1 2 3 = 255.0, 615.0, 46'])
        for line, dihedral in zip(lines, self.ethanol_dihedrals):
            start, end, points = self._scan_arguments(line)
            self.assertAlmostEqual(start, dihedral, delta=0.05)
            self.assertAlmostEqual(end - start, 360.0, places=6)
            self.assertEqual(points, 46)

    def test_a_dihedral_above_180_degrees_is_not_folded(self):
        """Test that a starting dihedral ORCA is handed is the 0-360 value, carried to its end"""
        line = self._scan_lines(self._input_file(self._job(torsions=[[0, 1, 2, 3]],
                                                           xyz=self.ethanol_xyz)))[0]
        start, end, points = self._scan_arguments(line)
        self.assertGreater(start, 180.0)
        self.assertAlmostEqual(start, 255.0, delta=0.05)
        self.assertAlmostEqual(end, 615.0, delta=0.05)

    def test_both_the_scan_and_the_geom_block_are_closed(self):
        """Test that the ``Scan`` block and the ``%geom`` block enclosing it are both terminated"""
        for torsions, xyz in [([[0, 1, 2, 3]], None), (self.ethanol_torsions, self.ethanol_xyz)]:
            content = self._input_file(self._job(torsions=torsions, xyz=xyz))
            self.assertIn('\n%geom Scan\n', content)
            self.assertEqual(content.rstrip('\n').split('\n')[-2:], ['end', 'end'])

    def test_a_transition_state_scan_holds_the_same_block(self):
        """Test that the scan of a transition state requests the same full rotation"""
        content = self._input_file(self._job(is_ts=True))
        self.assertIn('!OptTs', content)
        self.assertIn('D 0 1 2 3 = 92.8, 452.8, 46', content)

    def test_the_orca_scan_samples_the_gaussian_angles(self):
        """Test that the scan visits the angles the Gaussian adapter renders for the same torsion"""
        gaussian_line = self._scan_lines(self._input_file(self._job(adapter=GaussianAdapter)))[0]
        gaussian_steps, gaussian_step = int(gaussian_line.split()[-2]), float(gaussian_line.split()[-1])
        start, end, points = self._scan_arguments(self._scan_lines(self._input_file(self._job()))[0])
        self.assertEqual(points, gaussian_steps + 1)
        self.assertAlmostEqual(end - start, gaussian_steps * gaussian_step, places=6)
        self.assertAlmostEqual((end - start) / (points - 1), gaussian_step, places=6)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
