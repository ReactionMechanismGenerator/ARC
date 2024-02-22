#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.job.adapters.qchem module
"""

import os
import shutil
import unittest

from arc.common import ARC_PATH
from arc.job.adapters.qchem import QChemAdapter
from arc.level import Level
from arc.settings.settings import input_filenames, output_filenames, servers, submit_filenames
from arc.species import ARCSpecies
import arc.job.trsh as trsh


class TestQChemAdapter(unittest.TestCase):
    """
    Contains unit tests for the QChemAdapter class.
    """
    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None
        cls.job_1 = QChemAdapter(execution_type='incore',
                                    job_type='conformers', # Changed from 'composite' to 'conformers' - No equivalent in QChem AFAIK
                                    level=Level(software='qchem',
                                                method='b3lyp',
                                                basis='def2-TZVP',),
                                    project='test',
                                    project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_QChemAdapter'),
                                    species=[ARCSpecies(label='spc1', xyz=['O 0 0 1'])],
                                    testing=True,
                                    )
        cls.job_2 = QChemAdapter(execution_type='queue',
                                    job_type='opt',
                                    level=Level(method='wb97x-d',
                                                basis='def2-TZVP',
                                                solvation_method='SMD',
                                                solvent='Water'),
                                    project='test',
                                    project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_QChemAdapter'),
                                    species=[ARCSpecies(label='spc1', xyz=['O 0 0 1']),
                                             ARCSpecies(label='spc2', xyz=['O 0 0 2'])],
                                    testing=True,
                                    )
        cls.job_3 = QChemAdapter(execution_type='queue',
                                    job_type='opt',
                                    level=Level(method='wb97x-d',
                                                basis='def2-TZVP',
                                                solvation_method='SMD',
                                                solvent='Water'),
                                    project='test',
                                    project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_QChemAdapter'),
                                    species=[ARCSpecies(label='spc1', xyz=['O 0 0 1'])],
                                    testing=True,
                                    )
        spc_4 = ARCSpecies(label='ethanol', xyz=["""C	1.1658210	-0.4043550	0.0000000
                                                    C	0.0000000	0.5518050	0.0000000
                                                    O	-1.1894600	-0.2141940	0.0000000
                                                    H	-1.9412580	0.3751850	0.0000000
                                                    H	2.1054020	0.1451160	0.0000000
                                                    H	1.1306240	-1.0387850	0.8830320
                                                    H	1.1306240	-1.0387850	-0.8830320
                                                    H	0.0476820	1.1930570	0.8835910
                                                    H	0.0476820	1.1930570	-0.8835910"""],
                           directed_rotors={'brute_force_sp': [[1, 2], [2, 3]]})
        spc_4.determine_rotors()  # also calls initialize_directed_rotors()
        cls.job_4 = QChemAdapter(execution_type='queue',
                                    job_type='scan',
                                    level=Level(method='wb97x-d',
                                                basis='def2-TZVP'),
                                    project='test',
                                    project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_QChemAdapter'),
                                    species=[spc_4],
                                    rotor_index=0,
                                    testing=True,
                                    )
        cls.job_5 = QChemAdapter(execution_type='queue',
                                    job_type='freq',
                                    level=Level(method='wb97x-d',
                                                basis='def2-TZVP'),
                                    project='test',
                                    project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_QChemAdapter'),
                                    species=[ARCSpecies(label='birad singlet',
                                                        xyz=['O 0 0 1'],
                                                        multiplicity=1,
                                                        number_of_radicals=2)],
                                    testing=True,
                                    )
        cls.job_6 = QChemAdapter(execution_type='queue',
                                    job_type='optfreq',
                                    level=Level(method='wb97x-d',
                                                basis='def2-TZVP'),
                                    project='test',
                                    project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_QChemAdapter'),
                                    species=[ARCSpecies(label='anion', xyz=['O 0 0 1'], charge=-1, is_ts=False)],
                                    testing=True,
                                    )
        cls.job_7 = QChemAdapter(execution_type='queue',
                                    job_type='irc',
                                    level=Level(method='wb97x-d',
                                                basis='def2-TZVP'),
                                    project='test',
                                    project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_QChemAdapter'),
                                    species=[ARCSpecies(label='IRC', xyz=['O 0 0 1'], is_ts=True)],
                                    irc_direction='reverse',
                                    testing=True,
                                    )
        cls.job_8 = QChemAdapter(execution_type='queue',
                                    job_type='composite',
                                    level=Level(method='cbs-qb3-paraskevas'),
                                    project='test',
                                    project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_QChemAdapter'),
                                    species=[ARCSpecies(label='spc1', xyz=['O 0 0 1'])],
                                    testing=True,
                                    args={'keyword': {'general': 'IOp(1/12=5,3/44=0)'}},
                                    )
        cls.job_9 = QChemAdapter(execution_type='local',
                            job_type='optfreq',
                            level=Level(method='wb97x-d',
                                        basis='def2-TZVP'),
                            project='test',
                            project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_QChemAdapter'),
                            species=[ARCSpecies(label='anion', xyz=['O 0 0 1'], charge=-1, is_ts=False)],
                            testing=True,
                            )
        # Setting up for ESS troubleshooting input file writing

        label = 'anion'
        level_of_theory = {'method': 'wb97xd'}
        server = 'server1'
        job_type = 'optfreq'
        software = 'qchem'
        fine = True
        memory_gb = 16
        num_heavy_atoms = 2
        ess_trsh_methods = []
        cpu_cores = 8
        
        job_status = {'keywords': ['MaxOptCycles']}
        output_errors, ess_trsh_methods, remove_checkfile, level_of_theory, software, job_type, fine, trsh_keyword, \
            memory, shift, cpu_cores, couldnt_trsh = trsh.trsh_ess_job(label, level_of_theory, server, job_status,
                                                                       job_type, software, fine, memory_gb,
                                                                       num_heavy_atoms, cpu_cores, ess_trsh_methods)
        args = {'keyword': {}, 'block': {}}
        if trsh_keyword:
            args['trsh'] = {'trsh': trsh_keyword}
        cls.job_10 = QChemAdapter(execution_type='local',
                            job_type='optfreq',
                            level=Level(method='wb97x-d',
                                        basis='def2-TZVP'),
                            project='test',
                            project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_QChemAdapter'),
                            species=[ARCSpecies(label='anion', xyz=['O 0 0 1'], charge=-1, is_ts=False)],
                            testing=True,
                            ess_trsh_methods=ess_trsh_methods,
                            args=args,
                            )
    
        job_status = {'keywords': ['SCF']}

        output_errors, ess_trsh_methods, remove_checkfile, level_of_theory, software, job_type, fine, trsh_keyword, \
            memory, shift, cpu_cores, couldnt_trsh = trsh.trsh_ess_job(label, level_of_theory, server, job_status,
                                                                       job_type, software, fine, memory_gb,
                                                                       num_heavy_atoms, cpu_cores, ess_trsh_methods)
        args = {'keyword': {}, 'block': {}}
        if trsh_keyword:
            args['trsh'] = {'trsh': trsh_keyword}
        cls.job_11 = QChemAdapter(execution_type='local',
                            job_type='optfreq',
                            level=Level(method='wb97x-d',
                                        basis='def2-TZVP'),
                            project='test',
                            project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_QChemAdapter'),
                            species=[ARCSpecies(label='anion', xyz=['O 0 0 1'], charge=-1, is_ts=False)],
                            testing=True,
                            ess_trsh_methods=ess_trsh_methods,
                            args=args,
                            )

    def test_set_cpu_and_mem(self):
        """Test assigning number of cpu's and memory"""
        self.job_8.input_file_memory = None
        self.job_8.submit_script_memory = None
        self.job_8.server = 'server2'
        self.job_8.set_cpu_and_mem()
        self.assertEqual(self.job_8.cpu_cores, 8)

    def test_set_input_file_memory(self):
        """
        Test setting the input_file_memory argument
        QChem manages its own memory, so this should be None for the time being
        https://manual.q-chem.com/5.4/CCparallel.html

        A discussion is to be had about better manipulation of assigning memory to QChem jobs
        """
        self.assertEqual(self.job_1.input_file_memory, None)
        self.assertEqual(self.job_2.input_file_memory, None)

    def test_write_input_file(self):
        """Test writing QChem input files"""
        self.job_1.write_input_file()
        with open(os.path.join(self.job_1.local_path, input_filenames[self.job_1.job_adapter]), 'r') as f:
            content_1 = f.read()
        job_1_expected_input_file = """$molecule
0 3
O       0.00000000    0.00000000    1.00000000
$end
$rem
   JOBTYPE       opt
   METHOD        b3lyp
   UNRESTRICTED  TRUE
   BASIS         def2-TZVP
   IQMOL_FCHK    FALSE
$end



"""
        self.assertEqual(content_1, job_1_expected_input_file)

        self.job_3.write_input_file()
        with open(os.path.join(self.job_3.local_path, input_filenames[self.job_3.job_adapter]), 'r') as f:
            content_3 = f.read()
        job_3_expected_input_file = """$molecule
0 3
O       0.00000000    0.00000000    1.00000000
$end
$rem
   JOBTYPE       opt
   METHOD        wb97x-d
   UNRESTRICTED  TRUE
   BASIS         def2-TZVP
   SCF_CONVERGENCE 8
   IQMOL_FCHK    FALSE
$end



"""
        self.assertEqual(content_3, job_3_expected_input_file)

        self.job_4.write_input_file()
        with open(os.path.join(self.job_4.local_path, input_filenames[self.job_4.job_adapter]), 'r') as f:
            content_4 = f.read()
        job_4_expected_input_file = """$molecule
0 1
C       1.16582100   -0.40435500    0.00000000
C       0.00000000    0.55180500    0.00000000
O      -1.18946000   -0.21419400    0.00000000
H      -1.94125800    0.37518500    0.00000000
H       2.10540200    0.14511600    0.00000000
H       1.13062400   -1.03878500    0.88303200
H       1.13062400   -1.03878500   -0.88303200
H       0.04768200    1.19305700    0.88359100
H       0.04768200    1.19305700   -0.88359100
$end
$rem
   JOBTYPE       pes_scan
   METHOD        wb97x-d
   UNRESTRICTED  FALSE
   BASIS         def2-TZVP
   IQMOL_FCHK    FALSE
$end


$scan
tors 5 1 2 3 -180.0 180 8.0
$end


"""
        self.assertEqual(content_4, job_4_expected_input_file)

        self.job_5.write_input_file()
        with open(os.path.join(self.job_5.local_path, input_filenames[self.job_5.job_adapter]), 'r') as f:
            content_5 = f.read()
        job_5_expected_input_file = """$molecule
0 1
O       0.00000000    0.00000000    1.00000000
$end
$rem
   JOBTYPE       freq
   METHOD        wb97x-d
   UNRESTRICTED  TRUE
   BASIS         def2-TZVP
   SCF_CONVERGENCE 8
   IQMOL_FCHK    FALSE
$end



"""
        self.assertEqual(content_5, job_5_expected_input_file)

        self.job_6.write_input_file()
        with open(os.path.join(self.job_6.local_path, input_filenames[self.job_6.job_adapter]), 'r') as f:
            content_6 = f.read()
        job_6_expected_input_file = """$molecule
-1 2
O       0.00000000    0.00000000    1.00000000
$end
$rem
   JOBTYPE       opt
   METHOD        wb97x-d
   UNRESTRICTED  TRUE
   BASIS         def2-TZVP
   IQMOL_FCHK    FALSE
$end



"""
        self.assertEqual(content_6, job_6_expected_input_file)

        self.job_7.write_input_file()
        with open(os.path.join(self.job_7.local_path, input_filenames[self.job_7.job_adapter]), 'r') as f:
            content_7 = f.read()
        job_7_expected_input_file = """$molecule
0 1
O       0.00000000    0.00000000    1.00000000
$end
$rem
   JOBTYPE       freq
   METHOD        wb97x-d
   UNRESTRICTED  FALSE
   BASIS         def2-TZVP
   IQMOL_FCHK    FALSE
$end


@@@
$molecule
read
$end
$rem
   JOBTYPE       rpath
   BASIS        def2-TZVP
   METHOD        wb97x-d
   RPATH_DIRECTION -1
   RPATH_MAX_CYCLES 20
   RPATH_MAX_STEPSIZE 150
   RPATH_TOL_DISPLACEMENT 5000
   RPATH_PRINT 2
   SCF_GUESS     read
$end



"""
        self.assertEqual(content_7, job_7_expected_input_file)

    def test_set_files(self):
        """Test setting files"""
        job_3_files_to_upload = [{'file_name': 'submit.sub',
                                  'local': os.path.join(self.job_3.local_path, submit_filenames[servers[self.job_3.server]['cluster_soft']]),
                                  'remote': os.path.join(self.job_3.remote_path, submit_filenames[servers[self.job_3.server]['cluster_soft']]),
                                  'make_x': False,
                                  'source': 'path'},
                                 {'file_name': 'input.in',
                                  'local': os.path.join(self.job_3.local_path, input_filenames[self.job_3.job_adapter]),
                                  'remote': os.path.join(self.job_3.remote_path, input_filenames[self.job_3.job_adapter]),
                                  'source': 'path',
                                  'make_x': False}]
        job_3_files_to_download = [{'file_name': 'output.out',
                                    'local': os.path.join(self.job_3.local_path, output_filenames[self.job_3.job_adapter]),
                                    'remote': os.path.join(self.job_3.remote_path, output_filenames[self.job_3.job_adapter]),
                                    'source': 'path',
                                    'make_x': False}]
        self.assertEqual(self.job_3.files_to_upload, job_3_files_to_upload)
        self.assertEqual(self.job_3.files_to_download, job_3_files_to_download)

    def test_set_files_for_pipe(self):
        """Test setting files for a pipe job"""
        job_2_files_to_upload = [{'file_name': 'submit.sub',
                                  'local': os.path.join(self.job_2.local_path, 'submit.sub'),
                                  'remote': os.path.join(self.job_2.remote_path, 'submit.sub'),
                                  'source': 'path',
                                  'make_x': False},
                                 {'file_name': 'data.hdf5',
                                  'local': os.path.join(self.job_2.local_path, 'data.hdf5'),
                                  'remote': os.path.join(self.job_2.remote_path, 'data.hdf5'),
                                  'source': 'path',
                                  'make_x': False}]
        job_2_files_to_download = [{'file_name': 'data.hdf5',
                                    'local': os.path.join(self.job_2.local_path, 'data.hdf5'),
                                    'remote': os.path.join(self.job_2.remote_path, 'data.hdf5'),
                                    'source': 'path',
                                    'make_x': False}]
        self.assertEqual(self.job_2.files_to_upload, job_2_files_to_upload)
        self.assertEqual(self.job_2.files_to_download, job_2_files_to_download)

    def test_QChemAdapter_def2tzvp(self):
        """Test a QChem job using def2-tzvp"""
        self.assertEqual(self.job_9.level.basis, 'def2-tzvp')
    
    def test_trsh_write_input_file(self):
        """
        QChem troubleshooting input file writing. Currently there are only few situations where we attempt troubleshooting. This is still under development.
        1. When the job status contains 'MaxOptCycles' in the output file
        2. When the job status contains 'SCF' in the output file
        """
        
        self.job_10.write_input_file()
        with open(os.path.join(self.job_10.local_path, input_filenames[self.job_10.job_adapter]), 'r') as f:
            content_10 = f.read()
        job_10_expected_input_file = """$molecule
-1 2
O       0.00000000    0.00000000    1.00000000
$end
$rem
   JOBTYPE       opt
   METHOD        wb97x-d
   UNRESTRICTED  TRUE
   BASIS         def2-TZVP 
   SYM_IGNORE    TRUE
   GEOM_OPT_MAX_CYCLES 250
   IQMOL_FCHK    FALSE
$end



"""
        self.assertEqual(content_10, job_10_expected_input_file)
        
        self.job_11.write_input_file()
        with open(os.path.join(self.job_11.local_path, input_filenames[self.job_11.job_adapter]), 'r') as f:
            content_11 = f.read()
        job_11_expected_input_file = """$molecule
-1 2
O       0.00000000    0.00000000    1.00000000
$end
$rem
   JOBTYPE       opt
   METHOD        wb97x-d
   UNRESTRICTED  TRUE
   BASIS         def2-TZVP
   SCF_ALGORITHM DIIS_GDM 
   MAX_SCF_CYCLES 1000
   GEOM_OPT_MAX_CYCLES 100
   IQMOL_FCHK    FALSE
$end



"""
        self.assertEqual(content_11, job_11_expected_input_file)

    @classmethod
    def tearDownClass(cls):
        """
        A function that is run ONCE after all unit tests in this class.
        Delete all project directories created during these unit tests.
        """
        shutil.rmtree(os.path.join(ARC_PATH, 'arc', 'testing', 'test_QChemAdapter'), ignore_errors=True)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
