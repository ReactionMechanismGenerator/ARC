#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.job.adapters.gaussian module
"""

import math
import os
import shutil
import unittest

from arc.common import ARC_PATH
from arc.job.adapters.gaussian import GaussianAdapter
from arc.level import Level
from arc.settings.settings import input_filenames, output_filenames, servers, submit_filenames
from arc.species import ARCSpecies
import arc.job.trsh as trsh

class TestGaussianAdapter(unittest.TestCase):
    """
    Contains unit tests for the GaussianAdapter class.
    """
    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None
        cls.job_1 = GaussianAdapter(execution_type='incore',
                                    job_type='composite',
                                    level=Level(method='cbs-qb3-paraskevas'),
                                    project='test',
                                    project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_GaussianAdapter'),
                                    species=[ARCSpecies(label='spc1', xyz=['O 0 0 1'])],
                                    testing=True,
                                    args={'keyword': {'general': 'IOp(1/12=5,3/44=0)'}},
                                    )
        cls.job_2 = GaussianAdapter(execution_type='queue',
                                    job_type='opt',
                                    level=Level(method='wb97xd',
                                                basis='def2-TZVP',
                                                solvation_method='SMD',
                                                solvent='Water'),
                                    project='test',
                                    project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_GaussianAdapter'),
                                    species=[ARCSpecies(label='spc1', xyz=['O 0 0 1']),
                                             ARCSpecies(label='spc2', xyz=['O 0 0 2'])],
                                    testing=True,
                                    )
        cls.job_3 = GaussianAdapter(execution_type='queue',
                                    job_type='opt',
                                    level=Level(method='wb97xd',
                                                basis='def2-TZVP',
                                                solvation_method='SMD',
                                                solvent='Water'),
                                    project='test',
                                    project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_GaussianAdapter'),
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
        cls.job_4 = GaussianAdapter(execution_type='queue',
                                    job_type='scan',
                                    level=Level(method='wb97xd',
                                                basis='def2-TZVP'),
                                    project='test',
                                    project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_GaussianAdapter'),
                                    species=[spc_4],
                                    rotor_index=0,
                                    testing=True,
                                    args={'block': {'general': 'additional\ngaussian\nblock'}},
                                    )
        cls.job_5 = GaussianAdapter(execution_type='queue',
                                    job_type='freq',
                                    level=Level(method='wb97xd',
                                                basis='def2-TZVP'),
                                    project='test',
                                    project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_GaussianAdapter'),
                                    species=[ARCSpecies(label='birad singlet',
                                                        xyz=['O 0 0 1'],
                                                        multiplicity=1,
                                                        number_of_radicals=2)],
                                    testing=True,
                                    )
        cls.job_6 = GaussianAdapter(execution_type='queue',
                                    job_type='optfreq',
                                    level=Level(method='wb97xd',
                                                basis='def2-TZVP'),
                                    project='test',
                                    project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_GaussianAdapter'),
                                    species=[ARCSpecies(label='anion', xyz=['O 0 0 1'], charge=-1, is_ts=False)],
                                    testing=True,
                                    )
        cls.job_7 = GaussianAdapter(execution_type='queue',
                                    job_type='irc',
                                    level=Level(method='wb97xd',
                                                basis='def2-TZVP'),
                                    project='test',
                                    project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_GaussianAdapter'),
                                    species=[ARCSpecies(label='IRC', xyz=['O 0 0 1'], is_ts=True)],
                                    irc_direction='reverse',
                                    testing=True,
                                    )
        cls.job_8 = GaussianAdapter(execution_type='queue',
                                    job_type='composite',
                                    level=Level(method='cbs-qb3-paraskevas'),
                                    project='test',
                                    project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_GaussianAdapter'),
                                    species=[ARCSpecies(label='spc1', xyz=['O 0 0 1'])],
                                    testing=True,
                                    args={'keyword': {'general': 'IOp(1/12=5,3/44=0)'}},
                                    )
        cls.job_9 = GaussianAdapter(execution_type='local',
                            job_type='optfreq',
                            level=Level(method='wb97xd',
                                        basis='def2-TZVP'),
                            project='test',
                            project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_GaussianAdapter'),
                            species=[ARCSpecies(label='anion', xyz=['O 0 0 1'], charge=-1, is_ts=False)],
                            testing=True,
                            )
        cls.job_10 = GaussianAdapter(execution_type='local',
                            job_type='optfreq',
                            level=Level(method='wb97xd'),
                            fine=True,
                            project='test',
                            project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_GaussianAdapter'),
                            species=[ARCSpecies(label='anion', xyz=['O 0 0 1'], charge=-1, is_ts=False)],
                            testing=True,
                            args={'trsh': {'trsh': ['int=(Acc2E=14)']}},
                            )

        # Setting up for ESS troubleshooting input file writing

        label = 'anion'
        level_of_theory = {'method': 'wb97xd'}
        server = 'server1'
        job_type = 'optfreq'
        software = 'gaussian'
        fine = True
        memory_gb = 16
        num_heavy_atoms = 2
        ess_trsh_methods = ['int=(Acc2E=14)']
        cpu_cores = 8

        # Gaussian: Checkfile error
        job_status = {'keywords': ['CheckFile']}
        output_errors, ess_trsh_methods, remove_checkfile, level_of_theory, software, job_type, fine, trsh_keyword, \
            memory, shift, cpu_cores, couldnt_trsh = trsh.trsh_ess_job(label, level_of_theory, server, job_status,
                                                                       job_type, software, fine, memory_gb,
                                                                       num_heavy_atoms, cpu_cores, ess_trsh_methods)
        args = {'keyword': {}, 'block': {}}
        if trsh_keyword:
            args['trsh'] = {'trsh': trsh_keyword}
        spc_11 = ARCSpecies(label='ethanol', xyz=["""C	1.1658210	-0.4043550	0.0000000
                                                    C	0.0000000	0.5518050	0.0000000
                                                    O	-1.1894600	-0.2141940	0.0000000
                                                    H	-1.9412580	0.3751850	0.0000000
                                                    H	2.1054020	0.1451160	0.0000000
                                                    H	1.1306240	-1.0387850	0.8830320
                                                    H	1.1306240	-1.0387850	-0.8830320
                                                    H	0.0476820	1.1930570	0.8835910
                                                    H	0.0476820	1.1930570	-0.8835910"""],
                           directed_rotors={'brute_force_sp': [[1, 2], [2, 3]]})
        cls.job_11 = GaussianAdapter(execution_type='local',
                            job_type='optfreq',
                            level=Level(method='wb97xd'),
                            fine=True,
                            ess_trsh_methods=ess_trsh_methods,
                            project='test',
                            project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_GaussianAdapter'),
                            species=[spc_11],
                            testing=True,
                            args=args
                            )
        
        # Gaussian: Checkfile error and SCF error
        # First SCF error - qc,nosymm
        job_status = {'keywords': ['SCF', 'NoSymm']}
        output_errors, ess_trsh_methods, remove_checkfile, level_of_theory, software, job_type, fine, trsh_keyword, \
            memory, shift, cpu_cores, couldnt_trsh = trsh.trsh_ess_job(label, level_of_theory, server, job_status,
                                                                       job_type, software, fine, memory_gb,
                                                                       num_heavy_atoms, cpu_cores, ess_trsh_methods)
        args = {'keyword': {}, 'block': {}}
        if trsh_keyword:
            args['trsh'] = {'trsh': trsh_keyword}
        cls.job_12 = GaussianAdapter(execution_type='local',
                                          job_type='optfreq',
                                          level=Level(method='wb97xd'),
                                            fine=True,
                                            ess_trsh_methods=ess_trsh_methods,
                                            project='test',
                                            project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_GaussianAdapter'),
                                            species=[spc_11],
                                            testing=True,
                                            args=args
                                            )
        
        # Gaussian: Additional SCF error
        # Second SCF error - Includes previous SCF error and NDamp=30
        job_status = {'keywords': ['SCF']}
        output_errors, ess_trsh_methods, remove_checkfile, level_of_theory, software, job_type, fine, trsh_keyword, \
            memory, shift, cpu_cores, couldnt_trsh = trsh.trsh_ess_job(label, level_of_theory, server, job_status,
                                                                       job_type, software, fine, memory_gb,
                                                                       num_heavy_atoms, cpu_cores, ess_trsh_methods)
        args = {'keyword': {}, 'block': {}}
        if trsh_keyword:
            args['trsh'] = {'trsh': trsh_keyword}
        cls.job_13 = GaussianAdapter(execution_type='local',
                                          job_type='optfreq',
                                          level=Level(method='wb97xd'),
                                            fine=True,
                                            ess_trsh_methods=ess_trsh_methods,
                                            project='test',
                                            project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_GaussianAdapter'),
                                            species=[spc_11],
                                            testing=True,
                                            args=args
                                            )
        
        # Gaussian: Additional SCF error
        # Third SCF error - Includes previous SCF errors and NoDIIS
        job_status = {'keywords': ['SCF']}
        output_errors, ess_trsh_methods, remove_checkfile, level_of_theory, software, job_type, fine, trsh_keyword, \
            memory, shift, cpu_cores, couldnt_trsh = trsh.trsh_ess_job(label, level_of_theory, server, job_status,
                                                                       job_type, software, fine, memory_gb,
                                                                       num_heavy_atoms, cpu_cores, ess_trsh_methods)
        args = {'keyword': {}, 'block': {}}
        if trsh_keyword:
            args['trsh'] = {'trsh': trsh_keyword}
        cls.job_14 = GaussianAdapter(execution_type='local',
                                          job_type='optfreq',
                                          level=Level(method='wb97xd'),
                                            fine=True,
                                            ess_trsh_methods=ess_trsh_methods,
                                            project='test',
                                            project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_GaussianAdapter'),
                                            species=[spc_11],
                                            testing=True,
                                            args=args
                                            )

        # Gaussian: Internal coordinate error including a checkfile error and SCF errors
        #           Job type is switched to opt
        job_status = {'keywords': ['InternalCoordinateError']}
        job_type_15 = 'opt' # Need to switch job types for this error
        output_errors, ess_trsh_methods, remove_checkfile, level_of_theory, software, job_type, fine, trsh_keyword, \
            memory, shift, cpu_cores, couldnt_trsh = trsh.trsh_ess_job(label, level_of_theory, server, job_status,
                                                                          job_type_15, software, fine, memory_gb,  
                                                                            num_heavy_atoms, cpu_cores, ess_trsh_methods)
        args = {'keyword': {}, 'block': {}}
        if trsh_keyword:
            args['trsh'] = {'trsh': trsh_keyword}
        cls.job_15 = GaussianAdapter(execution_type='local',
                                            job_type='opt',
                                            level=Level(method='wb97xd'),
                                            fine=True,
                                            ess_trsh_methods=ess_trsh_methods,
                                            project='test',
                                            project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_GaussianAdapter'),
                                            species=[spc_11],
                                            testing=True,
                                            args=args
                                            )

        # Gaussian: Checkfile error and SCF error
        # First SCF error - qc,nosymm
        job_status = {'keywords': ['SCF', 'NoSymm', 'GL301']}
        output_errors, ess_trsh_methods, remove_checkfile, level_of_theory, software, job_type, fine, trsh_keyword, \
            memory, shift, cpu_cores, couldnt_trsh = trsh.trsh_ess_job(label, level_of_theory, server, job_status,
                                                                       job_type, software, fine, memory_gb,
                                                                       num_heavy_atoms, cpu_cores, ess_trsh_methods)
        args = {'keyword': {}, 'block': {}}
        if trsh_keyword:
            args['trsh'] = {'trsh': trsh_keyword}
        cls.job_16 = GaussianAdapter(execution_type='local',
                                          job_type='sp',
                                          level=Level(method='wb97xd'),
                                            fine=True,
                                            ess_trsh_methods=ess_trsh_methods,
                                            project='test',
                                            project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_GaussianAdapter'),
                                            species=[spc_11],
                                            testing=True,
                                            args=args
                                            )

    def test_set_cpu_and_mem(self):
        """Test assigning number of cpu's and memory"""
        self.job_8.input_file_memory = None
        self.job_8.submit_script_memory = None
        self.job_8.server = 'server2'
        self.job_8.set_cpu_and_mem()
        self.assertEqual(self.job_8.cpu_cores, 8)

    def test_set_input_file_memory(self):
        """Test setting the input_file_memory argument"""
        expected_memory = math.ceil(14 * 1024)
        self.assertEqual(self.job_1.input_file_memory, expected_memory)
        self.assertEqual(self.job_2.input_file_memory, 14336)

    def test_write_input_file(self):
        """Test writing Gaussian input files"""
        self.job_1.write_input_file()
        with open(os.path.join(self.job_1.local_path, input_filenames[self.job_1.job_adapter]), 'r') as f:
            content_1 = f.read()
        job_1_expected_input_file = """%chk=check.chk
%mem=14336mb
%NProcShared=8

#P opt=(calcfc)  cbs-qb3   IOp(2/9=2000) IOp(1/12=5,3/44=0)  

spc1

0 3
O       0.00000000    0.00000000    1.00000000


"""
        self.assertEqual(content_1, job_1_expected_input_file)

        self.job_3.write_input_file()
        with open(os.path.join(self.job_3.local_path, input_filenames[self.job_3.job_adapter]), 'r') as f:
            content_3 = f.read()
        job_3_expected_input_file = """%chk=check.chk
%mem=14336mb
%NProcShared=8

#P opt=(calcfc) SCRF=(smd, Solvent=water) uwb97xd/def2tzvp   IOp(2/9=2000)   

spc1

0 3
O       0.00000000    0.00000000    1.00000000


"""
        self.assertEqual(content_3, job_3_expected_input_file)

        self.job_4.write_input_file()
        with open(os.path.join(self.job_4.local_path, input_filenames[self.job_4.job_adapter]), 'r') as f:
            content_4 = f.read()
        job_4_expected_input_file = """%chk=check.chk
%mem=14336mb
%NProcShared=8

#P opt=(calcfc,maxStep=5,modredundant,noeigentest) integral=(grid=ultrafine, Acc2E=12) guess=mix wb97xd/def2tzvp   IOp(2/9=2000)    scf=(direct,tight)

ethanol

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

D 5 1 2 3 S 45 8.0


additional
gaussian
block


"""
        self.assertEqual(content_4, job_4_expected_input_file)

        self.job_5.write_input_file()
        with open(os.path.join(self.job_5.local_path, input_filenames[self.job_5.job_adapter]), 'r') as f:
            content_5 = f.read()
        job_5_expected_input_file = """%chk=check.chk
%mem=14336mb
%NProcShared=8

#P  uwb97xd/def2tzvp freq IOp(7/33=1)  integral=(grid=ultrafine, Acc2E=12)  IOp(2/9=2000)    scf=(direct,tight)

birad_singlet

0 1
O       0.00000000    0.00000000    1.00000000


"""
        self.assertEqual(content_5, job_5_expected_input_file)

        self.job_6.write_input_file()
        with open(os.path.join(self.job_6.local_path, input_filenames[self.job_6.job_adapter]), 'r') as f:
            content_6 = f.read()
        job_6_expected_input_file = """%chk=check.chk
%mem=14336mb
%NProcShared=8

#P opt=(calcfc)  uwb97xd/def2tzvp   IOp(2/9=2000)   

anion

-1 2
O       0.00000000    0.00000000    1.00000000


"""
        self.assertEqual(content_6, job_6_expected_input_file)

        self.job_7.write_input_file()
        with open(os.path.join(self.job_7.local_path, input_filenames[self.job_7.job_adapter]), 'r') as f:
            content_7 = f.read()
        job_7_expected_input_file = """%chk=check.chk
%mem=14336mb
%NProcShared=8

#P irc=(CalcAll, reverse, maxpoints=50, stepsize=7) wb97xd/def2tzvp   IOp(2/9=2000)   

IRC

0 1
O       0.00000000    0.00000000    1.00000000


"""
        self.assertEqual(content_7, job_7_expected_input_file)

    def test_set_files(self):
        """Test setting files"""
        job_3_files_to_upload = [{'file_name': 'submit.sub',
                                  'local': os.path.join(self.job_3.local_path, submit_filenames[servers[self.job_3.server]['cluster_soft']]),
                                  'remote': os.path.join(self.job_3.remote_path, submit_filenames[servers[self.job_3.server]['cluster_soft']]),
                                  'make_x': False,
                                  'source': 'path'},
                                 {'file_name': 'input.gjf',
                                  'local': os.path.join(self.job_3.local_path, input_filenames[self.job_3.job_adapter]),
                                  'remote': os.path.join(self.job_3.remote_path, input_filenames[self.job_3.job_adapter]),
                                  'source': 'path',
                                  'make_x': False}]
        job_3_files_to_download = [{'file_name': 'input.log',
                                    'local': os.path.join(self.job_3.local_path, output_filenames[self.job_3.job_adapter]),
                                    'remote': os.path.join(self.job_3.remote_path, output_filenames[self.job_3.job_adapter]),
                                    'source': 'path',
                                    'make_x': False},
                                   {'file_name': 'check.chk',
                                    'local': os.path.join(self.job_3.local_path, 'check.chk'),
                                    'remote': os.path.join(self.job_3.remote_path, 'check.chk'),
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

    def test_gaussian_def2tzvp(self):
        """Test a Gaussian job using def2-tzvp"""
        self.assertEqual(self.job_9.level.basis.lower(), 'def2tzvp')
    
    def test_trsh_write_input_file(self):
        """Test writing a trsh input file
        1. Test with getting Acc2E14 as the trsh method and thus changing the input file integral algorithm, and is also 'fine' thus it will have direct and tight SCF (but not xqc)
        2. Test with getting Acc2E14 as the trsh method but also checkfile=None in ess_trsh_methods, thus it will have both changes in the input file
        3. Test with getting Acc2E14 as the trsh method but also checkfile=None in ess_trsh_methods and first SCF error thus it will have all three changes in the input file
        4. Test with getting Acc2E14 as the trsh method but also checkfile=None in ess_trsh_methods and first and second SCF error and also the input file already has the integral algorithm change thus it will have all four changes in the input file
        5. Test with getting Acc2E14 as the trsh method but also checkfile=None in ess_trsh_methods and first, second and third SCF error and also the input file already has the integral algorithm change and also the input file already has the scf algorithm change thus it will have all five changes in the input file
        6. Test with all previous errors but now include an internal coordinate error thus it will have all six changes in the input file
        """
        self.job_10.write_input_file()
        with open(os.path.join(self.job_10.local_path, input_filenames[self.job_10.job_adapter]), 'r') as f:
            content_10 = f.read()
        job_10_expected_input_file = """%chk=check.chk
%mem=14336mb
%NProcShared=8

#P opt=(calcfc,maxstep=5,tight)  uwb97xd  integral=(grid=ultrafine, Acc2E=14) IOp(2/9=2000)    scf=(direct,tight)

anion

-1 2
O       0.00000000    0.00000000    1.00000000


"""
        self.assertEqual(content_10, job_10_expected_input_file)

        self.job_11.write_input_file()
        with open(os.path.join(self.job_11.local_path, input_filenames[self.job_11.job_adapter]), 'r') as f:
            content_11 = f.read()
        job_11_expected_input_file = """%chk=check.chk
%mem=14336mb
%NProcShared=8

#P opt=(calcfc,maxstep=5,tight)  guess=mix wb97xd  integral=(grid=ultrafine, Acc2E=14) IOp(2/9=2000)    scf=(direct,tight)

ethanol

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


"""
        self.assertEqual(content_11, job_11_expected_input_file)

        self.job_12.write_input_file()
        with open(os.path.join(self.job_12.local_path, input_filenames[self.job_12.job_adapter]), 'r') as f:
            content_12 = f.read()
        job_12_expected_input_file = """%chk=check.chk
%mem=14336mb
%NProcShared=8

#P opt=(calcfc,maxstep=5,tight)  guess=mix wb97xd  integral=(grid=ultrafine, Acc2E=14) IOp(2/9=2000)     nosymm scf=(direct,tight,xqc)

ethanol

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


"""
        self.assertEqual(content_12, job_12_expected_input_file)

        self.job_13.write_input_file()
        with open(os.path.join(self.job_13.local_path, input_filenames[self.job_13.job_adapter]), 'r') as f:
            content_13 = f.read()
        job_13_expected_input_file = """%chk=check.chk
%mem=14336mb
%NProcShared=8

#P opt=(calcfc,maxstep=5,tight)  guess=mix wb97xd  integral=(grid=ultrafine, Acc2E=14) IOp(2/9=2000)     nosymm scf=(NDamp=30,direct,tight,xqc)

ethanol

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


"""
        self.assertEqual(content_13, job_13_expected_input_file)

        self.job_14.write_input_file()
        with open(os.path.join(self.job_14.local_path, input_filenames[self.job_14.job_adapter]), 'r') as f:
            content_14 = f.read()
        job_14_expected_input_file = """%chk=check.chk
%mem=14336mb
%NProcShared=8

#P opt=(calcfc,maxstep=5,tight)  guess=mix wb97xd  integral=(grid=ultrafine, Acc2E=14) IOp(2/9=2000)     nosymm scf=(NDamp=30,NoDIIS,direct,tight,xqc)

ethanol

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


"""
        self.assertEqual(content_14, job_14_expected_input_file)

        self.job_15.write_input_file()
        with open(os.path.join(self.job_15.local_path, input_filenames[self.job_15.job_adapter]), 'r') as f:
            content_15 = f.read()
        job_15_expected_input_file = """%chk=check.chk
%mem=14336mb
%NProcShared=8

#P opt=(calcfc,cartesian,maxstep=5,tight)  guess=mix wb97xd  integral=(grid=ultrafine, Acc2E=14) IOp(2/9=2000)      nosymm scf=(NDamp=30,NoDIIS,direct,tight,xqc)

ethanol

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


"""
        self.assertEqual(content_15, job_15_expected_input_file)

        self.job_16.write_input_file()
        with open(os.path.join(self.job_16.local_path, input_filenames[self.job_16.job_adapter]), 'r') as f:
            content_16 = f.read()

        job_16_expected_input_file = """%chk=check.chk
%mem=14336mb
%NProcShared=8

#P opt=(cartesian) integral=(grid=ultrafine, Acc2E=14) guess=INDO wb97xd   IOp(2/9=2000)       nosymm  scf=(NDamp=30,NoDIIS,direct,tight,xqc)

ethanol

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


"""

        self.assertEqual(content_16, job_16_expected_input_file)

    @classmethod
    def tearDownClass(cls):
        """
        A function that is run ONCE after all unit tests in this class.
        Delete all project directories created during these unit tests.
        """
        shutil.rmtree(os.path.join(ARC_PATH, 'arc', 'testing', 'test_GaussianAdapter'), ignore_errors=True)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
