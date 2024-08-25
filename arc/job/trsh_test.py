#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.job.trsh module
"""

import os
import unittest
from unittest.mock import patch

import arc.job.trsh as trsh
from arc.common import ARC_PATH
from arc.imports import settings
from arc.parser import parse_1d_scan_energies

supported_ess = settings["supported_ess"]


class TestTrsh(unittest.TestCase):
    """
    Contains unit tests for the job.trsh module
    """

    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None
        path = os.path.join(ARC_PATH, "arc", "testing", "trsh")
        cls.base_path = {ess: os.path.join(path, ess) for ess in supported_ess}
        cls.server = "test_server"
        cls.job_name = "test_job"
        cls.job_id = "123"
        cls.servers = {
            "test_server": {
                "queue": {"short_queue": "1:00:00", "long_queue": "100:00:00"},
                " cluster_soft": "pbs",
            }
        }

    def test_determine_ess_status(self):
        """Test the determine_ess_status() function"""

        # Gaussian

        path = os.path.join(self.base_path["gaussian"], "converged.out")
        status, keywords, error, line = trsh.determine_ess_status(
            output_path=path, species_label="OH", job_type="opt"
        )
        self.assertEqual(status, "done")
        self.assertEqual(keywords, list())
        self.assertEqual(error, "")
        self.assertEqual(line, "")

        path = os.path.join(self.base_path["gaussian"], "l913.out")
        status, keywords, error, line = trsh.determine_ess_status(
            output_path=path, species_label="tst", job_type="composite"
        )
        self.assertEqual(status, "errored")
        self.assertEqual(keywords, ["MaxOptCycles", "GL913"])
        self.assertEqual(error, "Maximum optimization cycles reached.")
        self.assertIn("Error termination via Lnk1e", line)
        self.assertIn("g09/l913.exe", line)

        path = os.path.join(self.base_path["gaussian"], "l301_checkfile.out")
        status, keywords, error, line = trsh.determine_ess_status(
            output_path=path, species_label="Zr2O4H", job_type="opt"
        )
        self.assertEqual(status, "errored")
        self.assertEqual(keywords, ["CheckFile"])
        self.assertEqual(error, "No data on chk file.")
        self.assertIn("Error termination via Lnk1e", line)
        self.assertIn("g09/l301.exe", line)

        path = os.path.join(self.base_path["gaussian"], "l301.out")
        status, keywords, error, line = trsh.determine_ess_status(
            output_path=path, species_label="Zr2O4H", job_type="opt"
        )
        self.assertEqual(status, "errored")
        self.assertEqual(keywords, ["GL301", "BasisSet"])
        self.assertEqual(
            error, "The basis set 6-311G is not appropriate for the this chemistry."
        )
        self.assertIn("Error termination via Lnk1e", line)
        self.assertIn("g16/l301.exe", line)

        path = os.path.join(self.base_path["gaussian"], "l401.out")
        status, keywords, error, line = trsh.determine_ess_status(
            output_path=path, species_label="Zr2O4H", job_type="opt"
        )
        self.assertEqual(status, "errored")
        self.assertEqual(keywords, ["CheckFile"])
        self.assertEqual(error, "Basis set data is not on the checkpoint file.")
        self.assertIn("Error termination via Lnk1e", line)
        self.assertIn("g09/l401.exe", line)

        path = os.path.join(self.base_path["gaussian"], "l9999.out")
        status, keywords, error, line = trsh.determine_ess_status(
            output_path=path, species_label="Zr2O4H", job_type="opt"
        )
        self.assertEqual(status, "errored")
        self.assertEqual(keywords, ["Unconverged", "GL9999"])
        self.assertEqual(error, "Unconverged")
        self.assertIn("Error termination via Lnk1e", line)
        self.assertIn("g16/l9999.exe", line)

        path = os.path.join(self.base_path["gaussian"], "syntax.out")
        status, keywords, error, line = trsh.determine_ess_status(
            output_path=path, species_label="Zr2O4H", job_type="opt"
        )
        self.assertEqual(status, "errored")
        self.assertEqual(keywords, ["Syntax"])
        self.assertEqual(
            error,
            "There was a syntax error in the Gaussian input file. Check your Gaussian input file "
            "template under arc/job/inputs.py. Alternatively, perhaps the level of theory is not "
            "supported by Gaussian in the specific format it was given.",
        )
        self.assertFalse(line)

        path = os.path.join(self.base_path["gaussian"], "maxsteps.out")
        status, keywords, error, line = trsh.determine_ess_status(
            output_path=path, species_label="Zr2O4H", job_type="opt"
        )
        self.assertEqual(status, "errored")
        self.assertEqual(keywords, ["MaxOptCycles", "GL9999",])
        self.assertEqual(error, "Maximum optimization cycles reached.")
        self.assertIn("Number of steps exceeded", line)
        
        path = os.path.join(self.base_path["gaussian"], "inaccurate_quadrature.out")
        status, keywords, error, line = trsh.determine_ess_status(
            output_path=path, species_label="Zr2O4H", job_type="opt"
        )
        self.assertEqual(status, "errored")
        self.assertEqual(keywords, ["InaccurateQuadrature", "GL502"])
        self.assertEqual(error, "Inaccurate quadrature in CalDSu")
        self.assertIn("Inaccurate quadrature in CalDSu", line)
        
        path = os.path.join(self.base_path["gaussian"], "l123_deltax.out")
        status, keywords, error, line = trsh.determine_ess_status(
            output_path=path, species_label="Zr2O4H", job_type="irc"
        )
        self.assertEqual(status, "errored")
        self.assertEqual(keywords, ["DeltaX", "GL123"])
        self.assertEqual(error, 'Delta-x Convergence NOT Met')
        self.assertIn("Delta-x Convergence NOT Met", line)


        # QChem

        path = os.path.join(self.base_path["qchem"], "H2_opt.out")
        status, keywords, error, line = trsh.determine_ess_status(
            output_path=path, species_label="H2", job_type="opt"
        )
        self.assertEqual(status, "done")
        self.assertEqual(keywords, list())
        self.assertEqual(error, "")
        self.assertEqual(line, "")

        # Molpro

        path = os.path.join(self.base_path["molpro"], "unrecognized_basis_set.out")
        status, keywords, error, line = trsh.determine_ess_status(
            output_path=path, species_label="I", job_type="sp"
        )
        self.assertEqual(status, "errored")
        self.assertEqual(keywords, ["BasisSet"])
        self.assertEqual(error, "Unrecognized basis set 6-311G**")
        self.assertIn(" ? Basis library exhausted", line)  # line includes '\n'

        # Orca

        # test detection of a successful job
        path = os.path.join(self.base_path["orca"], "orca_successful_sp.log")
        status, keywords, error, line = trsh.determine_ess_status(
            output_path=path, species_label="test", job_type="sp", software="orca"
        )
        self.assertEqual(status, "done")
        self.assertEqual(keywords, list())
        self.assertEqual(error, "")
        self.assertEqual(line, "")

        # test detection of a successful job
        # notice that the log file in this example has a different format under the line
        # ***  Starting incremental Fock matrix formation  ***
        # compared to the above example. It is important to make sure that ARC's Orca trsh algorithm parse this
        # log file successfully
        path = os.path.join(self.base_path["orca"], "orca_successful_sp_scf.log")
        status, keywords, error, line = trsh.determine_ess_status(
            output_path=path, species_label="test", job_type="sp", software="orca"
        )
        self.assertEqual(status, "done")
        self.assertEqual(keywords, list())
        self.assertEqual(error, "")
        self.assertEqual(line, "")

        # test detection of SCF energy diverge issue
        path = os.path.join(self.base_path["orca"], "orca_scf_blow_up_error.log")
        status, keywords, error, line = trsh.determine_ess_status(
            output_path=path, species_label="test", job_type="sp", software="orca"
        )
        self.assertEqual(status, "errored")
        self.assertEqual(keywords, ["SCF"])
        expected_error_msg = (
            "The SCF energy seems diverged during iterations. "
            "SCF energy after initial iteration is -1076.6615662471. "
            "SCF energy after final iteration is -20006124.68383977. "
            "The ratio between final and initial SCF energy is 18581.627979509627. "
            "This ratio is greater than the default threshold of 2. "
            "Please consider using alternative methods or larger basis sets."
        )
        self.assertEqual(error, expected_error_msg)
        self.assertIn("", line)

        # test detection of insufficient memory causes SCF failure
        path = os.path.join(self.base_path["orca"], "orca_scf_memory_error.log")
        status, keywords, error, line = trsh.determine_ess_status(
            output_path=path, species_label="test", job_type="sp", software="orca"
        )
        self.assertEqual(status, "errored")
        self.assertEqual(keywords, ["SCF", "Memory"])
        expected_error_msg = (
            "Orca suggests to increase per cpu core memory to 789.0 MB."
        )
        self.assertEqual(error, expected_error_msg)
        self.assertEqual(" Error  (ORCA_SCF): Not enough memory available!", line)

        # test detection of insufficient memory causes MDCI failure
        path = os.path.join(self.base_path["orca"], "orca_mdci_memory_error.log")
        status, keywords, error, line = trsh.determine_ess_status(
            output_path=path, species_label="test", job_type="sp", software="orca"
        )
        self.assertEqual(status, "errored")
        self.assertEqual(keywords, ["MDCI", "Memory"])
        expected_error_msg = (
            "Orca suggests to increase per cpu core memory to 10218 MB."
        )
        self.assertEqual(error, expected_error_msg)
        self.assertIn("Please increase MaxCore", line)

        # test detection of too many cpu cores causes MDCI failure
        path = os.path.join(self.base_path["orca"], "orca_too_many_cores.log")
        status, keywords, error, line = trsh.determine_ess_status(
            output_path=path, species_label="test", job_type="sp", software="orca"
        )
        self.assertEqual(status, "errored")
        self.assertEqual(keywords, ["MDCI", "cpu"])
        expected_error_msg = (
            "Orca cannot utilize cpu cores more than electron pairs in a molecule. "
            "The maximum number of cpu cores can be used for this job is 10."
        )
        self.assertEqual(error, expected_error_msg)
        self.assertIn("parallel calculation exceeds", line)

        # test detection of generic GTOInt failure
        path = os.path.join(self.base_path["orca"], "orca_GTOInt_error.log")
        status, keywords, error, line = trsh.determine_ess_status(
            output_path=path, species_label="test", job_type="sp", software="orca"
        )
        self.assertEqual(status, "errored")
        self.assertEqual(keywords, ["GTOInt", "Memory"])
        expected_error_msg = "GTOInt error in Orca. Assuming memory allocation error."
        self.assertEqual(error, expected_error_msg)
        self.assertIn("ORCA finished by error termination in GTOInt", line)

        # test detection of generic MDCI failure
        path = os.path.join(self.base_path["orca"], "orca_mdci_error.log")
        status, keywords, error, line = trsh.determine_ess_status(
            output_path=path, species_label="test", job_type="sp", software="orca"
        )
        self.assertEqual(status, "errored")
        self.assertEqual(keywords, ["MDCI", "Memory"])
        expected_error_msg = "MDCI error in Orca. Assuming memory allocation error."
        self.assertEqual(error, expected_error_msg)
        self.assertIn("ORCA finished by error termination in MDCI", line)

        # test detection of generic MDCI failure in Orca version 4.2.x log files
        path = os.path.join(self.base_path["orca"], "orca_mdci_error_2.log")
        status, keywords, error, line = trsh.determine_ess_status(
            output_path=path, species_label="test", job_type="sp", software="orca"
        )
        self.assertEqual(status, "errored")
        self.assertEqual(keywords, ["MDCI", "Memory"])
        expected_error_msg = "MDCI error in Orca. Assuming memory allocation error."
        self.assertEqual(error, expected_error_msg)
        self.assertIn("ORCA finished by error termination in MDCI", line)

        # test detection of MDCI failure in Orca version 4.1.x log files (no memory/cpu suggestions compared to 4.2.x)
        path = os.path.join(self.base_path["orca"], "orca_mdci_error_3.log")
        status, keywords, error, line = trsh.determine_ess_status(
            output_path=path, species_label="test", job_type="sp", software="orca"
        )
        self.assertEqual(status, "errored")
        self.assertEqual(keywords, ["MDCI", "cpu"])
        expected_error_msg = (
            "Orca cannot utilize cpu cores more than electron pairs in a molecule. ARC will "
            "estimate the number of cpu cores needed based on the number of heavy atoms in the "
            "molecule."
        )
        self.assertEqual(error, expected_error_msg)
        self.assertIn(
            "Number of processes in parallel calculation exceeds number of pairs", line
        )

        # test detection of multiplicty and charge combination error
        path = os.path.join(self.base_path["orca"], "orca_multiplicity_error.log")
        status, keywords, error, line = trsh.determine_ess_status(
            output_path=path, species_label="test", job_type="sp", software="orca"
        )
        self.assertEqual(status, "errored")
        self.assertEqual(keywords, ["Input"])
        expected_error_msg = (
            "The multiplicity and charge combination for species test are wrong."
        )
        self.assertEqual(error, expected_error_msg)
        self.assertIn("Error : multiplicity", line)

        # test detection of input keyword error
        path = os.path.join(self.base_path["orca"], "orca_input_error.log")
        status, keywords, error, line = trsh.determine_ess_status(
            output_path=path, species_label="test", job_type="sp", software="orca"
        )
        self.assertEqual(status, "errored")
        self.assertEqual(keywords, ["Syntax"])
        expected_error_msg = (
            "There was keyword syntax error in the Orca input file. In particular, keywords "
            "XTB1 can either be duplicated or illegal. Please check your Orca "
            "input file template under arc/job/inputs.py. Alternatively, perhaps the level of "
            "theory or the job option is not supported by Orca in the format it was given."
        )
        self.assertEqual(error, expected_error_msg)
        self.assertIn("XTB1", line)

        # test detection of basis set error (e.g., input contains elements not supported by specified basis)
        path = os.path.join(self.base_path["orca"], "orca_basis_error.log")
        status, keywords, error, line = trsh.determine_ess_status(
            output_path=path, species_label="test", job_type="sp", software="orca"
        )
        self.assertEqual(status, "errored")
        self.assertEqual(keywords, ["Basis"])
        expected_error_msg = (
            "There was a basis set error in the Orca input file. In particular, basis for atom type "
            "Br is missing. Please check if specified basis set supports this atom."
        )
        self.assertEqual(error, expected_error_msg)
        self.assertIn("There are no CABS", line)

        # test detection of wavefunction convergence failure
        path = os.path.join(
            self.base_path["orca"], "orca_wavefunction_not_converge_error.log"
        )
        status, keywords, error, line = trsh.determine_ess_status(
            output_path=path, species_label="test", job_type="sp", software="orca"
        )
        self.assertEqual(status, "errored")
        self.assertEqual(keywords, ["Convergence"])
        expected_error_msg = (
            "Specified wavefunction method is not converged. Please restart calculation with larger "
            "max iterations or with different convergence flags."
        )
        self.assertEqual(error, expected_error_msg)
        self.assertIn("This wavefunction IS NOT FULLY CONVERGED!", line)

    def test_trsh_ess_job(self):
        """Test the trsh_ess_job() function"""

        # Test Gaussian
        label = 'ethanol'
        level_of_theory = {'method': 'ccsd', 'basis': 'vdz'}
        server = 'server1'
        job_type = 'opt'
        software = 'gaussian'
        fine = False
        memory_gb = 16
        num_heavy_atoms = 2
        ess_trsh_methods = ['change_node', 'int=(Acc2E=14)']
        cpu_cores = 8

        # Gaussian: test 1
        job_status = {'keywords': ['CheckFile']}
        output_errors, ess_trsh_methods, remove_checkfile, level_of_theory, software, job_type, fine, trsh_keyword, \
            memory, shift, cpu_cores, couldnt_trsh = trsh.trsh_ess_job(label, level_of_theory, server, job_status,
                                                                       job_type, software, fine, memory_gb,
                                                                       num_heavy_atoms, cpu_cores, ess_trsh_methods)

        self.assertTrue(remove_checkfile)
        self.assertEqual(software, 'gaussian')
        self.assertEqual(memory, 16)
        self.assertFalse(couldnt_trsh)

        # Gaussian: test 2
        job_status = {'keywords': ['InternalCoordinateError', 'NoSymm']}
        output_errors, ess_trsh_methods, remove_checkfile, level_of_theory, software, job_type, fine, trsh_keyword, \
            memory, shift, cpu_cores, couldnt_trsh = trsh.trsh_ess_job(label, level_of_theory, server, job_status,
                                                                       job_type, software, fine, memory_gb,
                                                                       num_heavy_atoms, cpu_cores, ess_trsh_methods)

        self.assertTrue(remove_checkfile)
        self.assertEqual(ess_trsh_methods, ['change_node', 'int=(Acc2E=14)', 'checkfile=None', 'cartesian', 'NoSymm'])
        self.assertEqual(trsh_keyword, ['opt=(cartesian)', 'int=(Acc2E=14)', 'nosymm'] )

        # Gaussian: test 3
        job_status = {'keywords': ['SCF', 'GL502', 'NoSymm']}
        output_errors, ess_trsh_methods, remove_checkfile, level_of_theory, software, job_type, fine, trsh_keyword, \
            memory, shift, cpu_cores, couldnt_trsh = trsh.trsh_ess_job(label, level_of_theory, server, job_status,
                                                                    job_type, software, fine, memory_gb,
                                                                    num_heavy_atoms, cpu_cores, ess_trsh_methods)

        self.assertTrue(remove_checkfile)
        self.assertIn('scf=(qc)', ess_trsh_methods)
        self.assertFalse(couldnt_trsh)

        # Gaussian: test 5
        job_status = {'keywords': ['DiskSpace']}
        output_errors, ess_trsh_methods, remove_checkfile, level_of_theory, software, job_type, fine, trsh_keyword, \
            memory, shift, cpu_cores, couldnt_trsh = trsh.trsh_ess_job(label, level_of_theory, server, job_status,
                                                                    job_type, software, fine, memory_gb,
                                                                    num_heavy_atoms, cpu_cores, ess_trsh_methods)

        self.assertTrue(couldnt_trsh)
        self.assertIn('Error: Could not troubleshoot opt for ethanol! The job ran out of disc space on server1; ', output_errors)

        # Gaussian: test 6
        job_status = {'keywords': ['SCF', 'GL502', 'NoSymm']}
        ess_trsh_methods = ['scf=(NoDIIS)', 'int=(Acc2E=14)', 'checkfile=None', 'scf=(qc)', 'NoSymm','scf=(NDamp=30)', 'guess=INDO', 'scf=(Fermi)',
                            'scf=(Noincfock)', 'scf=(NoVarAcc)']
        output_errors, ess_trsh_methods, remove_checkfile, level_of_theory, software, job_type, fine, trsh_keyword, \
            memory, shift, cpu_cores, couldnt_trsh = trsh.trsh_ess_job(label, level_of_theory, server, job_status,
                                                                    job_type, software, fine, memory_gb,
                                                                    num_heavy_atoms, cpu_cores, ess_trsh_methods)
        self.assertTrue(couldnt_trsh)
        self.assertIn(
            "Error: Could not troubleshoot opt for ethanol! Tried troubleshooting with the following methods: ['scf=(NoDIIS)', 'int=(Acc2E=14)', 'checkfile=None', 'scf=(qc)', 'NoSymm', 'scf=(NDamp=30)', 'guess=INDO', 'scf=(Fermi)', 'scf=(Noincfock)', 'scf=(NoVarAcc)', 'all_attempted']; ",
            output_errors,
        )

        # Gaussian: test 7
        job_status = {'keywords': ['MaxOptCycles', 'GL9999','SCF']}
        ess_trsh_methods = ['int=(Acc2E=14)']
        output_errors, ess_trsh_methods, remove_checkfile, level_of_theory, software, job_type, fine, trsh_keyword, \
            memory, shift, cpu_cores, couldnt_trsh = trsh.trsh_ess_job(label, level_of_theory, server, job_status,
                                                                    job_type, software, fine, memory_gb,
                                                                    num_heavy_atoms, cpu_cores, ess_trsh_methods)
        self.assertFalse(couldnt_trsh)
        self.assertIn('opt=(maxcycle=200)', ess_trsh_methods)

        # Gaussian: test 8 - part 1
        # 'InaccurateQuadrature', 'GL502'
        job_status = {'keywords': ['InaccurateQuadrature', 'GL502']}
        ess_trsh_methods = ['int=(Acc2E=14)']
        output_errors, ess_trsh_methods, remove_checkfile, level_of_theory, software, job_type, fine, trsh_keyword, \
            memory, shift, cpu_cores, couldnt_trsh = trsh.trsh_ess_job(label, level_of_theory, server, job_status,
                                                                    job_type, software, fine, memory_gb,
                                                                    num_heavy_atoms, cpu_cores, ess_trsh_methods)
        self.assertFalse(couldnt_trsh)
        self.assertIn('int=(Acc2E=14)', ess_trsh_methods)
        self.assertIn('int=grid=300590', ess_trsh_methods)
        
        # Gaussian: test 8 - part 2
        # 'InaccurateQuadrature', 'GL502'
        job_status = {'keywords': ['InaccurateQuadrature', 'GL502']}
        ess_trsh_methods = ['int=(Acc2E=14)', 'int=grid=300590']
        output_errors, ess_trsh_methods, remove_checkfile, level_of_theory, software, job_type, fine, trsh_keyword, \
            memory, shift, cpu_cores, couldnt_trsh = trsh.trsh_ess_job(label, level_of_theory, server, job_status,
                                                                    job_type, software, fine, memory_gb,
                                                                    num_heavy_atoms, cpu_cores, ess_trsh_methods)
        self.assertFalse(couldnt_trsh)
        self.assertIn('int=(Acc2E=14)', ess_trsh_methods)
        self.assertIn('int=grid=300590', ess_trsh_methods)
        self.assertIn('scf=(NoVarAcc)', ess_trsh_methods)
        
        # Gaussian: test 8 - part 3
        # 'InaccurateQuadrature', 'GL502'
        job_status = {'keywords': ['InaccurateQuadrature', 'GL502']}
        ess_trsh_methods = ['int=(Acc2E=14)', 'int=grid=300590', 'scf=(NoVarAcc)']
        output_errors, ess_trsh_methods, remove_checkfile, level_of_theory, software, job_type, fine, trsh_keyword, \
            memory, shift, cpu_cores, couldnt_trsh = trsh.trsh_ess_job(label, level_of_theory, server, job_status,
                                                                    job_type, software, fine, memory_gb,
                                                                    num_heavy_atoms, cpu_cores, ess_trsh_methods)
        self.assertFalse(couldnt_trsh)
        self.assertIn('int=(Acc2E=14)', ess_trsh_methods)
        self.assertIn('int=grid=300590', ess_trsh_methods)
        self.assertIn('scf=(NoVarAcc)', ess_trsh_methods)
        self.assertIn('guess=INDO', ess_trsh_methods)
        
        # Gaussian: test 9 - part 1
        # 'MaxOptCycles', 'GL9999'
        # Adding maxcycle=200 to opt
        job_status = {'keywords': ['MaxOptCycles', 'GL9999']}
        ess_trsh_methods = ['int=(Acc2E=14)']
        output_errors, ess_trsh_methods, remove_checkfile, level_of_theory, software, job_type, fine, trsh_keyword, \
            memory, shift, cpu_cores, couldnt_trsh = trsh.trsh_ess_job(label, level_of_theory, server, job_status,
                                                                    job_type, software, fine, memory_gb,
                                                                    num_heavy_atoms, cpu_cores, ess_trsh_methods)
        self.assertFalse(couldnt_trsh)
        self.assertIn('opt=(maxcycle=200)', ess_trsh_methods)
        
        # Gaussian: test 9 - part 2
        # 'MaxOptCycles', 'GL9999'
        # Adding RFO to opt
        job_status = {'keywords': ['MaxOptCycles', 'GL9999']}
        ess_trsh_methods = ['int=(Acc2E=14)', 'opt=(maxcycle=200)']
        output_errors, ess_trsh_methods, remove_checkfile, level_of_theory, software, job_type, fine, trsh_keyword, \
            memory, shift, cpu_cores, couldnt_trsh = trsh.trsh_ess_job(label, level_of_theory, server, job_status,
                                                                    job_type, software, fine, memory_gb,
                                                                    num_heavy_atoms, cpu_cores, ess_trsh_methods)
        self.assertFalse(couldnt_trsh)
        self.assertIn('opt=(maxcycle=200)', ess_trsh_methods)
        self.assertIn('opt=(RFO)', ess_trsh_methods)
        self.assertIn('opt=(maxcycle=200,RFO)', trsh_keyword)
        
        # Gaussian: test 9 - part 3
        # 'MaxOptCycles', 'GL9999'
        # Adding GDIIS to opt
        # Removing RFO from opt
        job_status = {'keywords': ['MaxOptCycles', 'GL9999']}
        ess_trsh_methods = ['int=(Acc2E=14)', 'opt=(maxcycle=200)', 'opt=(RFO)']
        output_errors, ess_trsh_methods, remove_checkfile, level_of_theory, software, job_type, fine, trsh_keyword, \
            memory, shift, cpu_cores, couldnt_trsh = trsh.trsh_ess_job(label, level_of_theory, server, job_status,
                                                                    job_type, software, fine, memory_gb,
                                                                    num_heavy_atoms, cpu_cores, ess_trsh_methods)
        self.assertFalse(couldnt_trsh)
        self.assertIn('opt=(maxcycle=200)', ess_trsh_methods)
        self.assertIn('opt=(RFO)', ess_trsh_methods)
        self.assertIn('opt=(GDIIS)', ess_trsh_methods)
        self.assertIn('opt=(maxcycle=200,GDIIS)', trsh_keyword)
        
        # Gaussian: test 9 - part 4
        # 'MaxOptCycles', 'GL9999'
        # Adding GEDIIS to opt
        # Removing RFO from opt
        # Removing GDIIS from opt
        job_status = {'keywords': ['MaxOptCycles', 'GL9999']}
        ess_trsh_methods = ['int=(Acc2E=14)', 'opt=(maxcycle=200)', 'opt=(RFO)', 'opt=(GDIIS)']
        output_errors, ess_trsh_methods, remove_checkfile, level_of_theory, software, job_type, fine, trsh_keyword, \
            memory, shift, cpu_cores, couldnt_trsh = trsh.trsh_ess_job(label, level_of_theory, server, job_status,
                                                                    job_type, software, fine, memory_gb,
                                                                    num_heavy_atoms, cpu_cores, ess_trsh_methods)
        self.assertFalse(couldnt_trsh)
        self.assertIn('opt=(maxcycle=200)', ess_trsh_methods)
        self.assertIn('opt=(RFO)', ess_trsh_methods)
        self.assertIn('opt=(GDIIS)', ess_trsh_methods)
        self.assertIn('opt=(GEDIIS)', ess_trsh_methods)
        self.assertIn('opt=(maxcycle=200,GEDIIS)', trsh_keyword)
        
        # Gaussian: test 9 - part 5
        # 'MaxOptCycles', 'GL9999'
        # Final test to ensure that it cannot troubleshoot the job further
        job_status = {'keywords': ['MaxOptCycles', 'GL9999']}
        ess_trsh_methods = ['int=(Acc2E=14)', 'opt=(maxcycle=200)', 'opt=(RFO)', 'opt=(GDIIS)', 'opt=(GEDIIS)']
        output_errors, ess_trsh_methods, remove_checkfile, level_of_theory, software, job_type, fine, trsh_keyword, \
            memory, shift, cpu_cores, couldnt_trsh = trsh.trsh_ess_job(label, level_of_theory, server, job_status,
                                                                    job_type, software, fine, memory_gb,
                                                                    num_heavy_atoms, cpu_cores, ess_trsh_methods)
        self.assertIn('all_attempted', ess_trsh_methods)
        self.assertTrue(couldnt_trsh)
        
        # Gaussian: test 10 - part 1
        # 'GL123', 'DeltaX'
        # Adding maxcycle=200 to irc
        job_status = {'keywords': ['GL123', 'DeltaX']}
        ess_trsh_methods = ['int=(Acc2E=14)']
        job_type = 'irc'
        output_errors, ess_trsh_methods, remove_checkfile, level_of_theory, software, job_type, fine, trsh_keyword, \
            memory, shift, cpu_cores, couldnt_trsh = trsh.trsh_ess_job(label, level_of_theory, server, job_status,
                                                                    job_type, software, fine, memory_gb,
                                                                    num_heavy_atoms, cpu_cores, ess_trsh_methods)
        self.assertFalse(couldnt_trsh)
        self.assertIn('irc=(maxcycle=200)', ess_trsh_methods)
        self.assertIn('irc=(maxcycle=200)', trsh_keyword)
        
        # Gaussian: test 10 - part 2
        # 'GL123', 'DeltaX'
        # Changing algorithm
        job_status = {'keywords': ['GL123', 'DeltaX']}
        ess_trsh_methods = ['int=(Acc2E=14)', 'irc=(maxcycle=200)']
        job_type = 'irc'
        output_errors, ess_trsh_methods, remove_checkfile, level_of_theory, software, job_type, fine, trsh_keyword, \
            memory, shift, cpu_cores, couldnt_trsh = trsh.trsh_ess_job(label, level_of_theory, server, job_status,
                                                                    job_type, software, fine, memory_gb,
                                                                    num_heavy_atoms, cpu_cores, ess_trsh_methods)
        self.assertFalse(couldnt_trsh)
        self.assertIn('irc=(maxcycle=200)', ess_trsh_methods)
        self.assertIn('irc=(LQA)', ess_trsh_methods)
        self.assertIn('irc=(maxcycle=200,LQA)', trsh_keyword)
        
        # Gaussian: test 11
        # 'NegEigenvalues', 'GL9999'
        # Adding noeigen to opt
        job_status = {'keywords': ['NegEigenvalues', 'GL9999']}
        ess_trsh_methods = ['int=(Acc2E=14)']
        job_type = 'opt'
        output_errors, ess_trsh_methods, remove_checkfile, level_of_theory, software, job_type, fine, trsh_keyword, \
            memory, shift, cpu_cores, couldnt_trsh = trsh.trsh_ess_job(label, level_of_theory, server, job_status,
                                                                    job_type, software, fine, memory_gb,
                                                                    num_heavy_atoms, cpu_cores, ess_trsh_methods)
        self.assertFalse(couldnt_trsh)
        self.assertIn('opt=(noeigen)', ess_trsh_methods)


        # Test Q-Chem
        software = "qchem"
        ess_trsh_methods = ["change_node"]
        job_status = {"keywords": ["MaxOptCycles", "Unconverged"]}
        # Q-Chem: test 1
        output_errors, ess_trsh_methods, remove_checkfile, level_of_theory, software, job_type, fine, trsh_keyword, \
            memory, shift, cpu_cores, couldnt_trsh = trsh.trsh_ess_job(label, level_of_theory, server, job_status,
                                                                       job_type, software, fine, memory_gb,
                                                                       num_heavy_atoms, cpu_cores, ess_trsh_methods)
        self.assertIn('max_cycles', ess_trsh_methods)

        # Q-Chem: test 2
        job_status = {'keywords': ['SCF']}
        output_errors, ess_trsh_methods, remove_checkfile, level_of_theory, software, job_type, fine, trsh_keyword, \
            memory, shift, cpu_cores, couldnt_trsh = trsh.trsh_ess_job(label, level_of_theory, server, job_status,
                                                                       job_type, software, fine, memory_gb,
                                                                       num_heavy_atoms, cpu_cores, ess_trsh_methods)
        self.assertIn('DIIS_GDM', ess_trsh_methods)

        # Test Molpro
        software = "molpro"

        # Molpro: test
        path = os.path.join(self.base_path["molpro"], "insufficient_memory.out")
        label = "TS"
        level_of_theory = {"method": "mrci", "basis": "aug-cc-pV(T+d)Z"}
        server = "server1"
        status, keywords, error, line = trsh.determine_ess_status(
            output_path=path, species_label="TS", job_type="sp"
        )
        job_status = {"keywords": keywords, "error": error}
        job_type = "sp"
        fine = True
        memory_gb = 32.0
        ess_trsh_methods = ['change_node']
        output_errors, ess_trsh_methods, remove_checkfile, level_of_theory, software, job_type, fine, trsh_keyword, \
            memory, shift, cpu_cores, couldnt_trsh = trsh.trsh_ess_job(label, level_of_theory, server, job_status,
                                                                       job_type, software, fine, memory_gb,
                                                                       num_heavy_atoms, cpu_cores, ess_trsh_methods)
        self.assertIn('memory', ess_trsh_methods)
        self.assertAlmostEqual(memory, 222.15625)

        path = os.path.join(self.base_path['molpro'], 'insufficient_memory_2.out')
        status, keywords, error, line = trsh.determine_ess_status(output_path=path, species_label='TS', job_type='sp')
        job_status = {'keywords': keywords, 'error': error}
        output_errors, ess_trsh_methods, remove_checkfile, level_of_theory, software, job_type, fine, trsh_keyword, \
            memory, shift, cpu_cores, couldnt_trsh = trsh.trsh_ess_job(label, level_of_theory, server, job_status,
                                                                       job_type, software, fine, memory_gb,
                                                                       num_heavy_atoms, cpu_cores, ess_trsh_methods)
        self.assertIn('memory', ess_trsh_methods)
        self.assertEqual(memory, 96.0)

        # Molpro: Insuffienct Memory 3 Test
        path = os.path.join(self.base_path['molpro'], 'insufficient_memory_3.out')
        status, keywords, error, line = trsh.determine_ess_status(output_path=path,
                                                                  species_label='TS',
                                                                  job_type='sp')
        job_status = {'keywords': keywords, 'error': error}
        output_errors, ess_trsh_methods, remove_checkfile, level_of_theory, software, job_type, fine, trsh_keyword, \
            memory, shift, cpu_cores, couldnt_trsh = trsh.trsh_ess_job(label, level_of_theory, server, job_status,
                                                                       job_type, software, fine, memory_gb,
                                                                       num_heavy_atoms, cpu_cores, ess_trsh_methods)
        self.assertIn('memory', ess_trsh_methods)
        self.assertEqual(memory, 62.0)

        # Test Orca
        # Orca: test 1
        # Test troubleshooting insufficient memory issue
        # Automatically increase memory provided not exceeding maximum available memory
        label = 'test'
        level_of_theory = {'method': 'dlpno-ccsd(T)'}
        server = 'server1'
        job_type = 'sp'
        software = 'orca'
        fine = True
        memory_gb = 250
        cpu_cores = 32
        num_heavy_atoms = 20
        ess_trsh_methods = ['memory']
        path = os.path.join(self.base_path['orca'], 'orca_mdci_memory_error.log')
        status, keywords, error, line = trsh.determine_ess_status(output_path=path, species_label='TS', job_type='sp',
                                                                  software=software)
        job_status = {'keywords': keywords, 'error': error}
        output_errors, ess_trsh_methods, remove_checkfile, level_of_theory, software, job_type, fine, trsh_keyword, \
            memory, shift, cpu_cores, couldnt_trsh = trsh.trsh_ess_job(label, level_of_theory, server, job_status,
                                                                       job_type, software, fine, memory_gb,
                                                                       num_heavy_atoms, cpu_cores, ess_trsh_methods)
        self.assertIn('memory', ess_trsh_methods)
        self.assertEqual(cpu_cores, 32)
        self.assertAlmostEqual(memory, 327)

        # Orca: test 2
        # Test troubleshooting insufficient memory issue
        # Automatically reduce cpu cores to ensure enough memory per core when maximum available memory is limited
        label = 'test'
        level_of_theory = {'method': 'dlpno-ccsd(T)'}
        server = 'server1'
        job_type = 'sp'
        software = 'orca'
        fine = True
        memory_gb = 250
        cpu_cores = 32
        num_heavy_atoms = 20
        ess_trsh_methods = ['memory']
        path = os.path.join(self.base_path['orca'], 'orca_mdci_memory_error.log')
        status, keywords, error, line = trsh.determine_ess_status(output_path=path, species_label='TS', job_type='sp',
                                                                  software=software)
        keywords.append('max_total_job_memory')
        job_status = {'keywords': keywords, 'error': error}
        output_errors, ess_trsh_methods, remove_checkfile, level_of_theory, software, job_type, fine, trsh_keyword, \
            memory, shift, cpu_cores, couldnt_trsh = trsh.trsh_ess_job(label, level_of_theory, server, job_status,
                                                                       job_type, software, fine, memory_gb,
                                                                       num_heavy_atoms, cpu_cores, ess_trsh_methods)
        self.assertIn('memory', ess_trsh_methods)
        self.assertEqual(cpu_cores, 22)
        self.assertAlmostEqual(memory, 227)

        # Orca: test 3
        # Test troubleshooting insufficient memory issue
        # Stop troubleshooting when ARC determined there is not enough computational resource to accomplish the job
        label = 'test'
        level_of_theory = {'method': 'dlpno-ccsd(T)'}
        server = 'server1'
        job_type = 'sp'
        software = 'orca'
        fine = True
        memory_gb = 1
        cpu_cores = 32
        num_heavy_atoms = 20
        ess_trsh_methods = ['memory']
        path = os.path.join(self.base_path['orca'], 'orca_mdci_memory_error.log')
        status, keywords, error, line = trsh.determine_ess_status(output_path=path, species_label='TS', job_type='sp',
                                                                  software=software)
        keywords.append('max_total_job_memory')
        job_status = {'keywords': keywords, 'error': error}
        output_errors, ess_trsh_methods, remove_checkfile, level_of_theory, software, job_type, fine, trsh_keyword, \
            memory, shift, cpu_cores, couldnt_trsh = trsh.trsh_ess_job(label, level_of_theory, server, job_status,
                                                                       job_type, software, fine, memory_gb,
                                                                       num_heavy_atoms, cpu_cores, ess_trsh_methods)
        self.assertIn('memory', ess_trsh_methods)
        self.assertEqual(couldnt_trsh, True)
        self.assertLess(cpu_cores, 1)  # can't really run job with less than 1 cpu ^o^

        # Orca: test 4
        # Test troubleshooting too many cpu cores
        # Automatically reduce cpu cores
        label = 'test'
        level_of_theory = {'method': 'dlpno-ccsd(T)'}
        server = 'server1'
        job_type = 'sp'
        software = 'orca'
        fine = True
        memory_gb = 16
        cpu_cores = 16
        num_heavy_atoms = 1
        ess_trsh_methods = ['cpu']
        path = os.path.join(self.base_path['orca'], 'orca_too_many_cores.log')
        status, keywords, error, line = trsh.determine_ess_status(output_path=path, species_label='TS', job_type='sp',
                                                                  software=software)
        job_status = {'keywords': keywords, 'error': error}
        output_errors, ess_trsh_methods, remove_checkfile, level_of_theory, software, job_type, fine, trsh_keyword, \
            memory, shift, cpu_cores, couldnt_trsh = trsh.trsh_ess_job(label, level_of_theory, server, job_status,
                                                                       job_type, software, fine, memory_gb,
                                                                       num_heavy_atoms, cpu_cores, ess_trsh_methods)
        self.assertIn('cpu', ess_trsh_methods)
        self.assertEqual(cpu_cores, 10)

    def test_determine_job_log_memory_issues(self):
        """Test the determine_job_log_memory_issues() function."""
        job_log_path_1 = os.path.join(ARC_PATH, 'arc', 'testing', 'job_log', 'no_issues.log')
        keywords, error, line = trsh.determine_job_log_memory_issues(job_log=job_log_path_1)
        self.assertEqual(keywords, [])
        self.assertEqual(error, '')
        self.assertEqual(line, '')

        job_log_path_2 = os.path.join(ARC_PATH, 'arc', 'testing', 'job_log', 'memory_exceeded.log')
        keywords, error, line = trsh.determine_job_log_memory_issues(job_log=job_log_path_2)
        self.assertEqual(keywords, ['Memory'])
        self.assertEqual(error, 'Insufficient job memory.')
        self.assertEqual(line, '\tMEMORY EXCEEDED\n')

        job_log_path_3 = os.path.join(ARC_PATH, 'arc', 'testing', 'job_log', 'using_to_few.log')
        keywords, error, line = trsh.determine_job_log_memory_issues(job_log=job_log_path_3)
        self.assertEqual(keywords, ['Memory'])
        self.assertIn('Memory requested is too high, used only', error)
        self.assertEqual(line, '\tJob Is Wasting Memory using less than 20 percent of requested Memory\n')

        with open(job_log_path_3, 'r') as f:
            job_log_content = f.read()
        keywords, error, line = trsh.determine_job_log_memory_issues(job_log=job_log_content)
        self.assertEqual(keywords, ['Memory'])
        self.assertIn('Memory requested is too high, used only', error)
        self.assertEqual(line, '\tJob Is Wasting Memory using less than 20 percent of requested Memory')

    def test_trsh_negative_freq(self):
        """Test troubleshooting a negative frequency"""
        gaussian_neg_freq_path = os.path.join(ARC_PATH, 'arc', 'testing', 'freq', 'Gaussian_neg_freq.out')
        current_neg_freqs_trshed, conformers, output_errors, output_warnings = \
            trsh.trsh_negative_freq(label='2-methoxy_n-methylaniline', log_file=gaussian_neg_freq_path)
        expected_current_neg_freqs_trshed = [-18.07]
        self.assertEqual(current_neg_freqs_trshed, expected_current_neg_freqs_trshed)
        expected_conformers = [{'symbols': ('C', 'N', 'C', 'C', 'O', 'C', 'C', 'C', 'C', 'C',
                                            'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'),
                                'isotopes': (12, 14, 12, 12, 16, 12, 12, 12, 12, 12, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
                                'coords': ((0.400778, 2.986759, -1.1916082863405995), (1.11819, 1.945985, -0.5248157036337059),
                                           (1.261692, 2.029477, 0.8795247459621556), (1.979036, 0.991891, 1.5241542217350892),
                                           (2.542752, -0.092156, 0.8416737861535863), (2.536206, -0.416927, -0.09758688694035716),
                                           (2.135627, 1.068178, 2.898943967697245), (1.619002, 2.117912, 3.7039782540378443),
                                           (0.917097, 3.141999, 3.151959048454133), (0.746925, 3.095369, 1.7965740484541326),
                                           (0.8924730978405825, 3.9642684755398547, -0.9936570489202912),
                                           (-0.6556710978405824, 3.0794535244601455, -0.7730690489202912),
                                           (0.377648, 2.76298, -2.2247747201413106), (2.954216133097816, 0.3442527133097816, -0.8289867485670152),
                                           (3.085125, -1.355514, -0.30031251079708793), (1.597416866902184, -0.5362537133097817, -0.6113597485670151),
                                           (2.680608, 0.26868, 3.414919566548908), (1.758456, 2.146776, 4.770743762230072),
                                           (0.510864, 3.967211, 3.6711780068315285), (0.203748, 3.890122, 1.2490044823713833))},
                               {'symbols': ('C', 'N', 'C', 'C', 'O', 'C', 'C', 'C', 'C', 'C',
                                            'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'),
                                'isotopes': (12, 14, 12, 12, 16, 12, 12, 12, 12, 12, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
                                'coords': ((0.400778, 2.986759, -1.0357237136594006), (1.11819, 1.945985, -0.337712296366294),
                                           (1.261692, 2.029477, 0.8968452540378444), (1.979036, 0.991891, 1.6453977782649107),
                                           (2.542752, -0.092156, 1.2416102138464136), (2.536206, -0.416927, -0.5825611130596429),
                                           (2.135627, 1.068178, 3.037508032302755), (1.619002, 2.117912, 3.6866577459621555),
                                           (0.917097, 3.141999, 2.9441129515458675), (0.746925, 3.095369, 1.5887279515458672),
                                           (0.8523169021594176, 3.9743075244601456, -0.9735789510797088),
                                           (-0.6155149021594175, 3.0694144755398547, -0.7529909510797088),
                                           (0.377648, 2.76298, -2.1344232798586895), (3.104801866902184, 0.3593112866902184, -1.0347872514329848),
                                           (3.085125, -1.355514, -0.501093489202912), (1.446831133097816, -0.5513122866902184, -0.8171602514329849),
                                           (2.680608, 0.26868, 3.490212433451092), (1.758456, 2.146776, 4.765724237769927),
                                           (0.510864, 3.967211, 3.565767993168471), (0.203748, 3.890122, 1.1536335176286168))}]
        self.assertEqual(conformers, expected_conformers)
        self.assertEqual(output_errors, list())
        self.assertEqual(output_warnings, list())

    def test_scan_quality_check(self):
        """Test scan quality check for 1D rotor"""
        log_file = os.path.join(ARC_PATH, 'arc', 'testing', 'rotor_scans', 'CH2OOH.out')
        # Case 1: non-smooth scan which troubleshot once
        case1 = {'label': 'CH2OOH',
                 'pivots': [1, 2],
                 'energies': parse_1d_scan_energies(log_file)[0],
                 'scan_res': 4.0,
                 'used_methods': None,
                 'log_file': log_file,
                 }
        invalidate, invalidation_reason, message, actions = trsh.scan_quality_check(**case1)
        self.assertTrue(invalidate)
        self.assertEqual(
            invalidation_reason, 'Significant difference observed between consecutive conformers')
        expect_message = 'Rotor scan of CH2OOH between pivots [1, 2] is inconsistent between ' \
                         'two consecutive conformers.\nInconsistent consecutive conformers and ' \
                         'problematic internal coordinates:\nconformer # 80 / # 81        D3, D2\n' \
                         'ARC will attempt to troubleshoot this rotor scan.'
        self.assertEqual(message, expect_message)
        self.assertEqual(len(actions.keys()), 1)
        self.assertIn('freeze', actions)
        self.assertIn([5, 1, 2, 3], actions['freeze'])
        self.assertIn([2, 1, 4, 5], actions['freeze'])

        # Case 2: Lower conformer
        log_file = os.path.join(ARC_PATH, 'arc', 'testing', 'rotor_scans', 'COCCOO.out')
        case2 = {'label': 'COCCOO',
                 'pivots': [2, 5],
                 'energies': parse_1d_scan_energies(log_file)[0],
                 'scan_res': 8.0,
                 'used_methods': None,
                 'log_file': log_file,
                 }
        invalidate, invalidation_reason, message, actions = trsh.scan_quality_check(**case2)
        self.assertTrue(invalidate)
        self.assertEqual(
            invalidation_reason, 'Another conformer for COCCOO exists which is 4.60 kJ/mol lower.')
        expect_message = 'Species COCCOO is not oriented correctly around pivots [2, 5], ' \
                         'searching for a better conformation...'
        self.assertEqual(message, expect_message)
        xyz = {'symbols': ('O', 'O', 'O', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'),
               'isotopes': (16, 16, 16, 12, 12, 12, 1, 1, 1, 1, 1, 1, 1, 1),
               'coords': ((1.064082, 0.653765, -0.451343),
                          (-2.342701, 0.031994, 0.662511),
                          (-2.398473, -1.385822, 0.327886),
                          (0.076296, -0.002042, 0.321439),
                          (-1.266646, 0.597254, -0.073572),
                          (2.370241, 0.177374, -0.19811),
                          (0.246556, 0.151577, 1.397399),
                          (0.074611, -1.082893, 0.12612),
                          (-1.343968, 1.669387, 0.129745),
                          (-1.395784, 0.428829, -1.147694),
                          (3.049182, 0.738271, -0.841111),
                          (2.661092, 0.333101, 0.85075),
                          (2.461024, -0.893605, -0.429469),
                          (-3.255509, -1.417186, -0.119474))}
        self.assertEqual(actions, {'change conformer': xyz})
        self.assertEqual(actions, {"change conformer": xyz})

    def test_trsh_scan_job(self):
        """Test troubleshooting problematic 1D rotor scan"""
        case = {
            "label": "CH2OOH",
            "scan_res": 4.0,
            "scan": [4, 1, 2, 3],
            "scan_list": [[4, 1, 2, 3], [1, 2, 3, 6]],
            "methods": {"freeze": [[5, 1, 2, 3], [2, 1, 4, 5]]},
            "log_file": os.path.join(
                ARC_PATH, "arc", "testing", "rotor_scans", "CH2OOH.out"
            ),
        }
        scan_trsh, scan_res = trsh.trsh_scan_job(**case)
        self.assertEqual(scan_trsh, "D 5 4 1 2 F\nD 1 2 3 6 F\nB 2 3 F\n")
        self.assertEqual(scan_res, 4.0)

    @patch(
        "arc.job.trsh.servers",
        {
            "test_server": {
                "cluster_soft": "PBS",
                "un": "test_user",
                "queues": {"short_queue": "24:00:0","middle_queue": "48:00:00", "long_queue": "3600:00:00"},
            }
        },
    )  
    @patch(
        "arc.job.trsh.execute_command"
    )
    def test_user_queue_setting_trsh(self, mock_execute_command):
        """ Test the trsh_job_queue function with user specified queue """
        # Mocking the groups and qstat command outputs
        mock_execute_command.side_effect = [
            (["users group1"], []),  # Simulates 'groups' command output
            (
                ["Queue Memory CPU Time Walltime Node Run Que Lm State"],
                [],
            ),  # Simulates 'qstat' command output
        ]

        # Call the trsh_job_queue function with test data
        result, success = trsh.trsh_job_queue("test_server", "test_job", 24)

        # Assertions
        self.assertIn("short_queue", result)
        self.assertIn("long_queue", result)
        self.assertTrue(success)

        # Now put in 'short_queue' in attempted_queues
        result, success = trsh.trsh_job_queue(
            "test_server", "test_job", 24, attempted_queues=["short_queue"]
        )

        # Assertions
        self.assertNotIn("short_queue", result)
        self.assertIn("long_queue", result)
        self.assertTrue(success)

        # Now put in 'long_queue' in attempted_queues
        result, success = trsh.trsh_job_queue(
            "test_server", "test_job", 24, attempted_queues=["long_queue"]
        )

        # Assertions
        self.assertIn("short_queue", result)
        self.assertNotIn("long_queue", result)
        self.assertTrue(success)

    @patch('arc.job.trsh.servers', {
        'test_server': {
            'cluster_soft': 'PBS',
            'un': 'test_user',
            'queue': {},
        }
    })
    @patch('arc.job.trsh.execute_command')
    def test_query_pbs_trsh_job_queue(self, mock_execute_command):
        """ Test the trsh_job_queue function with PBS queue """
        # Setting up the mock responses for execute_command
        mock_execute_command.side_effect = [
            (["users group1"], []),
            (["Queue Memory CPU Time Walltime Node Run Que Lm State",
              "---------------- ------ -------- -------- ---- ----- ----- ----  -----",
              "workq -- -- -- -- 0 0 -- D S",
              "maytal_q -- -- -- -- 7 0 -- E R",
              # ... add other queue lines as needed
              ], []),  # Simulates 'qstat -q' command output
            # Simulate 'qstat -Qf {queue_name}' for each queue
            (["Queue: maytal_q", "other info", "resources_default.walltime = 48:00:00", "acl_groups = group1"], []),  # For maytal_q
        ]

        # Call the trsh_job_queue function with test data
        result, success = trsh.trsh_job_queue("test_server", "test_job", 24, attempted_queues=None)

        # Assertions to verify function behavior
        self.assertIsNotNone(result)
        self.assertIn('maytal_q', result.keys())
        self.assertIn('48:00:00', result.values())
        self.assertTrue(success)


if __name__ == "__main__":
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
