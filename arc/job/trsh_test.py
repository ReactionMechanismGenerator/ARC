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
from arc.parser.parser import parse_1d_scan_energies

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
        path = os.path.join(self.base_path["orca"], "O2_MRCI.log")
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
                                'coords': ((1.594007, 2.403416, -0.07830928634059947),
                                           (1.38126, 0.982746, -0.09401470363370595),
                                           (0.132808, 0.524622, -0.008713254037844386),
                                           (-0.094298, -0.898343, -0.060980778264910704),
                                           (0.838851, -1.856028, -0.2008542138464136),
                                           (2.24123, -1.636965, 0.24357411305964283),
                                           (-1.398852, -1.390697, -0.06968303230275509),
                                           (-2.487721, -0.546972, 0.008673254037844387),
                                           (-2.300616, 0.840612, 0.10443604845413262),
                                           (-1.033759, 1.353715, 0.10441904845413262),
                                           (1.1793300978405823, 2.8826114755398544, 0.8734619510797088),
                                           (1.1385669021594176, 2.8928465244601456, -0.8938670489202911),
                                           (2.665025, 2.602779, -0.04588672014131042),
                                           (2.4813881330978163, -1.0934802866902185, 0.9879432514329849),
                                           (2.672003, -2.637149, 0.10192048920291205),
                                           (2.634424866902184, -1.0780867133097816, -0.7788617485670152),
                                           (-1.525589, -2.465278, -0.03837743345109202),
                                           (-3.487346, -0.961916, 0.002509762230072801),
                                           (-3.155537, 1.504321, 0.053595006831528826),
                                           (-0.898688, 2.426019, 0.04851648237138322))},
                               {'symbols': ('C', 'N', 'C', 'C', 'O', 'C', 'C', 'C', 'C', 'C',
                                            'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'),
                                'isotopes': (12, 14, 12, 12, 16, 12, 12, 12, 12, 12, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
                                'coords': ((1.594007, 2.403416, 0.07757528634059946),
                                           (1.38126, 0.982746, 0.09308870363370594),
                                           (0.132808, 0.524622, 0.008607254037844387),
                                           (-0.094298, -0.898343, 0.06026277826491071),
                                           (0.838851, -1.856028, 0.1990822138464136),
                                           (2.24123, -1.636965, -0.24140011305964282),
                                           (-1.398852, -1.390697, 0.0688810323027551),
                                           (-2.487721, -0.546972, -0.008647254037844386),
                                           (-2.300616, 0.840612, -0.10341004845413262),
                                           (-1.033759, 1.353715, -0.10342704845413263),
                                           (1.1391739021594176, 2.8926505244601453, 0.8935400489202912),
                                           (1.1787230978405823, 2.8828074755398547, -0.8737889510797088),
                                           (2.665025, 2.602779, 0.044464720141310414),
                                           (2.631973866902184, -1.0784217133097818, 0.7821427485670152),
                                           (2.672003, -2.637149, -0.09886048920291204),
                                           (2.483839133097816, -1.0931452866902183, -0.9846622514329849),
                                           (-1.525589, -2.465278, 0.036915433451092015),
                                           (-3.487346, -0.961916, -0.002509762230072801),
                                           (-3.155537, 1.504321, -0.05181500683152882),
                                           (-0.898688, 2.426019, -0.046854482371383226))}]
        self.assertEqual(conformers, expected_conformers)
        self.assertEqual(output_errors, list())
        self.assertEqual(output_warnings, list())

    def test_scan_quality_check(self):
        """Test scan quality check for 1D rotor"""
        log_file = os.path.join(ARC_PATH, 'arc', 'testing', 'rotor_scans', 'CH2OOH.out')
        # Case 1: non-smooth scan which troubleshot once
        case1 = {'label': 'CH2OOH',
                 'pivots': [1, 2],
                 'energies': parse_1d_scan_energies(log_file_path=log_file)[0],
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
                 'energies': parse_1d_scan_energies(log_file_path=log_file)[0],
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
               'coords': ((-1.639663, -0.368391, -0.311199), (1.930494, -0.414242, 0.395362),
                          (2.336326, 0.860359, -0.182795), (-0.455544, 0.118546, 0.291877),
                          (0.685253, -0.766654, -0.191387), (-2.786077, 0.380303, 0.038396),
                          (-0.519751, 0.071124, 1.38908), (-0.265861, 1.161629, 0.004955),
                          (0.524003, -1.798434, 0.134335), (0.752145, -0.745698, -1.283946),
                          (-3.635246, -0.06902, -0.477369), (-2.971297, 0.353777, 1.121854),
                          (-2.692729, 1.430409, -0.273456), (3.12969, 0.581231, -0.659714))}
        self.assertEqual(actions, {'change conformer': xyz})

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
