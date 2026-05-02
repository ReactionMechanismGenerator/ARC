#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.job.adapters.molpro module
"""

import os
import shutil
import unittest

from arc.common import ARC_TESTING_PATH
from arc.job.adapters.molpro import MolproAdapter
from arc.level import Level
from arc.settings.settings import input_filenames, output_filenames
from arc.species import ARCSpecies


class TestMolproAdapter(unittest.TestCase):
    """
    Contains unit tests for the MolproAdapter class.
    """
    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None
        cls.job_1 = MolproAdapter(execution_type='queue',
                                  job_type='sp',
                                  level=Level(method='CCSD(T)-F12', basis='cc-pVTZ-f12'),
                                  project='test',
                                  project_directory=os.path.join(ARC_TESTING_PATH, 'test_MolproAdapter_1'),
                                  species=[ARCSpecies(label='spc1', xyz=['O 0 0 1'], multiplicity=3)],
                                  testing=True,
                                  )
        cls.job_2 = MolproAdapter(execution_type='queue',
                                  job_type='opt',
                                  level=Level(method='CCSD(T)', basis='cc-pVQZ'),
                                  project='test',
                                  project_directory=os.path.join(ARC_TESTING_PATH, 'test_MolproAdapter_2'),
                                  species=[ARCSpecies(label='spc1', xyz=['O 0 0 1'], multiplicity=3)],
                                  testing=True,
                                  )
        cls.job_3 = MolproAdapter(execution_type='queue',
                                  job_type='sp',
                                  level=Level(method='MRCI', basis='aug-cc-pvtz-f12'),
                                  project='test',
                                  project_directory=os.path.join(ARC_TESTING_PATH, 'test_MolproAdapter_3'),
                                  species=[ARCSpecies(label='HNO_t', xyz=["""N     -0.08142    0.37454    0.00000
                                                                             O      1.01258   -0.17285    0.00000
                                                                             H     -0.93116   -0.20169    0.00000"""],
                                                      multiplicity=3)],
                                  testing=True,
                                  )
        cls.job_4 = MolproAdapter(execution_type='queue',
                                  job_type='sp',
                                  level=Level(method='MRCI-F12', basis='aug-cc-pvtz-f12'),
                                  project='test',
                                  project_directory=os.path.join(ARC_TESTING_PATH, 'test_MolproAdapter_4'),
                                  species=[ARCSpecies(label='HNO_t', xyz=["""N     -0.08142    0.37454    0.00000
                                                                             O      1.01258   -0.17285    0.00000
                                                                             H     -0.93116   -0.20169    0.00000"""],
                                                      multiplicity=3)],
                                  testing=True,
                                  )
        cls.job_5 = MolproAdapter(execution_type='queue',
                                  job_type='sp',
                                  level=Level(method='MP2_CASSCF_MRCI-F12', basis='aug-cc-pVTZ-F12'),
                                  project='test',
                                  project_directory=os.path.join(ARC_TESTING_PATH, 'test_MolproAdapter_5'),
                                  species=[ARCSpecies(label='HNO_t', xyz=["""N     -0.08142    0.37454    0.00000
                                                                             O      1.01258   -0.17285    0.00000
                                                                             H     -0.93116   -0.20169    0.00000"""],
                                                      multiplicity=3)],
                                  testing=True,
                                  )
        cls.job_6 = MolproAdapter(execution_type='queue',
                                  job_type='sp',
                                  level=Level(method='MP2_CASSCF_RS2C', basis='aug-cc-pVTZ'),  # CASPT2
                                  project='test',
                                  project_directory=os.path.join(ARC_TESTING_PATH, 'test_MolproAdapter_6'),
                                  species=[ARCSpecies(label='HNO_t', xyz=["""N     -0.08142    0.37454    0.00000
                                                                             O      1.01258   -0.17285    0.00000
                                                                             H     -0.93116   -0.20169    0.00000"""],
                                                      multiplicity=3)],
                                  testing=True,
                                  )
        cls.job_7 = MolproAdapter(execution_type='queue',
                                  job_type='sp',
                                  level=Level(method='MP2_CASSCF_RS2C', basis='aug-cc-pVTZ'),  # CASPT2
                                  project='test',
                                  project_directory=os.path.join(ARC_TESTING_PATH, 'test_MolproAdapter_7'),
                                  species=[ARCSpecies(label='N', xyz=["""N     0.0    0.0    0.0"""],
                                                      multiplicity=3,
                                                      active={'occ': [3, 1, 1, 0, 1, 0, 0, 0],
                                                              'closed': [1, 0, 0, 0, 0, 0, 0, 0]})],
                                  testing=True,
                                  )
        cls.job_mrcc_ccsdt = MolproAdapter(execution_type='queue',
                                           job_type='sp',
                                           level=Level(method='CCSDT', basis='cc-pVDZ'),
                                           project='test',
                                           project_directory=os.path.join(ARC_TESTING_PATH,
                                                                          'test_MolproAdapter_mrcc_ccsdt'),
                                           species=[ARCSpecies(label='spc1', xyz=['O 0 0 1'], multiplicity=3)],
                                           testing=True,
                                           )
        cls.job_mrcc_ccsdtq = MolproAdapter(execution_type='queue',
                                            job_type='sp',
                                            level=Level(method='CCSDT(Q)', basis='cc-pVDZ'),
                                            project='test',
                                            project_directory=os.path.join(ARC_TESTING_PATH,
                                                                           'test_MolproAdapter_mrcc_ccsdtq'),
                                            species=[ARCSpecies(label='spc1', xyz=['O 0 0 1'], multiplicity=1)],
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
        self.assertEqual(self.job_1.input_file_memory, 438)

        self.job_1.cpu_cores = 8
        self.job_1.set_input_file_memory()
        self.assertEqual(self.job_1.input_file_memory, 438)

        self.job_1.input_file_memory = None
        self.job_1.cpu_cores = 1
        self.job_1.set_input_file_memory()
        self.assertEqual(self.job_1.input_file_memory, 438)

    def test_write_input_file(self):
        """Test writing Molpro input files"""
        self.job_1.cpu_cores = 48
        self.job_1.set_input_file_memory()
        self.job_1.write_input_file()
        with open(os.path.join(self.job_1.local_path, input_filenames[self.job_1.job_adapter]), 'r') as f:
            content_1 = f.read()
        job_1_expected_input_file = """***,spc1
memory,Total=438,m;

geometry={angstrom;
O       0.00000000    0.00000000    1.00000000}

gprint,orbitals;

basis=cc-pvtz-f12



int;

{hf;
 maxit,999;
 wf,spin=2,charge=0;
}

uccsd(t)-f12;



---;

"""
        self.assertEqual(content_1, job_1_expected_input_file)

        self.job_2.cpu_cores = 48
        self.job_2.set_input_file_memory()
        self.job_2.write_input_file()
        with open(os.path.join(self.job_2.local_path, input_filenames[self.job_2.job_adapter]), 'r') as f:
            content_2 = f.read()
        job_2_expected_input_file = """***,spc1
memory,Total=438,m;

geometry={angstrom;
O       0.00000000    0.00000000    1.00000000}

gprint,orbitals;

basis=cc-pvqz



int;

{hf;
 maxit,999;
 wf,spin=2,charge=0;
}

uccsd(t);

optg, savexyz='geometry.xyz'

---;

"""
        self.assertEqual(content_2, job_2_expected_input_file)

    def test_write_mrci_input_file(self):
        """Test writing MRCI Molpro input files"""
        self.job_3.cpu_cores = 48
        self.job_3.set_input_file_memory()
        self.job_3.write_input_file()
        with open(os.path.join(self.job_3.local_path, input_filenames[self.job_3.job_adapter]), 'r') as f:
            content_3 = f.read()
        job_3_expected_input_file = """***,HNO_t
memory,Total=438,m;

geometry={angstrom;
N      -0.08142000    0.37454000    0.00000000
O       1.01258000   -0.17285000    0.00000000
H      -0.93116000   -0.20169000    0.00000000}

gprint,orbitals;

basis=aug-cc-pvtz-f12



int;

{hf;
 maxit,999;
 wf,spin=2,charge=0;
}

{casscf;
 maxit,999;
 wf,spin=2,charge=0;
}

{mrci;
 maxit,999;
 wf,spin=2,charge=0;
}




E_mrci=energy;
E_mrci_Davidson=energd;

table,E_mrci,E_mrci_Davidson;
---;

"""
        self.assertEqual(content_3, job_3_expected_input_file)

        self.job_4.cpu_cores = 48
        self.job_4.set_input_file_memory()
        self.job_4.write_input_file()
        with open(os.path.join(self.job_4.local_path, input_filenames[self.job_4.job_adapter]), 'r') as f:
            content_4 = f.read()
        job_4_expected_input_file = """***,HNO_t
memory,Total=438,m;

geometry={angstrom;
N      -0.08142000    0.37454000    0.00000000
O       1.01258000   -0.17285000    0.00000000
H      -0.93116000   -0.20169000    0.00000000}

gprint,orbitals;

basis=aug-cc-pvtz-f12



int;

{hf;
 maxit,999;
 wf,spin=2,charge=0;
}

{casscf;
 maxit,999;
 wf,spin=2,charge=0;
}

{mrci-f12;
 maxit,999;
 wf,spin=2,charge=0;
}




E_mrci=energy;
E_mrci_Davidson=energd;

table,E_mrci,E_mrci_Davidson;
---;

"""
        self.assertEqual(content_4, job_4_expected_input_file)

        self.job_5.cpu_cores = 48
        self.job_5.set_input_file_memory()
        self.job_5.write_input_file()
        with open(os.path.join(self.job_5.local_path, input_filenames[self.job_5.job_adapter]), 'r') as f:
            content_5 = f.read()
        job_5_expected_input_file = """***,HNO_t
memory,Total=438,m;

geometry={angstrom;
N      -0.08142000    0.37454000    0.00000000
O       1.01258000   -0.17285000    0.00000000
H      -0.93116000   -0.20169000    0.00000000}

gprint,orbitals;

basis=aug-cc-pvtz-f12



int;

{hf;
 maxit,999;
 wf,spin=2,charge=0;
}


{mp2;
 wf,spin=2,charge=0;
}

{casscf;
 maxit,999;
 wf,spin=2,charge=0;
}

{mrci-f12;
 maxit,999;
 wf,spin=2,charge=0;
}





E_mrci=energy;
E_mrci_Davidson=energd;

table,E_mrci,E_mrci_Davidson;
---;

"""
        self.assertEqual(content_5, job_5_expected_input_file)

        self.job_6.cpu_cores = 48
        self.job_6.set_input_file_memory()
        self.job_6.write_input_file()
        with open(os.path.join(self.job_6.local_path, input_filenames[self.job_6.job_adapter]), 'r') as f:
            content_6 = f.read()
        job_6_expected_input_file = """***,HNO_t
memory,Total=438,m;

geometry={angstrom;
N      -0.08142000    0.37454000    0.00000000
O       1.01258000   -0.17285000    0.00000000
H      -0.93116000   -0.20169000    0.00000000}

gprint,orbitals;

basis=aug-cc-pvtz



int;

{hf;
 maxit,999;
 wf,spin=2,charge=0;
}


{mp2;
 wf,spin=2,charge=0;
}

{casscf;
 maxit,999;
 wf,spin=2,charge=0;
}

{rs2c;
 maxit,999;
 wf,spin=2,charge=0;
}




---;

"""
        self.assertEqual(content_6, job_6_expected_input_file)

        self.job_7.cpu_cores = 48
        self.job_7.set_input_file_memory()
        self.job_7.write_input_file()
        with open(os.path.join(self.job_7.local_path, input_filenames[self.job_7.job_adapter]), 'r') as f:
            content_7 = f.read()
        job_7_expected_input_file = """***,N
memory,Total=438,m;

geometry={angstrom;
N       0.00000000    0.00000000    0.00000000}

gprint,orbitals;

basis=aug-cc-pvtz



int;

{hf;
 maxit,999;
 wf,spin=2,charge=0;
}


{mp2;
 wf,spin=2,charge=0;
}

{casscf;
 maxit,999;
 wf,spin=2,charge=0;
 occ,3,1,1,0,1,0,0,0;
 closed,1,0,0,0,0,0,0,0;
 state,1;
}

{rs2c;
 maxit,999;
 wf,spin=2,charge=0;
}




---;

"""
        self.assertEqual(content_7, job_7_expected_input_file)

    def test_write_input_file_mrcc_routing(self):
        """Methods unsupported by native Molpro but supported by MRCC are routed through the MRCC plugin.

        For an open-shell wavefunction, the SCF reference is switched from
        ``{hf;...}`` (which gives Molpro's ROHF for open-shell) to
        ``{uhf;...}``. MRCC's approximate-CC family (``CCSDT(Q)``,
        ``CCSDTQ(P)``, and the perturbative-``(T)`` variants) refuses
        standard ROHF orbitals with the error::

            Approximate CC methods are not implemented for standard ROHF orbitals!
            Use semicanonical orbitals!

        UHF orbitals are semicanonical by construction (alpha and beta Fock
        matrices are separately diagonal), saved to the default record 2100.2
        which MRCC reads — MRCC then reports ``Type=UHF/CANONICAL`` and runs
        the requested approximate-CC method.
        """
        self.job_mrcc_ccsdt.cpu_cores = 48
        self.job_mrcc_ccsdt.set_input_file_memory()
        self.job_mrcc_ccsdt.write_input_file()
        with open(os.path.join(self.job_mrcc_ccsdt.local_path,
                               input_filenames[self.job_mrcc_ccsdt.job_adapter]), 'r') as f:
            content_ccsdt = f.read()
        # spc1 has multiplicity=3 (open-shell triplet) — UHF reference expected.
        expected_ccsdt = """***,spc1
memory,Total=438,m;

geometry={angstrom;
O       0.00000000    0.00000000    1.00000000}

gprint,orbitals;

basis=cc-pvdz



int;

{uhf;
 maxit,999;
 wf,spin=2,charge=0;
}

{mrcc,method=CCSDT}



---;

"""
        self.assertEqual(content_ccsdt, expected_ccsdt)
        # Sanity: the bare directive Molpro rejects must NOT appear on its own line.
        self.assertNotIn('\nccsdt;\n', content_ccsdt)
        self.assertNotIn('\nuccsdt;\n', content_ccsdt)
        # An earlier (insufficient) fix used `{uccsd}` between HF and MRCC —
        # this contract has been replaced with UHF, so {uccsd} must NOT appear.
        self.assertNotIn('{uccsd}', content_ccsdt)
        # UHF must replace HF as the only SCF reference (no {hf;...} block).
        self.assertNotIn('{hf;', content_ccsdt)
        self.assertIn('{uhf;', content_ccsdt)

        self.job_mrcc_ccsdtq.cpu_cores = 48
        self.job_mrcc_ccsdtq.set_input_file_memory()
        self.job_mrcc_ccsdtq.write_input_file()
        with open(os.path.join(self.job_mrcc_ccsdtq.local_path,
                               input_filenames[self.job_mrcc_ccsdtq.job_adapter]), 'r') as f:
            content_ccsdtq = f.read()
        expected_ccsdtq = """***,spc1
memory,Total=438,m;

geometry={angstrom;
O       0.00000000    0.00000000    1.00000000}

gprint,orbitals;

basis=cc-pvdz



int;

{hf;
 maxit,999;
 wf,spin=0,charge=0;
}

{mrcc,method=CCSDT(Q)}



---;

"""
        self.assertEqual(content_ccsdtq, expected_ccsdtq)
        self.assertNotIn('\nccsdt(q);\n', content_ccsdtq)
        # spc1 here has multiplicity=1 (closed-shell) — RHF gives canonical
        # orbitals MRCC accepts directly. No UHF/UCCSD pre-step needed.
        self.assertNotIn('{uccsd}', content_ccsdtq)
        self.assertNotIn('{uhf;', content_ccsdtq)
        self.assertIn('{hf;', content_ccsdtq)

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
        job_1_files_to_download = [{'file_name': 'input.out',
                                    'local': os.path.join(self.job_1.local_path, output_filenames[self.job_1.job_adapter]),
                                    'remote': os.path.join(self.job_1.remote_path, output_filenames[self.job_1.job_adapter]),
                                    'source': 'path',
                                    'make_x': False}]
        self.assertEqual(self.job_1.files_to_upload, job_1_files_to_upload)
        self.assertEqual(self.job_1.files_to_download, job_1_files_to_download)

    @classmethod
    def tearDownClass(cls):
        """
        A function that is run ONCE after all unit tests in this class.
        Delete all project directories created during these unit tests
        """
        for attr in vars(cls).values():
            if isinstance(attr, MolproAdapter):
                shutil.rmtree(attr.project_directory, ignore_errors=True)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
