#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module contains unit tests of the arc.species.species module
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)
import unittest
import os

from rmgpy.molecule.molecule import Molecule
from rmgpy.species import Species
from rmgpy.reaction import Reaction

from arc.species.species import ARCSpecies, TSGuess, get_min_energy_conformer,\
    determine_rotor_type, determine_rotor_symmetry, check_species_xyz
from arc.species.converter import get_xyz_string, get_xyz_matrix, molecules_from_xyz
from arc.settings import arc_path

################################################################################


class TestARCSpecies(unittest.TestCase):
    """
    Contains unit tests for the ARCSpecies class
    """
    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None
        # Method 1: RMG Species object (here by SMILES)
        cls.spc1_rmg = Species(molecule=[Molecule().fromSMILES(str('C=C[O]'))])  # delocalized radical + amine
        cls.spc1_rmg.label = str('vinoxy')
        cls.spc1 = ARCSpecies(rmg_species=cls.spc1_rmg)

        # Method 2: ARCSpecies object by XYZ (also give SMILES for thermo BAC)
        oh_xyz = str("""O       0.00000000    0.00000000   -0.12002167
        H       0.00000000    0.00000000    0.85098324""")
        cls.spc2 = ARCSpecies(label=str('OH'), xyz=oh_xyz, smiles=str('[OH]'), multiplicity=2, charge=0)

        # Method 3: ARCSpecies object by SMILES
        cls.spc3 = ARCSpecies(label=str('methylamine'), smiles=str('CN'), multiplicity=1, charge=0)

        # Method 4: ARCSpecies object by RMG Molecule object
        mol4 = Molecule().fromSMILES(str('C=CC'))
        cls.spc4 = ARCSpecies(label=str('propene'), mol=mol4, multiplicity=1, charge=0)

        # Method 5: ARCSpecies by AdjacencyList (to generate AdjLists, see https://rmg.mit.edu/molecule_search)
        n2h4_adj = str("""1 N u0 p1 c0 {2,S} {3,S} {4,S}
        2 N u0 p1 c0 {1,S} {5,S} {6,S}
        3 H u0 p0 c0 {1,S}
        4 H u0 p0 c0 {1,S}
        5 H u0 p0 c0 {2,S}
        6 H u0 p0 c0 {2,S}""")
        cls.spc5 = ARCSpecies(label=str('N2H4'), adjlist=n2h4_adj, multiplicity=1, charge=0)

        n3_xyz = str("""N      -1.1997440839    -0.1610052059     0.0274738287
        H      -1.4016624407    -0.6229695533    -0.8487034080
        H      -0.0000018759     1.2861082773     0.5926077870
        N       0.0000008520     0.5651072858    -0.1124621525
        H      -1.1294692206    -0.8709078271     0.7537518889
        N       1.1997613019    -0.1609980472     0.0274604887
        H       1.1294795781    -0.8708998550     0.7537444446
        H       1.4015274689    -0.6230592706    -0.8487058662""")
        cls.spc6 = ARCSpecies(label=str('N3'), xyz=n3_xyz, multiplicity=1, charge=0, smiles=str('NNN'))

        xyz1 = os.path.join(arc_path, 'arc', 'testing', 'xyz', 'AIBN.gjf')
        cls.spc7 = ARCSpecies(label='AIBN', smiles=str('N#CC(C)(C)N=NC(C)(C)C#N'), xyz=xyz1)

        hso3_xyz = str("""S      -0.12383700    0.10918200   -0.21334200
        O       0.97332200   -0.98800100    0.31790100
        O      -1.41608500   -0.43976300    0.14487300
        O       0.32370100    1.42850400    0.21585900
        H       1.84477700   -0.57224200    0.35517700""")
        cls.spc8 = ARCSpecies(label=str('HSO3'), xyz=hso3_xyz, multiplicity=2, charge=0, smiles=str('O[S](=O)=O'))

        nh_s_adj = str("""1 N u0 p2 c0 {2,S}
                          2 H u0 p0 c0 {1,S}""")
        nh_s_xyz = str("""N       0.50949998    0.00000000    0.00000000
                          H      -0.50949998    0.00000000    0.00000000""")
        cls.spc9 = ARCSpecies(label=str('NH2(S)'), adjlist=nh_s_adj, xyz=nh_s_xyz, multiplicity=1, charge=0)

    def test_conformers(self):
        """Test conformer generation"""
        self.spc1.generate_conformers()  # vinoxy has two res. structures, each is assigned two conformers (RDkit/ob)
        self.assertEqual(len(self.spc1.conformers), 4)
        self.assertEqual(len(self.spc1.conformers), len(self.spc1.conformer_energies))

    def test_rmg_species_conversion_into_arc_species(self):
        """Test the conversion of an RMG species into an ARCSpecies"""
        self.spc1_rmg.label = None
        self.spc = ARCSpecies(rmg_species=self.spc1_rmg, label=str('vinoxy'))
        self.assertEqual(self.spc.label, str('vinoxy'))
        self.assertEqual(self.spc.multiplicity, 2)
        self.assertEqual(self.spc.charge, 0)

    def test_determine_rotors(self):
        """Test determination of rotors in ARCSpecies"""
        self.spc1.determine_rotors()
        self.spc2.determine_rotors()
        self.spc3.determine_rotors()
        self.spc4.determine_rotors()
        self.spc5.determine_rotors()
        self.spc6.determine_rotors()

        self.assertEqual(len(self.spc1.rotors_dict), 1)
        self.assertEqual(len(self.spc2.rotors_dict), 0)
        self.assertEqual(len(self.spc3.rotors_dict), 1)
        self.assertEqual(len(self.spc4.rotors_dict), 1)
        self.assertEqual(len(self.spc5.rotors_dict), 1)
        self.assertEqual(len(self.spc6.rotors_dict), 2)

        self.assertEqual(self.spc1.rotors_dict[0][str('pivots')], [2, 3])
        self.assertEqual(self.spc1.rotors_dict[0][str('scan')], [4, 2, 3, 1])
        self.assertTrue(all([t in [2, 4, 5] for t in self.spc1.rotors_dict[0][str('top')]]))
        self.assertEqual(self.spc1.rotors_dict[0][str('times_dihedral_set')], 0)
        self.assertEqual(self.spc3.rotors_dict[0][str('pivots')], [1, 2])
        self.assertEqual(self.spc4.rotors_dict[0][str('pivots')], [1, 2])
        self.assertEqual(self.spc5.rotors_dict[0][str('pivots')], [1, 2])
        self.assertEqual(self.spc6.rotors_dict[0][str('pivots')], [1, 4])
        self.assertEqual(self.spc6.rotors_dict[0][str('scan')], [2, 1, 4, 6])
        self.assertEqual(len(self.spc6.rotors_dict[0][str('top')]), 3)
        self.assertTrue(all([t in [1, 5, 2] for t in self.spc6.rotors_dict[0][str('top')]]))
        self.assertEqual(self.spc6.rotors_dict[1][str('pivots')], [4, 6])
        self.assertEqual(self.spc6.rotors_dict[1][str('scan')], [1, 4, 6, 7])
        self.assertEqual(len(self.spc6.rotors_dict[1][str('top')]), 3)
        self.assertTrue(all([t in [6, 7, 8] for t in self.spc6.rotors_dict[1][str('top')]]))

    def test_symmetry(self):
        """Test external symmetry and chirality determination"""
        allene = ARCSpecies(label=str('allene'), smiles=str('C=C=C'), multiplicity=1, charge=0)
        allene.final_xyz = """C  -1.01646   0.10640  -0.91445
                              H  -1.39000   1.03728  -1.16672
                              C   0.00000   0.00000   0.00000
                              C   1.01653  -0.10640   0.91438
                              H  -1.40975  -0.74420  -1.35206
                              H   0.79874  -0.20864   1.92036
                              H   2.00101  -0.08444   0.59842"""
        allene.determine_symmetry()
        self.assertEqual(allene.optical_isomers, 1)
        self.assertEqual(allene.external_symmetry, 4)

        ammonia = ARCSpecies(label=str('ammonia'), smiles=str('N'), multiplicity=1, charge=0)
        ammonia.final_xyz = """N  0.06617   0.20024   0.13886
                               H  -0.62578  -0.34119   0.63709
                               H  -0.32018   0.51306  -0.74036
                               H   0.87976  -0.37219  -0.03564"""
        ammonia.determine_symmetry()
        self.assertEqual(ammonia.optical_isomers, 1)
        self.assertEqual(ammonia.external_symmetry, 3)

        methane = ARCSpecies(label=str('methane'), smiles=str('C'), multiplicity=1, charge=0)
        methane.final_xyz = """C   0.00000   0.00000   0.00000
                               H  -0.29717   0.97009  -0.39841
                               H   1.08773  -0.06879   0.01517
                               H  -0.38523  -0.10991   1.01373
                               H -0.40533  -0.79140  -0.63049"""
        methane.determine_symmetry()
        self.assertEqual(methane.optical_isomers, 1)
        self.assertEqual(methane.external_symmetry, 12)

        chiral = ARCSpecies(label=str('chiral'), smiles=str('C(C)(O)(N)'), multiplicity=1, charge=0)
        chiral.final_xyz = """C                 -0.49341625    0.37828349    0.00442108
                              H                 -1.56331545    0.39193350    0.01003359
                              N                  0.01167132    1.06479568    1.20212111
                              H                  1.01157784    1.05203730    1.19687531
                              H                 -0.30960193    2.01178202    1.20391932
                              O                 -0.03399634   -0.97590449    0.00184366
                              H                 -0.36384913   -1.42423238   -0.78033350
                              C                  0.02253835    1.09779040   -1.25561654
                              H                 -0.34510997    0.59808430   -2.12741255
                              H                 -0.32122209    2.11106387   -1.25369100
                              H                  1.09243518    1.08414066   -1.26122530"""
        chiral.determine_symmetry()
        self.assertEqual(chiral.optical_isomers, 2)
        self.assertEqual(chiral.external_symmetry, 1)

        s8 = ARCSpecies(label=str('s8'), smiles=str('S1SSSSSSS1'), multiplicity=1, charge=0)
        s8.final_xyz = """S   2.38341   0.12608   0.09413
                          S   1.45489   1.88955  -0.13515
                          S  -0.07226   2.09247   1.14966
                          S  -1.81072   1.52327   0.32608
                          S  -2.23488  -0.39181   0.74645
                          S  -1.60342  -1.62383  -0.70542
                          S   0.22079  -2.35820  -0.30909
                          S   1.66220  -1.25754  -1.16665"""
        s8.determine_symmetry()
        self.assertEqual(s8.optical_isomers, 1)
        self.assertEqual(s8.external_symmetry, 8)

        water = ARCSpecies(label=str('H2O'), smiles=str('O'), multiplicity=1, charge=0)
        water.final_xyz = """O   0.19927   0.29049  -0.11186
                             H   0.50770  -0.61852  -0.09124
                             H  -0.70697   0.32803   0.20310"""
        water.determine_symmetry()
        self.assertEqual(water.optical_isomers, 1)
        self.assertEqual(water.external_symmetry, 2)

    def test_xyz_format_conversion(self):
        """Test conversions from string to list xyz formats"""
        xyz_str0 = """N       2.24690600   -0.00006500    0.11597700
C      -1.05654800    1.29155000   -0.02642500
C      -1.05661400   -1.29150400   -0.02650600
C      -0.30514100    0.00000200    0.00533200
C       1.08358900   -0.00003400    0.06558000
H      -0.39168300    2.15448600   -0.00132500
H      -1.67242600    1.35091400   -0.93175000
H      -1.74185400    1.35367700    0.82742800
H      -0.39187100   -2.15447800    0.00045500
H      -1.74341400   -1.35278100    0.82619100
H      -1.67091600   -1.35164600   -0.93286400
"""

        xyz_list, atoms, x, y, z = get_xyz_matrix(xyz_str0)

        # test all forms of input for get_xyz_string():
        xyz_str1 = get_xyz_string(xyz_list, symbol=atoms)
        xyz_str2 = get_xyz_string(xyz_list, number=[7, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1])
        mol, _ = molecules_from_xyz(xyz_str0)
        xyz_str3 = get_xyz_string(xyz_list, mol=mol)

        self.assertEqual(xyz_str0, xyz_str1)
        self.assertEqual(xyz_str1, xyz_str2)
        self.assertEqual(xyz_str2, xyz_str3)
        self.assertEqual(atoms, ['N', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H'])
        self.assertEqual(x, [2.246906, -1.056548, -1.056614, -0.305141, 1.083589, -0.391683, -1.672426, -1.741854,
                             -0.391871, -1.743414, -1.670916])
        self.assertEqual(y[1], 1.29155)
        self.assertEqual(z[-1], -0.932864)

    def test_is_linear(self):
        """Test determination of molecule linearity by xyz"""
        xyz1 = """C  0.000000    0.000000    0.000000
                  O  0.000000    0.000000    1.159076
                  O  0.000000    0.000000   -1.159076"""  # a trivial case
        xyz2 = """S      -0.06618943   -0.12360663   -0.07631983
                  O      -0.79539707    0.86755487    1.02675668
                  O      -0.68919931    0.25421823   -1.34830853
                  N       0.01546439   -1.54297548    0.44580391
                  C       1.59721519    0.47861334    0.00711000
                  H       1.94428095    0.40772394    1.03719428
                  H       2.20318015   -0.14715186   -0.64755729
                  H       1.59252246    1.51178950   -0.33908352
                  H      -0.87856890   -2.02453514    0.38494433
                  H      -1.34135876    1.49608206    0.53295071"""  # a non linear molecule
        xyz3 = """N  0.0000000000     0.0000000000     0.3146069129
                  O -1.0906813653     0.0000000000    -0.1376405244
                  O  1.0906813653     0.0000000000    -0.1376405244"""  # a non linear 3-atom molecule
        xyz4 = """N  0.0000000000     0.0000000000     0.1413439534
                  H -0.8031792912     0.0000000000    -0.4947038368
                  H  0.8031792912     0.0000000000    -0.4947038368"""  # a non linear 3-atom molecule
        xyz5 = """S -0.5417345330        0.8208150346        0.0000000000
                  O  0.9206183692        1.6432038228        0.0000000000
                  H -1.2739176462        1.9692549926        0.0000000000"""  # a non linear 3-atom molecule
        xyz6 = """N  1.18784533    0.98526702    0.00000000
                  C  0.04124533    0.98526702    0.00000000
                  H -1.02875467    0.98526702    0.00000000""" # linear
        xyz7 = """C -4.02394116    0.56169428    0.00000000
                  H -5.09394116    0.56169428    0.00000000
                  C -2.82274116    0.56169428    0.00000000
                  H -1.75274116    0.56169428    0.00000000""" # linear
        xyz8 = """C -1.02600933    2.12845307    0.00000000
                  C -0.77966935    0.95278385    0.00000000
                  H -1.23666197    3.17751246    0.00000000
                  H -0.56023545   -0.09447399    0.00000000""" # just 0.5 degree off from linearity, so NOT linear...
        xyz9 = """O -1.1998 0.1610 0.0275
                  O -1.4021 0.6223 -0.8489
                  O -1.48302 0.80682 -1.19946"""  # just 3 points in space on a straight line (not a physical molecule)
        spc1 = ARCSpecies(label=str('test_spc'), xyz=xyz1, multiplicity=1, charge=0, smiles=str('O=C=O'))
        spc2 = ARCSpecies(label=str('test_spc'), xyz=xyz2, multiplicity=1, charge=0, smiles=str('[NH-][S+](=O)(O)C'))
        spc3 = ARCSpecies(label=str('test_spc'), xyz=xyz3, multiplicity=2, charge=0, smiles=str('[O]N=O'))
        spc4 = ARCSpecies(label=str('test_spc'), xyz=xyz4, multiplicity=2, charge=0, smiles=str('[NH2]'))
        spc5 = ARCSpecies(label=str('test_spc'), xyz=xyz5, multiplicity=2, charge=0, smiles=str('[O]S'))
        spc6 = ARCSpecies(label=str('test_spc'), xyz=xyz6, multiplicity=1, charge=0, smiles=str('C#N'))
        spc7 = ARCSpecies(label=str('test_spc'), xyz=xyz7, multiplicity=1, charge=0, smiles=str('C#C'))
        spc8 = ARCSpecies(label=str('test_spc'), xyz=xyz8, multiplicity=1, charge=0, smiles=str('C#C'))
        spc9 = ARCSpecies(label=str('test_spc'), xyz=xyz9, multiplicity=1, charge=0, smiles=str('[O-][O+]=O'))

        self.assertTrue(spc1.is_linear())
        self.assertTrue(spc6.is_linear())
        self.assertTrue(spc7.is_linear())
        self.assertTrue(spc9.is_linear())
        self.assertFalse(spc2.is_linear())
        self.assertFalse(spc3.is_linear())
        self.assertFalse(spc4.is_linear())
        self.assertFalse(spc5.is_linear())
        self.assertFalse(spc8.is_linear())

    def test_charge_and_multiplicity(self):
        """Test determination of molecule charge and multiplicity"""
        spc1 = ARCSpecies(label='spc1', mol=Molecule(SMILES=str('C[CH]C')), generate_thermo=False)
        spc2 = ARCSpecies(label='spc2', mol=Molecule(SMILES=str('CCC')), generate_thermo=False)
        spc3 = ARCSpecies(label='spc3', smiles=str('N[NH]'), generate_thermo=False)
        spc4 = ARCSpecies(label='spc4', smiles=str('NNN'), generate_thermo=False)
        adj1 = """multiplicity 2
                  1 O u1 p2 c0 {2,S}
                  2 H u0 p0 c0 {1,S}
               """
        adj2 = """1 C u0 p0 c0 {2,S} {4,S} {5,S} {6,S}
                  2 N u0 p1 c0 {1,S} {3,S} {7,S}
                  3 O u0 p2 c0 {2,S} {8,S}
                  4 H u0 p0 c0 {1,S}
                  5 H u0 p0 c0 {1,S}
                  6 H u0 p0 c0 {1,S}
                  7 H u0 p0 c0 {2,S}
                  8 H u0 p0 c0 {3,S}
               """
        spc5 = ARCSpecies(label='spc5', adjlist=str(adj1), generate_thermo=False)
        spc6 = ARCSpecies(label='spc6', adjlist=str(adj2), generate_thermo=False)
        xyz1 = """O       0.00000000    0.00000000   -0.10796235
                  H       0.00000000    0.00000000    0.86318839"""
        xyz2 = """N      -0.74678912   -0.11808620    0.00000000
                  C       0.70509190    0.01713703    0.00000000
                  H       1.11547042   -0.48545356    0.87928385
                  H       1.11547042   -0.48545356   -0.87928385
                  H       1.07725194    1.05216961    0.00000000
                  H      -1.15564250    0.32084669    0.81500594
                  H      -1.15564250    0.32084669   -0.81500594"""
        spc7 = ARCSpecies(label='spc7', xyz=xyz1, generate_thermo=False)
        spc8 = ARCSpecies(label='spc8', xyz=xyz2, generate_thermo=False)

        self.assertEqual(spc1.charge, 0)
        self.assertEqual(spc2.charge, 0)
        self.assertEqual(spc3.charge, 0)
        self.assertEqual(spc4.charge, 0)
        self.assertEqual(spc5.charge, 0)
        self.assertEqual(spc6.charge, 0)
        self.assertEqual(spc7.charge, 0)
        self.assertEqual(spc8.charge, 0)

        self.assertEqual(spc1.multiplicity, 2)
        self.assertEqual(spc2.multiplicity, 1)
        self.assertEqual(spc3.multiplicity, 2)
        self.assertEqual(spc4.multiplicity, 1)
        self.assertEqual(spc5.multiplicity, 2)
        self.assertEqual(spc6.multiplicity, 1)
        self.assertEqual(spc7.multiplicity, 2)
        self.assertEqual(spc8.multiplicity, 1)

    def test_as_dict(self):
        """Test Species.as_dict()"""
        spc_dict = self.spc3.as_dict()
        expected_dict = {'optical_isomers': None,
                         'number_of_rotors': 0,
                         'neg_freqs_trshed': [],
                         'external_symmetry': None,
                         'multiplicity': 1,
                         'arkane_file': None,
                         'E0': None,
                         'mol': """1 C u0 p0 c0 {2,S} {3,S} {4,S} {5,S}
2 N u0 p1 c0 {1,S} {6,S} {7,S}
3 H u0 p0 c0 {1,S}
4 H u0 p0 c0 {1,S}
5 H u0 p0 c0 {1,S}
6 H u0 p0 c0 {2,S}
7 H u0 p0 c0 {2,S}
""",
                         'generate_thermo': True,
                         'label': 'methylamine',
                         'long_thermo_description': spc_dict['long_thermo_description'],
                         'charge': 0,
                         'is_ts': False,
                         'final_xyz': '',
                         'opt_level': '',
                         't1': None,
                         'bond_corrections': {'C-H': 3, 'C-N': 1, 'H-N': 2},
                         'rotors_dict': {}}
        self.assertEqual(spc_dict, expected_dict)

    def test_from_dict(self):
        """Test Species.from_dict()"""
        species_dict = self.spc2.as_dict()
        spc = ARCSpecies(species_dict=species_dict)
        self.assertEqual(spc.multiplicity, 2)
        self.assertEqual(spc.charge, 0)
        self.assertEqual(spc.label, 'OH')
        self.assertEqual(spc.mol.toSMILES(), '[OH]')
        self.assertFalse(spc.is_ts)

    def test_determine_rotor_type(self):
        """Test that we correctly determine whether a rotor is FreeRotor or HinderedRotor"""
        free_path = os.path.join(arc_path, 'arc', 'testing', 'rotor_scans', 'CH3C(O)O_FreeRotor.out')
        hindered_path = os.path.join(arc_path, 'arc', 'testing', 'rotor_scans', 'H2O2.out')
        self.assertEqual(determine_rotor_type(free_path), 'FreeRotor')
        self.assertEqual(determine_rotor_type(hindered_path), 'HinderedRotor')

    def test_rotor_symmetry(self):
        """Test that ARC automatically determines a correct rotor symmetry"""
        path1 = os.path.join(arc_path, 'arc', 'testing', 'rotor_scans', 'OOC1CCOCO1.out')  # symmetry = 1; min at -10 o
        path2 = os.path.join(arc_path, 'arc', 'testing', 'rotor_scans', 'H2O2.out')  # symmetry = 1
        path3 = os.path.join(arc_path, 'arc', 'testing', 'rotor_scans', 'N2O3.out')  # symmetry = 2
        path4 = os.path.join(arc_path, 'arc', 'testing', 'rotor_scans', 'sBuOH.out')  # symmetry = 3
        path5 = os.path.join(arc_path, 'arc', 'testing', 'rotor_scans', 'CH3C(O)O_FreeRotor.out')  # symmetry = 6

        symmetry1, _ = determine_rotor_symmetry(rotor_path=path1, label='label', pivots=[3,4])
        symmetry2, _ = determine_rotor_symmetry(rotor_path=path2, label='label', pivots=[3,4])
        symmetry3, _ = determine_rotor_symmetry(rotor_path=path3, label='label', pivots=[3,4])
        symmetry4, _ = determine_rotor_symmetry(rotor_path=path4, label='label', pivots=[3,4])
        symmetry5, _ = determine_rotor_symmetry(rotor_path=path5, label='label', pivots=[3,4])

        self.assertEqual(symmetry1, 1)
        self.assertEqual(symmetry2, 1)
        self.assertEqual(symmetry3, 2)
        self.assertEqual(symmetry4, 3)
        self.assertEqual(symmetry5, 6)

    def test_xyz_from_file(self):
        """Test parsing xyz from a file and saving it in the .initial_xyz attribute"""
        self.assertTrue(' N                 -2.36276900    2.14528400   -0.76917500' in self.spc7.initial_xyz)

    def test_check_species_xyz(self):
        """Test the check_xyz() function"""
        xyz = """
        
        
 C                 -0.67567701    1.18507660    0.04672449
 H                 -0.25592948    1.62415961    0.92757746
 H                 -2.26870864    1.38030564    0.05865317
 O                 -0.36671999   -0.21081064    0.01630374
 H                 -0.73553821   -0.63718986    0.79332805
 C                 -0.08400571    1.86907236   -1.19973252
 
 H                 -0.50375517    1.42998100   -2.08057962
 H                 -0.31518819    2.91354759   -1.17697025
 H                  0.97802159    1.73893214   -1.20769117
 O                 -3.69788377    1.55609096    0.07050345
 O                 -4.28667752    0.37487691    0.04916102
 H                 -4.01978712   -0.12970163    0.82103635
 
 """
        expected_xyz1 = """ C                 -0.67567701    1.18507660    0.04672449
 H                 -0.25592948    1.62415961    0.92757746
 H                 -2.26870864    1.38030564    0.05865317
 O                 -0.36671999   -0.21081064    0.01630374
 H                 -0.73553821   -0.63718986    0.79332805
 C                 -0.08400571    1.86907236   -1.19973252
 H                 -0.50375517    1.42998100   -2.08057962
 H                 -0.31518819    2.91354759   -1.17697025
 H                  0.97802159    1.73893214   -1.20769117
 O                 -3.69788377    1.55609096    0.07050345
 O                 -4.28667752    0.37487691    0.04916102
 H                 -4.01978712   -0.12970163    0.82103635"""
        new_xyz1 = check_species_xyz(xyz)
        self.assertEqual(new_xyz1, expected_xyz1)

        xyz_path = os.path.join(arc_path, 'arc', 'testing', 'xyz', 'CH3C(O)O.xyz')
        expected_xyz2 = """O      -0.53466300   -1.24850800   -0.02156300
O      -0.79314200    1.04818800    0.18134200
C      -0.02397300    0.01171700   -0.37827400
C       1.40511900    0.21728200    0.07675200
H      -0.09294500    0.02877800   -1.47163200
H       2.04132100   -0.57108600   -0.32806800
H       1.45535600    0.19295200    1.16972300
H       1.77484100    1.18704300   -0.25986700
H      -0.43701200   -1.34990600    0.92900600
H      -1.69944700    0.93441600   -0.11271200"""
        new_xyz2 = check_species_xyz(xyz_path)
        self.assertEqual(new_xyz2, expected_xyz2)

    def test_get_min_energy_conformer(self):
        """Test that the xyz with the minimum specified energy is returned from get_min_energy_conformer()"""
        xyzs = ['xyz1', 'xyz2', 'xyz3']
        energies = [-5, -30, -1.5]
        min_xyz = get_min_energy_conformer(xyzs, energies)
        self.assertEqual(min_xyz, 'xyz2')

    def test_mol_from_xyz_atom_id_1(self):
        """Test that atom ids are saved properly when loading both xyz and smiles."""
        mol = self.spc6.mol
        mol_list = self.spc6.mol_list

        self.assertEqual(len(mol_list), 1)
        res = mol_list[0]

        self.assertTrue(mol.atomIDValid())
        self.assertTrue(res.atomIDValid())

        self.assertTrue(mol.isIsomorphic(res))
        self.assertTrue(mol.isIdentical(res))

    def test_mol_from_xyz_atom_id_2(self):
        """Test that atom ids are saved properly when loading both xyz and smiles."""
        mol = self.spc8.mol
        mol_list = self.spc8.mol_list

        self.assertEqual(len(mol_list), 2)
        res1, res2 = mol_list

        self.assertTrue(mol.atomIDValid())
        self.assertTrue(res1.atomIDValid())
        self.assertTrue(res2.atomIDValid())

        self.assertTrue(mol.isIsomorphic(res1))
        self.assertTrue(mol.isIdentical(res1))

        # Check that atom ordering is consistent, ignoring specific oxygen ordering
        mol_ids = [(a.element.symbol, a.id) if a.element.symbol != 'O' else (a.element.symbol,) for a in mol.atoms]
        res1_ids = [(a.element.symbol, a.id) if a.element.symbol != 'O' else (a.element.symbol,) for a in res1.atoms]
        res2_ids = [(a.element.symbol, a.id) if a.element.symbol != 'O' else (a.element.symbol,) for a in res2.atoms]
        self.assertEqual(mol_ids, res1_ids)
        self.assertEqual(mol_ids, res2_ids)

    def test_preserving_multiplicity(self):
        """Test that multiplicity is being preserved, especially when it is guessed differently from xyz"""
        multiplicity_list = [2, 2, 1, 1, 1, 1, 1, 2, 1]
        for i, spc in enumerate([self.spc1, self.spc2, self.spc3, self.spc4, self.spc5, self.spc6, self.spc7,
                                 self.spc8, self.spc9]):
            self.assertEqual(spc.multiplicity, multiplicity_list[i])
            self.assertEqual(spc.mol.multiplicity, multiplicity_list[i])
            self.assertTrue(all([structure.multiplicity == spc.multiplicity for structure in spc.mol_list]))


class TestTSGuess(unittest.TestCase):
    """
    Contains unit tests for the TSGuess class
    """
    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None
        spc1 = Species().fromSMILES(str('CON=O'))
        spc1.label = str('CONO')
        spc2 = Species().fromSMILES(str('C[N+](=O)[O-]'))
        spc2.label = str('CNO2')
        rmg_reaction = Reaction(reactants=[spc1], products=[spc2])
        cls.tsg1 = TSGuess(rmg_reaction=rmg_reaction, method='AutoTST', family='H_Abstraction')
        xyz = """N       0.9177905887     0.5194617797     0.0000000000
                 H       1.8140204898     1.0381941417     0.0000000000
                 H      -0.4763167868     0.7509348722     0.0000000000
                 N       0.9992350860    -0.7048575683     0.0000000000
                 N      -1.4430010939     0.0274543367     0.0000000000
                 H      -0.6371484821    -0.7497769134     0.0000000000
                 H      -2.0093636431     0.0331190314    -0.8327683174
                 H      -2.0093636431     0.0331190314     0.8327683174"""
        cls.tsg2 = TSGuess(xyz=xyz)

    def test_as_dict(self):
        """Test TSGuess.as_dict()"""
        tsg_dict = self.tsg1.as_dict()
        expected_dict = {'method': u'autotst',
                         'energy': None,
                         'family': 'H_Abstraction',
                         'index': None,
                         'rmg_reaction': u'CON=O <=> [O-][N+](=O)C',
                         'success': None,
                         't0': None,
                         'execution_time': None}
        self.assertEqual(tsg_dict, expected_dict)

    def test_from_dict(self):
        """Test TSGuess.from_dict()
        Also tests that the round trip to and from a dictionary ended in an RMG Reaction object"""
        ts_dict = self.tsg1.as_dict()
        tsg = TSGuess(ts_dict=ts_dict)
        self.assertEqual(tsg.method, 'autotst')
        self.assertTrue(isinstance(tsg.rmg_reaction, Reaction))


################################################################################

if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
