#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.species.species module
"""

import os
import shutil
import unittest

from rmgpy.molecule.molecule import Molecule
from rmgpy.reaction import Reaction
from rmgpy.species import Species
from rmgpy.transport import TransportData

from arc.common import almost_equal_coords_lists
from arc.plotter import save_conformers_file
from arc.settings import arc_path
from arc.species.converter import molecules_from_xyz, check_isomorphism, str_to_xyz, xyz_to_str, xyz_to_x_y_z
from arc.species.species import ARCSpecies, TSGuess, determine_rotor_type, determine_rotor_symmetry, check_xyz


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
        cls.spc1_rmg = Species(molecule=[Molecule().from_smiles('C=C[O]')])  # delocalized radical + amine
        cls.spc1_rmg.label = str('vinoxy')
        cls.spc1 = ARCSpecies(rmg_species=cls.spc1_rmg)

        # Method 2: ARCSpecies object by XYZ (also give SMILES for thermo BAC)
        oh_xyz = str("""O       0.00000000    0.00000000   -0.12002167
        H       0.00000000    0.00000000    0.85098324""")
        cls.spc2 = ARCSpecies(label=str('OH'), xyz=oh_xyz, smiles=str('[OH]'), multiplicity=2, charge=0)

        # Method 3: ARCSpecies object by SMILES
        cls.spc3 = ARCSpecies(label=str('methylamine'), smiles=str('CN'), multiplicity=1, charge=0)

        # Method 4: ARCSpecies object by RMG Molecule object
        mol4 = Molecule().from_smiles('C=CC')
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
        cls.spc6 = ARCSpecies(label=str('N3'), xyz=n3_xyz, multiplicity=1, smiles=str('NNN'))

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

        cls.spc10 = ARCSpecies(label='CCCCC', smiles='CCCCC')
        cls.spc11 = ARCSpecies(label='CCCNO', smiles='CCCNO')  # has chiral N
        cls.spc12 = ARCSpecies(label='[CH](CC[CH]c1ccccc1)c1ccccc1', smiles='[CH](CC[CH]c1ccccc1)c1ccccc1')

    def test_set_mol_list(self):
        """Test preserving atom order in the .mol_list attribute"""
        bond_dict = dict()
        for index1, atom1 in enumerate(self.spc12.mol.atoms):
            for atom2 in atom1.edges.keys():
                index2 = self.spc12.mol.atoms.index(atom2)
                if index1 < index2:
                    if index1 not in bond_dict:
                        bond_dict[index1] = [index2]
                    else:
                        bond_dict[index1].append(index2)
        for mol in self.spc12.mol_list:
            for index1, atom1 in enumerate(mol.atoms):
                for atom2 in atom1.edges.keys():
                    index2 = mol.atoms.index(atom2)
                    if index1 < index2:
                        self.assertIn(index2, bond_dict[index1])  # check that these atoms are connected in all mols

    def test_conformers(self):
        """Test conformer generation"""
        self.spc1.conformers = list()
        self.spc1.conformer_energies = list()
        self.assertEqual(len(self.spc1.conformers), len(self.spc1.conformer_energies))
        self.spc1.generate_conformers()
        self.assertEqual(len(self.spc1.conformers), 1)
        self.assertEqual(len(self.spc1.conformers), len(self.spc1.conformer_energies))

        self.spc2.conformers = list()
        self.spc2.generate_conformers()
        self.assertEqual(len(self.spc2.conformers), 1)

        self.spc3.conformers = list()
        self.spc3.generate_conformers()
        self.assertEqual(len(self.spc3.conformers), 1)

        self.spc4.conformers = list()
        self.spc4.generate_conformers()
        self.assertEqual(len(self.spc4.conformers), 1)

        self.spc5.conformers = list()
        self.spc5.generate_conformers()
        self.assertEqual(len(self.spc5.conformers), 3)

        self.spc6.conformers = list()
        self.spc6.generate_conformers()
        self.assertEqual(len(self.spc6.conformers), 5)

        self.spc8.conformers = list()
        self.spc8.generate_conformers()
        self.assertEqual(len(self.spc8.conformers), 2)

        self.spc9.conformers = list()
        self.spc9.generate_conformers()
        self.assertEqual(len(self.spc9.conformers), 1)

        self.spc10.conformers = list()
        self.spc10.generate_conformers(confs_to_dft=1)
        self.assertEqual(len(self.spc10.conformers), 1)

        self.spc10.conformers = list()
        self.spc10.generate_conformers(confs_to_dft=2)
        self.assertEqual(len(self.spc10.conformers), 2)

        self.spc10.conformers = list()
        self.spc10.generate_conformers(confs_to_dft=3)
        self.assertEqual(len(self.spc10.conformers), 3)

        self.spc11.conformers = list()
        self.spc11.generate_conformers(confs_to_dft=1)
        self.assertEqual(len(self.spc11.conformers), 1)

        self.spc11.conformers = list()
        self.spc11.generate_conformers(confs_to_dft=2)
        self.assertEqual(len(self.spc11.conformers), 2)

        self.spc11.conformers = list()
        self.spc11.generate_conformers(confs_to_dft=3)
        self.assertEqual(len(self.spc11.conformers), 3)

        xyz12 = """C       0.00000000    0.00000000    0.00000000
H       1.07008000   -0.14173100    0.00385900
H      -0.65776100   -0.85584100   -0.00777700
H      -0.41231900    0.99757300    0.00391900"""
        spc12 = ARCSpecies(label='CH3', smiles='[CH3]', xyz=xyz12)
        spc12.generate_conformers()
        self.assertEqual(len(spc12.conformers), 2)
        self.assertEqual(len(spc12.conformer_energies), 2)

    def test_from_rmg_species(self):
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

        self.assertEqual(self.spc1.rotors_dict[0][str('pivots')], [1, 2])
        self.assertEqual(self.spc1.rotors_dict[0][str('scan')], [4, 1, 2, 3])
        self.assertTrue(all([t in [1, 4, 5] for t in self.spc1.rotors_dict[0][str('top')]]))
        self.assertEqual(self.spc1.rotors_dict[0][str('times_dihedral_set')], 0)
        self.assertEqual(self.spc3.rotors_dict[0][str('pivots')], [1, 2])
        self.assertEqual(self.spc4.rotors_dict[0][str('pivots')], [2, 3])
        self.assertEqual(self.spc5.rotors_dict[0][str('pivots')], [1, 2])
        self.assertEqual(self.spc6.rotors_dict[0][str('pivots')], [1, 4])
        self.assertEqual(self.spc6.rotors_dict[0][str('scan')], [2, 1, 4, 6])
        self.assertEqual(len(self.spc6.rotors_dict[0][str('top')]), 3)
        self.assertTrue(all([t in [1, 5, 2] for t in self.spc6.rotors_dict[0][str('top')]]))
        self.assertEqual(self.spc6.rotors_dict[1][str('pivots')], [4, 6])
        self.assertEqual(self.spc6.rotors_dict[1][str('scan')], [1, 4, 6, 7])
        self.assertEqual(len(self.spc6.rotors_dict[1][str('top')]), 3)
        self.assertTrue(all([t in [6, 7, 8] for t in self.spc6.rotors_dict[1][str('top')]]))

    def test_initialize_directed_rotors(self):
        """Test the initialize_directed_rotors() method"""
        xyz1 = """C       0.56128965    0.34357304    0.38495536
C      -0.56128955   -0.34357303   -0.38993306
C       1.92143728   -0.23016746    0.02127176
C      -1.92143731    0.23016755   -0.02625006
H       0.39721807    0.22480962    1.46233923
H       0.55622819    1.41911534    0.17290968
H      -0.55622807   -1.41911525   -0.17788706
H      -0.39721778   -0.22481004   -1.46731694
H       2.13051816   -0.09814896   -1.04512584
H       1.97060552   -1.29922203    0.25160298
H       2.70987905    0.27583092    0.58717473
H      -2.13051778    0.09815130    1.04014792
H      -2.70987907   -0.27583253   -0.59215160
H      -1.97060638    1.29922153   -0.25658392"""
        directed_rotors1 = {'cont_opt': ['all'],
                            'brute_force_sp': [['all'], [1, 2]],
                            'brute_force_opt': [[1, 2], [1, 3], [2, 4], [[1, 2], [1, 3]]]
                            }
        spc1 = ARCSpecies(label='spc1', smiles='CCCC', xyz=xyz1, directed_rotors=directed_rotors1)
        spc1.determine_rotors()  # also initializes directed_rotors
        self.assertIn([[3, 1, 2, 4]], spc1.directed_rotors['cont_opt'])
        self.assertIn([[2, 1, 3, 9]], spc1.directed_rotors['cont_opt'])
        self.assertIn([[1, 2, 4, 12]], spc1.directed_rotors['cont_opt'])
        self.assertIn([3, 1, 2, 4], spc1.directed_rotors['brute_force_sp'][0])
        self.assertIn([[3, 1, 2, 4], [2, 1, 3, 9]], spc1.directed_rotors['brute_force_opt'])
        self.assertEqual(len(spc1.rotors_dict.keys()), 9)
        self.assertEqual(spc1.rotors_dict[3]['dimensions'], 3)
        self.assertIn([1, 3], spc1.rotors_dict[3]['pivots'])
        self.assertIn([1, 2], spc1.rotors_dict[3]['pivots'])
        self.assertIn([2, 4], spc1.rotors_dict[3]['pivots'])
        self.assertEqual(spc1.rotors_dict[3]['cont_indices'], [])

        spc2 = ARCSpecies(label='propanol', smiles='CCO', directed_rotors={'brute_force_sp': [['all']]})
        spc2.determine_rotors()  # also initializes directed_rotors
        self.assertEqual(spc2.directed_rotors, {'brute_force_sp': [[[4, 1, 2, 3], [1, 2, 3, 9]]]})
        self.assertEqual(len(spc2.rotors_dict), 1)
        self.assertEqual(spc2.rotors_dict[0]['dimensions'], 2)

    def test_symmetry(self):
        """Test external symmetry and chirality determination"""
        allene = ARCSpecies(label=str('allene'), smiles=str('C=C=C'), multiplicity=1, charge=0)
        allene.final_xyz = str_to_xyz("""C  -1.01646   0.10640  -0.91445
                              H  -1.39000   1.03728  -1.16672
                              C   0.00000   0.00000   0.00000
                              C   1.01653  -0.10640   0.91438
                              H  -1.40975  -0.74420  -1.35206
                              H   0.79874  -0.20864   1.92036
                              H   2.00101  -0.08444   0.59842""")
        allene.determine_symmetry()
        self.assertEqual(allene.optical_isomers, 1)
        self.assertEqual(allene.external_symmetry, 4)

        ammonia = ARCSpecies(label=str('ammonia'), smiles=str('N'), multiplicity=1, charge=0)
        ammonia.final_xyz = str_to_xyz("""N  0.06617   0.20024   0.13886
                               H  -0.62578  -0.34119   0.63709
                               H  -0.32018   0.51306  -0.74036
                               H   0.87976  -0.37219  -0.03564""")
        ammonia.determine_symmetry()
        self.assertEqual(ammonia.optical_isomers, 1)
        self.assertEqual(ammonia.external_symmetry, 3)

        methane = ARCSpecies(label=str('methane'), smiles=str('C'), multiplicity=1, charge=0)
        methane.final_xyz = str_to_xyz("""C   0.00000   0.00000   0.00000
                               H  -0.29717   0.97009  -0.39841
                               H   1.08773  -0.06879   0.01517
                               H  -0.38523  -0.10991   1.01373
                               H -0.40533  -0.79140  -0.63049""")
        methane.determine_symmetry()
        self.assertEqual(methane.optical_isomers, 1)
        self.assertEqual(methane.external_symmetry, 12)

        chiral = ARCSpecies(label=str('chiral'), smiles=str('C(C)(O)(N)'), multiplicity=1, charge=0)
        chiral.final_xyz = str_to_xyz("""C                 -0.49341625    0.37828349    0.00442108
                              H                 -1.56331545    0.39193350    0.01003359
                              N                  0.01167132    1.06479568    1.20212111
                              H                  1.01157784    1.05203730    1.19687531
                              H                 -0.30960193    2.01178202    1.20391932
                              O                 -0.03399634   -0.97590449    0.00184366
                              H                 -0.36384913   -1.42423238   -0.78033350
                              C                  0.02253835    1.09779040   -1.25561654
                              H                 -0.34510997    0.59808430   -2.12741255
                              H                 -0.32122209    2.11106387   -1.25369100
                              H                  1.09243518    1.08414066   -1.26122530""")
        chiral.determine_symmetry()
        self.assertEqual(chiral.optical_isomers, 2)
        self.assertEqual(chiral.external_symmetry, 1)

        s8 = ARCSpecies(label=str('s8'), smiles=str('S1SSSSSSS1'), multiplicity=1, charge=0)
        s8.final_xyz = str_to_xyz("""S   2.38341   0.12608   0.09413
                          S   1.45489   1.88955  -0.13515
                          S  -0.07226   2.09247   1.14966
                          S  -1.81072   1.52327   0.32608
                          S  -2.23488  -0.39181   0.74645
                          S  -1.60342  -1.62383  -0.70542
                          S   0.22079  -2.35820  -0.30909
                          S   1.66220  -1.25754  -1.16665""")
        s8.determine_symmetry()
        self.assertEqual(s8.optical_isomers, 1)
        self.assertEqual(s8.external_symmetry, 8)

        water = ARCSpecies(label=str('H2O'), smiles=str('O'), multiplicity=1, charge=0)
        water.final_xyz = str_to_xyz("""O   0.19927   0.29049  -0.11186
                             H   0.50770  -0.61852  -0.09124
                             H  -0.70697   0.32803   0.20310""")
        water.determine_symmetry()
        self.assertEqual(water.optical_isomers, 1)
        self.assertEqual(water.external_symmetry, 2)

    def test_xyz_format_conversion(self):
        """Test conversions from string to dict xyz formats"""
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
H      -1.67091600   -1.35164600   -0.93286400"""

        xyz_dict = str_to_xyz(xyz_str0)
        xyz_str1 = xyz_to_str(xyz_dict)
        s_mol, b_mol = molecules_from_xyz(xyz_dict)
        x, y, z = xyz_to_x_y_z(xyz_dict)

        self.assertEqual(xyz_str0, xyz_str1)
        self.assertEqual(xyz_dict['symbols'], ('N', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H'))
        self.assertEqual(xyz_dict['symbols'], tuple(atom.symbol for atom in s_mol.atoms))
        self.assertEqual(xyz_dict['symbols'], tuple(atom.symbol for atom in b_mol.atoms))
        self.assertEqual(x, (2.246906, -1.056548, -1.056614, -0.305141, 1.083589, -0.391683, -1.672426, -1.741854,
                             -0.391871, -1.743414, -1.670916))
        self.assertEqual(y[1], 1.29155)
        self.assertEqual(z[-1], -0.932864)

    def test_charge_and_multiplicity(self):
        """Test determination of molecule charge and multiplicity"""
        spc1 = ARCSpecies(label='spc1', mol=Molecule(smiles='C[CH]C'), generate_thermo=False)
        spc2 = ARCSpecies(label='spc2', mol=Molecule(smiles='CCC'), generate_thermo=False)
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
                         'consider_all_diastereomers': True,
                         'force_field': 'MMFF94s',
                         'is_ts': False,
                         't1': None,
                         'bond_corrections': {'C-H': 3, 'C-N': 1, 'H-N': 2}}
        self.assertEqual(spc_dict, expected_dict)

    def test_from_dict(self):
        """Test Species.from_dict()"""
        species_dict = self.spc2.as_dict()
        spc = ARCSpecies(species_dict=species_dict)
        self.assertEqual(spc.multiplicity, 2)
        self.assertEqual(spc.charge, 0)
        self.assertEqual(spc.label, 'OH')
        self.assertEqual(spc.mol.to_smiles(), '[OH]')
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

        sym1, e1 = determine_rotor_symmetry(label='label', pivots=[3, 4], rotor_path=path1)
        sym2, e2 = determine_rotor_symmetry(label='label', pivots=[3, 4], rotor_path=path2)
        sym3, e3, n3 = determine_rotor_symmetry(label='label', pivots=[3, 4], rotor_path=path3, return_num_wells=True)
        sym4, e4 = determine_rotor_symmetry(label='label', pivots=[3, 4], rotor_path=path4)
        sym5, e5, n5 = determine_rotor_symmetry(label='label', pivots=[3, 4], rotor_path=path5, return_num_wells=True)

        self.assertEqual(sym1, 1)
        self.assertAlmostEqual(e1, 20.87723711)
        self.assertEqual(sym2, 1)
        self.assertAlmostEqual(e2, 35.40014232)
        self.assertEqual(sym3, 2)
        self.assertAlmostEqual(e3, 18.26215261)
        self.assertEqual(n3, 2)
        self.assertEqual(sym4, 3)
        self.assertAlmostEqual(e4, 11.14737880)
        self.assertEqual(sym5, 6)
        self.assertAlmostEqual(e5, 0.099359417)
        self.assertEqual(n5, 6)

    def test_xyz_from_file(self):
        """Test parsing xyz from a file and saving it in the .initial_xyz attribute"""
        expected_xyz = {'symbols': ('N', 'N', 'N', 'N', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C',
                                    'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'),
                        'isotopes': (14, 14, 14, 14, 12, 12, 12, 12, 12, 12, 12, 12,
                                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
                        'coords': ((0.342697, 0.256671, -0.178208),
                                   (-0.433495, -0.657695, 0.054512), (2.41731872, -1.07916417, 2.08039935),
                                   (-2.362769, 2.145284, -0.769175), (1.770075, -0.169463, -0.293983),
                                   (-1.853044, -0.247836, 0.17823), (2.55942025, 1.11530153, -0.58321611),
                                   (2.04300531, -1.25673666, -1.33341538), (-2.184306, -0.295998, 1.676229),
                                   (-2.679947, -1.265, -0.615282), (2.12217963, -0.66843078, 1.04808732),
                                   (-2.10138, 1.108101, -0.348404), (3.63048242, 0.90778556, -0.5731045),
                                   (2.33388242, 1.88038907, 0.16008174), (2.28038909, 1.48886093, -1.57118254),
                                   (1.4720186, -2.16169122, -1.12648223), (3.10742908, -1.49794218, -1.3260035),
                                   (1.7671938, -0.89392348, -2.32499076), (-1.967535, -1.295127, 2.059502),
                                   (-3.242229, -0.077235, 1.830058), (-1.586982, 0.433821, 2.22524),
                                   (-2.479993, -2.265879, -0.228516), (-3.743773, -1.046138, -0.509693),
                                   (-2.420253, -1.234761, -1.674964))}
        self.assertTrue(almost_equal_coords_lists(self.spc7.conformers[0], expected_xyz))

    def test_process_xyz(self):
        """Test the process_xyz() function"""
        # the last four H's in xyz hve a leading TAB character which is removed when processing
        xyz = """
        
        
        
   O    3.1024    0.1216    1.0455
   O    1.4602   -3.3145   -0.2099
C   -2.2924    0.2555   -0.8205
   C   -2.3929   -0.1455   -2.3013
   C   -3.2177   -0.6103    0.0500
                    C    3.7529    1.2995    1.5066
   C   -0.8566    0.2292   -0.3220
      C    1.2086   -0.9791    0.1635
       C    1.8156    0.2120    0.6120
       C   -0.1097   -0.9444   -0.2920
   C   -0.2294    1.3920    0.1279
   C    1.0837    1.3976    0.5896
   

   C    1.9403   -2.2644    0.1662
    H   -2.6363    1.2898   -0.7367
H   -2.0724   -1.1772   -2.4533
   H   -1.7687    0.4927   -2.9270
   H   -3.4228   -0.0630   -2.6520
   H   -4.2511   -0.5298   -0.2908
   H   -3.1797   -0.3025    1.0951
   H   -2.9338   -1.6626    0.0021
H    3.8200    2.0502    0.7167
   H    4.7525    0.9900    1.7960
                                                                           H    3.2393    1.7229    2.3720
	H   -0.5302   -1.8837   -0.6253
	H   -0.7777    2.3260    0.1202
	H    1.5203    2.3247    0.9261
	H    2.9757   -2.2266    0.5368
 
     
   
   
 
 
 """
        expected_xyz2 = """O    3.1024    0.1216    1.0455
O    1.4602   -3.3145   -0.2099
C   -2.2924    0.2555   -0.8205
C   -2.3929   -0.1455   -2.3013
C   -3.2177   -0.6103    0.0500
C    3.7529    1.2995    1.5066
C   -0.8566    0.2292   -0.3220
C    1.2086   -0.9791    0.1635
C    1.8156    0.2120    0.6120
C   -0.1097   -0.9444   -0.2920
C   -0.2294    1.3920    0.1279
C    1.0837    1.3976    0.5896
C    1.9403   -2.2644    0.1662
H   -2.6363    1.2898   -0.7367
H   -2.0724   -1.1772   -2.4533
H   -1.7687    0.4927   -2.9270
H   -3.4228   -0.0630   -2.6520
H   -4.2511   -0.5298   -0.2908
H   -3.1797   -0.3025    1.0951
H   -2.9338   -1.6626    0.0021
H    3.8200    2.0502    0.7167
H    4.7525    0.9900    1.7960
H    3.2393    1.7229    2.3720
H   -0.5302   -1.8837   -0.6253
H   -0.7777    2.3260    0.1202
H    1.5203    2.3247    0.9261
H    2.9757   -2.2266    0.5368"""

        spc1 = ARCSpecies(label='test_spc1', xyz=[xyz, xyz])
        self.assertIsNone(spc1.initial_xyz)
        self.assertIsNone(spc1.final_xyz)
        self.assertEqual(len(spc1.conformers), 2)
        self.assertEqual(len(spc1.conformer_energies), 2)
        self.assertEqual(spc1.multiplicity, 1)

        spc2 = ARCSpecies(label='test_spc2', xyz=xyz)
        self.assertIsNone(spc1.initial_xyz)
        self.assertIsNone(spc1.final_xyz)
        self.assertEqual(spc2.conformers[0], str_to_xyz(expected_xyz2))
        self.assertIsNone(spc2.final_xyz)
        self.assertEqual(len(spc2.conformers), 1)
        self.assertEqual(len(spc2.conformer_energies), 1)
        self.assertEqual(spc2.multiplicity, 1)

        xyz_path = os.path.join(arc_path, 'arc', 'testing', 'xyz', 'CH3C(O)O.xyz')
        expected_xyz3 = """O      -0.53466300   -1.24850800   -0.02156300
O      -0.79314200    1.04818800    0.18134200
C      -0.02397300    0.01171700   -0.37827400
C       1.40511900    0.21728200    0.07675200
H      -0.09294500    0.02877800   -1.47163200
H       2.04132100   -0.57108600   -0.32806800
H       1.45535600    0.19295200    1.16972300
H       1.77484100    1.18704300   -0.25986700
H      -0.43701200   -1.34990600    0.92900600
H      -1.69944700    0.93441600   -0.11271200"""

        spc3 = ARCSpecies(label='test_spc3', xyz=xyz_path)
        self.assertIsNone(spc1.initial_xyz)
        self.assertIsNone(spc1.final_xyz)
        self.assertEqual(spc3.conformers[0], str_to_xyz(expected_xyz3))
        self.assertEqual(len(spc3.conformers), 1)
        self.assertEqual(len(spc3.conformer_energies), 1)
        self.assertEqual(spc3.multiplicity, 1)

        conformers_path = os.path.join(arc_path, 'arc', 'testing', 'xyz', 'conformers_file.txt')
        spc4 = ARCSpecies(label='test_spc3', xyz=conformers_path)
        self.assertEqual(len(spc4.conformers), 4)
        self.assertEqual(len(spc4.conformer_energies), 4)
        self.assertIsNotNone(spc4.conformer_energies[0])
        self.assertIsNotNone(spc4.conformer_energies[1])
        self.assertIsNone(spc4.conformer_energies[2])
        self.assertIsNotNone(spc4.conformer_energies[3])
        self.assertEqual(spc4.multiplicity, 2)

    def test_mol_from_xyz_atom_id_1(self):
        """Test that atom ids are saved properly when loading both xyz and smiles (1)."""
        mol = self.spc6.mol
        mol_list = self.spc6.mol_list

        self.assertEqual(len(mol_list), 1)
        res = mol_list[0]

        self.assertTrue(mol.atom_ids_valid())
        self.assertTrue(res.atom_ids_valid())

        self.assertTrue(mol.is_isomorphic(res))
        self.assertTrue(mol.is_identical(res))

    def test_mol_from_xyz_atom_id_2(self):
        """Test that atom ids are saved properly when loading both xyz and smiles (2)."""
        mol = self.spc8.mol
        mol_list = self.spc8.mol_list

        self.assertEqual(len(mol_list), 2)
        res1, res2 = mol_list

        self.assertTrue(mol.atom_ids_valid())
        self.assertTrue(res1.atom_ids_valid())
        self.assertTrue(res2.atom_ids_valid())

        self.assertTrue(mol.is_isomorphic(res1))
        self.assertTrue(mol.is_identical(res1))

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

    def test_append_conformers(self):
        """Test that ARC correctly parses its own conformer files"""
        project_directory = os.path.join(arc_path, 'Projects', 'arc_project_for_testing_delete_after_usage4')
        xyzs = [{'symbols': ('O', 'C', 'C', 'H', 'H', 'H'), 'isotopes': (16, 12, 12, 1, 1, 1),
                 'coords': ((1.090687, 0.265168, -0.167063), (2.922041, -1.183357, -0.388849),
                            (2.276555, -0.003739, 0.085435), (2.365448, -1.88781, -0.999146),
                            (3.96112, -1.388545, -0.149588), (2.878135, 0.688284, 0.703994))},
                {'symbols': ('O', 'C', 'C', 'H', 'H', 'H'), 'isotopes': (16, 12, 12, 1, 1, 1),
                 'coords': ((1.193961, -0.060037, 0.038901), (3.18797, 0.770613, -0.873527),
                            (2.435912, -0.044393, 0.021716), (4.2737, 0.760902, -0.862861),
                            (2.666417, 1.411557, -1.577573), (3.00398, -0.683368, 0.723598))},
                {'symbols': ('O', 'C', 'C', 'H', 'H', 'H'), 'isotopes': (16, 12, 12, 1, 1, 1),
                 'coords': ((1.352411, -1.02956, -0.240562), (-0.720843, 0.013082, 0.09573),
                            (0.692177, 0.011851, -0.090443), (-1.258038, -0.930181, 0.109268),
                            (-1.268612, 0.941771, 0.224201), (1.202904, 0.993037, -0.098194))},
                {'symbols': ('O', 'C', 'C', 'H', 'H', 'H'), 'isotopes': (16, 12, 12, 1, 1, 1),
                 'coords': ((-1.401029, -0.985751, -0.115885), (0.72457, -0.010767, 0.064488),
                            (-0.694946, 0.0345, -0.062063), (1.22539, -0.97248, 0.117412),
                            (1.312774, 0.900871, 0.108784), (-1.166758, 1.033626, -0.112737))}]
        energies = [0, 5, 5, 5]  # J/mol

        save_conformers_file(project_directory=project_directory, label='vinoxy', xyzs=xyzs, level_of_theory='level1',
                             multiplicity=2, charge=0)
        self.assertTrue(os.path.isfile(os.path.join(project_directory, 'output', 'Species', 'vinoxy', 'geometry',
                                                    'conformers', 'conformers_before_optimization.txt')))

        save_conformers_file(project_directory=project_directory, label='vinoxy', xyzs=xyzs, level_of_theory='level1',
                             multiplicity=2, charge=0, energies=energies)
        self.assertTrue(os.path.isfile(os.path.join(project_directory, 'output', 'Species', 'vinoxy', 'geometry',
                                                    'conformers', 'conformers_after_optimization.txt')))

        spc2 = ARCSpecies(label=str('vinoxy'), smiles='C=C[O]', xyz=os.path.join(
            project_directory, 'output', 'Species', 'vinoxy', 'geometry', 'conformers',
            'conformers_before_optimization.txt'))

        spc3 = ARCSpecies(label=str('vinoxy'), smiles='C=C[O]', xyz=os.path.join(
            project_directory, 'output', 'Species', 'vinoxy', 'geometry', 'conformers',
            'conformers_after_optimization.txt'))

        self.assertEqual(spc2.conformers[2], xyzs[2])
        self.assertEqual(spc3.conformers[2], xyzs[2])
        self.assertEqual(spc3.conformer_energies[2], energies[2])

    def test_number_of_atoms_property(self):
        """Test that the number_of_atoms property functions correctly"""
        self.assertEqual(self.spc1.number_of_atoms, 6)
        self.assertEqual(self.spc2.number_of_atoms, 2)
        self.assertEqual(self.spc3.number_of_atoms, 7)
        self.assertEqual(self.spc4.number_of_atoms, 9)
        self.assertEqual(self.spc5.number_of_atoms, 6)
        self.assertEqual(self.spc6.number_of_atoms, 8)
        self.assertEqual(self.spc7.number_of_atoms, 24)
        self.assertEqual(self.spc8.number_of_atoms, 5)
        self.assertEqual(self.spc9.number_of_atoms, 2)

        xyz10 = """N       0.82269400    0.19834500   -0.33588000
C      -0.57469800   -0.02442800    0.04618900
H      -1.08412400   -0.56416500   -0.75831900
H      -0.72300600   -0.58965300    0.98098100
H      -1.07482500    0.94314300    0.15455500
H       1.31266200   -0.68161600   -0.46770200
H       1.32129900    0.71837500    0.38017700

"""
        spc10 = ARCSpecies(label='spc10', xyz=xyz10)
        self.assertEqual(spc10.number_of_atoms, 7)

    def test_number_of_heavy_atoms_property(self):
        """Test that the number_of_heavy_atoms property functions correctly"""
        self.assertEqual(self.spc1.number_of_heavy_atoms, 3)
        self.assertEqual(self.spc2.number_of_heavy_atoms, 1)
        self.assertEqual(self.spc3.number_of_heavy_atoms, 2)
        self.assertEqual(self.spc4.number_of_heavy_atoms, 3)
        self.assertEqual(self.spc5.number_of_heavy_atoms, 2)
        self.assertEqual(self.spc6.number_of_heavy_atoms, 3)
        self.assertEqual(self.spc7.number_of_heavy_atoms, 12)
        self.assertEqual(self.spc8.number_of_heavy_atoms, 4)
        self.assertEqual(self.spc9.number_of_heavy_atoms, 1)

        xyz10 = """N       0.82269400    0.19834500   -0.33588000
C      -0.57469800   -0.02442800    0.04618900
H      -1.08412400   -0.56416500   -0.75831900
H      -0.72300600   -0.58965300    0.98098100
H      -1.07482500    0.94314300    0.15455500
H       1.31266200   -0.68161600   -0.46770200
H       1.32129900    0.71837500    0.38017700

"""
        spc10 = ARCSpecies(label='spc10', xyz=xyz10)
        self.assertEqual(spc10.number_of_heavy_atoms, 2)

    def test_set_transport_data(self):
        """Test the set_transport_data method"""
        self.assertIsInstance(self.spc1.transport_data, TransportData)
        lj_path = os.path.join(arc_path, 'arc', 'testing', 'NH3_oneDMin.dat')
        opt_path = os.path.join(arc_path, 'arc', 'testing', 'SO2OO_CBS-QB3.log')
        bath_gas = 'N2'
        opt_level = 'CBS-QB3'
        freq_path = os.path.join(arc_path, 'arc', 'testing', 'SO2OO_CBS-QB3.log')
        freq_level = 'CBS-QB3'
        self.spc1.set_transport_data(lj_path, opt_path, bath_gas, opt_level, freq_path, freq_level)
        self.assertIsInstance(self.spc1.transport_data, TransportData)
        self.assertEqual(self.spc1.transport_data.shapeIndex, 2)
        self.assertAlmostEqual(self.spc1.transport_data.epsilon.value_si, 1420.75, 2)
        self.assertAlmostEqual(self.spc1.transport_data.sigma.value_si, 3.57813e-10, 4)
        self.assertAlmostEqual(self.spc1.transport_data.dipoleMoment.value_si, 2.10145e-30, 4)
        self.assertAlmostEqual(self.spc1.transport_data.polarizability.value_si, 3.99506e-30, 4)
        self.assertEqual(self.spc1.transport_data.rotrelaxcollnum, 2)
        self.assertEqual(self.spc1.transport_data.comment, 'L-J coefficients calculated by OneDMin using a '
                                                           'DF-MP2/aug-cc-pVDZ potential energy surface with N2 as '
                                                           'the bath gas; Dipole moment was calculated at the CBS-QB3 '
                                                           'level of theory; Polarizability was calculated at the '
                                                           'CBS-QB3 level of theory; Rotational Relaxation Collision '
                                                           'Number was not determined, default value is 2')

    def test_xyz_from_dict(self):
        """Test correctly assigning xyz from dictionary"""
        species_dict1 = {'label': 'tst_spc_1', 'xyz': 'C 0.1 0.5 0.0'}
        species_dict2 = {'label': 'tst_spc_2', 'initial_xyz': 'C 0.2 0.5 0.0'}
        species_dict3 = {'label': 'tst_spc_3', 'final_xyz': 'C 0.3 0.5 0.0'}
        species_dict4 = {'label': 'tst_spc_4', 'xyz': ['C 0.4 0.5 0.0', 'C 0.5 0.5 0.0']}

        spc1 = ARCSpecies(species_dict=species_dict1)
        spc2 = ARCSpecies(species_dict=species_dict2)
        spc3 = ARCSpecies(species_dict=species_dict3)
        spc4 = ARCSpecies(species_dict=species_dict4)

        self.assertIsNone(spc1.initial_xyz)
        self.assertIsNone(spc1.final_xyz)
        self.assertEqual(spc1.conformers, [{'coords': ((0.1, 0.5, 0.0),), 'isotopes': (12,), 'symbols': ('C',)}])
        self.assertEqual(spc1.conformer_energies, [None])

        self.assertEqual(spc2.initial_xyz, {'coords': ((0.2, 0.5, 0.0),), 'isotopes': (12,), 'symbols': ('C',)})
        self.assertIsNone(spc2.final_xyz)
        self.assertEqual(spc2.conformers, [])
        self.assertEqual(spc2.conformer_energies, [])

        self.assertIsNone(spc3.initial_xyz)
        self.assertEqual(spc3.final_xyz, {'coords': ((0.3, 0.5, 0.0),), 'isotopes': (12,), 'symbols': ('C',)})
        self.assertEqual(spc3.conformers, [])
        self.assertEqual(spc3.conformer_energies, [])

        self.assertIsNone(spc4.initial_xyz)
        self.assertIsNone(spc4.final_xyz)
        self.assertEqual(spc4.conformers, [{'coords': ((0.4, 0.5, 0.0),), 'isotopes': (12,), 'symbols': ('C',)},
                                           {'coords': ((0.5, 0.5, 0.0),), 'isotopes': (12,), 'symbols': ('C',)}])
        self.assertEqual(spc4.conformer_energies, [None, None])

    def test_consistent_atom_order(self):
        """Test that the atom order is preserved whether starting from SMILES or from xyz"""
        spc1 = ARCSpecies(label='spc1', smiles='CCCO')
        xyz1 = spc1.get_xyz()
        for atom, symbol in zip(spc1.mol.atoms, xyz1['symbols']):
            self.assertEqual(atom.symbol, symbol)

        xyz2 = """C      -0.37147383   -0.54225753    0.07779977
C       0.99011397    0.11006088   -0.10715587
H      -0.33990169   -1.22256017    0.93731544
H      -0.60100180   -1.16814809   -0.79292035
H       1.26213386    0.70273091    0.77209458
O       1.96607463   -0.90691160   -0.28642183
H       0.99631715    0.75813344   -0.98936747
H      -1.27803075    1.09840370    1.16400304
C      -1.46891192    0.48768649    0.27579733
H      -2.43580767   -0.00829320    0.40610628
H      -1.54270451    1.15356356   -0.58992943
H       2.82319256   -0.46240839   -0.40178723"""
        spc2 = ARCSpecies(label='spc2', xyz=xyz2)
        for adj, coord in zip(spc2.mol.to_adjacency_list().splitlines(), xyz2.splitlines()):
            if adj and coord:
                self.assertEqual(adj.split()[1], coord.split()[0])

        n3_xyz = str("""N      -1.1997440839    -0.1610052059     0.0274738287
        H      -1.4016624407    -0.6229695533    -0.8487034080
        H      -0.0000018759     1.2861082773     0.5926077870
        N       0.0000008520     0.5651072858    -0.1124621525
        H      -1.1294692206    -0.8709078271     0.7537518889
        N       1.1997613019    -0.1609980472     0.0274604887
        H       1.1294795781    -0.8708998550     0.7537444446
        H       1.4015274689    -0.6230592706    -0.8487058662""")
        spc3 = ARCSpecies(label=str('N3'), xyz=n3_xyz, multiplicity=1, smiles=str('NNN'))
        self.assertEqual(spc3.mol.atoms[1].symbol, 'H')
        spc3.generate_conformers()
        self.assertEqual(len(spc3.conformers), 6)

        xyz4 = """O      -1.48027320    0.36597456    0.41386552
        C      -0.49770656   -0.40253648   -0.26500019
        C       0.86215119    0.24734211   -0.11510338
        H      -0.77970114   -0.46128090   -1.32025907
        H      -0.49643724   -1.41548311    0.14879346
        H       0.84619526    1.26924854   -0.50799415
        H       1.14377239    0.31659216    0.94076336
        H       1.62810781   -0.32407050   -0.64676910
        H      -1.22610851    0.40421362    1.35170355"""
        spc4 = ARCSpecies(label='CCO', smiles='CCO', xyz=xyz4)  # define from xyz for consistent atom order
        for atom1, atom2 in zip(spc4.mol.atoms, spc4.mol_list[0].atoms):
            self.assertEqual(atom1.symbol, atom2.symbol)

        xyz5 = {'symbols': ('S', 'O', 'O', 'O', 'C', 'C', 'H', 'H', 'H', 'H'),
                'isotopes': (32, 16, 16, 16, 12, 12, 1, 1, 1, 1),
                'coords': ((0.35915171, 1.99254721, 1.1849049),
                           (0.40385373, -0.65769862, 1.03431374),
                           (-1.23178399, -0.59559801, -1.39114493),
                           (0.6901556, -1.65712867, 0.01239391),
                           (-0.0426136, 0.49595776, 0.40364219),
                           (-0.80103934, 0.51314044, -0.70610325),
                           (-1.17387862, 1.41490429, -1.17716515),
                           (0.95726719, 1.46882836, 2.26423536),
                           (-0.83008868, -1.36939497, -0.94170868),
                           (1.65888059, -1.54205855, 0.02674995))}
        spc5 = ARCSpecies(label='chiral1', smiles='SC(OO)=CO', xyz=xyz5)
        for atom, symbol in zip(spc5.mol.atoms, xyz5['symbols']):
            self.assertEqual(atom.symbol, symbol)

    def test_get_radius(self):
        """Test determining the species radius"""
        spc1 = ARCSpecies(label='r1', smiles='O=C=O')
        self.assertAlmostEqual(spc1.radius, 2.065000, 5)

        spc2 = ARCSpecies(label='r2', smiles='CCCCC')
        self.assertAlmostEqual(spc2.radius, 3.734040, 5)

        spc3 = ARCSpecies(label='r3', smiles='CCO')
        self.assertAlmostEqual(spc3.radius, 2.495184, 5)

        xyz = """
        C       0.05984800   -0.62319600    0.00000000
        H      -0.46898100   -1.02444400    0.87886100
        H      -0.46898100   -1.02444400   -0.87886100
        H       1.08093800   -1.00826200    0.00000000
        N       0.05980600    0.81236000    0.00000000
        H      -0.92102100    1.10943400    0.00000000
        """
        spc4 = ARCSpecies(label='r4', xyz=xyz)
        self.assertAlmostEqual(spc4.radius, 1.81471201, 5)

    def test_check_xyz(self):
        """Test the check_xyz() function"""
        xyz1 = """C       0.62797113   -0.03193934   -0.15151370
C       1.35170118   -1.00275231   -0.48283333
O      -0.67437022    0.01989281    0.16029161
H      -1.14812497    0.95492850    0.42742905
H      -1.27300665   -0.88397696    0.14797321
H       1.11582953    0.94384729   -0.10134685"""
        xyz_dict1 = str_to_xyz(xyz1)
        self.assertTrue(check_xyz(xyz_dict1, multiplicity=2, charge=0))
        self.assertFalse(check_xyz(xyz_dict1, multiplicity=1, charge=0))
        self.assertFalse(check_xyz(xyz_dict1, multiplicity=2, charge=1))
        self.assertTrue(check_xyz(xyz_dict1, multiplicity=1, charge=-1))

    def test_check_xyz_isomorphism(self):
        """Test the check_xyz_isomorphism() method"""
        xyz1 = """C  -1.9681540   0.0333440  -0.0059220
                  C  -0.6684360  -0.7562450   0.0092140
                  C   0.5595480   0.1456260  -0.0036480
                  O   0.4958540   1.3585920   0.0068500
                  N   1.7440770  -0.5331650  -0.0224050
                  H  -2.8220500  -0.6418490   0.0045680
                  H  -2.0324190   0.6893210   0.8584580
                  H  -2.0300690   0.6574940  -0.8939330
                  H  -0.6121640  -1.4252590  -0.8518010
                  H  -0.6152180  -1.3931710   0.8949150
                  H   1.7901420  -1.5328370   0.0516350
                  H   2.6086580  -0.0266360   0.0403330"""
        spc1 = ARCSpecies(label='propanamide1', smiles='CCC(=O)N', xyz=xyz1)
        spc1.final_xyz = spc1.conformers[0]
        is_isomorphic1 = spc1.check_xyz_isomorphism()
        self.assertTrue(is_isomorphic1)

        xyz2 = """C   0.6937910  -0.8316510   0.0000000
                  O   0.2043990  -1.9431180   0.0000000
                  N   0.0000000   0.3473430   0.0000000
                  C  -1.4529360   0.3420060   0.0000000
                  C   0.6724520   1.6302410   0.0000000
                  H   1.7967050  -0.6569720   0.0000000
                  H  -1.7909050  -0.7040620   0.0000000
                  H  -1.8465330   0.8552430   0.8979440
                  H  -1.8465330   0.8552430  -0.8979440
                  H   1.7641260   1.4761740   0.0000000
                  H   0.4040540   2.2221690  -0.8962980
                  H   0.4040540   2.2221690   0.8962980"""  # dimethylformamide, O=CN(C)C
        spc2 = ARCSpecies(label='propanamide2', smiles='CCC(=O)N', xyz=xyz1)  # define w/ the correct xyz
        spc2.final_xyz = str_to_xyz(xyz2)  # set .final_xyz to the incorrect isomer

        spc2.conf_is_isomorphic = True  # set to True so that isomorphism is strictly enforced
        is_isomorphic2 = spc2.check_xyz_isomorphism()
        self.assertFalse(is_isomorphic2)

        is_isomorphic3 = spc2.check_xyz_isomorphism(allow_nonisomorphic_2d=True)
        self.assertTrue(is_isomorphic3)

        spc2.conf_is_isomorphic = False  # set to False so that isomorphism is not strictly enforced
        is_isomorphic4 = spc2.check_xyz_isomorphism()
        self.assertTrue(is_isomorphic4)

    def test_scissors(self):
        """Test the scissors method in Species"""
        ch3oc2h5_xyz = """C  1.3324310  1.2375310  0.0000000
                          O  0.0018390  0.7261240  0.0000000
                          C  0.0000000 -0.7039850  0.0000000
                          C -1.4484340 -1.1822370  0.0000000
                          H  1.2595400  2.3362390  0.0000000
                          H  1.8883830  0.9070030  0.9017920
                          H  1.8883830  0.9070030 -0.9017920
                          H  0.5387190 -1.0786510 -0.8972750
                          H  0.5387190 -1.0786510  0.8972750
                          H -1.4840700 -2.2862140  0.0000000
                          H -1.9741850 -0.8117890  0.8963760
                          H -1.9741850 -0.8117890 -0.8963760"""
        ch3oc2h5 = ARCSpecies(label='ch3oc2h5', smiles='COCC', xyz=ch3oc2h5_xyz, bdes=[(1, 2)])
        ch3oc2h5.final_xyz = ch3oc2h5.conformers[0]

        resulting_species = ch3oc2h5.scissors()
        self.assertEqual(len(resulting_species), 2)
        for spc in resulting_species:
            self.assertIn(spc.label, ['ch3oc2h5_BDE_1_2_A', 'ch3oc2h5_BDE_1_2_B'])
            self.assertIn(spc.mol.to_smiles(), ['CC[O]', '[CH3]'])

        ch3oc2h5.bdes = ['all_h']
        resulting_species = ch3oc2h5.scissors()
        self.assertEqual(len(resulting_species), 9)  # inc. H
        self.assertIn('H', [spc.label for spc in resulting_species])
        xyz1 = """C       1.37610855    1.22086621   -0.01539125
                  O       0.04551655    0.70945921   -0.01539125
                  C       0.04367755   -0.72064979   -0.01539125
                  C      -1.40475645   -1.19890179   -0.01539125
                  H       1.30321755    2.31957421   -0.01539125
                  H       1.93206055    0.89033821    0.88640075
                  H       0.58239655   -1.09531579   -0.91266625
                  H       0.58239655   -1.09531579    0.88188375
                  H      -1.44039245   -2.30287879   -0.01539125
                  H      -1.93050745   -0.82845379    0.88098475
                  H      -1.93050745   -0.82845379   -0.91176725"""
        self.assertTrue(any([almost_equal_coords_lists(str_to_xyz(xyz1), spc.conformers) for spc in resulting_species]))
        self.assertEqual(resulting_species[0].multiplicity, 2)
        self.assertEqual(resulting_species[0].multiplicity, 2)

        ch3ch2o_xyz = """C  1.0591720 -0.5840990  0.0000000
                         C  0.0000000  0.5217430  0.0000000
                         O -1.3168640  0.0704390  0.0000000
                         H  2.0740310 -0.1467690  0.0000000
                         H  0.9492620 -1.2195690  0.8952700
                         H  0.9492620 -1.2195690 -0.8952700
                         H  0.1036620  1.1982630  0.8800660
                         H  0.1036620  1.1982630 -0.8800660"""
        ch3ch2o = ARCSpecies(label='ch3ch2o', smiles='CC[O]', xyz=ch3ch2o_xyz, multiplicity=2, bdes=[(1, 2)])
        ch3ch2o.final_xyz = ch3ch2o.conformers[0]
        spc1, spc2 = ch3ch2o.scissors()
        self.assertEqual(spc2.mol.to_smiles(), '[CH3]')
        expected_conformer0 = {'symbols': ('C', 'O', 'H', 'H'), 'isotopes': (12, 16, 1, 1),
                               'coords': ((0.6948946528715227, 0.1950960079388373, 0.0),
                                          (-0.6219693471284773, -0.2562079920611626, 0.0),
                                          (0.7985566528715228, 0.8716160079388374, 0.880066),
                                          (0.7985566528715228, 0.8716160079388374, -0.880066))}
        self.assertTrue(almost_equal_coords_lists(spc1.conformers[0], expected_conformer0))
        self.assertEqual(spc1.multiplicity, 3)
        self.assertEqual(spc2.multiplicity, 2)

        xyz0 = """ C                  2.66919769   -3.26620310   -0.74374716
                   H                  2.75753007   -4.15036595   -1.33567513
                   H                  1.77787951   -3.31806642   -0.15056757
                   H                  3.52403140   -3.18425165   -0.10554699
                   O                  2.58649229   -2.11402779   -1.57991769
                   N                  1.53746313   -2.25602697   -2.43070981
                   H                  1.84309991   -2.72880952   -3.25551907
                   S                  0.92376743   -0.70765861   -2.81176827
                   C                  0.24515773   -0.06442163   -1.29416377
                   H                  0.49440635   -0.20089032   -0.26642534
                   C                  0.91953671    1.90163977   -2.36604187
                   H                  1.60635907    2.48833509   -2.94076293
                   H                  1.05606550    1.06933422   -3.02405448
                   C                 -0.52564195    1.78158435   -2.66011055
                   H                 -0.65465370    1.24680889   -3.58012497
                   H                 -1.05821524    2.70904379   -2.74109386
                   C                 -0.94377584    0.94856998   -1.47293837
                   H                 -1.88836228    0.48072549   -1.65625619
                   H                 -1.02428550    1.53977616   -0.58246004
                   O                  1.32323842    0.95413994   -1.37785658"""
        spc0 = ARCSpecies(label='0',
                          smiles='CONSC1OCCC1', xyz=xyz0, bdes=[(6, 8), 'all_h'])
        spc0.final_xyz = spc0.conformers[0]
        spc_list = spc0.scissors()
        self.assertEqual(len(spc_list), 14)  # 11 H's, one H species, two non-H cut fragments
        for spc in spc_list:
            self.assertTrue(check_isomorphism(mol1=spc.mol,
                                              mol2=molecules_from_xyz(xyz=spc.conformers[0],
                                                                      multiplicity=spc.multiplicity,
                                                                      charge=spc.charge)[1]))
        self.assertTrue(any(spc.mol.to_smiles() == 'CO[NH]' for spc in spc_list))

    def test_net_charged_species(self):
        """Test that we can define and manipulate ions"""
        nh4 = ARCSpecies(label='NH4', smiles='[NH4+]', charge=1)
        nh4.determine_multiplicity(smiles='', adjlist='', mol=None)
        self.assertEqual(nh4.multiplicity, 1)

        cation = ARCSpecies(label='OCCCOH2', smiles='OCCC[OH2+]', charge=1)
        cation.determine_multiplicity(smiles='', adjlist='', mol=None)
        self.assertEqual(cation.multiplicity, 1)
        cation.generate_conformers()
        self.assertTrue(len(cation.conformers))

        anion = ARCSpecies(label='CCC(=O)[O-]', smiles='CCC(=O)[O-]', charge=-1)
        anion.determine_multiplicity(smiles='', adjlist='', mol=None)
        self.assertEqual(anion.multiplicity, 1)
        anion.generate_conformers()
        self.assertTrue(len(anion.conformers))

        anion_rad = ARCSpecies(label='[CH2]CC(=O)[O-]', smiles='[CH2]CC(=O)[O-]', charge=-1)
        anion_rad.determine_multiplicity(smiles='', adjlist='', mol=None)
        self.assertEqual(anion_rad.multiplicity, 2)
        self.assertEqual(anion_rad.charge, -1)
        anion_rad.generate_conformers()
        self.assertTrue(len(anion_rad.conformers))

        cation_rad = ARCSpecies(label='C1=[C]C=C([NH3])C=C1', smiles='C1=[C]C=C([NH3+])C=C1', charge=1)
        cation_rad.determine_multiplicity(smiles='', adjlist='', mol=None)
        self.assertEqual(cation_rad.multiplicity, 2)
        self.assertEqual(cation_rad.charge, 1)
        cation_rad.generate_conformers()
        self.assertTrue(len(cation_rad.conformers))


    @classmethod
    def tearDownClass(cls):
        """
        A function that is run ONCE after all unit tests in this class.
        Delete all project directories created during these unit tests
        """
        projects = ['arc_project_for_testing_delete_after_usage4']
        for project in projects:
            project_directory = os.path.join(arc_path, 'Projects', project)
            shutil.rmtree(project_directory)


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
        spc1 = Species().from_smiles('CON=O')
        spc1.label = str('CONO')
        spc2 = Species().from_smiles('C[N+](=O)[O-]')
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
        expected_dict = {'method': 'autotst',
                         'energy': None,
                         'family': 'H_Abstraction',
                         'index': None,
                         'rmg_reaction': 'CON=O <=> [O-][N+](=O)C',
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


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
