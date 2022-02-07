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

from arc.common import ARC_PATH, almost_equal_coords_lists
from arc.exceptions import SpeciesError
from arc.level import Level
from arc.plotter import save_conformers_file
from arc.species.converter import (check_isomorphism,
                                   molecules_from_xyz,
                                   str_to_xyz,
                                   xyz_to_str,
                                   xyz_to_x_y_z,
                                   )
from arc.species.species import (ARCSpecies,
                                 TSGuess,
                                 are_coords_compliant_with_graph,
                                 check_atom_balance,
                                 check_label,
                                 check_xyz,
                                 colliding_atoms,
                                 determine_rotor_symmetry,
                                 determine_rotor_type,
                                 )
from arc.species.xyz_to_2d import MolGraph


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
        cls.spc1_rmg = Species(molecule=[Molecule(smiles='C=C[O]')])  # delocalized radical + amine
        cls.spc1_rmg.label = 'vinoxy'
        cls.spc1 = ARCSpecies(rmg_species=cls.spc1_rmg)

        # Method 2: ARCSpecies object by XYZ (also give SMILES for thermo BAC)
        oh_xyz = """O       0.00000000    0.00000000   -0.12002167
        H       0.00000000    0.00000000    0.85098324"""
        cls.spc2 = ARCSpecies(label='OH', xyz=oh_xyz, smiles='[OH]', multiplicity=2, charge=0)

        # Method 3: ARCSpecies object by SMILES
        cls.spc3 = ARCSpecies(label='methylamine', smiles='CN', multiplicity=1, charge=0)

        # Method 4: ARCSpecies object by RMG Molecule object
        mol4 = Molecule().from_smiles('C=CC')
        cls.spc4 = ARCSpecies(label='propene', mol=mol4, multiplicity=1, charge=0)

        # Method 5: ARCSpecies by AdjacencyList (to generate AdjLists, see https://rmg.mit.edu/molecule_search)
        n2h4_adj = """1 N u0 p1 c0 {2,S} {3,S} {4,S}
        2 N u0 p1 c0 {1,S} {5,S} {6,S}
        3 H u0 p0 c0 {1,S}
        4 H u0 p0 c0 {1,S}
        5 H u0 p0 c0 {2,S}
        6 H u0 p0 c0 {2,S}"""
        cls.spc5 = ARCSpecies(label='N2H4', adjlist=n2h4_adj, multiplicity=1, charge=0)

        cls.n3_xyz = """N      -1.1997440839    -0.1610052059     0.0274738287
        H      -1.4016624407    -0.6229695533    -0.8487034080
        H      -0.0000018759     1.2861082773     0.5926077870
        N       0.0000008520     0.5651072858    -0.1124621525
        H      -1.1294692206    -0.8709078271     0.7537518889
        N       1.1997613019    -0.1609980472     0.0274604887
        H       1.1294795781    -0.8708998550     0.7537444446
        H       1.4015274689    -0.6230592706    -0.8487058662"""
        cls.spc6 = ARCSpecies(label='N3', xyz=cls.n3_xyz, multiplicity=1, smiles='NNN')

        xyz1 = os.path.join(ARC_PATH, 'arc', 'testing', 'xyz', 'AIBN.gjf')
        cls.spc7 = ARCSpecies(label='AIBN', smiles='N#CC(C)(C)N=NC(C)(C)C#N', xyz=xyz1)

        hso3_xyz = """S      -0.12383700    0.10918200   -0.21334200
        O       0.97332200   -0.98800100    0.31790100
        O      -1.41608500   -0.43976300    0.14487300
        O       0.32370100    1.42850400    0.21585900
        H       1.84477700   -0.57224200    0.35517700"""
        cls.spc8 = ARCSpecies(label='HSO3', xyz=hso3_xyz, multiplicity=2, charge=0, smiles='O[S](=O)=O')

        nh_s_adj = """1 N u0 p2 c0 {2,S}
                      2 H u0 p0 c0 {1,S}"""
        nh_s_xyz = """N       0.50949998    0.00000000    0.00000000
                      H      -0.50949998    0.00000000    0.00000000"""
        cls.spc9 = ARCSpecies(label='NH2_S_', adjlist=nh_s_adj, xyz=nh_s_xyz, multiplicity=1, charge=0)

        cls.spc10 = ARCSpecies(label='CCCCC', smiles='CCCCC')
        cls.spc11 = ARCSpecies(label='CCCNO', smiles='CCCNO')  # has chiral N
        cls.spc12 = ARCSpecies(label='[CH](CC[CH]c1ccccc1)c1ccccc1', smiles='[CH](CC[CH]c1ccccc1)c1ccccc1')
        cls.spc13 = ARCSpecies(label='CH3CHCH3', smiles='C[CH]C')

    def test_from_yml_file(self):
        """Test that an ARCSpecies object can successfully be loaded from an Arkane YAML file"""
        n4h6_adj_list = """1  N u0 p1 c0 {2,S} {3,S} {4,S}
2  H u0 p0 c0 {1,S}
3  H u0 p0 c0 {1,S}
4  N u0 p1 c0 {1,S} {5,S} {6,S}
5  H u0 p0 c0 {4,S}
6  N u0 p1 c0 {4,S} {7,S} {8,S}
7  H u0 p0 c0 {6,S}
8  N u0 p1 c0 {6,S} {9,S} {10,S}
9  H u0 p0 c0 {8,S}
10 H u0 p0 c0 {8,S}
"""
        n4h6_xyz = {'symbols': ('N', 'H', 'H', 'N', 'H', 'N', 'H', 'N', 'H', 'H'),
                    'isotopes': (14, 1, 1, 14, 1, 14, 1, 14, 1, 1),
                    'coords': ((1.359965, -0.537228, -0.184462),
                               (2.339584, -0.30093, -0.289911),
                               (1.2713739999999998, -1.27116, 0.51544),
                               (0.669838, 0.659561, 0.217548),
                               (0.61618, 0.715758, 1.2316809999999996),
                               (-0.669836, 0.659561, -0.217548),
                               (-0.616179, 0.715757, -1.231682),
                               (-1.3599669999999997, -0.537227, 0.184463),
                               (-2.339586, -0.300928, 0.289904),
                               (-1.2713739999999998, -1.271158, -0.51544))}

        n4h6_yml_path = os.path.join(ARC_PATH, 'arc', 'testing', 'yml_testing', 'N4H6.yml')
        n4h6 = ARCSpecies(yml_path=n4h6_yml_path)
        self.assertEqual(n4h6.mol.to_adjacency_list(), n4h6_adj_list)
        self.assertEqual(n4h6.charge, 0)
        self.assertEqual(n4h6.multiplicity, 1)
        self.assertEqual(n4h6.external_symmetry, 2)
        self.assertEqual(n4h6.mol.to_smiles(), 'NNNN')
        self.assertEqual(n4h6.optical_isomers, 2)
        self.assertEqual(n4h6.get_xyz(), n4h6_xyz)
        self.assertAlmostEqual(n4h6.e0, 273.2465365710362)

        c3_1_yml_path = os.path.join(ARC_PATH, 'arc', 'testing', 'yml_testing', 'C3_1.yml')
        c3_1 = ARCSpecies(yml_path=c3_1_yml_path)
        self.assertAlmostEqual(c3_1.e0, 86.34867237178679)

        c3_2_yml_path = os.path.join(ARC_PATH, 'arc', 'testing', 'yml_testing', 'C3_2.yml')
        c3_2 = ARCSpecies(yml_path=c3_2_yml_path)
        self.assertAlmostEqual(c3_2.e0, 72.98479932780415)

    def test_str(self):
        """Test the string representation of the object"""
        str_representation = str(self.spc9)
        expected_representation = 'ARCSpecies(label="NH2_S_", smiles="[NH]", is_ts=False, multiplicity=1, charge=0)'
        self.assertEqual(str_representation, expected_representation)

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

    def test_get_xyz(self):
        """Test the get_xyz() method."""
        n3 = ARCSpecies(label='N3', smiles='NNN', xyz=self.n3_xyz, multiplicity=1)
        xyz = n3.get_xyz()
        self.assertIsInstance(xyz, dict)
        expected_xyz = {'symbols': ('N', 'H', 'H', 'N', 'H', 'N', 'H', 'H'), 'isotopes': (14, 1, 1, 14, 1, 14, 1, 1),
                        'coords': ((-1.1997440839, -0.1610052059, 0.0274738287), (-1.4016624407, -0.6229695533, -0.848703408),
                                   (-1.8759e-06, 1.2861082773, 0.592607787), (8.52e-07, 0.5651072858, -0.1124621525),
                                   (-1.1294692206, -0.8709078271, 0.7537518889), (1.1997613019, -0.1609980472, 0.0274604887),
                                   (1.1294795781, -0.870899855, 0.7537444446), (1.4015274689, -0.6230592706, -0.8487058662))}
        self.assertTrue(almost_equal_coords_lists(xyz, expected_xyz))

        xyz = n3.get_xyz(return_format='str')
        self.assertIsInstance(xyz, str)
        self.assertEqual(xyz, xyz_to_str(expected_xyz))

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
        self.assertEqual(len(self.spc6.conformers), 8)

        self.spc8.conformers = list()
        self.spc8.generate_conformers()
        self.assertEqual(len(self.spc8.conformers), 1)

        self.spc8.conformers = list()
        self.spc8.generate_conformers(e_confs=10)
        self.assertEqual(len(self.spc8.conformers), 2)

        self.spc9.conformers = list()
        self.spc9.generate_conformers()
        self.assertEqual(len(self.spc9.conformers), 1)

        self.spc10.conformers = list()
        self.spc10.generate_conformers(n_confs=1)
        self.assertEqual(len(self.spc10.conformers), 1)

        self.spc10.conformers = list()
        self.spc10.generate_conformers(n_confs=2)
        self.assertEqual(len(self.spc10.conformers), 2)

        self.spc10.conformers = list()
        self.spc10.generate_conformers(n_confs=3)
        self.assertEqual(len(self.spc10.conformers), 3)

        self.spc11.conformers = list()
        self.spc11.generate_conformers(n_confs=1)
        self.assertEqual(len(self.spc11.conformers), 1)

        self.spc11.conformers = list()
        self.spc11.generate_conformers(n_confs=2)
        self.assertEqual(len(self.spc11.conformers), 2)

        self.spc11.conformers = list()
        self.spc11.generate_conformers(n_confs=3)
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
        self.spc = ARCSpecies(rmg_species=self.spc1_rmg, label='vinoxy')
        self.assertEqual(self.spc.label, 'vinoxy')
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

        self.assertEqual(self.spc1.rotors_dict[0]['pivots'], [1, 2])
        self.assertEqual(self.spc1.rotors_dict[0]['scan'], [4, 1, 2, 3])
        self.assertTrue(all([t in [1, 4, 5] for t in self.spc1.rotors_dict[0]['top']]))
        self.assertEqual(self.spc1.rotors_dict[0]['times_dihedral_set'], 0)
        self.assertEqual(self.spc3.rotors_dict[0]['pivots'], [1, 2])
        self.assertEqual(self.spc4.rotors_dict[0]['pivots'], [2, 3])
        self.assertEqual(self.spc5.rotors_dict[0]['pivots'], [1, 2])
        self.assertEqual(self.spc6.rotors_dict[0]['pivots'], [1, 4])
        self.assertEqual(self.spc6.rotors_dict[0]['scan'], [2, 1, 4, 6])
        self.assertEqual(len(self.spc6.rotors_dict[0]['top']), 3)
        self.assertTrue(all([t in [1, 5, 2] for t in self.spc6.rotors_dict[0]['top']]))
        self.assertEqual(self.spc6.rotors_dict[1]['pivots'], [4, 6])
        self.assertEqual(self.spc6.rotors_dict[1]['scan'], [1, 4, 6, 7])
        self.assertEqual(len(self.spc6.rotors_dict[1]['top']), 3)
        self.assertTrue(all([t in [6, 7, 8] for t in self.spc6.rotors_dict[1]['top']]))

        ts_xyz1 = {'symbols': ('O', 'C', 'N', 'C', 'H', 'H', 'H', 'H'),
                   'isotopes': (16, 12, 14, 12, 1, 1, 1, 1),
                   'coords': ((-1.46891188, 0.4782021, -0.74907357), (-0.77981513, -0.5067346, 0.0024359),
                              (0.86369081, 0.1484285, 0.8912832), (1.78225246, 0.27014716, 0.17691),
                              (2.61878546, 0.38607062, -0.47459418), (-1.62732717, 1.19177937, -0.10791543),
                              (-1.40237804, -0.74595759, 0.87143836), (-0.39285462, -1.26299471, -0.69270021))}
        spc7 = ARCSpecies(label='TS1', xyz=ts_xyz1)
        spc7.determine_rotors()
        self.assertEqual(len(spc7.rotors_dict), 1)
        self.assertEqual(self.spc1.rotors_dict[0]['pivots'], [1, 2])

        ts_xyz2 = {'symbols': ('N', 'C', 'C', 'C', 'H', 'H', 'C', 'C', 'C', 'C', 'H', 'H', 'C', 'C', 'C', 'H', 'C',
                               'C', 'N', 'H', 'H', 'C', 'H', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'C', 'C', 'C', 'H',
                               'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'O', 'O', 'C', 'H', 'H', 'H'),
                   'isotopes': (14, 12, 12, 12, 1, 1, 12, 12, 12, 12, 1, 1, 12, 12, 12, 1, 12, 12, 14, 1, 1, 12, 1, 12,
                                12, 12, 1, 1, 1, 1, 12, 12, 12, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 16, 16, 12, 1, 1,
                                1),
                   'coords': ((0.172994, 0.023999, -0.057725), (1.623343, 0.008256, -0.050587),
                              (-0.436897, 1.293199, -0.199903), (2.215336, -1.118676, -0.852764),
                              (1.964199, 0.961552, -0.490082), (2.050636, -0.010449, 0.97411),
                              (-0.507086, -1.061033, 0.546055), (0.095881, 2.419891, 0.442226),
                              (-1.538625, 1.444158, -1.059615), (3.589515, -1.56886, -0.426249),
                              (1.50829, -1.903881, -1.154574), (2.487071, -0.520945, -2.076727),
                              (0.211549, -1.876423, 1.446456), (-1.860548, -1.388722, 0.279489),
                              (-0.449507, 3.684447, 0.22785), (0.942862, 2.303356, 1.12196),
                              (-2.101046, 0.228199, -1.723519), (-2.086935, 2.71489, -1.248249),
                              (3.573457, -2.277274, 0.845492), (4.221519, -0.672818, -0.309381),
                              (4.065649, -2.176083, -1.227842), (-0.376908, -2.96072, 2.087481),
                              (1.264831, -1.671103, 1.64345), (-2.773075, -0.686025, -0.702983),
                              (-2.426791, -2.477745, 0.957426), (-1.546572, 3.837073, -0.619888),
                              (-0.019933, 4.552027, 0.736196), (-2.833211, 0.523007, -2.490838),
                              (-1.29597, -0.328211, -2.23538), (-2.943449, 2.82556, -1.920552),
                              (4.841766, -2.220868, 1.531231), (3.090796, -3.632265, 0.715735),
                              (-1.715824, -3.265496, 1.856119), (0.221986, -3.563668, 2.775908),
                              (-3.3472, -1.463634, -1.23465), (-3.520842, -0.093098, -0.145563),
                              (-3.473406, -2.720774, 0.746466), (-1.980665, 4.825631, -0.790906),
                              (4.757108, -2.706464, 2.515664), (5.664566, -2.726167, 0.976822),
                              (5.140202, -1.17481, 1.698765), (2.989906, -4.094171, 1.709796),
                              (3.768404, -4.273274, 0.107669), (2.096249, -3.645583, 0.248617),
                              (-2.196951, -4.110638, 2.354751), (2.839195, 0.106683, -2.975438),
                              (2.533432, 1.405371, -2.628984), (1.274047, 1.739318, -3.167728),
                              (1.054829, 2.755858, -2.812066), (0.496295, 1.05356, -2.797583),
                              (1.298699, 1.715159, -4.269555))}
        spc8 = ARCSpecies(label='TS2', xyz=ts_xyz2)
        spc8.determine_rotors()
        self.assertEqual(len(spc8.rotors_dict), 8)
        self.assertEqual(spc8.rotors_dict[0]['pivots'], [1, 2])
        self.assertEqual(spc8.rotors_dict[1]['pivots'], [2, 4])
        self.assertEqual(spc8.rotors_dict[2]['pivots'], [4, 10])
        self.assertEqual(spc8.rotors_dict[3]['pivots'], [10, 19])
        self.assertEqual(spc8.rotors_dict[4]['pivots'], [19, 31])
        self.assertEqual(spc8.rotors_dict[5]['pivots'], [19, 32])
        self.assertEqual(spc8.rotors_dict[6]['pivots'], [46, 47])
        self.assertEqual(spc8.rotors_dict[7]['pivots'], [47, 48])

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
        self.assertEqual(len(spc1.rotors_dict.keys()), 12)
        self.assertEqual(spc1.rotors_dict[6]['dimensions'], 3)
        self.assertIn([1, 3], spc1.rotors_dict[6]['pivots'])
        self.assertIn([1, 2], spc1.rotors_dict[6]['pivots'])
        self.assertIn([2, 4], spc1.rotors_dict[6]['pivots'])
        self.assertEqual(spc1.rotors_dict[6]['cont_indices'], [])

        spc2 = ARCSpecies(label='propanol', smiles='CCO', directed_rotors={'brute_force_sp': [['all']]})
        spc2.determine_rotors()  # also initializes directed_rotors
        self.assertEqual(spc2.directed_rotors, {'brute_force_sp': [[[4, 1, 2, 3], [1, 2, 3, 9]]]})
        self.assertEqual(len(spc2.rotors_dict), 3)
        self.assertEqual(spc2.rotors_dict[0]['dimensions'], 1)
        self.assertEqual(spc2.rotors_dict[1]['dimensions'], 1)
        self.assertEqual(spc2.rotors_dict[2]['dimensions'], 2)

    def test_symmetry(self):
        """Test external symmetry and chirality determination"""
        allene = ARCSpecies(label='allene', smiles='C=C=C', multiplicity=1, charge=0)
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

        ammonia = ARCSpecies(label='ammonia', smiles='N', multiplicity=1, charge=0)
        ammonia.final_xyz = str_to_xyz("""N  0.06617   0.20024   0.13886
                               H  -0.62578  -0.34119   0.63709
                               H  -0.32018   0.51306  -0.74036
                               H   0.87976  -0.37219  -0.03564""")
        ammonia.determine_symmetry()
        self.assertEqual(ammonia.optical_isomers, 1)
        self.assertEqual(ammonia.external_symmetry, 3)

        methane = ARCSpecies(label='methane', smiles='C', multiplicity=1, charge=0)
        methane.final_xyz = str_to_xyz("""C   0.00000   0.00000   0.00000
                               H  -0.29717   0.97009  -0.39841
                               H   1.08773  -0.06879   0.01517
                               H  -0.38523  -0.10991   1.01373
                               H -0.40533  -0.79140  -0.63049""")
        methane.determine_symmetry()
        self.assertEqual(methane.optical_isomers, 1)
        self.assertEqual(methane.external_symmetry, 12)

        chiral = ARCSpecies(label='chiral', smiles='C(C)(O)(N)', multiplicity=1, charge=0)
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

        s8 = ARCSpecies(label='s8', smiles='S1SSSSSSS1', multiplicity=1, charge=0)
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

        water = ARCSpecies(label='H2O', smiles='O', multiplicity=1, charge=0)
        water.final_xyz = str_to_xyz("""O   0.19927   0.29049  -0.11186
                             H   0.50770  -0.61852  -0.09124
                             H  -0.70697   0.32803   0.20310""")
        water.determine_symmetry()
        self.assertEqual(water.optical_isomers, 1)
        self.assertEqual(water.external_symmetry, 2)

        # test setting only symmetry, preserving optical isomers
        ch2oh = ARCSpecies(label='CH2OH', smiles='[CH2]O', optical_isomers=1)
        ch2oh.determine_symmetry()
        self.assertEqual(ch2oh.optical_isomers, 1)
        self.assertEqual(ch2oh.external_symmetry, 1)

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
        spc1 = ARCSpecies(label='spc1', mol=Molecule(smiles='C[CH]C'), compute_thermo=False)
        spc2 = ARCSpecies(label='spc2', mol=Molecule(smiles='CCC'), compute_thermo=False)
        spc3 = ARCSpecies(label='spc3', smiles='N[NH]', compute_thermo=False)
        spc4 = ARCSpecies(label='spc4', smiles='NNN', compute_thermo=False)
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
        spc5 = ARCSpecies(label='spc5', adjlist=adj1, compute_thermo=False)
        spc6 = ARCSpecies(label='spc6', adjlist=adj2, compute_thermo=False)
        xyz1 = """O       0.00000000    0.00000000   -0.10796235
                  H       0.00000000    0.00000000    0.86318839"""
        xyz2 = """N      -0.74678912   -0.11808620    0.00000000
                  C       0.70509190    0.01713703    0.00000000
                  H       1.11547042   -0.48545356    0.87928385
                  H       1.11547042   -0.48545356   -0.87928385
                  H       1.07725194    1.05216961    0.00000000
                  H      -1.15564250    0.32084669    0.81500594
                  H      -1.15564250    0.32084669   -0.81500594"""
        spc7 = ARCSpecies(label='spc7', xyz=xyz1, compute_thermo=False)
        spc8 = ARCSpecies(label='spc8', xyz=xyz2, compute_thermo=False)

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
        expected_dict = {'number_of_rotors': 0,
                         'multiplicity': 1,
                         'arkane_file': None,
                         'mol': {'atom_order': spc_dict['mol']['atom_order'],
                                 'atoms': spc_dict['mol']['atoms'],
                                 'multiplicity': 1, 'props': {}},
                         'compute_thermo': True,
                         'label': 'methylamine',
                         'long_thermo_description': spc_dict['long_thermo_description'],
                         'charge': 0,
                         'consider_all_diastereomers': True,
                         'force_field': 'MMFF94s',
                         'is_ts': False,
                         'bond_corrections': {'C-H': 3, 'C-N': 1, 'H-N': 2}}
        self.assertEqual(spc_dict, expected_dict)
        self.assertEqual(len(set([spc_dict['mol']['atoms'][i]['id'] for i in range(len(spc_dict['mol']['atoms']))])),
                         len(spc_dict['mol']['atoms']))
        self.assertEqual(spc_dict['mol']['atoms'][0]['radical_electrons'], 0)
        self.assertEqual(spc_dict['mol']['atoms'][0]['charge'], 0)
        self.assertEqual(spc_dict['mol']['atoms'][0]['lone_pairs'], 0)
        self.assertEqual(spc_dict['mol']['atoms'][0]['props'], {'inRing': False})
        self.assertEqual(spc_dict['mol']['atoms'][0]['element']['number'], 6)
        self.assertEqual(spc_dict['mol']['atoms'][0]['element']['isotope'], -1)
        self.assertEqual(spc_dict['mol']['atoms'][0]['atomtype'], 'Cs')

    def test_from_dict(self):
        """Test Species.from_dict()"""
        species_dict = self.spc2.as_dict()
        spc = ARCSpecies(species_dict=species_dict)
        self.assertEqual(spc.multiplicity, 2)
        self.assertEqual(spc.charge, 0)
        self.assertEqual(spc.label, 'OH')
        self.assertEqual(spc.mol.to_smiles(), '[OH]')
        self.assertFalse(spc.is_ts)

        species_dict = self.spc13.as_dict()
        spc = ARCSpecies(species_dict=species_dict)
        self.assertEqual(spc.multiplicity, 2)
        self.assertEqual(spc.charge, 0)
        self.assertEqual(spc.label, 'CH3CHCH3')
        self.assertEqual(spc.mol.to_smiles(), 'C[CH]C')
        self.assertFalse(spc.is_ts)

        species_dict = {
            'arkane_file': None,
            'bond_corrections': {'C-C': 2, 'C-H': 7},
            'charge': 0,
            'cheap_conformer': """C      -1.28873024    0.06292844    0.10889819
    C       0.01096161   -0.45756396   -0.39342150
    C       1.28410310    0.11324608    0.12206177
    H      -1.49844465    1.04581965   -0.32238736
    H      -1.28247249    0.14649430    1.19953628
    H      -2.09838469   -0.61664655   -0.17318515
    H       0.02736023   -1.06013834   -1.29522253
    H       2.12255117   -0.53409831   -0.15158596
    H       1.26342625    0.19628892    1.21256167
    H       1.45962973    1.10366979   -0.30725541""",
            'compute_thermo': True,
            'conf_is_isomorphic': True,
            'conformer_energies': {-310736.67239208287, -310736.6722398039},
            'conformers': ["""C       1.29970500    0.14644400    0.33188600
    C      -0.03768600   -0.16670600    0.90640300
    C      -1.27231500    0.46087300    0.35997600
    H       1.44120400    1.23441200    0.20525800
    H       2.11937800   -0.23320300    0.95927900
    H       1.43056000   -0.29786500   -0.67654600
    H      -0.13536400   -1.02567100    1.57749500
    H      -1.52797900    0.06385600   -0.64425600
    H      -2.14561400    0.28815000    1.00583400
    H      -1.15029500    1.55122700    0.23360600""",
                           """C       1.30103900    0.16393300    0.34879900
                           C      -0.04152900   -0.19847900    0.88071100
                           C      -1.28197500    0.35864300    0.27435100
                           H       1.56963800    1.21419300    0.58585100
                           H       2.09641900   -0.47262500    0.76327200
                           H       1.33590500    0.08223900   -0.75201500
                           H      -0.10685900   -0.69364300    1.85448600
                           H      -1.26682800    0.27562700   -0.82682700
                           H      -2.18634800   -0.14787200    0.64197300
                           H      -1.40179300    1.43901800    0.49754100"""],
            'conformers_before_opt': ["""C       1.29196387    0.15815210    0.32047503
    C      -0.03887789   -0.17543467    0.89494533
    C      -1.26222918    0.47039644    0.34836510
    H       1.40933232    1.23955428    0.20511486
    H       2.08593721   -0.19903577    0.98301313
    H       1.41699441   -0.31973461   -0.65525752
    H      -0.13933823   -1.05339936    1.52398873
    H      -1.51964710    0.03926484   -0.62319221
    H      -2.10441807    0.31322346    1.02876738
    H      -1.11812298    1.54852996    0.23271515""",
                                      """C       1.29310340    0.17362129    0.33579983
                                      C      -0.04076055   -0.18590306    0.88714873
                                      C      -1.27189499    0.36735010    0.26212474
                                      H       1.55592178    1.19894970    0.61091750
                                      H       2.05776887   -0.49750870    0.73805031
                                      H       1.30408843    0.08336402   -0.75426781
                                      H      -0.10304141   -0.63591124    1.87214802
                                      H      -1.23406042    0.27508450   -0.82717286
                                      H      -2.15031563   -0.17969330    0.61716567
                                      H      -1.39313889    1.42168001    0.52622776"""],
            'consider_all_diastereomers': True,
            'e_elect': -310565.27164853434,
            'final_xyz': """C       1.30187300    0.14621800    0.33152500
    C      -0.03806200   -0.16945500    0.90272000
    C      -1.27448200    0.46117100    0.35965700
    H       1.44177100    1.22731000    0.20493900
    H       2.11385600   -0.22881400    0.95955100
    H       1.43997500   -0.29695900   -0.66977600
    H      -0.13482800   -1.02046600    1.56833500
    H      -1.53677300    0.06694200   -0.63727300
    H      -2.13918500    0.29111100    1.00599100
    H      -1.15255100    1.54445900    0.23326700""",
            'force_field': 'MMFF94s',
            'initial_xyz': """C       1.29970500    0.14644400    0.33188600
    C      -0.03768600   -0.16670600    0.90640300
    C      -1.27231500    0.46087300    0.35997600
    H       1.44120400    1.23441200    0.20525800
    H       2.11937800   -0.23320300    0.95927900
    H       1.43056000   -0.29786500   -0.67654600
    H      -0.13536400   -1.02567100    1.57749500
    H      -1.52797900    0.06385600   -0.64425600
    H      -2.14561400    0.28815000    1.00583400
    H      -1.15029500    1.55122700    0.23360600""",
            'is_ts': False,
            'label': 'C3_2',
            'long_thermo_description': "Bond corrections: {'C-H': 7, 'C-C': 2}",
            'mol': {
                'atom_order': [-32758, -32757, -32756, -32755, -32754, -32753, -32752, -32751, -32750, -32749],
                'atoms': [
                    {'charge': 0, 'edges': {-32757: 1.0, -32755: 1.0, -32754: 1.0, -32753: 1.0},
                     'element': {'isotope': -1, 'mass': 0.01201064046472311, 'name': 'carbon', 'number': 6, 'symbol': 'C'},
                     'id': -32758, 'label': '', 'lone_pairs': 0, 'props': {'inRing': False}, 'radical_electrons': 0},
                    {'charge': 0, 'edges': {-32758: 1.0, -32756: 1.0, -32752: 1.0},
                     'element': {'isotope': -1, 'mass': 0.01201064046472311, 'name': 'carbon', 'number': 6, 'symbol': 'C'},
                     'id': -32757, 'label': '', 'lone_pairs': 0, 'props': {'inRing': False}, 'radical_electrons': 1},
                    {'charge': 0, 'edges': {-32757: 1.0, -32751: 1.0, -32750: 1.0, -32749: 1.0},
                     'element': {'isotope': -1, 'mass': 0.01201064046472311, 'name': 'carbon', 'number': 6, 'symbol': 'C'},
                     'id': -32756, 'label': '', 'lone_pairs': 0, 'props': {'inRing': False}, 'radical_electrons': 0},
                    {'charge': 0, 'edges': {-32758: 1.0},
                     'element': {'isotope': -1, 'mass': 0.0010079710045829415, 'name': 'hydrogen', 'number': 1, 'symbol': 'H'},
                     'id': -32755, 'label': '', 'lone_pairs': 0, 'props': {'inRing': False}, 'radical_electrons': 0},
                    {'charge': 0, 'edges': {-32758: 1.0},
                     'element': {'isotope': -1, 'mass': 0.0010079710045829415, 'name': 'hydrogen', 'number': 1, 'symbol': 'H'},
                     'id': -32754, 'label': '', 'lone_pairs': 0, 'props': {'inRing': False}, 'radical_electrons': 0},
                    {'charge': 0, 'edges': {-32758: 1.0},
                     'element': {'isotope': -1, 'mass': 0.0010079710045829415, 'name': 'hydrogen', 'number': 1, 'symbol': 'H'},
                     'id': -32753, 'label': '', 'lone_pairs': 0, 'props': {'inRing': False}, 'radical_electrons': 0},
                    {'charge': 0, 'edges': {-32757: 1.0},
                     'element': {'isotope': -1, 'mass': 0.0010079710045829415, 'name': 'hydrogen', 'number': 1, 'symbol': 'H'},
                     'id': -32752, 'label': '', 'lone_pairs': 0, 'props': {'inRing': False}, 'radical_electrons': 0},
                    {'charge': 0, 'edges': {-32756: 1.0},
                     'element': {'isotope': -1, 'mass': 0.0010079710045829415, 'name': 'hydrogen', 'number': 1, 'symbol': 'H'},
                     'id': -32751, 'label': '', 'lone_pairs': 0, 'props': {'inRing': False}, 'radical_electrons': 0},
                    {'charge': 0, 'edges': {-32756: 1.0},
                     'element': {'isotope': -1, 'mass': 0.0010079710045829415, 'name': 'hydrogen', 'number': 1, 'symbol': 'H'},
                     'id': -32750, 'label': '', 'lone_pairs': 0, 'props': {'inRing': False}, 'radical_electrons': 0},
                    {'charge': 0, 'edges': {-32756: 1.0},
                     'element': {'isotope': -1, 'mass': 0.0010079710045829415, 'name': 'hydrogen', 'number': 1, 'symbol': 'H'},
                     'id': -32749, 'label': '', 'lone_pairs': 0, 'props': {'inRing': False}, 'radical_electrons': 0}],
                'multiplicity': 2,
                'props': {}},
        }
        spc = ARCSpecies(species_dict=species_dict)
        self.assertEqual(spc.mol.copy(deep=True).to_smiles(), 'C[CH]C')
        self.assertEqual(spc.force_field, 'MMFF94s')
        for index in [0, 1, 2]:
            self.assertEqual(spc.mol.atoms[index].element.symbol, 'C')
        for index in [3, 4, 5, 6, 7, 8, 9]:
            self.assertEqual(spc.mol.atoms[index].element.symbol, 'H')
        self.assertFalse(spc.mol.is_aromatic())

        species_dict = {'arkane_file': None, 'bond_corrections': {}, 'charge': 0,
                        'chosen_ts': 12, 'chosen_ts_list': [15, 14, 12], 'chosen_ts_method': 'gcn',
                        'compute_thermo': False, 'consider_all_diastereomers': True, 'e_elect': -310382.39770689985,
                        'final_xyz': """C       1.24308800   -0.63728100   -0.83084500
                                     C       0.47242200   -0.00167700    0.28034600
                                     C      -0.97840400   -0.32227800    0.48081000
                                     H       2.31900300   -0.72163700   -0.75942700
                                     H       0.71709200   -1.25228600   -1.54944600
                                     H       0.82531600    0.59717200   -0.81605600
                                     H       1.05269200    0.35819200    1.12279600
                                     H      -1.10906700   -1.30741000    0.95493200
                                     H      -1.47674600    0.41900300    1.11033300
                                     H      -1.51333600   -0.36069700   -0.47419600""",
                        'initial_xyz': """C       1.24048800   -0.62930500   -0.83486000
                                       C       0.47277300   -0.00261400    0.27886600
                                       C      -0.97665700   -0.32246100    0.47924600
                                       H       2.32489500   -0.72600100   -0.75707500
                                       H       0.70824900   -1.25659800   -1.55384200
                                       H       0.82237000    0.60748600   -0.81298600
                                       H       1.05824200    0.35217700    1.13233100
                                       H      -1.10589600   -1.30876200    0.96446300
                                       H      -1.48016200    0.42782200    1.10607600
                                       H      -1.51224100   -0.37064400   -0.48297200""",
                        'is_ts': True, 'label': 'TS0', 'long_thermo_description': '', 'force_field': 'MMFF94s',
                        'mol': {'atom_order': [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                'atoms': [{'charge': 0, 'edges': {-1: 1.0}, 'radical_electrons': 0,
                                           'element': {'isotope': -1, 'mass': 0.01201064046472311, 'name': 'carbon',
                                                       'number': 6, 'symbol': 'C'},
                                           'id': -1, 'label': '', 'lone_pairs': 0, 'props': {'inRing': False}},
                                          {'charge': 0, 'edges': {-1: 1.0}, 'radical_electrons': 1,
                                           'element': {'isotope': -1, 'mass': 0.01201064046472311, 'name': 'carbon',
                                                       'number': 6, 'symbol': 'C'},
                                           'id': -1, 'label': '', 'lone_pairs': 0, 'props': {'inRing': False}},
                                          {'charge': 0, 'edges': {-1: 1.0}, 'radical_electrons': 0,
                                           'element': {'isotope': -1, 'mass': 0.01201064046472311, 'name': 'carbon',
                                                       'number': 6, 'symbol': 'C'},
                                           'id': -1, 'label': '', 'lone_pairs': 0, 'props': {'inRing': False}},
                                          {'charge': 0, 'edges': {-1: 1.0}, 'radical_electrons': 0,
                                           'element': {'isotope': -1, 'mass': 0.0010079710045829415, 'name': 'hydrogen',
                                                       'number': 1, 'symbol': 'H'},
                                           'id': -1, 'label': '', 'lone_pairs': 0, 'props': {'inRing': False}},
                                          {'charge': 0, 'edges': {-1: 1.0}, 'radical_electrons': 0,
                                           'element': {'isotope': -1, 'mass': 0.0010079710045829415, 'name': 'hydrogen',
                                                       'number': 1, 'symbol': 'H'},
                                           'id': -1, 'label': '', 'lone_pairs': 0, 'props': {'inRing': False}},
                                          {'charge': 0, 'edges': {-1: 1.0}, 'radical_electrons': 0,
                                           'element': {'isotope': -1, 'mass': 0.0010079710045829415, 'name': 'hydrogen',
                                                       'number': 1, 'symbol': 'H'},
                                           'id': -1, 'label': '', 'lone_pairs': 0, 'props': {'inRing': False}},
                                          {'charge': 0, 'edges': {-1: 1.0}, 'radical_electrons': 0,
                                           'element': {'isotope': -1, 'mass': 0.0010079710045829415, 'name': 'hydrogen',
                                                       'number': 1, 'symbol': 'H'},
                                           'id': -1, 'label': '', 'lone_pairs': 0, 'props': {'inRing': False}},
                                          {'charge': 0, 'edges': {-1: 1.0}, 'radical_electrons': 0,
                                           'element': {'isotope': -1, 'mass': 0.0010079710045829415, 'name': 'hydrogen',
                                                       'number': 1, 'symbol': 'H'},
                                           'id': -1, 'label': '', 'lone_pairs': 0, 'props': {'inRing': False}},
                                          {'charge': 0, 'edges': {-1: 1.0}, 'radical_electrons': 0,
                                           'element': {'isotope': -1, 'mass': 0.0010079710045829415, 'name': 'hydrogen',
                                                       'number': 1, 'symbol': 'H'},
                                           'id': -1, 'label': '', 'lone_pairs': 0, 'props': {'inRing': False}},
                                          {'charge': 0, 'edges': {-1: 1.0}, 'radical_electrons': 0,
                                           'element': {'isotope': -1, 'mass': 0.0010079710045829415, 'name': 'hydrogen',
                                                       'number': 1, 'symbol': 'H'},
                                           'id': -1, 'label': '', 'lone_pairs': 0, 'props': {'inRing': False}}],
                                'multiplicity': 2, 'props': {}},
                        'multiplicity': 2, 'number_of_rotors': 2, 'opt_level': 'cbs-qb3',
                        'rotors_dict': {0: {'cont_indices': [], 'dimensions': 1, 'directed_scan': {},
                                            'directed_scan_type': 'ess',
                                            'invalidation_reason': 'Pivots participate in the TS reaction zone (code: pivTS). ',
                                            'number_of_running_jobs': 0, 'original_dihedrals': [], 'pivots': [1, 2],
                                            'scan': [4, 1, 2, 3], 'scan_path': '', 'success': False,
                                            'times_dihedral_set': 0, 'top': [1, 4, 5, 6], 'torsion': [3, 0, 1, 2],
                                            'trsh_counter': 0, 'trsh_methods': []},
                                        1: {'cont_indices': [], 'dimensions': 1, 'directed_scan': {},
                                            'directed_scan_type': 'ess', 'invalidation_reason': '',
                                            'number_of_running_jobs': 0, 'original_dihedrals': [], 'pivots': [2, 3],
                                            'scan': [1, 2, 3, 8], 'scan_path': '', 'success': None,
                                            'times_dihedral_set': 0, 'top': [3, 8, 9, 10], 'torsion': [0, 1, 2, 7],
                                            'trsh_counter': 0, 'trsh_methods': []}},
                        'run_time': 1238.0, 'rxn_index': 0,
                        'rxn_label': 'C3_1 <=> C3_2',
                        'successful_methods': ['autotst', 'autotst', 'autotst', 'autotst',
                                               'gcn', 'gcn', 'gcn', 'gcn', 'gcn', 'gcn', 'gcn', 'gcn', 'gcn', 'gcn',
                                               'kinbot', 'kinbot'],
                        'ts_checks': {'E0': False, 'IRC': False, 'e_elect': True, 'freq': True,
                                      'normal_mode_displacement': True, 'warnings': ''},
                        'ts_conf_spawned': True,
                        'ts_guesses': [{'conformer_index': 0, 'energy': 455.24421061505564,
                                        'execution_time': '0:00:25.483065',
                                        'imaginary_freqs': None, 'index': 0,
                                        'initial_xyz': """C       0.06870000   -0.52310000   -0.65000000
                                                          C       1.32690000   -0.17800000    0.12310000
                                                          C      -1.61580000    0.23640000    0.43190000
                                                          H      -0.94590000   -0.88470000    0.05810000
                                                          H       0.00080000   -0.05630000   -1.65490000
                                                          H       1.36660000   -0.77570000    1.05730000
                                                          H       2.21840000   -0.40920000   -0.49590000
                                                          H       1.32650000    0.90240000    0.37660000
                                                          H      -1.31000000    1.13360000    1.01690000
                                                          H      -2.43630000    0.55470000   -0.26310000""",
                                        'method': 'autotst', 'method_direction': 'F', 'method_index': 0,
                                        'opt_xyz': """C       0.47274600   -0.76452100   -0.88459200
                                                      C       1.44656200   -0.23763200    0.09307300
                                                      C      -1.72938800    0.32911600    0.28782700
                                                      H      -1.69734300   -0.70598100    0.63266200
                                                      H      -0.10051300    0.01283500   -1.43240400
                                                      H       1.69764900   -0.96652300    0.88086600
                                                      H       2.37671100   -0.12775000   -0.50513500
                                                      H       1.24091800    0.74738900    0.54623600
                                                      H      -1.30693200    1.11824100    0.91137000
                                                      H      -2.40051000    0.59492500   -0.52990300""",
                                        'success': True, 'successful_irc': None, 'successful_normal_mode': None,
                                        't0': '2021-08-22T07:40:04.565352'},
                                       {'conformer_index': 1, 'energy': 485.01425728958566, 'execution_time': '0:00:25.483065',
                                        'imaginary_freqs': None, 'index': 1,
                                        'initial_xyz': """C       1.27140000   -0.19880000    0.27060000
                                                          C      -1.52300000    0.74650000   -0.35470000
                                                          C       0.02370000   -0.73290000   -0.41700000
                                                          H      -0.96230000   -0.12240000   -1.12430000
                                                          H       1.61070000    0.72270000   -0.24570000
                                                          H       1.04290000    0.03790000    1.33060000
                                                          H       2.07780000   -0.96010000    0.22660000
                                                          H      -1.03890000    1.73950000   -0.25260000
                                                          H      -2.20860000    0.44390000    0.46320000
                                                          H      -0.29360000   -1.67630000    0.10330000""",
                                        'method': 'autotst', 'method_direction': 'F', 'method_index': 1,
                                        'opt_xyz': """C       1.66858400   -0.47894800    0.18483000
                                                      C      -2.24898800    1.37779300   -0.16738600
                                                      C       0.19196200   -0.39189200    0.04997800
                                                      H      -0.30502700    0.59228700    0.04964900
                                                      H       2.17438800   -0.42762500   -0.80039200
                                                      H       2.07468500    0.34685900    0.78841600
                                                      H       1.98194700   -1.42970200    0.64501700
                                                      H      -2.70375200    0.74656200    0.62073400
                                                      H      -2.42994200    0.93691500   -1.16726700
                                                      H      -0.40375800   -1.27225000   -0.20358000""",
                                        'success': True, 'successful_irc': None, 'successful_normal_mode': None,
                                        't0': '2021-08-22T07:40:04.565352'},
                                       {'conformer_index': 2, 'energy': None, 'execution_time': '0:00:21.880109',
                                        'imaginary_freqs': None, 'index': 2,
                                        'initial_xyz': """C       1.71270000   -0.29390000    0.04290000
                                                          C      -1.29740000   -0.16230000   -0.10640000
                                                          C      -0.10790000    0.76300000   -0.32970000
                                                          H       1.08610000    0.51490000   -0.81510000
                                                          H       1.48670000   -1.37900000   -0.00500000
                                                          H       2.09490000    0.08260000    1.01380000
                                                          H      -2.24090000    0.38520000   -0.31090000
                                                          H      -1.29900000   -0.52400000    0.94300000
                                                          H      -1.22320000   -1.03060000   -0.79300000
                                                          H      -0.21210000    1.64410000    0.36030000""",
                                        'method': 'autotst', 'method_direction': 'R', 'method_index': 2,
                                        'success': True, 'successful_irc': None, 'successful_normal_mode': None,
                                        't0': '2021-08-22T07:40:30.067177'},
                                       {'conformer_index': 3, 'energy': None, 'execution_time': '0:00:21.880109',
                                        'imaginary_freqs': None, 'index': 3,
                                        'initial_xyz': """C       0.14170000    0.77420000   -0.31580000
                                                          C       1.33000000   -0.02950000    0.17580000
                                                          C      -1.64460000   -0.24370000    0.27000000
                                                          H      -0.90890000    0.81700000    0.50710000
                                                          H       0.07300000    0.86010000   -1.42040000
                                                          H       1.38540000    0.02680000    1.28260000
                                                          H       2.26370000    0.38470000   -0.25780000
                                                          H       1.22000000   -1.08930000   -0.13430000
                                                          H      -2.43100000   -0.17220000   -0.52760000
                                                          H      -1.42930000   -1.32810000    0.42060000""",
                                        'method': 'autotst', 'method_direction': 'R', 'method_index': 3,
                                        'success': True, 'successful_irc': None, 'successful_normal_mode': None,
                                        't0': '2021-08-22T07:40:30.067177'},
                                       {'conformer_index': 4, 'energy': None, 'execution_time': '0:00:06.423237',
                                        'imaginary_freqs': None, 'index': 4,
                                        'initial_xyz': """C      -0.50205731    0.69866323    0.68881840
                                                          C       0.86601830   -0.07523254    0.81684917
                                                          C       1.79938066   -0.06883909   -0.47658044
                                                          H      -1.20263302    0.73346782    1.43291938
                                                          H      -0.66235149    1.47553778   -0.12268760
                                                          H      -0.09058958    0.58470058    0.54704428
                                                          H       0.80732340   -0.87154227    1.32850742
                                                          H       2.17241597    0.87013972   -0.69461155
                                                          H       2.53096390   -0.75815833   -0.33109212
                                                          H       1.31342387   -0.25284114   -1.40562236""",
                                        'method': 'gcn', 'method_direction': 'F', 'method_index': None,
                                        'success': True, 'successful_irc': None, 'successful_normal_mode': None,
                                        't0': '2021-08-22T07:40:56.929027'},
                                       {'conformer_index': 5, 'energy': 179.82936736522242,
                                        'execution_time': '0:00:03.044867', 'imaginary_freqs': None, 'index': 5,
                                        'initial_xyz': """C      -0.79042125    0.52490389   -0.05857688
                                                          C       0.10447983   -0.01884642    1.04805517
                                                          C       0.97251391   -1.07929420    0.99123168
                                                          H      -1.48476017    1.27402782   -0.17248495
                                                          H      -0.35744023    0.41111273   -0.96343821
                                                          H      -1.22865522    0.02419715    0.87522262
                                                          H      -0.07661759    0.46040890    2.08498836
                                                          H       1.43652236   -1.30764163    0.05607589
                                                          H       1.61815691   -1.20334327    1.70499873
                                                          H       0.39542824   -1.93504274    0.76601362""",
                                        'method': 'gcn', 'method_direction': 'F', 'method_index': None,
                                        'opt_xyz': """C      -0.79042125    0.52490389   -0.05857688
                                                      C       0.10447983   -0.01884642    1.04805517
                                                      C       0.97251391   -1.07929420    0.99123168
                                                      H      -1.48476017    1.27402782   -0.17248495
                                                      H      -0.35744023    0.41111273   -0.96343821
                                                      H      -1.22865522    0.02419715    0.87522262
                                                      H      -0.07661759    0.46040890    2.08498836
                                                      H       1.43652236   -1.30764163    0.05607589
                                                      H       1.61815691   -1.20334327    1.70499873
                                                      H       0.39542824   -1.93504274    0.76601362""",
                                        'success': True, 'successful_irc': None, 'successful_normal_mode': None,
                                        't0': '2021-08-22T07:41:06.893439'},
                                       {'conformer_index': 6, 'energy': 179.8294986402616,
                                        'execution_time': '0:00:02.757137', 'imaginary_freqs': None, 'index': 6,
                                        'initial_xyz': """C      -0.79042125    0.52490389   -0.05857688
                                                          C       0.10447983   -0.01884642    1.04805517
                                                          C       0.97251391   -1.07929420    0.99123168
                                                          H      -1.48476017    1.27402782   -0.17248495
                                                          H      -0.35744023    0.41111273   -0.96343821
                                                          H      -1.22865522    0.02419715    0.87522262
                                                          H      -0.07661759    0.46040890    2.08498836
                                                          H       1.43652236   -1.30764163    0.05607589
                                                          H       1.61815691   -1.20334327    1.70499873
                                                          H       0.39542824   -1.93504274    0.76601362""",
                                        'method': 'gcn', 'method_direction': 'F', 'method_index': None,
                                        'opt_xyz': """C      -0.79042125    0.52490389   -0.05857688
                                                      C       0.10447983   -0.01884642    1.04805517
                                                      C       0.97251391   -1.07929420    0.99123168
                                                      H      -1.48476017    1.27402782   -0.17248495
                                                      H      -0.35744023    0.41111273   -0.96343821
                                                      H      -1.22865522    0.02419715    0.87522262
                                                      H      -0.07661759    0.46040890    2.08498836
                                                      H       1.43652236   -1.30764163    0.05607589
                                                      H       1.61815691   -1.20334327    1.70499873
                                                      H       0.39542824   -1.93504274    0.76601362""",
                                        'success': True, 'successful_irc': None, 'successful_normal_mode': None,
                                        't0': '2021-08-22T07:41:13.388827'},
                                       {'conformer_index': 7, 'energy': 179.8292282137554,
                                        'execution_time': '0:00:02.840379', 'imaginary_freqs': None, 'index': 7,
                                        'initial_xyz': """C      -0.79042125    0.52490389   -0.05857688
                                                          C       0.10447983   -0.01884642    1.04805517
                                                          C       0.97251391   -1.07929420    0.99123168
                                                          H      -1.48476017    1.27402782   -0.17248495
                                                          H      -0.35744023    0.41111273   -0.96343821
                                                          H      -1.22865522    0.02419715    0.87522262
                                                          H      -0.07661759    0.46040890    2.08498836
                                                          H       1.43652236   -1.30764163    0.05607589
                                                          H       1.61815691   -1.20334327    1.70499873
                                                          H       0.39542824   -1.93504274    0.76601362""",
                                        'method': 'gcn', 'method_direction': 'F', 'method_index': None,
                                        'opt_xyz': """C      -0.79042125    0.52490389   -0.05857688
                                                      C       0.10447983   -0.01884642    1.04805517
                                                      C       0.97251391   -1.07929420    0.99123168
                                                      H      -1.48476017    1.27402782   -0.17248495
                                                      H      -0.35744023    0.41111273   -0.96343821
                                                      H      -1.22865522    0.02419715    0.87522262
                                                      H      -0.07661759    0.46040890    2.08498836
                                                      H       1.43652236   -1.30764163    0.05607589
                                                      H       1.61815691   -1.20334327    1.70499873
                                                      H       0.39542824   -1.93504274    0.76601362""",
                                        'success': True, 'successful_irc': None, 'successful_normal_mode': None,
                                        't0': '2021-08-22T07:41:19.049270'},
                                       {'conformer_index': 8, 'energy': 179.82915207423503,
                                        'execution_time': '0:00:03.363040', 'imaginary_freqs': None, 'index': 8,
                                        'initial_xyz': """C      -0.79042125    0.52490389   -0.05857688
                                                          C       0.10447983   -0.01884642    1.04805517
                                                          C       0.97251391   -1.07929420    0.99123168
                                                          H      -1.48476017    1.27402782   -0.17248495
                                                          H      -0.35744023    0.41111273   -0.96343821
                                                          H      -1.22865522    0.02419715    0.87522262
                                                          H      -0.07661759    0.46040890    2.08498836
                                                          H       1.43652236   -1.30764163    0.05607589
                                                          H       1.61815691   -1.20334327    1.70499873
                                                          H       0.39542824   -1.93504274    0.76601362""",
                                        'method': 'gcn', 'method_direction': 'F', 'method_index': None,
                                        'opt_xyz': """C      -0.79042125    0.52490389   -0.05857688
                                                      C       0.10447983   -0.01884642    1.04805517
                                                      C       0.97251391   -1.07929420    0.99123168
                                                      H      -1.48476017    1.27402782   -0.17248495
                                                      H      -0.35744023    0.41111273   -0.96343821
                                                      H      -1.22865522    0.02419715    0.87522262
                                                      H      -0.07661759    0.46040890    2.08498836
                                                      H       1.43652236   -1.30764163    0.05607589
                                                      H       1.61815691   -1.20334327    1.70499873
                                                      H       0.39542824   -1.93504274    0.76601362""",
                                        'success': True, 'successful_irc': None, 'successful_normal_mode': None,
                                        't0': '2021-08-22T07:41:24.792257'},
                                       {'conformer_index': 9, 'energy': 179.82950651680585,
                                        'execution_time': '0:00:03.128026', 'imaginary_freqs': None, 'index': 9,
                                        'initial_xyz': """C      -0.79042125    0.52490389   -0.05857688
                                                          C       0.10447983   -0.01884642    1.04805517
                                                          C       0.97251391   -1.07929420    0.99123168
                                                          H      -1.48476017    1.27402782   -0.17248495
                                                          H      -0.35744023    0.41111273   -0.96343821
                                                          H      -1.22865522    0.02419715    0.87522262
                                                          H      -0.07661759    0.46040890    2.08498836
                                                          H       1.43652236   -1.30764163    0.05607589
                                                          H       1.61815691   -1.20334327    1.70499873
                                                          H       0.39542824   -1.93504274    0.76601362""",
                                        'method': 'gcn', 'method_direction': 'F', 'method_index': None,
                                        'opt_xyz': """C      -0.79042125    0.52490389   -0.05857688
                                                      C       0.10447983   -0.01884642    1.04805517
                                                      C       0.97251391   -1.07929420    0.99123168
                                                      H      -1.48476017    1.27402782   -0.17248495
                                                      H      -0.35744023    0.41111273   -0.96343821
                                                      H      -1.22865522    0.02419715    0.87522262
                                                      H      -0.07661759    0.46040890    2.08498836
                                                      H       1.43652236   -1.30764163    0.05607589
                                                      H       1.61815691   -1.20334327    1.70499873
                                                      H       0.39542824   -1.93504274    0.76601362""",
                                        'success': True, 'successful_irc': None, 'successful_normal_mode': None,
                                        't0': '2021-08-22T07:41:31.062516'},
                                       {'conformer_index': 10, 'energy': 179.82957740523852,
                                        'execution_time': '0:00:02.815705', 'imaginary_freqs': None, 'index': 10,
                                        'initial_xyz': """C      -0.79042125    0.52490389   -0.05857688
                                                          C       0.10447983   -0.01884642    1.04805517
                                                          C       0.97251391   -1.07929420    0.99123168
                                                          H      -1.48476017    1.27402782   -0.17248495
                                                          H      -0.35744023    0.41111273   -0.96343821
                                                          H      -1.22865522    0.02419715    0.87522262
                                                          H      -0.07661759    0.46040890    2.08498836
                                                          H       1.43652236   -1.30764163    0.05607589
                                                          H       1.61815691   -1.20334327    1.70499873
                                                          H       0.39542824   -1.93504274    0.76601362""",
                                        'method': 'gcn', 'method_direction': 'F', 'method_index': None,
                                        'opt_xyz': """C      -0.79042125    0.52490389   -0.05857688
                                                      C       0.10447983   -0.01884642    1.04805517
                                                      C       0.97251391   -1.07929420    0.99123168
                                                      H      -1.48476017    1.27402782   -0.17248495
                                                      H      -0.35744023    0.41111273   -0.96343821
                                                      H      -1.22865522    0.02419715    0.87522262
                                                      H      -0.07661759    0.46040890    2.08498836
                                                      H       1.43652236   -1.30764163    0.05607589
                                                      H       1.61815691   -1.20334327    1.70499873
                                                      H       0.39542824   -1.93504274    0.76601362""",
                                        'success': True, 'successful_irc': None, 'successful_normal_mode': None,
                                        't0': '2021-08-22T07:41:37.126828'},
                                       {'conformer_index': 11, 'energy': 179.8292675963021,
                                        'execution_time': '0:00:03.054725', 'imaginary_freqs': None, 'index': 11,
                                        'initial_xyz': """C      -0.79042125    0.52490389   -0.05857688
                                                          C       0.10447983   -0.01884642    1.04805517
                                                          C       0.97251391   -1.07929420    0.99123168
                                                          H      -1.48476017    1.27402782   -0.17248495
                                                          H      -0.35744023    0.41111273   -0.96343821
                                                          H      -1.22865522    0.02419715    0.87522262
                                                          H      -0.07661759    0.46040890    2.08498836
                                                          H       1.43652236   -1.30764163    0.05607589
                                                          H       1.61815691   -1.20334327    1.70499873
                                                          H       0.39542824   -1.93504274    0.76601362""",
                                        'method': 'gcn', 'method_direction': 'F', 'method_index': None,
                                        'opt_xyz': """C      -0.79042125    0.52490389   -0.05857688
                                                      C       0.10447983   -0.01884642    1.04805517
                                                      C       0.97251391   -1.07929420    0.99123168
                                                      H      -1.48476017    1.27402782   -0.17248495
                                                      H      -0.35744023    0.41111273   -0.96343821
                                                      H      -1.22865522    0.02419715    0.87522262
                                                      H      -0.07661759    0.46040890    2.08498836
                                                      H       1.43652236   -1.30764163    0.05607589
                                                      H       1.61815691   -1.20334327    1.70499873
                                                      H       0.39542824   -1.93504274    0.76601362""",
                                        'success': True, 'successful_irc': None, 'successful_normal_mode': None,
                                        't0': '2021-08-22T07:41:42.887084'},
                                       {'conformer_index': 12, 'energy': 179.82911269180477,
                                        'execution_time': '0:00:03.280951', 'imaginary_freqs': [-1927.8631], 'index': 12,
                                        'initial_xyz': """C      -0.79042125    0.52490389   -0.05857688
                                                          C       0.10447983   -0.01884642    1.04805517
                                                          C       0.97251391   -1.07929420    0.99123168
                                                          H      -1.48476017    1.27402782   -0.17248495
                                                          H      -0.35744023    0.41111273   -0.96343821
                                                          H      -1.22865522    0.02419715    0.87522262
                                                          H      -0.07661759    0.46040890    2.08498836
                                                          H       1.43652236   -1.30764163    0.05607589
                                                          H       1.61815691   -1.20334327    1.70499873
                                                          H       0.39542824   -1.93504274    0.76601362""",
                                        'method': 'gcn', 'method_direction': 'F', 'method_index': None,
                                        'opt_xyz': """C      -0.79042125    0.52490389   -0.05857688
                                                      C       0.10447983   -0.01884642    1.04805517
                                                      C       0.97251391   -1.07929420    0.99123168
                                                      H      -1.48476017    1.27402782   -0.17248495
                                                      H      -0.35744023    0.41111273   -0.96343821
                                                      H      -1.22865522    0.02419715    0.87522262
                                                      H      -0.07661759    0.46040890    2.08498836
                                                      H       1.43652236   -1.30764163    0.05607589
                                                      H       1.61815691   -1.20334327    1.70499873
                                                      H       0.39542824   -1.93504274    0.76601362""",
                                        'success': True, 'successful_irc': None, 'successful_normal_mode': None,
                                        't0': '2021-08-22T07:41:48.895399'},
                                       {'conformer_index': 13, 'energy': 179.82924134132918,
                                        'execution_time': '0:00:02.906303', 'imaginary_freqs': None, 'index': 13,
                                        'initial_xyz': """C      -0.79042125    0.52490389   -0.05857688
                                                          C       0.10447983   -0.01884642    1.04805517
                                                          C       0.97251391   -1.07929420    0.99123168
                                                          H      -1.48476017    1.27402782   -0.17248495
                                                          H      -0.35744023    0.41111273   -0.96343821
                                                          H      -1.22865522    0.02419715    0.87522262
                                                          H      -0.07661759    0.46040890    2.08498836
                                                          H       1.43652236   -1.30764163    0.05607589
                                                          H       1.61815691   -1.20334327    1.70499873
                                                          H       0.39542824   -1.93504274    0.76601362""",
                                        'method': 'gcn', 'method_direction': 'F', 'method_index': None,
                                        'opt_xyz': """C      -0.79042125    0.52490389   -0.05857688
                                                      C       0.10447983   -0.01884642    1.04805517
                                                      C       0.97251391   -1.07929420    0.99123168
                                                      H      -1.48476017    1.27402782   -0.17248495
                                                      H      -0.35744023    0.41111273   -0.96343821
                                                      H      -1.22865522    0.02419715    0.87522262
                                                      H      -0.07661759    0.46040890    2.08498836
                                                      H       1.43652236   -1.30764163    0.05607589
                                                      H       1.61815691   -1.20334327    1.70499873
                                                      H       0.39542824   -1.93504274    0.76601362""",
                                        'success': True, 'successful_irc': None, 'successful_normal_mode': None,
                                        't0': '2021-08-22T07:41:55.030683'},
                                       {'conformer_index': 14, 'energy': 16.056782611296512,
                                        'execution_time': '0:00:00.009075', 'imaginary_freqs': [-76.0137], 'index': 14,
                                        'initial_xyz': """C      -0.79042125    0.52490389   -0.05857688
                                                          C       0.10447983   -0.01884642    1.04805517
                                                          C       0.97251391   -1.07929420    0.99123168
                                                          H      -1.48476017    1.27402782   -0.17248495
                                                          H      -0.35744023    0.41111273   -0.96343821
                                                          H      -1.22865522    0.02419715    0.87522262
                                                          H      -0.07661759    0.46040890    2.08498836
                                                          H       1.43652236   -1.30764163    0.05607589
                                                          H       1.61815691   -1.20334327    1.70499873
                                                          H       0.39542824   -1.93504274    0.76601362""",
                                        'method': 'kinbot', 'method_direction': 'F', 'method_index': 0,
                                        'opt_xyz': """C      -0.79042125    0.52490389   -0.05857688
                                                      C       0.10447983   -0.01884642    1.04805517
                                                      C       0.97251391   -1.07929420    0.99123168
                                                      H      -1.48476017    1.27402782   -0.17248495
                                                      H      -0.35744023    0.41111273   -0.96343821
                                                      H      -1.22865522    0.02419715    0.87522262
                                                      H      -0.07661759    0.46040890    2.08498836
                                                      H       1.43652236   -1.30764163    0.05607589
                                                      H       1.61815691   -1.20334327    1.70499873
                                                      H       0.39542824   -1.93504274    0.76601362""",
                                        'success': True, 'successful_irc': None, 'successful_normal_mode': None,
                                        't0': '2021-08-22T07:42:00.994920'},
                                       {'conformer_index': 15, 'energy': 0.0,
                                        'execution_time': '0:00:00.008971', 'imaginary_freqs': [-117.6316], 'index': 15,
                                        'initial_xyz': """C      -0.79042125    0.52490389   -0.05857688
                                                          C       0.10447983   -0.01884642    1.04805517
                                                          C       0.97251391   -1.07929420    0.99123168
                                                          H      -1.48476017    1.27402782   -0.17248495
                                                          H      -0.35744023    0.41111273   -0.96343821
                                                          H      -1.22865522    0.02419715    0.87522262
                                                          H      -0.07661759    0.46040890    2.08498836
                                                          H       1.43652236   -1.30764163    0.05607589
                                                          H       1.61815691   -1.20334327    1.70499873
                                                          H       0.39542824   -1.93504274    0.76601362""",
                                        'method': 'kinbot', 'method_direction': 'R', 'method_index': 1,
                                        'opt_xyz': """C      -0.79042125    0.52490389   -0.05857688
                                                      C       0.10447983   -0.01884642    1.04805517
                                                      C       0.97251391   -1.07929420    0.99123168
                                                      H      -1.48476017    1.27402782   -0.17248495
                                                      H      -0.35744023    0.41111273   -0.96343821
                                                      H      -1.22865522    0.02419715    0.87522262
                                                      H      -0.07661759    0.46040890    2.08498836
                                                      H       1.43652236   -1.30764163    0.05607589
                                                      H       1.61815691   -1.20334327    1.70499873
                                                      H       0.39542824   -1.93504274    0.76601362""",
                                        'success': True, 'successful_irc': None, 'successful_normal_mode': None,
                                        't0': '2021-08-22T07:42:01.052311'}],
                        'ts_guesses_exhausted': False, 'ts_number': 0,
                        'ts_report': 'TS method summary for TS0 in C3_1 <=> C3_2:\n'
                                     'Methods that successfully generated a TS guess:\n'
                                     'autotst,autotst,autotst,autotst,gcn,gcn,gcn,gcn,gcn,gcn,gcn,gcn,gcn,gcn,kinbot,kinbot,\n'
                                     'The method that generated the best TS guess and its output used '
                                     'for the optimization: gcn\n',
                        'tsg_spawned': True, 'unsuccessful_methods': []}
        spc = ARCSpecies(species_dict=species_dict)
        self.assertTrue(spc.is_ts)

    def test_copy(self):
        """Test the copy() method."""
        spc_copy = self.spc6.copy()
        self.assertIsNot(self.spc6, spc_copy)
        self.assertEqual(len(self.spc6.mol.get_all_edges()), len(spc_copy.mol.get_all_edges()))
        self.assertEqual(spc_copy.multiplicity, self.spc6.multiplicity)
        self.assertEqual(spc_copy.get_xyz()['symbols'], self.spc6.get_xyz()['symbols'])
        self.assertNotEqual(spc_copy.mol.atoms[0].id, self.spc6.mol.atoms[0].id)
        self.assertEqual(spc_copy.mol.to_smiles(), self.spc6.mol.to_smiles())

    def test_mol_dict_repr_round_trip(self):
        """Test that a Molecule object survives the as_dict() and from_dict() round trip with emphasis on atom IDs."""
        mol = Molecule(smiles='NCC')
        mol.assign_atom_ids()
        original_symbols = [atom.element.symbol for atom in mol.atoms]
        original_ids = [atom.id for atom in mol.atoms]
        original_adjlist = mol.to_adjacency_list()
        spc = ARCSpecies(label='EA', mol=mol)
        species_dict = spc.as_dict()
        new_spc = ARCSpecies(species_dict=species_dict)
        new_symbols = [atom.element.symbol for atom in new_spc.mol.atoms]
        new_ids = [atom.id for atom in new_spc.mol.atoms]
        new_adjlist = new_spc.mol.to_adjacency_list()
        self.assertEqual(original_symbols, new_symbols)
        self.assertEqual(original_ids, new_ids)
        self.assertEqual(original_adjlist, new_adjlist)

    def test_determine_rotor_type(self):
        """Test that we correctly determine whether a rotor is FreeRotor or HinderedRotor"""
        free_path = os.path.join(ARC_PATH, 'arc', 'testing', 'rotor_scans', 'CH3C(O)O_FreeRotor.out')
        hindered_path = os.path.join(ARC_PATH, 'arc', 'testing', 'rotor_scans', 'H2O2.out')
        self.assertEqual(determine_rotor_type(free_path), 'FreeRotor')
        self.assertEqual(determine_rotor_type(hindered_path), 'HinderedRotor')

    def test_rotor_symmetry(self):
        """Test that ARC automatically determines a correct rotor symmetry"""
        path1 = os.path.join(ARC_PATH, 'arc', 'testing', 'rotor_scans', 'OOC1CCOCO1.out')  # symmetry = 1; min at -10 o
        path2 = os.path.join(ARC_PATH, 'arc', 'testing', 'rotor_scans', 'H2O2.out')  # symmetry = 1
        path3 = os.path.join(ARC_PATH, 'arc', 'testing', 'rotor_scans', 'N2O3.out')  # symmetry = 2
        path4 = os.path.join(ARC_PATH, 'arc', 'testing', 'rotor_scans', 'sBuOH.out')  # symmetry = 3
        path5 = os.path.join(ARC_PATH, 'arc', 'testing', 'rotor_scans', 'CH3C(O)O_FreeRotor.out')  # symmetry = 6

        sym1, e1, _ = determine_rotor_symmetry(label='label', pivots=[3, 4], rotor_path=path1)
        sym2, e2, _ = determine_rotor_symmetry(label='label', pivots=[3, 4], rotor_path=path2)
        sym3, e3, n3 = determine_rotor_symmetry(label='label', pivots=[3, 4], rotor_path=path3, return_num_wells=True)
        sym4, e4, _ = determine_rotor_symmetry(label='label', pivots=[3, 4], rotor_path=path4)
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
        
        
        
        O   3.1024  0.1216  1.0455
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
   

   C,    1.9403,   -2.2644,    0.1662
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
H,-0.7777,2.3260,0.1202
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

        xyz_path = os.path.join(ARC_PATH, 'arc', 'testing', 'xyz', 'CH3C(O)O.xyz')
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

        conformers_path = os.path.join(ARC_PATH, 'arc', 'testing', 'xyz', 'conformers_file.txt')
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
        project_directory = os.path.join(ARC_PATH, 'Projects', 'arc_project_for_testing_delete_after_usage4')
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

        # test w/o energies
        save_conformers_file(project_directory=project_directory, label='vinoxy', xyzs=xyzs,
                             level_of_theory=Level(repr='level1'), multiplicity=2, charge=0)
        self.assertTrue(os.path.isfile(os.path.join(project_directory, 'output', 'Species', 'vinoxy', 'geometry',
                                                    'conformers', 'conformers_before_optimization.txt')))

        # test with energies
        save_conformers_file(project_directory=project_directory, label='vinoxy', xyzs=xyzs,
                             level_of_theory=Level(repr='level1'), multiplicity=2, charge=0, energies=energies)
        self.assertTrue(os.path.isfile(os.path.join(project_directory, 'output', 'Species', 'vinoxy', 'geometry',
                                                    'conformers', 'conformers_after_optimization.txt')))

        spc2 = ARCSpecies(label='vinoxy', smiles='C=C[O]', xyz=os.path.join(
            project_directory, 'output', 'Species', 'vinoxy', 'geometry', 'conformers',
            'conformers_before_optimization.txt'))

        spc3 = ARCSpecies(label='vinoxy', smiles='C=C[O]', xyz=os.path.join(
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

        spc11 = ARCSpecies(label='C--H', xyz='C 0 0 0\nH 1 2 5')  # test using mol_s instead of mol_b
        self.assertEqual(spc11.number_of_atoms, 2)

        spc12 = ARCSpecies(label='C--H-TS', xyz='C 0 0 0\nH 1 2 5', is_ts=True)
        self.assertEqual(spc12.number_of_atoms, 2)

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
        lj_path = os.path.join(ARC_PATH, 'arc', 'testing', 'NH3_oneDMin.dat')
        opt_path = os.path.join(ARC_PATH, 'arc', 'testing', 'composite', 'SO2OO_CBS-QB3.log')
        bath_gas = 'N2'
        opt_level = Level(repr='CBS-QB3')
        freq_path = os.path.join(ARC_PATH, 'arc', 'testing', 'composite', 'SO2OO_CBS-QB3.log')
        freq_level = Level(repr='CBS-QB3')
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
                                                           'the bath gas; Dipole moment was calculated at the cbs-qb3 '
                                                           'level of theory; Polarizability was calculated at the '
                                                           'cbs-qb3 level of theory; Rotational Relaxation Collision '
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
        for i, atom in enumerate(spc2.mol.atoms):
            self.assertEqual(atom.symbol, spc2.get_xyz()['symbols'][i])

        n3_xyz = """N      -1.1997440839    -0.1610052059     0.0274738287
        H      -1.4016624407    -0.6229695533    -0.8487034080
        H      -0.0000018759     1.2861082773     0.5926077870
        N       0.0000008520     0.5651072858    -0.1124621525
        H      -1.1294692206    -0.8709078271     0.7537518889
        N       1.1997613019    -0.1609980472     0.0274604887
        H       1.1294795781    -0.8708998550     0.7537444446
        H       1.4015274689    -0.6230592706    -0.8487058662"""
        spc3 = ARCSpecies(label='N3', xyz=n3_xyz, multiplicity=1, smiles='NNN')
        for i, atom in enumerate(spc3.mol.atoms):
            self.assertEqual(atom.symbol, spc3.get_xyz()['symbols'][i])
        spc3.generate_conformers()
        self.assertEqual(len(spc3.conformers), 9)

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
        for i, atom in enumerate(spc4.mol.atoms):
            self.assertEqual(atom.symbol, spc4.get_xyz()['symbols'][i])

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

        xyz6 = """C                  0.50180491   -0.93942231   -0.57086745
        C                  0.01278145    0.13148427    0.42191407
        H                  0.28549447    0.06799101    1.45462711
        H                  1.44553946   -1.32386345   -0.24456986
        H                  0.61096295   -0.50262210   -1.54153222
        H                 -0.24653265    2.11136864   -0.37045418
        C                 -0.86874485    1.29377369   -0.07163907
        H                 -0.21131163   -1.73585284   -0.61629002
        H                 -1.51770930    1.60958621    0.71830245
        H                 -1.45448167    0.96793094   -0.90568876"""
        spc6 = ARCSpecies(label='C[CH]C', smiles='C[CH]C', xyz=xyz6)
        for atom, symbol in zip(spc6.mol.atoms, spc6.get_xyz()['symbols']):
            self.assertEqual(atom.symbol, symbol)
        for atom, symbol in zip(spc6.mol.atoms, ['C', 'C', 'H', 'H', 'H', 'H', 'C', 'H', 'H', 'H']):
            self.assertEqual(atom.symbol, symbol)

        xyz7 = """C                  0.42854493    1.37396218   -0.06378771
                  H                  0.92258324    0.49575491   -0.42375735
                  C                  1.14683468    2.48797667    0.21834452
                  H                  0.65279661    3.36618328    0.57831609
                  C                  2.67411846    2.48994130    0.02085939
                  H                  3.00074251    3.47886113   -0.22460814
                  O                  3.01854194    1.59675339   -1.04144368
                  H                  3.97061495    1.59797810   -1.16455130
                  N                  3.32919606    2.05137935    1.26159979
                  H                  3.08834048    2.67598628    2.00446907
                  O                 -0.98964750    1.37213874    0.11958876
                  H                 -1.39314061    0.77169906   -0.51149404
                  O                  4.67796616    2.05311435    1.08719734
                  H                  5.10577193    1.76670654    1.89747679"""
        spc7 = ARCSpecies(label='spc7', smiles='OC=CC(O)NO', xyz=xyz7)
        for atom, symbol in zip(spc7.mol.atoms, spc7.get_xyz()['symbols']):
            self.assertEqual(atom.symbol, symbol)
        for atom, symbol in zip(spc7.mol.atoms, ['C', 'H', 'C', 'H', 'C', 'H', 'O', 'H', 'N', 'H', 'O', 'H', 'O', 'H']):
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

        xyz4 = """N       2.25402700    0.45886100    0.16098200
                  C       0.96879200    0.14668800    0.06929300
                  C       0.01740100    1.20875600   -0.00119800
                  H       0.40464100    2.21851400    0.02457300
                  C      -1.33137900    0.96373600   -0.09758800
                  C      -1.81578100   -0.35215900   -0.12971800
                  C      -0.87660400   -1.35782200   -0.06021800
                  C       0.46214300   -1.20057200    0.03562800
                  H       2.81718400   -0.39044400    0.20315600
                  H      -2.03062400    1.78920300   -0.14948800
                  H      -2.87588800   -0.55598400   -0.20545500
                  H       1.15316700   -2.03432300    0.08695500"""
        spc4 = ARCSpecies(label='anilino_radical_BDE_7_12_A', smiles='N=C1[CH]C=C[C]=C1', xyz=xyz4)
        spc4.final_xyz = xyz4
        is_isomorphic4 = spc4.check_xyz_isomorphism()
        self.assertTrue(is_isomorphic4)

        xyz5 = """C       0.08059628    1.32037195   -0.29800610
                  C      -1.28794158    1.26062641   -0.03029704
                  C      -1.89642539    0.02969442    0.21332426
                  C      -1.13859198   -1.14110023    0.18344613
                  C       0.23092154   -1.08214781   -0.08393383
                  C       0.84282867    0.15119681   -0.31285027
                  O       2.17997981    0.29916802   -0.59736431
                  O       2.90066125   -0.82056323   -0.00921949
                  H       0.55201906    2.27952184   -0.49410221
                  H      -1.87925130    2.17240519   -0.01581738
                  H      -2.96278939   -0.01860646    0.41888241
                  H      -1.61463364   -2.10195688    0.36125669
                  H       0.80478689   -2.00346200   -0.12519327"""
        with self.assertRaises(SpeciesError):
            ARCSpecies(label='c1ccccc1OO', smiles='c1ccccc1OO', xyz=xyz5)

        xyz6 = """C    1.1709385492    0.1763143411    0.0
                  Cl  -0.5031634975   -0.0109430036    0.0
                  H    1.5281481620   -0.8718549847    0.0"""
        spc6 = ARCSpecies(label='[CH]Cl', smiles='[CH]Cl', xyz=xyz6)
        self.assertEqual(spc6.get_xyz(), str_to_xyz(xyz6))

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
        """Test that we can define, process, and manipulate ions"""
        nh4 = ARCSpecies(label='NH4', smiles='[NH4+]', charge=1)
        nh4.determine_multiplicity(smiles='', adjlist='', mol=None)
        self.assertEqual(nh4.multiplicity, 1)

        cation1 = ARCSpecies(label='OCCCOH2', smiles='OCCC[OH2+]', charge=1)
        cation1.determine_multiplicity(smiles='', adjlist='', mol=None)
        self.assertEqual(cation1.multiplicity, 1)
        self.assertEqual(cation1.charge, 1)
        cation1.generate_conformers()
        self.assertTrue(len(cation1.conformers))

        cation2 = ARCSpecies(label='C(C)(C)C[NH2+]CO', smiles='C(C)(C)C[NH2+]CO', charge=1)
        cation2.determine_multiplicity(smiles='', adjlist='', mol=None)
        self.assertEqual(cation2.multiplicity, 1)
        self.assertEqual(cation2.charge, 1)
        cation2.generate_conformers()
        self.assertEqual(len(cation2.conformers), 10)

        anion = ARCSpecies(label='CCC(=O)[O-]', smiles='CCC(=O)[O-]', charge=-1)
        anion.determine_multiplicity(smiles='', adjlist='', mol=None)
        self.assertEqual(anion.multiplicity, 1)
        self.assertEqual(anion.charge, -1)
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

    def test_determine_multiplicity(self):
        """Test determining a species multiplicity"""
        h_rad = ARCSpecies(label='H', smiles='[H]')
        self.assertEqual(h_rad.multiplicity, 2)

        n2 = ARCSpecies(label='N#N', smiles='N#N')
        self.assertEqual(n2.multiplicity, 1)

        methyl_peroxyl = ARCSpecies(label='CH3OO', smiles='CO[O]')
        self.assertEqual(methyl_peroxyl.multiplicity, 2)

        ch_ts = ARCSpecies(label='C--H-TS', xyz='C 0 0 0\nH 1 2 5', is_ts=True)
        self.assertEqual(ch_ts.multiplicity, 2)

    def test_are_coords_compliant_with_graph(self):
        """Test coordinates compliant with 2D graph connectivity"""
        self.assertTrue(are_coords_compliant_with_graph(xyz=self.spc6.get_xyz(), mol=self.spc6.mol))

        xyz_non_split = str_to_xyz(""" O                 -1.77782500    0.16383000   -1.00470900
 O                  2.82634100   -0.49758800   -0.22053900
 O                 -3.00549000    0.30995600   -0.63994200
 O                  3.40372500    0.73816600   -0.00411700
 N                  1.58283100   -0.44410100   -0.22685100
 C                 -0.87680000   -0.04688100    0.15263000
 C                 -0.91300400    1.19079300    1.04397100
 C                 -1.28190800   -1.33133800    0.86726200
 C                  0.45671000   -0.17198800   -0.46072100
 H                 -0.57829200    2.08105700    0.49410300
 H                 -1.94993600    1.34102100    1.37594600
 H                 -0.26515400    1.04703000    1.91910800
 H                 -0.63490800   -1.49884100    1.73900500
 H                 -2.32291100   -1.22406700    1.20286200
 H                 -1.21261900   -2.19592900    0.19252600""")
        xyz_split = str_to_xyz(""" O                  0.07716500   -0.35216600   -0.85343400
 O                  2.62355500    1.10632700    0.00977500
 O                  0.76150400   -1.46024200   -0.66136400
 O                  3.11333100   -0.00414700    0.06703600
 N                 -2.16062200    2.08015200   -0.47197200
 C                 -1.01293500   -0.20025800    0.17009400
 C                 -1.99168200   -1.36471200    0.01039700
 C                 -0.37657800   -0.11697600    1.55811500
 C                 -1.64082800    1.07802900   -0.20550900
 H                 -2.40531700   -1.38599900   -1.00008100
 H                 -1.45900400   -2.29793500    0.20302000
 H                 -2.81142200   -1.26290200    0.72553300
 H                 -1.15306500    0.01609000    2.31512700
 H                  0.16104500   -1.04691100    1.75262900
 H                  0.31981100    0.72190400    1.61289000""")
        adj_list = """multiplicity 2
    1  O u0 p2 c0 {3,S} {6,S}
    2  O u0 p2 c0 {4,S} {5,S}
    3  O u1 p2 c0 {1,S}
    4  O u0 p3 c-1 {2,S}
    5  N u0 p0 c+1 {2,S} {9,T}
    6  C u0 p0 c0 {1,S} {7,S} {8,S} {9,S}
    7  C u0 p0 c0 {6,S} {10,S} {11,S} {12,S}
    8  C u0 p0 c0 {6,S} {13,S} {14,S} {15,S}
    9  C u0 p0 c0 {5,T} {6,S}
    10 H u0 p0 c0 {7,S}
    11 H u0 p0 c0 {7,S}
    12 H u0 p0 c0 {7,S}
    13 H u0 p0 c0 {8,S}
    14 H u0 p0 c0 {8,S}
    15 H u0 p0 c0 {8,S}"""
        mol = Molecule().from_adjacency_list(adj_list)
        self.assertTrue(are_coords_compliant_with_graph(xyz=xyz_non_split, mol=mol))
        self.assertFalse(are_coords_compliant_with_graph(xyz=xyz_split, mol=mol))

    def test_check_label(self):
        """Test the species check_label() method"""
        label, original_label = check_label('HCN')
        self.assertEqual(label, 'HCN')
        self.assertIsNone(original_label)

        label, original_label = check_label('H-N')
        self.assertEqual(label, 'H-N')
        self.assertIsNone(original_label)

        label, original_label = check_label('C#N')
        self.assertEqual(label, 'CtN')
        self.assertEqual(original_label, 'C#N')

        label, original_label = check_label('C+N')
        self.assertEqual(label, 'CpN')
        self.assertEqual(original_label, 'C+N')

        label, original_label = check_label('C?N')
        self.assertEqual(label, 'C_N')
        self.assertEqual(original_label, 'C?N')

    def test_check_atom_balance(self):
        """Test the check_atom_balance function"""
        entry_mol = Molecule(smiles='C')
        entry_str = """C  0.0000000  0.0000000  0.0000000
                     H  0.6325850  0.6325850  0.6325850
                     H -0.6325850 -0.6325850  0.6325850
                     H -0.6325850  0.6325850 -0.6325850
                     H  0.6325850 -0.6325850 -0.6325850"""
        entry_dict = {'symbols': ('C', 'H', 'H', 'H', 'H'),
                      'isotopes': (12, 1, 1, 1, 1),
                      'coords': ((0.0, 0.0, 0.0),
                                 (0.6300326, 0.6300326, 0.6300326),
                                 (-0.6300326, -0.6300326, 0.6300326),
                                 (-0.6300326, 0.6300326, -0.6300326),
                                 (0.6300326, -0.6300326, -0.6300326))}
        entry_wrong = Molecule(smiles='N')

        self.assertTrue(check_atom_balance(entry_mol, entry_str))
        self.assertTrue(check_atom_balance(entry_str, entry_dict))
        self.assertTrue(check_atom_balance(entry_dict, entry_dict))
        self.assertFalse(check_atom_balance(entry_wrong, entry_dict))
        self.assertFalse(check_atom_balance(entry_mol, entry_wrong))

    def test_ts_mol_attribute(self):
        """Test that a TS species has a .mol attribute generated from xyz"""
        ts_xyz = """O      -0.63023600    0.92494700    0.43958200
C       0.14513500   -0.07880000   -0.04196400
C      -0.97050300   -1.02992900   -1.65916600
N      -0.75664700   -2.16458700   -1.81286400
H      -1.25079800    0.57954500    1.08412300
H       0.98208300    0.28882200   -0.62114100
H       0.30969500   -0.94370100    0.59100600
H      -1.47626400   -0.10694600   -1.88883800"""
        ts_spc = ARCSpecies(label='TS', is_ts=True, xyz=ts_xyz)
        ts_spc.mol_from_xyz()
        self.assertEqual(ts_spc.mol.to_smiles(), 'C#N.[CH2]O')
        self.assertEqual(len(ts_spc.mol.atoms), 8)

    def test_colliding_atoms(self):
        """Check that we correctly determine when atoms collide in xyz"""
        xyz_no_0 = """C	0.0000000	0.0000000	0.6505570"""  # Monoatomic
        xyz_no_1 = """C      -0.84339557   -0.03079260   -0.13110478
                      N       0.53015060    0.44534713   -0.25006000
                      O       1.33245258   -0.55134720    0.44204567
                      H      -1.12632103   -0.17824612    0.91628291
                      H      -1.52529493    0.70480833   -0.56787044
                      H      -0.97406455   -0.97317212   -0.67214713
                      H       0.64789210    1.26863944    0.34677470
                      H       1.98414750   -0.79355889   -0.24492049"""  # No colliding atoms.
        xyz_no_2 = """C      0.0 0.0 0.0
                      H       0.0 0.0 1.09"""  # No colliding atoms.
        xyz_no_3 = """N      -0.29070308    0.26322835    0.48770927
                      N       0.29070351   -0.26323281   -0.48771096
                      N      -2.61741263    1.38275080    2.63428181
                      N       2.61742270   -1.38276006   -2.63427425
                      C      -1.77086206    0.18100754    0.43957605
                      C       1.77086254   -0.18101028   -0.43957552
                      C      -2.22486176   -1.28143567    0.45202312
                      C      -2.30707039    0.92407663   -0.78734681
                      C       2.30707074   -0.92407071    0.78735246
                      C       2.22485929    1.28143406   -0.45203080
                      C      -2.23868798    0.85547218    1.67084736
                      C       2.23869247   -0.85548109   -1.67084185
                      H      -1.90398693   -1.81060764   -0.45229645
                      H      -3.31681639   -1.35858536    0.51240600
                      H      -1.80714051   -1.81980551    1.31137107
                      H      -3.40300863    0.95379538   -0.78701415
                      H      -1.98806037    0.44494681   -1.71978670
                      H      -1.94802915    1.96005927   -0.81269573
                      H       1.98805486   -0.44493850    1.71978893
                      H       1.94803425   -1.96005464    0.81270509
                      H       3.40300902   -0.95378386    0.78702431
                      H       1.90398036    1.81061002    0.45228426
                      H       3.31681405    1.35858667   -0.51241516
                      H       1.80713611    1.81979843   -1.31138136"""  # Check that N=N and C#N do not collide.
        xyz_no_4 = """C      -1.33285177    0.30272690   -2.83811851
                      C      -1.33285177    1.50627688   -3.54494783
                      C      -1.32845119    2.71920095   -2.85862837
                      C      -1.32234713    2.73004576   -1.46505859
                      C      -1.32231954    1.52762830   -0.75637284
                      C      -1.33285177    0.30272690   -1.43685068
                      C      -1.31312206   -0.99764423   -0.67121404
                      C       0.11690866   -1.47784322   -0.40640250
                      C       0.12987915   -2.78199406    0.34160671
                      C       0.66174761   -2.94316897    1.55972735
                      H      -1.33264187   -0.63592930   -3.38748277
                      H      -1.33515167    1.49722919   -4.63168080
                      H      -1.32825370    3.65579521   -3.40966787
                      H      -1.31648537    3.67551631   -0.92922749
                      H      -1.31376930    1.55151899    0.33101393
                      H      -1.87364153   -1.75646869   -1.23312193
                      H      -1.85321213   -0.86924064    0.27615721
                      H       0.78333694   -0.55673605    0.25844147
                      H       0.64816845   -1.61976732   -1.35587312
                      H      -0.31460521   -3.64071363   -0.15781584
                      H       1.12279222   -2.11920392    2.09567728
                      H       0.64740116   -3.91271098    2.04834754
                      C       2.18921590   -0.43919208    4.34098460
                      C       1.87571447    0.35085963    3.08717896
                      O       1.66949109   -0.56359341    2.01183519
                      O       1.37483207    0.26080190    0.84853043
                      H       3.08302153   -1.05485636    4.19553888
                      H       1.36799933   -1.12234572    4.58180000
                      H       2.35454842    0.22787988    5.19160950
                      H       2.70974870    1.01655203    2.84107717
                      H       0.96938754    0.94803086    3.23320843"""  # Check a valid TS.

        self.assertFalse(colliding_atoms(str_to_xyz(xyz_no_0)))
        self.assertFalse(colliding_atoms(str_to_xyz(xyz_no_1)))
        self.assertFalse(colliding_atoms(str_to_xyz(xyz_no_2)))
        self.assertFalse(colliding_atoms(str_to_xyz(xyz_no_3)))

        xyz_0 = """C      0.0 0.0 0.0
                   H       0.0 0.0 0.5"""  # colliding atoms
        xyz_1 = """C      -0.84339557   -0.03079260   -0.13110478
                   N       0.53015060    0.44534713   -0.25006000
                   O       1.33245258   -0.55134720    0.44204567
                   H      -1.12632103   -0.17824612    0.91628291
                   H      -1.52529493    0.70480833   -0.56787044
                   H      -0.97406455   -0.97317212   -0.67214713
                   H       1.33245258   -0.55134720    0.48204567
                   H       1.98414750   -0.79355889   -0.24492049"""  # colliding atoms
        xyz_2 = """N                 -0.29070308    0.26322835    0.48770927
                   N                  0.29070351   -0.26323281   -0.48771096
                   N                 -2.48318439    1.19587180    2.29281971
                   N                  2.61742270   -1.38276006   -2.63427425
                   C                 -1.77086206    0.18100754    0.43957605
                   C                  1.77086254   -0.18101028   -0.43957552
                   C                 -2.22486176   -1.28143567    0.45202312
                   C                 -2.30707039    0.92407663   -0.78734681
                   C                  2.30707074   -0.92407071    0.78735246
                   C                  2.22485929    1.28143406   -0.45203080
                   C                 -2.23868798    0.85547218    1.67084736
                   C                  2.23869247   -0.85548109   -1.67084185
                   H                 -1.90398693   -1.81060764   -0.45229645
                   H                 -3.31681639   -1.35858536    0.51240600
                   H                 -1.80714051   -1.81980551    1.31137107
                   H                 -3.40300863    0.95379538   -0.78701415
                   H                 -1.98806037    0.44494681   -1.71978670
                   H                 -1.94802915    1.96005927   -0.81269573
                   H                  1.98805486   -0.44493850    1.71978893
                   H                  1.94803425   -1.96005464    0.81270509
                   H                  3.40300902   -0.95378386    0.78702431
                   H                  1.90398036    1.81061002    0.45228426
                   H                  3.31681405    1.35858667   -0.51241516
                   H                  1.80713611    1.81979843   -1.31138136"""  # check that C-N collide
        xyz_3 = """N                 -0.29070308    0.26322835    0.48770927
                   N                  0.29070351   -0.26323281   -0.48771096
                   N                 -2.61741263    1.38275080    2.63428181
                   N                  2.61742270   -1.38276006   -2.63427425
                   C                 -1.77086206    0.18100754    0.43957605
                   C                  1.77086254   -0.18101028   -0.43957552
                   C                 -2.22486176   -1.28143567    0.45202312
                   C                 -2.30707039    0.92407663   -0.78734681
                   C                  2.30707074   -0.92407071    0.78735246
                   C                  2.22485929    1.28143406   -0.45203080
                   C                 -2.23868798    0.85547218    1.67084736
                   C                  2.23869247   -0.85548109   -1.67084185
                   H                 -1.90398693   -1.81060764   -0.45229645
                   H                 -2.77266137   -1.32013927    0.48231533
                   H                 -1.80714051   -1.81980551    1.31137107
                   H                 -3.40300863    0.95379538   -0.78701415
                   H                 -1.98806037    0.44494681   -1.71978670
                   H                 -1.94802915    1.96005927   -0.81269573
                   H                  1.98805486   -0.44493850    1.71978893
                   H                  1.94803425   -1.96005464    0.81270509
                   H                  3.40300902   -0.95378386    0.78702431
                   H                  1.90398036    1.81061002    0.45228426
                   H                  3.31681405    1.35858667   -0.51241516
                   H                  1.80713611    1.81979843   -1.31138136"""  # check that C-H collide
        xyz_4 = """C                 -0.50291748    0.30789571   -1.58840193
                   C                 -0.50291748    1.51144570   -2.29523125
                   C                 -0.49851690    2.72436977   -1.60891178
                   C                 -0.49241284    2.73521458   -0.21534200
                   C                 -0.49238525    1.53279712    0.49334375
                   C                 -0.50291748    0.30789571   -0.18713410
                   C                 -0.48318777   -0.99247541    0.57850254
                   C                  0.94684295   -1.47267440    0.84331408
                   C                  0.95981345   -2.77682525    1.59132329
                   C                  1.49168190   -2.93800015    2.80944393
                   H                 -0.50270758   -0.63076048   -2.13776618
                   H                 -0.50521738    1.50239800   -3.38196422
                   H                 -0.49831941    3.66096402   -2.15995128
                   H                 -0.48655108    3.68068513    0.32048909
                   H                 -0.48383501    1.55668781    1.58073051
                   H                 -1.04370724   -1.75129988    0.01659465
                   H                 -1.02327784   -0.86407182    1.52587380
                   H                  1.61327123   -0.55156723    1.50815806
                   H                  1.47810274   -1.61459850   -0.10615653
                   H                  0.51532908   -3.63554481    1.09190074
                   H                  1.95272651   -2.11403510    3.34539387
                   H                  1.47733545   -3.90754216    3.29806413
                   C                 -1.77154392    0.32434223   -0.71017644
                   C                 -0.51618806   -0.49816092   -0.50508209
                   O                  0.44688790    0.30258121    0.17786256
                   O                  1.61812416   -0.54706350    0.33922867
                   H                 -1.55163746    1.22502635   -1.29283820
                   H                 -2.17532536    0.65812254    0.25140501
                   H                 -2.53737957   -0.25582137   -1.23246804
                   H                 -0.10675560   -0.81091180   -1.47145547
                   H                 -0.73982217   -1.38616269    0.09559021"""

        self.assertTrue(colliding_atoms(str_to_xyz(xyz_0)))
        self.assertTrue(colliding_atoms(str_to_xyz(xyz_1)))
        self.assertTrue(colliding_atoms(str_to_xyz(xyz_2)))
        self.assertTrue(colliding_atoms(str_to_xyz(xyz_3)))
        self.assertTrue(colliding_atoms(str_to_xyz(xyz_4)))

    @classmethod
    def tearDownClass(cls):
        """
        A function that is run ONCE after all unit tests in this class.
        Delete all project directories created during these unit tests
        """
        projects = ['arc_project_for_testing_delete_after_usage4',
                    os.path.join(ARC_PATH, 'arc', 'testing', 'gcn_tst')]
        for project in projects:
            project_directory = os.path.join(ARC_PATH, 'Projects', project)
            shutil.rmtree(project_directory, ignore_errors=True)


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
        spc1.label = 'CONO'
        spc2 = Species().from_smiles('C[N+](=O)[O-]')
        spc2.label = 'CNO2'
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
                         'conformer_index': None,
                         'imaginary_freqs': None,
                         'successful_irc': None,
                         'successful_normal_mode': None,
                         'energy': None,
                         'family': 'H_Abstraction',
                         'index': None,
                         'rmg_reaction': 'CON=O <=> [O-][N+](=O)C',
                         'success': None,
                         'method_direction': None,
                         'method_index': None,
                         't0': None,
                         'execution_time': None}
        self.assertEqual(tsg_dict, expected_dict)

    def test_from_dict(self):
        """
        Test TSGuess.from_dict()

        Also tests that the round trip to and from a dictionary ended in an RMG Reaction object.
        """
        ts_dict = self.tsg1.as_dict()
        tsg = TSGuess(ts_dict=ts_dict)
        self.assertEqual(tsg.method, 'autotst')
        self.assertTrue(isinstance(tsg.rmg_reaction, Reaction))

    def test_xyz_perception(self):
        """Test MolGraph.get_formula()"""
        xyz_arb = {'symbols': ('H', 'C', 'H', 'H', 'O', 'N', 'O'),
                   'isotopes': (1, 13, 1, 1, 16, 14, 16),
                   'coords': ((-1.0, 0.0, 0.0),
                              (0.0, 0.0, 0.0),
                              (0.0, -1.0, 0.0),
                              (0.0, 1.0, 0.0),
                              (1.0, 0.0, 0.0),
                              (2.0, 0.0, 0.0),
                              (3.0, 0.0, 0.0),)}
        mol_graph_1 = MolGraph(symbols=xyz_arb['symbols'], coords=xyz_arb['coords'])
        self.assertEqual(mol_graph_1.get_formula(), 'CH3NO2')


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
