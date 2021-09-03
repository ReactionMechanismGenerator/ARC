#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.species.conformers module
"""

import unittest

from rdkit.Chem import rdMolTransforms as rdMT

from rmgpy.molecule.atomtype import ATOMTYPES
from rmgpy.molecule.group import GroupAtom, GroupBond, Group
from rmgpy.molecule.molecule import Molecule

import arc.species.conformers as conformers
import arc.species.converter as converter
import arc.species.vectors as vectors
from arc.common import almost_equal_coords_lists
from arc.exceptions import ConformerError
from arc.species.species import ARCSpecies


class TestConformers(unittest.TestCase):
    """
    Contains unit tests for the conformers module
    """

    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None
        xyz1 = """O      -1.48027320    0.36597456    0.41386552
C      -0.49770656   -0.40253648   -0.26500019
C       0.86215119    0.24734211   -0.11510338
H      -0.77970114   -0.46128090   -1.32025907
H      -0.49643724   -1.41548311    0.14879346
H       0.84619526    1.26924854   -0.50799415
H       1.14377239    0.31659216    0.94076336
H       1.62810781   -0.32407050   -0.64676910
H      -1.22610851    0.40421362    1.35170355"""
        cls.spc0 = ARCSpecies(label='CCO', smiles='CCO', xyz=xyz1)  # define from xyz for consistent atom order
        cls.mol0 = cls.spc0.mol

        adj1 = """multiplicity 2
1 C u0 p0 c0 {2,S} {5,S} {6,S} {7,S}
2 C u0 p0 c0 {1,S} {3,D} {8,S}
3 C u0 p0 c0 {2,D} {4,S} {9,S}
4 O u1 p2 c0 {3,S}
5 H u0 p0 c0 {1,S}
6 H u0 p0 c0 {1,S}
7 H u0 p0 c0 {1,S}
8 H u0 p0 c0 {2,S}
9 H u0 p0 c0 {3,S}"""
        cls.mol1 = Molecule().from_adjacency_list(adj1)

        cls.cj_xyz = {'symbols': ('C', 'O', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'N', 'C', 'C', 'C', 'C', 'N', 'C', 'C',
                                  'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C',
                                  'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H',
                                  'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H',
                                  'H', 'H', 'H', 'H'),
                      'isotopes': (12, 16, 12, 12, 12, 12, 12, 12, 12, 14, 12, 12, 12, 12, 14, 12, 12, 12, 12, 12, 12,
                                   12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
                      'coords': (
                      (5.675, 2.182, 1.81), (4.408, 1.923, 1.256), (4.269, 0.813, 0.479), (5.303, -0.068, 0.178),
                      (5.056, -1.172, -0.639), (3.794, -1.414, -1.169), (2.77, -0.511, -0.851),
                      (2.977, 0.59, -0.032), (1.872, 1.556, 0.318), (0.557, 1.029, -0.009),
                      (-0.537, 1.879, 0.448), (-0.535, 3.231, -0.298), (-1.831, 3.983, 0.033),
                      (-3.003, 3.199, -0.61), (-2.577, 1.854, -0.99), (-1.64, 1.962, -2.111),
                      (-0.501, 2.962, -1.805), (-1.939, 1.236, 0.178), (-1.971, -0.305, 0.069),
                      (-3.385, -0.794, -0.209), (-4.336, -0.893, 0.81), (-5.631, -1.324, 0.539),
                      (-5.997, -1.673, -0.759), (-5.056, -1.584, -1.781), (-3.764, -1.147, -1.505),
                      (-1.375, -1.024, 1.269), (-1.405, -0.508, 2.569), (-0.871, -1.226, 3.638),
                      (-0.296, -2.475, 3.429), (-0.259, -3.003, 2.14), (-0.794, -2.285, 1.078),
                      (3.533, -2.614, -2.056), (2.521, -3.574, -1.424), (3.087, -2.199, -3.461),
                      (5.569, 3.097, 2.395), (6.433, 2.338, 1.031), (6.003, 1.368, 2.47), (6.302, 0.091, 0.57),
                      (5.874, -1.854, -0.864), (1.772, -0.654, -1.257), (1.963, 1.832, 1.384),
                      (2.033, 2.489, -0.239), (0.469, 0.13, 0.461), (-0.445, 2.089, 1.532), (0.328, 3.83, 0.012),
                      (-1.953, 4.059, 1.122), (-1.779, 5.008, -0.352), (-3.365, 3.702, -1.515),
                      (-3.856, 3.118, 0.074), (-1.226, 0.969, -2.31), (-2.211, 2.259, -2.999),
                      (-0.639, 3.906, -2.348), (0.466, 2.546, -2.105), (-2.586, 1.501, 1.025),
                      (-1.36, -0.582, -0.799), (-4.057, -0.647, 1.831), (-6.355, -1.396, 1.347),
                      (-7.006, -2.015, -0.97), (-5.329, -1.854, -2.798), (-3.038, -1.07, -2.311),
                      (-1.843, 0.468, 2.759), (-0.904, -0.802, 4.638), (0.125, -3.032, 4.262),
                      (0.189, -3.977, 1.961), (-0.772, -2.708, 0.075), (4.484, -3.155, -2.156),
                      (1.543, -3.093, -1.308), (2.383, -4.464, -2.049), (2.851, -3.899, -0.431),
                      (3.826, -1.542, -3.932), (2.134, -1.659, -3.429), (2.951, -3.078, -4.102))}
        cls.cj_spc = ARCSpecies(label='CJ', xyz=cls.cj_xyz,
                                smiles='COC1=C(CN[C@H]2C3CCN(CC3)[C@H]2C(C2=CC=CC=C2)C2=CC=CC=C2)C=C(C=C1)C(C)C')

    def test_CONFS_VS_HEAVY_ATOMS(self):
        """Test that the CONFS_VS_HEAVY_ATOMS dictionary has 0 and 'inf' in its keys"""
        found_zero = False
        for key in conformers.CONFS_VS_HEAVY_ATOMS.keys():
            if key[0] == 0:
                found_zero = True
                break
        self.assertTrue(found_zero, 'The CONFS_VS_HEAVY_ATOMS dictionary has to have a key that srarts at zero. '
                                    'got:\n{0}'.format(conformers.CONFS_VS_HEAVY_ATOMS))
        found_inf = False
        for key in conformers.CONFS_VS_HEAVY_ATOMS.keys():
            if key[1] == 'inf':
                found_inf = True
                break
        self.assertTrue(found_inf, "The CONFS_VS_HEAVY_ATOMS dictionary has to have a key that ends with 'inf'. "
                                   "got:\n{0}".format(conformers.CONFS_VS_HEAVY_ATOMS))

    def test_CONFS_VS_TORSIONS(self):
        """Test that the CONFS_VS_TORSIONS dictionary has 0 and 'inf' in its keys"""
        found_zero = False
        for key in conformers.CONFS_VS_TORSIONS.keys():
            if key[0] == 0:
                found_zero = True
                break
        self.assertTrue(found_zero, 'The CONFS_VS_TORSIONS dictionary has to have a key that srarts at zero. '
                                    'got:\n{0}'.format(conformers.CONFS_VS_TORSIONS))
        found_inf = False
        for key in conformers.CONFS_VS_TORSIONS.keys():
            if key[1] == 'inf':
                found_inf = True
                break
        self.assertTrue(found_inf, "The CONFS_VS_TORSIONS dictionary has to have a key that ends with 'inf'. "
                                   "got:\n{0}".format(conformers.CONFS_VS_TORSIONS))

    def test_generate_conformers_with_openbabel(self):
        """Test the main conformer generation function with a species for which RDKit fails to generate conformers"""
        xyz = converter.str_to_xyz("""C         -2.18276        2.03598        0.00028
        C         -0.83696        1.34108       -0.05231
        H         -2.23808        2.82717       -0.75474
        H         -2.33219        2.51406        0.97405
        H         -2.99589        1.32546       -0.17267
        O          0.18176        2.30786        0.17821
        H         -0.69161        0.88171       -1.03641
        H         -0.78712        0.56391        0.71847
        O          1.39175        1.59510        0.11494""")
        spc = ARCSpecies(label='CCO[O]', smiles='CCO[O]', xyz=xyz)
        lowest_confs = conformers.generate_conformers(mol_list=spc.mol_list, label=spc.label,
                                                      charge=spc.charge, multiplicity=spc.multiplicity,
                                                      force_field='MMFF94s', print_logs=False, diastereomers=None,
                                                      n_confs=1, return_all_conformers=False)
        self.assertEqual(len(lowest_confs), 1)
        self.assertAlmostEqual(lowest_confs[0]['FF energy'], 1.7935, 3)
        expected_xyz = {'symbols': ('C', 'C', 'H', 'H', 'H', 'O', 'H', 'H', 'O'),
                        'isotopes': (12, 12, 1, 1, 1, 16, 1, 1, 16),
                        'coords': ((-1.06401, 0.15134, -0.02907),
                                   (0.39059, -0.26595, -0.11181),
                                   (-1.38709, 0.21893, 1.01503),
                                   (-1.21986, 1.1209, -0.51032),
                                   (-1.70646, -0.59124, -0.51377),
                                   (0.53686, -1.52925, 0.52713),
                                   (0.6972, -0.347, -1.16062),
                                   (1.0214, 0.47542, 0.39124),
                                   (1.90095, -1.84857, 0.41142))}
        # Only symbols instead of the coordinate values are compared.
        # This is due to the unknown behavior of OpenBabel optimization function.
        # With the same iteration number and same initial xyz, the optimized xyzs can
        # be different on different machines, while energies are still consistent.
        # More info can be found from PR #332
        self.assertEqual(lowest_confs[0]['xyz']['symbols'], expected_xyz['symbols'])

        # HO2
        lowest_confs = conformers.generate_conformers(mol_list=[Molecule(smiles='O[O]')],
                                                      label='HO2',
                                                      charge=0,
                                                      multiplicity=2,
                                                      force_field='MMFF94s',
                                                      )
        self.assertEqual(lowest_confs[0]['xyz']['symbols'], ('O', 'O', 'H'))

    def test_generate_conformers_with_specific_diastereomers(self):
        """Test the main conformer generation function, considering a specific diastereomer"""
        spc1 = ARCSpecies(label='spc1', smiles='IC=CC(Cl)(I)NF')
        lowest_confs = conformers.generate_conformers(mol_list=spc1.mol_list, label=spc1.label,
                                                      charge=spc1.charge, multiplicity=spc1.multiplicity,
                                                      force_field='MMFF94s', print_logs=False, diastereomers=None,
                                                      n_confs=1, return_all_conformers=False)
        self.assertEqual(len(lowest_confs), 1)
        self.assertEqual(lowest_confs[0]['chirality'], {(1, 2): 'E', (3,): 'S', (6,): 'NS'})

        diastereomers = ["""I       0.44040267   -1.86187033   -1.61216600
                            C       0.44040267   -1.86187033    0.42506052
                            C       0.44040267   -0.80545983    1.25546732
                            C       0.51339746    0.67654644    0.99549106
                            Cl      1.15587774    1.51105829    2.44189131
                            I      -1.44518629    1.43298811    0.68505581
                            N       1.32635015    1.11409115   -0.12238582
                            F       2.63651016    0.68779407   -0.01892330
                            H       0.39731749   -2.85756010    0.86584403
                            H       0.38923719   -1.05997885    2.31383716
                            H       0.89917512    0.75112899   -0.97160458"""]
        lowest_confs = conformers.generate_conformers(mol_list=spc1.mol_list, label=spc1.label,
                                                      charge=spc1.charge, multiplicity=spc1.multiplicity,
                                                      force_field='MMFF94s', print_logs=False,
                                                      n_confs=1, return_all_conformers=False,
                                                      diastereomers=diastereomers)
        self.assertEqual(len(lowest_confs), 1)
        lowest_confs = conformers.determine_chirality(lowest_confs, spc1.label, spc1.mol)
        self.assertEqual(lowest_confs[0]['chirality'], {(1, 2): 'Z', (3,): 'S', (6,): 'NR'})

    def test_deduce_new_conformers(self):
        """Test deducing new conformers"""
        confs = [{'index': 0, 'xyz': {'symbols': ('O', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'),
                                      'isotopes': (16, 12, 12, 12, 1, 1, 1, 1, 1, 1, 1, 1),
                                      'coords': ((1.49632323, 0.74450682, 0.93344565),
                                                 (-0.27411342, -0.73369536, 0.23273076),
                                                 (1.00918754, -0.0158525, -0.16288919),
                                                 (-1.40106678, 0.23458846, 0.56318695),
                                                 (-0.08785698, -1.36178955, 1.11163236),
                                                 (-0.59069375, -1.38959579, -0.58564668),
                                                 (0.83747022, 0.65606977, -1.00993468),
                                                 (1.77546073, -0.7455614, -0.44289977),
                                                 (-1.14379759, 0.86897309, 1.41716858),
                                                 (-2.31108163, -0.31832139, 0.81688419),
                                                 (-1.62427975, 0.88145943, -0.29124479),
                                                 (2.31444818, 1.17921842, 0.63815656))},
                  'torsion_dihedrals': {(9, 4, 2, 3): -61.78942, (4, 2, 3, 1): 63.79634, (2, 3, 1, 12): 179.70585},
                  'source': 'MMFF94s', 'FF energy': -1.5241875152610689},
                 {'index': 1, 'xyz': {'symbols': ('O', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'),
                                      'isotopes': (16, 12, 12, 12, 1, 1, 1, 1, 1, 1, 1, 1),
                                      'coords': ((2.09496537, -0.68203123, 0.41738811),
                                                 (-0.17540789, 0.11818414, 0.51180976),
                                                 (0.92511172, -0.46810337, -0.36086829),
                                                 (-1.45486974, 0.34772573, -0.27221056),
                                                 (-0.37104415, -0.55290987, 1.35664654),
                                                 (0.16538089, 1.0635112, 0.95069762),
                                                 (0.61854668, -1.431406, -0.78023161),
                                                 (1.17698645, 0.20191002, -1.18924062),
                                                 (-2.22790196, 0.76917222, 0.37791757),
                                                 (-1.8347639, -0.59150616, -0.68674647),
                                                 (-1.28838516, 1.04591267, -1.0987324),
                                                 (2.3713817, 0.17954064, 0.77357037))},
                  'torsion_dihedrals': {(9, 4, 2, 3): -179.98511, (4, 2, 3, 1): -179.22951, (2, 3, 1, 12): -60.04260},
                  'source': 'MMFF94s', 'FF energy': -1.641467160999287},
                 {'index': 2, 'xyz': {'symbols': ('O', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'),
                                      'isotopes': (16, 12, 12, 12, 1, 1, 1, 1, 1, 1, 1, 1),
                                      'coords': ((1.68732977, -0.20570878, 0.8604398),
                                                 (-0.41954613, 0.64790835, 0.00757451),
                                                 (1.08358067, 0.51540154, -0.2039521),
                                                 (-1.1458917, -0.68756266, -0.0638385),
                                                 (-0.83158455, 1.31816915, -0.7550012),
                                                 (-0.61484441, 1.09768535, 0.98791309),
                                                 (1.31520902, 0.01348062, -1.14907531),
                                                 (1.54143425, 1.50913982, -0.22977912),
                                                 (-0.95895886, -1.1849333, -1.02082479),
                                                 (-0.83081898, -1.35799497, 0.74145742),
                                                 (-2.22543313, -0.5341506, 0.03344438),
                                                 (1.39952405, -1.13143453, 0.79164182))},
                  'torsion_dihedrals': {(9, 4, 2, 3): -57.02407, (4, 2, 3, 1): -66.21040, (2, 3, 1, 12): 69.65707},
                  'source': 'MMFF94s', 'FF energy': -1.0563449757315282}]
        torsions = [[9, 4, 2, 3], [4, 2, 3, 1], [2, 3, 1, 12]]
        tops = [[4, 9, 10, 11], [3, 7, 8, 1, 12], [1, 12]]

        spc1 = ARCSpecies(label='propanol', smiles='CCCO', xyz=confs[0]['xyz'])
        new_conformers, symmetries = conformers.deduce_new_conformers(
            label='', conformers=confs, torsions=torsions, tops=tops, mol_list=[spc1.mol], plot_path=None,
            combination_threshold=10, force_field='MMFF94s', max_combination_iterations=25)

        self.assertEqual(len(new_conformers), 6)
        self.assertEqual(symmetries, {(2, 3, 1, 12): 1, (4, 2, 3, 1): 1, (9, 4, 2, 3): 3})

        expected_new_conformers = [{'index': 3,
                                    'xyz': {'symbols': ('O', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'),
                                            'isotopes': (16, 12, 12, 12, 1, 1, 1, 1, 1, 1, 1, 1),
                                            'coords': ((2.094965350070438, -0.6820312883655302, 0.41738812543556636),
                                                       (-0.17540789, 0.11818414, 0.5118097600000001),
                                                       (0.9251116997351838, -0.4681034307777509, -0.36086827472356287),
                                                       (-1.45486974, 0.34772573, -0.27221056),
                                                       (-0.37104416020111164, -0.5529098320998265, 1.3566465677436679),
                                                       (0.1653809158507713, 1.063511209038263, 0.9506975804596753),
                                                       (0.6185466335892884, -1.4314060698728557, -0.7802315547182584),
                                                       (1.176986440026325, 0.20190992061504034, -1.1892406328211544),
                                                       (-2.22790196, 0.76917222, 0.37791757),
                                                       (-1.8347639, -0.59150616, -0.68674647),
                                                       (-1.28838516, 1.04591267, -1.0987324),
                                                       (2.371381703321999, 0.17954058897321318, 0.7735703496393789))},
                                    'FF energy': -1.641,
                                    'source': 'Changing dihedrals on most stable conformer, iteration 0',
                                    'torsion': (9, 4, 2, 3), 'dihedral': -179.99},
                                   {'index': 4,
                                    'xyz': {'symbols': ('O', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'),
                                            'isotopes': (16, 12, 12, 12, 1, 1, 1, 1, 1, 1, 1, 1),
                                            'coords': ((1.5629128183069128, 1.150273092080661, 1.8221106171165564),
                                                       (-0.17540789000000023, 0.11818413999999994, 0.5118097599999999),
                                                       (0.3581470519572676, 1.4100190033709756, 1.1142425322000384),
                                                       (-1.45486974, 0.34772573, -0.27221056),
                                                       (0.5867474458121764, -0.3255284963612595, -0.1398252925806644),
                                                       (-0.35452520109837593, -0.6142892439766408, 1.3079269167226824),
                                                       (0.5782647385545507, 2.1431855647619886, 0.3320567025011878),
                                                       (-0.3629845778602747, 1.8545497751561877, 1.8077272868372494),
                                                       (-2.22790196, 0.76917222, 0.37791757),
                                                       (-1.8347639, -0.59150616, -0.68674647),
                                                       (-1.28838516, 1.04591267, -1.0987324),
                                                       (1.357345644952105, 0.5110314515764744, 2.525452738199251))},
                                    'FF energy': -1.641,
                                    'source': 'Changing dihedrals on most stable conformer, iteration 0',
                                    'torsion': (9, 4, 2, 3), 'dihedral': -59.41},
                                   {'index': 5,
                                    'xyz': {'symbols': ('O', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'),
                                            'isotopes': (16, 12, 12, 12, 1, 1, 1, 1, 1, 1, 1, 1),
                                            'coords': ((0.4044889246474961, -1.5363469758977075, -1.1404920577633173),
                                                       (-0.17540789, 0.11818414, 0.51180976),
                                                       (0.9251117200000001, -0.4681033699999999, -0.3608682900000001),
                                                       (-1.45486974, 0.34772573, -0.27221056),
                                                       (-0.37104415, -0.55290987, 1.35664654),
                                                       (0.16538089, 1.0635112, 0.95069762),
                                                       (1.3228072266085593, 0.2866998875438399, -1.0463681086003478),
                                                       (1.7527703334738094, -0.8514884848383631, 0.24457089998069526),
                                                       (-2.22790196, 0.76917222, 0.37791757),
                                                       (-1.8347639, -0.59150616, -0.68674647),
                                                       (-1.28838516, 1.04591267, -1.0987324),
                                                       (
                                                       0.06773060495376348, -2.2104474122994415, -0.5258699683744645))},
                                    'FF energy': -1.405,
                                    'source': 'Changing dihedrals on most stable conformer, iteration 0',
                                    'torsion': (4, 2, 3, 1), 'dihedral': -46.21},
                                   {'index': 6,
                                    'xyz': {'symbols': ('O', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'),
                                            'isotopes': (16, 12, 12, 12, 1, 1, 1, 1, 1, 1, 1, 1),
                                            'coords': ((0.3648443372009176, -0.9869852599973693, -1.5595624794610512),
                                                       (-0.17540789, 0.11818414, 0.51180976),
                                                       (0.9251117200000001, -0.46810337000000013, -0.36086828999999987),
                                                       (-1.45486974, 0.34772573, -0.27221056),
                                                       (-0.37104415, -0.55290987, 1.35664654),
                                                       (0.16538089, 1.0635112, 0.95069762),
                                                       (1.655719796634958, 0.29927576676270773, -0.6349864646868111),
                                                       (1.4540141045677961, -1.2768891252490158, 0.1536095124427027),
                                                       (-2.22790196, 0.76917222, 0.37791757),
                                                       (-1.8347639, -0.59150616, -0.68674647),
                                                       (-1.28838516, 1.04591267, -1.0987324),
                                                       (-0.271008482835584, -1.6790856272786052, -1.3100306180375452))},
                                    'FF energy': -1.405,
                                    'source': 'Changing dihedrals on most stable conformer, iteration 0',
                                    'torsion': (4, 2, 3, 1), 'dihedral': -16.21},
                                   {'index': 7,
                                    'xyz': {'symbols': ('O', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'),
                                            'isotopes': (16, 12, 12, 12, 1, 1, 1, 1, 1, 1, 1, 1),
                                            'coords': ((0.5687233426911779, -0.3483474092261868, -1.7315073928355307),
                                                       (-0.17540789, 0.11818414, 0.51180976),
                                                       (0.9251117200000001, -0.46810337, -0.3608682900000002),
                                                       (-1.45486974, 0.34772573, -0.27221056),
                                                       (-0.37104415, -0.55290987, 1.35664654),
                                                       (0.16538089, 1.0635112, 0.95069762),
                                                       (1.8684013537054927, 0.06599282702430986, -0.2100518646860865),
                                                       (1.090845033585963, -1.5267601082842295, -0.13650715532042257),
                                                       (-2.22790196, 0.76917222, 0.37791757),
                                                       (-1.8347639, -0.59150616, -0.68674647),
                                                       (-1.28838516, 1.04591267, -1.0987324),
                                                       (
                                                       -0.26099475485192725, -0.8382937416014827, -1.862268040608934))},
                                    'FF energy': -1.056,
                                    'source': 'Changing dihedrals on most stable conformer, iteration 0',
                                    'torsion': (4, 2, 3, 1), 'dihedral': 13.79},
                                   {'index': 8,
                                    'xyz': {'symbols': ('O', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'),
                                            'isotopes': (16, 12, 12, 12, 1, 1, 1, 1, 1, 1, 1, 1),
                                            'coords': ((0.9614967262435001, 0.20844408004575415, -1.6102542972054235),
                                                       (-0.17540789, 0.11818414, 0.51180976),
                                                       (0.9251117199999999, -0.4681033699999999, -0.36086828999999976),
                                                       (-1.45486974, 0.34772573, -0.27221056),
                                                       (-0.37104415, -0.55290987, 1.35664654),
                                                       (0.16538089, 1.0635112, 0.95069762),
                                                       (1.9038640463581198, -0.3506409563604913, 0.11457480849559187),
                                                       (0.760573979813852, -1.5341487058277492, -0.548042576470742),
                                                       (-2.22790196, 0.76917222, 0.37791757),
                                                       (-1.8347639, -0.59150616, -0.68674647),
                                                       (-1.28838516, 1.04591267, -1.0987324),
                                                       (
                                                       0.09508861857828704, 0.08663873796210059, -2.0346106646803817))},
                                    'FF energy': -1.056,
                                    'source': 'Changing dihedrals on most stable conformer, iteration 0',
                                    'torsion': (4, 2, 3, 1), 'dihedral': 43.79},
                                   {'index': 9,
                                    'xyz': {'symbols': ('O', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'),
                                            'isotopes': (16, 12, 12, 12, 1, 1, 1, 1, 1, 1, 1, 1),
                                            'coords': ((2.0949653699999997, -0.6820312299999999, 0.4173881099999994),
                                                       (-0.17540789, 0.11818414, 0.51180976),
                                                       (0.92511172, -0.46810337, -0.36086829),
                                                       (-1.45486974, 0.34772573, -0.27221056),
                                                       (-0.37104415, -0.55290987, 1.35664654),
                                                       (0.16538089, 1.0635112, 0.95069762),
                                                       (0.61854668, -1.431406, -0.78023161),
                                                       (1.17698645, 0.20191002, -1.18924062),
                                                       (-2.22790196, 0.76917222, 0.37791757),
                                                       (-1.8347639, -0.59150616, -0.68674647),
                                                       (-1.28838516, 1.04591267, -1.0987324),
                                                       (2.079736137980861, -1.6056159752671268, 0.7212584208344683))},
                                    'FF energy': -1.641,
                                    'source': 'Changing dihedrals on most stable conformer, iteration 0',
                                    'torsion': (2, 3, 1, 12), 'dihedral': 94.68},
                                   {'index': 10,
                                    'xyz': {'symbols': ('O', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'),
                                            'isotopes': (16, 12, 12, 12, 1, 1, 1, 1, 1, 1, 1, 1),
                                            'coords': ((2.0949653699999997, -0.6820312299999999, 0.41738810999999976),
                                                       (-0.17540789, 0.11818414, 0.51180976),
                                                       (0.92511172, -0.46810337, -0.36086829),
                                                       (-1.45486974, 0.34772573, -0.27221056),
                                                       (-0.37104415, -0.55290987, 1.35664654),
                                                       (0.16538089, 1.0635112, 0.95069762),
                                                       (0.61854668, -1.431406, -0.78023161),
                                                       (1.17698645, 0.20191002, -1.18924062),
                                                       (-2.22790196, 0.76917222, 0.37791757),
                                                       (-1.8347639, -0.59150616, -0.68674647),
                                                       (-1.28838516, 1.04591267, -1.0987324),
                                                       (2.3440717266511886, -1.617013290161668, 0.3207835176842774))},
                                    'FF energy': -1.810,
                                    'source': 'Changing dihedrals on most stable conformer, iteration 0',
                                    'torsion': (2, 3, 1, 12), 'dihedral': 124.68},
                                   {'index': 11,
                                    'xyz': {'symbols': ('O', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'),
                                            'isotopes': (16, 12, 12, 12, 1, 1, 1, 1, 1, 1, 1, 1),
                                            'coords': ((2.09496537, -0.68203123, 0.41738810999999987),
                                                       (-0.17540789, 0.11818414, 0.51180976),
                                                       (0.92511172, -0.46810337, -0.36086829),
                                                       (-1.45486974, 0.34772573, -0.27221056),
                                                       (-0.37104415, -0.55290987, 1.35664654),
                                                       (0.16538089, 1.0635112, 0.95069762),
                                                       (0.61854668, -1.431406, -0.78023161),
                                                       (1.17698645, 0.20191002, -1.18924062),
                                                       (-2.22790196, 0.76917222, 0.37791757),
                                                       (-1.8347639, -0.59150616, -0.68674647),
                                                       (-1.28838516, 1.04591267, -1.0987324),
                                                       (
                                                       2.606253045729423, -1.3896949636775977, -0.010834763851175933))},
                                    'FF energy': -1.810,
                                    'source': 'Changing dihedrals on most stable conformer, iteration 0',
                                    'torsion': (2, 3, 1, 12), 'dihedral': 154.68}]

        for i, (new_conformer, expected_new_conformer) in enumerate(zip(new_conformers, expected_new_conformers)):
            self.assertEqual(new_conformer['index'], expected_new_conformer['index'])
            self.assertEqual(new_conformer['FF energy'], expected_new_conformer['FF energy'])
            self.assertEqual(new_conformer['source'], expected_new_conformer['source'])
            self.assertEqual(new_conformer['torsion'], expected_new_conformer['torsion'])
            self.assertEqual(new_conformer['dihedral'], expected_new_conformer['dihedral'])
            self.assertTrue(almost_equal_coords_lists(new_conformer['xyz'], expected_new_conformer['xyz']))

    def test_get_force_field_energies(self):
        """Test attaining force field conformer energies"""
        xyzs, energies = conformers.get_force_field_energies(label='', mol=self.mol0, num_confs=10)
        self.assertEqual(len(xyzs), 10)
        self.assertEqual(len(energies), 10)
        mol0 = converter.molecules_from_xyz(xyzs[0])[1]
        self.assertTrue(self.mol0.is_isomorphic(mol0), 'Could not complete a round trip from Molecule to xyz and back '
                                                       'via RDKit')

        ch2oh_xyz = converter.str_to_xyz("""O       0.83632835   -0.29575461    0.40459411
C      -0.43411393   -0.07778692   -0.05635829
H      -1.16221394   -0.80894238    0.24815765
H      -0.64965442    0.77699377   -0.67782845
H       1.40965394    0.40549015    0.08143497""")
        ch2oh_mol = Molecule(smiles='[CH2]O')
        energies = conformers.get_force_field_energies(label='', mol=ch2oh_mol, xyz=ch2oh_xyz, optimize=True)[1]
        self.assertAlmostEqual(energies[0], 13.466911, 5)

        # Test peroxyl radical (uses UFF by default if MMFF94s is still buggy in RDKit).
        xyzs, energies = conformers.get_force_field_energies(label='', mol=Molecule(smiles='CCO[O]'), num_confs=10)
        self.assertEqual(len(energies), 10)
        self.assertEqual(len(xyzs), 10)
        self.assertEqual(xyzs[0]['symbols'], ('C', 'C', 'O', 'O', 'H', 'H', 'H', 'H', 'H'))

    def test_generate_force_field_conformers(self):
        """Test generating conformers from RDKit """
        mol_list = [self.mol0]
        label = 'ethanol'
        xyzs = [converter.str_to_xyz("""O       1.22700646   -0.74306134   -0.46642912
C       0.44275447    0.24386237    0.18670577
C      -1.02171998   -0.04112700   -0.06915927
H       0.65837353    0.21260997    1.25889575
H       0.71661793    1.22791280   -0.20534525
H      -1.29179957   -1.03551511    0.30144722
H      -1.23432965   -0.03370364   -1.14332936
H      -1.65581027    0.70264687    0.42135230
H       2.15890709   -0.53362491   -0.28413803"""),
                converter.str_to_xyz("""O       0.97434661    0.08786861   -0.03552502
C       2.39398906    0.09669513   -0.02681997
C       2.90028285   -0.91910243   -1.02843131
H       2.74173418   -0.15134085    0.98044759
H       2.74173583    1.10037467   -0.28899552
H       2.53391442   -1.92093297   -0.78104074
H       2.53391750   -0.68582304   -2.03364248
H       3.99353906   -0.93757558   -1.04664384
H       0.68104300    0.74807180    0.61546062""")]
        torsion_num = 2
        charge = 0
        multiplicity = 1

        confs = conformers.generate_force_field_conformers(mol_list=mol_list, label=label, torsion_num=torsion_num,
                                                           charge=charge, multiplicity=multiplicity, xyzs=xyzs,
                                                           num_confs=50)

        self.assertEqual(len(confs), 52)
        self.assertEqual(confs[0]['source'], 'MMFF94s')
        self.assertEqual(confs[0]['index'], 0)
        self.assertEqual(confs[1]['index'], 1)
        self.assertEqual(confs[-1]['index'], 51)
        self.assertEqual(confs[-1]['source'], 'User Guess')
        self.assertFalse(any([confs[i]['xyz'] == confs[0]['xyz'] for i in range(1, 52)]))

    def test_determine_number_of_conformers_to_generate(self):
        """Test that the correct number of conformers to generate is determined"""
        self.assertEqual(conformers.determine_number_of_conformers_to_generate(heavy_atoms=0, torsion_num=0,
                                                                               label=''), (75, 0))
        self.assertEqual(conformers.determine_number_of_conformers_to_generate(heavy_atoms=15, torsion_num=0,
                                                                               label=''), (1000, 0))
        self.assertEqual(conformers.determine_number_of_conformers_to_generate(heavy_atoms=5, torsion_num=31,
                                                                               label=''), (2500, 0))
        self.assertEqual(conformers.determine_number_of_conformers_to_generate(heavy_atoms=150, torsion_num=0,
                                                                               label=''), (7500, 0))
        xyz = {'symbols': ('S', 'O', 'O', 'N', 'N', 'C', 'C', 'C', 'C', 'C',
                           'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'),
               'isotopes': (32, 16, 16, 14, 14, 12, 12, 12, 12, 12, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
               'coords': ((-1.7297500450289969, 0.515438796636824, 1.9168096618045691),
                          (-1.4803657376736399, 1.1318228490801632, -1.2741346588410747),
                          (1.6684385634300425, -1.4080623343331675, -0.69146134720917),
                          (1.4447732964722193, 0.26810320987982, 1.834186396466309),
                          (2.9639741941549724, 0.45637046406705295, -0.6018493257174595),
                          (0.7107726366328042, 0.4671661961997124, 0.5489199037723008),
                          (-0.68225248960311, -0.2422492925439827, 0.5854344518475088),
                          (-1.416609693814781, -0.21038706266511795, -0.7805348852864422),
                          (1.5885097252447307, -0.0015515049234385968, -0.6624596185356767),
                          (-2.818273455523886, -0.8196266870848813, -0.7693094853681998),
                          (0.5645026891741011, 1.5505258112503815, 0.44320480645337756),
                          (-0.5489109707645772, -1.2919888767636876, 0.8751860070317062),
                          (-0.8321905851878897, -0.784265725208026, -1.507876072604498),
                          (1.1510890997187422, 0.3429505362529166, -1.6042827339693007),
                          (-3.539862945639595, -0.18351203035164942, -0.24723208418918377),
                          (-2.81666678799753, -1.815870747483849, -0.3170415955378251),
                          (-3.191244105746757, -0.9126055866431627, -1.795841007782851),
                          (3.3764976755549823, 0.08016081376522317, 0.24941044358861653),
                          (2.959962629146053, 1.469289098466773, -0.48374550042835573),
                          (1.7124864428642297, -0.7164193231909225, 1.8874284688782355),
                          (2.3042342900725874, 0.8091358385609299, 1.8231896193937174),
                          (2.4468466474830963, -1.5710191541618868, -1.2534461848335496),
                          (-1.779032202163809, 1.7365703107348773, 1.363680443883325),
                          (-2.0569288708039766, 1.130024400459187, -2.059052520249386))}
        self.assertEqual(conformers.determine_number_of_conformers_to_generate(heavy_atoms=30, torsion_num=10,
                                                                               label='', xyz=xyz), (10000, 4))
        mol = Molecule(smiles='CNC(O)(S)C=CO')
        self.assertEqual(conformers.determine_number_of_conformers_to_generate(heavy_atoms=30, torsion_num=10,
                                                                               label='', mol=mol), (7500, 3))

    def test_openbabel_force_field(self):
        """Test Open Babel force field"""
        xyz = """S      -0.19093478    0.57933906    0.00000000
                 O      -1.21746139   -0.72237602    0.00000000
                 O       1.40839617    0.14303696    0.00000000"""
        spc = ARCSpecies(label='SO2', smiles='O=S=O', xyz=xyz)
        xyzs, energies = conformers.openbabel_force_field(label='',
                                                          mol=spc.mol,
                                                          num_confs=1,
                                                          force_field='GAFF',
                                                          method='diverse',
                                                          )
        self.assertEqual(len(xyzs), 1)
        self.assertAlmostEqual(energies[0], 2.931930, 3)

    def test_openbabel_force_field_on_rdkit_conformers(self):
        """Test Open Babel force field on RDKit conformers"""
        xyz = converter.str_to_xyz("""C         -2.18276        2.03598        0.00028
                                      C         -0.83696        1.34108       -0.05231
                                      H         -2.23808        2.82717       -0.75474
                                      H         -2.33219        2.51406        0.97405
                                      H         -2.99589        1.32546       -0.17267
                                      O          0.18176        2.30786        0.17821
                                      H         -0.69161        0.88171       -1.03641
                                      H         -0.78712        0.56391        0.71847
                                      O          1.39175        1.59510        0.11494""")
        spc = ARCSpecies(label='CCO[O]', smiles='CCO[O]', xyz=xyz)
        rd_mol = conformers.embed_rdkit(label='', mol=spc.mol, num_confs=2, xyz=xyz)
        xyzs, energies = conformers.openbabel_force_field_on_rdkit_conformers(label='', rd_mol=rd_mol,
                                                                              force_field='MMFF94s',)
        expected_xyzs = [{'symbols': ('C', 'C', 'H', 'H', 'H', 'O', 'H', 'H', 'O'),
                          'isotopes': (12, 12, 1, 1, 1, 16, 1, 1, 16),
                          'coords': ((-0.92004, 0.17591, -0.00274),
                                     (0.41052, -0.54976, 0.06772),
                                     (-1.70176, -0.48351, -0.39026),
                                     (-0.85205, 1.05583, -0.65051),
                                     (-1.21725, 0.53214, 0.98907),
                                     (1.40523, 0.29451, 0.63948),
                                     (0.7238, -0.90625, -0.92072),
                                     (0.3145, -1.42616, 0.71666),
                                     (1.87763, 1.10137, -0.41125))},
                         {'symbols': ('C', 'C', 'H', 'H', 'H', 'O', 'H', 'H', 'O'),
                          'isotopes': (12, 12, 1, 1, 1, 16, 1, 1, 16),
                          'coords': ((-1.06853, -0.10587, 0.01916),
                                     (0.39395, 0.27799, -0.08402),
                                     (-1.33109, -0.35023, 1.05379),
                                     (-1.71119, 0.70925, -0.32507),
                                     (-1.2765, -0.99769, -0.58128),
                                     (1.18277, -0.81419, 0.37478),
                                     (0.64876, 0.50323, -1.12562),
                                     (0.59336, 1.16044, 0.53412),
                                     (2.51642, -0.38757, 0.25036))}]
        self.assertEqual(len(energies), 2)
        self.assertAlmostEqual(energies[0], 3.49581, 5)
        self.assertAlmostEqual(energies[1], -0.702848, 5)
        # Only symbols instead of the coordinate values are compared.
        # This is due to the unknown behavior of OpenBabel optimization function.
        # With the same iteration number and same initial xyz, the optimized xyzs can
        # be different on different machines, while energies are still consistent.
        # More info can be found from PR #332
        self.assertEqual(xyzs[0]['symbols'], expected_xyzs[0]['symbols'])
        self.assertEqual(xyzs[1]['symbols'], expected_xyzs[1]['symbols'])

    def test_embed_rdkit(self):
        """Test embedding in RDKit"""
        rd_mol = conformers.embed_rdkit(label='CJ', mol=self.cj_spc.mol, num_confs=1)
        xyzs, energies = conformers.rdkit_force_field(label='CJ', rd_mol=rd_mol)
        for atom, symbol in zip(self.cj_spc.mol.atoms, xyzs[0]['symbols']):
            self.assertEqual(atom.symbol, symbol)

    def test_read_rdkit_embedded_conformers(self):
        """Test reading coordinates from embedded RDKit conformers"""
        xyz = """S      -0.19093478    0.57933906    0.00000000
O      -1.21746139   -0.72237602    0.00000000
O       1.40839617    0.14303696    0.00000000"""
        spc = ARCSpecies(label='SO2', smiles='O=S=O', xyz=xyz)
        rd_mol = conformers.embed_rdkit(label='', mol=spc.mol, num_confs=3, xyz=xyz)
        xyzs = conformers.read_rdkit_embedded_conformers(label='', rd_mol=rd_mol)
        expected_xyzs = [{'symbols': ('S', 'O', 'O'),
                          'isotopes': (32, 16, 16),
                          'coords': ((-0.0007230118849883151, 0.4313717594780365, -0.0),
                                     (-1.1690509082499037, -0.21589003940054327, -0.0),
                                     (1.169773920134892, -0.2154817200774931, 0.0))},
                         {'symbols': ('S', 'O', 'O'),
                          'isotopes': (32, 16, 16),
                          'coords': ((-0.0014867645051002793, 0.45116968803294244, -0.0),
                                     (-1.1527224078813536, -0.2260261622412862, -0.0),
                                     (1.154209172386454, -0.2251435257916561, 0.0))},
                         {'symbols': ('S', 'O', 'O'),
                          'isotopes': (32, 16, 16),
                          'coords': ((-0.0021414150293084812, 0.42384518006634525, -0.0),
                                     (-1.176450577057819, -0.21250387455074374, -0.0),
                                     (1.1785919920871275, -0.21134130551560104, 0.0))}]
        self.assertTrue(almost_equal_coords_lists(xyzs, expected_xyzs))

    def test_rdkit_force_field(self):
        """Test embedding molecule and applying force field using RDKit"""
        xyz = """S      -0.19093478    0.57933906    0.00000000
O      -1.21746139   -0.72237602    0.00000000
O       1.40839617    0.14303696    0.00000000"""
        spc = ARCSpecies(label='SO2', smiles='O=S=O', xyz=xyz)
        rd_mol = conformers.embed_rdkit(label='', mol=spc.mol, num_confs=3, xyz=xyz)
        xyzs, energies = conformers.rdkit_force_field(label='', rd_mol=rd_mol,
                                                      force_field='MMFF94s', optimize=True)
        self.assertEqual(len(energies), 3)
        self.assertAlmostEqual(energies[0], 2.8820960262158292e-11, 5)
        self.assertAlmostEqual(energies[1], 4.496464369416183e-14, 5)
        self.assertAlmostEqual(energies[2], 1.8168786624672814e-12, 5)
        expected_xyzs1 = [{'coords': ((-0.048697770520464284, 0.6080446547953217, 0.0),
                                      (-1.353634030198068, -0.41438354841733305, 0.0),
                                      (1.402331800718534, -0.19366110637798933, 0.0)),
                           'isotopes': (32, 16, 16),
                           'symbols': ('S', 'O', 'O')},
                          {'coords': ((-0.05319780617176572, 0.607667542238931, 0.0),
                                      (-1.3505293496269846, -0.42439346057066346, 0.0),
                                      (1.4037271557987503, -0.18327408166827253, 0.0)),
                           'isotopes': (32, 16, 16),
                           'symbols': ('S', 'O', 'O')},
                          {'coords': ((0.07722439890127433, 0.6050837187112847, 0.0),
                                      (-1.4098847335098195, -0.12753188106391833, 0.0),
                                      (1.3326603346085464, -0.4775518376473646, 0.0)),
                           'isotopes': (32, 16, 16),
                           'symbols': ('S', 'O', 'O')}]

        self.assertTrue(almost_equal_coords_lists(xyzs, expected_xyzs1))
        xyzs, energies = conformers.rdkit_force_field(label='', rd_mol=rd_mol,
                                                      force_field='MMFF94s', optimize=False)
        self.assertEqual(len(energies), 0)
        expected_xyzs2 = [{'symbols': ('S', 'O', 'O'),
                           'isotopes': (32, 16, 16),
                           'coords': ((-0.048697770520464284, 0.6080446547953217, 0.0),
                                      (-1.353634030198068, -0.41438354841733305, 0.0),
                                      (1.402331800718534, -0.19366110637798933, 0.0))},
                          {'symbols': ('S', 'O', 'O'),
                           'isotopes': (32, 16, 16),
                           'coords': ((-0.05319780617176572, 0.607667542238931, 0.0),
                                      (-1.3505293496269846, -0.42439346057066346, 0.0),
                                      (1.4037271557987503, -0.18327408166827253, 0.0))},
                          {'symbols': ('S', 'O', 'O'),
                           'isotopes': (32, 16, 16),
                           'coords': ((0.07722439890127433, 0.6050837187112847, 0.0),
                                      (-1.4098847335098195, -0.12753188106391833, 0.0),
                                      (1.3326603346085464, -0.4775518376473646, 0.0))}]
        self.assertTrue(almost_equal_coords_lists(xyzs, expected_xyzs2))

    def test_determine_rotors(self):
        """Test determining the rotors"""
        mol = Molecule(smiles='C=[C]C(=O)O[O]')
        mol_list = mol.generate_resonance_structures()
        torsions, tops = conformers.determine_rotors(mol_list)
        self.assertEqual(torsions, [[3, 1, 4, 2], [1, 4, 6, 5]])
        self.assertEqual(sum(tops[0]), 4)
        self.assertEqual(sum(tops[1]), 10)

        mol_list = [Molecule(smiles='CCCO')]
        torsions, tops = conformers.determine_rotors(mol_list)
        self.assertEqual(torsions, [[5, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 12]])
        self.assertEqual(sum(tops[0]), 19)
        self.assertEqual(sum(tops[1]), 40)

    def test_get_wells(self):
        """Test determining wells characteristics from a list of angles"""
        scan = [-179, -178, -175, -170, -61, -59, -58, -50, -40, -30, -20, -10, 0, 10, 150, 160]
        wells = conformers.get_wells(label='', angles=scan)
        self.assertEqual(wells, [{'angles': [-179, -178, -175, -170],
                                  'end_angle': -170,
                                  'end_idx': 3,
                                  'start_angle': -179,
                                  'start_idx': 0},
                                 {'angles': [-61, -59, -58, -50, -40, -30, -20, -10, 0, 10],
                                  'end_angle': 10,
                                  'end_idx': 13,
                                  'start_angle': -61,
                                  'start_idx': 4},
                                 {'angles': [150, 160],
                                  'end_angle': 160,
                                  'end_idx': 15,
                                  'start_angle': 150,
                                  'start_idx': 14}])

        scan = [-73.68336628556762, -73.6698270329844, -73.23989527267882, -73.21461046998135, -72.98404533147658,
                -72.98164225603858, -69.39937902123044, -69.3988528569368, -67.77744864514945, -67.77148282968275,
                -65.17239618718334, -65.15907633527517, -64.26652813162181, -64.24439959181805, -63.240801195360554,
                -63.23822372040009, -63.1200724631762, -63.11971243825017, -60.63742589032739, -60.63139420286131,
                -60.173003247364996, -60.16784651044193, -59.468854511900325, -59.46786228068227, -58.65720743567386,
                -58.65683432963624, -57.98873394285648, -57.982941836860924, -57.40064081386206, -57.40044768735372,
                -53.74282998560906, -53.73596914102977, -52.68824260862714, -52.68597330918877, -46.27234815136188,
                -46.25545908648955, -45.31808116497048, -45.31451574906208, -37.35933646586181, -37.35852002741229,
                -36.88328657195036, -36.87053094294823, -36.01343709893113, -36.012150772093335, -34.86948467964051,
                -34.8688938780246, -34.72494851508614, -34.72482090531513, 37.92144833495903, 37.998896394653386,
                41.96215337416899, 42.02448094052937, 45.810576693278705, 45.817331523047116, 51.29257628301334,
                51.30311799717333, 51.61623061568205, 51.617786414116125, 53.34477784221306, 53.34580850447212,
                53.48942661659711, 53.4901617185321, 56.47511675765871, 56.47592191783421, 56.89219524846863,
                56.89667826685056, 58.21791320569343, 58.22084771674736, 58.56402491310407, 58.56610158979941,
                58.575585167060105, 58.61170917818044, 58.72432829799178, 58.73704018872237, 58.86716586121682,
                58.86911130502126, 58.88121731495366, 58.88577729197807, 59.080599884436346, 59.08922104350318,
                59.11353250453865, 59.12147007319143, 59.47503346965784, 59.49304004152549, 60.01935141189763,
                60.02668302599606, 60.251501427706465, 60.25315632266764, 60.46989855245081, 60.47493321918853,
                60.682093573955186, 60.70801046425914, 60.720951804151774, 60.7245195946674, 61.25184661219729,
                61.25714897361994, 61.70601073036617, 61.707906267456195, 61.83301219526896, 61.86382363861279,
                62.13779819494127, 62.13972013334631, 62.220744529176095, 62.223284838706526, 62.36544009963831,
                62.3720484546934, 62.81168956759717, 62.83423018985191, 64.9521798738434, 64.9629609319081,
                66.16746748827471, 66.17893910758922, 66.7668740766453, 66.76788523167403, 67.60590251354962,
                67.62187329547093, 68.06080902669052, 68.0842956220337, 68.09623011492323, 68.10199767805355,
                68.48522980414184, 68.48583421208666, 73.26005419282097, 73.26808846126718, 73.45281733280729,
                73.45469759141737, 74.51490380770962, 74.53803845428752, 74.58370293245521, 74.60616218954709,
                83.33528261426564, 83.35937288850262, 83.43016173886006, 83.45479594676438, 158.0577370327542,
                158.0618058607164, 158.9044370338183, 158.9221003782838, 162.31749728503348, 162.3209967086902,
                164.52195355333234, 164.53582069333316, 164.74621421776186, 164.7467225861628, 165.2164723949667,
                165.2265767573193, 168.07838462664859, 168.09950407181762, 168.88464375222418, 168.88580356066808,
                171.689937353885, 171.72548498246235, 178.33869822654015, 178.3814143461112, 178.63816074390414,
                178.69372759467555, 179.27900877194176, 179.28377950998077, 179.5472101554281, 179.5534956989596,
                179.61109850679009, 179.61481948231528, 180.72497299223755, 180.74053436010533, 180.9941404710718,
                181.02077030292295, 181.11974593201793, 181.1243463457858, 181.37177080545533, 181.37611255671746,
                181.4051354501779, 181.41711108890976, 181.48880148090316, 181.49758562851264, 182.26201874998708,
                182.26780045263806, 182.28202738170322, 182.2825980166329, 182.4882342397495, 182.5114677002835,
                183.34205829177768, 183.35856051636569, 187.10490626605906, 187.10934663583262, 191.6284272944037,
                191.6332491563272, 191.65467120190945, 191.65980785427425, 194.12584582734019, 194.1354751568166,
                194.25775316213299, 194.26909817380425, 203.2455591397814, 203.25703331666415, 203.36649309444803,
                203.37095080136055, 205.1767218350291, 205.17998346889013, 205.78470608775663, 205.78904336991963]
        wells = conformers.get_wells(label='', angles=scan)
        self.assertEqual(wells, [{'angles': [-73.68336628556762, -73.6698270329844, -73.23989527267882,
                                             -73.21461046998135, -72.98404533147658, -72.98164225603858,
                                             -69.39937902123044, -69.3988528569368, -67.77744864514945,
                                             -67.77148282968275, -65.17239618718334, -65.15907633527517,
                                             -64.26652813162181, -64.24439959181805, -63.240801195360554,
                                             -63.23822372040009, -63.1200724631762, -63.11971243825017,
                                             -60.63742589032739, -60.63139420286131, -60.173003247364996,
                                             -60.16784651044193, -59.468854511900325, -59.46786228068227,
                                             -58.65720743567386, -58.65683432963624, -57.98873394285648,
                                             -57.982941836860924, -57.40064081386206, -57.40044768735372,
                                             -53.74282998560906, -53.73596914102977, -52.68824260862714,
                                             -52.68597330918877, -46.27234815136188, -46.25545908648955,
                                             -45.31808116497048, -45.31451574906208, -37.35933646586181,
                                             -37.35852002741229, -36.88328657195036, -36.87053094294823,
                                             -36.01343709893113, -36.012150772093335, -34.86948467964051,
                                             -34.8688938780246, -34.72494851508614, -34.72482090531513],
                                  'end_angle': -34.72482090531513,
                                  'end_idx': 47,
                                  'start_angle': -73.68336628556762,
                                  'start_idx': 0},
                                 {'angles': [37.92144833495903, 37.998896394653386, 41.96215337416899,
                                             42.02448094052937, 45.810576693278705, 45.817331523047116,
                                             51.29257628301334, 51.30311799717333, 51.61623061568205,
                                             51.617786414116125, 53.34477784221306, 53.34580850447212,
                                             53.48942661659711, 53.4901617185321, 56.47511675765871,
                                             56.47592191783421, 56.89219524846863, 56.89667826685056,
                                             58.21791320569343, 58.22084771674736, 58.56402491310407,
                                             58.56610158979941, 58.575585167060105, 58.61170917818044,
                                             58.72432829799178, 58.73704018872237, 58.86716586121682,
                                             58.86911130502126, 58.88121731495366, 58.88577729197807,
                                             59.080599884436346, 59.08922104350318, 59.11353250453865,
                                             59.12147007319143, 59.47503346965784, 59.49304004152549,
                                             60.01935141189763, 60.02668302599606, 60.251501427706465,
                                             60.25315632266764, 60.46989855245081, 60.47493321918853,
                                             60.682093573955186, 60.70801046425914, 60.720951804151774,
                                             60.7245195946674, 61.25184661219729, 61.25714897361994,
                                             61.70601073036617, 61.707906267456195, 61.83301219526896,
                                             61.86382363861279, 62.13779819494127, 62.13972013334631,
                                             62.220744529176095, 62.223284838706526, 62.36544009963831,
                                             62.3720484546934, 62.81168956759717, 62.83423018985191,
                                             64.9521798738434, 64.9629609319081, 66.16746748827471,
                                             66.17893910758922, 66.7668740766453, 66.76788523167403,
                                             67.60590251354962, 67.62187329547093, 68.06080902669052,
                                             68.0842956220337, 68.09623011492323, 68.10199767805355,
                                             68.48522980414184, 68.48583421208666, 73.26005419282097,
                                             73.26808846126718, 73.45281733280729, 73.45469759141737,
                                             74.51490380770962, 74.53803845428752, 74.58370293245521,
                                             74.60616218954709, 83.33528261426564, 83.35937288850262,
                                             83.43016173886006, 83.45479594676438],
                                  'end_angle': 83.45479594676438,
                                  'end_idx': 133,
                                  'start_angle': 37.92144833495903,
                                  'start_idx': 48},
                                 {'angles': [158.0577370327542, 158.0618058607164, 158.9044370338183,
                                             158.9221003782838, 162.31749728503348, 162.3209967086902,
                                             164.52195355333234, 164.53582069333316, 164.74621421776186,
                                             164.7467225861628, 165.2164723949667, 165.2265767573193,
                                             168.07838462664859, 168.09950407181762, 168.88464375222418,
                                             168.88580356066808, 171.689937353885, 171.72548498246235,
                                             178.33869822654015, 178.3814143461112, 178.63816074390414,
                                             178.69372759467555, 179.27900877194176, 179.28377950998077,
                                             179.5472101554281, 179.5534956989596, 179.61109850679009,
                                             179.61481948231528, 180.72497299223755, 180.74053436010533,
                                             180.9941404710718, 181.02077030292295, 181.11974593201793,
                                             181.1243463457858, 181.37177080545533, 181.37611255671746,
                                             181.4051354501779, 181.41711108890976, 181.48880148090316,
                                             181.49758562851264, 182.26201874998708, 182.26780045263806,
                                             182.28202738170322, 182.2825980166329, 182.4882342397495,
                                             182.5114677002835, 183.34205829177768, 183.35856051636569,
                                             187.10490626605906, 187.10934663583262, 191.6284272944037,
                                             191.6332491563272, 191.65467120190945, 191.65980785427425,
                                             194.12584582734019, 194.1354751568166, 194.25775316213299,
                                             194.26909817380425, 203.2455591397814, 203.25703331666415,
                                             203.36649309444803, 203.37095080136055, 205.1767218350291,
                                             205.17998346889013, 205.78470608775663, 205.78904336991963],
                                  'end_angle': 205.78904336991963,
                                  'end_idx': 199,
                                  'start_angle': 158.0577370327542,
                                  'start_idx': 134}])

    def test_determine_dihedrals(self):
        """Test determining the dihedrals in a molecule"""
        confs = list()
        rd_xyzs, rd_energies = conformers.get_force_field_energies(label='', mol=self.mol0, num_confs=10)
        for xyz, energy in zip(rd_xyzs, rd_energies):
            confs.append({'xyz': xyz,
                          'FF energy': energy,
                          'molecule': self.mol0,
                          'source': 'RDKit'})
        self.spc0.determine_rotors()
        torsions = [rotor_dict['scan'] for rotor_dict in self.spc0.rotors_dict.values()]
        confs = conformers.determine_dihedrals(confs, torsions)
        self.assertAlmostEqual(confs[0]['torsion_dihedrals'][(9, 1, 2, 3)], 300.14463, 3)
        self.assertAlmostEqual(confs[0]['torsion_dihedrals'][(1, 2, 3, 6)], 301.81907, 3)

    def test_check_special_non_rotor_cases(self):
        """Test that special cases (cyano and azide groups) are not considered as rotors"""
        adj0 = """1 N u0 p1 c0 {3,T}
2 C u0 p0 c0 {3,S} {4,S} {5,S} {6,S}
3 C u0 p0 c0 {2,S} {1,T}
4 H u0 p0 c0 {2,S}
5 H u0 p0 c0 {2,S}
6 H u0 p0 c0 {2,S}"""
        adj1 = """1 C u0 p0 c0 {2,S} {5,S} {6,S} {7,S}
2 N u0 p2 c-1 {1,S} {3,S}
3 N u0 p0 c+1 {2,S} {4,T}
4 N u0 p1 c0 {3,T}
5 H u0 p0 c0 {1,S}
6 H u0 p0 c0 {1,S}
7 H u0 p0 c0 {1,S}"""
        adj2 = """1 C u0 p0 c0 {2,S} {4,S} {5,S} {6,S}
2 C u0 p0 c0 {1,S} {3,T}
3 C u0 p0 c0 {2,T} {7,S}
4 H u0 p0 c0 {1,S}
5 H u0 p0 c0 {1,S}
6 H u0 p0 c0 {1,S}
7 H u0 p0 c0 {3,S}"""
        adj3 = """1 C u0 p0 c0 {2,S} {3,S} {4,S} {5,S}
2 C u0 p0 c0 {1,S} {6,S} {7,S} {8,S}
3 H u0 p0 c0 {1,S}
4 H u0 p0 c0 {1,S}
5 H u0 p0 c0 {1,S}
6 H u0 p0 c0 {2,S}
7 H u0 p0 c0 {2,S}
8 H u0 p0 c0 {2,S}"""
        mol0 = Molecule().from_adjacency_list(adj0)
        mol1 = Molecule().from_adjacency_list(adj1)
        mol2 = Molecule().from_adjacency_list(adj2)
        mol3 = Molecule().from_adjacency_list(adj3)
        non_rotor0 = conformers.check_special_non_rotor_cases(mol=mol0, top1=[3, 1], top2=[2, 4, 5, 6])
        non_rotor1 = conformers.check_special_non_rotor_cases(mol=mol1, top1=[2, 1, 5, 6, 7], top2=[3, 4])
        non_rotor2 = conformers.check_special_non_rotor_cases(mol=mol2, top1=[1, 4, 5, 6], top2=[2, 3, 7])
        rotor3 = conformers.check_special_non_rotor_cases(mol=mol3, top1=[1, 3, 4, 5], top2=[2, 6, 7, 8])
        self.assertTrue(non_rotor0)
        self.assertTrue(non_rotor1)
        self.assertTrue(non_rotor2)
        self.assertFalse(rotor3)

        spc0 = ARCSpecies(label='spc0', mol=mol0)
        spc1 = ARCSpecies(label='spc1', mol=mol1)
        spc2 = ARCSpecies(label='spc2', mol=mol2)

        spc0.determine_rotors()
        spc1.determine_rotors()
        spc2.determine_rotors()

        torsions0 = [rotor_dict['scan'] for rotor_dict in spc0.rotors_dict.values()]
        self.assertFalse(torsions0)

        torsions1 = [rotor_dict['scan'] for rotor_dict in spc1.rotors_dict.values()]
        self.assertEqual(len(torsions1), 1)  # expecting only the CH3 rotor

        torsions2 = [rotor_dict['scan'] for rotor_dict in spc2.rotors_dict.values()]
        self.assertFalse(len(torsions2))

        mol4 = Molecule(smiles='c1ccccc1C#C')
        rotors = conformers.find_internal_rotors(mol4)
        self.assertFalse(len(rotors))

        mol5 = Molecule(smiles='N#CCC')
        rotors = conformers.find_internal_rotors(mol5)
        self.assertEqual(len(rotors), 1)

    def test_find_internal_rotors(self):
        """Test finding internal rotors in a molecule list"""
        mol_list = self.mol1.generate_resonance_structures()
        rotors = []
        for mol in mol_list:
            rotors.append(conformers.find_internal_rotors(mol))
        self.assertEqual(len(rotors[0]), 1)
        self.assertEqual(len(rotors[1]), 2)
        self.assertEqual(rotors[0][0]['pivots'], [2, 3])
        self.assertEqual(rotors[1][0]['pivots'], [2, 3])
        self.assertEqual(rotors[1][1]['pivots'], [3, 4])
        self.assertTrue('invalidation_reason' in rotors[0][0])
        self.assertTrue('pivots' in rotors[0][0])
        self.assertTrue('scan' in rotors[0][0])
        self.assertTrue('scan_path' in rotors[0][0])
        self.assertTrue('success' in rotors[0][0])
        self.assertTrue('times_dihedral_set' in rotors[0][0])
        self.assertTrue('top' in rotors[0][0])

        cj_adj = """1  C u0 p0 c0 {2,S} {37,S} {38,S} {39,S}
2  O u0 p2 c0 {1,S} {3,S}
3  C u0 p0 c0 {2,S} {4,D} {11,S}
4  C u0 p0 c0 {3,D} {5,S} {12,S}
5  C u0 p0 c0 {4,S} {6,D} {40,S}
6  C u0 p0 c0 {5,D} {7,S} {10,S}
7  C u0 p0 c0 {6,S} {8,S} {9,S} {41,S}
8  C u0 p0 c0 {7,S} {42,S} {43,S} {44,S}
9  C u0 p0 c0 {7,S} {45,S} {46,S} {47,S}
10 C u0 p0 c0 {6,S} {11,D} {48,S}
11 C u0 p0 c0 {3,S} {10,D} {49,S}
12 C u0 p0 c0 {4,S} {13,S} {50,S} {51,S}
13 N u0 p1 c0 {12,S} {14,S} {52,S}
14 C u0 p0 c0 {13,S} {15,S} {16,S} {20,S}
15 H u0 p0 c0 {14,S}
16 C u0 p0 c0 {14,S} {17,S} {36,S} {53,S}
17 C u0 p0 c0 {16,S} {18,S} {54,S} {55,S}
18 C u0 p0 c0 {17,S} {19,S} {56,S} {57,S}
19 N u0 p1 c0 {18,S} {20,S} {35,S}
20 C u0 p0 c0 {14,S} {19,S} {21,S} {22,S}
21 H u0 p0 c0 {20,S}
22 C u0 p0 c0 {20,S} {23,S} {29,S} {58,S}
23 C u0 p0 c0 {22,S} {24,D} {28,S}
24 C u0 p0 c0 {23,D} {25,S} {59,S}
25 C u0 p0 c0 {24,S} {26,D} {60,S}
26 C u0 p0 c0 {25,D} {27,S} {61,S}
27 C u0 p0 c0 {26,S} {28,D} {62,S}
28 C u0 p0 c0 {23,S} {27,D} {63,S}
29 C u0 p0 c0 {22,S} {30,D} {34,S}
30 C u0 p0 c0 {29,D} {31,S} {64,S}
31 C u0 p0 c0 {30,S} {32,D} {65,S}
32 C u0 p0 c0 {31,D} {33,S} {66,S}
33 C u0 p0 c0 {32,S} {34,D} {67,S}
34 C u0 p0 c0 {29,S} {33,D} {68,S}
35 C u0 p0 c0 {19,S} {36,S} {69,S} {70,S}
36 C u0 p0 c0 {16,S} {35,S} {71,S} {72,S}
37 H u0 p0 c0 {1,S}
38 H u0 p0 c0 {1,S}
39 H u0 p0 c0 {1,S}
40 H u0 p0 c0 {5,S}
41 H u0 p0 c0 {7,S}
42 H u0 p0 c0 {8,S}
43 H u0 p0 c0 {8,S}
44 H u0 p0 c0 {8,S}
45 H u0 p0 c0 {9,S}
46 H u0 p0 c0 {9,S}
47 H u0 p0 c0 {9,S}
48 H u0 p0 c0 {10,S}
49 H u0 p0 c0 {11,S}
50 H u0 p0 c0 {12,S}
51 H u0 p0 c0 {12,S}
52 H u0 p0 c0 {13,S}
53 H u0 p0 c0 {16,S}
54 H u0 p0 c0 {17,S}
55 H u0 p0 c0 {17,S}
56 H u0 p0 c0 {18,S}
57 H u0 p0 c0 {18,S}
58 H u0 p0 c0 {22,S}
59 H u0 p0 c0 {24,S}
60 H u0 p0 c0 {25,S}
61 H u0 p0 c0 {26,S}
62 H u0 p0 c0 {27,S}
63 H u0 p0 c0 {28,S}
64 H u0 p0 c0 {30,S}
65 H u0 p0 c0 {31,S}
66 H u0 p0 c0 {32,S}
67 H u0 p0 c0 {33,S}
68 H u0 p0 c0 {34,S}
69 H u0 p0 c0 {35,S}
70 H u0 p0 c0 {35,S}
71 H u0 p0 c0 {36,S}
72 H u0 p0 c0 {36,S}
"""
        cj_mol = Molecule().from_adjacency_list(adjlist=cj_adj)
        rotors = conformers.find_internal_rotors(cj_mol)
        for rotor in rotors:
            if rotor['pivots'] == [6, 7]:
                self.assertEqual(rotor['scan'], [5, 6, 7, 8])
                self.assertEqual(sum(rotor['top']), 332)  # [7,41,8,44,42,43,9,46,47,45] in non-deterministic order

        mol = Molecule(smiles='CCCO')
        rotors = conformers.find_internal_rotors(mol)
        for rotor in rotors:
            self.assertIn(rotor['scan'], [[5, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 12]])

    def test_to_group(self):
        """Test converting a part of a molecule into a group"""
        atom_indices = [0, 3, 8]
        group0 = conformers.to_group(mol=self.mol1, atom_indices=atom_indices)

        atom0 = GroupAtom(atomtype=[ATOMTYPES['Cd']], radical_electrons=[0], charge=[0],
                          label='', lone_pairs=[0])
        atom1 = GroupAtom(atomtype=[ATOMTYPES['O2s']], radical_electrons=[1], charge=[0],
                          label='', lone_pairs=[2])
        atom2 = GroupAtom(atomtype=[ATOMTYPES['H']], radical_electrons=[0], charge=[0],
                          label='', lone_pairs=[0])
        group1 = Group(atoms=[atom0, atom1, atom2], multiplicity=[2])
        bond01 = GroupBond(atom0, atom1, order=[1.0])
        bond02 = GroupBond(atom0, atom2, order=[1.0])
        group1.add_bond(bond01)
        group1.add_bond(bond02)

        self.assertTrue(group0.is_isomorphic(group1))

    def test_get_torsion_angles(self):
        """Test determining the torsion angles from all conformers"""
        torsions = conformers.determine_rotors(self.spc0.mol_list)[0]
        confs = [{'FF energy': -1.5170975770157071,
                  'molecule': self.mol0,
                  'source': 'RDKit',
                  'xyz': 'O       1.22700646   -0.74306134   -0.46642912\nC       0.44275447    0.24386237    '
                         '0.18670577\nC      -1.02171998   -0.04112700   -0.06915927\nH       0.65837353    '
                         '0.21260997    1.25889575\nH       0.71661793    1.22791280   -0.20534525\nH      '
                         '-1.29179957   -1.03551511    0.30144722\nH      -1.23432965   -0.03370364   -1.14332936\n'
                         'H      -1.65581027    0.70264687    0.42135230\nH       2.15890709   -0.53362491   '
                         '-0.28413803\n'},
                 {'FF energy': -1.336857063940798,
                  'molecule': self.mol0,
                  'source': 'RDKit',
                  'xyz': 'O       1.45741739    0.61020598   -0.00386149\nC       0.49432210   -0.39620513   '
                         '-0.28042773\nC      -0.89596057    0.11891813    0.02906668\nH       0.57628556   '
                         '-0.66552902   -1.33744779\nH       0.72801279   -1.28050031    0.32015417\n'
                         'H      -1.11282223    1.01829132   -0.55660310\nH      -0.97906607    0.39858005    '
                         '1.08445491\nH      -1.65147628   -0.63942945   -0.19448738\nH       1.38328732    '
                         '0.83566844    0.93915173\n'},
                 {'FF energy': -1.3368570629495138,
                  'molecule': self.mol0,
                  'source': 'RDKit',
                  'xyz': 'O      -1.48027320    0.36597456    0.41386552\nC      -0.49770656   -0.40253648   '
                         '-0.26500019\nC       0.86215119    0.24734211   -0.11510338\nH      -0.77970114   '
                         '-0.46128090   -1.32025907\nH      -0.49643724   -1.41548311    0.14879346\nH       '
                         '0.84619526    1.26924854   -0.50799415\nH       1.14377239    0.31659216    0.94076336\n'
                         'H       1.62810781   -0.32407050   -0.64676910\nH      -1.22610851    0.40421362    '
                         '1.35170355\n'}]
        confs = conformers.determine_dihedrals(confs, torsions)
        torsion_angles = conformers.get_torsion_angles(label='', conformers=confs, torsions=torsions)
        self.assertEqual(torsions[0], [9, 1, 2, 3])
        self.assertEqual(torsions[1], [1, 2, 3, 6])
        self.assertTrue(all([int(round(angle / 5.0) * 5.0) in [60, 180, 300]
                             for angle in torsion_angles[tuple(torsions[0])]]))  # batch check almost equal
        self.assertTrue(all([int(round(angle / 5.0) * 5.0) in [60, 300]
                             for angle in torsion_angles[tuple(torsions[1])]]))  # batch check almost equal

    def test_determine_torsion_symmetry(self):
        """Test that we correctly determine the torsion symmetry"""
        adj0 = """1 O u0 p2 c0 {2,S} {9,S}
2 C u0 p0 c0 {1,S} {3,S} {4,S} {5,S}
3 C u0 p0 c0 {2,S} {6,S} {7,S} {8,S}
4 H u0 p0 c0 {2,S}
5 H u0 p0 c0 {2,S}
6 H u0 p0 c0 {3,S}
7 H u0 p0 c0 {3,S}
8 H u0 p0 c0 {3,S}
9 H u0 p0 c0 {1,S}"""
        mol0 = Molecule().from_adjacency_list(adj0)
        torsion_scan0 = [-179.5316509266246, -179.5316509266246, -179.5316509266246, -179.5316509266246,
                         -179.5316509266246, -179.5316509266246, -179.5316509266246, -179.5316509266246,
                         -60.11816797679059, -60.11810624433042, -60.11804469757614, -58.18097368133174,
                         -58.18045845481007, 58.18063730000221, 58.18087223853245, 58.18090794745689, 60.1180249902725,
                         60.11822550406657, 179.06251373730467, 179.0631045514683, 179.06310597052072,
                         179.0633554023178, 179.99994796571877, 179.99999648225656]
        top0 = [3, 6, 7, 8]
        self.assertEqual(conformers.determine_torsion_symmetry(label='', top1=top0, mol_list=[mol0],
                                                               torsion_scan=torsion_scan0), 3)

        # both rotors are symmetric with symmetry numbers of 2 and 3
        mol1 = Molecule(smiles='CC[N+](=O)[O-]')
        mol1.update()
        torsions, tops = conformers.determine_rotors([mol1])
        confs = conformers.generate_force_field_conformers(mol_list=[mol1], label='mol1', num_confs=50,
                                                           torsion_num=len(torsions), charge=0, multiplicity=1)
        with self.assertRaises(ConformerError):
            conformers.get_torsion_angles(label='', conformers=confs, torsions=torsions)
        confs = conformers.determine_dihedrals(conformers=confs, torsions=torsions)
        torsion_angles = conformers.get_torsion_angles(label='', conformers=confs, torsions=torsions)
        self.assertEqual(conformers.determine_torsion_symmetry(label='', top1=tops[0], mol_list=[mol1],
                                                               torsion_scan=torsion_angles[tuple(torsions[0])]), 2)
        self.assertEqual(conformers.determine_torsion_symmetry(label='', top1=tops[1], mol_list=[mol1],
                                                               torsion_scan=torsion_angles[tuple(torsions[1])]), 3)

        # only one rotor is symmetric
        mol2 = Molecule(smiles='CC[N+](=S)[O-]')
        mol2.update()
        torsions, tops = conformers.determine_rotors([mol2])
        confs = conformers.generate_force_field_conformers(mol_list=[mol2], label='mol2', num_confs=50,
                                                           torsion_num=len(torsions), charge=0, multiplicity=1)
        confs = conformers.determine_dihedrals(conformers=confs, torsions=torsions)
        torsion_angles = conformers.get_torsion_angles(label='', conformers=confs, torsions=torsions)
        # NSO rotor:
        self.assertEqual(conformers.determine_torsion_symmetry(label='', top1=tops[0], mol_list=[mol2],
                                                               torsion_scan=torsion_angles[tuple(torsions[0])]), 1)
        # CH3 rotor:
        self.assertEqual(conformers.determine_torsion_symmetry(label='', top1=tops[1], mol_list=[mol2],
                                                               torsion_scan=torsion_angles[tuple(torsions[1])]), 3)

        # The COH rotor is symmetric because of the bottom of the molecule
        mol3 = Molecule(smiles='c1ccccc1C(c1ccccc1)(c1ccccc1)O')
        mol3.update()
        torsions, tops = conformers.determine_rotors([mol3])
        confs = conformers.generate_force_field_conformers(mol_list=[mol3], label='mol3', num_confs=100,
                                                           torsion_num=len(torsions), charge=0, multiplicity=1)
        confs = conformers.determine_dihedrals(conformers=confs, torsions=torsions)
        torsion_angles = conformers.get_torsion_angles(label='', conformers=confs, torsions=torsions)
        self.assertEqual(conformers.determine_torsion_symmetry(label='', top1=tops[1], mol_list=[mol3],
                                                               torsion_scan=torsion_angles[tuple(torsions[1])]), 2)
        self.assertEqual(conformers.determine_torsion_symmetry(label='', top1=tops[2], mol_list=[mol3],
                                                               torsion_scan=torsion_angles[tuple(torsions[2])]), 2)
        self.assertEqual(conformers.determine_torsion_symmetry(label='', top1=tops[3], mol_list=[mol3],
                                                               torsion_scan=torsion_angles[tuple(torsions[3])]), 2)

        mol4 = Molecule(smiles='c1ccccc1CO')
        mol4.update()
        torsions, tops = conformers.determine_rotors([mol4])
        confs = conformers.generate_force_field_conformers(mol_list=[mol4], label='mol4', num_confs=100,
                                                           torsion_num=len(torsions), charge=0, multiplicity=1)
        confs = conformers.determine_dihedrals(conformers=confs, torsions=torsions)
        torsion_angles = conformers.get_torsion_angles(label='', conformers=confs, torsions=torsions)
        self.assertTrue(any([conformers.determine_torsion_symmetry(label='', top1=tops[i], mol_list=[mol4],
                                                                   torsion_scan=torsion_angles[tuple(torsions[i])])
                             == 2 for i in range(2)]))

        mol5 = Molecule(smiles='OCC(C)C')
        mol5.update()
        torsions, tops = conformers.determine_rotors([mol5])
        confs = conformers.generate_force_field_conformers(mol_list=[mol5], label='mol5', num_confs=100,
                                                           torsion_num=len(torsions), charge=0, multiplicity=1)
        confs = conformers.determine_dihedrals(conformers=confs, torsions=torsions)
        torsion_angles = conformers.get_torsion_angles(label='', conformers=confs, torsions=torsions)
        self.assertEqual(len(torsions), 4)
        self.assertEqual(sum([conformers.determine_torsion_symmetry(label='', top1=tops[i], mol_list=[mol5],
                                                                    torsion_scan=torsion_angles[tuple(torsions[i])])
                              for i in range(len(torsions))]), 8)

        mol7 = Molecule(smiles='CC')
        mol7.update()
        torsions, tops = conformers.determine_rotors([mol7])
        confs = conformers.generate_force_field_conformers(mol_list=[mol7], label='mol7', num_confs=50,
                                                           torsion_num=len(torsions), charge=0, multiplicity=1)
        confs = conformers.determine_dihedrals(conformers=confs, torsions=torsions)
        torsion_angles = conformers.get_torsion_angles(label='', conformers=confs, torsions=torsions)
        self.assertEqual(len(torsions), 1)
        self.assertEqual(conformers.determine_torsion_symmetry(label='', top1=tops[0], mol_list=[mol7],
                                                               torsion_scan=torsion_angles[tuple(torsions[0])]), 9)

        mol8 = Molecule(smiles='C[N+](=O)[O-]')
        mol8.update()
        torsions, tops = conformers.determine_rotors([mol8])
        confs = conformers.generate_force_field_conformers(mol_list=[mol8], label='mol8', num_confs=200,
                                                           torsion_num=len(torsions), charge=0, multiplicity=1)
        confs = conformers.determine_dihedrals(conformers=confs, torsions=torsions)
        torsion_angles = conformers.get_torsion_angles(label='', conformers=confs, torsions=torsions)
        self.assertEqual(len(torsions), 1)
        self.assertEqual(conformers.determine_torsion_symmetry(label='', top1=tops[0], mol_list=[mol8],
                                                               torsion_scan=torsion_angles[tuple(torsions[0])]), 6)

        mol9 = Molecule(smiles='Cc1ccccc1')
        mol9.update()
        torsions, tops = conformers.determine_rotors([mol9])
        confs = conformers.generate_force_field_conformers(mol_list=[mol9], label='mol9', num_confs=50,
                                                           torsion_num=len(torsions), charge=0, multiplicity=1)
        confs = conformers.determine_dihedrals(conformers=confs, torsions=torsions)
        torsion_angles = conformers.get_torsion_angles(label='', conformers=confs, torsions=torsions)
        self.assertEqual(len(torsions), 1)
        self.assertEqual(conformers.determine_torsion_symmetry(label='', top1=tops[0], mol_list=[mol9],
                                                               torsion_scan=torsion_angles[tuple(torsions[0])]), 6)

        mol10 = Molecule(smiles='O[CH2]')
        mol10.update()
        torsions, tops = conformers.determine_rotors([mol10])
        confs = conformers.generate_force_field_conformers(mol_list=[mol10], label='mol10', num_confs=50,
                                                           torsion_num=len(torsions), charge=0, multiplicity=2)
        confs = conformers.determine_dihedrals(conformers=confs, torsions=torsions)
        torsion_angles = conformers.get_torsion_angles(label='', conformers=confs, torsions=torsions)
        self.assertEqual(len(torsions), 1)
        self.assertEqual(conformers.determine_torsion_symmetry(label='', top1=tops[0], mol_list=[mol10],
                                                               torsion_scan=torsion_angles[tuple(torsions[0])]), 2)

        # test symmetry for a torsion involving a benzene ring
        spc1 = ARCSpecies(label='aniline', smiles='Nc1ccccc1')
        torsions, tops = conformers.determine_rotors(spc1.mol_list)
        confs = conformers.generate_force_field_conformers(mol_list=spc1.mol_list, label='aniline', num_confs=50,
                                                           torsion_num=len(torsions), charge=0, multiplicity=1)
        confs = conformers.determine_dihedrals(conformers=confs, torsions=torsions)
        torsion_angles = conformers.get_torsion_angles(label='aniline', conformers=confs, torsions=torsions)
        symmetry = conformers.determine_torsion_symmetry(label='aniline', top1=tops[0], mol_list=spc1.mol_list,
                                                         torsion_scan=torsion_angles[tuple(torsions[0])])
        sampling_points, wells = conformers.determine_torsion_sampling_points(
            label='aniline', torsion_angles=list(torsion_angles.values())[0], symmetry=symmetry)
        self.assertEqual(len(sampling_points), 1)
        self.assertEqual(len(wells), 2)

    def test_determine_torsion_sampling_points(self):
        """test determining torsion sampling points"""
        torsion_angles = [10, 150, 160, 181, 182, 185, 190, 299, 301, 302, 310, 320, 330, 340, 350, 360]
        sampling_points, wells = conformers.determine_torsion_sampling_points(label='', torsion_angles=torsion_angles)
        expected_sampling_points = [155.0, 184.5, 313.2, 343.2]
        for i, entry in enumerate(sampling_points):
            self.assertAlmostEqual(entry, expected_sampling_points[i])
        self.assertEqual(wells,
                         [{'start_idx': 0, 'end_idx': 1, 'start_angle': 150, 'end_angle': 160, 'angles': [150, 160]},
                          {'start_idx': 2, 'end_idx': 5, 'start_angle': 181, 'end_angle': 190,
                           'angles': [181, 182, 185, 190]},
                          {'start_idx': 6, 'end_idx': 15, 'start_angle': 299, 'end_angle': 370,
                           'angles': [299, 301, 302, 310, 320, 330, 340, 350, 360, 370]}])

        torsion_angles = [0, 10, 150, 160, 181, 182, 185, 190, 299, 301, 302, 310, 320, 330, 340, 350]
        sampling_points = conformers.determine_torsion_sampling_points(
            label='', torsion_angles=torsion_angles, symmetry=3)[0]
        self.assertEqual(sampling_points, [155.0])

        torsion_angles = [76.44226785146085, 76.44226796206006, 78.8000414071327, 78.80248169737817, 97.86638633387129,
                          97.86690688613884, 101.5620159123216, 101.56211964495982, 101.60273917268496,
                          101.62220693019022, 167.9999696197525, 168.015496963019, 173.50361410674796,
                          173.50361428302764, 179.87419886059158, 179.87466884415952, 180.40790487151293,
                          180.40960625182666, 181.4618139310557, 181.4618139310557, 259.9786638343785,
                          259.9879721883361, 262.42991601908506, 262.4343645429452, 272.77041649599323,
                          272.7712694868169, 277.236254373247, 277.23718191509414, 282.67947978189875,
                          282.68320790650165, 283.6851961459463, 283.68520119431787, 284.2591371701531,
                          284.2593431730432, 285.5500256417829, 285.5549586681715, 288.27884019164287,
                          288.3144235971651, 290.7904254426122, 290.7904490651266]

        sampling_points = conformers.determine_torsion_sampling_points(label='', torsion_angles=torsion_angles)[0]
        self.assertEqual(sampling_points, [91.25694337981986, 176.65127016627497, 278.7688363417129])

        torsion_angles = [88.10464521993997, 95.68946323823499, 95.69459080513944, 97.33055063592927, 97.33445482833133,
                          97.49952914172731, 97.49958226697981, 108.85221511940138, 112.13184073171584,
                          112.1329020043724, 113.43204366643745, 113.43508423709228, 219.06299556385207,
                          236.63467930635943, 236.63806547008573, 248.56723166099772, 251.14996197507145,
                          251.1937115885355, 253.2519118139127, 253.25193018843223, 259.0821271105939,
                          259.10901129617685, 259.1280227306398, 259.1329899507279, 259.9207311502777,
                          259.9265997028191, 261.1447417086918, 261.14907932120946, 262.93322577568483,
                          262.9400097961174, 265.08121133145704, 266.0095202772411, 266.0128551733065,
                          279.16046334807686, 279.2014185247318, 286.6025958554289, 286.6032721934276,
                          293.97652065163896, 297.0524153609526, 297.0539875926821]
        sampling_points = conformers.determine_torsion_sampling_points(label='', torsion_angles=torsion_angles)[0]
        self.assertEqual(sampling_points, [102.4280751579418, 248.2489745149689, 278.2489745149689])

        torsion_angles = [25.809606429720823, 48.25078667982481, 48.27271314406701, 50.17085636345509,
                          50.171666700725766, 54.825786971684956, 54.8270959675416, 54.82801972678648,
                          54.83998709758166, 55.217356366890705, 55.222640628360544, 55.63284261357208,
                          55.63470843229376, 56.06991437742509, 56.072010852678616, 56.16268683800219,
                          56.17532397584247, 57.563521896714526, 57.56379561118491, 57.58220162436901,
                          57.58238054823384, 57.773083571358136, 57.77759654244658, 58.02807739899398, 58.0366638254267,
                          58.6602605445575, 58.8186647391465, 59.078472206071325, 59.08343206068942, 59.13088372662295,
                          59.13833825100024, 61.16787507452899, 61.19150600898746, 61.46146030387509, 61.48207696062333,
                          62.00611886241341, 62.007061944422155, 62.302417977239266, 62.30258831738516,
                          62.304595670178955, 62.30752051958369, 62.3928572433257, 62.394690027161005,
                          62.39507804066775, 62.40761950553415, 62.43300411487009, 62.438690527218625,
                          62.84124026732686, 62.8506486485056, 63.19285387679225, 63.19421010254325, 63.75017305004417,
                          63.7647047937204, 63.821893406046755, 63.84121559550037, 63.848486286450935,
                          63.87463519698978, 64.09430722365335, 64.09441331439972, 64.2455888373737, 64.24757421619192,
                          64.30553145396728, 64.30574858897346, 64.91706498924965, 64.93292069662333, 64.94411171945082,
                          64.94469126990732, 65.10560181105447, 65.14520620184406, 65.15039226448064, 65.15873721160678,
                          65.17031707742622, 65.20502692437536, 65.75025128293476, 65.7518510416996, 67.48435748370368,
                          67.48993483575366, 67.78223180118508, 67.79865159086725, 68.57163156828182, 68.58278969476429,
                          68.9835342681565, 68.98502489257125, 69.02485893048764, 69.02531831229979, 69.30083134611509,
                          69.32639761608996, 69.78727996013161, 69.79003948478928, 69.81646328793155, 69.81650466417702,
                          70.75457219440229, 70.7711916067069, 70.79267671910102, 70.7944938028614, 70.79540510667567,
                          70.94549462058072, 70.9531816227119, 71.07947581077214, 72.10598479387446, 72.106971801984,
                          72.14765977159695, 72.15631579934205, 72.15809553596962, 72.16477284932475, 72.55513044781213,
                          72.55524795156039, 75.28412012313748, 75.55541153577926, 75.56300919137861, 76.39916569514754,
                          76.40012661444992, 77.50634180413392, 77.6575588981426, 78.0301542193255, 78.03303193448812,
                          78.78787044406549, 78.79082915408495, 78.80116145781852, 78.82627274476296, 79.10247828055287,
                          79.12271367210117, 79.4377244591139, 79.45142875982222, 79.4782352817987, 79.47896782421039,
                          79.47949360294194, 79.47952864923629, 79.80643678517221, 79.81306411762233, 81.98172874001276,
                          81.98685806299791, 89.73711444222226, 89.74110812758445, 91.04881077542787, 91.04924826404962,
                          91.60403875417842, 91.60672185467155, 92.13554278700374, 92.15206547263318, 93.5653324027846,
                          93.56591614818197, 93.56718452183638, 93.56880449913933, 94.17783361803986, 94.19514712640881,
                          95.54189421759462, 95.54196730280678, 99.45620288193062, 99.46318922816309, 99.4783063650388,
                          99.4847190781842, 99.48634425879055, 101.65124311361308, 101.65178162868837,
                          102.85679501819145, 102.85872102229916, 102.8597074345882, 102.85995894555535,
                          104.86907002547052, 104.90241841732139, 105.44836556399554, 105.45349950525973,
                          105.45748075964613, 105.46428509851808, 105.8141771701505, 105.81833520271985,
                          106.61321552233747, 106.61497971551212, 108.55192687444472, 108.55315365741819,
                          108.55393853803722, 108.5547410527443, 115.08686295289823, 115.0967600690075,
                          130.84084515283902, 130.84104663344476, 130.878423658668, 130.88412149967814,
                          132.52441710224534, 132.5302238969977, 132.5338586978291, 132.5357908447517,
                          138.3415593476458, 138.34462568197915, 155.72658039999197, 155.8138698964472,
                          156.1174332413083, 157.60010974730972, 158.46355864481313, 158.4843791142827,
                          164.91378968522747, 164.91381098080547, 168.67215306371415, 168.67217353280967,
                          168.8914569801022, 168.90350649651364, 168.98439201781522, 169.04580022622054,
                          173.35440894486018, 173.3630011494851, 183.67632213503813, 183.67633243487035,
                          187.58160465560428, 187.58224511178042, 189.08027971262942, 189.08315337855532,
                          191.32634064909385, 191.3505176680047, 202.6621344302995, 202.6696821390786,
                          229.11888110411834, 229.1198966779721, 236.96557507408176, 236.96745073942841,
                          253.37962674865196, 253.38508701631037, 253.50629644566735, 253.50937496025952,
                          253.55184690722564, 253.5533844348513, 253.8815946050616, 253.97643265625734,
                          254.544286049067, 254.54580474374734, 257.12146290159944, 257.12529776649166,
                          258.3474164128502, 258.3485239942917, 260.9327110491532, 260.9340257375877,
                          262.90751669716633, 262.9077362057592, 262.94993018168645, 262.9525447447239,
                          265.14039329636495, 265.16311246583433, 265.8445365616285, 265.8606448312495,
                          268.62976537680413, 268.63159928574873, 268.8718080221958, 268.872324172423,
                          270.7477589908166, 270.7817809750326, 275.77429707044286, 275.7742975202485,
                          278.07342940885303, 278.0829946932511, 278.4606492994791, 278.4809285321251,
                          279.2305995251554, 279.2316995312577, 279.2330770449011, 279.23414985192943,
                          279.23647285274717, 279.2399648419613, 279.47327799724104, 279.60781054672896,
                          280.5486046027173, 280.5519740393946, 281.17060373377404, 281.1761125184928,
                          281.1782859536547, 281.17877183338936, 281.17883175956706, 281.1800037326778,
                          281.18126699890075, 281.1901149895282, 281.20890519287775, 281.20895202619016,
                          281.9751327622438, 281.9792056770387, 282.2352172351048, 282.24894348009593,
                          283.2372532048137, 283.24005693955536, 283.242794833729, 283.24344070634504,
                          283.59934863950974, 283.59964735830494, 283.5997510106667, 283.60142341893817,
                          283.81851817928805, 283.82241312156043, 284.4411674976406, 284.44207715759086,
                          284.7277936412685, 285.5652902038477, 285.5687128200898, 285.571375610106, 285.57745104631795,
                          285.57878624593604, 285.5793867135136, 285.5814505538898, 285.5830162422366,
                          285.58757883625356, 285.5886545079454, 285.96805661917745, 285.96806461833705,
                          287.01221507585865, 287.4348661093819, 287.4357982914525, 287.4439322909748,
                          287.4486815283523, 287.8443389454168, 287.8459598455542, 287.8474060318161,
                          287.85118465266135, 287.8646316134916, 287.86882418538295, 287.8889012906417,
                          288.23661962674197, 288.3136449213081, 288.50344059348913, 288.5058362757437,
                          288.98634042336664, 289.00419909326314, 289.20399617181045, 289.20863904927603,
                          291.00672343238887, 291.01371610041707, 291.198568574, 291.35928180484376, 291.43201973788473,
                          291.4485711946821, 291.61004256620595, 291.6527429104865, 292.36623024644456,
                          292.36752049326765, 292.75775628323777, 292.79983812415094, 293.13559560455406,
                          293.61734265782434, 293.6186299590924, 293.9579964279248, 293.9799190672663,
                          294.8967003093777, 295.0441492236449, 295.0519723980249, 295.05433592770856,
                          295.0544936894122, 295.05886864153734, 295.68007601545025, 295.6945051516295,
                          295.7183588533513, 296.0674472970444, 296.0682336650365, 296.1614649805777, 296.1624307391884,
                          296.3277630515221, 296.3290021292955, 296.6868274672562, 296.68727025305293,
                          297.4702156750467, 297.4825194669855, 297.56452104700895, 297.57204305454707,
                          297.6937101348846, 297.70005737487384, 298.02221267624725, 298.03734313684635,
                          298.97186097147403, 299.0184223850097, 299.23331704782026, 299.23534755084154,
                          299.5130951042285, 299.52838843771406, 299.98401910210333, 299.99503681827366,
                          300.079131132314, 300.08622868906383, 300.86451413937823, 300.8646965978713,
                          300.89205310921864, 300.90662094815576, 301.01533630438803, 301.01568758314113,
                          301.2723758562778, 301.3035193584378, 301.3043451245349, 301.37564016952484,
                          301.96284799814464, 301.96572778371973, 301.96780000396984, 301.9681236208642,
                          301.9751231546964, 301.97931037669036, 302.40628022227656, 302.4392141547419,
                          302.44438816600507, 302.4992715979114, 302.96724299827497, 302.98972561354395,
                          303.6470612486681, 303.6680043902503, 303.9250142266542, 303.9250957807027,
                          312.72467742132835, 312.7334224192651]
        sampling_points = conformers.determine_torsion_sampling_points(label='', torsion_angles=torsion_angles)[0]
        self.assertEqual(sampling_points, [25.809606429720823, 66.64007136763342, 96.64007136763342, 126.64007136763342,
                                           156.64007136763342, 186.64007136763342, 270.1055076930085,
                                           300.1055076930085])

        torsion_angles = [21.029952658998006, 21.03683763790061, 22.318426332229294, 22.336215183961727,
                          23.333588670057164, 23.53377212293256, 25.49301296124286, 25.557679425383313,
                          27.37408014443026, 27.375068021347385, 28.302628635236076, 28.30666188407542,
                          35.11676917687218, 35.122528645010405, 35.38600242978858, 35.423586629995526,
                          36.49418243038614, 36.49461194898948, 38.08080106534097, 38.08143077323737, 39.63415129543948,
                          39.63522549410671, 41.15932732141922, 50.9477443315772, 50.95423555170428, 56.17775368411345,
                          56.17902819213001, 59.46352938407191, 59.47772321411631, 59.53170081489957, 60.70182746213326,
                          60.709505081165965, 64.79089192650625, 64.80853497686292, 64.9224755210989, 64.92656059591395,
                          68.68587366805725, 68.821036584528, 69.78204869965228, 69.81931017893343, 72.73119662519044,
                          72.74064499939671, 76.13914875143365, 76.14034180826447, 80.51285821060877, 80.51360853678266,
                          82.62855279474225, 82.62962281642685, 82.66153389416449, 82.67279209613228, 85.01626903974508,
                          85.02283268800834, 85.38800082360545, 85.38986517017983, 86.10988589746262, 86.11012064044289,
                          86.26592905483845, 86.2679012539459, 96.20740209593461, 96.21360170732405, 100.16258231764904,
                          100.16297697301955, 104.69626153683052, 106.92608118016551, 106.93146790288502,
                          116.61863123769633, 116.61864592924242, 118.24417546451886, 118.2493793312905,
                          120.1668712499721, 120.182922789032, 121.53479077752151, 121.53654524750456,
                          123.38491194765737, 123.4205961326216, 141.79830426741128, 141.84598851738662,
                          142.15429059974863, 142.16774851816908, 145.57607725441045, 145.57876371361897,
                          146.14196953860653, 146.1481645699347, 146.29244880461678, 146.3017402883685,
                          146.8840668702791, 146.89163317627043, 146.99548693580584, 147.0234330114963,
                          152.95573242995786, 152.95868183843845, 154.80682233127425, 154.80812898533412,
                          154.84602670053238, 207.73838602518913, 207.7412740770099, 210.14767760782937,
                          210.14868603812678, 213.22807520800336, 213.232115972531, 214.42148293923918,
                          214.42204428698125, 214.70131469094957, 214.7082883280577, 214.70948688563115,
                          214.72102073285245, 215.0646707286595, 215.09968560230425, 215.98093944599214,
                          215.98631496506223, 218.16637267967428, 218.16926310192207, 218.68310325593043,
                          218.6844547989021, 233.4741622446865, 233.4862509389519, 234.5172074467245,
                          234.51947934484195, 243.23967816490176, 243.313302680576, 245.5758741758238,
                          245.57602070099125, 245.6780799776852, 245.68310935421135, 247.0714203096195,
                          247.0971066518543, 248.3548371310418, 248.35825591171374, 252.216241106907,
                          252.21637331229618, 252.21727622490317, 252.21860279602214, 253.89500698181092,
                          253.92079194087057, 258.6902669332376, 258.692294283941, 262.0376785171378, 262.0419673198777,
                          265.31890933596605, 265.3199870866621, 265.54045126884927, 265.5414689557946,
                          268.8911876234665, 268.8950161977786, 270.2027846354444, 270.2045910209839, 270.7549100421501,
                          270.7582871485056, 272.84096993148665, 272.85901544460035, 284.9281072205472,
                          284.9299275177437, 286.7629083528301, 288.7121057475683, 288.73357811189237,
                          290.0191281086919, 290.0193204389709, 292.9188091677114, 293.92529074839973,
                          293.92733301910664, 293.98308673772544, 293.98692202903635, 297.6010232329676,
                          299.43002688567344, 300.54340628387564, 300.552522625597, 302.5923545236912,
                          302.6067661698267, 303.5750848000767, 303.6203720935542, 316.59887076547494, 316.617063221576,
                          318.7922195449375, 318.80880399370477, 319.34373259740704, 319.36687190118676,
                          320.42276391991555, 320.42616370985473, 324.1111895343747, 324.11141803287205,
                          324.8885643034773, 324.88889535280134, 325.38795495890577, 325.40753211241736,
                          325.9992475509422, 325.9998945453133, 326.0160343785387, 326.01850999047934,
                          327.15035431329005, 327.15111873652575, 329.1128133825743, 329.11595670623296,
                          331.64845728507674, 331.6566099651715, 331.68363641179496, 331.68501074128363,
                          331.6870781692291, 331.692395468951, 331.7051800706131, 331.75669183607647]
        sampling_points = conformers.determine_torsion_sampling_points(label='', torsion_angles=torsion_angles,
                                                                       symmetry=2)[0]
        self.assertEqual(sampling_points, [39.52873168116747, 69.52873168116747, 99.52873168116747, 129.52873168116747])

    def test_change_dihedrals_and_force_field_it(self):
        """Test changing a dihedral and retrieving the new FF energy"""
        ncc_xyz = {'coords': ((0.92795, -0.065916, -0.036432),
                              (2.389325, -0.061851, -0.064911),
                              (2.913834, 1.357417, -0.223617),
                              (2.741111, -0.474299, 0.885656),
                              (2.810508, -0.695037, -0.861612),
                              (2.543779, 1.992973, 0.584107),
                              (4.00671, 1.373862, -0.212637),
                              (2.583945, 1.791163, -1.17337),
                              (0.552434, 0.274266, -0.914418),
                              (0.566796, -1.001559, 0.102471)),
                   'isotopes': (14, 12, 12, 1, 1, 1, 1, 1, 1, 1),
                   'symbols': ('N', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H')}
        ncc_spc = ARCSpecies(label='NCC', smiles='NCC', xyz=ncc_xyz)
        ncc_mol = ncc_spc.mol
        energies = conformers.get_force_field_energies(label='NCC', mol=ncc_mol, xyz=ncc_xyz, optimize=True)[1]
        self.assertAlmostEqual(energies[0], -6.15026868, 5)
        idx0 = 10
        for i, atom in enumerate(ncc_mol.atoms):
            if atom.is_nitrogen():
                idx1 = i + 1
            elif atom.is_carbon():
                for atom2 in atom.edges.keys():
                    if atom2.is_nitrogen():
                        idx2 = i + 1
                        break
                else:
                    idx3 = i + 1
            elif atom.is_hydrogen():
                for atom2 in atom.edges.keys():
                    if atom2.is_nitrogen():
                        if i + 1 < idx0:
                            idx0 = i + 1
        torsion = (idx0, idx1, idx2, idx3)

        rd_conf = converter.rdkit_conf_from_mol(ncc_mol, ncc_xyz)[0]
        angle = rdMT.GetDihedralDeg(rd_conf, torsion[0] - 1, torsion[1] - 1, torsion[2] - 1, torsion[3] - 1)

        self.assertAlmostEqual(angle, 62.9431377, 5)

        xyzs, energies = conformers.change_dihedrals_and_force_field_it(label='NCC', mol=ncc_mol, xyz=ncc_xyz,
                                                                        torsions=[torsion], new_dihedrals=[180])

        expected_xyz = {'symbols': ('N', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H'),
                        'isotopes': (14, 12, 12, 1, 1, 1, 1, 1, 1, 1),
                        'coords': ((0.9341864117063167, -0.03285603972745727, -0.030012065030801692),
                                   (2.38993503322623, -0.074847908015265, -0.06911647066955567),
                                   (2.941403903438726, -0.5795255724709422, 1.2527502260139094),
                                   (2.717849986905357, -0.7297746478103556, -0.8830884226475003),
                                   (2.7758526938420145, 0.9294965458224461, -0.27090233721816753),
                                   (2.5902921769786222, -1.5943817769131658, 1.4682044523529993),
                                   (4.035470799632481, -0.6013337725677417, 1.2213623546604429),
                                   (2.6407717942045714, 0.07038089530452964, 2.0815301335388927),
                                   (0.5795889232686523, 0.2571468947460437, -0.94038297444683),
                                   (0.5673605907115399, -0.969470108525465, 0.1343930969631629))}

        self.assertAlmostEqual(energies[0], -6.1502687, 5)
        self.assertEqual(xyzs[0]['symbols'], expected_xyz['symbols'])
        self.assertTrue(almost_equal_coords_lists(xyzs[0], expected_xyz))
        self.assertEqual(len(energies), 1)

        xyzs, energies = conformers.change_dihedrals_and_force_field_it(label='NCC', mol=ncc_mol, xyz=ncc_xyz,
                                                                        torsions=[torsion],
                                                                        new_dihedrals=[0])
        self.assertEqual(len(energies), 1)

        xyzs, energies = conformers.change_dihedrals_and_force_field_it(label='NCC', mol=ncc_mol, xyz=ncc_xyz,
                                                                        torsions=[torsion, torsion],
                                                                        new_dihedrals=[[0, 180], [90, -120]])
        self.assertEqual(len(energies), 2)

    def test_determine_well_width_tolerance(self):
        """Test determining well width tolerance"""
        tols = list()
        tols.append(conformers.determine_well_width_tolerance(mean_width=101))
        tols.append(conformers.determine_well_width_tolerance(mean_width=90))
        tols.append(conformers.determine_well_width_tolerance(mean_width=50))
        tols.append(conformers.determine_well_width_tolerance(mean_width=45))
        tols.append(conformers.determine_well_width_tolerance(mean_width=23))
        tols.append(conformers.determine_well_width_tolerance(mean_width=7))
        tols.append(conformers.determine_well_width_tolerance(mean_width=1.5))
        self.assertEqual(tols, [0.1, 0.10530934999999897, 0.15021874999999973, 0.16273341406249986, 0.2647389825514999,
                                0.46149436430350005, 0.5777707774184844])

    def test_get_lowest_confs(self):
        """Test getting the n lowest conformers"""

        # test a case where confs is a list of dicts:
        confs = [{'index': 0,
                  'FF energy': 20,
                  'xyz': converter.str_to_xyz('C 0 0 0\nC 1 0 0')},
                 {'index': 1,
                  'FF energy': 30,
                  'xyz': converter.str_to_xyz('C 0 0 0\nC 2 0 0')},
                 {'index': 2,
                  'FF energy': 40,
                  'xyz': converter.str_to_xyz('C 0 0 0\nC 3 0 0')},
                 {'index': 3,
                  'some other energy': 10,
                  'xyz': converter.str_to_xyz('C 0 0 0\nC 4 0 0')},
                 ]

        lowest_confs = conformers.get_lowest_confs(label='', confs=confs, n=2, e=None)
        self.assertEqual(len(lowest_confs), 2)
        for conf in lowest_confs:
            self.assertTrue(conf['FF energy'] in [20, 30])
            self.assertTrue(conf['index'] in [0, 1])

        lowest_confs = conformers.get_lowest_confs(label='', confs=confs, e=5, energy='FF energy')
        self.assertEqual(len(lowest_confs), 1)
        self.assertEqual(lowest_confs[0]['FF energy'], 20)
        self.assertEqual(lowest_confs[0]['index'], 0)

        lowest_confs = conformers.get_lowest_confs(label='', confs=confs, e=15, energy='FF energy')
        self.assertEqual(len(lowest_confs), 2)
        for conf in lowest_confs:
            self.assertTrue(conf['FF energy'] in [20, 30])
            self.assertTrue(conf['index'] in [0, 1])

        # test a case where confs is a list of lists:
        confs = [['C 0 0 0\nC 1 0 0', 8],
                 ['C 0 0 0\nC 2 0 0', 7],
                 ['C 0 0 0\nC 3 0 0', 6],
                 ['C 0 0 0\nC 4 0 0', 5]]

        lowest_confs = conformers.get_lowest_confs(label='', confs=confs, n=1)
        self.assertEqual(len(lowest_confs), 1)
        self.assertEqual(lowest_confs[0]['xyz'], 'C 0 0 0\nC 4 0 0')
        self.assertEqual(lowest_confs[0]['FF energy'], 5)

        lowest_confs = conformers.get_lowest_confs(label='', confs=confs, e=2)
        self.assertEqual(len(lowest_confs), 3)

        # test a case where the number of confs is also the number to return:
        confs = [{'index': 0,
                  'FF energy': 20,
                  'xyz': converter.str_to_xyz('C 0 0 0\nC 1 0 0')},
                 {'index': 1,
                  'FF energy': 10,
                  'xyz': converter.str_to_xyz('C 0 0 0\nC 2 0 0')}]
        lowest_confs = conformers.get_lowest_confs(label='', confs=confs, n=2, e=None, energy='FF energy')
        self.assertEqual(len(lowest_confs), 2)
        lowest_confs = conformers.get_lowest_confs(label='', confs=confs, n=None, e=5, energy='FF energy')
        self.assertEqual(len(lowest_confs), 1)

        # test a case where the number of confs is lower than the number to return:
        confs = [{'index': 0,
                  'FF energy': 20},
                 {'index': 1,
                  'some other energy': 10}]
        lowest_confs = conformers.get_lowest_confs(label='', confs=confs, n=2, e=None, energy='FF energy')
        self.assertEqual(len(lowest_confs), 1)  # only 1, not 2

    def test_update_mol(self):
        """Test that atom ordering remains the same after updating a molecule in update_mol()"""
        xyz = {'symbols': ('S', 'O', 'O', 'N', 'N', 'C', 'C', 'C', 'C', 'C',
                           'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'),
               'isotopes': (32, 16, 16, 14, 14, 12, 12, 12, 12, 12,
                            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
               'coords': ((-1.7297500450289969, 0.515438796636824, 1.9168096618045691),
                          (-1.4803657376736399, 1.1318228490801632, -1.2741346588410747),
                          (1.6684385634300425, -1.4080623343331675, -0.69146134720917),
                          (1.4447732964722193, 0.26810320987982, 1.834186396466309),
                          (2.9639741941549724, 0.45637046406705295, -0.6018493257174595),
                          (0.7107726366328042, 0.4671661961997124, 0.5489199037723008),
                          (-0.68225248960311, -0.2422492925439827, 0.5854344518475088),
                          (-1.416609693814781, -0.21038706266511795, -0.7805348852864422),
                          (1.5885097252447307, -0.0015515049234385968, -0.6624596185356767),
                          (-2.818273455523886, -0.8196266870848813, -0.7693094853681998),
                          (0.5645026891741011, 1.5505258112503815, 0.44320480645337756),
                          (-0.5489109707645772, -1.2919888767636876, 0.8751860070317062),
                          (-0.8321905851878897, -0.784265725208026, -1.507876072604498),
                          (1.1510890997187422, 0.3429505362529166, -1.6042827339693007),
                          (-3.539862945639595, -0.18351203035164942, -0.24723208418918377),
                          (-2.81666678799753, -1.815870747483849, -0.3170415955378251),
                          (-3.191244105746757, -0.9126055866431627, -1.795841007782851),
                          (3.3764976755549823, 0.08016081376522317, 0.24941044358861653),
                          (2.959962629146053, 1.469289098466773, -0.48374550042835573),
                          (1.7124864428642297, -0.7164193231909225, 1.8874284688782355),
                          (2.3042342900725874, 0.8091358385609299, 1.8231896193937174),
                          (2.4468466474830963, -1.5710191541618868, -1.2534461848335496),
                          (-1.779032202163809, 1.7365703107348773, 1.363680443883325),
                          (-2.0569288708039766, 1.130024400459187, -2.059052520249386))}
        mol = ARCSpecies(label='OC(N)C(N)C(S)C(C)O', smiles='OC(N)C(N)C(S)C(C)O', xyz=xyz).mol
        for i, atom in enumerate(mol.atoms):
            self.assertEqual(atom.symbol, xyz['symbols'][i])
        mol = conformers.update_mol(mol)
        for i, atom in enumerate(mol.atoms):
            self.assertEqual(atom.symbol, xyz['symbols'][i])

    def test_translate_group(self):
        """Test translating groups within a molecule"""
        xyz1 = {'symbols': ('O', 'O', 'C'), 'isotopes': (16, 16, 12),
                'coords': ((1.40486421, 0.01953338, 0.0), (-1.40486421, -0.01953339, 0.0), (0.0, 0.0, 0.0))}
        spc1 = ARCSpecies(label='CO2', smiles='O=C=O', xyz=xyz1)
        # translate the first O by 90 degrees:
        new_xyz1 = conformers.translate_group(mol=spc1.mol, xyz=xyz1, pivot=2, anchor=0, vector=[0, 0, 1])
        expected_xyz1 = {'symbols': ('O', 'O', 'C'), 'isotopes': (16, 16, 12),
                         'coords': ((8.749511531958021e-17, 3.469446951953614e-18, 1.405000000524252),
                                    (-1.40486421, -0.01953339, 0.0), (0.0, 0.0, 0.0))}
        self.assertEqual(new_xyz1, expected_xyz1)

        xyz2 = converter.str_to_xyz("""Cl      1.47512188   -0.78746253   -0.20393322
S      -1.45707856   -0.94104506   -0.20275830
O      -0.03480906    1.11948179   -0.82988874
C      -0.02416711    0.17703194    0.08644641
H       0.04093286    0.43199386    1.15013385""")
        spc2 = ARCSpecies(label='chiral1', smiles='[S]C([O])Cl', xyz=xyz2)
        vector1 = vectors.unit_vector(vectors.get_vector(pivot=3, anchor=1, xyz=xyz2))
        vector2 = vectors.unit_vector(vectors.get_vector(pivot=3, anchor=2, xyz=xyz2))
        new_xyz2 = conformers.translate_group(mol=spc2.mol, xyz=xyz2, pivot=3, anchor=2, vector=vector1)
        new_xyz2 = conformers.translate_group(mol=spc2.mol, xyz=new_xyz2, pivot=3, anchor=1, vector=vector2)
        expected_xyz2 = converter.str_to_xyz("""Cl      1.47512188   -0.78746253   -0.20393322
S      -0.03906606    1.49648129   -1.19644182
O      -1.04766012   -0.62158265   -0.12012532
C      -0.02416711    0.17703194    0.08644641
H       0.04093286    0.43199386    1.15013385
""")
        self.assertTrue(almost_equal_coords_lists(new_xyz2, expected_xyz2))

        xyz3 = converter.str_to_xyz(""" C                  1.50048866   -0.50848248   -0.64006761
 F                  0.27568368    0.01156702   -0.41224910
 Cl                 3.09727149   -1.18647281   -0.93707550
 O                  1.93561914    0.74741610   -1.16759033
 O                  2.33727805    1.90670710   -1.65453438
 H                  3.25580294    1.83954553   -1.92546105""")
        spc3 = ARCSpecies(label='chiral2', smiles='OO[C](F)Cl', xyz=xyz3)
        vector1 = vectors.unit_vector(vectors.get_vector(pivot=0, anchor=1, xyz=xyz3))
        vector2 = vectors.unit_vector(vectors.get_vector(pivot=0, anchor=3, xyz=xyz3))
        new_xyz3 = conformers.translate_group(mol=spc3.mol, xyz=xyz3, pivot=0, anchor=3, vector=vector1)
        new_xyz3 = conformers.translate_group(mol=spc3.mol, xyz=new_xyz3, pivot=0, anchor=1, vector=vector2)
        expected_xyz3 = converter.str_to_xyz("""C       1.50048866   -0.50848248   -0.64006761
F       1.91127618    0.67715604   -1.13807857
Cl      3.09727149   -1.18647281   -0.93707550
O       0.20310264    0.04238477   -0.39874874
O      -0.99448445    0.55087762   -0.17599287
H      -1.00986076    1.46908426   -0.45574283
""")
        self.assertTrue(almost_equal_coords_lists(new_xyz3, expected_xyz3))

        xyz4 = {'symbols': ('S', 'O', 'O', 'O', 'C', 'C', 'H', 'H', 'H', 'H'),
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
        spc4 = ARCSpecies(label='chiral3', smiles='SC(OO)=CO', xyz=xyz4)
        vector1 = vectors.unit_vector(vectors.get_vector(pivot=4, anchor=0, xyz=xyz4))
        vector2 = vectors.unit_vector(vectors.get_vector(pivot=4, anchor=2, xyz=xyz4))
        new_xyz4 = conformers.translate_group(mol=spc4.mol, xyz=xyz4, pivot=4, anchor=2, vector=vector1)
        new_xyz4 = conformers.translate_group(mol=spc4.mol, xyz=new_xyz4, pivot=4, anchor=0, vector=vector2)
        expected_xyz4 = {'symbols': ('S', 'O', 'O', 'O', 'C', 'C', 'H', 'H', 'H', 'H'),
                         'isotopes': (32, 16, 16, 16, 12, 12, 1, 1, 1, 1),
                         'coords': ((-0.8975272666391765, -0.28877919066680047, -0.8866590574065785),
                                    (0.40385373, -0.65769862, 1.03431374),
                                    (0.5162351224205494, 2.5776882661021125, 1.4903653502407788),
                                    (0.6901556, -1.65712867, 0.01239391), (-0.0426136, 0.49595776, 0.40364219),
                                    (0.6096964210696835, 1.217485186769998, 1.3315051098421813),
                                    (1.2757172600702331, 0.7966267238977445, 2.0754008550613348),
                                    (-1.5831217972336848, 0.7805289916315858, -1.3150490129767116),
                                    (-0.06378382313185912, 2.8994155510586217, 0.7677427813224768),
                                    (1.65888059, -1.54205855, 0.02674995))}
        self.assertTrue(almost_equal_coords_lists(new_xyz4, expected_xyz4))

        xyz5 = converter.str_to_xyz(""" C                  1.18149528   -0.70041459   -0.31471741
 C                  0.93123100    0.49135224   -1.25740484
 C                  0.68096672    1.68311908   -2.20009227
 C                  0.90913879    1.54885051   -3.56990280
 C                  0.22609012    2.89672219   -1.68453149
 C                  0.68310320    2.62819874   -4.42379802
 H                  1.26848882    0.59217080   -3.97594868
 C                 -0.00100499    3.97619810   -2.53867988
 H                  0.04603318    3.00266663   -0.60495551
 C                  0.22754321    3.84217921   -3.90811103
 H                  0.86352778    2.52263257   -5.50342726
 H                 -0.36004784    4.93282504   -2.13195230
 H                  0.04933714    4.69304911   -4.58157919
 F                  0.25569319   -1.42216987   -0.98139560
 Cl                 2.38846685    0.24054066    0.55443324""")
        spc5 = ARCSpecies(label='chiral4', smiles='ClC(F)=[C]c1ccccc1', xyz=xyz5)
        vector1 = vectors.unit_vector(vectors.get_vector(pivot=0, anchor=1, xyz=xyz5))
        vector2 = vectors.unit_vector(vectors.get_vector(pivot=0, anchor=13, xyz=xyz5))
        new_xyz5 = conformers.translate_group(mol=spc5.mol, xyz=xyz5, pivot=0, anchor=13, vector=vector1)
        new_xyz5 = conformers.translate_group(mol=spc5.mol, xyz=new_xyz5, pivot=0, anchor=1, vector=vector2)
        expected_xyz5 = converter.str_to_xyz("""C       1.18149528   -0.70041459   -0.31471741
C       0.12539512   -1.52375024   -1.07522438
C      -0.93070504   -2.34708589   -1.83573135
C      -0.56553338   -3.53564257   -2.46854578
C      -2.25231991   -1.90466577   -1.89182711
C      -1.52168662   -4.28119970   -3.15784780
H       0.47664391   -3.88377085   -2.42478536
C      -3.20909105   -2.65084001   -2.58046554
H      -2.54034188   -0.96801190   -1.39297653
C      -2.84398183   -3.83887752   -3.21355255
H      -1.23379910   -5.21772042   -3.65719364
H      -4.25113565   -2.30197553   -2.62417846
H      -3.59771174   -4.42692116   -3.75705255
F       0.96210776    0.34431608   -1.14109925
Cl      2.38846685    0.24054066    0.55443324
""")
        self.assertTrue(almost_equal_coords_lists(new_xyz5, expected_xyz5))

    def test_convert_chirality(self):
        """Test converting a chiral center"""
        xyz1 = {'coords': ((1.38346248, 1.33352376, 0.05890374),
                           (0.4240511, -0.73855006, 1.08316776),
                           (-0.85054134, 0.17787474, -0.37480771),
                           (0.50839421, -0.20402577, -0.10066066),
                           (0.92397006, -0.715863, -0.97353222),
                           (-1.03560134, 0.80870122, 0.40254095),
                           (-1.35373518, -0.66166089, -0.09561187)),
                'isotopes': (35, 16, 14, 12, 1, 1, 1),
                'symbols': ('Cl', 'O', 'N', 'C', 'H', 'H', 'H')}
        spc1 = ARCSpecies(label='chiral1', smiles='[O]C(N)Cl', xyz=xyz1)
        new_xyz1 = conformers.translate_groups(label='', mol=spc1.mol, xyz=xyz1, pivot=3)
        expected_xyz1 = {'symbols': ('Cl', 'O', 'N', 'C', 'H', 'H', 'H'),
                         'isotopes': (35, 16, 14, 12, 1, 1, 1),
                         'coords': ((1.38346248, 1.33352376, 0.05890374),
                                    (1.0028995740089042, -0.8130752509418283, -1.1393150199942588),
                                    (-0.85054134, 0.17787474, -0.37480771),
                                    (0.50839421, -0.20402577, -0.10066066),
                                    (0.4375133618212483, -0.6532330027822042, 0.8942132841515917),
                                    (-1.03560134, 0.80870122, 0.40254095), (-1.35373518, -0.66166089, -0.09561187))}
        self.assertTrue(almost_equal_coords_lists(new_xyz1, expected_xyz1))

    def test_get_number_of_chiral_centers(self):
        mol1 = Molecule(smiles='CNC(O)(S)C=CO')
        xyz = {'symbols': ('S', 'O', 'O', 'N', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'),
               'isotopes': (32, 16, 16, 14, 12, 12, 12, 12, 1, 1, 1, 1, 1, 1, 1, 1, 1),
               'coords': ((0.6420879930160509, 1.4059756276576734, -1.9004366023735562),
                          (-1.0389352827938405, -0.5566748017044921, -1.840496829821974),
                          (2.5668341201990437, 0.49790540744287926, 0.36347381229156395),
                          (-1.0760490888259793, 0.8343723253593865, 0.027417240207766196),
                          (-0.22885676667434462, 0.15057082360988466, -0.9240814312969589),
                          (-1.9902899508325493, -0.03808658641522206, 0.7491391566675523),
                          (0.7005970180291918, -0.8480132723154269, -0.29672702659709393),
                          (1.9133334329533498, -0.6952316522517126, 0.2525043929875734),
                          (-2.6734628821867386, -0.5712458130150081, 0.08031482787764908),
                          (-1.459385075800411, -0.7630948848733338, 1.3737382788435994),
                          (-2.6038897158166754, 0.569802586439533, 1.4221886384919287),
                          (-1.6220195571611036, 1.4945440103218024, -0.5185627968437716),
                          (0.29690439670859164, -1.8591482100344014, -0.28128779841506435),
                          (2.4642558717204968, -1.525939409138389, 0.6779057538876055),
                          (-0.7512476960860016, -0.19385302132671417, -2.694945733376175),
                          (1.4724369524535623, 1.7465840526502638, -0.9017293410970915),
                          (3.387686231096821, 0.35153281759343014, 0.8677681809936627))}
        chirality_dict = conformers.get_number_of_chiral_centers(label='CNC(O)(S)C=CO', mol=mol1,
                                                                 xyz=xyz, just_get_the_number=False)
        expected_chirality_dict = {'C': 1, 'N': 1, 'D': 1}
        self.assertEqual(chirality_dict, expected_chirality_dict)
        number = conformers.get_number_of_chiral_centers(label='CNC(O)(S)C=CO', mol=mol1,
                                                         xyz=xyz, just_get_the_number=True)
        self.assertEqual(number, 3)

        mol2 = Molecule(smiles='OC(N)C(N)C(S)C(C)O')
        conformer = {'xyz': {'symbols': ('S', 'O', 'O', 'N', 'N', 'C', 'C', 'C', 'C', 'C',
                                         'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'),
                             'isotopes': (32, 16, 16, 14, 14, 12, 12, 12, 12, 12,
                                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
                             'coords': ((-1.7297500450289969, 0.515438796636824, 1.9168096618045691),
                                        (-1.4803657376736399, 1.1318228490801632, -1.2741346588410747),
                                        (1.6684385634300425, -1.4080623343331675, -0.69146134720917),
                                        (1.4447732964722193, 0.26810320987982, 1.834186396466309),
                                        (2.9639741941549724, 0.45637046406705295, -0.6018493257174595),
                                        (0.7107726366328042, 0.4671661961997124, 0.5489199037723008),
                                        (-0.68225248960311, -0.2422492925439827, 0.5854344518475088),
                                        (-1.416609693814781, -0.21038706266511795, -0.7805348852864422),
                                        (1.5885097252447307, -0.0015515049234385968, -0.6624596185356767),
                                        (-2.818273455523886, -0.8196266870848813, -0.7693094853681998),
                                        (0.5645026891741011, 1.5505258112503815, 0.44320480645337756),
                                        (-0.5489109707645772, -1.2919888767636876, 0.8751860070317062),
                                        (-0.8321905851878897, -0.784265725208026, -1.507876072604498),
                                        (1.1510890997187422, 0.3429505362529166, -1.6042827339693007),
                                        (-3.539862945639595, -0.18351203035164942, -0.24723208418918377),
                                        (-2.81666678799753, -1.815870747483849, -0.3170415955378251),
                                        (-3.191244105746757, -0.9126055866431627, -1.795841007782851),
                                        (3.3764976755549823, 0.08016081376522317, 0.24941044358861653),
                                        (2.959962629146053, 1.469289098466773, -0.48374550042835573),
                                        (1.7124864428642297, -0.7164193231909225, 1.8874284688782355),
                                        (2.3042342900725874, 0.8091358385609299, 1.8231896193937174),
                                        (2.4468466474830963, -1.5710191541618868, -1.2534461848335496),
                                        (-1.779032202163809, 1.7365703107348773, 1.363680443883325),
                                        (-2.0569288708039766, 1.130024400459187, -2.059052520249386))},
                     'index': 0, 'FF energy': 22.961887225620355, 'source': 'MMFF94',
                     'torsion_dihedrals': {(23, 1, 7, 6): 63.252162092007836, (24, 2, 8, 7): 188.18176931477524,
                                           (22, 3, 9, 5): 321.8543702025301, (20, 4, 6, 7): 69.42534853556789,
                                           (18, 5, 9, 3): 299.56981935604216, (4, 6, 9, 3): 72.49491068065248,
                                           (4, 6, 7, 1): 62.60285015856057, (1, 7, 8, 2): 71.176964208596,
                                           (2, 8, 10, 15): 307.898746690347}}
        chirality_dict = conformers.get_number_of_chiral_centers(label='OC(N)C(N)C(S)C(C)O', mol=mol2,
                                                                 conformer=conformer, just_get_the_number=False)
        expected_chirality_dict = {'C': 4, 'N': 0, 'D': 0}
        self.assertEqual(chirality_dict, expected_chirality_dict)
        number = conformers.get_number_of_chiral_centers(label='OC(N)C(N)C(S)C(C)O', mol=mol2,
                                                         conformer=conformer, just_get_the_number=True)
        self.assertEqual(number, 4)

    def test_determine_chirality(self):
        """Test determining R/S/E/Z chirality of atom centers and double bonds"""
        # One chiral C center.
        confs = [{'xyz': {'symbols': ('O', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'),
                          'isotopes': (16, 12, 12, 12, 12, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
                          'coords': ((-1.3836321615294294, -1.3053014308406108, -0.9378250622062406),
                                     (-0.6696330674918395, -0.5206694436671537, 0.017366092492431772),
                                     (0.7600028422950423, -0.31244312508738487, -0.48984341243804375),
                                     (-1.4375832510055804, 0.779775353185485, 0.20647632598093374),
                                     (1.6491881296919153, 0.44188292995535106, 0.4848591853644157),
                                     (-0.6606929547148451, -1.0771525336410563, 0.9613257999392658),
                                     (1.210567291806299, -1.2914746565750734, -0.6966296665542017),
                                     (0.7414760036284255, 0.21038137537248922, -1.4542799433502682),
                                     (2.6758767671675066, 0.4771730436037278, 0.10643306236375147),
                                     (1.311440175413646, 1.4734363557117354, 0.620459389920203),
                                     (1.6656277360415193, -0.04974635614075438, 1.4626130356981),
                                     (-1.0016043498768117, 1.399185839025344, 0.995066997067928),
                                     (-1.4622480112395122, 1.3588735975343944, -0.7232294702657215),
                                     (-2.479744308228501, 0.5711234850055331, 0.47272491808492745),
                                     (-0.9190408419578571, -2.155044433442066, -1.0255172520974882))},
                  'index': 0, 'FF energy': 1.4674221390129727, 'source': 'MMFF94',
                  'torsion_dihedrals': {(15, 1, 2, 3): 59.84839196064534, (1, 2, 3, 5): 183.49860912922097,
                                        (1, 2, 4, 12): 174.74760319759486, (2, 3, 5, 9): 174.81552259870148}},
                 {'xyz': {'symbols': ('O', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'),
                          'isotopes': (16, 12, 12, 12, 12, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
                          'coords': ((0.674497217076435, -0.26231958203527134, 1.7970022065880684),
                                     (0.28756925889735574, 0.5119002517924365, 0.6648520968794853),
                                     (-0.46201904241049996, -0.36550915052842486, -0.3411739680851192),
                                     (1.5340613565581214, 1.1304797154265056, 0.04669745553541255),
                                     (-1.7442554444408034, -0.9441814470086994, 0.24198816347133886),
                                     (-0.3692532108656602, 1.3116656951757875, 1.0241362712844757),
                                     (-0.7087504119126617, 0.2202708661474925, -1.2342070490325108),
                                     (0.17600704844977633, -1.1992564185451453, -0.6585999108587859),
                                     (-1.5354108135625717, -1.5931711831228046, 1.0981531595999416),
                                     (-2.4193977443622225, -0.14737187978087019, 0.5698787550543981),
                                     (-2.265680564059287, -1.5412220834810455, -0.5129216222825455),
                                     (2.233464986190819, 0.3562849165858558, -0.2876846606926968),
                                     (2.0665809677306575, 1.7362922480079568, 0.7877648580734955),
                                     (1.2808255678852123, 1.7661452099631867, -0.8071405153386059),
                                     (1.251760828825342, -0.9800071585968614, 1.484840950550354))},
                  'index': 1, 'FF energy': 1.1177706999925383, 'source': 'MMFF94',
                  'torsion_dihedrals': {(15, 1, 2, 3): 59.859190489723126, (1, 2, 3, 5): 61.70252194292617,
                                        (1, 2, 4, 12): 60.87837765729409, (2, 3, 5, 9): 298.05629041710233}}]
        mol = ARCSpecies(label='CCC(O)C', smiles='CCC(O)C', xyz=confs[0]['xyz']).mol  # preserves atom order
        confs = conformers.determine_chirality(conformers=confs, label='2-butanol', mol=mol)
        self.assertEqual(confs[0]['chirality'], {(1,): 'R'})
        self.assertEqual(confs[1]['chirality'], {(1,): 'S'})

        # One chiral N center.
        confs = [{'xyz': {'symbols': ('S', 'O', 'N', 'C', 'H', 'H', 'H', 'H', 'H'),
                          'isotopes': (32, 16, 14, 12, 1, 1, 1, 1, 1),
                          'coords': ((0.6642693414954796, 1.2877501906584146, -1.012730946867456),
                                     (1.4434020248169315, -0.38401499067541955, 0.742113508402271),
                                     (0.27842261260800394, 0.36827437724805046, 0.309124990733691),
                                     (-0.792179078734418, -0.6250047255451342, 0.23270021781619676),
                                     (-0.9417015295075069, -1.1002179337839513, 1.2084570329632383),
                                     (-0.5728811846002392, -1.4068932176620021, -0.5035019715368415),
                                     (-1.7406369286234398, -0.14967627161847707, -0.04118908533890797),
                                     (2.1170524073217942, -0.017576732086427016, 0.14110126958923766),
                                     (-0.45574766477657924, 2.027359303464945, -1.076075015761433))},
                  'index': 0, 'FF energy': 11.313324099313466, 'source': 'MMFF94',
                  'torsion_dihedrals': {(9, 1, 3, 2): 189.73319214323027, (8, 2, 3, 1): 5.523715708214589,
                                        (1, 3, 4, 5): 182.8555331584859}},
                 {'xyz': {'symbols': ('S', 'O', 'N', 'C', 'H', 'H', 'H', 'H', 'H'),
                          'isotopes': (32, 16, 14, 12, 1, 1, 1, 1, 1),
                          'coords': ((1.0084332582992683, -1.149848836139103, 1.0505954063768639),
                                     (-1.4211947643583844, -0.7242545673787971, 0.43044054259763487),
                                     (-0.06254377914282004, -0.5047725595919565, -0.036074937116642086),
                                     (-0.01451295996155549, 0.9215373257863942, -0.35795045808837966),
                                     (-0.20522047536190932, 1.5466747772296041, 0.5220106693916197),
                                     (0.9652420332986122, 1.1917858818632272, -0.767326696516769),
                                     (-0.7597970492723831, 1.1644301578930112, -1.123376622505416),
                                     (-1.6565217873850617, -1.5130591988407518, -0.08973656186875435),
                                     (2.146115523884054, -0.932492980821605, 0.37141865772986954))},
                  'index': 5, 'FF energy': 10.673849828339602, 'source': 'MMFF94',
                  'torsion_dihedrals': {(9, 1, 3, 2): 175.1514313020827, (8, 2, 3, 1): 261.8174240396574,
                                        (1, 3, 4, 5): 298.2847739052176}}]
        mol = ARCSpecies(label='ON(C)(S)', smiles='ON(C)(S)', xyz=confs[0]['xyz']).mol  # preserves atom order
        confs = conformers.determine_chirality(conformers=confs, label='ON(C)(S)', mol=mol)
        self.assertEqual(confs[0]['chirality'], {(2,): 'NS'})
        self.assertEqual(confs[1]['chirality'], {(2,): 'NR'})

        # One chiral N center.
        confs = [{'xyz': {'symbols': ('O', 'N', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'),
                          'isotopes': (16, 14, 12, 12, 12, 1, 1, 1, 1, 1, 1, 1, 1, 1), 'coords': (
            (-1.1260113581472435, 1.3761961818488566, 0.09344599103558728),
            (-1.007935513086041, -0.060277862163366974, -0.09993420755804434),
            (0.151358859843175, -0.27030737367364815, -0.9795144959896613),
            (1.4906256072798636, 0.2796855525759796, -0.4957720189189826),
            (-0.8225472029133591, -0.6198916544114855, 1.236450696209674),
            (-0.07761195961803313, 0.17763805874443062, -1.9544637548276826),
            (0.25834932156021245, -1.3457949407658736, -1.167089042946175),
            (1.4403695778627317, 1.355537171420406, -0.3016073959171323),
            (1.8300405092867553, -0.22342396900707007, 0.41456530364965505),
            (2.2550119159612887, 0.11755528333371892, -1.2631875084858506),
            (-1.7453782035104, -0.5017121642006336, 1.8148154521877977),
            (-0.6160862591652101, -1.6939507925721629, 1.1803550095576598),
            (-0.015633065127377256, -0.1348394230710416, 1.7958114555334979),
            (-2.0145522302265086, 1.5435859319421577, -0.273875483530568))}, 'index': 2, 'FF energy': 20.2904299980655,
                  'source': 'MMFF94',
                  'torsion_dihedrals': {(14, 1, 2, 3): 119.43272880797798, (1, 2, 5, 11): 68.37929823399126,
                                        (1, 2, 3, 4): 57.86935830901248, (2, 3, 4, 8): 302.8320372319507}},
                 {'xyz': {'symbols': ('O', 'N', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'),
                          'isotopes': (16, 14, 12, 12, 12, 1, 1, 1, 1, 1, 1, 1, 1, 1), 'coords': (
                     (1.5655272732117604, 0.741879169647508, -1.2062846243961904),
                     (0.5633566474165278, -0.15418061414647216, -0.6423115896914202),
                     (-0.511769740850912, 0.7358672851934527, -0.1798524588971155),
                     (-1.7120688384845566, -0.0308203921710982, 0.3615285164098966),
                     (1.2502841581658586, -0.8126642623570183, 0.4701156007474357),
                     (-0.8587852733983324, 1.3329637132185015, -1.032484205031793),
                     (-0.14921730987267087, 1.4422219078982466, 0.5776055526573312),
                     (-2.528543982683172, 0.6639594124604079, 0.5846376708253733),
                     (-1.4713883777486563, -0.5640772542627065, 1.2861373160467724),
                     (-2.080774893369899, -0.7563091230479586, -0.371287616932611),
                     (0.5938202155462396, -1.5340729192591642, 0.9662682005139577),
                     (1.6156617273425515, -0.09806062247169192, 1.21637578375925),
                     (2.1088944673719134, -1.3798450282247212, 0.09354820894490706),
                     (1.6150039273574184, 0.4131387275224817, -2.1239963549485936))}, 'index': 1,
                  'FF energy': 19.516538133934294, 'source': 'MMFF94',
                  'torsion_dihedrals': {(14, 1, 2, 3): 239.0576218005308, (1, 2, 5, 11): 181.0950382395368,
                                        (1, 2, 3, 4): 175.97347188945042, (2, 3, 4, 8): 186.5196017991176}}]
        mol = ARCSpecies(label='ON(C)CC', smiles='ON(C)CC', xyz=confs[0]['xyz']).mol  # preserves atom order
        confs = conformers.determine_chirality(conformers=confs, label='ON(C)CC', mol=mol)
        self.assertEqual(confs[0]['chirality'], {(1,): 'NR'})
        self.assertEqual(confs[1]['chirality'], {(1,): 'NS'})

        # One chiral N center.
        confs = [{'xyz': {'symbols': ('N', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'),
                          'isotopes': (14, 12, 12, 12, 1, 1, 1, 1, 1, 1, 1, 1, 1),
                          'coords': ((-0.7073511380114899, 0.8197164661073145, 0.24999885684842907),
                                     (0.5801699241438923, 0.6591912227657216, -0.4240530456792972),
                                     (1.4472113196665168, -0.4372777699985311, 0.17945348165229474),
                                     (-1.6390090531407464, -0.25796648974989134, -0.04936095294345401),
                                     (1.1197404668780022, 1.6093134315498747, -0.3376879040777191),
                                     (0.43764603823503984, 0.4845854286631773, -1.4968921979101455),
                                     (1.002550211214117, -1.4275789874428901, 0.04242740542022188),
                                     (2.4294750170597803, -0.4452330671213071, -0.30432398624467893),
                                     (1.6034105333687751, -0.27376799189464784, 1.2509388990817716),
                                     (-1.8125204547484717, -0.3462467078827822, -1.1266788056658708),
                                     (-2.6039691776051646, -0.041004688407494536, 0.41960198456069625),
                                     (-1.2927485939442889, -1.2203699903546943, 0.338772805584756),
                                     (-0.5646050931159982, 0.8766391437661488, 1.2578034593729632))},
                  'index': 0, 'FF energy': -1.859731850553788, 'source': 'MMFF94',
                  'torsion_dihedrals': {(2, 1, 4, 10): 302.68771874988295, (4, 1, 2, 3): 286.5769827243587,
                                        (1, 2, 3, 7): 65.87874548828209}},
                 {'xyz': {'symbols': ('N', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'),
                          'isotopes': (14, 12, 12, 12, 1, 1, 1, 1, 1, 1, 1, 1, 1),
                          'coords': ((0.7419854952964929, -0.18588322859265055, -0.8060295165375653),
                                     (-0.38476668007897186, -0.8774643553523614, -0.1815530887172187),
                                     (-1.4977348513273125, 0.05995564693605262, 0.26652181022311233),
                                     (1.5633235727172392, 0.5360966415350092, 0.15477859056711452),
                                     (-0.04458112063725271, -1.4936027355391557, 0.6589418973690523),
                                     (-0.7986335015469359, -1.5715787743431335, -0.9219907626214912),
                                     (-2.348455608682208, -0.5210498432021002, 0.6375394558854425),
                                     (-1.8523669868240424, 0.6790455638159553, -0.5642494434208211),
                                     (-1.170505453235269, 0.7210016856743618, 1.0746899133307615),
                                     (2.4283037770451084, 0.9651590522064675, -0.36083882142892065),
                                     (1.945994527876002, -0.1322800197070601, 0.9328203647772167),
                                     (1.0178974719106297, 1.3595978302624294, 0.6250164549219148),
                                     (0.39953935748654607, 0.4610025363062083, -1.5156468543485933))},
                  'index': 2, 'FF energy': -1.8597318482610092, 'source': 'MMFF94',
                  'torsion_dihedrals': {(2, 1, 4, 10): 175.65579979115313, (4, 1, 2, 3): 73.42320852093809,
                                        (1, 2, 3, 7): 174.68783904235727}}]
        mol = ARCSpecies(label='CNCC', smiles='CNCC', xyz=confs[0]['xyz']).mol  # preserves atom order
        confs = conformers.determine_chirality(conformers=confs, label='CNCC', mol=mol)
        self.assertEqual(confs[0]['chirality'], {(0,): 'NS'})
        self.assertEqual(confs[1]['chirality'], {(0,): 'NR'})

        # One chiral double bond.
        confs = [{'xyz': {'symbols': ('N', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H'),
                          'isotopes': (14, 12, 12, 12, 1, 1, 1, 1, 1, 1, 1),
                          'coords': ((1.6790693468233917, -0.0528568295658936, -0.45240516692920485),
                                     (-1.1415432095482703, 0.8159978183020002, 0.45777653889749736),
                                     (-0.6030745653144279, -0.5588309590596223, 0.23246181251406198),
                                     (0.6339013864077682, -0.8831860930051899, -0.15776847330162308),
                                     (-1.9778945186106556, 1.0025324715053494, -0.22290121239631047),
                                     (-0.3957446439345486, 1.5999948333310583, 0.3027471612426946),
                                     (-1.510839172457255, 0.9036705220969689, 1.4843947319916737),
                                     (-1.3027235745367962, -1.369975084517498, 0.4249808960994018),
                                     (0.8972744852259353, -1.9357535283955147, -0.2634276250968679),
                                     (2.314682816337745, -0.4270626936455623, -1.1531647677060783),
                                     (1.4068916496070927, 0.9054695429538218, -0.6526938953152706))},
                  'index': 0, 'FF energy': -0.24381581344325376, 'source': 'MMFF94',
                  'torsion_dihedrals': {(10, 1, 4, 3): 210.4306870643632, (5, 2, 3, 4): 118.8944142787326}},
                 {'xyz': {'symbols': ('N', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H'),
                          'isotopes': (14, 12, 12, 12, 1, 1, 1, 1, 1, 1, 1),
                          'coords': ((-2.0444722658072676, 0.021096781039271272, 0.30298806055406596),
                                     (1.6583031733640239, 0.00851091952710539, -0.4944103909895636),
                                     (0.26810087021443746, -0.4448130149550418, -0.19070593904913016),
                                     (-0.7551907722509675, 0.38633053142055346, 0.029185846843007235),
                                     (1.7440364246039963, 1.0996203081736124, -0.5120458434762838),
                                     (1.970035026863879, -0.37414588279153344, -1.471052828043817),
                                     (2.349037813280732, -0.37581346737605087, 0.2622971512323338),
                                     (0.10649090396788516, -1.5188739358061625, -0.16340650129386183),
                                     (-0.6163826189655114, 1.4661454618836856, -0.007128797639924795),
                                     (-2.5356581672628917, 0.6750310304472394, 0.908223106013873),
                                     (-2.1443003880083205, -0.9430887315627433, 0.6110064895340819))},
                  'index': 4, 'FF energy': -1.312714178518334, 'source': 'MMFF94',
                  'torsion_dihedrals': {(10, 1, 4, 3): 148.39156975667314, (5, 2, 3, 4): 0.243893275336888}}]
        mol = ARCSpecies(label='CC=CN', smiles='CC=CN', xyz=confs[0]['xyz']).mol  # preserves atom order
        confs = conformers.determine_chirality(conformers=confs, label='CC=CN', mol=mol)
        self.assertEqual(confs[0]['chirality'], {(2, 3): 'Z'})
        self.assertEqual(confs[1]['chirality'], {(2, 3): 'E'})

        # One chiral double bond, one chiral C center, one chiral N center.
        confs = [{'xyz': {'symbols': ('S', 'O', 'O', 'N', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H'),
                          'isotopes': (32, 16, 16, 14, 12, 12, 12, 1, 1, 1, 1, 1, 1, 1),
                          'coords': ((1.730624392278447, 0.35615417645030134, 1.6248629220037842),
                                     (0.6807202076552984, 1.949430671070447, -0.2353806638510147),
                                     (-2.746407458295995, -0.9138273136209699, 0.3545271400021022),
                                     (0.7032858579085668, -0.28597182075547545, -0.7820856256658617),
                                     (1.4143066998411842, 0.7604564595772063, -0.09298883621289974),
                                     (-0.5650014170868946, -0.7732982106030353, -0.5443979867563395),
                                     (-1.5322919454724873, -0.2947040553199928, 0.24009717315511855),
                                     (2.379409070741076, 0.9091340746308052, -0.5866484741455296),
                                     (1.3245240729295662, -0.9815808859760832, -1.1870567507275864),
                                     (-0.7636604502233536, -1.6863826947117113, -1.1056300203331668),
                                     (-1.4417611494498759, 0.5994723074582383, 0.8436786880185323),
                                     (0.05376726478993601, 1.7087857697464142, -0.9373861680976588),
                                     (2.032723739968267, -0.9262182582387579, 1.3967628167913386),
                                     (-3.2702388855837525, -0.4214502197073594, 1.0116457858191263))},
                  'index': 0, 'FF energy': -58.91943006722673, 'source': 'MMFF94',
                  'torsion_dihedrals': {(13, 1, 5, 2): 162.3401547589165, (12, 2, 5, 1): 221.93303981536653,
                                        (14, 3, 7, 6): 176.89271972908574, (6, 4, 5, 1): 63.38944206413592,
                                        (5, 4, 6, 7): 10.747867003395235}},
                 {'xyz': {'symbols': ('S', 'O', 'O', 'N', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H'),
                          'isotopes': (32, 16, 16, 14, 12, 12, 12, 1, 1, 1, 1, 1, 1, 1),
                          'coords': ((-2.2630067975987265, 1.7570992049576681, -0.2714867230167323),
                                     (-1.9035349945512323, -0.4686779657234163, 1.0856513630465972),
                                     (2.0086421655680256, -0.1468077158986447, -2.1094212366463174),
                                     (-0.2636653972177391, 0.060760334376423544, -0.3965146786620352),
                                     (-1.169304990163232, 0.6067990405891494, 0.5654038051474856),
                                     (0.8711133028078012, -0.6569585161001706, -0.09881641541894212),
                                     (1.9376516771684236, -0.7718363048482877, -0.8926198400840747),
                                     (-0.6470282827235821, 1.1229515687436151, 1.3768757293494518),
                                     (-0.13273466092936573, 0.6647070877488709, -1.20835854817492),
                                     (0.848381201118087, -1.1538433007417384, 0.8693847762014714),
                                     (2.815351987662298, -1.3493924431334008, -0.6351672832198746),
                                     (-1.82154740999548, -1.1085626770827177, 0.3575085806579677),
                                     (-3.1146579060437762, 1.868727394100491, 0.7555364506071955),
                                     (2.8343401048977714, -0.42496570698788333, -2.542966597331772))},
                  'index': 2, 'FF energy': -68.39144685068642, 'source': 'MMFF94',
                  'torsion_dihedrals': {(13, 1, 5, 2): 51.469450566143024, (12, 2, 5, 1): 93.78019119118923,
                                        (14, 3, 7, 6): 184.5102421075699, (6, 4, 5, 1): 167.16203608651372,
                                        (5, 4, 6, 7): 204.57856732311797}}]
        mol = ARCSpecies(label='C(O)(S)NC=CO', smiles='C(O)(S)NC=CO', xyz=confs[0]['xyz']).mol
        confs = conformers.determine_chirality(conformers=confs, label='C(O)(S)NC=CO', mol=mol)
        self.assertEqual(confs[0]['chirality'], {(0,): 'R', (3,): 'NR', (4, 5): 'E'})
        self.assertEqual(confs[1]['chirality'], {(0,): 'S', (3,): 'NR', (4, 5): 'Z'})

        confs = [{'xyz': converter.str_to_xyz("""C                  0.42854493    1.37396218   -0.06378771
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
                                                 H                  5.10577193    1.76670654    1.89747679"""),
                  'index': 0, 'FF energy': -58.91943006722673, 'source': 'MMFF94'},
                 {'xyz': converter.str_to_xyz("""C                  0.70455114    1.02471502   -0.59812334
                                                 H                 -0.29550712    0.65127249   -0.67109010
                                                 C                  0.99699795    2.01057459    0.28455600
                                                 H                  0.22784149    2.41551873    0.90850999
                                                 C                  2.43633411    2.54805275    0.38957346
                                                 H                  2.91991643    2.46069364   -0.56090880
                                                 O                  2.40440760    3.92231746    0.78362575
                                                 H                  1.91959416    4.43147118    0.12992978
                                                 N                  3.17870469    1.76841111    1.39054541
                                                 H                  3.20103091    0.80738684    1.11498438
                                                 O                  1.73248925    0.48352787   -1.43200566
                                                 H                  2.39027727    1.15999922   -1.60892777
                                                 O                  2.56405800    1.87944698    2.59863502
                                                 H                  2.54262482    2.80203029    2.86317361"""),
                  'index': 2, 'FF energy': -68.39144685068642, 'source': 'MMFF94'}]
        mol = ARCSpecies(label='S1', smiles='OC=CC(O)NO', xyz=confs[0]['xyz']).mol  # Preserves atom order.
        confs = conformers.determine_chirality(conformers=confs, label='S1', mol=mol)
        self.assertEqual(confs[0]['chirality'], {(0, 2): 'E', (4,): 'S', (8,): 'NR'})
        self.assertEqual(confs[1]['chirality'], {(0, 2): 'Z', (4,): 'R', (8,): 'NR'})

    def test_get_lowest_diastereomers(self):
        """Test the getting the lowest diasteroemrs from a given conformers list"""
        smiles = 'N=N'  # test chirality of a double bond between nitrogen atoms
        mol = Molecule(smiles=smiles)
        torsions, tops = conformers.determine_rotors([mol])
        confs = conformers.generate_force_field_conformers(mol_list=[mol], label=smiles, torsion_num=len(torsions),
                                                           charge=0, multiplicity=mol.multiplicity, num_confs=10)
        confs = conformers.determine_dihedrals(confs, torsions)
        diastereomeric_conformers = conformers.get_lowest_diastereomers(label=smiles, mol=mol, conformers=confs)
        diastereomers = [conf['chirality'] for conf in diastereomeric_conformers]
        self.assertEqual(len(diastereomers), 2)
        self.assertIn({(0, 1): 'E'}, diastereomers)
        self.assertIn({(0, 1): 'Z'}, diastereomers)

        smiles = 'C=C'  # test no chirality of a double bond with trivial identical groups
        mol = Molecule(smiles=smiles)
        torsions, tops = conformers.determine_rotors([mol])
        confs = conformers.generate_force_field_conformers(mol_list=[mol], label=smiles, torsion_num=len(torsions),
                                                           charge=0, multiplicity=mol.multiplicity, num_confs=10)
        confs = conformers.determine_dihedrals(confs, torsions)
        diastereomeric_conformers = conformers.get_lowest_diastereomers(label=smiles, mol=mol, conformers=confs)
        diastereomers = [conf['chirality'] for conf in diastereomeric_conformers]
        self.assertEqual(len(diastereomers), 1)
        self.assertIn({}, diastereomers)

        smiles = 'C1(Cl)C(Cl)CCCC1'  # test chiralities in a ring
        spc1 = ARCSpecies(label=smiles, smiles=smiles)
        lowest_confs = conformers.generate_conformers(mol_list=spc1.mol_list, label=spc1.label,
                                                      charge=spc1.charge, multiplicity=spc1.multiplicity,
                                                      force_field='MMFF94s', print_logs=False,
                                                      n_confs=10, return_all_conformers=False)
        diastereomers = list()
        for conf in lowest_confs:
            if conf['chirality'] not in diastereomers:
                diastereomers.append(conf['chirality'])
        self.assertEqual(len(diastereomers), 2)
        self.assertIn({(0,): 'S', (2,): 'S'}, diastereomers)
        self.assertIn({(0,): 'R', (2,): 'S'}, diastereomers)

        smiles = 'CC(Cl)C(Cl)CCC'  # test chiralities not in a ring
        spc1 = ARCSpecies(label=smiles, smiles=smiles)
        lowest_confs = conformers.generate_conformers(mol_list=spc1.mol_list, label=spc1.label,
                                                      charge=spc1.charge, multiplicity=spc1.multiplicity,
                                                      force_field='MMFF94s', print_logs=False,
                                                      n_confs=10, return_all_conformers=False)
        diastereomers = list()
        for conf in lowest_confs:
            if conf['chirality'] not in diastereomers:
                diastereomers.append(conf['chirality'])
        self.assertEqual(len(diastereomers), 2)
        self.assertIn({(1,): 'S', (3,): 'S'}, diastereomers)
        self.assertIn({(1,): 'R', (3,): 'S'}, diastereomers)

    def test_prune_enantiomers_dict(self):
        """Test pruning the enantiomers_dict, removing exact mirror images"""
        enantiomers_dict = {(((4,), 'S'), ((3,), 'NR'), ((5, 6), 'E')): {'FF energy': 10},
                            (((4,), 'S'), ((3,), 'NR'), ((5, 6), 'Z')): {'FF energy': 10},
                            (((4,), 'S'), ((3,), 'NS'), ((5, 6), 'E')): {'FF energy': 10},
                            (((4,), 'S'), ((3,), 'NS'), ((5, 6), 'Z')): {'FF energy': 10},
                            (((4,), 'R'), ((3,), 'NR'), ((5, 6), 'E')): {'FF energy': 10},
                            (((4,), 'R'), ((3,), 'NR'), ((5, 6), 'Z')): {'FF energy': 10},
                            (((4,), 'R'), ((3,), 'NS'), ((5, 6), 'E')): {'FF energy': 10},
                            (((4,), 'R'), ((3,), 'NS'), ((5, 6), 'Z')): {'FF energy': 10}}
        pruned_enantiomers_dict = conformers.prune_enantiomers_dict(label='label', enantiomers_dict=enantiomers_dict)
        expected_enantiomers_dict = {(((4,), 'S'), ((3,), 'NR'), ((5, 6), 'E')): {'FF energy': 10},
                                     (((4,), 'S'), ((3,), 'NS'), ((5, 6), 'E')): {'FF energy': 10},
                                     (((4,), 'S'), ((3,), 'NR'), ((5, 6), 'Z')): {'FF energy': 10},
                                     (((4,), 'S'), ((3,), 'NS'), ((5, 6), 'Z')): {'FF energy': 10}}
        self.assertEqual(pruned_enantiomers_dict, expected_enantiomers_dict)

        enantiomers_dict = {(((4,), 'S'), ((3,), 'NR'), ((5, 6), 'E')): {'FF energy': 10},
                            (((4,), 'S'), ((3,), 'NR'), ((5, 6), 'Z')): {'FF energy': None},
                            (((4,), 'S'), ((3,), 'NS'), ((5, 6), 'E')): {'FF energy': 20},
                            (((4,), 'S'), ((3,), 'NS'), ((5, 6), 'Z')): {'FF energy': 30},
                            (((4,), 'R'), ((3,), 'NR'), ((5, 6), 'Z')): {'FF energy': 50},
                            (((4,), 'R'), ((3,), 'NS'), ((5, 6), 'E')): {'FF energy': 50},
                            (((4,), 'R'), ((3,), 'NS'), ((5, 6), 'Z')): {'FF energy': 60}}
        # enantiomers_dict is missing (((4,), 'R'), ((3,), 'NR'), ((5, 6), 'E')) on purpose
        pruned_enantiomers_dict = conformers.prune_enantiomers_dict(label='label', enantiomers_dict=enantiomers_dict)
        expected_enantiomers_dict = {(((4,), 'S'), ((3,), 'NR'), ((5, 6), 'E')): {'FF energy': 10},
                                     (((4,), 'S'), ((3,), 'NS'), ((5, 6), 'E')): {'FF energy': 20},
                                     (((4,), 'S'), ((3,), 'NS'), ((5, 6), 'Z')): {'FF energy': 30},
                                     (((4,), 'R'), ((3,), 'NS'), ((5, 6), 'Z')): {'FF energy': 60}}
        self.assertEqual(pruned_enantiomers_dict, expected_enantiomers_dict)

    def test_inverse_chirality_symbol(self):
        """Test inversing the chirality symbol"""
        symbols = ['R', 'S', 'NR', 'NS', 'E', 'Z']
        expected_inversed_symbols = ['S', 'R', 'NS', 'NR', 'E', 'Z']  # 'E' and 'Z' are not inverted
        inversed_symbols = list()
        for symbol in symbols:
            inversed_symbols.append(conformers.inverse_chirality_symbol(symbol))
        self.assertEqual(inversed_symbols, expected_inversed_symbols)

    def test_chirality_dict_to_tuple(self):
        """Test generating a deterministic tupe from the conformer chirality dictionary"""
        chirality_dict = {(3,): 'NR', (4,): 'S', (5, 6): 'E'}
        chirality_tupe = conformers.chirality_dict_to_tuple(chirality_dict)
        expected_tuple = (((4,), 'S'), ((3,), 'NR'), ((5, 6), 'E'))
        self.assertEqual(chirality_tupe, expected_tuple)

        chirality_dict = {(3,): 'NR', (4,): 'S', (7,): 'NS', (2,): 'R', (21,): 'S', (19,): 'S',
                          (5, 10): 'E', (15, 6): 'Z', (9, 0): 'E', (18,): 'NR', (17,): 'S'}
        chirality_tupe = conformers.chirality_dict_to_tuple(chirality_dict)
        expected_tuple = (((2,), 'R'), ((4,), 'S'), ((17,), 'S'), ((19,), 'S'), ((21,), 'S'), ((3,), 'NR'),
                          ((7,), 'NS'), ((18,), 'NR'), ((0, 9), 'E'), ((5, 10), 'E'), ((6, 15), 'Z'))
        self.assertEqual(chirality_tupe, expected_tuple)

    def test_identify_chiral_nitrogen_centers(self):
        """Test identifying chiral nitrogen centers (umbrella modes)"""
        xyz1 = {'symbols': ('N', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'),
                'isotopes': (14, 12, 12, 12, 1, 1, 1, 1, 1, 1, 1, 1, 1),
                'coords': ((-0.7073511380114899, 0.8197164661073145, 0.24999885684842907),
                           (0.5801699241438923, 0.6591912227657216, -0.4240530456792972),
                           (1.4472113196665168, -0.4372777699985311, 0.17945348165229474),
                           (-1.6390090531407464, -0.25796648974989134, -0.04936095294345401),
                           (1.1197404668780022, 1.6093134315498747, -0.3376879040777191),
                           (0.43764603823503984, 0.4845854286631773, -1.4968921979101455),
                           (1.002550211214117, -1.4275789874428901, 0.04242740542022188),
                           (2.4294750170597803, -0.4452330671213071, -0.30432398624467893),
                           (1.6034105333687751, -0.27376799189464784, 1.2509388990817716),
                           (-1.8125204547484717, -0.3462467078827822, -1.1266788056658708),
                           (-2.6039691776051646, -0.041004688407494536, 0.41960198456069625),
                           (-1.2927485939442889, -1.2203699903546943, 0.338772805584756),
                           (-0.5646050931159982, 0.8766391437661488, 1.2578034593729632))}
        spc1 = ARCSpecies(label='CNCC', smiles='CNCC', xyz=xyz1)
        nitrogen_chiral_centers1 = conformers.identify_chiral_nitrogen_centers(spc1.mol)
        self.assertEqual(nitrogen_chiral_centers1, [0])

        xyz2 = {'symbols': ('S', 'O', 'N', 'N', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'),
                'isotopes': (32, 16, 14, 14, 12, 12, 12, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
                'coords': ((2.951110081218288, 0.671911406761362, -1.5025398129041452),
                           (1.4428046314202434, -1.3647155681966856, -1.5268897885735253),
                           (-0.4152168782986968, 0.025751653448532214, 0.45917210342119374),
                           (1.6705192648554394, -0.12214480762739517, -0.8054536444235497),
                           (0.35542387239917184, 0.5372476906230289, -0.6902625247641537),
                           (-1.466940215540971, 0.985917957418011, 0.8075193590264669),
                           (-1.0271484513526679, -1.273946812093523, 0.17969565674753799),
                           (0.5461028243490972, 1.603991362257306, -0.5090188121480972),
                           (-0.21201202121287238, 0.4836947814643129, -1.6299428423709177),
                           (-1.0328536170772404, 1.9522587589275429, 1.0882227785532794),
                           (-2.033304535282507, 0.6398190872369356, 1.6796892222352768),
                           (-2.1692275497427262, 1.1480293716036831, -0.018477509854500935),
                           (-1.639110479495997, -1.2598631931458917, -0.7294144620439414),
                           (-0.2733872489872659, -2.060305256507347, 0.0864655449663449),
                           (-1.6680746888672895, -1.5846364964768231, 1.0135559968297658),
                           (3.141072684518697, 1.6068921838382135, -0.5574948624946222),
                           (1.8302423270979375, -1.989902119530246, -0.88864367977472))}
        spc2 = ARCSpecies(label='SN(O)CN(C)C', smiles='SN(O)CN(C)C', xyz=xyz2)
        nitrogen_chiral_centers2 = conformers.identify_chiral_nitrogen_centers(spc2.mol)
        self.assertEqual(nitrogen_chiral_centers2, [3])

        mol3 = Molecule(smiles='OC(N)N')
        nitrogen_chiral_centers3 = conformers.identify_chiral_nitrogen_centers(mol3)
        self.assertEqual(nitrogen_chiral_centers3, [])

        mol4 = Molecule(smiles='COC1=C(C=C(C(C)C)C=C1)CN[C@H]2C3CCN([C@H]2C(C4=CC=CC=C4)C5=CC=CC=C5)CC3')
        nitrogen_chiral_centers4 = conformers.identify_chiral_nitrogen_centers(mol4)
        self.assertEqual(nitrogen_chiral_centers4, [12])
        self.assertEqual(mol4.atoms[19].symbol, 'C')
        self.assertEqual(mol4.atoms[12].symbol, 'N')
        self.assertEqual(mol4.atoms[13].symbol, 'C')

        mol5 = Molecule(smiles='NNNN')  # two chiral Ns
        nitrogen_chiral_centers5 = conformers.identify_chiral_nitrogen_centers(mol5)
        self.assertEqual(nitrogen_chiral_centers5, [1, 2])
        self.assertEqual(mol5.atoms[1].symbol, 'N')
        self.assertEqual(mol5.atoms[2].symbol, 'N')

        mol6 = Molecule(smiles='NCC')  # no chiral N
        nitrogen_chiral_centers6 = conformers.identify_chiral_nitrogen_centers(mol6)
        self.assertEqual(nitrogen_chiral_centers6, [])

        mol7 = Molecule(smiles='FN[3H]')  # no umbrella modes
        nitrogen_chiral_centers7 = conformers.identify_chiral_nitrogen_centers(mol7)
        self.assertEqual(nitrogen_chiral_centers7, [1])

    def test_replace_n_with_c_in_mol(self):
        """Test replacing N with C-H in a molecule"""
        mol = Molecule(smiles='CN(CC)CCC')
        new_mol, inserted_elements = conformers.replace_n_with_c_in_mol(
            mol, conformers.identify_chiral_nitrogen_centers(mol))
        self.assertEqual(new_mol.to_smiles(), 'CCCC(C)CC')
        self.assertEqual(inserted_elements, ['H'])

        mol = Molecule(smiles='CNCC')
        new_mol, inserted_elements = conformers.replace_n_with_c_in_mol(
            mol, conformers.identify_chiral_nitrogen_centers(mol))
        self.assertEqual(new_mol.to_smiles(), 'CCC(C)F')
        self.assertEqual(inserted_elements, ['F'])

        mol = Molecule(smiles='CNCCN')
        new_mol, inserted_elements = conformers.replace_n_with_c_in_mol(
            mol, conformers.identify_chiral_nitrogen_centers(mol))
        self.assertEqual(new_mol.to_smiles(), 'CC(CCN)F')
        self.assertEqual(inserted_elements, ['F'])

        mol = Molecule(smiles='NNNN')
        new_mol, inserted_elements = conformers.replace_n_with_c_in_mol(
            mol, conformers.identify_chiral_nitrogen_centers(mol))
        self.assertEqual(new_mol.to_smiles(), 'NC(C(F)N)F')
        self.assertEqual(inserted_elements, ['F', 'F'])

        mol = Molecule(smiles='[2H]N[3H]')
        new_mol, inserted_elements = conformers.replace_n_with_c_in_mol(
            mol, conformers.identify_chiral_nitrogen_centers(mol))
        self.assertEqual(new_mol.to_smiles(), '[2H]C([3H])F')
        self.assertEqual(inserted_elements, ['F'])

        mol = Molecule(smiles='FN[3H]')
        new_mol, inserted_elements = conformers.replace_n_with_c_in_mol(
            mol, conformers.identify_chiral_nitrogen_centers(mol))
        self.assertEqual(new_mol.to_smiles(), '[3H]C(F)Cl')
        self.assertEqual(inserted_elements, ['Cl'])

        mol = Molecule(smiles='FNCl')
        new_mol, inserted_elements = conformers.replace_n_with_c_in_mol(
            mol, conformers.identify_chiral_nitrogen_centers(mol))
        self.assertEqual(new_mol.to_smiles(), 'FC(Cl)I')
        self.assertEqual(inserted_elements, ['I'])

        mol = Molecule(smiles='C(O)(S)NC=CO')
        new_mol, inserted_elements = conformers.replace_n_with_c_in_mol(
            mol, conformers.identify_chiral_nitrogen_centers(mol))
        self.assertEqual(inserted_elements, ['F'])
        self.assertEqual(len(mol.atoms), 14)
        self.assertEqual(len(new_mol.atoms), 15)
        self.assertEqual(new_mol.atoms[-1].symbol, 'F')
        # test that atom order is preserved in new_mol:
        for i in range(len(mol.atoms)):
            if not mol.atoms[i].is_nitrogen():
                self.assertEqual(new_mol.atoms[i].symbol, mol.atoms[i].symbol)
            else:
                self.assertEqual(new_mol.atoms[i].symbol, 'C')
        self.assertEqual(new_mol.to_smiles(), 'OC=CC(C(S)O)F')  # changes atom order, check last

    def test_replace_n_with_c_in_xyz(self):
        """Test replacing N with C-H in xyz"""
        xyz = {'symbols': ('N', 'C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H',
                           'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'),
               'isotopes': (14, 12, 12, 12, 12, 12, 12, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
               'coords': ((-0.576807794430736, 0.35198590827267795, -0.028013114446316406),
                          (1.921586273854039, 0.5047436065860704, -0.26969327980033797),
                          (0.7222949301042161, -0.28165278619109574, 0.27594152797609256),
                          (-1.6378200711465631, -0.297429478240597, 0.7656657521694586),
                          (3.2374101833920235, -0.10256608475632271, 0.19663891893480523),
                          (-2.965489531748123, 0.45325206867425405, 0.7284587100940173),
                          (-0.876124908772818, 0.2923437697671098, -1.4623448616637884),
                          (1.8700624454501567, 1.5470233325916922, 0.06688774247332803),
                          (1.918726140695422, 0.512576418938087, -1.3645338051195977),
                          (-1.3300996673476249, -0.32959824543996896, 1.8187131713232585),
                          (-1.7898221300327064, -1.3363278728424122, 0.44648744289534936),
                          (0.8440505869687669, -0.3361546092391798, 1.365858492113345),
                          (0.7476583283721643, -1.3140299215686377, -0.09713473154426461),
                          (-2.830544089155255, 1.5079013999630924, 0.9904167855100331),
                          (-3.4422698361474744, 0.39593283373644217, -0.25429298060138067),
                          (-3.6600627566418167, 0.015223612299218104, 1.4531890487832204),
                          (3.3332290815959014, -1.1384601906926344, -0.1440670424796236),
                          (4.080973845176929, 0.4672259191442818, -0.2055354199821314),
                          (3.3106135732933084, -0.09096382724759984, 1.2888028722851448),
                          (-1.8177291594054494, 0.7995852305193636, -1.6943141141031526),
                          (-0.11858048074138193, 0.8185896685737892, -2.050874557508544),
                          (-0.9412549633329137, -0.7392988091007344, -1.8262565573090315))}
        spc = ARCSpecies(label='CN(CC)CCC', smiles='CN(CC)CCC', xyz=xyz)
        chiral_nitrogen_centers = conformers.identify_chiral_nitrogen_centers(spc.mol)
        elements_to_insert = conformers.replace_n_with_c_in_mol(spc.mol, chiral_nitrogen_centers)[1]
        new_xyz = conformers.replace_n_with_c_in_xyz(label=spc.label, mol=spc.mol, xyz=xyz,
                                                     chiral_nitrogen_centers=chiral_nitrogen_centers,
                                                     elements_to_insert=elements_to_insert)
        expected_xyz = {'symbols': ('C', 'C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H',
                                    'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'),
                        'isotopes': (12, 12, 12, 12, 12, 12, 12, 1, 1, 1, 1,
                                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
                        'coords': ((-0.576807794430736, 0.35198590827267795, -0.028013114446316406),
                                   (1.921586273854039, 0.5047436065860704, -0.26969327980033797),
                                   (0.7222949301042161, -0.28165278619109574, 0.27594152797609256),
                                   (-1.6378200711465631, -0.297429478240597, 0.7656657521694586),
                                   (3.2374101833920235, -0.10256608475632271, 0.19663891893480523),
                                   (-2.965489531748123, 0.45325206867425405, 0.7284587100940173),
                                   (-0.876124908772818, 0.2923437697671098, -1.4623448616637884),
                                   (1.8700624454501567, 1.5470233325916922, 0.06688774247332803),
                                   (1.918726140695422, 0.512576418938087, -1.3645338051195977),
                                   (-1.3300996673476249, -0.32959824543996896, 1.8187131713232585),
                                   (-1.7898221300327064, -1.3363278728424122, 0.44648744289534936),
                                   (0.8440505869687669, -0.3361546092391798, 1.365858492113345),
                                   (0.7476583283721643, -1.3140299215686377, -0.09713473154426461),
                                   (-2.830544089155255, 1.5079013999630924, 0.9904167855100331),
                                   (-3.4422698361474744, 0.39593283373644217, -0.25429298060138067),
                                   (-3.6600627566418167, 0.015223612299218104, 1.4531890487832204),
                                   (3.3332290815959014, -1.1384601906926344, -0.1440670424796236),
                                   (4.080973845176929, 0.4672259191442818, -0.2055354199821314),
                                   (3.3106135732933084, -0.09096382724759984, 1.2888028722851448),
                                   (-1.8177291594054494, 0.7995852305193636, -1.6943141141031526),
                                   (-0.11858048074138193, 0.8185896685737892, -2.050874557508544),
                                   (-0.9412549633329137, -0.7392988091007344, -1.8262565573090315),
                                   (-0.5258187657388067, 1.4160407364858265, 0.24619017056197964))}
        self.assertEqual(new_xyz, expected_xyz)

    def test_get_top_element_count(self):
        """Test getting an element and isotope count for top atoms in a molecule"""
        mol = Molecule(smiles='[2H]N[3H]CCCS')
        top = list(range(10))
        top_element_count = conformers.get_top_element_count(mol, top)
        expected_top_element_count = {('C', -1): 3, ('H', -1): 3, ('H', 2): 1, ('H', 3): 1, ('N', -1): 1, ('S', -1): 1}
        self.assertEqual(top_element_count, expected_top_element_count)

    def test_preserve_atom_order(self):
        """Test that atom order is being preserved when generating conformers"""
        spc1 = ARCSpecies(label='formic_acid', smiles='O=CO')
        lowest_confs = conformers.generate_conformers(mol_list=spc1.mol_list, label=spc1.label, charge=spc1.charge,
                                                      multiplicity=spc1.multiplicity, print_logs=False, n_confs=1)
        for mol_atom, conf_atom in zip(spc1.mol.atoms, lowest_confs[0]['xyz']['symbols']):
            self.assertEqual(mol_atom.element.symbol, conf_atom)

        spc1.generate_conformers(n_confs=1, plot_path=None)
        for mol_atom, conf_atom in zip(spc1.mol.atoms, spc1.conformers[0]['symbols']):
            self.assertEqual(mol_atom.element.symbol, conf_atom)

    def test_generate_monoatomic_conformer(self):
        """Test generating a monoatomic conformer"""
        conf = conformers.generate_monoatomic_conformer('N')
        expected_conf = {'xyz': {'symbols': ('N',), 'isotopes': (14,), 'coords': ((0.0, 0.0, 0.0),)},
                         'index': 0, 'FF energy': 0.0, 'chirality': None,
                         'source': 'monoatomic species', 'torsion_dihedrals': None}
        self.assertEqual(conf, expected_conf)

    def test_determine_smallest_atom_index_in_scan(self):
        """Test determine_smallest_atom_index_in_scan()"""
        xyz_0 = """O                  2.26520400   -1.90260200   -2.47777200
                   O                  1.51800600   -1.97132200   -1.39924100
                   C                  1.83952600   -0.92109900   -0.43295700
                   C                  3.09267100   -1.35786700    0.33590700
                   N                  3.61417000   -0.30259800    1.19303500
                   C                  4.41571600   -0.83324600    2.28631100
                   C                  4.34546000    0.71666100    0.45207800
                   C                  0.62056900   -0.75704800    0.46435900
                   N                 -0.56235500   -0.18762100   -0.18174300
                   H                  2.03054900   -0.02361500   -1.01570000
                   H                  3.83658000   -1.70694600   -0.39693000
                   H                  2.82985700   -2.21452800    0.96004800
                   H                  5.32392600   -1.35409800    1.93989400
                   H                  4.72583400   -0.01814900    2.94028200
                   H                  3.82560400   -1.53427200    2.87628600
                   H                  3.71739800    1.16256600   -0.31702000
                   H                  5.24879000    0.31495900   -0.03601700
                   H                  4.64979400    1.51214900    1.13174700
                   H                  0.34943300   -1.73462900    0.86289600
                   H                  0.94533300   -0.14188300    1.31211800
                   H                 -1.43404175   -0.66435521   -0.06822342
                   H                 -0.53193587    0.79042827   -0.38788446"""
        mol_0 = ARCSpecies(label='imipramine_3_peroxy', xyz=xyz_0).mol
        self.assertEqual(conformers.determine_smallest_atom_index_in_scan(
            atom1=mol_0.atoms[2],
            atom2=mol_0.atoms[3],
            mol=mol_0), 2)
        self.assertEqual(conformers.determine_smallest_atom_index_in_scan(
            atom1=mol_0.atoms[3],
            atom2=mol_0.atoms[2],
            mol=mol_0), 5)

        xyz_1 = """N                  4.80913900   -0.18142200   -0.31894800
                   C                  3.70376300   -1.17433400   -0.29502100
                   C                  2.45778600   -0.74850100    0.50058600
                   H                  4.10978900   -2.11991000    0.13095600
                   C                  1.62116900    0.29476100   -0.25698800
                   H                  1.84074900   -1.64060500    0.72043500
                   H                  2.74746400   -0.34556500    1.49184900
                   N                  0.56572800    0.83928800    0.66053500
                   H                  2.27298300    1.13086100   -0.58875700
                   C                 -0.58051700   -0.06632200    0.80490800
                   C                  0.13746100    2.21077400    0.25626800
                   H                 -1.35936800    0.37327500    1.45247800
                   H                 -0.25811600   -1.06042800    1.17657000
                   H                 -0.49831100    2.63509900    1.05018600
                   H                 -0.43367100    2.23564600   -0.68857500
                   H                  1.01817400    2.86272700    0.15917800
                   H                  5.11146200    0.04022000    0.62542100
                   H                  4.49938400    0.69076900   -0.73712200
                   H                  3.43073500   -1.40966400   -1.34806700
                   H                  1.18593500   -0.15003100   -1.17952600
                   O                 -1.18790200   -0.45549200   -0.48823800
                   O                 -2.01122000    0.48874300   -0.93507100
                   H                 -3.16340600    0.19846300   -0.51358000
                   C                 -4.45612600   -0.13163000   -0.38252500
                   H                 -4.89014100    0.77163700    0.03528500
                   H                 -4.72782500   -0.41388400   -1.39028500
                   O                 -4.49534800   -1.12219800    0.58015100
                   H                 -4.25535300   -2.00057000    0.19434000"""
        spc_1 = ARCSpecies(label='imipramine_3_peroxy', xyz=xyz_1, is_ts=True)
        spc_1.mol_from_xyz()
        mol_1 = spc_1.mol
        self.assertEqual(conformers.determine_smallest_atom_index_in_scan(
            atom1=mol_1.atoms[7],
            atom2=mol_1.atoms[4],
            mol=mol_1), 10)
        self.assertEqual(conformers.determine_smallest_atom_index_in_scan(
            atom1=mol_1.atoms[21],
            atom2=mol_1.atoms[20],
            mol=mol_1), 23)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
