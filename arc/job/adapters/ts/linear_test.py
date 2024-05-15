#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.job.adapters.ts.heuristics module
"""

import os
import shutil
import unittest

from rmgpy.data.kinetics import KineticsFamily

from arc.common import ARC_PATH, almost_equal_coords
from arc.job.adapters.ts.linear import (LinearAdapter,
                                        average_zmat_params,
                                        get_r_constraints,
                                        get_rxn_weight,
                                        get_weight,
                                        interpolate_isomerization,
                                        )
from arc.reaction import ARCReaction
from arc.rmgdb import make_rmg_database_object, load_families_only
from arc.species.converter import str_to_xyz, xyz_to_str
from arc.species.species import ARCSpecies
from arc.species.zmat import _compare_zmats


class TestHeuristicsAdapter(unittest.TestCase):
    """
    Contains unit tests for the HeuristicsAdapter class.
    """

    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None
        cls.rmgdb = make_rmg_database_object()
        load_families_only(cls.rmgdb)

        cls.rxn_1 = ARCReaction(r_species=[ARCSpecies(label='CPD', smiles='C1C=CC=C1',
                                                      xyz="""C      -1.11689933   -0.16076292   -0.17157587
                                                             C      -0.34122713    1.12302797   -0.12498608
                                                             C       0.95393962    0.86179733    0.10168911
                                                             C       1.14045506   -0.56033684    0.22004768
                                                             C      -0.03946631   -1.17782376    0.06650470
                                                             H      -1.58827673   -0.30386166   -1.14815401
                                                             H      -1.87502410   -0.19463481    0.61612857
                                                             H      -0.77193310    2.10401684   -0.25572143
                                                             H       1.74801386    1.58807889    0.18578522
                                                             H       2.09208098   -1.03534789    0.40412258
                                                             H      -0.20166282   -2.24415315    0.10615953""")],
                                p_species=[ARCSpecies(label='C5_carbene', adjlist="""1  C u0 p1 c0 {2,S} {6,S}
                                                                                     2  C u0 p0 c0 {1,S} {3,D} {7,S}
                                                                                     3  C u0 p0 c0 {2,D} {4,S} {8,S}
                                                                                     4  C u0 p0 c0 {3,S} {5,D} {9,S}
                                                                                     5  C u0 p0 c0 {4,D} {10,S} {11,S}
                                                                                     6  H u0 p0 c0 {1,S}
                                                                                     7  H u0 p0 c0 {2,S}
                                                                                     8  H u0 p0 c0 {3,S}
                                                                                     9  H u0 p0 c0 {4,S}
                                                                                     10 H u0 p0 c0 {5,S}
                                                                                     11 H u0 p0 c0 {5,S}""",
                                                      xyz="""C       2.62023459    0.49362130   -0.23013873
                                                             C       1.48006570   -0.33866786   -0.38699247
                                                             C       1.53457595   -1.45115429   -1.13132450
                                                             C       0.40179762   -2.32741928   -1.31937443
                                                             C       0.45595744   -3.43865596   -2.06277224
                                                             H       3.47507694    1.11901971   -0.11163109
                                                             H       0.56454036   -0.04212124    0.11659958
                                                             H       2.46516705   -1.72493574   -1.62516589
                                                             H      -0.53390611   -2.06386676   -0.83047533
                                                             H      -0.42088759   -4.06846526   -2.17670487
                                                             H       1.36205133   -3.75009763   -2.57288841""")])
        cls.rxn_1.determine_family(rmg_database=cls.rmgdb)

        cls.rxn_2 = ARCReaction(r_species=[ARCSpecies(label='CCONO', smiles='CCON=O',
                                                      xyz="""C      -1.36894499    0.07118059   -0.24801399
                                                             C      -0.01369535    0.17184136    0.42591278
                                                             O      -0.03967083   -0.62462610    1.60609048
                                                             N       1.23538512   -0.53558048    2.24863846
                                                             O       1.25629155   -1.21389295    3.27993827
                                                             H      -2.16063255    0.41812452    0.42429392
                                                             H      -1.39509985    0.66980796   -1.16284741
                                                             H      -1.59800183   -0.96960842   -0.49986392
                                                             H       0.19191326    1.21800574    0.68271847
                                                             H       0.76371340   -0.19234475   -0.25650067""")],
                                p_species=[ARCSpecies(label='CCNO2', smiles='CC[N+](=O)[O-]',
                                                      xyz="""C      -1.12362739   -0.04664655   -0.08575959
                                                             C       0.24488022   -0.51587553    0.36119196
                                                             N       0.57726975   -1.77875156   -0.37104243
                                                             O       1.16476543   -1.66382529   -1.45384186
                                                             O       0.24561669   -2.84385320    0.16410116
                                                             H      -1.87655344   -0.80826847    0.13962125
                                                             H      -1.14729169    0.14493421   -1.16405294
                                                             H      -1.41423043    0.87863077    0.42354512
                                                             H       1.02430791    0.21530309    0.12674144
                                                             H       0.27058353   -0.73979548    1.43184405""")])
        cls.rxn_2.determine_family(rmg_database=cls.rmgdb)

    def test_average_zmat_params(self):
        """Test the average_zmat_params() function."""
        zmat_1 = {'symbols': ('H', 'H'),
                  'coords': ((None, None, None),
                             ('R_1_0', None, None)),
                  'vars': {'R_1_0': 0.7},
                  'map': {0: 0, 1: 1}}
        zmat_2 = {'symbols': ('H', 'H'),
                  'coords': ((None, None, None),
                             ('R_1_0', None, None)),
                  'vars': {'R_1_0': 1.3},
                  'map': {0: 0, 1: 1}}
        expected_zmat = {'symbols': ('H', 'H'),
                         'coords': ((None, None, None),
                                    ('R_1_0', None, None)),
                         'vars': {'R_1_0': 1.0},
                         'map': {0: 0, 1: 1}}
        zmat = average_zmat_params(zmat_1, zmat_2)
        self.assertTrue(_compare_zmats(zmat, expected_zmat))

        expected_zmat = {'symbols': ('H', 'H'),
                         'coords': ((None, None, None),
                                    ('R_1_0', None, None)),
                         'vars': {'R_1_0': 0.85},
                         'map': {0: 0, 1: 1}}
        zmat = average_zmat_params(zmat_1, zmat_2, weight=0.25)
        self.assertTrue(_compare_zmats(zmat, expected_zmat))
        zmat = average_zmat_params(zmat_2, zmat_1, weight=0.75)
        self.assertTrue(_compare_zmats(zmat, expected_zmat))

        zmat_1 = {'symbols': ('C', 'N', 'H', 'H', 'H', 'H', 'H'),
                  'coords': ((None, None, None), ('R_1_0', None, None), ('R_2|4_0|0', 'A_2|4_0|0_1|1', None),
                             ('R_3|6_1|1', 'A_3|6_1|1_0|0', 'D_3_1_0_2'),
                             ('R_2|4_0|0', 'A_2|4_0|0_1|1', 'D_4_0_1_3'), ('R_5_0', 'A_5_0_1', 'D_5_0_1_4'),
                             ('R_3|6_1|1', 'A_3|6_1|1_0|0', 'D_6_1_0_5')),
                  'vars': {'R_1_0': 1.451965854148702, 'D_3_1_0_2': 60.83821034525936,
                           'D_4_0_1_3': 301.30263742432356, 'R_5_0': 1.0936965384360282,
                           'A_5_0_1': 110.59878027260544, 'D_5_0_1_4': 239.76779188408136,
                           'D_6_1_0_5': 65.17113681053117, 'R_2|4_0|0': 1.0935188594180785,
                           'R_3|6_1|1': 1.019169330302324, 'A_2|4_0|0_1|1': 110.20495980110817,
                           'A_3|6_1|1_0|0': 109.41187648524644},
                  'map': {0: 0, 1: 1, 2: 2, 3: 5, 4: 3, 5: 4, 6: 6}}
        zmat_2 = {'symbols': ('C', 'N', 'H', 'H', 'H', 'H', 'H'),
                  'coords': ((None, None, None), ('R_1_0', None, None), ('R_2|4_0|0', 'A_2|4_0|0_1|1', None),
                             ('R_3|6_1|1', 'A_3|6_1|1_0|0', 'D_3_1_0_2'),
                             ('R_2|4_0|0', 'A_2|4_0|0_1|1', 'D_4_0_1_3'), ('R_5_0', 'A_5_0_1', 'D_5_0_1_4'),
                             ('R_3|6_1|1', 'A_3|6_1|1_0|0', 'D_6_1_0_5')),
                  'vars': {'R_1_0': 1.2, 'D_3_1_0_2': 50,
                           'D_4_0_1_3': 250, 'R_5_0': 1.0936965384360282,
                           'A_5_0_1': 110.59878027260544, 'D_5_0_1_4': 239.76779188408136,
                           'D_6_1_0_5': 120, 'R_2|4_0|0': 1.0935188594180785,
                           'R_3|6_1|1': 1.6, 'A_2|4_0|0_1|1': 110.20495980110817,
                           'A_3|6_1|1_0|0': 109.41187648524644},
                  'map': {0: 0, 1: 1, 2: 2, 3: 5, 4: 3, 5: 4, 6: 6}}
        expected_zmat = {'symbols': ('C', 'N', 'H', 'H', 'H', 'H', 'H'),
                         'coords': ((None, None, None), ('R_1_0', None, None), ('R_2|4_0|0', 'A_2|4_0|0_1|1', None),
                                    ('R_3|6_1|1', 'A_3|6_1|1_0|0', 'D_3_1_0_2'), ('R_2|4_0|0', 'A_2|4_0|0_1|1', 'D_4_0_1_3'),
                                    ('R_5_0', 'A_5_0_1', 'D_5_0_1_4'), ('R_3|6_1|1', 'A_3|6_1|1_0|0', 'D_6_1_0_5')),
                         'vars': {'R_1_0': 1.3259829270743508, 'D_3_1_0_2': 55.419105172629685,
                                  'D_4_0_1_3': 275.6513187121618, 'R_5_0': 1.0936965384360282,
                                  'A_5_0_1': 110.59878027260544, 'D_5_0_1_4': 239.76779188408136,
                                  'D_6_1_0_5': 92.58556840526558, 'R_2|4_0|0': 1.0935188594180785,
                                  'R_3|6_1|1': 1.309584665151162, 'A_2|4_0|0_1|1': 110.20495980110817,
                                  'A_3|6_1|1_0|0': 109.41187648524644},
                         'map': {0: 0, 1: 1, 2: 2, 3: 5, 4: 3, 5: 4, 6: 6}}
        zmat = average_zmat_params(zmat_1, zmat_2)
        self.assertTrue(_compare_zmats(zmat, expected_zmat))

    def test_get_weight(self):
        """Test the get_weight() function."""
        self.assertEqual(get_weight([0], [0], 4), 0.5)  # 4 / 8
        self.assertEqual(get_weight([0], [8], 12), 0.75)  # 12 / 20
        self.assertEqual(get_weight([0], [2], 6), 0.6)  # 6 / 10
        self.assertEqual(get_weight([10], [0], 30), 0.4)  # 20 / 50
        self.assertEqual(get_weight([20], [10], 40), 0.4)  # 20 / 50
        self.assertIsNone(get_weight([20], [None], 40), 0.4)  # 20 / 50
        self.assertEqual(get_weight([8, 2], [0], 30), 0.4)  # 20 / 50
        self.assertEqual(get_weight([4, 1], [5.5, 1.5], 11), 0.6)  # 6 / 10

    def test_get_rxn_weight(self):
        """Test the get_rxn_weight() function."""
        rxn_1 = ARCReaction(r_species=[ARCSpecies(label='HO2', smiles='[O]O'),
                                       ARCSpecies(label='NH', smiles='[NH]')],
                            p_species=[ARCSpecies(label='N', smiles='[N]'),
                                       ARCSpecies(label='H2O2', smiles='OO')])
        rxn_1.r_species[0].e0 = 252.0
        rxn_1.r_species[1].e0 = 100.5
        rxn_1.p_species[0].e0 = 116.0
        rxn_1.p_species[1].e0 = 200.3
        rxn_1.ts_species = ARCSpecies(label='TS', is_ts=True)
        rxn_1.ts_species.e0 = 391.6
        self.assertAlmostEquals(get_rxn_weight(rxn_1), 0.3417832)

    def test_interpolate_isomerization_intra_h_migration(self):
        """Test the interpolate_isomerization() function for intra H migration reactions."""
        nc3h7_xyz = """C                  0.00375165   -0.48895802   -1.20586379
                       C                  0.00375165   -0.48895802    0.28487510
                       C                  0.00375165    0.91997987    0.85403684
                       H                  0.41748586   -1.33492098   -1.74315104
                       H                 -0.57506729    0.24145491   -1.76006154
                       H                 -0.87717095   -1.03203740    0.64280162
                       H                  0.88948616   -1.02465371    0.64296621
                       H                  0.88512433    1.48038223    0.52412379
                       H                  0.01450405    0.88584135    1.94817394
                       H                 -0.88837301    1.47376959    0.54233121"""
        ic3h7_xyz = """C                 -0.40735690   -0.74240205   -0.34312948
                       C                  0.38155377   -0.25604705    0.82450968
                       C                  0.54634593    1.25448345    0.81064511
                       H                  0.00637731   -1.58836501   -0.88041673
                       H                 -0.98617584   -0.01198912   -0.89732723
                       H                 -1.29710684   -1.29092340    0.08598983
                       H                  1.36955428   -0.72869684    0.81102246
                       H                  1.06044877    1.58846788   -0.09702437
                       H                  1.13774084    1.57830484    1.67308862
                       H                 -0.42424546    1.75989927    0.85794283"""
        nc3h7 = ARCSpecies(label='nC3H7', smiles='[CH2]CC', xyz=nc3h7_xyz)
        ic3h7 = ARCSpecies(label='iC3H7', smiles='C[CH]C', xyz=ic3h7_xyz)
        rxn = ARCReaction(r_species=[nc3h7], p_species=[ic3h7])
        expected_ts_xyz = str_to_xyz("""C       0.00598652   -0.48762088   -1.18600054
                                        C       0.00598652   -0.48762088    0.30473835
                                        C       0.00598652    0.92131703    0.87390011
                                        H       0.57807817   -1.25594905   -1.69382911
                                        H      -0.42698663    0.35443110   -1.71434916
                                        H      -1.27461406   -1.27709743   -0.24121083
                                        H       0.89172104   -1.02331658    0.66282944
                                        H       0.88735917    1.48171935    0.54398704
                                        H       0.01673891    0.88717852    1.96803717
                                        H      -0.88613815    1.47510670    0.56219446""")
        ts_xyzs = interpolate_isomerization(rxn, use_weights=False)
        self.assertEqual(len(ts_xyzs), 2)
        self.assertTrue(almost_equal_coords(ts_xyzs[0], expected_ts_xyz))

        nc3h7.e0 = 101.55
        ic3h7.e0 = 88.91
        ts = ARCSpecies(label='TS', is_ts=True, multiplicity=2, xyz=expected_ts_xyz)
        ts.e0 = 105
        rxn.ts_species = ts
        expected_ts_xyz = str_to_xyz("""C       0.00591772   -0.48764618   -1.20069282
                                        C       0.00591772   -0.48764618    0.29004607
                                        C       0.00591772    0.92129176    0.85920784
                                        H       0.47693974   -1.30982443   -1.72763512
                                        H      -0.52424330    0.28530048   -1.74580953
                                        H      -1.07348606   -1.15308709    0.40763997
                                        H       0.89165221   -1.02334186    0.64813718
                                        H       0.88729039    1.48169408    0.52929476
                                        H       0.01667012    0.88715326    1.95334482
                                        H      -0.88620694    1.47508143    0.54750218""")
        ts_xyzs = interpolate_isomerization(rxn, use_weights=True)
        self.assertEqual(len(ts_xyzs), 2)
        self.assertTrue(almost_equal_coords(ts_xyzs[0], expected_ts_xyz))

        # r_xyz = """C      -1.05582103   -0.03329574   -0.10080257
        #            C       0.41792695    0.17831205    0.21035514
        #            O       1.19234020   -0.65389683   -0.61111443
        #            O       2.44749684   -0.41401220   -0.28381363
        #            H      -1.33614002   -1.09151783    0.08714882
        #            H      -1.25953618    0.21489046   -1.16411897
        #            H      -1.67410396    0.62341419    0.54699514
        #            H       0.59566350   -0.06437686    1.28256640
        #            H       0.67254676    1.24676329    0.02676370"""
        # p_xyz = """C      -1.40886397    0.22567351   -0.37379668
        #            C       0.06280787    0.04097694   -0.38515682
        #            O       0.44130326   -0.57668419    0.84260864
        #            O       1.89519755   -0.66754203    0.80966180
        #            H      -1.87218376    0.90693511   -1.07582340
        #            H      -2.03646287   -0.44342165    0.20255768
        #            H       0.35571681   -0.60165457   -1.22096147
        #            H       0.56095122    1.01161503   -0.47393734
        #            H       2.05354047   -0.10415729    1.58865243"""
        # r = ARCSpecies(label='R', smiles='CCO[O]', xyz=r_xyz)
        # p = ARCSpecies(label='P', smiles='[CH2]COO', xyz=p_xyz)
        # rxn = ARCReaction(r_species=[r], p_species=[p])
        # expected_ts_xyz = str_to_xyz("""C       0.00598652   -0.48762088   -1.18600054
        #                                 C       0.00598652   -0.48762088    0.30473835
        #                                 C       0.00598652    0.92131703    0.87390011
        #                                 H       0.57807817   -1.25594905   -1.69382911
        #                                 H      -0.42698663    0.35443110   -1.71434916
        #                                 H      -1.27461406   -1.27709743   -0.24121083
        #                                 H       0.89172104   -1.02331658    0.66282944
        #                                 H       0.88735917    1.48171935    0.54398704
        #                                 H       0.01673891    0.88717852    1.96803717
        #                                 H      -0.88613815    1.47510670    0.56219446""")
        # ts_xyzs = interpolate_isomerization(rxn, use_weights=False)
        # self.assertEqual(len(ts_xyzs), 3)
        # for ts_xyz in ts_xyzs:
        #     print(f'\nTS xyz:\n\n')
        #     print(xyz_to_str(ts_xyz))
        # self.assertTrue(almost_equal_coords(ts_xyzs[0], expected_ts_xyz))

    def test_linear_adapter(self):
        """Test the LinearAdapter class."""
        self.assertEqual(self.rxn_1.family.label, 'Cyclopentadiene_scission')
        linear_1 = LinearAdapter(job_type='tsg',
                                 reactions=[self.rxn_1],
                                 testing=True,
                                 project='test',
                                 project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_linear', 'rxn_1'),
                                 )
        self.assertIsNone(self.rxn_1.ts_species)
        linear_1.execute()
        self.assertEqual(len(self.rxn_1.ts_species.ts_guesses), 1)
        self.assertEqual(self.rxn_1.ts_species.ts_guesses[0].initial_xyz['symbols'],
                         ('C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H'))

    def test_linear_adapter_2(self):
        self.rxn_2.family = KineticsFamily(label='intra_NO2_ONO_conversion')
        self.rxn_2.atom_map = [0, 1, 3, 2, 4, 5, 7, 6, 9, 8]
        self.assertEqual(self.rxn_2.family.label, 'intra_NO2_ONO_conversion')
        linear_2 = LinearAdapter(job_type='tsg',
                                 reactions=[self.rxn_2],
                                 testing=True,
                                 project='test',
                                 project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_linear', 'rxn_2'),
                                 )
        self.assertIsNone(self.rxn_2.ts_species)
        linear_2.execute()
        self.assertEqual(len(self.rxn_2.ts_species.ts_guesses), 1)
        print(xyz_to_str(self.rxn_2.ts_species.ts_guesses[0].initial_xyz))
        self.assertEqual(self.rxn_2.ts_species.ts_guesses[0].initial_xyz['symbols'],
                         ('C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H'))

    def test_get_r_constraints(self):
        """Test the get_r_constraints() function."""
        self.assertEqual(get_r_constraints([(1, 5)], [(0, 5)]), {'R_atom': [(5, 1)]})
        self.assertEqual(get_r_constraints([(1, 5), (7, 2), (8, 2)], [(0, 5), (7, 4), (8, 1)]), {'R_atom': [(1, 5), (5, 0), (7, 2), (2, 8)]})

    @classmethod
    def tearDownClass(cls):
        """
        A function that is run ONCE after all unit tests in this class.
        Delete all project directories created during these unit tests.
        """
        shutil.rmtree(os.path.join(ARC_PATH, 'arc', 'testing', 'test_linear'), ignore_errors=True)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
