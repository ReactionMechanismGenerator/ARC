#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.job.adapters.ts.heuristics module
"""

import copy
import itertools
import os
import unittest
import shutil

from rmgpy.reaction import Reaction
from rmgpy.species import Species

from arc.common import ARC_PATH, _check_r_n_p_symbols_between_rmg_and_arc_rxns, almost_equal_coords
from arc.job.adapters.ts.heuristics import (HeuristicsAdapter,
                                            combine_coordinates_with_redundant_atoms,
                                            determine_glue_params,
                                            find_distant_neighbor,
                                            generate_the_two_constrained_zmats,
                                            get_modified_params_from_zmat_2,
                                            get_new_zmat_2_map,
                                            react,
                                            stretch_zmat_bond,
                                            )
from arc.reaction import ARCReaction
from arc.rmgdb import load_families_only, make_rmg_database_object
from arc.species.converter import str_to_xyz, xyz_to_str, zmat_to_xyz
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
        cls.ccooh_xyz = {'symbols': ('C', 'C', 'O', 'O', 'H', 'H', 'H', 'H', 'H', 'H'),
                         'isotopes': (12, 12, 16, 16, 1, 1, 1, 1, 1, 1),
                         'coords': ((-1.34047, -0.03188, 0.16703), (0.07658, -0.19298, -0.34334),
                                    (0.27374, 0.70670, -1.43275), (1.64704, 0.49781, -1.86879),
                                    (-2.06314, -0.24194, -0.62839), (-1.53242, -0.70687, 1.00574),
                                    (-1.51781, 0.99794, 0.49424), (0.24018, -1.21958, -0.68782),
                                    (0.79344, 0.03863, 0.45152), (1.95991, 1.39912, -1.67215))}
        cls.ccooj_xyz = {'symbols': ('C', 'C', 'O', 'O', 'H', 'H', 'H', 'H', 'H'),
                         'isotopes': (12, 12, 16, 16, 1, 1, 1, 1, 1),
                         'coords': ((-1.10653, -0.06552, 0.042602), (0.385508, 0.205048, 0.049674),
                                    (0.759622, 1.114927, -1.032928), (0.675395, 0.525342, -2.208593),
                                    (-1.671503, 0.860958, 0.166273), (-1.396764, -0.534277, -0.898851),
                                    (-1.36544, -0.740942, 0.862152), (0.97386, -0.704577, -0.082293),
                                    (0.712813, 0.732272, 0.947293))}
        cls.c2h6_xyz = {'symbols': ('C', 'C', 'H', 'H', 'H', 'H', 'H', 'H'),
                        'isotopes': (12, 12, 1, 1, 1, 1, 1, 1),
                        'coords': ((0.75560, 0.02482, 0.00505), (-0.75560, -0.02482, -0.00505),
                                   (1.17380, -0.93109, -0.32410), (1.11891, 0.80903, -0.66579),
                                   (1.12656, 0.23440, 1.01276), (-1.17380, 0.93109, 0.32410),
                                   (-1.11891, -0.809039, 0.66579), (-1.12656, -0.23440, -1.01276))}
        cls.c2h5_xyz = """C      -0.62870399    0.02330636   -0.00849448
                          C       0.85160870   -0.05497517    0.04976674
                          H      -1.06100002   -0.98045393   -0.04651515
                          H      -0.94436731    0.57609393   -0.89725353
                          H      -1.01744999    0.53143528    0.87837737
                          H       1.37701694   -0.84406878   -0.47511337
                          H       1.42289567    0.74866241    0.49923247"""
        cls.oh_xyz = """O 0.0000000 0.0000000 0.1078170
                        H 0.0000000 0.0000000 -0.8625320"""
        cls.h2_xyz = {'coords': ((0, 0, 0.3736550), (0, 0, -0.3736550)),
                      'isotopes': (1, 1), 'symbols': ('H', 'H')}
        cls.h2_mol = ARCSpecies(label='H2', smiles='[H][H]', xyz=cls.h2_xyz).mol
        cls.ch3_xyz = """C       0.00000000    0.00000001   -0.00000000
                         H       1.06690511   -0.17519582    0.05416493
                         H      -0.68531716   -0.83753536   -0.02808565
                         H      -0.38158795    1.01273118   -0.02607927"""
        cls.ch4_xyz = {'symbols': ('C', 'H', 'H', 'H', 'H'), 'isotopes': (12, 1, 1, 1, 1),
                       'coords': ((-5.45906343962835e-10, 4.233517924761169e-10, 2.9505240956083194e-10),
                                  (-0.6505520089868748, -0.7742801979689132, -0.4125187934483119),
                                  (-0.34927557824779626, 0.9815958255612931, -0.3276823191685369),
                                  (-0.022337921721882443, -0.04887374527620588, 1.0908766524267022),
                                  (1.0221655095024578, -0.15844188273952128, -0.350675540104908))}
        cls.ch4_mol = ARCSpecies(label='CH4', smiles='C', xyz=cls.ch4_xyz).mol
        cls.n2h4_xyz = str_to_xyz("""N      -0.67026921   -0.02117571   -0.25636419
                          N       0.64966276    0.05515705    0.30069593
                          H      -1.27787600    0.74907557    0.03694453
                          H      -1.14684483   -0.88535632    0.02014513
                          H       0.65472168    0.28979031    1.29740292
                          H       1.21533718    0.77074524   -0.16656810""")
        cls.n2h4_mol = ARCSpecies(label='N2H4', smiles='NN', xyz=cls.n2h4_xyz).mol
        cls.nh3_xyz = str_to_xyz("""N       0.00064924   -0.00099698    0.29559292
                         H      -0.41786606    0.84210396   -0.09477452
                         H      -0.52039228   -0.78225292   -0.10002797
                         H       0.93760911   -0.05885406   -0.10079043""")
        cls.nh3_mol = ARCSpecies(label='NH3', smiles='N', xyz=cls.nh3_xyz).mol
        cls.nh2_xyz = str_to_xyz("""N       0.00022972    0.40059496    0.00000000
                                    H      -0.83174214   -0.19982058    0.00000000
                                    H       0.83151242   -0.20077438    0.00000000""")
        cls.h2o_xyz = str_to_xyz("""O      -0.00032832    0.39781490    0.00000000
                                    H      -0.76330345   -0.19953755    0.00000000
                                    H       0.76363177   -0.19827735    0.00000000""")
        cls.ch3ch2oh = ARCSpecies(label='CH3CH2OH', smiles='CCO', xyz="""C      -0.97459464    0.29181710    0.10303882
                                                                         C       0.39565894   -0.35143697    0.10221676
                                                                         O       0.30253309   -1.63748710   -0.49196889
                                                                         H      -1.68942501   -0.32359616    0.65926091
                                                                         H      -0.93861751    1.28685508    0.55523033
                                                                         H      -1.35943743    0.38135479   -0.91822428
                                                                         H       0.76858330   -0.46187184    1.12485643
                                                                         H       1.10301149    0.25256708   -0.47388355
                                                                         H       1.19485981   -2.02360458   -0.47786539""")
        cls.ch3ooh = ARCSpecies(label='CH3OOH', smiles='COO', xyz="""C      -0.76039072    0.01483858   -0.00903344
                                                                     O       0.44475333    0.76952102    0.02291303
                                                                     O       0.16024511    1.92327904    0.86381800
                                                                     H      -1.56632337    0.61401630   -0.44251282
                                                                     H      -1.02943316   -0.30449156    1.00193709
                                                                     H      -0.60052507   -0.86954495   -0.63086438
                                                                     H       0.30391344    2.59629139    0.17435159""")
        cls.zmat_1 = {'symbols': ('C', 'C', 'O', 'O', 'H', 'H', 'H', 'H', 'H', 'H'),
                      'coords': ((None, None, None), ('R_1_0', None, None), ('R_2_1', 'A_2_1_0', None),
                                 ('R_3_2', 'A_3_2_1', 'D_3_2_1_0'), ('R_4_0', 'A_4_0_1', 'D_4_0_1_3'),
                                 ('R_5_0', 'A_5_0_1', 'D_5_0_1_4'), ('R_6_0', 'A_6_0_1', 'D_6_0_1_5'),
                                 ('R_7_1', 'A_7_1_0', 'D_7_1_0_6'), ('R_8_1', 'A_8_1_0', 'D_8_1_0_7'),
                                 ('R_9_3', 'A_9_3_2', 'D_9_3_2_1')),
                      'vars': {'R_1_0': 1.5147479951212197, 'R_2_1': 1.4265728986680748,
                               'A_2_1_0': 108.63387152978416, 'R_3_2': 1.4559254886404387,
                               'A_3_2_1': 105.58023544826183, 'D_3_2_1_0': 179.9922243050821,
                               'R_4_0': 1.0950205915944824, 'A_4_0_1': 110.62463321031589,
                               'D_4_0_1_3': 59.13545080998071, 'R_5_0': 1.093567969297245,
                               'A_5_0_1': 110.91425998596507, 'D_5_0_1_4': 120.87266977773987,
                               'R_6_0': 1.0950091062890002, 'A_6_0_1': 110.62270362433773,
                               'D_6_0_1_5': 120.87301274044218, 'R_7_1': 1.0951433842986755,
                               'A_7_1_0': 110.20822115119915, 'D_7_1_0_6': 181.16392677464265,
                               'R_8_1': 1.0951410439636102, 'A_8_1_0': 110.20143800025897,
                               'D_8_1_0_7': 239.4199964284852, 'R_9_3': 0.9741224704818748,
                               'A_9_3_2': 96.30065819269021, 'D_9_3_2_1': 242.3527063196313},
                      'map': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9}}
        cls.zmat_2 = {'symbols': ('H', 'C', 'C', 'H', 'H', 'H', 'H', 'H'),
                      'coords': ((None, None, None), ('R_1_0', None, None), ('R_2_1', 'A_2_1_0', None),
                                 ('R_3_1', 'A_3_1_2', 'D_3_1_2_0'), ('R_4_1', 'A_4_1_2', 'D_4_1_2_3'),
                                 ('R_5_2', 'A_5_2_1', 'D_5_2_1_4'), ('R_6_2', 'A_6_2_1', 'D_6_2_1_5'),
                                 ('R_7_2', 'A_7_2_1', 'D_7_2_1_6')),
                      'vars': {'R_1_0': 1.0940775789443724, 'R_2_1': 1.5120487296562577,
                               'A_2_1_0': 110.56801921096591, 'R_3_1': 1.0940725668318991,
                               'A_3_1_2': 110.56890700195424, 'D_3_1_2_0': 239.99938309284212,
                               'R_4_1': 1.0940817193677925, 'A_4_1_2': 110.56754686774481,
                               'D_4_1_2_3': 239.9997190582892, 'R_5_2': 1.0940725668318991,
                               'A_5_2_1': 110.56890700195424, 'D_5_2_1_4': 59.99971758419434,
                               'R_6_2': 1.0940840619688397, 'A_6_2_1': 110.56790845138725,
                               'D_6_2_1_5': 239.99905123159166, 'R_7_2': 1.0940817193677925,
                               'A_7_2_1': 110.56754686774481, 'D_7_2_1_6': 240.00122783407815},
                      'map': {0: 3, 1: 0, 2: 1, 3: 2, 4: 4, 5: 5, 6: 6, 7: 7}}

    def test_heuristics_for_h_abstraction(self):
        """
        Test that ARC can generate TS guesses based on heuristics for H Abstraction reactions.
        """
        # H2 + O <=> H + OH
        h2_xyz = """H 0.0000000  0.0000000  0.3714780
                    H 0.0000000  0.0000000 -0.3714780"""
        h2 = ARCSpecies(label='H2', smiles='[H][H]', xyz=h2_xyz)
        o = ARCSpecies(label='O', smiles='[O]')
        h = ARCSpecies(label='H', smiles='[H]')
        oh = ARCSpecies(label='OH', smiles='[OH]', xyz=self.oh_xyz)
        rxn1 = ARCReaction(r_species=[h2, o], p_species=[h, oh])
        rxn1.determine_family(rmg_database=self.rmgdb)
        self.assertEqual(rxn1.family.label, 'H_Abstraction')
        heuristics_1 = HeuristicsAdapter(job_type='tsg',
                                         reactions=[rxn1],
                                         testing=True,
                                         project='test',
                                         project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'heuristics'),
                                         dihedral_increment=10,
                                         )
        heuristics_1.execute_incore()
        self.assertTrue(rxn1.ts_species.is_ts)
        self.assertEqual(rxn1.ts_species.charge, 0)
        self.assertEqual(rxn1.ts_species.multiplicity, 3)
        self.assertEqual(len(rxn1.ts_species.ts_guesses), 1)
        self.assertTrue(almost_equal_coords(rxn1.ts_species.ts_guesses[0].initial_xyz,
                                            {'symbols': ('H', 'H', 'O'), 'isotopes': (1, 1, 16),
                                             'coords': ((0.0, 0.0, -1.875762), (0.0, 0.0, -0.984214), (0.0, 0.0, 0.180204))}))

        # H + OH <=> H2 + O
        rxn2 = ARCReaction(r_species=[h, oh], p_species=[h2, o])
        rxn2.determine_family(rmg_database=self.rmgdb)
        heuristics_2 = HeuristicsAdapter(job_type='tsg',
                                         reactions=[rxn2],
                                         testing=True,
                                         project='test',
                                         project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'heuristics'),
                                         dihedral_increment=10,
                                         )
        heuristics_2.execute_incore()
        self.assertEqual(len(rxn2.ts_species.ts_guesses), 1)
        self.assertEqual(rxn2.ts_species.ts_guesses[0].initial_xyz['symbols'], ('H', 'O', 'H'))
        self.assertTrue(almost_equal_coords(rxn2.ts_species.ts_guesses[0].initial_xyz,
                                            {'symbols': ('H', 'O', 'H'), 'isotopes': (1, 16, 1),
                                             'coords': ((0.0, 0.0, 1.875762), (0.0, 0.0, -0.180204), (0.0, 0.0, 0.984214))}))

        # OH + H <=> H2 + O
        rxn3 = ARCReaction(r_species=[oh, h], p_species=[h2, o])
        rxn3.determine_family(rmg_database=self.rmgdb)
        heuristics_3 = HeuristicsAdapter(job_type='tsg',
                                         reactions=[rxn3],
                                         testing=True,
                                         project='test',
                                         project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'heuristics'),
                                         dihedral_increment=10,
                                         )
        heuristics_3.execute_incore()
        self.assertEqual(len(rxn3.ts_species.ts_guesses), 1)
        self.assertEqual(rxn3.ts_species.ts_guesses[0].initial_xyz['symbols'], ('O', 'H', 'H'))
        self.assertTrue(almost_equal_coords(rxn3.ts_species.ts_guesses[0].initial_xyz,
                                            {'symbols': ('O', 'H', 'H'), 'isotopes': (16, 1, 1),
                                             'coords': ((0.0, 0.0, -0.180204), (0.0, 0.0, 0.984214), (0.0, 0.0, 1.8757615))}))

        # CH4 + H <=> CH3 + H2
        ch4 = ARCSpecies(label='CH4', smiles='C', xyz=self.ch4_xyz)
        ch3 = ARCSpecies(label='CH3', smiles='[CH3]', xyz=self.ch3_xyz)
        rxn4 = ARCReaction(reactants=['CH4', 'H'], products=['CH3', 'H2'],
                           r_species=[ch4, h], p_species=[ch3, h2],
                           rmg_reaction=Reaction(reactants=[Species(smiles='C'), Species(smiles='[H]')],
                                                 products=[Species(smiles='[CH3]'), Species(smiles='[H][H]')]))
        rxn4.determine_family(rmg_database=self.rmgdb)
        self.assertEqual(rxn4.family.label, 'H_Abstraction')
        self.assertEqual(rxn4.atom_map[0], 0)
        for index in [1, 2, 3, 4]:
            self.assertIn(rxn4.atom_map[index], [1, 2, 3, 4, 5])
        self.assertIn(rxn4.atom_map[5], [4, 5])
        heuristics_4 = HeuristicsAdapter(job_type='tsg',
                                         reactions=[rxn4],
                                         testing=True,
                                         project='test',
                                         project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'heuristics'),
                                         dihedral_increment=120,
                                         )
        heuristics_4.execute_incore()
        self.assertTrue(rxn4.ts_species.is_ts)
        self.assertEqual(rxn4.ts_species.charge, 0)
        self.assertEqual(rxn4.ts_species.multiplicity, 2)
        self.assertEqual(len(rxn4.ts_species.ts_guesses), 1)  # No dihedral scans for H attacking at 180 degrees.
        self.assertTrue(rxn4.ts_species.ts_guesses[0].success)
        self.assertTrue(almost_equal_coords(rxn4.ts_species.ts_guesses[0].initial_xyz,
                                            {'symbols': ('C', 'H', 'H', 'H', 'H', 'H'), 'isotopes': (12, 1, 1, 1, 1, 1),
                                             'coords': ((2.032228145856716e-09, -0.1349861936729119, -0.047724833206401085),
                                                        (2.032228145856716e-09, 1.1006963897520332, 0.3891549615762657),
                                                        (0.8917770595461324, -0.6498538917136708, 0.31634164254626373),
                                                        (-0.8917770898402092, -0.6498539493490708, 0.31634164059575576),
                                                        (2.032228145856716e-09, -0.1349861936729119, -1.139924203999533),
                                                        (2.032228145856716e-09, 1.9412551274403573, 0.6863373721320729))}))

        # C3H8 + HO2 <=> C3H7 + H2O2
        c3h8_xyz = """C	0.0000000 0.0000000 0.5949240
                      C 0.0000000 1.2772010 -0.2630030
                      C 0.0000000 -1.2772010 -0.2630030
                      H 0.8870000 0.0000000 1.2568980
                      H -0.8870000 0.0000000 1.2568980
                      H 0.0000000 2.1863910 0.3643870
                      H 0.0000000 -2.1863910 0.3643870
                      H 0.8933090 1.3136260 -0.9140200
                      H -0.8933090 1.3136260 -0.9140200
                      H -0.8933090 -1.3136260 -0.9140200
                      H 0.8933090 -1.3136260 -0.9140200"""
        ho2_xyz = """O 0.0553530 -0.6124600 0.0000000
                     O 0.0553530 0.7190720 0.0000000
                     H -0.8856540 -0.8528960 0.0000000"""
        c3h7_xyz = """C 1.3077700 -0.2977690 0.0298660
                      C 0.0770610 0.5654390 -0.0483740
                      C -1.2288150 -0.2480100 0.0351080
                      H -2.1137100 0.4097560 -0.0247200
                      H -1.2879330 -0.9774520 -0.7931500
                      H -1.2803210 -0.8079420 0.9859990
                      H 0.1031750 1.3227340 0.7594170
                      H 0.0813910 1.1445730 -0.9987260
                      H 2.2848940 0.1325040 0.2723890
                      H 1.2764100 -1.3421290 -0.3008110"""
        h2o2_xyz = """O 0.0000000 0.7275150 -0.0586880
                      O 0.0000000 -0.7275150 -0.0586880
                      H 0.7886440 0.8942950 0.4695060
                      H -0.7886440 -0.8942950 0.4695060"""
        c3h8 = ARCSpecies(label='C3H8', smiles='CCC', xyz=c3h8_xyz)
        ho2 = ARCSpecies(label='HO2', smiles='O[O]', xyz=ho2_xyz)
        c3h7 = ARCSpecies(label='C3H7', smiles='[CH2]CC', xyz=c3h7_xyz)
        h2o2 = ARCSpecies(label='H2O2', smiles='OO', xyz=h2o2_xyz)
        rxn5 = ARCReaction(reactants=['C3H8', 'HO2'], products=['C3H7', 'H2O2'],
                           r_species=[c3h8, ho2], p_species=[c3h7, h2o2],
                           rmg_reaction=Reaction(reactants=[Species().from_smiles('CCC'),
                                                            Species().from_smiles('O[O]')],
                                                 products=[Species().from_smiles('[CH2]CC'),
                                                           Species().from_smiles('OO')]))
        rxn5.determine_family(rmg_database=self.rmgdb)
        self.assertEqual(rxn5.family.label, 'H_Abstraction')
        heuristics_5 = HeuristicsAdapter(job_type='tsg',
                                         reactions=[rxn5],
                                         testing=True,
                                         project='test',
                                         project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'heuristics'),
                                         dihedral_increment=120,
                                         )
        heuristics_5.execute_incore()
        self.assertTrue(rxn5.ts_species.is_ts)
        self.assertEqual(rxn5.ts_species.charge, 0)
        self.assertEqual(rxn5.ts_species.multiplicity, 2)
        self.assertEqual(len(rxn5.ts_species.ts_guesses), 3)
        self.assertTrue(almost_equal_coords(rxn5.ts_species.ts_guesses[0].initial_xyz,
                                            {'symbols': ('C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'O', 'O', 'H'),
                                             'isotopes': (12, 12, 12, 1, 1, 1, 1, 1, 1, 1, 1, 16, 16, 1),
                                             'coords': ((-0.01147913752923963, -0.5562661419535987, -1.1846714308119068),
                                                        (-0.01147913752923963, 0.8680770847359621, -0.6028402086476463),
                                                        (-0.01147913752923963, -0.5562661419535987, -2.723268583578539),
                                                        (0.8755208740199835, -1.105775790466776, -0.8155524647838386),
                                                        (-0.8984791490784627, -1.105775790466776, -0.8155524647838386),
                                                        (-0.01147913752923963, 0.8514766161550243, 0.7226321171473922),
                                                        (-0.011479137529239503, -1.5840347367204934, -3.1281592859817424),
                                                        (0.8818298910743927, 1.4288019032460375, -0.9356128425491042),
                                                        (-0.9047881446185388, 1.4288019172197552, -0.9356128767572558),
                                                        (-0.904788166132872, -0.0361627138809536, -3.1165145270718613),
                                                        (0.8818298910743926, -0.03616271388095338, -3.1165145270718613),
                                                        (-0.01147913752923963, -0.599123069148373, 2.112848077609368),
                                                        (-0.01147913752923963, 0.8369939164492187, 1.8790101777575878),
                                                        (0.8662379361688366, -0.7056654789761558, 2.4962868861193237))}))
        self.assertTrue(almost_equal_coords(rxn5.ts_species.ts_guesses[1].initial_xyz,
                                            {'symbols': ('C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'O', 'O', 'H'),
                                             'isotopes': (12, 12, 12, 1, 1, 1, 1, 1, 1, 1, 1, 16, 16, 1),
                                             'coords': ((-0.2703079353599017, -1.0436234985919701, -1.1825306017455417),
                                                        (-0.2703079353599017, 0.3807197280975907, -0.6006993795812812),
                                                        (-0.2703079353599017, -1.0436234985919701, -2.7211277545121737),
                                                        (0.6166920761893214, -1.5931331471051475, -0.8134116357174734),
                                                        (-1.1573079469091248, -1.5931331471051475, -0.8134116357174734),
                                                        (-0.2703079353599017, 0.3641192595166529, 0.7247729462137573),
                                                        (-0.2703079353599016, -2.071392093358865, -3.1260184569153773),
                                                        (0.6230010932437307, 0.941444546607666, -0.933472013482739),
                                                        (-1.1636169424492009, 0.9414445605813839, -0.9334720476908906),
                                                        (-1.163616963963534, -0.523520070519325, -3.114373698005496),
                                                        (0.6230010932437305, -0.5235200705193248, -3.114373698005496),
                                                        (0.9707722573196677, 1.0629649547967785, 2.141908971178336),
                                                        (-0.2703079353599017, 0.3496365598108473, 1.881151006823953),
                                                        (0.7011171921899904, 1.958006862823161, 1.9074948181266))}))
        self.assertTrue(almost_equal_coords(rxn5.ts_species.ts_guesses[2].initial_xyz,
                                            {'symbols': ('C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'O', 'O', 'H'),
                                             'isotopes': (12, 12, 12, 1, 1, 1, 1, 1, 1, 1, 1, 16, 16, 1),
                                             'coords': ((0.27698580107090603, -1.0268295459131784, -1.1966886956594065),
                                                        (0.27698580107090603, 0.3975136807763824, -0.614857473495146),
                                                        (0.27698580107090603, -1.0268295459131784, -2.7352858484260385),
                                                        (1.1639858126201292, -1.5763391944263558, -0.8275697296313382),
                                                        (-0.610014210478317, -1.5763391944263558, -0.8275697296313382),
                                                        (0.27698580107090603, 0.3809132121954446, 0.7106148522998925),
                                                        (0.27698580107090615, -2.0545981406800733, -3.140176550829242),
                                                        (1.1702948296745384, 0.9582384992864577, -0.9476301073966038),
                                                        (-0.6163232060183932, 0.9582385132601756, -0.9476301416047554),
                                                        (-0.6163232275327264, -0.5067261178405333, -3.128531791919361),
                                                        (1.1702948296745384, -0.5067261178405331, -3.128531791919361),
                                                        (-0.9640943916086628, 1.0797589074755711, 2.127750877264471),
                                                        (0.27698580107090603, 0.366430512489639, 1.8669929129100882),
                                                        (-1.2050419194946638, 0.6907026570107317, 2.9758920148143635))}))
        self.assertTrue(rxn5.ts_species.ts_guesses[0].success)
        self.assertTrue(rxn5.ts_species.ts_guesses[1].success)
        self.assertTrue(rxn5.ts_species.ts_guesses[2].success)

        # CCCOH + OH <=> CCCO + H2O
        cccoh_xyz = """C -1.4562640 1.2257490 0.0000000
                       C 0.0000000 0.7433860 0.0000000
                       C 0.1008890 -0.7771710 0.0000000
                       O 1.4826600 -1.1256940 0.0000000
                       H -1.5081640 2.3212940 0.0000000
                       H -1.9909330 0.8624630 0.8882620
                       H -1.9909330 0.8624630 -0.8882620
                       H 0.5289290 1.1236530 0.8845120
                       H 0.5289290 1.1236530 -0.8845120
                       H -0.4109400 -1.1777970 0.8923550
                       H -0.4109400 -1.1777970 -0.8923550
                       H 1.5250230 -2.0841670 0.0000000"""
        ccco_xyz = """C      -1.22579665    0.34157501   -0.08330600
                      C      -0.04626439   -0.57243496    0.22897599
                      C      -0.11084721   -1.88672335   -0.59040103
                      O       0.94874959   -2.60335587   -0.24842497
                      H      -2.17537216   -0.14662734    0.15781317
                      H      -1.15774972    1.26116047    0.50644174
                      H      -1.23871523    0.61790236   -1.14238547
                      H       0.88193016   -0.02561912    0.01201028
                      H      -0.05081615   -0.78696747    1.30674288
                      H      -1.10865982   -2.31155703   -0.39617740
                      H      -0.21011639   -1.57815495   -1.64338139"""
        h2o_xyz = """O      -0.00032832    0.39781490    0.00000000
                     H      -0.76330345   -0.19953755    0.00000000
                     H       0.76363177   -0.19827735    0.00000000"""
        cccoh = ARCSpecies(label='CCCOH', smiles='CCCO', xyz=cccoh_xyz)
        ccco = ARCSpecies(label='CCCO', smiles='CCC[O]', xyz=ccco_xyz)
        h2o = ARCSpecies(label='H2O', smiles='O', xyz=h2o_xyz)
        rxn6 = ARCReaction(reactants=['CCCOH', 'OH'], products=['CCCO', 'H2O'],
                           r_species=[cccoh, oh], p_species=[ccco, h2o],
                           rmg_reaction=Reaction(reactants=[Species().from_smiles('CCCO'),
                                                            Species().from_smiles('[OH]')],
                                                 products=[Species().from_smiles('CCC[O]'),
                                                           Species().from_smiles('O')]))
        rxn6.determine_family(rmg_database=self.rmgdb)
        self.assertEqual(rxn6.family.label, 'H_Abstraction')
        heuristics_6 = HeuristicsAdapter(job_type='tsg',
                                         reactions=[rxn6],
                                         testing=True,
                                         project='test',
                                         project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'heuristics'),
                                         dihedral_increment=20,
                                         )
        heuristics_6.execute_incore()
        self.assertTrue(rxn6.ts_species.is_ts)
        self.assertEqual(rxn6.ts_species.charge, 0)
        self.assertEqual(rxn6.ts_species.multiplicity, 2)
        self.assertEqual(rxn6.ts_species.ts_guesses[0].initial_xyz['symbols'],
                         ('C', 'C', 'C', 'O', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'O', 'H'))
        self.assertEqual(len(rxn6.ts_species.ts_guesses[1].initial_xyz['coords']), 14)

        # C=COH + H <=> C=CO + H2
        cdcoh_xyz = """C      -0.80601307   -0.11773769    0.32792128
                       C       0.23096883    0.47536513   -0.26437348
                       O       1.44620485   -0.11266560   -0.46339257
                       H      -1.74308628    0.41660480    0.45016601
                       H      -0.75733964   -1.13345488    0.70278513
                       H       0.21145717    1.48838416   -0.64841675
                       H       1.41780836   -1.01649567   -0.10468897"""
        cdco_xyz = """C      -0.68324480   -0.04685539   -0.10883672
                      C       0.63642204    0.05717653    0.10011041
                      O       1.50082619   -0.82476680    0.32598015
                      H      -1.27691852    0.84199331   -0.29048852
                      H      -1.17606821   -1.00974165   -0.10030145
                      H       0.99232452    1.08896899    0.06242974"""
        cdcoh = ARCSpecies(label='C=COH', smiles='C=CO', xyz=cdcoh_xyz)
        cdco = ARCSpecies(label='C=CO', smiles='C=C[O]', xyz=cdco_xyz)
        rxn7 = ARCReaction(reactants=['C=COH', 'H'], products=['C=CO', 'H2'],
                           r_species=[cdcoh, h], p_species=[cdco, h2],
                           rmg_reaction=Reaction(reactants=[Species().from_smiles('C=CO'),
                                                            Species().from_smiles('[H]')],
                                                 products=[Species().from_smiles('C=C[O]'),
                                                           Species().from_smiles('[H][H]')]))
        rxn7.determine_family(rmg_database=self.rmgdb)
        self.assertEqual(rxn7.family.label, 'H_Abstraction')
        heuristics_7 = HeuristicsAdapter(job_type='tsg',
                                         reactions=[rxn7],
                                         testing=True,
                                         project='test',
                                         project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'heuristics'),
                                         dihedral_increment=120,
                                         )
        heuristics_7.execute_incore()
        self.assertTrue(rxn7.ts_species.is_ts)
        self.assertEqual(rxn7.ts_species.charge, 0)
        self.assertEqual(rxn7.ts_species.multiplicity, 2)
        self.assertEqual(len(rxn7.ts_species.ts_guesses), 1)  # No dihedral scans for H attacking at 180 degrees.
        self.assertTrue(almost_equal_coords(rxn7.ts_species.ts_guesses[0].initial_xyz,
                                            {'symbols': ('C', 'C', 'O', 'H', 'H', 'H', 'H', 'H'),
                                             'isotopes': (12, 12, 16, 1, 1, 1, 1, 1),
                                             'coords': ((-5.668810146262865e-08, -0.48719172182900145, -1.1721907663384206),
                                                        (-5.668810146262865e-08, -0.48719172182900145, 0.16119415878387744),
                                                        (-5.668810146262865e-08, 0.6381065684637319, 0.933134016171312),
                                                        (-3.6906864264223316e-08, -1.4258733886217165, -1.7175793139772886),
                                                        (-9.526192583832159e-08, 0.42803267481449664, -1.7526542401902248),
                                                        (0.0, -1.3854217138117004, 0.7672141857856329),
                                                        (8.461119254489423e-07, 1.5722313670073633, 0.23298892236129287),
                                                        (1.5355909530954166e-06, 2.2856334921752297, -0.30172014288208393))}))

        # NCO + NH2 <=> HNCO + NH
        nco_xyz = """N       1.36620399    0.00000000    0.00000000
                     C      -0.09510200    0.00000000    0.00000000
                     O      -1.27110200    0.00000000    0.00000000"""
        nh2_xyz = """N       0.00022972    0.40059496    0.00000000
                     H      -0.83174214   -0.19982058    0.00000000
                     H       0.83151242   -0.20077438    0.00000000"""
        hnco_xyz = """N      -0.70061553    0.28289128   -0.18856549
                      C       0.42761869    0.11537693    0.07336374
                      O       1.55063087   -0.07323229    0.35677630
                      H      -1.27763403   -0.32503592    0.39725197"""
        nh_xyz = """N       0.50949998    0.00000000    0.00000000
                    H      -0.50949998    0.00000000    0.00000000"""
        nco = ARCSpecies(label='NCO', smiles='[N]=C=O', xyz=nco_xyz)
        nh2 = ARCSpecies(label='NH2', smiles='[NH2]', xyz=nh2_xyz)
        hnco = ARCSpecies(label='HNCO', smiles='N=C=O', xyz=hnco_xyz)
        nh = ARCSpecies(label='NH', smiles='[NH]', xyz=nh_xyz)
        rxn8 = ARCReaction(r_species=[nco, nh2], p_species=[hnco, nh])
        rxn8.determine_family(rmg_database=self.rmgdb)
        self.assertEqual(rxn8.family.label, 'H_Abstraction')
        heuristics_8 = HeuristicsAdapter(job_type='tsg',
                                         reactions=[rxn8],
                                         testing=True,
                                         project='test',
                                         project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'heuristics'),
                                         dihedral_increment=120,
                                         )
        heuristics_8.execute_incore()
        self.assertTrue(rxn8.ts_species.is_ts)
        self.assertEqual(rxn8.ts_species.charge, 0)
        self.assertEqual(rxn8.ts_species.multiplicity, 3)
        self.assertEqual(len(rxn8.ts_species.ts_guesses), 3)
        self.assertTrue(almost_equal_coords(rxn8.ts_species.ts_guesses[0].initial_xyz,
                                            {'symbols': ('N', 'C', 'O', 'N', 'H', 'H'), 'isotopes': (14, 12, 16, 14, 1, 1),
                                             'coords': ((-3.657596721635545e-09, 0.08876698337705413, 0.9329034620293603),
                                                        (-3.657596721635545e-09, 0.8029731964901674, 0.005816759497990542),
                                                        (9.609228590831403e-09, 1.4947776654181317, -0.9420542025729876),
                                                        (-3.657596721635545e-09, -2.2452231277287726, 0.16101001965294726),
                                                        (-3.657596721635545e-09, -1.0762904363668742, 0.547597552628467),
                                                        (-3.657596721635545e-09, -2.2452231277287726, -0.8649899620469162))}))
        self.assertTrue(almost_equal_coords(rxn8.ts_species.ts_guesses[1].initial_xyz,
                                            {'symbols': ('N', 'C', 'O', 'N', 'H', 'H'), 'isotopes': (14, 12, 16, 14, 1, 1),
                                             'coords': ((0.7304309896785263, 0.4813753237349452, -0.26044855417424406),
                                                        (-0.22605518455485318, 0.6753956743209286, 0.3853614572987496),
                                                        (-1.2013936879956266, 0.8535814918821567, 1.0130695121236126),
                                                        (0.7304309896785263, -1.8526147873708816, -1.032341996550657),
                                                        (0.7304309896785263, -0.6836820960089831, -0.6457544635751373),
                                                        (0.7304309896785263, -1.8526147873708816, -2.0583419782505206))}))
        self.assertTrue(almost_equal_coords(rxn8.ts_species.ts_guesses[2].initial_xyz,
                                            {'symbols': ('N', 'C', 'O', 'N', 'H', 'H'), 'isotopes': (14, 12, 16, 14, 1, 1),
                                             'coords': ((-0.7304309882994099, 0.48137532979234665, -0.26044855803662603),
                                                        (0.22605518593396912, 0.6753956803783296, 0.3853614534363681),
                                                        (1.201393684372416, 0.8535814759681686, 1.0130695222708512),
                                                        (-0.7304309882994099, -1.8526147813134801, -1.032342000413039),
                                                        (-0.7304309882994099, -0.6836820899515816, -0.6457544674375193),
                                                        (-0.7304309882994099, -1.8526147813134801, -2.0583419821129025))}))

        # butenylnebzene + CCOO <=> butenylnebzene_rad + CCOOH
        butenylnebzene_xyz = """C      -1.71226453   -0.84554485    0.38526063
                                C      -3.04422965   -0.42914264    0.41012166
                                C      -3.42029899    0.74723413   -0.23559352
                                C      -2.46544277    1.50753989   -0.90818256
                                C      -1.13299223    1.09288376   -0.93424603
                                C      -0.74341165   -0.08513522   -0.28306031
                                C       0.69357354   -0.54411184   -0.32741768
                                C       0.96111341   -1.45397252   -1.53003141
                                C       2.38975018   -1.92155599   -1.55932472
                                C       3.25117244   -1.64080060   -2.54512753
                                H      -1.43447830   -1.76767058    0.89062734
                                H      -3.78886508   -1.02369999    0.93270709
                                H      -4.45754439    1.07063686   -0.21594061
                                H      -2.75843025    2.42373310   -1.41396154
                                H      -0.39940594    1.69445181   -1.46624302
                                H       0.93550160   -1.06555454    0.60815996
                                H       1.35380726    0.33269483   -0.35901826
                                H       0.70851184   -0.92832434   -2.45979346
                                H       0.31045732   -2.33637041   -1.48650631
                                H       2.72442421   -2.53507765   -0.72507466
                                H       2.96318630   -1.03906198   -3.40168902
                                H       4.27083810   -2.01270921   -2.51458029"""
        peroxyl_xyz = """C      -1.05582103   -0.03329574   -0.10080257
                         C       0.41792695    0.17831205    0.21035514
                         O       1.19234020   -0.65389683   -0.61111443
                         O       2.44749684   -0.41401220   -0.28381363
                         H      -1.33614002   -1.09151783    0.08714882
                         H      -1.25953618    0.21489046   -1.16411897
                         H      -1.67410396    0.62341419    0.54699514
                         H       0.59566350   -0.06437686    1.28256640
                         H       0.67254676    1.24676329    0.02676370"""
        peroxide_xyz = """C      -1.34047532   -0.03188958    0.16703197
                          C       0.07658214   -0.19298564   -0.34334978
                          O       0.27374087    0.70670928   -1.43275058
                          O       1.64704173    0.49781461   -1.86879814
                          H      -2.06314673   -0.24194344   -0.62839800
                          H      -1.53242454   -0.70687225    1.00574309
                          H      -1.51781587    0.99794893    0.49424821
                          H       0.24018222   -1.21958121   -0.68782344
                          H       0.79344780    0.03863434    0.45152272
                          H       1.95991273    1.39912383   -1.67215155"""
        butenylnebzene_rad1_xyz = """C      -1.88266976   -0.87687529   -0.63607576
                                     C      -3.06025073   -0.20236914   -0.30870889
                                     C      -3.07096712    0.71611053    0.73950868
                                     C      -1.90403911    0.96368054    1.45998739
                                     C      -0.72563713    0.29023608    1.13460173
                                     C      -0.70473521   -0.64221365    0.08717649
                                     C       0.57567571   -1.35823396   -0.27232070
                                     C       1.47982006   -0.51744899   -1.11962456
                                     C       2.90883262   -0.59134257   -0.96873657
                                     C       3.75183420    0.10706027   -1.73634793
                                     H      -1.88917338   -1.58645450   -1.46031384
                                     H      -3.96913141   -0.39219681   -0.87352986
                                     H      -3.98783434    1.24160615    0.99265871
                                     H      -1.91005379    1.68377589    2.27388853
                                     H       0.18057426    0.49821548    1.69940227
                                     H       0.34051301   -2.28273463   -0.81410484
                                     H       1.07516828   -1.66856944    0.65418851
                                     H       1.05110536    0.09965361   -1.90323723
                                     H       3.30722691   -1.24504321   -0.19595001
                                     H       4.82473956    0.02594832   -1.59289339
                                     H       3.40309973    0.77336779   -2.51917155"""
        butenylnebzene_rad2_xyz = """C       0.86557186    0.88201467    0.63421602
                                     C       2.10949019    1.51449365    0.71246655
                                     C       3.25224601    0.86186581    0.25912210
                                     C       3.14957012   -0.42059281   -0.27342877
                                     C       1.90421947   -1.05251000   -0.35083861
                                     C       0.74699841   -0.41118094    0.10593634
                                     C      -0.58479314   -1.01936532    0.04460380
                                     C      -0.85980476   -2.42967089   -0.36144757
                                     C      -1.19954275   -2.52225943   -1.82034160
                                     C      -2.37408626   -2.95334594   -2.29914371
                                     H      -0.01246578    1.41307286    0.99522772
                                     H       2.18292853    2.51633055    1.12713435
                                     H       4.22023612    1.35204102    0.31838735
                                     H       4.03880532   -0.93210312   -0.63266934
                                     H       1.86256917   -2.04849879   -0.78187681
                                     H      -1.44058451   -0.38162928    0.24463988
                                     H      -0.01456714   -3.08941132   -0.14056348
                                     H      -1.69219098   -2.80180034    0.24897564
                                     H      -0.42576150   -2.21925087   -2.52323755
                                     H      -2.55420789   -2.99614384   -3.36897081
                                     H      -3.17872709   -3.26780622   -1.64194211"""
        butenylnebzene = ARCSpecies(label='butenylnebzene', smiles='c1ccccc1CCC=C', xyz=butenylnebzene_xyz)
        peroxyl = ARCSpecies(label='CCOO', smiles='CCO[O]', xyz=peroxyl_xyz)
        peroxide = ARCSpecies(label='CCOOH', smiles='CCOO', xyz=peroxide_xyz)
        butenylnebzene_rad1 = ARCSpecies(label='butenylnebzene_rad1', smiles='c1ccccc1C[CH]C=C', xyz=butenylnebzene_rad1_xyz)
        butenylnebzene_rad2 = ARCSpecies(label='butenylnebzene_rad2', smiles='c1ccccc1[CH]CC=C', xyz=butenylnebzene_rad2_xyz)
        rxn9 = ARCReaction(r_species=[butenylnebzene, peroxyl], p_species=[peroxide, butenylnebzene_rad1])
        rxn10 = ARCReaction(r_species=[butenylnebzene, peroxyl], p_species=[peroxide, butenylnebzene_rad2])
        rxn9.determine_family(rmg_database=self.rmgdb)
        rxn10.determine_family(rmg_database=self.rmgdb)
        self.assertEqual(rxn9.family.label, 'H_Abstraction')
        self.assertEqual(rxn10.family.label, 'H_Abstraction')
        heuristics_9 = HeuristicsAdapter(job_type='tsg',
                                         reactions=[rxn9],
                                         testing=True,
                                         project='test',
                                         project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'heuristics'),
                                         dihedral_increment=120,
                                         )
        heuristics_9.execute_incore()
        self.assertTrue(rxn9.ts_species.is_ts)
        self.assertEqual(rxn9.ts_species.charge, 0)
        self.assertEqual(rxn9.ts_species.multiplicity, 2)
        self.assertEqual(len(rxn9.ts_species.ts_guesses), 3)
        self.assertTrue(almost_equal_coords(rxn9.ts_species.ts_guesses[2].initial_xyz,
                                            {'symbols': ('C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'H', 'H',
                                                         'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'C', 'C',
                                                         'O', 'O', 'H', 'H', 'H', 'H', 'H'),
                                             'isotopes': (12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 1, 1, 1, 1, 1, 1, 1,
                                                          1, 1, 1, 1, 1, 12, 12, 16, 16, 1, 1, 1, 1, 1),
                                             'coords': ((-1.6782375798443572, 0.29919363015288947, -2.01806933306555),
                                                        (-1.6782375798443572, 1.5027435678701453, -2.7248986560063386),
                                                        (-1.6738369971006504, 2.7156676603762113, -2.038579233325109),
                                                        (-1.6677329412943638, 2.7265125184344017, -0.6450094538178275),
                                                        (-1.6677053461953915, 1.5240949895060363, 0.06367634062402505),
                                                        (-1.6782375798443572, 0.29919363015288947, -0.6168015066664458),
                                                        (-1.6585078694334607, -1.0011774963388973, 0.1488351360576159),
                                                        (-0.22847715115175005, -1.4813764870892399, 0.4136466779649939),
                                                        (-0.21550665223290233, -2.785527329778983, 1.1616558824137488),
                                                        (0.31636180414283266, -2.946702231731262, 2.3797765274806224),
                                                        (-1.678027681823609, -0.6394625668383056, -2.5674335899178775),
                                                        (-1.6805374816037417, 1.4936958699611083, -3.811631623110614),
                                                        (-1.673639503902614, 3.652261895102285, -2.5896187666855774),
                                                        (-1.6618711770245946, 3.6719830916188174, -0.10917839329676005),
                                                        (-1.659155104998602, 1.547985675237973, 1.151063106976439),
                                                        (-2.219027332783967, -1.760001959587851, -0.4130727559788232),
                                                        (-2.198597939911353, -0.8727739071372331, 1.096206390467767),
                                                        (0.43795113273770614, -0.5602693112997892, 1.078490651365168),
                                                        (0.30278264523831466, -1.623300586160451, -0.5358239385363122),
                                                        (-0.6599910199181409, -3.6442468927242135, 0.6622333384371668),
                                                        (0.7774064133373235, -2.1227371856777904, 2.9157264595113626),
                                                        (0.30201535275503, -3.916244249061097, 2.86839672092038),
                                                        (4.099315214001509, -0.48487667138136925, -0.16869388097797056),
                                                        (2.9824137349346773, -0.6248127016370444, 0.8449345858315107),
                                                        (2.0683807009783797, 0.45612083231342077, 0.6681784201249208),
                                                        (1.0294462430506086, 0.2572686049690234, 1.6685795910063232),
                                                        (4.618937325294738, 0.47046657633339406, -0.04064071032117367),
                                                        (3.6987957431812384, -0.4951582635350503, -1.187794495313047),
                                                        (4.823883478797864, -1.2974708501182275, -0.06586407038633357),
                                                        (3.3880933310949213, -0.5929701052157578, 1.86166251199261),
                                                        (2.454473857018558, -1.5729134990251377, 0.6974526112820145))}))
        heuristics_10 = HeuristicsAdapter(job_type='tsg',
                                          reactions=[rxn10],
                                          testing=True,
                                          project='test',
                                          project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'heuristics'),
                                          dihedral_increment=120,
                                          )
        heuristics_10.execute_incore()
        self.assertTrue(rxn10.ts_species.is_ts)
        self.assertEqual(rxn10.ts_species.charge, 0)
        self.assertEqual(rxn10.ts_species.multiplicity, 2)
        self.assertEqual(len(rxn10.ts_species.ts_guesses), 3)
        self.assertTrue(almost_equal_coords(rxn10.ts_species.ts_guesses[1].initial_xyz,
                                            {'symbols': ('C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'H', 'H',
                                                         'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'C', 'C',
                                                         'O', 'O', 'H', 'H', 'H', 'H', 'H'),
                                             'isotopes': (12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 1, 1, 1, 1, 1, 1, 1,
                                                          1, 1, 1, 1, 1, 12, 12, 16, 16, 1, 1, 1, 1, 1),
                                             'coords': ((-0.25317681113274554, 1.5464908800909205, -1.1860120627685324),
                                                        (-0.25317681113274554, 2.750040817808176, -1.8928413857093211),
                                                        (-0.24877622838903884, 3.962964910314242, -1.2065219630280912),
                                                        (-0.24267217258275225, 3.973809768372433, 0.18704781647919),
                                                        (-0.2426445774837797, 2.7713922394440673, 0.8957336109210425),
                                                        (-0.25317681113274554, 1.5464908800909205, 0.21525576363057164),
                                                        (-0.23344710072184902, 0.24611975359913374, 0.9808924063546334),
                                                        (1.1965836175598616, -0.23407923715120882, 1.2457039482620114),
                                                        (1.2095541164787094, -1.538230079840952, 1.9937131527107663),
                                                        (1.7414225728544444, -1.699404981793231, 3.21183379777764),
                                                        (-0.2529669131119974, 0.6078346830997254, -1.73537631962086),
                                                        (-0.25547671289213014, 2.740993119899139, -2.9795743528135965),
                                                        (-0.2485787351910023, 4.899559145040316, -1.75756149638856),
                                                        (-0.23681040831298303, 4.919280341556849, 0.7228788770002574),
                                                        (-0.23409433628699036, 2.795282925176004, 1.9831203772734565),
                                                        (-0.9060704567424569, -0.6644696022996106, 0.3066029359109066),
                                                        (-0.7735371711997414, 0.3745233428007979, 1.9282636607647845),
                                                        (1.7519405208010752, 0.5335100760066669, 1.7997405927621566),
                                                        (1.7278434139499264, -0.37600333622242, 0.2962333317607053),
                                                        (0.7650697487934708, -2.3969496427861827, 1.4942906087341843),
                                                        (2.202467182048935, -0.8754399357397595, 3.74778372980838),
                                                        (1.7270761214666417, -2.668946999123066, 3.7004539912173975),
                                                        (0.40677214002562073, -4.1756095986706185, -1.8395123233828858),
                                                        (-0.7086457539915318, -3.1772355462720103, -1.6080261500691102),
                                                        (-0.3950340436834628, -2.403482787253425, -0.45127604863576076),
                                                        (-1.5027753693097672, -1.4722814885993731, -0.2915800381186999),
                                                        (0.5242075146700145, -4.831155150378781, -0.970285979769179),
                                                        (1.3625343973711994, -3.6602019038736495, -1.9807958516487556),
                                                        (0.20360884057053463, -4.790576749365072, -2.7206607313906463),
                                                        (-1.6572087403470876, -3.700357862133809, -1.4471011206701503),
                                                        (-0.8063915817826355, -2.512199529901875, -2.47262977000082))}))

        # C2H5O + CH3OH <=> C2H5OH + CH3O
        c2h5o_xyz = """C      -0.74046271    0.02568566   -0.00568694
                       C       0.79799272   -0.01511040    0.00517437
                       O       1.17260343   -0.72227959   -1.04851579
                       H      -1.13881231   -0.99286049    0.06963185
                       H      -1.14162013    0.59700303    0.84092854
                       H      -1.13266865    0.46233725   -0.93283228
                       H       1.11374677    1.03794239    0.06905096
                       H       1.06944350   -0.38306117    1.00698657"""
        ch3oh_xyz = """C      -0.36862686   -0.00871354    0.04292587
                       O       0.98182901   -0.04902010    0.46594709
                       H      -0.57257378    0.95163086   -0.43693396
                       H      -0.55632373   -0.82564527   -0.65815446
                       H      -1.01755588   -0.12311763    0.91437513
                       H       1.10435907    0.67758465    1.10037299"""
        c2h5oh_xyz = """C      -0.97459464    0.29181710    0.10303882
                        C       0.39565894   -0.35143697    0.10221676
                        O       0.30253309   -1.63748710   -0.49196889
                        H      -1.68942501   -0.32359616    0.65926091
                        H      -0.93861751    1.28685508    0.55523033
                        H      -1.35943743    0.38135479   -0.91822428
                        H       0.76858330   -0.46187184    1.12485643
                        H       1.10301149    0.25256708   -0.47388355
                        H       1.19485981   -2.02360458   -0.47786539"""
        ch3o_xyz = """C       0.03807240    0.00035621   -0.00484242
                      O       1.35198769    0.01264937   -0.17195885
                      H      -0.33965241   -0.14992727    1.02079480
                      H      -0.51702680    0.90828035   -0.29592912
                      H      -0.53338088   -0.77135867   -0.54806440"""
        c2h5o = ARCSpecies(label='C2H5O', smiles='CC[O]', xyz=c2h5o_xyz)
        ch3oh = ARCSpecies(label='CH3OH', smiles='CO', xyz=ch3oh_xyz)
        c2h5oh = ARCSpecies(label='C2H5OH', smiles='CCO', xyz=c2h5oh_xyz)
        ch3o = ARCSpecies(label='CH3O', smiles='C[O]', xyz=ch3o_xyz)
        rxn11 = ARCReaction(r_species=[c2h5o, ch3oh], p_species=[c2h5oh, ch3o])
        rxn11.determine_family(rmg_database=self.rmgdb)
        self.assertEqual(rxn11.family.label, 'H_Abstraction')
        heuristics_11 = HeuristicsAdapter(job_type='tsg',
                                          reactions=[rxn11],
                                          testing=True,
                                          project='test',
                                          project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'heuristics'),
                                          dihedral_increment=30,
                                          )
        heuristics_11.execute_incore()
        self.assertTrue(rxn11.ts_species.is_ts)
        self.assertEqual(rxn11.ts_species.charge, 0)
        self.assertEqual(rxn11.ts_species.multiplicity, 2)
        self.assertEqual(len(rxn11.ts_species.ts_guesses), 12)
        self.assertTrue(almost_equal_coords(rxn11.ts_species.ts_guesses[0].initial_xyz,
                                            {'symbols': ('C', 'C', 'O', 'H', 'H', 'H', 'H', 'H', 'C', 'O', 'H', 'H', 'H', 'H'),
                                             'isotopes': (12, 12, 16, 1, 1, 1, 1, 1, 12, 16, 1, 1, 1, 1),
                                             'coords': ((0.0001680285531732084, 2.5770474225077824, 0.1679368032120243),
                                                        (0.00018012176459177072, 1.1158677934318275, -0.22744331382375593),
                                                        (0.00018012176459177072, 0.3187981141998928, 0.9474327541498717),
                                                        (0.8796922989370138, 2.812139957756975, 0.776432804127883),
                                                        (0.00017379057608612246, 3.220807881040447, -0.7160567825175632),
                                                        (-0.879371991863, 2.812129791699508, 0.7764144557866137),
                                                        (-0.8911956013160998, 0.8771578165958389, -0.8152586387659195),
                                                        (0.8915649122746292, 0.8771703200314263, -0.8152504730704038),
                                                        (0.00018012176459177072, -1.9118296410415634, -1.1541375883576068),
                                                        (0.00018012176459177072, -1.9118296410415634, 0.26159640505254145),
                                                        (0.8957121157789865, -1.4031805894810436, -1.519404220290248),
                                                        (-0.021608921236217812, -2.9415027531385087, -1.519404287358789),
                                                        (-0.887154096327402, -1.3827009462369495, -1.5094989958271163),
                                                        (0.00018012176459177072, -0.7965359247781687, 0.6045083807221359))}))
        self.assertTrue(almost_equal_coords(rxn11.ts_species.ts_guesses[2].initial_xyz,
                                            {'symbols': ('C', 'C', 'O', 'H', 'H', 'H', 'H', 'H', 'C', 'O', 'H', 'H', 'H', 'H'),
                                             'isotopes': (12, 12, 16, 1, 1, 1, 1, 1, 12, 16, 1, 1, 1, 1),
                                             'coords': ((-1.134004075265406, 2.106914345635799, 1.0399329396496988),
                                                        (-0.6207602525122963, 1.0779688004252954, 0.05544160453432445),
                                                        (0.5546455384436214, 0.48033625355649145, 0.5816643845034846),
                                                        (-0.3766439101440957, 2.8760357938507166, 1.2242061163139133),
                                                        (-2.041532463623169, 2.587905422825105, 0.6645504738851482),
                                                        (-1.3547862623942049, 1.6403966815865265, 2.005705126284722),
                                                        (-1.3694923850133613, 0.29740901845530776, -0.10944201959765931),
                                                        (-0.3781814231894316, 1.5496951699532293, -0.9014754916021778),
                                                        (0.5546455384436214, -1.7502915016849647, -1.519905958003994),
                                                        (0.5546455384436214, -1.7502915016849647, -0.1041719645938457),
                                                        (1.450177532458016, -1.241642450124445, -1.8851725899366352),
                                                        (0.5328564954428119, -2.77996461378191, -1.8851726570051761),
                                                        (-0.33268867964837245, -1.221162806880351, -1.8752673654735035),
                                                        (0.5546455384436214, -0.63499778542157, 0.23874001107574871))}))
        self.assertTrue(almost_equal_coords(rxn11.ts_species.ts_guesses[4].initial_xyz,
                                            {'symbols': ('C', 'C', 'O', 'H', 'H', 'H', 'H', 'H', 'C', 'O', 'H', 'H', 'H', 'H'),
                                             'isotopes': (12, 12, 16, 1, 1, 1, 1, 1, 12, 16, 1, 1, 1, 1),
                                             'coords': ((-0.4477164764963672, 1.7159699936537183, 2.0154238000694313),
                                                        (-0.7874785138121851, 0.8687253327375415, 0.8079017141148697),
                                                        (0.3879272771437328, 0.6699670505949986, 0.03681791807509471),
                                                        (-0.03875559138170409, 2.68328917869697, 1.705353785156344),
                                                        (-1.3335305353186275, 1.8887989234564668, 2.6329477282135247),
                                                        (0.3190396781344225, 1.2276997811660224, 2.6259781550896744),
                                                        (-1.1736746515510768, -0.1063951464166284, 1.1194343501343251),
                                                        (-1.536300031452119, 1.36880157380032, 0.18641102804956766),
                                                        (0.3879272771437328, -1.5606607046464576, -2.064752424432384),
                                                        (0.3879272771437328, -1.5606607046464576, -0.6490184310222356),
                                                        (1.2834592711581274, -1.0520116530859378, -2.4300190563650252),
                                                        (0.36613823414292324, -2.590333816743403, -2.430019123433566),
                                                        (-0.49940694094826105, -1.0315320098418437, -2.4201138319018933),
                                                        (0.3879272771437328, -0.4453669883830629, -0.30610645535264114))}))
        self.assertTrue(almost_equal_coords(rxn11.ts_species.ts_guesses[6].initial_xyz,
                                            {'symbols': ('C', 'C', 'O', 'H', 'H', 'H', 'H', 'H', 'C', 'O', 'H', 'H', 'H', 'H'),
                                             'isotopes': (12, 12, 16, 1, 1, 1, 1, 1, 12, 16, 1, 1, 1, 1),
                                             'coords': ((0.0001874755718819162, 2.063180434892475, 1.8392479936813926),
                                                        (
                                                        0.00017538236046314698, 0.6322709618874114, 1.3454166201963274),
                                                        (0.0001753823604633132, 0.6329498121079993, -0.07432046384791535),
                                                        (-0.8793367948119585, 2.5995264781198886, 1.467941105548702),
                                                        (0.00018171354896911316, 2.099097162328274, 2.932217424302187),
                                                        (0.8797274959880551, 2.5995077597308427, 1.4679505729804174),
                                                        (0.8915511054411546, 0.10455108473483721, 1.697582831715791),
                                                        (-0.8912094081495741, 0.10456601597306348, 1.6975831011778144),
                                                        (0.0001753823604633132, -1.597677943133457, -2.175890806355394),
                                                        (0.0001753823604633132, -1.597677943133457, -0.7601568129452456),
                                                        (0.895707376374858, -1.0890288915729371, -2.5411574382880353),
                                                        (-0.02161366064034627, -2.627351055230402, -2.5411575053565763),
                                                        (-0.8871588357315305, -1.0685492483288432, -2.5312522138249034),
                                                        (0.0001753823604633132, -0.4823842268700622, -0.4172448372756512))}))

        # NH3 + OH <=> NH2 + H2O
        nh3 = ARCSpecies(label='NH3', smiles='N', xyz=self.nh3_xyz)
        oh = ARCSpecies(label='OH', smiles='[OH]', xyz=self.oh_xyz)
        nh2 = ARCSpecies(label='NH2', smiles='[NH2]', xyz=self.nh2_xyz)
        h2o = ARCSpecies(label='H2O', smiles='O', xyz=self.h2o_xyz)
        rxn12 = ARCReaction(r_species=[nh3, oh], p_species=[nh2, h2o])
        rxn12.determine_family(rmg_database=self.rmgdb)
        self.assertEqual(rxn12.family.label, 'H_Abstraction')
        heuristics_12 = HeuristicsAdapter(job_type='tsg',
                                          reactions=[rxn12],
                                          testing=True,
                                          project='test',
                                          project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'heuristics'),
                                          dihedral_increment=60,
                                          )
        heuristics_12.execute_incore()
        self.assertTrue(rxn12.ts_species.is_ts)
        self.assertEqual(rxn12.ts_species.charge, 0)
        self.assertEqual(rxn12.ts_species.multiplicity, 2)
        self.assertEqual(len(rxn12.ts_species.ts_guesses), 6)
        for i in range(6):
            self.assertTrue(rxn12.ts_species.ts_guesses[i].success)
        self.assertTrue(almost_equal_coords(rxn12.ts_species.ts_guesses[0].initial_xyz,
                                            {'symbols': ('N', 'H', 'H', 'H', 'O', 'H'), 'isotopes': (14, 1, 1, 1, 16, 1),
                                             'coords': ((0.026828661202958397, -1.183916026522652, -0.2917700509562986),
                                                        (0.026828661202958397, -0.008473410456561581, 0.045238258724209945),
                                                        (-0.8790431218203842, -1.5565901159683337, -0.010929724242103145),
                                                        (0.026828661202958397, -1.183916026522652, -1.310770145043425),
                                                        (0.026828661202958397, 1.1092928314476138, 0.3657103146572922),
                                                        (0.026828661202958397, 1.5934412844746615, -0.47367118649951234))}))
        self.assertTrue(almost_equal_coords(rxn12.ts_species.ts_guesses[1].initial_xyz,
                                            {'symbols': ('N', 'H', 'H', 'H', 'O', 'H'), 'isotopes': (14, 1, 1, 1, 16, 1),
                                             'coords': ((0.05094616431400374, -1.1800784514883642, -0.30515503022262447),
                                                        (0.05094616431400374, -0.0046358354222737486, 0.031853279457884076),
                                                        (-0.8549256187093388, -1.552752540934046, -0.024314703508429014),
                                                        (0.05094616431400374, -1.1800784514883642, -1.324155124309751),
                                                        (0.05094616431400374, 1.1131304064819016, 0.35232533539096633),
                                                        (-0.7633831598367127, 1.4677028470311904, -0.0351113243147112))}))
        self.assertTrue(almost_equal_coords(rxn12.ts_species.ts_guesses[2].initial_xyz,
                                            {'symbols': ('N', 'H', 'H', 'H', 'O', 'H'), 'isotopes': (14, 1, 1, 1, 16, 1),
                                             'coords': ((0.05094616431400374, -1.1724033014197888, -0.331924988755276),
                                                        (0.05094616431400374, 0.003039314646301694, 0.0050833209252325595),
                                                        (-0.8549256187093388, -1.5450773908654705, -0.05108466204108053),
                                                        (0.05094616431400374, -1.1724033014197888, -1.3509250828424024),
                                                        (0.05094616431400374, 1.120805556550477, 0.3255553768583148),
                                                        (-0.7633831598367128, 1.216225972144247, 0.8420084000548913))}))
        self.assertTrue(almost_equal_coords(rxn12.ts_species.ts_guesses[3].initial_xyz,
                                            {'symbols': ('N', 'H', 'H', 'H', 'O', 'H'), 'isotopes': (14, 1, 1, 1, 16, 1),
                                             'coords': ((0.0268286612029584, -1.168565726385501, -0.34530996802160185),
                                                        (0.0268286612029584, 0.006876889680589526, -0.00830165834109331),
                                                        (-0.8790431218203842, -1.5412398158311826, -0.0644696413074064),
                                                        (0.0268286612029584, -1.168565726385501, -1.3643100621087283),
                                                        (0.0268286612029584, 1.124643131584765, 0.31217039759198895),
                                                        (0.026828661202958286, 1.0904875347007754, 1.280568262239693))}))
        self.assertTrue(almost_equal_coords(rxn12.ts_species.ts_guesses[4].initial_xyz,
                                            {'symbols': ('N', 'H', 'H', 'H', 'O', 'H'), 'isotopes': (14, 1, 1, 1, 16, 1),
                                             'coords': ((0.002711158091913058, -1.1724033014197888, -0.331924988755276),
                                                        (0.002711158091913058, 0.003039314646301694, 0.0050833209252325595),
                                                        (-0.9031606249314296, -1.5450773908654705, -0.05108466204108053),
                                                        (0.002711158091913058, -1.1724033014197888, -1.3509250828424024),
                                                        (0.002711158091913058, 1.120805556550477, 0.3255553768583148),
                                                        (0.8170404822426293, 1.216225972144247, 0.8420084000548922))}))
        self.assertTrue(almost_equal_coords(rxn12.ts_species.ts_guesses[5].initial_xyz,
                                            {'symbols': ('N', 'H', 'H', 'H', 'O', 'H'), 'isotopes': (14, 1, 1, 1, 16, 1),
                                             'coords': ((0.0027111580919130514, -1.1800784514883642, -0.30515503022262447),
                                                        (0.0027111580919130514, -0.0046358354222737486, 0.031853279457884076),
                                                        (-0.9031606249314296, -1.552752540934046, -0.024314703508429014),
                                                        (0.0027111580919130514, -1.1800784514883642, -1.324155124309751),
                                                        (0.0027111580919130514, 1.1131304064819016, 0.35232533539096633),
                                                        (0.8170404822426295, 1.4677028470311904, -0.0351113243147112))}))

        # Reverse order
        # NH2 + H2O <=> NH3 + OH
        rxn13 = ARCReaction(r_species=[nh2, h2o], p_species=[nh3, oh])
        rxn13.determine_family(rmg_database=self.rmgdb)
        self.assertEqual(rxn13.family.label, 'H_Abstraction')
        heuristics_13 = HeuristicsAdapter(job_type='tsg',
                                          reactions=[rxn13],
                                          testing=True,
                                          project='test',
                                          project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'heuristics'),
                                          dihedral_increment=180,
                                          )
        heuristics_13.execute_incore()
        self.assertTrue(rxn13.ts_species.is_ts)
        self.assertEqual(rxn13.ts_species.charge, 0)
        self.assertEqual(rxn13.ts_species.multiplicity, 2)
        self.assertEqual(len(rxn13.ts_species.ts_guesses), 2)
        self.assertTrue(almost_equal_coords(rxn13.ts_species.ts_guesses[0].initial_xyz,
                                            {'symbols': ('N', 'H', 'H', 'O', 'H', 'H'), 'isotopes': (14, 1, 1, 16, 1, 1),
                                             'coords': ((0.02682866175036011, 1.1806016621578093, 0.3382386588330619),
                                                        (0.02682866175036011, 1.6897319854135922, -0.5444545984662927),
                                                        (-0.8790431397560428, 1.049996458867457, 0.7862336514836761),
                                                        (0.02682866175036011, -1.1343573639334195, -0.2380014751731998),
                                                        (0.02682866175036011, -0.005989553115606228, 0.04287210020574728),
                                                        (0.02682866175036011, -1.1343573639334195, -1.2070014286527884))}))

        # different reactant order order
        # H2O + NH2 <=> NH3 + OH
        rxn14 = ARCReaction(r_species=[h2o, nh2], p_species=[nh3, oh])
        rxn14.determine_family(rmg_database=self.rmgdb)
        self.assertEqual(rxn14.family.label, 'H_Abstraction')
        heuristics_14 = HeuristicsAdapter(job_type='tsg',
                                          reactions=[rxn14],
                                          testing=True,
                                          project='test',
                                          project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'heuristics'),
                                          dihedral_increment=180,
                                          )
        heuristics_14.execute_incore()
        self.assertTrue(rxn14.ts_species.is_ts)
        self.assertEqual(rxn14.ts_species.charge, 0)
        self.assertEqual(rxn14.ts_species.multiplicity, 2)
        self.assertEqual(len(rxn14.ts_species.ts_guesses), 2)
        self.assertTrue(almost_equal_coords(rxn14.ts_species.ts_guesses[0].initial_xyz,
                                            {'symbols': ('O', 'H', 'H', 'N', 'H', 'H'), 'isotopes': (16, 1, 1, 14, 1, 1),
                                             'coords': ((0.02682866175036011, -1.1343573639334195, -0.2380014751731998),
                                                        (0.02682866175036011, -0.005989553115606228, 0.04287210020574728),
                                                        (0.02682866175036011, -1.1343573639334195, -1.2070014286527884),
                                                        (0.02682866175036011, 1.1806016621578093, 0.3382386588330619),
                                                        (0.02682866175036011, 1.6897319854135922, -0.5444545984662927),
                                                        (-0.8790431397560428, 1.049996458867457, 0.7862336514836761))}))

        # different product order order
        # NH2 + H2O <=> OH + NH3
        rxn15 = ARCReaction(r_species=[h2o, nh2], p_species=[nh3, oh])
        rxn15.determine_family(rmg_database=self.rmgdb)
        self.assertEqual(rxn15.family.label, 'H_Abstraction')
        heuristics_15 = HeuristicsAdapter(job_type='tsg',
                                          reactions=[rxn15],
                                          testing=True,
                                          project='test',
                                          project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'heuristics'),
                                          dihedral_increment=180,
                                          )
        heuristics_15.execute_incore()
        self.assertTrue(rxn15.ts_species.is_ts)
        self.assertEqual(rxn15.ts_species.charge, 0)
        self.assertEqual(rxn15.ts_species.multiplicity, 2)
        self.assertEqual(len(rxn15.ts_species.ts_guesses), 2)
        self.assertTrue(almost_equal_coords(rxn15.ts_species.ts_guesses[0].initial_xyz,
                                            {'symbols': ('O', 'H', 'H', 'N', 'H', 'H'), 'isotopes': (16, 1, 1, 14, 1, 1),
                                             'coords': ((0.02682866175036011, -1.1343573639334195, -0.2380014751731998),
                                                        (0.02682866175036011, -0.005989553115606228, 0.04287210020574728),
                                                        (0.02682866175036011, -1.1343573639334195, -1.2070014286527884),
                                                        (0.02682866175036011, 1.1806016621578093, 0.3382386588330619),
                                                        (0.02682866175036011, 1.6897319854135922, -0.5444545984662927),
                                                        (-0.8790431397560428, 1.049996458867457, 0.7862336514836761))}))

        # NFCl + H2O <=> NHFCl + OH
        hnfcl = ARCSpecies(label='NHFCl', smiles='N(F)Cl', xyz="""N      -0.14626256    0.12816405    0.30745256
                                                                  F      -0.94719775   -0.91910939   -0.09669786
                                                                  Cl      1.53982436   -0.20497454   -0.07627978
                                                                  H      -0.44636405    0.99591988   -0.13447493""")
        nfcl = ARCSpecies(label='NFCl', smiles='[N](F)(Cl)', xyz="""N      -0.17697493    0.58788903    0.00000000
                                                                    F      -1.17300047   -0.36581404    0.00000000
                                                                    Cl      1.34997541   -0.22207499    0.00000000""")
        rxn16 = ARCReaction(r_species=[nfcl, h2o], p_species=[hnfcl, oh])
        rxn16.determine_family(rmg_database=self.rmgdb)
        self.assertEqual(rxn16.family.label, 'H_Abstraction')
        heuristics_16 = HeuristicsAdapter(job_type='tsg',
                                          reactions=[rxn16],
                                          testing=True,
                                          project='test',
                                          project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'heuristics'),
                                          dihedral_increment=360,
                                          )
        heuristics_16.execute_incore()
        self.assertEqual(len(rxn16.ts_species.ts_guesses), 1)
        self.assertTrue(rxn16.ts_species.ts_guesses[0].success)
        self.assertTrue(almost_equal_coords(rxn16.ts_species.ts_guesses[0].initial_xyz,
                                            {'symbols': ('N', 'F', 'Cl', 'O', 'H', 'H'), 'isotopes': (14, 19, 35, 16, 1, 1),
                                             'coords': ((-0.24126731380591912, 0.0011327422566265177, 0.5459320349532694),
                                                        (0.8506338994152145, 0.3005202886236016, 1.3331646388015477),
                                                        (-0.24126731380591912, 0.9954816716934416, -0.9074747062308236),
                                                        (-0.24126731380591912, -2.3138262157211655, -0.030308082098179923),
                                                        (-0.24126731380591912, -1.1854584049033523, 0.25056549328076716),
                                                        (-0.24126731380591912, -2.3138262157211655, -0.9993080355777685))}))



    def test_keeping_atom_order_in_ts(self):
        """Test that the generated TS has the same atom order as in the reactants"""
        # reactant_reversed, products_reversed = False, False
        rxn_1 = ARCReaction(r_species=[ARCSpecies(label='C2H6', smiles='CC', xyz=self.c2h6_xyz),
                                       ARCSpecies(label='CCOOj', smiles='CCO[O]', xyz=self.ccooj_xyz)],
                            p_species=[ARCSpecies(label='C2H5', smiles='C[CH2]', xyz=self.c2h5_xyz),
                                       ARCSpecies(label='CCOOH', smiles='CCOO', xyz=self.ccooh_xyz)])
        rxn_1.determine_family(rmg_database=self.rmgdb)
        self.assertIn(rxn_1.atom_map[0], [0, 1])
        self.assertIn(rxn_1.atom_map[1], [0, 1])
        for index in [2, 3, 4, 5, 6, 7]:
            self.assertIn(rxn_1.atom_map[index], [2, 3, 4, 5, 6, 16])
        self.assertEqual(rxn_1.atom_map[8:], [7, 8, 9, 10, 13, 11, 12, 14, 15])
        heuristics_1 = HeuristicsAdapter(job_type='tsg',
                                         reactions=[rxn_1],
                                         testing=True,
                                         project='test_1',
                                         project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'heuristics_1'),
                                         dihedral_increment=120,
                                         )
        heuristics_1.execute_incore()
        for tsg in rxn_1.ts_species.ts_guesses:
            self.assertEqual(tsg.initial_xyz['symbols'],
                             ('C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'C', 'C', 'O', 'O', 'H', 'H', 'H', 'H', 'H'))

        # reactant_reversed, products_reversed = False, True
        rxn_2 = ARCReaction(r_species=[ARCSpecies(label='C2H6', smiles='CC', xyz=self.c2h6_xyz),
                                       ARCSpecies(label='CCOOj', smiles='CCO[O]', xyz=self.ccooj_xyz)],
                            p_species=[ARCSpecies(label='CCOOH', smiles='CCOO', xyz=self.ccooh_xyz),
                                       ARCSpecies(label='C2H5', smiles='C[CH2]', xyz=self.c2h5_xyz)])
        rxn_2.determine_family(rmg_database=self.rmgdb)
        self.assertEqual(rxn_2.family.label, 'H_Abstraction')
        self.assertEqual(rxn_2.atom_map, [11, 10, 9, 16, 15, 14, 12, 13, 0, 1, 2, 3, 6, 4, 5, 7, 8])
        heuristics_2 = HeuristicsAdapter(job_type='tsg',
                                         reactions=[rxn_2],
                                         testing=True,
                                         project='test_1',
                                         project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'heuristics_1'),
                                         dihedral_increment=120,
                                         )
        heuristics_2.execute_incore()
        for tsg in rxn_2.ts_species.ts_guesses:
            self.assertEqual(tsg.initial_xyz['symbols'],
                             ('C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'C', 'C', 'O', 'O', 'H', 'H', 'H', 'H', 'H'))

        # reactant_reversed, products_reversed = True, False
        rxn_3 = ARCReaction(r_species=[ARCSpecies(label='CCOOj', smiles='CCO[O]', xyz=self.ccooj_xyz),
                                       ARCSpecies(label='C2H6', smiles='CC', xyz=self.c2h6_xyz)],
                            p_species=[ARCSpecies(label='C2H5', smiles='C[CH2]', xyz=self.c2h5_xyz),
                                       ARCSpecies(label='CCOOH', smiles='CCOO', xyz=self.ccooh_xyz)])
        rxn_3.determine_family(rmg_database=self.rmgdb)
        self.assertEqual(rxn_3.atom_map, [7, 8, 9, 10, 13, 11, 12, 14, 15, 1, 0, 16, 6, 5, 4, 2, 3])
        heuristics_3 = HeuristicsAdapter(job_type='tsg',
                                         reactions=[rxn_3],
                                         testing=True,
                                         project='test_1',
                                         project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'heuristics_1'),
                                         dihedral_increment=120,
                                         )
        heuristics_3.execute_incore()
        for tsg in rxn_3.ts_species.ts_guesses:
            self.assertEqual(tsg.initial_xyz['symbols'],
                             ('C', 'C', 'O', 'O', 'H', 'H', 'H', 'H', 'H', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H'))

        # reactant_reversed, products_reversed = True, True
        rxn_4 = ARCReaction(r_species=[ARCSpecies(label='CCOOj', smiles='CCO[O]', xyz=self.ccooj_xyz),
                                       ARCSpecies(label='C2H6', smiles='CC', xyz=self.c2h6_xyz)],
                            p_species=[ARCSpecies(label='CCOOH', smiles='CCOO', xyz=self.ccooh_xyz),
                                       ARCSpecies(label='C2H5', smiles='C[CH2]', xyz=self.c2h5_xyz)])
        rxn_4.determine_family(rmg_database=self.rmgdb)
        self.assertEqual(rxn_4.atom_map, [0, 1, 2, 3, 6, 4, 5, 7, 8, 11, 10, 9, 16, 15, 14, 12, 13])
        heuristics_4 = HeuristicsAdapter(job_type='tsg',
                                         reactions=[rxn_4],
                                         testing=True,
                                         project='test_1',
                                         project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'heuristics_1'),
                                         dihedral_increment=120,
                                         )
        heuristics_4.execute_incore()
        for tsg in rxn_4.ts_species.ts_guesses:
            self.assertEqual(tsg.initial_xyz['symbols'],
                             ('C', 'C', 'O', 'O', 'H', 'H', 'H', 'H', 'H', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H'))

    def test_combine_coordinates_with_redundant_atoms(self):
        """Test the combine_coordinates_with_redundant_atoms() function."""
        ts_xyz = combine_coordinates_with_redundant_atoms(xyz_1=self.ccooh_xyz,
                                                          xyz_2=self.c2h6_xyz,
                                                          mol_1=ARCSpecies(label='CCOOH', smiles='CCOO', xyz=self.ccooh_xyz).mol,
                                                          mol_2=ARCSpecies(label='C2H6', smiles='CC', xyz=self.c2h6_xyz).mol,
                                                          reactant_2=ARCSpecies(label='C2H5', smiles='C[CH2]', xyz=self.c2h5_xyz),
                                                          h1=9,
                                                          h2=5,
                                                          c=2,
                                                          d=0,
                                                          a2=180,
                                                          d2=None,
                                                          d3=0,
                                                          reactants_reversed=False,
                                                          )
        expected_xyz = {'symbols': ('C', 'C', 'O', 'O', 'H', 'H', 'H', 'H', 'H', 'H', 'C', 'C', 'H', 'H', 'H', 'H', 'H'),
                        'isotopes': (12, 12, 16, 16, 1, 1, 1, 1, 1, 1, 12, 12, 1, 1, 1, 1, 1),
                        'coords': ((0.8187041630923411, -1.305974629356673, -2.1958802480028368),
                                   (0.8187041630923411, -1.305974629356673, -0.681132252881617),
                                   (0.8187041630923411, 0.04581686905211524, -0.22531432865752898),
                                   (0.818513837528933, -0.03174140807870107, 1.2285438851644308),
                                   (1.6983292424348284, -0.7800904747940467, -2.5815947431771877),
                                   (0.81871196114839, -2.3274935937515515, -2.586251749003749),
                                   (-0.06092771998370783, -0.7800974898360937, -2.5815561835518057),
                                   (1.7113460420860538, -1.8153071614909009, -0.3028337456942676),
                                   (-0.07395075446213972, -1.8153703481853025, -0.30295622775763786),
                                   (-0.2107062743589989, 0.499850734706512, 1.3852370918171815),
                                   (-1.9718315856107542, 1.4854394299684088, 0.2311504747215567),
                                   (-1.366661074542108, 1.096901338454432, 1.5612249706918742),
                                   (-2.9351272524300116, 1.9829815997583418, 0.3778070404504672),
                                   (-2.1325990339431034, 0.6001332666491552, -0.39126549020899803),
                                   (-1.3092655055659, 2.1694913032490764, -0.30744489564843125),
                                   (-2.02923513711407, 0.41285009988656296, 2.099824496568834),
                                   (-1.2058935954739198, 1.9822075052998445, 2.1836409226679843))}
        self.assertTrue(almost_equal_coords(ts_xyz, expected_xyz))

        ts_xyz = combine_coordinates_with_redundant_atoms(xyz_1=self.ccooh_xyz,
                                                          xyz_2=self.c2h6_xyz,
                                                          mol_1=ARCSpecies(label='CCOOH', smiles='CCOO', xyz=self.ccooh_xyz).mol,
                                                          mol_2=ARCSpecies(label='C2H6', smiles='CC', xyz=self.c2h6_xyz).mol,
                                                          reactant_2=ARCSpecies(label='C2H5', smiles='C[CH2]', xyz=self.c2h5_xyz),
                                                          h1=9,
                                                          h2=5,
                                                          c=2,
                                                          d=0,
                                                          a2=150,
                                                          d2=30,
                                                          d3=120,
                                                          reactants_reversed=False,
                                                          )
        expected_xyz = {'symbols': ('C', 'C', 'O', 'O', 'H', 'H', 'H', 'H', 'H', 'H', 'C', 'C', 'H', 'H', 'H', 'H', 'H'),
                        'isotopes': (12, 12, 16, 16, 1, 1, 1, 1, 1, 1, 12, 12, 1, 1, 1, 1, 1),
                        'coords': ((0.9077118476535074, -1.055592038162869, -2.378138314624377),
                                   (0.9077118476535074, -1.055592038162869, -0.8633903195031574),
                                   (0.9077118476535074, 0.29619946024591925, -0.4075723952790693),
                                   (0.9075215220900994, 0.21864118311510294, 1.0462858185428905),
                                   (1.787336926995995, -0.5297078836002427, -2.763852809798728),
                                   (0.9077196457095563, -2.0771110025577473, -2.7685098156252894),
                                   (0.028079964577458538, -0.5297148986422897, -2.763814250173346),
                                   (1.80035372664722, -1.5649245702970966, -0.4850918123158079),
                                   (0.015056930099026644, -1.5649877569914987, -0.4852142943791782),
                                   (-0.12169858979783255, 0.750233325900316, 1.2029790251956411),
                                   (-2.3703610220276, 0.5257644475934526, 1.7488676094956128),
                                   (-1.3302908291148148, 1.035981385760534, 0.7771557022531046),
                                   (-3.3775212214584185, 0.7638878308103012, 1.3940148403768324),
                                   (-2.234527957629507, 0.984077927652405, 2.732998160422298),
                                   (-2.292191326440104, -0.5597760994377934, 1.8606574488891878),
                                   (-1.408466403673402, 2.1215276573011588, 0.6653621132557062),
                                   (-1.4661238729956745, 0.5776678806782756, -0.2069748398519864))}
        self.assertTrue(almost_equal_coords(ts_xyz, expected_xyz))

    def test_get_new_zmat2_map(self):
        """Test the get_new_zmat_2_map() function."""
        zmat_1 = {'symbols': ('C', 'C', 'O', 'O', 'H', 'H', 'H', 'H', 'H', 'H', 'X'),
                  'coords': ((None, None, None), ('R_1_0', None, None), ('R_2_1', 'A_2_1_0', None),
                             ('R_3_2', 'A_3_2_1', 'D_3_2_1_0'), ('R_4_0', 'A_4_0_1', 'D_4_0_1_3'),
                             ('R_5_0', 'A_5_0_1', 'D_5_0_1_4'), ('R_6_0', 'A_6_0_1', 'D_6_0_1_5'),
                             ('R_7_1', 'A_7_1_0', 'D_7_1_0_6'), ('R_8_1', 'A_8_1_0', 'D_8_1_0_7'),
                             ('R_9_3', 'A_9_3_2', 'D_9_3_2_1'), ('RX_10_9', 'AX_10_9_3', 'DX_10_9_3_2')),
                  'vars': {'R_1_0': 1.5147479951212197, 'R_2_1': 1.4265728986680748, 'A_2_1_0': 108.63387152978416,
                           'R_3_2': 1.4559254886404387, 'A_3_2_1': 105.58023544826183, 'D_3_2_1_0': 179.9922243050821,
                           'R_4_0': 1.0950205915944824, 'A_4_0_1': 110.62463321031589, 'D_4_0_1_3': 59.13545080998071,
                           'R_5_0': 1.093567969297245, 'A_5_0_1': 110.91425998596507, 'D_5_0_1_4': 120.87266977773987,
                           'R_6_0': 1.0950091062890002, 'A_6_0_1': 110.62270362433773, 'D_6_0_1_5': 120.87301274044218,
                           'R_7_1': 1.0951433842986755, 'A_7_1_0': 110.20822115119915, 'D_7_1_0_6': 181.16392677464265,
                           'R_8_1': 1.0951410439636102, 'A_8_1_0': 110.20143800025897, 'D_8_1_0_7': 239.4199964284852,
                           'R_9_3': 1.1689469645782498, 'A_9_3_2': 96.30065819269021, 'D_9_3_2_1': 242.3527063196313,
                           'RX_10_9': 1.0, 'AX_10_9_3': 90.0, 'DX_10_9_3_2': 0},
                  'map': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 'X10'}}
        zmat_2 = {'symbols': ('H', 'C', 'C', 'H', 'H', 'H', 'H', 'H'),
                  'coords': ((None, None, None), ('R_1_0', None, None), ('R_2_1', 'A_2_1_0', None),
                             ('R_3_1', 'A_3_1_2', 'D_3_1_2_0'), ('R_4_1', 'A_4_1_2', 'D_4_1_2_3'),
                             ('R_5_2', 'A_5_2_1', 'D_5_2_1_4'), ('R_6_2', 'A_6_2_1', 'D_6_2_1_5'),
                             ('R_7_2', 'A_7_2_1', 'D_7_2_1_6')),
                  'vars': {'R_1_0': 1.3128870801982788, 'R_2_1': 1.5120487296562577, 'A_2_1_0': 110.56890700195424,
                           'R_3_1': 1.0940775789443724, 'A_3_1_2': 110.56801921096591, 'D_3_1_2_0': 120.00061587714492,
                           'R_4_1': 1.0940817193677925, 'A_4_1_2': 110.56754686774481, 'D_4_1_2_3': 119.99910067703652,
                           'R_5_2': 1.0940725668318991, 'A_5_2_1': 110.56890700195424, 'D_5_2_1_4': 59.99971758419434,
                           'R_6_2': 1.0940840619688397, 'A_6_2_1': 110.56790845138725, 'D_6_2_1_5': 239.99905123159166,
                           'R_7_2': 1.0940817193677925, 'A_7_2_1': 110.56754686774481, 'D_7_2_1_6': 240.00122783407815},
                  'map': {0: 2, 1: 0, 2: 1, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}}
        new_map = get_new_zmat_2_map(zmat_1=zmat_1,
                                     zmat_2=zmat_2,
                                     reactant_2=ARCSpecies(label='C2H5', smiles='C[CH2]', xyz=self.c2h5_xyz),
                                     reactants_reversed=False,
                                     )
        expected_new_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 'X10',
                            11: 12, 12: 11, 13: 17, 14: 16, 15: 15, 16: 13, 17: 14}
        self.assertEqual(new_map, expected_new_map)

        new_map = get_new_zmat_2_map(zmat_1=zmat_1,
                                     zmat_2=zmat_2,
                                     reactant_2=ARCSpecies(label='C2H5', smiles='C[CH2]', xyz=self.c2h5_xyz),
                                     reactants_reversed=True,
                                     )
        expected_new_map = {0: 7, 1: 8, 2: 9, 3: 10, 4: 11, 5: 12, 6: 13, 7: 14, 8: 15, 9: 16, 10: 'X17',
                            11: 1, 12: 0, 13: 6, 14: 5, 15: 4, 16: 2, 17: 3}
        self.assertEqual(new_map, expected_new_map)

    def test_react(self):
        """Test the react() function and specifically that atom order in kept."""
        rxn_1 = ARCReaction(r_species=[ARCSpecies(label='C2H6', smiles='CC', xyz=self.c2h6_xyz),
                                       ARCSpecies(label='CCOOj', smiles='CCO[O]', xyz=self.ccooj_xyz)],
                            p_species=[ARCSpecies(label='C2H5', smiles='C[CH2]', xyz=self.c2h5_xyz),
                                       ARCSpecies(label='CCOOH', smiles='CCOO', xyz=self.ccooh_xyz)])
        rxn_1.determine_family(rmg_database=self.rmgdb, save_order=True)
        reactants, products = rxn_1.get_reactants_and_products(arc=True)
        reactant_mol_combinations = list(itertools.product(*list(reactant.mol_list for reactant in reactants)))
        product_mol_combinations = list(itertools.product(*list(product.mol_list for product in products)))
        reactants = list(reactant_mol_combinations)[0]
        products = list(product_mol_combinations)[0]
        rmg_reactions = react(reactants=list(reactants),
                              products=list(products),
                              family=rxn_1.family,
                              arc_reaction=rxn_1,
                              )
        self.assertTrue(_check_r_n_p_symbols_between_rmg_and_arc_rxns(rxn_1, rmg_reactions))

    def test_generate_the_two_constrained_zmats(self):
        """Test the generate_the_two_constrained_zmats() function."""
        zmat_1, zmat_2 = generate_the_two_constrained_zmats(xyz_1=self.ccooh_xyz,
                                                            xyz_2=self.c2h6_xyz,
                                                            mol_1=ARCSpecies(label='CCOOH', smiles='CCOO', xyz=self.ccooh_xyz).mol,
                                                            mol_2=ARCSpecies(label='C2H6', smiles='CC', xyz=self.c2h6_xyz).mol,
                                                            h1=9,
                                                            h2=3,
                                                            a=3,
                                                            b=0,
                                                            c=2,
                                                            d=1,
                                                            )
        self.assertTrue(_compare_zmats(zmat_1, self.zmat_1, r_tol=0.01, a_tol=0.01, d_tol=0.01))
        self.assertTrue(_compare_zmats(zmat_2, self.zmat_2, r_tol=0.01, a_tol=0.01, d_tol=0.01))

        zmat_1, zmat_2 = generate_the_two_constrained_zmats(xyz_1=self.ch4_xyz,
                                                            xyz_2=self.h2_xyz,
                                                            mol_1=self.ch4_mol,
                                                            mol_2=self.h2_mol,
                                                            h1=2,
                                                            h2=0,
                                                            a=0,
                                                            b=1,
                                                            c=None,
                                                            d=None,
                                                            )
        expected_zmat_1 = {'symbols': ('C', 'H', 'H', 'H', 'H'),
                           'coords': ((None, None, None), ('R_1_0', None, None), ('R_2_0', 'A_2_0_1', None),
                                      ('R_3_0', 'A_3_0_1', 'D_3_0_1_2'), ('R_4_0', 'A_4_0_1', 'D_4_0_1_3')),  # R_4_0
                           'vars': {'R_1_0': 1.092199370793132, 'R_2_0': 1.0921994253661749,
                                    'A_2_0_1': 109.47122156965536, 'R_3_0': 1.092199370793132,
                                    'A_3_0_1': 109.47122278898594, 'D_3_0_1_2': 120.00000135665665,
                                    'R_4_0': 1.0921994253661749, 'A_4_0_1': 109.47121850997881,
                                    'D_4_0_1_3': 120.00000068116007},
                           'map': {0: 0, 1: 1, 2: 3, 3: 4, 4: 2}}
        expected_zmat_2 = {'symbols': ('H', 'H'), 'coords': ((None, None, None), ('R_1_0', None, None)),  # R_1_0
                           'vars': {'R_1_0': 0.7473099866382779}, 'map': {0: 0, 1: 1}}
        self.assertTrue(_compare_zmats(zmat_1, expected_zmat_1, r_tol=0.01, a_tol=0.01, d_tol=0.01))
        self.assertTrue(_compare_zmats(zmat_2, expected_zmat_2, r_tol=0.01, a_tol=0.01, d_tol=0.01))

        zmat_1, zmat_2 = generate_the_two_constrained_zmats(xyz_1=self.n2h4_xyz,
                                                            xyz_2=self.nh3_xyz,
                                                            mol_1=self.n2h4_mol,
                                                            mol_2=self.nh3_mol,
                                                            h1=2,
                                                            h2=1,
                                                            a=0,
                                                            b=0,
                                                            c=1,
                                                            d=None,
                                                            )
        expected_zmat_1 = {'symbols': ('N', 'N', 'H', 'H', 'H', 'H'),
                           'coords': ((None, None, None), ('R_1_0', None, None), ('R_2_1', 'A_2_1_0', None),  # A_2_1_0
                                      ('R_3_1', 'A_3_1_0', 'D_3_1_0_2'), ('R_4_0', 'A_4_0_1', 'D_4_0_1_3'),
                                      ('R_5_0', 'A_5_0_1', 'D_5_0_1_4')),
                           'vars': {'R_1_0': 1.4346996064735746, 'R_2_1': 1.023964433208832,
                                    'A_2_1_0': 113.24587551512498, 'R_3_1': 1.0248853619364922,
                                    'A_3_1_0': 111.58697299955385, 'D_3_1_0_2': 240.07077704046898,
                                    'R_4_0': 1.0239645496281908, 'A_4_0_1': 113.24586240810203,
                                    'D_4_0_1_3': 284.3887260014507, 'R_5_0': 1.024885187464345,
                                    'A_5_0_1': 111.5869758085818, 'D_5_0_1_4': 240.07079516383422},
                           'map': {0: 1, 1: 0, 2: 2, 3: 3, 4: 4, 5: 5}}
        expected_zmat_2 = {'symbols': ('H', 'N', 'H', 'H'),
                           'coords': ((None, None, None), ('R_1_0', None, None), ('R_2_1', 'A_2_1_0', None),  # R_1_0
                                      ('R_3_1', 'A_3_1_0', 'D_3_1_0_2')),
                           'vars': {'R_1_0': 1.0190000355938578, 'R_2_1': 1.0189999771005855,
                                    'A_2_1_0': 105.99799962283844, 'R_3_1': 1.0190000940871264,
                                    'A_3_1_0': 105.99799852287603, 'D_3_1_0_2': 112.36218461898632},
                           'map': {0: 1, 1: 0, 2: 2, 3: 3}}
        self.assertTrue(_compare_zmats(zmat_1, expected_zmat_1, r_tol=0.01, a_tol=0.01, d_tol=0.01))
        self.assertTrue(_compare_zmats(zmat_2, expected_zmat_2, r_tol=0.01, a_tol=0.01, d_tol=0.01))

        zmat_1, zmat_2 = generate_the_two_constrained_zmats(xyz_2=self.n2h4_xyz,
                                                            xyz_1=self.nh3_xyz,
                                                            mol_2=self.n2h4_mol,
                                                            mol_1=self.nh3_mol,
                                                            h2=2,
                                                            h1=1,
                                                            b=0,
                                                            a=0,
                                                            d=1,
                                                            c=None,
                                                            )
        expected_zmat_1 = {'symbols': ('N', 'H', 'H', 'H'),
                           'coords': ((None, None, None), ('R_1_0', None, None), ('R_2_0', 'A_2_0_1', None),
                                      ('R_3_0', 'A_3_0_1', 'D_3_0_1_2')),
                           'vars': {'R_1_0': 1.0189999771005855, 'R_2_0': 1.0190000940871264,
                                    'A_2_0_1': 105.9980011877756, 'R_3_0': 1.0190000355938578,  # R_3_0
                                    'A_3_0_1': 105.99799962283844, 'D_3_0_1_2': 112.36217876566015},
                           'map': {0: 0, 1: 2, 2: 3, 3: 1}}
        expected_zmat_2 = {'symbols': ('H', 'N', 'N', 'H', 'H', 'H'),
                           'coords': ((None, None, None), ('R_1_0', None, None), ('R_2_1', 'A_2_1_0', None),  # A_2_1_0
                                      ('R_3_1', 'A_3_1_2', 'D_3_1_2_0'), ('R_4_2', 'A_4_2_1', 'D_4_2_1_3'),
                                      ('R_5_2', 'A_5_2_1', 'D_5_2_1_4')),
                           'vars': {'R_1_0': 1.023964433208832, 'R_2_1': 1.4346996064735746,
                                    'A_2_1_0': 113.24587551512498, 'R_3_1': 1.0248853619364922,
                                    'A_3_1_2': 111.58697299955385, 'D_3_1_2_0': 240.07077704046898,
                                    'R_4_2': 1.0239645496281908, 'A_4_2_1': 113.24586240810203,
                                    'D_4_2_1_3': 284.3887260014507, 'R_5_2': 1.024885187464345,
                                    'A_5_2_1': 111.5869758085818, 'D_5_2_1_4': 240.07079516383422},
                           'map': {0: 2, 1: 0, 2: 1, 3: 3, 4: 4, 5: 5}}
        self.assertTrue(_compare_zmats(zmat_1, expected_zmat_1, r_tol=0.01, a_tol=0.01, d_tol=0.01))
        self.assertTrue(_compare_zmats(zmat_2, expected_zmat_2, r_tol=0.01, a_tol=0.01, d_tol=0.01))

        zmat_1, zmat_2 = generate_the_two_constrained_zmats(xyz_1=self.ch3ch2oh.get_xyz(),
                                                            xyz_2=self.ch3ooh.get_xyz(),
                                                            mol_1=self.ch3ch2oh.mol,
                                                            mol_2=self.ch3ooh.mol,
                                                            h1=8,
                                                            a=2,
                                                            c=1,
                                                            h2=6,
                                                            b=2,
                                                            d=1,
                                                            )
        expected_zmat_1 = {'symbols': ('C', 'C', 'O', 'H', 'H', 'H', 'H', 'H', 'H'),
                           'coords': ((None, None, None), ('R_1_0', None, None), ('R_2_1', 'A_2_1_0', None),
                                      ('R_3_0', 'A_3_0_1', 'D_3_0_1_2'), ('R_4_0', 'A_4_0_1', 'D_4_0_1_3'),
                                      ('R_5_0', 'A_5_0_1', 'D_5_0_1_4'), ('R_6_1', 'A_6_1_0', 'D_6_1_0_5'),
                                      ('R_7_1', 'A_7_1_0', 'D_7_1_0_6'), ('R_8_2', 'A_8_2_1', 'D_8_2_1_0')),  # A_8_2_1
                           'vars': {'R_1_0': 1.5137276325416074, 'R_2_1': 1.4197372463410514,
                                    'A_2_1_0': 109.01303538097567, 'R_3_0': 1.0950337097344136,
                                    'A_3_0_1': 110.63258593497066, 'D_3_0_1_2': 59.12026293645304,
                                    'R_4_0': 1.0935594120185885, 'A_4_0_1': 110.92258531860486,
                                    'D_4_0_1_3': 120.87939658650379, 'R_5_0': 1.095033981893329,
                                    'A_5_0_1': 110.63254968567193, 'D_5_0_1_4': 120.8793955156551,
                                    'R_6_1': 1.0941026391623285, 'A_6_1_0': 110.5440977183771,
                                    'D_6_1_0_5': 181.34310124526917, 'R_7_1': 1.0941023667717409,
                                    'A_7_1_0': 110.54410081124645, 'D_7_1_0_6': 239.07189759901027,
                                    'R_8_2': 0.9723850776119742, 'A_8_2_1': 107.06334992092434,
                                    'D_8_2_1_0': 179.99951584936258},
                           'map': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8}}
        expected_zmat_2 = {'symbols': ('H', 'O', 'O', 'C', 'H', 'H', 'H'),
                           'coords': ((None, None, None), ('R_1_0', None, None), ('R_2_1', 'A_2_1_0', None),  # A_2_1_0
                                      ('R_3_2', 'A_3_2_1', 'D_3_2_1_0'), ('R_4_3', 'A_4_3_2', 'D_4_3_2_1'),
                                      ('R_5_3', 'A_5_3_2', 'D_5_3_2_4'), ('R_6_3', 'A_6_3_2', 'D_6_3_2_5')),
                           'vars': {'R_1_0': 0.9741406737537205, 'R_2_1': 1.4557553347542735,
                                    'A_2_1_0': 96.30924284405943, 'R_3_2': 1.4223004218216138,
                                    'A_3_2_1': 105.53988198924208, 'D_3_2_1_0': 242.22744613021646,
                                    'R_4_3': 1.093821659465454, 'A_4_3_2': 110.03248558165065,
                                    'D_4_3_2_1': 60.84271231265853, 'R_5_3': 1.0938084178037755,
                                    'A_5_3_2': 110.03299489037433, 'D_5_3_2_4': 238.4134975536592,
                                    'R_6_3': 1.0928700313199922, 'A_6_3_2': 108.55511996651099,
                                    'D_6_3_2_5': 240.7911024479184},
                           'map': {0: 6, 1: 2, 2: 1, 3: 0, 4: 3, 5: 4, 6: 5}}
        self.assertTrue(_compare_zmats(zmat_1, expected_zmat_1, r_tol=0.01, a_tol=0.01, d_tol=0.01))
        self.assertTrue(_compare_zmats(zmat_2, expected_zmat_2, r_tol=0.01, a_tol=0.01, d_tol=0.01))

        zmat_1, zmat_2 = generate_the_two_constrained_zmats(xyz_2=self.ch3ch2oh.get_xyz(),
                                                            xyz_1=self.ch3ooh.get_xyz(),
                                                            mol_2=self.ch3ch2oh.mol,
                                                            mol_1=self.ch3ooh.mol,
                                                            h2=8,
                                                            b=2,
                                                            d=1,
                                                            h1=6,
                                                            a=2,
                                                            c=1,
                                                            )
        expected_zmat_1 = {'symbols': ('C', 'O', 'O', 'H', 'H', 'H', 'H'),
                           'coords': ((None, None, None), ('R_1_0', None, None), ('R_2_1', 'A_2_1_0', None),
                                      ('R_3_0', 'A_3_0_1', 'D_3_0_1_2'), ('R_4_0', 'A_4_0_1', 'D_4_0_1_3'),
                                      ('R_5_0', 'A_5_0_1', 'D_5_0_1_4'), ('R_6_2', 'A_6_2_1', 'D_6_2_1_0')),  # A_6_2_1
                           'vars': {'R_1_0': 1.4223004218216138, 'R_2_1': 1.4557553347542735,
                                    'A_2_1_0': 105.53988198924208, 'R_3_0': 1.093821659465454,
                                    'A_3_0_1': 110.03248558165065, 'D_3_0_1_2': 60.84271231265853,
                                    'R_4_0': 1.0938084178037755, 'A_4_0_1': 110.03299489037433,
                                    'D_4_0_1_3': 238.4134975536592, 'R_5_0': 1.0928700313199922,
                                    'A_5_0_1': 108.55511996651099, 'D_5_0_1_4': 240.7911024479184,
                                    'R_6_2': 0.9741406737537205, 'A_6_2_1': 96.30924284405943,
                                    'D_6_2_1_0': 242.22744613021646},
                           'map': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}}
        expected_zmat_2 = {'symbols': ('H', 'O', 'C', 'C', 'H', 'H', 'H', 'H', 'H'),
                           'coords': ((None, None, None), ('R_1_0', None, None), ('R_2_1', 'A_2_1_0', None),  # A_2_1_0
                                      ('R_3_2', 'A_3_2_1', 'D_3_2_1_0'), ('R_4_3', 'A_4_3_2', 'D_4_3_2_1'),
                                      ('R_5_3', 'A_5_3_2', 'D_5_3_2_4'), ('R_6_3', 'A_6_3_2', 'D_6_3_2_5'),
                                      ('R_7_2', 'A_7_2_3', 'D_7_2_3_6'), ('R_8_2', 'A_8_2_3', 'D_8_2_3_7')),
                           'vars': {'R_1_0': 0.9723850776119742, 'R_2_1': 1.4197372463410514,
                                    'A_2_1_0': 107.06334992092434, 'R_3_2': 1.5137276325416074,
                                    'A_3_2_1': 109.01303538097567, 'D_3_2_1_0': 179.99951584936258,
                                    'R_4_3': 1.0950337097344136, 'A_4_3_2': 110.63258593497066,
                                    'D_4_3_2_1': 59.12026293645304, 'R_5_3': 1.0935594120185885,
                                    'A_5_3_2': 110.92258531860486, 'D_5_3_2_4': 120.87939658650379,
                                    'R_6_3': 1.095033981893329, 'A_6_3_2': 110.63254968567193,
                                    'D_6_3_2_5': 120.8793955156551, 'R_7_2': 1.0941026391623285,
                                    'A_7_2_3': 110.5440977183771, 'D_7_2_3_6': 181.34310124526917,
                                    'R_8_2': 1.0941023667717409, 'A_8_2_3': 110.54410081124645,
                                    'D_8_2_3_7': 239.07189759901027},
                           'map': {0: 8, 1: 2, 2: 1, 3: 0, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7}}
        self.assertTrue(_compare_zmats(zmat_1, expected_zmat_1, r_tol=0.01, a_tol=0.01, d_tol=0.01))
        self.assertTrue(_compare_zmats(zmat_2, expected_zmat_2, r_tol=0.01, a_tol=0.01, d_tol=0.01))

        zmat_1, zmat_2 = generate_the_two_constrained_zmats(xyz_1=self.ch3ch2oh.get_xyz(),
                                                            xyz_2=self.ch3ooh.get_xyz(),
                                                            mol_1=self.ch3ch2oh.mol,
                                                            mol_2=self.ch3ooh.mol,
                                                            h1=7,
                                                            a=4,
                                                            c=2,
                                                            h2=3,
                                                            b=0,
                                                            d=1,
                                                            )
        expected_zmat_1 = {'symbols': ('C', 'C', 'O', 'H', 'H', 'H', 'H', 'H', 'H'),
                           'coords': ((None, None, None), ('R_1_0', None, None), ('R_2_1', 'A_2_1_0', None),
                                      ('R_3_0', 'A_3_0_1', 'D_3_0_1_2'), ('R_4_0', 'A_4_0_1', 'D_4_0_1_3'),
                                      ('R_5_0', 'A_5_0_1', 'D_5_0_1_4'), ('R_6_1', 'A_6_1_0', 'D_6_1_0_5'),
                                      ('R_7_4', 'A_7_4_2', 'D_7_4_2_1'), ('R_8_2', 'A_8_2_1', 'D_8_2_1_0')),  # A_7_4_2
                           'vars': {'R_1_0': 1.5137276325416074, 'R_2_1': 1.4197372463410514,
                                    'A_2_1_0': 109.01303538097567, 'R_3_0': 1.0950337097344136,
                                    'A_3_0_1': 110.63258593497066, 'D_3_0_1_2': 59.12026293645304,
                                    'R_4_0': 1.0935594120185885, 'A_4_0_1': 110.92258531860486,
                                    'D_4_0_1_3': 120.87939658650379, 'R_5_0': 1.095033981893329,
                                    'A_5_0_1': 110.63254968567193, 'D_5_0_1_4': 120.8793955156551,
                                    'R_6_1': 1.0941026391623285, 'A_6_1_0': 110.5440977183771,
                                    'D_6_1_0_5': 181.34310124526917, 'R_7_4': 2.509397582146114,
                                    'A_7_4_2': 37.75756623354705, 'D_7_4_2_1': 35.45850610925192,
                                    'R_8_2': 0.9723850776119742, 'A_8_2_1': 107.06334992092434,
                                    'D_8_2_1_0': 179.99951584936258},
                           'map': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8}}
        expected_zmat_2 = {'symbols': ('H', 'C', 'O', 'O', 'H', 'H', 'H'),
                           'coords': ((None, None, None), ('R_1_0', None, None), ('R_2_1', 'A_2_1_0', None),  # A_2_1_0
                                      ('R_3_2', 'A_3_2_1', 'D_3_2_1_0'), ('R_4_1', 'A_4_1_2', 'D_4_1_2_3'),
                                      ('R_5_1', 'A_5_1_2', 'D_5_1_2_4'), ('R_6_3', 'A_6_3_2', 'D_6_3_2_1')),
                           'vars': {'R_1_0': 1.093821659465454, 'R_2_1': 1.4223004218216138,
                                    'A_2_1_0': 110.03248558165065, 'R_3_2': 1.4557553347542735,
                                    'A_3_2_1': 105.53988198924208, 'D_3_2_1_0': 60.84271231265853,
                                    'R_4_1': 1.0938084178037755, 'A_4_1_2': 110.03299489037433,
                                    'D_4_1_2_3': 299.2562112207745, 'R_5_1': 1.0928700313199922,
                                    'A_5_1_2': 108.55511996651099, 'D_5_1_2_4': 240.7911024479184,
                                    'R_6_3': 0.9741406737537205, 'A_6_3_2': 96.30924284405943,
                                    'D_6_3_2_1': 242.22744613021646},
                           'map': {0: 3, 1: 0, 2: 1, 3: 2, 4: 4, 5: 5, 6: 6}}
        self.assertTrue(_compare_zmats(zmat_1, expected_zmat_1, r_tol=0.01, a_tol=0.01, d_tol=0.01))
        self.assertTrue(_compare_zmats(zmat_2, expected_zmat_2, r_tol=0.01, a_tol=0.01, d_tol=0.01))

    def test_stretch_zmat_bond(self):
        """Test the stretch_zmat_bond function."""
        zmat2_copy = copy.deepcopy(self.zmat_2)
        stretch_zmat_bond(zmat=zmat2_copy, indices=(1, 0), stretch=1.5)
        self.assertEqual(zmat2_copy['vars']['R_2_1'], self.zmat_2['vars']['R_2_1'] * 1.5)

    def test_determine_glue_params(self):
        """Test the determine_glue_params() function."""
        zmat_0 = {'symbols': ('O', 'H'), 'coords': ((None, None, None), ('R_1_0', None, None)),
                  'vars': {'R_1_0': 1.1644188088546794}, 'map': {0: 0, 1: 1}}
        param_a2, param_d2, param_d3 = determine_glue_params(zmat=zmat_0,
                                                             add_dummy=False,
                                                             h1=0,
                                                             a=1,
                                                             c=None,
                                                             d=None,
                                                             )
        self.assertEqual(param_a2, 'A_2_0_1')  # B-H-A
        self.assertEqual(param_d2, None)  # B-H-A-C
        self.assertEqual(param_d3, None)  # D-B-H-A

        # None linear
        zmat_1 = {'symbols': ('H', 'N', 'C', 'H', 'H', 'H', 'H'),
                  'coords': ((None, None, None), ('R_1_0', None, None), ('R_2_1', 'A_2_1_0', None),
                             ('R_3_2', 'A_3_2_1', 'D_3_2_1_0'), ('R_4_2', 'A_4_2_1', 'D_4_2_1_3'),
                             ('R_5_2', 'A_5_2_1', 'D_5_2_1_4'), ('R_6_1', 'A_6_1_2', 'D_6_1_2_5')),
                  'vars': {'R_1_0': 1.019169359544101, 'R_2_1': 1.451965854148702, 'A_2_1_0': 109.41187480598941,
                           'R_3_2': 1.0935188594180785, 'A_3_2_1': 110.20496026605478,
                           'D_3_2_1_0': 60.83821034525936, 'R_4_2': 1.0935188594180785,
                           'A_4_2_1': 110.20495933616156, 'D_4_2_1_3': 240.4644263689792,
                           'R_5_2': 1.0936965384360282, 'A_5_2_1': 110.59878027260544,
                           'D_5_2_1_4': 239.76779188408136, 'R_6_1': 1.0191693010605467,
                           'A_6_1_2': 109.41187816450345, 'D_6_1_2_5': 65.17113681053117},
                  'map': {0: 5, 1: 1, 2: 0, 3: 2, 4: 3, 5: 4, 6: 6}}
        param_a2, param_d2, param_d3 = determine_glue_params(zmat=zmat_1,
                                                             add_dummy=False,
                                                             h1=5,
                                                             a=1,
                                                             c=0,
                                                             d=1,
                                                             )
        self.assertEqual(param_a2, 'A_7_0_1')  # B-H-A
        self.assertEqual(param_d2, 'D_7_0_1_2')  # B-H-A-C
        self.assertEqual(param_d3, 'D_8_7_0_1')  # D-B-H-A

        # Linear
        param_a2, param_d2, param_d3 = determine_glue_params(zmat=zmat_1,
                                                             add_dummy=True,
                                                             h1=5,
                                                             a=1,
                                                             c=0,
                                                             d=1,
                                                             )
        self.assertEqual(param_a2, 'A_8_0_1')  # B-H-A
        self.assertEqual(param_d2, 'D_8_0_7_1')  # B-H-X-A
        self.assertEqual(param_d3, 'D_9_8_1_2')  # D-B-A-C/X

        zmat_1_copy = copy.deepcopy(self.zmat_1)
        param_a2, param_d2, param_d3 = determine_glue_params(zmat=zmat_1_copy,
                                                             add_dummy=True,
                                                             h1=9,
                                                             a=3,
                                                             c=2,
                                                             d=0,
                                                             )
        self.assertEqual(param_a2, 'A_11_9_3')
        self.assertEqual(param_d2, 'D_11_9_10_3')
        self.assertEqual(param_d3, 'D_12_11_3_2')
        self.assertEqual(zmat_1_copy['symbols'][-1], 'X')
        self.assertEqual(zmat_1_copy['coords'][-1], ('RX_10_9', 'AX_10_9_3', 'DX_10_9_3_2'))
        self.assertEqual(zmat_1_copy['vars']['RX_10_9'], 1.0)
        self.assertEqual(zmat_1_copy['vars']['AX_10_9_3'], 90)
        self.assertEqual(zmat_1_copy['vars']['DX_10_9_3_2'], 0)
        self.assertEqual(zmat_1_copy['map'][10], 'X10')
        expected_xyz_1 = {'symbols': ('C', 'C', 'O', 'O', 'H', 'H', 'H', 'H', 'H', 'H', 'X'),
                          'isotopes': (12, 12, 16, 16, 1, 1, 1, 1, 1, 1, None),
                          'coords': ((0.01398594476849108, -0.6889055930112338, -1.784376061009113),
                                     (0.01398594476849108, -0.6889055930112338, -0.26962806588789334),
                                     (0.01398594476849108, 0.6628859053975544, 0.18618985833619472),
                                     (0.013795619205083086, 0.5853276282667381, 1.6400480721581545),
                                     (0.8936110241109786, -0.1630214384486075, -2.1700905561834642),
                                     (0.013993742824539971, -1.7104245574061123, -2.174747562010025),
                                     (-0.8656459383075578, -0.16302845349065453, -2.170051996558082),
                                     (0.9066278741353695, -1.1982381538878737, 0.10867044129945613),
                                     (-0.8786689727859842, -1.1983013118398729, 0.10854795923608584),
                                     (-0.8438878073681937, 1.028321080587749, 1.77062574436878),
                                     (-0.9409710712603131, 1.1321270125843694, 0.7807776912038682))}
        self.assertTrue(almost_equal_coords(zmat_to_xyz(zmat_1_copy, keep_dummy=True), expected_xyz_1))

        zmat_2 = {'symbols': ('C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'),
                  'coords': ((None, None, None), ('R_1_0', None, None), ('R_2_1', 'A_2_1_0', None),
                             ('R_3_1', 'A_3_1_2', 'D_3_1_2_0'), ('R_4_1', 'A_4_1_2', 'D_4_1_2_3'),
                             ('R_5_2', 'A_5_2_1', 'D_5_2_1_4'), ('R_6_0', 'A_6_0_1', 'D_6_0_1_2'),
                             ('R_7_2', 'A_7_2_1', 'D_7_2_1_6'), ('R_8_2', 'A_8_2_1', 'D_8_2_1_7'),
                             ('R_9_0', 'A_9_0_1', 'D_9_0_1_2'), ('R_10_0', 'A_10_0_1', 'D_10_0_1_2')),
                  'vars': {'R_1_0': 1.538597152766632, 'R_2_1': 1.538597152766632, 'A_2_1_0': 112.21956716174662,
                           'R_3_1': 1.1067875520527994, 'A_3_1_2': 109.48164208764113, 'D_3_1_2_0': 121.77882792718154,
                           'R_4_1': 1.1067875520527994, 'A_4_1_2': 109.48164208764113, 'D_4_1_2_3': 116.44234414563692,
                           'R_5_2': 1.325576275438579, 'A_5_2_1': 111.50202139070583, 'D_5_2_1_4': 58.22117207281846,
                           'R_6_0': 1.1046468961988158, 'A_6_0_1': 111.50202139070583, 'D_6_0_1_2': 180.0,
                           'R_7_2': 1.1059615583516615, 'A_7_2_1': 110.8283376252455, 'D_7_2_1_6': 300.2088700889345,
                           'R_8_2': 1.1059615583516615, 'A_8_2_1': 110.8283376252455, 'D_8_2_1_7': 119.58225745206313,
                           'R_9_0': 1.1059615583516615, 'A_9_0_1': 110.8283376252455, 'D_9_0_1_2': 300.2088700889345,
                           'R_10_0': 1.1059615583516615, 'A_10_0_1': 110.8283376252455, 'D_10_0_1_2': 59.79112991106552},
                  'map': {0: 2, 1: 0, 2: 1, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10}}
        param_a2, param_d2, param_d3 = determine_glue_params(zmat=zmat_2,
                                                             add_dummy=False,
                                                             h1=1,
                                                             a=5,
                                                             c=0,
                                                             d=1,
                                                             )
        self.assertEqual(param_a2, 'A_11_2_5')
        self.assertEqual(param_d2, 'D_11_2_5_1')
        self.assertEqual(param_d3, 'D_12_11_2_5')

    def test_get_modified_params_from_zmat_2(self):
        """Test the get_modified_params_from_zmat_2() function."""
        zmat_1 = {'symbols': ('C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'X'),
                  'coords': ((None, None, None), ('R_1_0', None, None), ('R_2_0', 'A_2_0_1', None),
                             ('R_3_0', 'A_3_0_1', 'D_3_0_1_2'), ('R_4_1', 'A_4_1_0', 'D_4_1_0_3'),
                             ('R_5_1', 'A_5_1_0', 'D_5_1_0_4'), ('R_6_1', 'A_6_1_0', 'D_6_1_0_5'),
                             ('R_7_0', 'A_7_0_1', 'D_7_0_1_6'), ('RX_8_7', 'AX_8_7_0', 'DX_8_7_0_1')),
                  'vars': {'R_1_0': 1.5120487296562577, 'R_2_0': 1.0940775789443724, 'A_2_0_1': 110.56801921096591,
                           'R_3_0': 1.0940817193677925, 'A_3_0_1': 110.56754686774481, 'D_3_0_1_2': 119.99910067703652,
                           'R_4_1': 1.0940725668318991, 'A_4_1_0': 110.56890700195424, 'D_4_1_0_3': 59.99971758419434,
                           'R_5_1': 1.0940840619688397, 'A_5_1_0': 110.56790845138725, 'D_5_1_0_4': 239.99905123159166,
                           'R_6_1': 1.0940817193677925, 'A_6_1_0': 110.56754686774481, 'D_6_1_0_5': 240.00122783407815,
                           'R_7_0': 1.3128870801982788, 'A_7_0_1': 110.56890700195424, 'D_7_0_1_6': 300.00028241580566,
                           'RX_8_7': 1.0, 'AX_8_7_0': 90.0, 'DX_8_7_0_1': 0},
                  'map': {0: 0, 1: 1, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 2, 8: 'X8'}}
        zmat_2 = {'symbols': ('H', 'O', 'O', 'C', 'C', 'H', 'H', 'H', 'H', 'H'),
                  'coords': ((None, None, None), ('R_1_0', None, None), ('R_2_1', 'A_2_1_0', None),
                             ('R_3_2', 'A_3_2_1', 'D_3_2_1_0'), ('R_4_3', 'A_4_3_2', 'D_4_3_2_1'),
                             ('R_5_4', 'A_5_4_3', 'D_5_4_3_2'), ('R_6_4', 'A_6_4_3', 'D_6_4_3_5'),
                             ('R_7_4', 'A_7_4_3', 'D_7_4_3_6'), ('R_8_3', 'A_8_3_4', 'D_8_3_4_7'),
                             ('R_9_3', 'A_9_3_4', 'D_9_3_4_8')),
                  'vars': {'R_1_0': 1.1689469645782498, 'R_2_1': 1.4559254886404387, 'A_2_1_0': 96.30065819269021,
                           'R_3_2': 1.4265728986680748, 'A_3_2_1': 105.58023544826183, 'D_3_2_1_0': 242.3527063196313,
                           'R_4_3': 1.5147479951212197, 'A_4_3_2': 108.63387152978416, 'D_4_3_2_1': 179.9922243050821,
                           'R_5_4': 1.0950205915944824, 'A_5_4_3': 110.62463321031589, 'D_5_4_3_2': 59.1268942923763,
                           'R_6_4': 1.093567969297245, 'A_6_4_3': 110.91425998596507, 'D_6_4_3_5': 120.87266977773987,
                           'R_7_4': 1.0950091062890002, 'A_7_4_3': 110.62270362433773, 'D_7_4_3_6': 120.87301274044218,
                           'R_8_3': 1.0951433842986755, 'A_8_3_4': 110.20822115119915, 'D_8_3_4_7': 181.16392677464265,
                           'R_9_3': 1.0951410439636102, 'A_9_3_4': 110.20143800025897, 'D_9_3_4_8': 239.4199964284852},
                  'map': {0: 9, 1: 3, 2: 2, 3: 1, 4: 0, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8}}
        new_symbols, new_coords, new_vars, new_map = \
            get_modified_params_from_zmat_2(zmat_1=zmat_1,
                                            zmat_2=zmat_2,
                                            reactant_2=ARCSpecies(label='CCOOj', smiles='CCO[O]', xyz=self.ccooj_xyz),
                                            add_dummy=True,
                                            glue_params=('A_9_7_8', 'D_9_7_8_0', 'D_10_9_7_0'),
                                            h1=2,
                                            a=0,
                                            c=1,
                                            a2=150,
                                            d2=0,
                                            d3=120,
                                            )
        expected_new_symbols = ('C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'X', 'O', 'O', 'C', 'C', 'H', 'H', 'H', 'H', 'H')
        expected_new_coords = (
            (None, None, None), ('R_1_0', None, None), ('R_2_0', 'A_2_0_1', None), ('R_3_0', 'A_3_0_1', 'D_3_0_1_2'),
            ('R_4_1', 'A_4_1_0', 'D_4_1_0_3'), ('R_5_1', 'A_5_1_0', 'D_5_1_0_4'), ('R_6_1', 'A_6_1_0', 'D_6_1_0_5'),
            ('R_7_0', 'A_7_0_1', 'D_7_0_1_6'), ('RX_8_7', 'AX_8_7_0', 'DX_8_7_0_1'), ('R_9_2', 'A_9_7_8', 'D_9_7_8_0'),
            ('R_10_9', 'A_10_9_0', 'D_10_9_7_0'), ('R_11_10', 'A_11_10_9', 'D_11_10_9_8'),
            ('R_12_11', 'A_12_11_10', 'D_12_11_10_9'), ('R_13_12', 'A_13_12_11', 'D_13_12_11_10'),
            ('R_14_12', 'A_14_12_11', 'D_14_12_11_13'), ('R_15_12', 'A_15_12_11', 'D_15_12_11_14'),
            ('R_16_11', 'A_16_11_12', 'D_16_11_12_15'), ('R_17_11', 'A_17_11_12', 'D_17_11_12_16'))
        expected_new_vars = {
            'R_1_0': 1.5120487296562577, 'R_2_0': 1.0940775789443724, 'A_2_0_1': 110.56801921096591,
            'R_3_0': 1.0940817193677925, 'A_3_0_1': 110.56754686774481, 'D_3_0_1_2': 119.99910067703652,
            'R_4_1': 1.0940725668318991, 'A_4_1_0': 110.56890700195424, 'D_4_1_0_3': 59.99971758419434,
            'R_5_1': 1.0940840619688397, 'A_5_1_0': 110.56790845138725, 'D_5_1_0_4': 239.99905123159166,
            'R_6_1': 1.0940817193677925, 'A_6_1_0': 110.56754686774481, 'D_6_1_0_5': 240.00122783407815,
            'R_7_0': 1.3128870801982788, 'A_7_0_1': 110.56890700195424, 'D_7_0_1_6': 300.00028241580566, 'RX_8_7': 1.0,
            'AX_8_7_0': 90.0, 'DX_8_7_0_1': 0, 'R_9_2': 1.1689469645782498, 'A_9_7_8': 240, 'D_9_7_8_0': 0,
            'R_10_9': 1.4559254886404387, 'A_10_9_0': 96.30065819269021, 'D_10_9_7_0': 120,
            'R_11_10': 1.4265728986680748, 'A_11_10_9': 105.58023544826183, 'D_11_10_9_8': 242.3527063196313,
            'R_12_11': 1.5147479951212197, 'A_12_11_10': 108.63387152978416, 'D_12_11_10_9': 179.9922243050821,
            'R_13_12': 1.0950205915944824, 'A_13_12_11': 110.62463321031589, 'D_13_12_11_10': 59.1268942923763,
            'R_14_12': 1.093567969297245, 'A_14_12_11': 110.91425998596507, 'D_14_12_11_13': 120.87266977773987,
            'R_15_12': 1.0950091062890002, 'A_15_12_11': 110.62270362433773, 'D_15_12_11_14': 120.87301274044218,
            'R_16_11': 1.0951433842986755, 'A_16_11_12': 110.20822115119915, 'D_16_11_12_15': 181.16392677464265,
            'R_17_11': 1.0951410439636102, 'A_17_11_12': 110.20143800025897, 'D_17_11_12_16': 239.4199964284852}
        expected_new_map = {0: 0, 1: 1, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 2, 8: 'X8', 9: 12,
                            10: 11, 11: 10, 12: 9, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17}
        self.assertTrue(_compare_zmats({'symbols': new_symbols, 'coords': new_coords, 'vars': new_vars, 'map': new_map},
                                       {'symbols': expected_new_symbols, 'coords': expected_new_coords,
                                        'vars': expected_new_vars, 'map': expected_new_map},
                                       r_tol=0.01, a_tol=0.01, d_tol=0.01))

    def test_find_distant_neighbor(self):
        """Test the find_distant_neighbor() function."""
        xyz_1 = {'symbols': ('S', 'O', 'O', 'N', 'C', 'H', 'H', 'H', 'H', 'H'),
                 'isotopes': (32, 16, 16, 14, 12, 1, 1, 1, 1, 1),
                 'coords': ((-0.06618943, -0.12360663, -0.07631983),
                            (-0.79539707, 0.86755487, 1.02675668),
                            (-0.68919931, 0.25421823, -1.34830853),
                            (0.01546439, -1.54297548, 0.44580391),
                            (1.59721519, 0.47861334, 0.00711),
                            (1.94428095, 0.40772394, 1.03719428),
                            (2.20318015, -0.14715186, -0.64755729),
                            (1.59252246, 1.5117895, -0.33908352),
                            (-0.8785689, -2.02453514, 0.38494433),
                            (-1.34135876, 1.49608206, 0.53295071))}
        mol_1 = ARCSpecies(label='CS(=O)(O)[NH]', xyz=xyz_1).mol
        self.assertEqual(find_distant_neighbor(rmg_mol=mol_1, start=8), 0)
        self.assertEqual(find_distant_neighbor(rmg_mol=mol_1, start=9), 0)
        self.assertEqual(find_distant_neighbor(rmg_mol=mol_1, start=5), 0)
        self.assertIn(find_distant_neighbor(rmg_mol=mol_1, start=2), [1, 3, 4])
        self.assertIn(find_distant_neighbor(rmg_mol=mol_1, start=4), [1, 2, 3])
        self.assertIn(find_distant_neighbor(rmg_mol=mol_1, start=3), [1, 2, 4])
        self.assertIn(find_distant_neighbor(rmg_mol=mol_1, start=0), [5, 6, 7, 8, 9])

        xyz_2 = {'symbols': ('S', 'H', 'O', 'N', 'C', 'H', 'H', 'H', 'H'),
                 'isotopes': (32, 1, 16, 14, 12, 1, 1, 1, 1),
                 'coords': ((-0.06618943, -0.12360663, -0.07631983),
                            (-0.79539707, 0.86755487, 1.02675668),
                            (-0.68919931, 0.25421823, -1.34830853),
                            (0.01546439, -1.54297548, 0.44580391),
                            (1.59721519, 0.47861334, 0.00711),
                            (1.94428095, 0.40772394, 1.03719428),
                            (2.20318015, -0.14715186, -0.64755729),
                            (1.59252246, 1.5117895, -0.33908352),
                            (-0.8785689, -2.02453514, 0.38494433))}
        mol_2 = ARCSpecies(label='C[SH](=O)[NH]', xyz=xyz_2).mol
        self.assertEqual(find_distant_neighbor(rmg_mol=mol_2, start=8), 0)
        self.assertEqual(find_distant_neighbor(rmg_mol=mol_2, start=5), 0)
        self.assertIn(find_distant_neighbor(rmg_mol=mol_2, start=2), [3, 4])
        self.assertIn(find_distant_neighbor(rmg_mol=mol_2, start=4), [2, 3])
        self.assertIn(find_distant_neighbor(rmg_mol=mol_2, start=3), [2, 4])
        self.assertIn(find_distant_neighbor(rmg_mol=mol_2, start=0), [5, 6, 7, 8])

        xyz_3 = """C                 -2.27234259   -0.78101274   -0.00989219
                   H                 -1.94047502   -1.78971513   -0.14135849
                   H                 -1.87026865   -0.16873290   -0.78986011
                   H                 -3.34123881   -0.74952096   -0.04689398
                   C                 -1.79025824   -0.25578524    1.35514655
                   H                 -2.12212581    0.75291715    1.48661285
                   H                 -2.19233218   -0.86806508    2.13511447
                   N                 -0.32177464   -0.29904964    1.40598078
                   H                  0.05399540    0.27317450    0.67703880
                   H                 -0.00873286    0.04200717    2.29236958"""
        mol_3 = ARCSpecies(label='EA', smiles='NCC', xyz=xyz_3).mol
        self.assertEqual(find_distant_neighbor(rmg_mol=mol_3, start=9), 4)
        self.assertEqual(find_distant_neighbor(rmg_mol=mol_3, start=8), 4)
        self.assertIn(find_distant_neighbor(rmg_mol=mol_3, start=5), [0, 7])
        self.assertIn(find_distant_neighbor(rmg_mol=mol_3, start=6), [0, 7])
        self.assertEqual(find_distant_neighbor(rmg_mol=mol_3, start=1), 4)
        self.assertEqual(find_distant_neighbor(rmg_mol=mol_3, start=2), 4)
        self.assertEqual(find_distant_neighbor(rmg_mol=mol_3, start=2), 4)

    @classmethod
    def tearDownClass(cls):
        """
        A function that is run ONCE after all unit tests in this class.
        Delete all project directories created during these unit tests.
        """
        shutil.rmtree(os.path.join(ARC_PATH, 'arc', 'testing', 'heuristics'), ignore_errors=True)
        shutil.rmtree(os.path.join(ARC_PATH, 'arc', 'testing', 'heuristics_1'), ignore_errors=True)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
