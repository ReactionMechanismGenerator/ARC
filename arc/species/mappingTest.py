#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.species.mapping module
"""

import unittest

from qcelemental.models.molecule import Molecule as QCMolecule

from rmgpy.reaction import Reaction
from rmgpy.species import Species

import arc.species.mapping as mapping
from arc.rmgdb import determine_family, load_families_only, make_rmg_database_object
from arc.species.species import ARCSpecies
from arc.reaction import ARCReaction


class TestMapping(unittest.TestCase):
    """
    Contains unit tests for the mapping module.
    """

    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None
        cls.rmgdb = make_rmg_database_object()
        load_families_only(cls.rmgdb)
        cls.arc_reaction_1 = ARCReaction(label='CH4 + OH <=> CH3 + H2O',
                                         r_species=[ARCSpecies(label='CH4', smiles='C',
                                                               xyz="""C      -0.00000000    0.00000000    0.00000000
                                                                      H      -0.65055201   -0.77428020   -0.41251879
                                                                      H      -0.34927558    0.98159583   -0.32768232
                                                                      H      -0.02233792   -0.04887375    1.09087665
                                                                      H       1.02216551   -0.15844188   -0.35067554"""),
                                                    ARCSpecies(label='OH', smiles='[OH]',
                                                               xyz="""O       0.48890387    0.00000000    0.00000000
                                                                      H      -0.48890387    0.00000000    0.00000000""")],
                                         p_species=[ARCSpecies(label='CH3', smiles='[CH3]',
                                                               xyz="""C       0.00000000    0.00000001   -0.00000000
                                                                      H       1.06690511   -0.17519582    0.05416493
                                                                      H      -0.68531716   -0.83753536   -0.02808565
                                                                      H      -0.38158795    1.01273118   -0.02607927"""),
                                                    ARCSpecies(label='H2O', smiles='O',
                                                               xyz="""O      -0.00032832    0.39781490    0.00000000
                                                                      H      -0.76330345   -0.19953755    0.00000000
                                                                      H       0.76363177   -0.19827735    0.00000000""")])
        cls.arc_reaction_2 = ARCReaction(label='C3H8 + NH2 <=> nC3H7 + NH3',
                                         r_species=[ARCSpecies(label='C3H8', smiles='CCC',
                                                               xyz="""C      -1.26511392    0.18518050   -0.19976825
                                                                      C       0.02461113   -0.61201635   -0.29700643
                                                                      C       0.09902018   -1.69054887    0.77051392
                                                                      H      -1.34710559    0.68170095    0.77242199
                                                                      H      -2.12941774   -0.47587010   -0.31761654
                                                                      H      -1.31335400    0.95021638   -0.98130653
                                                                      H       0.88022594    0.06430231   -0.19248282
                                                                      H       0.09389171   -1.07422931   -1.28794952
                                                                      H      -0.73049348   -2.39807515    0.67191015
                                                                      H       1.03755706   -2.24948851    0.69879172
                                                                      H       0.04615234   -1.24181601    1.76737952"""),
                                                    ARCSpecies(label='NH2', smiles='[NH2]',
                                                               xyz="""N       0.00022972    0.40059496    0.00000000
                                                                      H      -0.83174214   -0.19982058    0.00000000
                                                                      H       0.83151242   -0.20077438    0.00000000""")],
                                         p_species=[ARCSpecies(label='nC3H7', smiles='[CH2]CC',
                                                               xyz="""C       1.37804355    0.27791700   -0.19511840
                                                                      C       0.17557158   -0.34036318    0.43265003
                                                                      C      -0.83187173    0.70418067    0.88324591
                                                                      H       2.32472110   -0.25029805   -0.17789388
                                                                      H       1.28332450    1.14667614   -0.83695597
                                                                      H      -0.29365298   -1.02042821   -0.28596734
                                                                      H       0.48922284   -0.93756983    1.29560539
                                                                      H      -1.19281782    1.29832390    0.03681748
                                                                      H      -1.69636720    0.21982441    1.34850246
                                                                      H      -0.39178710    1.38838724    1.61666119"""),
                                                    ARCSpecies(label='NH3', smiles='N',
                                                               xyz="""N       0.00064924   -0.00099698    0.29559292
                                                                      H      -0.41786606    0.84210396   -0.09477452
                                                                      H      -0.52039228   -0.78225292   -0.10002797
                                                                      H       0.93760911   -0.05885406   -0.10079043""")])
        cls.arc_reaction_4 = ARCReaction(label='CH2CH2NH2 <=> CH3CH2NH',
                                         r_species=[ARCSpecies(label='CH2CH2NH2', smiles='[CH2]CN',
                                                               xyz="""C      -1.24450121    0.17451352    0.00786829
                                                                      C       0.09860657   -0.41192142   -0.18691029
                                                                      N       0.39631461   -0.45573259   -1.60474376
                                                                      H      -2.04227601   -0.03772349   -0.69530380
                                                                      H      -1.50683666    0.61023628    0.96427405
                                                                      H       0.11920004   -1.42399272    0.22817674
                                                                      H       0.85047586    0.18708096    0.33609395
                                                                      H       0.46736985    0.49732569   -1.95821046
                                                                      H       1.31599714   -0.87204344   -1.73848831""")],
                                         p_species=[ARCSpecies(label='CH3CH2NH', smiles='CC[NH]',
                                                               xyz="""C      -1.03259818   -0.08774861    0.01991495
                                                                      C       0.48269985   -0.19939835    0.09039740
                                                                      N       0.94816502   -1.32096642   -0.71111614
                                                                      H      -1.51589318   -0.99559128    0.39773163
                                                                      H      -1.37921961    0.75694189    0.62396315
                                                                      H      -1.37189872    0.07009088   -1.00987126
                                                                      H       0.78091492   -0.31605120    1.13875203
                                                                      H       0.92382278    0.74158978   -0.25822764
                                                                      H       1.97108857   -1.36649904   -0.64094836""")])
        cls.rmg_reaction_1 = Reaction(reactants=[Species(smiles='C'), Species(smiles='[OH]')],
                                      products=[Species(smiles='[CH3]'), Species(smiles='O')])
        cls.rmg_reaction_2 = Reaction(reactants=[Species(smiles='[OH]'), Species(smiles='C')],
                                      products=[Species(smiles='[CH3]'), Species(smiles='O')])
        cls.arc_reaction_3 = ARCReaction(label='CH3 + CH3 <=> C2H6',
                                         r_species=[ARCSpecies(label='CH3', smiles='[CH3]')],
                                         p_species=[ARCSpecies(label='C2H6', smiles='CC')])
        cls.rmg_reaction_3 = Reaction(reactants=[Species(smiles='[CH3]'), Species(smiles='[CH3]')],
                                      products=[Species(smiles='CC')])

        cls.r_xyz_2a = """C                  0.50180491   -0.93942231   -0.57086745
        C                  0.01278145    0.13148427    0.42191407
        C                 -0.86874485    1.29377369   -0.07163907
        H                  0.28549447    0.06799101    1.45462711
        H                  1.44553946   -1.32386345   -0.24456986
        H                  0.61096295   -0.50262210   -1.54153222
        H                 -0.24653265    2.11136864   -0.37045418
        H                 -0.21131163   -1.73585284   -0.61629002
        H                 -1.51770930    1.60958621    0.71830245
        H                 -1.45448167    0.96793094   -0.90568876"""
        cls.r_xyz_2b = """C                  0.50180491   -0.93942231   -0.57086745
        C                  0.01278145    0.13148427    0.42191407
        H                  0.28549447    0.06799101    1.45462711
        H                  1.44553946   -1.32386345   -0.24456986
        H                  0.61096295   -0.50262210   -1.54153222
        H                 -0.24653265    2.11136864   -0.37045418
        C                 -0.86874485    1.29377369   -0.07163907
        H                 -0.21131163   -1.73585284   -0.61629002
        H                 -1.51770930    1.60958621    0.71830245
        H                 -1.45448167    0.96793094   -0.90568876"""
        cls.p_xyz_2 = """C                  0.48818717   -0.94549701   -0.55196729
        C                  0.35993708    0.29146456    0.35637075
        C                 -0.91834764    1.06777042   -0.01096751
        H                  0.30640232   -0.02058840    1.37845537
        H                  1.37634603   -1.48487836   -0.29673876
        H                  0.54172192   -0.63344406   -1.57405191
        H                  1.21252186    0.92358349    0.22063264
        H                 -0.36439762   -1.57761595   -0.41622918
        H                 -1.43807526    1.62776079    0.73816131
        H                 -1.28677889    1.04716138   -1.01532486"""
        cls.ts_xyz_2 = """C       0.52123900   -0.93806900   -0.55301700
        C       0.15387500    0.18173100    0.37122900
        C      -0.89554000    1.16840700   -0.01362800
        H       0.33997700    0.06424800    1.44287100
        H       1.49602200   -1.37860200   -0.29763200
        H       0.57221700   -0.59290500   -1.59850500
        H       0.39006800    1.39857900   -0.01389600
        H      -0.23302200   -1.74751100   -0.52205400
        H      -1.43670700    1.71248300    0.76258900
        H      -1.32791000    1.11410600   -1.01554900"""  # C[CH]C <=> [CH2]CC
        cls.ts_spc_2 = ARCSpecies(label='TS', is_ts=True, xyz=cls.ts_xyz_2)
        cls.ts_spc_2.mol_from_xyz()
        cls.reactant_2a = ARCSpecies(label='C[CH]C', smiles='C[CH]C', xyz=cls.r_xyz_2a)
        cls.reactant_2b = ARCSpecies(label='C[CH]C', smiles='C[CH]C',
                                     xyz=cls.r_xyz_2b)  # same as 2a, only one C atom shifted place in the reactant xyz
        cls.product_2 = ARCSpecies(label='[CH2]CC', smiles='[CH2]CC', xyz=cls.p_xyz_2)
        cls.rxn_2a = ARCReaction(r_species=[cls.reactant_2a], p_species=[cls.product_2])
        cls.rxn_2a.ts_species = cls.ts_spc_2
        cls.rxn_2b = ARCReaction(r_species=[cls.reactant_2b], p_species=[cls.product_2])
        cls.rxn_2b.ts_species = cls.ts_spc_2

    def test_map_h_abstraction(self):
        """Test the map_h_abstraction() function."""

        # CH4 + OH <=> CH3 + H2O
        ch4_xyz = """C      -0.00000000    0.00000000    0.00000000
                     H      -0.65055201   -0.77428020   -0.41251879
                     H      -0.34927558    0.98159583   -0.32768232
                     H      -0.02233792   -0.04887375    1.09087665
                     H       1.02216551   -0.15844188   -0.35067554"""
        oh_xyz = """O       0.48890387    0.00000000    0.00000000
                    H      -0.48890387    0.00000000    0.00000000"""
        ch3_xyz = """C       0.00000000    0.00000001   -0.00000000
                     H       1.06690511   -0.17519582    0.05416493
                     H      -0.68531716   -0.83753536   -0.02808565
                     H      -0.38158795    1.01273118   -0.02607927"""
        h2o_xyz = """O      -0.00032832    0.39781490    0.00000000
                     H      -0.76330345   -0.19953755    0.00000000
                     H       0.76363177   -0.19827735    0.00000000"""
        r_1 = ARCSpecies(label='CH4', smiles='C', xyz=ch4_xyz)
        r_2 = ARCSpecies(label='OH', smiles='[OH]', xyz=oh_xyz)
        p_1 = ARCSpecies(label='CH3', smiles='[CH3]', xyz=ch3_xyz)
        p_2 = ARCSpecies(label='H2O', smiles='O', xyz=h2o_xyz)
        rxn = ARCReaction(r_species=[r_1, r_2], p_species=[p_1, p_2])
        atom_map = mapping.map_h_abstraction(rxn=rxn, db=self.rmgdb)
        self.assertEqual(atom_map[0], 0)
        self.assertIn(atom_map[1], [1, 2, 3, 5, 6])
        self.assertIn(atom_map[2], [1, 2, 3, 5, 6])
        self.assertIn(atom_map[3], [1, 2, 3, 5, 6])
        self.assertIn(atom_map[4], [1, 2, 3, 5, 6])
        self.assertEqual(atom_map[5], 4)
        self.assertIn(atom_map[6], [5, 6])
        self.assertTrue(any(atom_map[r_index] in [5, 6] for r_index in [1, 2, 3, 4]))

        # NH2 + N2H4 <=> NH3 + N2H3
        nh2_xyz = """N       0.00022972    0.40059496    0.00000000
                     H      -0.83174214   -0.19982058    0.00000000
                     H       0.83151242   -0.20077438    0.00000000"""
        n2h4_xyz = """N      -0.67026921   -0.02117571   -0.25636419
                      N       0.64966276    0.05515705    0.30069593
                      H      -1.27787600    0.74907557    0.03694453
                      H      -1.14684483   -0.88535632    0.02014513
                      H       0.65472168    0.28979031    1.29740292
                      H       1.21533718    0.77074524   -0.16656810"""
        nh3_xyz = """N       0.00064924   -0.00099698    0.29559292
                     H      -0.41786606    0.84210396   -0.09477452
                     H      -0.52039228   -0.78225292   -0.10002797
                     H       0.93760911   -0.05885406   -0.10079043"""
        n2h3_xyz = """N      -0.46371338    0.04553420    0.30600516
                      N       0.79024530   -0.44272936   -0.27090857
                      H      -1.18655934   -0.63438343    0.06795859
                      H      -0.71586186    0.90189070   -0.18800765
                      H       1.56071894    0.18069099    0.00439608"""
        r_1 = ARCSpecies(label='NH2', smiles='[NH2]', xyz=nh2_xyz)
        r_2 = ARCSpecies(label='N2H4', smiles='NN', xyz=n2h4_xyz)
        p_1 = ARCSpecies(label='NH3', smiles='N', xyz=nh3_xyz)
        p_2 = ARCSpecies(label='N2H3', smiles='N[NH]', xyz=n2h3_xyz)
        rxn = ARCReaction(r_species=[r_1, r_2], p_species=[p_1, p_2])
        atom_map = mapping.map_h_abstraction(rxn=rxn, db=self.rmgdb)
        self.assertEqual(atom_map[0], 0)
        self.assertIn(atom_map[1], [1, 2, 3])
        self.assertIn(atom_map[2], [1, 2, 3])
        self.assertIn(atom_map[3], [4, 5])
        self.assertIn(atom_map[4], [4, 5])
        self.assertTrue(any(atom_map[r_index] in [1, 2, 3] for r_index in [5, 6, 7, 8]))

        # NH2 + N2H4 <=> N2H3 + NH3
        rxn = ARCReaction(r_species=[r_1, r_2], p_species=[p_2, p_1])
        atom_map = mapping.map_h_abstraction(rxn=rxn, db=self.rmgdb)
        self.assertEqual(atom_map[0], 5)
        self.assertIn(atom_map[1], [6, 7, 8])
        self.assertIn(atom_map[2], [6, 7, 8])
        self.assertIn(atom_map[3], [0, 1])
        self.assertIn(atom_map[4], [0, 1])
        self.assertTrue(any(atom_map[r_index] in [6, 7, 8] for r_index in [5, 6, 7, 8]))

        # C3H6O + C4H9O <=> C3H5O + C4H10O
        c3h6o_xyz = {'coords': ((-1.0614352911982476, -0.35086070951203013, 0.3314546936475969),
                                (0.08232694092180896, 0.5949821397504677, 0.020767511136565348),
                                (1.319643623472743, -0.1238222051358961, -0.4579284002686819),
                                (1.4145501246584122, -1.339374145335546, -0.5896335370976351),
                                (-0.7813545474862899, -1.0625754884160945, 1.1151404910689675),
                                (-1.3481804813952152, -0.9258389945508673, -0.5552942813558058),
                                (-1.9370566523150816, 0.2087367432207233, 0.6743848589525232),
                                (-0.2162279757671984, 1.3021306884228383, -0.7596873819624604),
                                (0.35220978385921775, 1.1650050778348893, 0.9154971248602527),
                                (2.1755244752498673, 0.5316168937214946, -0.6947010789813145)),
                     'isotopes': (12, 12, 12, 16, 1, 1, 1, 1, 1, 1),
                     'symbols': ('C', 'C', 'C', 'O', 'H', 'H', 'H', 'H', 'H', 'H')}
        c4h9o_xyz = {'coords': ((0.025711531222639566, 1.5002469234994276, -0.018809721320361607),
                                (-0.2501237905589279, 2.283276320160058, 0.6795778782867752),
                                (0.21710649528235348, 1.7701501165266882, -1.0518607878262018),
                                (-0.1296127183749531, 0.05931626777072968, 0.3829802045651552),
                                (-1.5215969202773243, -0.4341372833972907, -0.0024458040153687616),
                                (0.954275466146204, -0.8261822387409435, -0.2512878552942834),
                                (2.238645869558612, -0.5229077195628998, 0.2868843893740711),
                                (-0.022719509344805086, 0.012299638536749403, 1.47391586262432),
                                (-1.6734988982808552, -1.4656213151526711, 0.3333615031669381),
                                (-1.6708084550075688, -0.40804497485420527, -1.0879383468423085),
                                (-2.3005261427143897, 0.18308085969254126, 0.45923715033920876),
                                (0.7583076310662862, -1.882720433150506, -0.04089782108496264),
                                (0.9972006722528377, -0.7025586995487184, -1.3391950754631268),
                                (2.377638769033351, 0.43380253822255727, 0.17647842348371048)),
                     'isotopes': (12, 1, 1, 12, 12, 12, 16, 1, 1, 1, 1, 1, 1, 1),
                     'symbols': ('C', 'H', 'H', 'C', 'C', 'C', 'O', 'H', 'H', 'H', 'H', 'H', 'H', 'H')}
        c3h5o_xyz = {'coords': ((-1.1339526749599567, -0.11366348271898848, -0.17361178233231772),
                                (0.1315989608873882, 0.19315012600914244, 0.5375291058021542),
                                (0.12186476447223683, 0.5479023323381329, 1.5587521800625246),
                                (1.435623589506148, 0.026762256080503182, -0.11697684942586563),
                                (1.5559845484585495, -0.3678359306766861, -1.2677014903374604),
                                (-1.6836994309836657, -0.8907558916446712, 0.3657463577153353),
                                (-1.7622426221647125, 0.7810307051429465, -0.21575166529131876),
                                (-0.9704526962734873, -0.4619573344933834, -1.1970278328709658),
                                (2.3052755610575106, 0.2853672199629854, 0.5090419766779545)),
                     'isotopes': (12, 12, 1, 12, 16, 1, 1, 1, 1),
                     'symbols': ('C', 'C', 'H', 'C', 'O', 'H', 'H', 'H', 'H')}
        c4h10o_xyz = {'coords': ((-1.0599869990613344, -1.2397714287161459, 0.010871360821665921),
                                 (-0.15570197396874313, -0.0399426912154684, -0.2627503141760959),
                                 (-0.8357120092418682, 1.2531917172190083, 0.1920887922885465),
                                 (1.2013757682054618, -0.22681093996845836, 0.42106399857821075),
                                 (2.0757871909243337, 0.8339710961541049, 0.05934908325727899),
                                 (-1.2566363886319676, -1.3536924078596617, 1.082401336123387),
                                 (-0.5978887839926055, -2.1649950925769703, -0.3492714363488459),
                                 (-2.0220571570609596, -1.1266512469159389, -0.4999630281827645),
                                 (0.0068492778433242255, 0.03845056912064928, -1.3453078463310726),
                                 (-0.22527545723287978, 2.1284779433126504, -0.05264318253022085),
                                 (-1.804297837475001, 1.3767516368254167, -0.30411519687565475),
                                 (-1.0079707678533625, 1.2514371624519658, 1.2738106811073706),
                                 (1.0967232048111195, -0.23572903005857432, 1.511374071529777),
                                 (1.6637048773271081, -1.1686406202494035, 0.10718319440789557),
                                 (2.9210870554073614, 0.6739533324768243, 0.512528859867013)),
                      'isotopes': (12, 12, 12, 12, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
                      'symbols': ('C', 'C', 'C', 'C', 'O', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H')}
        r_1 = ARCSpecies(label='C3H6O', smiles='CCC=O', xyz=c3h6o_xyz)
        r_2 = ARCSpecies(label='C4H9O', smiles='[CH2]C(C)CO', xyz=c4h9o_xyz)
        p_1 = ARCSpecies(label='C3H5O', smiles='C[CH]C=O', xyz=c3h5o_xyz)
        p_2 = ARCSpecies(label='C4H10O', smiles='CC(C)CO', xyz=c4h10o_xyz)
        rxn = ARCReaction(reactants=['C3H6O', 'C4H9O'], products=['C3H5O', 'C4H10O'],
                          r_species=[r_1, r_2], p_species=[p_1, p_2])
        atom_map = mapping.map_h_abstraction(rxn=rxn, db=self.rmgdb)
        self.assertEqual(atom_map[0], 0)
        self.assertEqual(atom_map[1], 1)
        self.assertEqual(atom_map[2], 3)
        self.assertEqual(atom_map[3], 4)
        self.assertIn(atom_map[4], [5, 6, 7])
        self.assertIn(atom_map[5], [5, 6, 7])
        self.assertIn(atom_map[6], [5, 6, 7])
        self.assertIn(atom_map[7], [2, 14, 15, 16, 18, 19, 20])
        self.assertIn(atom_map[8], [2, 14, 15, 16, 18, 19, 20])
        self.assertTrue(any(entry == 2 for entry in [atom_map[7], atom_map[8]]))
        self.assertEqual(atom_map[9], 8)
        self.assertIn(atom_map[10], [9, 11])
        self.assertIn(atom_map[11], [14, 15, 16, 18, 19, 20])
        self.assertIn(atom_map[12], [14, 15, 16, 18, 19, 20])
        self.assertEqual(atom_map[13], 10)
        self.assertIn(atom_map[14], [9, 11])
        self.assertEqual(atom_map[15], 12)
        self.assertEqual(atom_map[16], 13)
        self.assertEqual(atom_map[17], 17)
        self.assertIn(atom_map[18], [14, 15, 16, 18, 19, 20])
        self.assertIn(atom_map[19], [14, 15, 16, 18, 19, 20])
        self.assertIn(atom_map[20], [14, 15, 16, 18, 19, 20])
        self.assertIn(atom_map[21], [21, 22])
        self.assertIn(atom_map[22], [21, 22])
        self.assertEqual(atom_map[23], 23)

    def test_map_ho2_elimination_from_peroxy_radical(self):
        """Test the map_ho2_elimination_from_peroxy_radical() function."""
        r_xyz = """N      -0.82151000   -0.98211000   -0.58727000
                   C      -0.60348000    0.16392000    0.30629000
                   C       0.85739000    0.41515000    0.58956000
                   C       1.91892000   -0.27446000    0.14220000
                   O      -1.16415000    1.38916000   -0.20784000
                   O      -2.39497344    1.57487672    0.46214548
                   H      -0.50088000   -0.69919000   -1.51181000
                   H      -1.83926000   -1.03148000   -0.69340000
                   H      -1.09049000   -0.04790000    1.26633000
                   H       1.04975000    1.25531000    1.25575000
                   H       2.92700000    0.00462000    0.43370000
                   H       1.81273000   -1.13911000   -0.50660000"""  # NC(C=C)O[O]
        p_1_xyz = """N       1.16378795    1.46842703   -0.82620909
                     C       0.75492192    0.42940001   -0.18269967
                     C      -0.66835457    0.05917401   -0.13490822
                     C      -1.06020680   -1.02517494    0.54162130
                     H       2.18280085    1.55132949   -0.73741996
                     H       1.46479392   -0.22062618    0.35707573
                     H      -1.36374229    0.69906451   -0.66578157
                     H      -2.11095970   -1.29660899    0.57562763
                     H      -0.36304116   -1.66498540    1.07269317"""  # N=CC=C
        p_2_xyz = """N      -1.60333711   -0.23049987   -0.35673484
                     C      -0.63074775    0.59837442    0.08043329
                     C       0.59441219    0.18489797    0.16411656
                     C       1.81978128   -0.23541908    0.24564488
                     H      -2.56057110    0.09083582   -0.42266843
                     H      -1.37296018   -1.18147301   -0.62077856
                     H      -0.92437032    1.60768040    0.35200716
                     H       2.49347824   -0.13648710   -0.59717108
                     H       2.18431385   -0.69791121    1.15515621"""  # NC=C=C
        ho2_xyz = """O      -0.18935000    0.42639000    0.00000000
                     O       1.07669000   -0.17591000    0.00000000
                     H      -0.88668000   -0.25075000    0.00000000"""  # O[O]
        rxn_1 = ARCReaction(r_species=[ARCSpecies(label='R', smiles='NC(C=C)O[O]', xyz=r_xyz)],
                            p_species=[ARCSpecies(label='P1', smiles='N=CC=C', xyz=p_1_xyz),
                                       ARCSpecies(label='HO2', smiles='O[O]', xyz=ho2_xyz)])
        atom_map = mapping.map_ho2_elimination_from_peroxy_radical(rxn_1)
        print(atom_map)  # [0, 1, 2, 3, 10, 4, 9, 6, 11, 7, 5, 8]
        self.assertEqual(atom_map[:6], [0, 1, 2, 3, 10, 9])
        self.assertIn(atom_map[6], [4, 11])
        self.assertIn(atom_map[7], [4, 11])
        self.assertEqual(atom_map[8], 5)
        self.assertEqual(atom_map[9], 6)
        self.assertIn(atom_map[10], [7, 8])
        self.assertIn(atom_map[11], [7, 8])

        # Todo: also test (and write func) in reverse, also test the other rxn in both dirs

    def test_map_intra_h_migration(self):
        """Test the map_intra_h_migration() function."""
        atom_map = mapping.map_intra_h_migration(self.arc_reaction_4)
        self.assertEqual(atom_map[0], 0)
        self.assertEqual(atom_map[1], 1)
        self.assertEqual(atom_map[2], 2)
        self.assertIn(atom_map[3], [3, 4, 5])
        self.assertIn(atom_map[4], [3, 4, 5])
        self.assertIn(atom_map[5], [6, 7])
        self.assertIn(atom_map[6], [6, 7])
        self.assertIn(atom_map[7], [3, 4, 5, 8])
        self.assertIn(atom_map[8], [3, 4, 5, 8])

    def test_create_qc_mol(self):
        """Test the create_qc_mol() function."""
        qcmol1 = mapping.create_qc_mol(species=ARCSpecies(label='S1', smiles='C'))
        self.assertIsInstance(qcmol1, QCMolecule)
        self.assertEqual(qcmol1.molecular_charge, 0)
        self.assertEqual(qcmol1.molecular_multiplicity, 1)
        for symbol, expected_symbol in zip(qcmol1.symbols, ['C', 'H', 'H', 'H', 'H']):
            self.assertEqual(symbol, expected_symbol)

        qcmol2 = mapping.create_qc_mol(species=[ARCSpecies(label='S1', smiles='C'),
                                                ARCSpecies(label='S2', smiles='N[CH2]')],
                                       charge=0,
                                       multiplicity=2,
                                       )
        self.assertIsInstance(qcmol2, QCMolecule)
        self.assertEqual(qcmol2.molecular_charge, 0)
        self.assertEqual(qcmol2.molecular_multiplicity, 2)
        for symbol, expected_symbol in zip(qcmol2.symbols, ['C', 'H', 'H', 'H', 'H', 'N', 'C', 'H', 'H', 'H', 'H']):
            self.assertEqual(symbol, expected_symbol)

    def test_map_two_species(self):
        """Test the map_two_species() function."""
        spc1 = ARCSpecies(label='H', smiles='[H]')
        spc2 = ARCSpecies(label='H', smiles='[H]')
        atom_map = mapping.map_two_species(spc1, spc2)
        self.assertEqual(atom_map, [0])

        spc1 = ARCSpecies(label='OH', smiles='[OH]', xyz="""O 0 0 0\nH 0.8 0 0""")
        spc2 = ARCSpecies(label='OH', smiles='[OH]', xyz="""H 0 0 0\nO 0 0.9 0""")
        atom_map = mapping.map_two_species(spc1, spc2)
        self.assertEqual(atom_map, [1, 0])

        spc1 = ARCSpecies(label='CH4', smiles='C',
                          xyz="""C      -0.00000000    0.00000000    0.00000000
                                 H      -0.65055201   -0.77428020   -0.41251879
                                 H      -0.34927558    0.98159583   -0.32768232
                                 H      -0.02233792   -0.04887375    1.09087665
                                 H       1.02216551   -0.15844188   -0.35067554""")
        spc2 = ARCSpecies(label='CH4', smiles='C',
                          xyz="""H      -0.65055201   -0.77428020   -0.41251879
                                 H      -0.34927558    0.98159583   -0.32768232
                                 C      -0.00000000    0.00000000    0.00000000
                                 H      -0.02233792   -0.04887375    1.09087665
                                 H       1.02216551   -0.15844188   -0.35067554""")
        atom_map = mapping.map_two_species(spc1, spc2)
        self.assertEqual(atom_map, [2, 0, 1, 3, 4])

        spc1 = ARCSpecies(label='CCHCHO', smiles='CC=C[O]',
                          xyz="""C      -1.13395267   -0.11366348   -0.17361178
                                 C       0.13159896    0.19315013    0.53752911
                                 H       0.12186476    0.54790233    1.55875218
                                 C       1.43562359    0.02676226   -0.11697685
                                 O       1.55598455   -0.36783593   -1.26770149
                                 H      -1.68369943   -0.89075589    0.36574636
                                 H      -1.76224262    0.78103071   -0.21575167
                                 H      -0.97045270   -0.46195733   -1.19702783
                                 H       2.30527556    0.28536722    0.50904198""")
        spc2 = ARCSpecies(label='CCCHO', smiles='C[CH]C=O',
                          xyz="""C      -1.06143529   -0.35086071    0.33145469
                                 C       0.08232694    0.59498214    0.02076751
                                 C       1.31964362   -0.12382221   -0.45792840
                                 O       1.41455012   -1.33937415   -0.58963354
                                 H      -0.78135455   -1.06257549    1.11514049
                                 H      -1.34818048   -0.92583899   -0.55529428
                                 H      -1.93705665    0.20873674    0.67438486
                                 H      -0.21622798    1.30213069   -0.75968738
                                 H       2.17552448    0.53161689   -0.69470108""")
        atom_map = mapping.map_two_species(spc1, spc2)
        self.assertEqual(atom_map, [0, 1, 7, 2, 3, 6, 5, 4, 8])

        # spc1 = ARCSpecies(label='C=CC=N', smiles='C=CC=N',
        #                   xyz={'symbols': ('N', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H'),
        #                        'isotopes': (14, 12, 12, 12, 1, 1, 1, 1, 1),
        #                        'coords': ((1.16378795, 1.46842703, -0.82620909),
        #                                   (0.75492192, 0.42940001, -0.18269967),
        #                                   (-0.66835457, 0.05917401, -0.13490822),
        #                                   (-1.0602068, -1.02517494, 0.5416213),
        #                                   (2.18280085, 1.55132949, -0.73741996),
        #                                   (1.46479392, -0.22062618, 0.35707573),
        #                                   (-1.36374229, 0.69906451, -0.66578157),
        #                                   (-2.1109597, -1.29660899, 0.57562763),
        #                                   (-0.36304116, -1.6649854, 1.07269317))})
        # spc2 = ARCSpecies(label='C=CC=N_mod', smiles='C=CC=N',
        #                   xyz={'symbols': ('N', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H'),
        #                        'isotopes': (14, 12, 12, 12, 1, 1, 1, 1, 1),
        #                        'coords': ((-0.82151, -0.98211, -0.58727),
        #                                   (-0.60348, 0.16392, 0.30629),
        #                                   (0.8573899999999992, 0.41514999999999985, 0.5895599999999999),
        #                                   (1.3997911040139661, 1.372785085984443, 1.358504046765431),
        #                                   (-1.83926, -1.03148, -0.6934),
        #                                   (-1.09049, -0.0479, 1.26633),
        #                                   (1.5389692539529671, -0.26718857011726227, 0.08305816623111278),
        #                                   (2.4768853016366608, 1.4534535661976533, 1.4700561357087325),
        #                                   (0.7918676809585643, 2.1071205322982185, 1.8790978832519483))})
        # atom_map = mapping.map_two_species(spc1, spc2)
        # self.assertEqual(atom_map, [0, 1, 2, 3, 4, 5, 6, 7, 8])

    def test_get_atom_indices_of_labeled_atoms_in_an_rmg_reaction(self):
        """Test the get_atom_indices_of_labeled_atoms_in_an_rmg_reaction() function."""
        determine_family(self.arc_reaction_1)
        rmg_reactions = mapping._get_rmg_reactions_from_arc_reaction(arc_reaction=self.arc_reaction_1, db=self.rmgdb)
        r_dict, p_dict = mapping.get_atom_indices_of_labeled_atoms_in_an_rmg_reaction(arc_reaction=self.arc_reaction_1,
                                                                                      rmg_reaction=rmg_reactions[0])
        self.assertEqual(r_dict['*1'], 0)
        self.assertIn(r_dict['*2'], [1, 2, 3, 4])
        self.assertEqual(r_dict['*3'], 5)
        self.assertEqual(p_dict['*1'], 0)
        self.assertIn(p_dict['*2'], [5, 6])
        self.assertEqual(p_dict['*3'], 4)

        determine_family(self.arc_reaction_2)
        rmg_reactions = mapping._get_rmg_reactions_from_arc_reaction(arc_reaction=self.arc_reaction_2, db=self.rmgdb)
        r_dict, p_dict = mapping.get_atom_indices_of_labeled_atoms_in_an_rmg_reaction(arc_reaction=self.arc_reaction_2,
                                                                                      rmg_reaction=rmg_reactions[0])
        self.assertIn(r_dict['*1'], [0, 2])
        self.assertIn(r_dict['*2'], [3, 4, 5, 8, 9, 10])
        self.assertEqual(r_dict['*3'], 11)
        self.assertEqual(p_dict['*1'], 0)
        self.assertIn(p_dict['*2'], [11, 12, 13])
        self.assertEqual(p_dict['*3'], 10)

        determine_family(self.arc_reaction_4)
        rmg_reactions = mapping._get_rmg_reactions_from_arc_reaction(arc_reaction=self.arc_reaction_4, db=self.rmgdb)
        r_dict, p_dict = mapping.get_atom_indices_of_labeled_atoms_in_an_rmg_reaction(arc_reaction=self.arc_reaction_4,
                                                                                      rmg_reaction=rmg_reactions[0])
        self.assertEqual(r_dict['*1'], 0)
        self.assertEqual(r_dict['*2'], 2)
        self.assertIn(r_dict['*3'], [7, 8])
        self.assertEqual(p_dict['*1'], 0)
        self.assertEqual(p_dict['*2'], 2)
        self.assertIn(p_dict['*3'], [3, 4, 5])

        determine_family(self.rxn_2a)
        for atom, symbol in zip(self.rxn_2a.r_species[0].mol.atoms, ['C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H']):
            self.assertEqual(atom.symbol, symbol)
        self.assertEqual(self.rxn_2a.r_species[0].mol.atoms[0].radical_electrons, 0)
        self.assertEqual(self.rxn_2a.r_species[0].mol.atoms[1].radical_electrons, 1)
        self.assertEqual(self.rxn_2a.r_species[0].mol.atoms[2].radical_electrons, 0)
        self.assertEqual(self.rxn_2a.p_species[0].mol.atoms[0].radical_electrons, 0)
        self.assertEqual(self.rxn_2a.p_species[0].mol.atoms[1].radical_electrons, 0)
        self.assertEqual(self.rxn_2a.p_species[0].mol.atoms[2].radical_electrons, 1)
        rmg_reactions = mapping._get_rmg_reactions_from_arc_reaction(arc_reaction=self.rxn_2a, db=self.rmgdb)
        r_dict, p_dict = mapping.get_atom_indices_of_labeled_atoms_in_an_rmg_reaction(arc_reaction=self.rxn_2a,
                                                                                      rmg_reaction=rmg_reactions[0])
        self.assertEqual(r_dict['*1'], 1)
        self.assertIn(r_dict['*2'], [0, 2])
        self.assertIn(r_dict['*3'], [4, 5, 6, 7, 8, 9])
        self.assertEqual(p_dict['*1'], 1)
        self.assertEqual(p_dict['*2'], 2)
        self.assertIn(p_dict['*3'], [3, 6])

        determine_family(self.rxn_2b)
        for atom, symbol in zip(self.rxn_2b.r_species[0].mol.atoms, ['C', 'C', 'H', 'H', 'H', 'H', 'C', 'H', 'H', 'H']):
            self.assertEqual(atom.symbol, symbol)
        rmg_reactions = mapping._get_rmg_reactions_from_arc_reaction(arc_reaction=self.rxn_2b, db=self.rmgdb)
        r_dict, p_dict = mapping.get_atom_indices_of_labeled_atoms_in_an_rmg_reaction(arc_reaction=self.rxn_2b,
                                                                                      rmg_reaction=rmg_reactions[0])
        self.assertEqual(r_dict['*1'], 1)
        self.assertIn(r_dict['*2'], [0, 6])
        self.assertIn(r_dict['*3'], [3, 4, 5, 7, 8, 9])
        self.assertEqual(p_dict['*1'], 1)
        self.assertEqual(p_dict['*2'], 2)
        self.assertIn(p_dict['*3'], [3, 6])

        # C3H6O + C4H9O <=> C3H5O + C4H10O
        r_1 = ARCSpecies(label='C3H6O', smiles='CCC=O')
        r_2 = ARCSpecies(label='C4H9O', smiles='[CH2]C(C)CO')
        p_1 = ARCSpecies(label='C3H5O', smiles='C[CH]C=O')
        p_2 = ARCSpecies(label='C4H10O', smiles='CC(C)CO')
        rxn_1 = ARCReaction(reactants=['C3H6O', 'C4H9O'], products=['C3H5O', 'C4H10O'],
                            r_species=[r_1, r_2], p_species=[p_1, p_2])
        determine_family(rxn_1)
        rmg_reactions = mapping._get_rmg_reactions_from_arc_reaction(arc_reaction=rxn_1, db=self.rmgdb)
        for rmg_reaction in rmg_reactions:
            r_dict, p_dict = mapping.get_atom_indices_of_labeled_atoms_in_an_rmg_reaction(arc_reaction=rxn_1,
                                                                                          rmg_reaction=rmg_reaction)
            for d in [r_dict, p_dict]:
                self.assertEqual(len(list(d.keys())), 3)
                keys = list(d.keys())
                for label in ['*1', '*2', '*3']:
                    self.assertIn(label, keys)

        p_1 = ARCSpecies(label='C3H5O', smiles='CC=C[O]')  # Use a wrong resonance structure and repeat the above.
        rxn_2 = ARCReaction(reactants=['C3H6O', 'C4H9O'], products=['C3H5O', 'C4H10O'],
                            r_species=[r_1, r_2], p_species=[p_1, p_2])
        determine_family(rxn_2)
        rmg_reactions = mapping._get_rmg_reactions_from_arc_reaction(arc_reaction=rxn_2, db=self.rmgdb)
        for rmg_reaction in rmg_reactions:
            r_dict, p_dict = mapping.get_atom_indices_of_labeled_atoms_in_an_rmg_reaction(arc_reaction=rxn_2,
                                                                                          rmg_reaction=rmg_reaction)
            for d in [r_dict, p_dict]:
                self.assertEqual(len(list(d.keys())), 3)
                keys = list(d.keys())
                for label in ['*1', '*2', '*3']:
                    self.assertIn(label, keys)

        # C3H6O + C4H9O <=> C3H5O + C4H10O
        c3h6o_xyz = {'coords': ((-1.0614352911982476, -0.35086070951203013, 0.3314546936475969),
                                (0.08232694092180896, 0.5949821397504677, 0.020767511136565348),
                                (1.319643623472743, -0.1238222051358961, -0.4579284002686819),
                                (1.4145501246584122, -1.339374145335546, -0.5896335370976351),
                                (-0.7813545474862899, -1.0625754884160945, 1.1151404910689675),
                                (-1.3481804813952152, -0.9258389945508673, -0.5552942813558058),
                                (-1.9370566523150816, 0.2087367432207233, 0.6743848589525232),
                                (-0.2162279757671984, 1.3021306884228383, -0.7596873819624604),
                                (0.35220978385921775, 1.1650050778348893, 0.9154971248602527),
                                (2.1755244752498673, 0.5316168937214946, -0.6947010789813145)),
                     'isotopes': (12, 12, 12, 16, 1, 1, 1, 1, 1, 1),
                     'symbols': ('C', 'C', 'C', 'O', 'H', 'H', 'H', 'H', 'H', 'H')}
        c4h9o_xyz = {'coords': ((0.025711531222639566, 1.5002469234994276, -0.018809721320361607),
                                (-0.2501237905589279, 2.283276320160058, 0.6795778782867752),
                                (0.21710649528235348, 1.7701501165266882, -1.0518607878262018),
                                (-0.1296127183749531, 0.05931626777072968, 0.3829802045651552),
                                (-1.5215969202773243, -0.4341372833972907, -0.0024458040153687616),
                                (0.954275466146204, -0.8261822387409435, -0.2512878552942834),
                                (2.238645869558612, -0.5229077195628998, 0.2868843893740711),
                                (-0.022719509344805086, 0.012299638536749403, 1.47391586262432),
                                (-1.6734988982808552, -1.4656213151526711, 0.3333615031669381),
                                (-1.6708084550075688, -0.40804497485420527, -1.0879383468423085),
                                (-2.3005261427143897, 0.18308085969254126, 0.45923715033920876),
                                (0.7583076310662862, -1.882720433150506, -0.04089782108496264),
                                (0.9972006722528377, -0.7025586995487184, -1.3391950754631268),
                                (2.377638769033351, 0.43380253822255727, 0.17647842348371048)),
                     'isotopes': (12, 1, 1, 12, 12, 12, 16, 1, 1, 1, 1, 1, 1, 1),
                     'symbols': ('C', 'H', 'H', 'C', 'C', 'C', 'O', 'H', 'H', 'H', 'H', 'H', 'H', 'H')}
        c3h5o_xyz = {'coords': ((-1.1339526749599567, -0.11366348271898848, -0.17361178233231772),
                                (0.1315989608873882, 0.19315012600914244, 0.5375291058021542),
                                (0.12186476447223683, 0.5479023323381329, 1.5587521800625246),
                                (1.435623589506148, 0.026762256080503182, -0.11697684942586563),
                                (1.5559845484585495, -0.3678359306766861, -1.2677014903374604),
                                (-1.6836994309836657, -0.8907558916446712, 0.3657463577153353),
                                (-1.7622426221647125, 0.7810307051429465, -0.21575166529131876),
                                (-0.9704526962734873, -0.4619573344933834, -1.1970278328709658),
                                (2.3052755610575106, 0.2853672199629854, 0.5090419766779545)),
                     'isotopes': (12, 12, 1, 12, 16, 1, 1, 1, 1),
                     'symbols': ('C', 'C', 'H', 'C', 'O', 'H', 'H', 'H', 'H')}
        c4h10o_xyz = {'coords': ((-1.0599869990613344, -1.2397714287161459, 0.010871360821665921),
                                 (-0.15570197396874313, -0.0399426912154684, -0.2627503141760959),
                                 (-0.8357120092418682, 1.2531917172190083, 0.1920887922885465),
                                 (1.2013757682054618, -0.22681093996845836, 0.42106399857821075),
                                 (2.0757871909243337, 0.8339710961541049, 0.05934908325727899),
                                 (-1.2566363886319676, -1.3536924078596617, 1.082401336123387),
                                 (-0.5978887839926055, -2.1649950925769703, -0.3492714363488459),
                                 (-2.0220571570609596, -1.1266512469159389, -0.4999630281827645),
                                 (0.0068492778433242255, 0.03845056912064928, -1.3453078463310726),
                                 (-0.22527545723287978, 2.1284779433126504, -0.05264318253022085),
                                 (-1.804297837475001, 1.3767516368254167, -0.30411519687565475),
                                 (-1.0079707678533625, 1.2514371624519658, 1.2738106811073706),
                                 (1.0967232048111195, -0.23572903005857432, 1.511374071529777),
                                 (1.6637048773271081, -1.1686406202494035, 0.10718319440789557),
                                 (2.9210870554073614, 0.6739533324768243, 0.512528859867013)),
                      'isotopes': (12, 12, 12, 12, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
                      'symbols': ('C', 'C', 'C', 'C', 'O', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H')}
        r_1 = ARCSpecies(label='C3H6O', smiles='CCC=O', xyz=c3h6o_xyz)
        r_2 = ARCSpecies(label='C4H9O', smiles='[CH2]C(C)CO', xyz=c4h9o_xyz)
        p_1 = ARCSpecies(label='C3H5O', smiles='C[CH]C=O', xyz=c3h5o_xyz)
        p_2 = ARCSpecies(label='C4H10O', smiles='CC(C)CO', xyz=c4h10o_xyz)
        rxn_3 = ARCReaction(reactants=['C3H6O', 'C4H9O'], products=['C3H5O', 'C4H10O'],
                            r_species=[r_1, r_2], p_species=[p_1, p_2])
        determine_family(rxn_3)
        rmg_reactions = mapping._get_rmg_reactions_from_arc_reaction(arc_reaction=rxn_3, db=self.rmgdb)
        r_dict, p_dict = mapping.get_atom_indices_of_labeled_atoms_in_an_rmg_reaction(arc_reaction=rxn_3,
                                                                                      rmg_reaction=rmg_reactions[0])
        self.assertEqual(r_dict, {'*3': 10, '*1': 1, '*2': 7})
        self.assertEqual(p_dict, {'*1': 1, '*3': 9, '*2': 16})

    def test_map_arc_rmg_species(self):
        """Test the map_arc_rmg_species() function."""
        r_map, p_map = mapping.map_arc_rmg_species(arc_reaction=ARCReaction(r_species=[ARCSpecies(label='CCjC', smiles='C[CH]C')],
                                                                            p_species=[ARCSpecies(label='CjCC', smiles='[CH2]CC')]),
                                                   rmg_reaction=Reaction(reactants=[Species(smiles='C[CH]C')],
                                                                         products=[Species(smiles='[CH2]CC')]),
                                                   concatenate=False)
        self.assertEqual(r_map, {0: 0})
        self.assertEqual(p_map, {0: 0})

        r_map, p_map = mapping.map_arc_rmg_species(arc_reaction=ARCReaction(r_species=[ARCSpecies(label='CCjC', smiles='C[CH]C')],
                                                                            p_species=[ARCSpecies(label='CjCC', smiles='[CH2]CC')]),
                                                   rmg_reaction=Reaction(reactants=[Species(smiles='C[CH]C')],
                                                                         products=[Species(smiles='[CH2]CC')]))
        self.assertEqual(r_map, {0: [0]})
        self.assertEqual(p_map, {0: [0]})

        r_map, p_map = mapping.map_arc_rmg_species(arc_reaction=self.arc_reaction_1, rmg_reaction=self.rmg_reaction_1)
        self.assertEqual(r_map, {0: [0], 1: [1]})
        self.assertEqual(p_map, {0: [0], 1: [1]})

        r_map, p_map = mapping.map_arc_rmg_species(arc_reaction=self.arc_reaction_1, rmg_reaction=self.rmg_reaction_2)
        self.assertEqual(r_map, {0: [1], 1: [0]})
        self.assertEqual(p_map, {0: [0], 1: [1]})

        r_map, p_map = mapping.map_arc_rmg_species(arc_reaction=self.arc_reaction_3, rmg_reaction=self.rmg_reaction_3)
        self.assertEqual(r_map, {0: [0, 1], 1: [0, 1]})
        self.assertEqual(p_map, {0: [0]})

        rmg_reaction_1 = Reaction(reactants=[Species(smiles='N'), Species(smiles='[H]')],
                                  products=[Species(smiles='[NH2]'), Species(smiles='[H][H]')])
        rmg_reaction_2 = Reaction(reactants=[Species(smiles='[H]'), Species(smiles='N')],
                                  products=[Species(smiles='[H][H]'), Species(smiles='[NH2]')])
        rmg_reaction_3 = Reaction(reactants=[Species(smiles='N'), Species(smiles='[H]')],
                                  products=[Species(smiles='[H][H]'), Species(smiles='[NH2]')])
        arc_reaction = ARCReaction(r_species=[ARCSpecies(label='NH3', smiles='N'), ARCSpecies(label='H', smiles='[H]')],
                                   p_species=[ARCSpecies(label='NH2', smiles='[NH2]'), ARCSpecies(label='H2', smiles='[H][H]')])

        r_map, p_map = mapping.map_arc_rmg_species(rmg_reaction=rmg_reaction_1, arc_reaction=arc_reaction)
        self.assertEqual(r_map, {0: [0], 1: [1]})
        self.assertEqual(p_map, {0: [0], 1: [1]})

        r_map, p_map = mapping.map_arc_rmg_species(rmg_reaction=rmg_reaction_2, arc_reaction=arc_reaction)
        self.assertEqual(r_map, {0: [1], 1: [0]})
        self.assertEqual(p_map, {0: [1], 1: [0]})

        r_map, p_map = mapping.map_arc_rmg_species(rmg_reaction=rmg_reaction_3, arc_reaction=arc_reaction)
        self.assertEqual(r_map, {0: [0], 1: [1]})
        self.assertEqual(p_map, {0: [1], 1: [0]})

        rmg_reaction = Reaction(reactants=[Species(smiles='[CH3]'), Species(smiles='[CH3]')],
                                products=[Species(smiles='CC')])
        arc_reaction = ARCReaction(r_species=[ARCSpecies(label='CH3', smiles='[CH3]'),
                                              ARCSpecies(label='CH3', smiles='[CH3]')],
                                   p_species=[ARCSpecies(label='C2H6', smiles='CC')])

        r_map, p_map = mapping.map_arc_rmg_species(rmg_reaction=rmg_reaction, arc_reaction=arc_reaction)
        self.assertEqual(r_map, {0: [0, 1], 1: [0, 1]})
        self.assertEqual(p_map, {0: [0]})

    def test_find_equivalent_atoms_in_reactants_and_products(self):
        """Test the find_equivalent_atoms_in_reactants_and_products() function"""
        equivalence_map_1 = mapping.find_equivalent_atoms_in_reactants(arc_reaction=self.rxn_2a)
        # Both C 0 and C 2 are equivalent, C 1 is unique, and H 4-9 are equivalent as well.
        self.assertEqual(equivalence_map_1, [[0, 2], [1], [4, 5, 6, 7, 8, 9]])
        equivalence_map_2 = mapping.find_equivalent_atoms_in_reactants(arc_reaction=self.rxn_2b)
        self.assertEqual(equivalence_map_2, [[0, 6], [1], [3, 4, 5, 7, 8, 9]])


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
