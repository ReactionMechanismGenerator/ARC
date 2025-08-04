#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests for the arc.checks.nmd module
"""

import unittest
import os
import shutil

import numpy as np

import arc.checks.nmd as nmd
from arc.common import ARC_PATH
from arc.job.factory import job_factory
from arc.level import Level
from arc.parser import parse_normal_mode_displacement
from arc.reaction import ARCReaction
from arc.species.species import ARCSpecies
from arc.species.converter import check_xyz_dict
from rmgpy.molecule import Molecule


class TestNMD(unittest.TestCase):
    """
    Contains unit tests for the nmd module.
    """
    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None
        cls.generic_job = job_factory(job_adapter='gaussian',
                                      species=[ARCSpecies(label='SPC', smiles='C')],
                                      job_type='composite',
                                      level=Level(method='CBS-QB3'),
                                      project='test_project',
                                      project_directory=os.path.join(ARC_PATH, 'Projects', 'tmp_nmd_project'),
                                      )
        cls.xyz_1 = {'symbols': ('C', 'N', 'H', 'H', 'H', 'H'),
                     'isotopes': (13, 14, 1, 1, 1, 1),
                     'coords': ((0.6616514836, 0.4027481525, -0.4847382281),
                                (-0.6039793084, 0.6637270105, 0.0671637135),
                                (-1.4226865648, -0.4973210697, -0.2238712255),
                                (-0.4993010635, 0.6531020442, 1.0853092315),
                                (-2.2115796924, -0.4529256762, 0.4144516252),
                                (-1.8113671395, -0.3268900681, -1.1468957003))}
        cls.weights_1 = np.array([[3.60601648], [3.74206815], [1.00390489], [1.00390489], [1.00390489], [1.00390489]])
        cls.nmd_1 = np.array([[-0.5, 0.0, -0.09], [-0.0, 0.0, -0.1], [0.0, 0.0, -0.01],
                              [0.0, 0.0, -0.07], [0.0, 0.0, -0.2], [-0.0, -0.0, 0.28]], np.float64)

        cls.ch4_xyz = """C      -0.00000000    0.00000000    0.00000000
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
        cls.rxn_1 = ARCReaction(r_species=[ARCSpecies(label='CH4', smiles='C', xyz=cls.ch4_xyz),
                                           ARCSpecies(label='OH', smiles='[OH]', xyz=oh_xyz)],
                                p_species=[ARCSpecies(label='CH3', smiles='[CH3]', xyz=ch3_xyz),
                                           ARCSpecies(label='H2O', smiles='O', xyz=h2o_xyz)])
        cls.ts_1_xyz = check_xyz_dict("""C                    -1.212192   -0.010161    0.000000
                                         H                     0.010122    0.150115    0.000001
                                         H                    -1.460491   -0.555461   -0.907884
                                         H                    -1.460499   -0.555391    0.907924
                                         H                    -1.600535    1.006984   -0.000041
                                         O                     1.295714    0.108618   -0.000000
                                         H                     1.418838   -0.854225    0.000000""")
        cls.rxn_1.ts_species = ARCSpecies(label='TS1', is_ts=True, xyz=cls.ts_1_xyz)

        cls.ho2_xyz = """O       1.00509800   -0.18331500   -0.00000000
                         O      -0.16548400    0.44416100    0.00000000
                         H      -0.83961400   -0.26084600    0.00000000"""
        cls.n2h4_xyz = """N      -0.66510800   -0.10671700   -0.25444200
                          N       0.63033400    0.04211900    0.34557500
                          H      -1.16070500    0.76768900   -0.12511600
                          H      -1.21272700   -0.83945300    0.19196500
                          H       1.26568700   -0.57247200   -0.14993500
                          H       0.63393800   -0.23649100    1.32457000"""
        cls.h2o2_xyz = """O       0.60045000   -0.40342400    0.24724100
                          O      -0.59754500    0.41963800    0.22641300
                          H       1.20401100    0.16350100   -0.25009400
                          H      -1.20691600   -0.17971500   -0.22356000"""
        cls.n2h3_xyz = """N       0.74263400   -0.29604200    0.40916100
                          N      -0.39213800   -0.13735700   -0.31177100
                          H       1.49348100    0.07315400   -0.18245700
                          H      -1.18274100   -0.63578900    0.07132400
                          H      -0.36438800   -0.12591900   -1.32684600"""
        cls.rxn_2 = ARCReaction(r_species=[ARCSpecies(label='HO2', smiles='O[O]', xyz=cls.ho2_xyz),
                                           ARCSpecies(label='N2H4', smiles='NN', xyz=cls.n2h4_xyz)],
                                p_species=[ARCSpecies(label='H2O2', smiles='OO', xyz=cls.h2o2_xyz),
                                           ARCSpecies(label='N2H3', smiles='N[NH]', xyz=cls.n2h3_xyz)])
        cls.ts_2_xyz = check_xyz_dict("""O      -1.275006   -0.656458   -0.217271
                                         O      -1.463976    0.650010    0.244741
                                         H      -1.239724    1.193859   -0.524436
                                         N       1.456296    0.587219   -0.119103
                                         N       1.134258   -0.749491    0.081665
                                         H       1.953072    0.715591   -0.992228
                                         H       1.926641    1.040462    0.653661
                                         H      -0.004455   -0.848273   -0.183117
                                         H       1.142454   -0.914148    1.088425""")
        cls.rxn_2.ts_species = ARCSpecies(label='TS2', is_ts=True, xyz=cls.ts_2_xyz)
        cls.freq_log_path_2 = os.path.join(ARC_PATH, 'arc', 'testing', 'freq', 'HO2+N2H4_H2O2+N2H3_freq.out')

        cls.displaced_xyz_3 = ({'symbols': ('C', 'N', 'H', 'H', 'H', 'H'), 'isotopes': (13, 14, 1, 1, 1, 1),
                                'coords': ((1.1124035436000002, 0.4027481525, -0.4036028573),
                                           (-0.6039793084, 0.6637270105, 0.16071541725),
                                           (-1.4226865648, -0.4973210697, -0.221361463275),
                                           (-0.4993010635, 0.6531020442, 1.102877567075),
                                           (-2.2115796924, -0.4529256762, 0.4646468697),
                                           (-1.8113671395, -0.3268900681, -1.2171690426))},
                               {'symbols': ('C', 'N', 'H', 'H', 'H', 'H'), 'isotopes': (13, 14, 1, 1, 1, 1),
                                'coords': ((0.21089942360000002, 0.4027481525, -0.5658735989),
                                           (-0.6039793084, 0.6637270105, -0.026387990250000007),
                                           (-1.4226865648, -0.4973210697, -0.22638098772499998),
                                           (-0.4993010635, 0.6531020442, 1.0677408959249999),
                                           (-2.2115796924, -0.4529256762, 0.3642563807),
                                           (-1.8113671395, -0.3268900681, -1.076622358))})

        cls.ts_xyz_2 = """C       -1.278012   -0.190724   -0.025183
                          C        0.041522    0.515907    0.026225
                          C        1.316457   -0.254613   -0.035061
                          H        0.091907    1.563260   -0.285810
                          H       -2.097896    0.442595    0.343854
                          H       -1.263271   -1.109504    0.583429
                          H        0.818808    0.292397    1.041458
                          H       -1.528388   -0.498101   -1.058573
                          H        2.225721    0.232255   -0.392262
                          H        1.273321   -1.346321   -0.027982"""  # C[CH]C <=> [CH2]CC
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
        cls.r_xyz_2c = """C                  1.09338094    0.27531747   -0.57096575
                          C                 -0.04640070    0.39437846    0.45778217
                          C                 -1.28106359   -0.51952520    0.34829392
                          H                  1.14957877   -0.73250497   -0.92598088
                          H                  2.02052140    0.54282736   -0.10863577
                          H                  0.01952328    1.11208766    1.24863458
                          H                  0.90196885    0.93290581   -1.39306002
                          H                 -1.11162616   -1.41634994    0.90677072
                          H                 -2.13694812   -0.01127740    0.74078066
                          H                 -1.45246666   -0.76593325   -0.67874262"""
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
        cls.ts_spc_2 = ARCSpecies(label='TS2', is_ts=True, xyz=cls.ts_xyz_2)
        cls.ts_spc_2.mol_from_xyz()
        cls.reactant_2a = ARCSpecies(label='C[CH]C', smiles='C[CH]C', xyz=cls.r_xyz_2a)
        cls.reactant_2b = ARCSpecies(label='C[CH]C', smiles='C[CH]C', xyz=cls.r_xyz_2b)  # Shuffled order, != ts_xyz_2
        cls.product_2 = ARCSpecies(label='[CH2]CC', smiles='[CH2]CC', xyz=cls.p_xyz_2)
        cls.rxn_2a = ARCReaction(r_species=[cls.reactant_2a], p_species=[cls.product_2])
        cls.rxn_2a.ts_species = cls.ts_spc_2
        cls.rxn_2b = ARCReaction(r_species=[cls.reactant_2b], p_species=[cls.product_2])
        cls.rxn_2b.ts_species = cls.ts_spc_2

        c2h5no2_xyz = {'coords': ((1.8953828083622057, 0.8695975650550358, 0.6461465212661076),
                                  (1.3601473931706598, -0.04212583715410005, 0.0034200061443233247),
                                  (1.8529583069008781, -0.6310931351538215, -0.9666668585141432),
                                  (-0.010154355673379136, -0.4652844276756663, 0.43320585211058743),
                                  (-1.0281604639422022, 0.36855062612122236, -0.3158851121891869),
                                  (-0.11071296591935365, -1.5314728469286516, 0.20909234121344752),
                                  (-0.07635985361458197, -0.31625218083177237, 1.5151037167736001),
                                  (-2.042322710601489, 0.08102183703582924, -0.021667016484293297),
                                  (-0.9033569412063314, 1.436005790671757, -0.10388682333330314),
                                  (-0.937421217476434, 0.23105260886017234, -1.3988626269871478)),
                       'isotopes': (16, 14, 16, 12, 12, 1, 1, 1, 1, 1),
                       'symbols': ('O', 'N', 'O', 'C', 'C', 'H', 'H', 'H', 'H', 'H')}
        c2h5ono_xyz = {'coords': ((-1.3334725178745668, 0.2849178019354427, 0.4149005134933577),
                                  (-0.08765353373275289, 0.24941420749682627, -0.4497882845360618),
                                  (1.0488580188184402, 0.3986394744609146, 0.39515448276833964),
                                  (2.2292240798482883, 0.36629637181188207, -0.4124684043339001),
                                  (3.2413605054484185, 0.4928521621538312, 0.283008378837631),
                                  (-1.3088339518827734, -0.5173661350567303, 1.1597967522753032),
                                  (-2.23462275856269, 0.17332354052924734, -0.19455307765792382),
                                  (-1.393828440234405, 1.2294860794610234, 0.9656140588162426),
                                  (-0.12370667081323389, 1.0672740524773998, -1.1795070012935482),
                                  (-0.037324731014725374, -0.7080479312151163, -0.9821574183694773)),
                       'isotopes': (12, 12, 16, 14, 16, 1, 1, 1, 1, 1),
                       'symbols': ('C', 'C', 'O', 'N', 'O', 'H', 'H', 'H', 'H', 'H')}
        cls.rxn_3 = ARCReaction(r_species=[ARCSpecies(label='C2H5NO2', smiles='[O-][N+](=O)CC', xyz=c2h5no2_xyz)],
                               p_species=[ARCSpecies(label='C2H5ONO', smiles='CCON=O', xyz=c2h5ono_xyz)], )
        cls.ts_3_xyz = check_xyz_dict("""O        0.520045    1.026544   -0.223307
                                         N        0.818877   -0.207900   -0.075436
                                         O        1.964221   -0.523711   -0.014266
                                         C       -0.968581    0.050866    0.695117
                                         C       -1.903603   -0.321292   -0.395596
                                         H       -1.145584    1.019535    1.170709
                                         H       -0.740906   -0.730110    1.427000
                                         H       -1.628826   -1.274421   -0.863423
                                         H       -2.906412   -0.425097    0.055493
                                         H       -1.951439    0.465285   -1.158262""")
        cls.rxn_3.ts_species = ARCSpecies(label='TS3', is_ts=True, xyz=cls.ts_3_xyz)

    def test_analyze_ts_normal_mode_displacement_simple_rxns(self):
        """Test the analyze_ts_normal_mode_displacement() function with simple reactions."""
        # CH4 + OH <=> CH3 + H2O
        self.generic_job.local_path_to_output_file = os.path.join(ARC_PATH, 'arc', 'testing', 'freq', 'TS_CH4_OH.log')
        valid = nmd.analyze_ts_normal_mode_displacement(reaction=self.rxn_1,
                                                        job=self.generic_job,
                                                        amplitude=0.25,
                                                        weights=True)
        self.assertTrue(valid)
        valid = nmd.analyze_ts_normal_mode_displacement(reaction=self.rxn_1,
                                                        job=self.generic_job,
                                                        amplitude=[0, 0.001, 0.5],
                                                        weights=True)
        self.assertTrue(valid)

        # N2H4 + HO2 <=> N2H3 + H2O2
        self.generic_job.local_path_to_output_file = self.freq_log_path_2
        valid = nmd.analyze_ts_normal_mode_displacement(reaction=self.rxn_2, job=self.generic_job, amplitude=0.25)
        self.assertTrue(valid)

        # NH2 + N2H3 <=> NH + N2H4:
        self.generic_job.local_path_to_output_file = os.path.join(ARC_PATH, 'arc', 'testing', 'freq', 'TS_NH2+N2H3.out')
        rxn_3 = ARCReaction(r_species=[ARCSpecies(label='NH2', xyz="""N       0.00000000   -0.00000000    0.14115400
                                                                      H      -0.80516800    0.00000000   -0.49355600
                                                                      H       0.80516800   -0.00000000   -0.49355600"""),
                                       ARCSpecies(label='N2H3', xyz="""N       0.59115400    0.02582600   -0.07080800
                                                                       H       1.01637000    0.90287000    0.19448100
                                                                       H       1.13108000   -0.79351700    0.15181600
                                                                       N      -0.73445800   -0.15245400    0.02565700
                                                                       H      -1.14969800    0.77790100   -0.02540800""")],
                            p_species=[ARCSpecies(label='NH', xyz="""N       0.00000000    0.00000000    0.13025700
                                                                     H       0.00000000    0.00000000   -0.90825700"""),
                                       ARCSpecies(label='N2H4', xyz="""N       0.70348300    0.09755100   -0.07212500
                                                                       N      -0.70348300   -0.09755100   -0.07212500
                                                                       H       1.05603900    0.38865300    0.83168200
                                                                       H      -1.05603900   -0.38865300    0.83168200
                                                                       H       1.14245100   -0.77661300   -0.32127200
                                                                       H      -1.14245100    0.77661300   -0.32127200""")])
        rxn_3.ts_species = ARCSpecies(label='TS3', is_ts=True, xyz="""N      -0.44734500    0.68033000   -0.09191900
                                                                      H      -0.45257300    1.14463200    0.81251500
                                                                      H       0.67532500    0.38185200   -0.23044400
                                                                      N      -1.22777700   -0.47121500   -0.00284000
                                                                      H      -1.81516400   -0.50310400    0.81640600
                                                                      H      -1.78119500   -0.57249600   -0.84071000
                                                                      N       1.91083300   -0.14543600   -0.06636000
                                                                      H       1.73701100   -0.85419700    0.66460600""")
        rxn_3.ts_species.mol_from_xyz()
        rxn_3.ts_species.populate_ts_checks()
        valid = nmd.analyze_ts_normal_mode_displacement(reaction=rxn_3, job=self.generic_job, amplitude=0.25)
        self.assertTrue(valid)

        # [CH2]CC=C <=> CCC=[CH] butylene intra H migration:
        self.generic_job.local_path_to_output_file = os.path.join(ARC_PATH, 'arc', 'testing', 'composite',
                                                                  'TS_butylene_intra_H_migration.out')
        rxn_4 = ARCReaction(r_species=[
            ARCSpecies(label='butylene',
                       xyz={'symbols': ('C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H'),
                            'isotopes': (12, 12, 12, 12, 1, 1, 1, 1, 1, 1, 1),
                            'coords': ((-1.5025309111564664, -0.534274223668814, -0.8036222996901808),
                                       (-0.7174177387201146, -0.023936112728414158, 0.35370258735369786),
                                       (-1.5230462996626752, -0.05695435961481443, 1.6163349692848272),
                                       (-1.8634470313869078, 1.0277715224421244, 2.324841919574016),
                                       (-0.984024423978003, -0.9539130636653048, -1.6577859414775906),
                                       (-2.550807526086091, -0.2789561000296545, -0.9131030981780086),
                                       (-0.3724512697624012, 0.9914237465990766, 0.12894489304781925),
                                       (0.1738420368001901, -0.6466414881716757, 0.48830614688365104),
                                       (-1.8352343593831375, -1.0368501719961523, 1.9724902744574715),
                                       (-1.57401834878684, 2.026695960278519, 2.0137658090390858),
                                       (-2.446426657980167, 0.9347672870076474, 3.235948559430434))})],
                            p_species=[ARCSpecies(label='CCC=[CH]', smiles='CCC=[CH]')])
        rxn_4.ts_species = ARCSpecies(label='TS4', is_ts=True,
                                      xyz="""C                 -1.21222600   -0.64083500    0.00000300
                                             C                 -0.63380200    0.77863500   -0.00000300
                                             C                  0.87097000    0.58302100    0.00000400
                                             C                  1.24629100   -0.68545200   -0.00000300
                                             H                 -1.72740700   -0.95796100    0.90446200
                                             H                 -1.72743700   -0.95796100   -0.90443900
                                             H                 -0.95478600    1.35296500    0.87649200
                                             H                 -0.95477600    1.35295100   -0.87651200
                                             H                  1.55506600    1.42902600    0.00001400
                                             H                  2.20977700   -1.17852900   -0.00000300
                                             H                 -0.02783300   -1.25271100   -0.00001000""")
        rxn_4.ts_species.mol_from_xyz()
        rxn_4.ts_species.populate_ts_checks()
        valid = nmd.analyze_ts_normal_mode_displacement(reaction=rxn_4, job=self.generic_job, amplitude=0.25)
        self.assertTrue(valid)

        # NCC + H <=> CH3CHNH2 + H2:
        self.generic_job.local_path_to_output_file = os.path.join(ARC_PATH, 'arc', 'testing', 'composite',
                                                                  'TS0_composite_2044.out')
        ea_xyz = """N                  1.27511929   -0.21413688   -0.09829069
                    C                  0.04568411    0.51479456    0.24529057
                    C                 -1.17314611   -0.39875221    0.01838707
                    H                  1.35437220   -1.02559828    0.48071654
                    H                  1.24076865   -0.49175940   -1.05836661
                    H                 -0.03911651    1.38305825   -0.37424716
                    H                  0.08243929    0.81185065    1.27257181
                    H                 -1.08834550   -1.26701591    0.63792481
                    H                 -2.06804111    0.13183054    0.26847684
                    H                 -1.20990129   -0.69580830   -1.00889416"""
        ch3chnh2_xyz = {'symbols': ('C', 'C', 'N', 'H', 'H', 'H', 'H', 'H', 'H'), 'isotopes': (12, 12, 14, 1, 1, 1, 1, 1, 1),
                        'coords': ((-1.126885721359211, 0.0525078449047029, 0.16096122288992248),
                                   (0.3110254709748632, 0.43198882113724923, 0.25317384633324647),
                                   (0.8353039548590167, 1.612144179849946, -0.43980881000393546),
                                   (-1.2280793873408098, -0.9063788436936245, -0.3556390245681601),
                                   (-1.5506369749373963, -0.04972561706616586, 1.1642482560565997),
                                   (-1.7098257757667992, 0.801238249703059, -0.3836181889319467),
                                   (0.9893789736981147, -0.22608070229597396, 0.7909844523882721),
                                   (1.6782473720416846, 1.9880253918558228, -0.007820540824533973),
                                   (0.15476721028478654, 2.3655991320212193, -0.527262390251838))}
        ea = ARCSpecies(label='EA', smiles='NCC', xyz=ea_xyz)
        ch3chnh2 = ARCSpecies(label='CH3CHNH2', smiles='C[CH]N', xyz=ch3chnh2_xyz)
        rxn_5 = ARCReaction(r_species=[ea, ARCSpecies(label='H', smiles='[H]')],
                                p_species=[ch3chnh2, ARCSpecies(label='H2', smiles='[H][H]')])
        rxn_5.ts_species = ARCSpecies(label='TS5', is_ts=True, xyz=self.generic_job.local_path_to_output_file)
        rxn_5.ts_species.populate_ts_checks()
        valid = nmd.analyze_ts_normal_mode_displacement(reaction=rxn_5, job=self.generic_job, amplitude=0.25)
        self.assertTrue(valid)

        # NH3 + H <=> NH2 + H2
        self.generic_job.local_path_to_output_file = os.path.join(ARC_PATH, 'arc', 'testing', 'freq', 'TS_NH3+H=NH2+H2.out')

        rxn_6 = ARCReaction(r_species=[ARCSpecies(label='NH3', smiles='N'), ARCSpecies(label='H', smiles='[H]')],
                            p_species=[ARCSpecies(label='NH2', smiles='[NH2]'), ARCSpecies(label='H2', smiles='[H][H]')])
        rxn_6.ts_species = ARCSpecies(label='TS3', is_ts=True, xyz=self.generic_job.local_path_to_output_file)
        rxn_6.ts_species.populate_ts_checks()
        valid = nmd.analyze_ts_normal_mode_displacement(reaction=rxn_6, job=self.generic_job, amplitude=1)
        self.assertTrue(valid)

    def test_analyze_ts_normal_mode_displacement_correct_and_incorrect_data(self):
        """Test the analyze_ts_normal_mode_displacement() function with correct and incorrect TSs for iC3H7 <=> nC3H7."""
        base_path = os.path.join(ARC_PATH, 'arc', 'testing', 'composite', 'C3H7')
        log_file_paths = {'iC3H7': os.path.join(base_path, 'iC3H7.gjf'),
                          'nC3H7': os.path.join(base_path, 'nC3H7.log'),
                          'TS1': os.path.join(base_path, 'TS1.log'),  # an iC3H7-like saddle point
                          'TS2': os.path.join(base_path, 'TS2.log'),  # an nC3H7-like saddle point
                          'TS3': os.path.join(base_path, 'TS3.log'),  # the correct TS for iC3H7 <=> nC3H7
                          'TS4': os.path.join(base_path, 'TS4.log'),  # a structure with one H far away
                          'TS5': os.path.join(base_path, 'TS5.log'),  # a TS for a wrong reaction of nC3H7 <=> nC3H7 (shifting sides in a 4 member ring TS)
                          'TS6': os.path.join(base_path, 'TS6.log'),  # a TS for another rxn on this PES: CH3CH=CH2 + H <=> CH3CH=CH + H2
                          'TS7': os.path.join(base_path, 'TS7.log'),  # a strange zwitterion saddle point with 4 H's on one side
        }

        amplitude = 0.25

        rxn = ARCReaction(r_species=[ARCSpecies(label='iC3H7', smiles='C[CH]C', xyz=log_file_paths['iC3H7'])],
                          p_species=[ARCSpecies(label='nC3H7', smiles='[CH2]CC', xyz=log_file_paths['nC3H7'])])

        self.generic_job.local_path_to_output_file = log_file_paths['TS1']
        rxn.ts_species = ARCSpecies(label='TS', is_ts=True, xyz=log_file_paths['TS1'])
        valid = nmd.analyze_ts_normal_mode_displacement(reaction=rxn, job=self.generic_job, amplitude=amplitude)
        self.assertFalse(valid)

        self.generic_job.local_path_to_output_file = log_file_paths['TS2']
        rxn.ts_species = ARCSpecies(label='TS', is_ts=True, xyz=log_file_paths['TS2'])
        valid = nmd.analyze_ts_normal_mode_displacement(reaction=rxn, job=self.generic_job, amplitude=amplitude)
        self.assertFalse(valid)

        self.generic_job.local_path_to_output_file = log_file_paths['TS3']
        rxn.ts_species = ARCSpecies(label='TS', is_ts=True, xyz=log_file_paths['TS3'])
        valid = nmd.analyze_ts_normal_mode_displacement(reaction=rxn, job=self.generic_job, amplitude=amplitude)
        self.assertTrue(valid)  # the correct TS

        self.generic_job.local_path_to_output_file = log_file_paths['TS4']
        rxn.ts_species = ARCSpecies(label='TS', is_ts=True, xyz=log_file_paths['TS4'])
        valid = nmd.analyze_ts_normal_mode_displacement(reaction=rxn, job=self.generic_job, amplitude=amplitude)
        self.assertFalse(valid)

        self.generic_job.local_path_to_output_file = log_file_paths['TS5']
        rxn.ts_species = ARCSpecies(label='TS', is_ts=True, xyz=log_file_paths['TS5'])
        valid = nmd.analyze_ts_normal_mode_displacement(reaction=rxn, job=self.generic_job, amplitude=amplitude)
        self.assertFalse(valid)

        self.generic_job.local_path_to_output_file = log_file_paths['TS6']
        rxn.ts_species = ARCSpecies(label='TS', is_ts=True, xyz=log_file_paths['TS6'])
        valid = nmd.analyze_ts_normal_mode_displacement(reaction=rxn, job=self.generic_job, amplitude=amplitude)
        self.assertFalse(valid)

        self.generic_job.local_path_to_output_file = log_file_paths['TS7']
        rxn.ts_species = ARCSpecies(label='TS', is_ts=True, xyz=log_file_paths['TS7'])
        valid = nmd.analyze_ts_normal_mode_displacement(reaction=rxn, job=self.generic_job, amplitude=amplitude)
        self.assertFalse(valid)

    def test_analyze_ts_normal_mode_displacement_for_hypervalence_nitrogen(self):
        """Test the analyze_ts_normal_mode_displacement() function for a hypervalence nitrogen."""
        # C2H5NO2 <=> C2H5ONO
        self.generic_job.local_path_to_output_file = os.path.join(ARC_PATH, 'arc', 'testing', 'composite', 'C2H5NO2__C2H5ONO.out')
        valid = nmd.analyze_ts_normal_mode_displacement(reaction=self.rxn_3,
                                                        job=self.generic_job,
                                                        amplitude=0.25,
                                                        weights=True)
        self.assertTrue(valid)

    def test_translate_all_tuples_simultaneously(self):
        """Test the translate_all_tuples_simultaneously() function."""
        translated_tuples = nmd.translate_all_tuples_simultaneously(list_1=[(0, 1)],
                                                                    list_2=[],
                                                                    list_3=[],
                                                                    equivalences=[])
        self.assertEqual(translated_tuples, [([(0, 1)], [], [])])

        translated_tuples = nmd.translate_all_tuples_simultaneously(list_1=[(0, 1)],
                                                                    list_2=[],
                                                                    list_3=[],
                                                                    equivalences=[[0, 3, 4]])
        self.assertEqual(translated_tuples, [([(0, 1)], [], []), ([(3, 1)], [], []), ([(4, 1)], [], [])])

        translated_tuples = nmd.translate_all_tuples_simultaneously(list_1=[(0, 1)],
                                                                    list_2=[(1, 5)],
                                                                    list_3=[],
                                                                    equivalences=[[1, 2, 3, 4]])
        self.assertEqual(translated_tuples, [([(0, 1)], [(1, 5)], []),
                                             ([(0, 2)], [(2, 5)], []),
                                             ([(0, 3)], [(3, 5)], []),
                                             ([(0, 4)], [(4, 5)], [])])

        translated_tuples = nmd.translate_all_tuples_simultaneously(list_1=[(0, 1)],
                                                                    list_2=[(1, 5)],
                                                                    list_3=[(2, 7)],
                                                                    equivalences=[[1, 2, 3, 4], [5, 6]])
        self.assertEqual(translated_tuples, [([(0, 1)], [(1, 5)], [(1, 7)]),
                                             ([(0, 1)], [(1, 6)], [(1, 7)]),
                                             ([(0, 1)], [(1, 5)], [(2, 7)]),
                                             ([(0, 1)], [(1, 6)], [(2, 7)]),
                                             ([(0, 1)], [(1, 5)], [(3, 7)]),
                                             ([(0, 1)], [(1, 6)], [(3, 7)]),
                                             ([(0, 1)], [(1, 5)], [(4, 7)]),
                                             ([(0, 1)], [(1, 6)], [(4, 7)]),
                                             ([(0, 2)], [(2, 5)], [(1, 7)]),
                                             ([(0, 2)], [(2, 6)], [(1, 7)]),
                                             ([(0, 2)], [(2, 5)], [(2, 7)]),
                                             ([(0, 2)], [(2, 6)], [(2, 7)]),
                                             ([(0, 2)], [(2, 5)], [(3, 7)]),
                                             ([(0, 2)], [(2, 6)], [(3, 7)]),
                                             ([(0, 2)], [(2, 5)], [(4, 7)]),
                                             ([(0, 2)], [(2, 6)], [(4, 7)]),
                                             ([(0, 3)], [(3, 5)], [(1, 7)]),
                                             ([(0, 3)], [(3, 6)], [(1, 7)]),
                                             ([(0, 3)], [(3, 5)], [(2, 7)]),
                                             ([(0, 3)], [(3, 6)], [(2, 7)]),
                                             ([(0, 3)], [(3, 5)], [(3, 7)]),
                                             ([(0, 3)], [(3, 6)], [(3, 7)]),
                                             ([(0, 3)], [(3, 5)], [(4, 7)]),
                                             ([(0, 3)], [(3, 6)], [(4, 7)]),
                                             ([(0, 4)], [(4, 5)], [(1, 7)]),
                                             ([(0, 4)], [(4, 6)], [(1, 7)]),
                                             ([(0, 4)], [(4, 5)], [(2, 7)]),
                                             ([(0, 4)], [(4, 6)], [(2, 7)]),
                                             ([(0, 4)], [(4, 5)], [(3, 7)]),
                                             ([(0, 4)], [(4, 6)], [(3, 7)]),
                                             ([(0, 4)], [(4, 5)], [(4, 7)]),
                                             ([(0, 4)], [(4, 6)], [(4, 7)])])

        translated_tuples = nmd.translate_all_tuples_simultaneously(list_1=[(0, 1), (0, 2)],
                                                                    list_2=[(1, 5)],
                                                                    list_3=[(2, 7)],
                                                                    equivalences=[[1, 2, 3, 4], [5, 6]])
        self.assertEqual(translated_tuples, [([(0, 1), (0, 2)], [(1, 5)], [(2, 7)]),
                                             ([(0, 1), (0, 2)], [(1, 6)], [(2, 7)]),
                                             ([(0, 1), (0, 3)], [(1, 5)], [(3, 7)]),
                                             ([(0, 1), (0, 3)], [(1, 6)], [(3, 7)]),
                                             ([(0, 1), (0, 4)], [(1, 5)], [(4, 7)]),
                                             ([(0, 1), (0, 4)], [(1, 6)], [(4, 7)]),
                                             ([(0, 2), (0, 1)], [(2, 5)], [(1, 7)]),
                                             ([(0, 2), (0, 1)], [(2, 6)], [(1, 7)]),
                                             ([(0, 2), (0, 3)], [(2, 5)], [(3, 7)]),
                                             ([(0, 2), (0, 3)], [(2, 6)], [(3, 7)]),
                                             ([(0, 2), (0, 4)], [(2, 5)], [(4, 7)]),
                                             ([(0, 2), (0, 4)], [(2, 6)], [(4, 7)]),
                                             ([(0, 3), (0, 1)], [(3, 5)], [(1, 7)]),
                                             ([(0, 3), (0, 1)], [(3, 6)], [(1, 7)]),
                                             ([(0, 3), (0, 2)], [(3, 5)], [(2, 7)]),
                                             ([(0, 3), (0, 2)], [(3, 6)], [(2, 7)]),
                                             ([(0, 3), (0, 4)], [(3, 5)], [(4, 7)]),
                                             ([(0, 3), (0, 4)], [(3, 6)], [(4, 7)]),
                                             ([(0, 4), (0, 1)], [(4, 5)], [(1, 7)]),
                                             ([(0, 4), (0, 1)], [(4, 6)], [(1, 7)]),
                                             ([(0, 4), (0, 2)], [(4, 5)], [(2, 7)]),
                                             ([(0, 4), (0, 2)], [(4, 6)], [(2, 7)]),
                                             ([(0, 4), (0, 3)], [(4, 5)], [(3, 7)]),
                                             ([(0, 4), (0, 3)], [(4, 6)], [(3, 7)])])

    def test_create_equivalence_mapping(self):
        """Test the create_equivalence_mapping() function."""
        mapping = nmd.create_equivalence_mapping(equivalences=[[1, 2, 3, 4]])
        self.assertEqual(mapping, {1: [1, 2, 3, 4], 2: [1, 2, 3, 4], 3: [1, 2, 3, 4], 4: [1, 2, 3, 4]})
        mapping = nmd.create_equivalence_mapping(equivalences=[[1, 2, 3, 4], [5, 6]])
        self.assertEqual(mapping, {1: [1, 2, 3, 4],
                                   2: [1, 2, 3, 4],
                                   3: [1, 2, 3, 4],
                                   4: [1, 2, 3, 4],
                                   5: [5, 6],
                                   6: [5, 6]})

    def test_get_weights_from_xyz(self):
        """Test the get_weights_from_xyz() function."""
        weights = nmd.get_weights_from_xyz(xyz=self.xyz_1, weights=False)
        np.testing.assert_array_equal(weights, np.array([[1], [1], [1], [1], [1], [1]]))

        weights = nmd.get_weights_from_xyz(xyz=self.xyz_1, weights=True)
        np.testing.assert_almost_equal(weights, self.weights_1)

        weights = nmd.get_weights_from_xyz(xyz=self.ts_2_xyz, weights=True)
        expected_w = np.array([[3.99936428], [3.99936428], [1.00390489], [3.74206815], [3.74206815],
                               [1.00390489], [1.00390489], [1.00390489], [1.00390489]])
        np.testing.assert_almost_equal(weights, expected_w, decimal=7)

        w_array = np.array([[10], [1], [1], [3], [1], [1]])
        weights = nmd.get_weights_from_xyz(xyz=self.xyz_1, weights=w_array)
        np.testing.assert_equal(weights, w_array)

    def test_get_bond_length_changes_baseline_and_std(self):
        """Test the get_bond_length_changes_baseline_and_std() function."""
        weights = nmd.get_weights_from_xyz(xyz=self.ts_2_xyz, weights=False)
        freqs, normal_mode_disp = parse_normal_mode_displacement(path=self.freq_log_path_2)
        np.testing.assert_array_almost_equal(freqs[0], -1350.1119)
        xyzs = nmd.get_displaced_xyzs(xyz=self.ts_2_xyz,
                                      amplitude=0.25,
                                      normal_mode_disp=normal_mode_disp[0],
                                      weights=weights,
                                      )
        baseline_max, std = nmd.get_bond_length_changes_baseline_and_std(
            non_reactive_bonds=[(0, 1), (1, 2), (4, 7), (4, 8), (3, 6), (3, 4)],
            xyzs=xyzs)
        self.assertAlmostEqual(baseline_max, 0.0913286, 4)
        self.assertAlmostEqual(std, 0.1657833, 4)

    def test_get_bond_length_in_reaction(self):
        """Test the get_bond_length_in_reaction() function."""
        bond_length = nmd.get_bond_length_in_reaction(bond=(0, 1), xyz=self.rxn_1.r_species[0].get_xyz())
        self.assertAlmostEqual(bond_length, 1.0922, 4)
        bond_length = nmd.get_bond_length_in_reaction(bond=(0, 1), xyz=self.rxn_1.r_species[1].get_xyz())
        self.assertAlmostEqual(bond_length, 0.9778, 4)

    def test_classic_intra_h_migration_through_all_major_functions(self):
        """Test the intermediate stages the nmd module takes for processing the classic intra H migration reaction."""
        xyz_path = os.path.join(ARC_PATH, 'arc', 'testing', 'composite', 'C3H7', 'TS3.log')
        standard_ts_orientation_xyz = check_xyz_dict(xyz_path)
        freqs, nmd_array = parse_normal_mode_displacement(xyz_path)
        self.assertEqual(freqs[0], -1927.8629)
        self.assertTrue(np.allclose(nmd_array[0], np.array([[0., 0., -0.],
                                                            [0., -0., 0.01],
                                                            [-0.02, -0.02, -0.],
                                                            [-0., -0., -0.],
                                                            [-0.04, 0.01, -0.06],
                                                            [-0.01, 0.03, 0.03],
                                                            [-0.02, 0.03, 0.07],
                                                            [-0.02, 0.02, -0.03],
                                                            [-0.03, -0., -0.03],
                                                            [0.85, -0.51, -0.06]])))
        weights = nmd.get_weights_from_xyz(standard_ts_orientation_xyz)
        self.assertTrue(np.allclose(weights, np.array([[3.46410162],
                                                       [1.00390489],
                                                       [1.00390489],
                                                       [1.00390489],
                                                       [3.46410162],
                                                       [1.00390489],
                                                       [3.46410162],
                                                       [1.00390489],
                                                       [1.00390489],
                                                       [1.00390489]])))
        xyzs = nmd.get_displaced_xyzs(xyz=standard_ts_orientation_xyz,
                                      amplitude=0.50,
                                      normal_mode_disp=nmd_array[0],
                                      weights=weights,
                                      )
        expected_xyz_0 = {'symbols': ('C', 'H', 'H', 'H', 'C', 'H', 'C', 'H', 'H', 'H'),
                          'isotopes': (12, 1, 1, 1, 12, 1, 12, 1, 1, 1),
                          'coords': ((-1.279906, -0.191149, -0.024558), (-1.269506, -1.096592, 0.5866444755398793),
                                     (-1.5148619510797587, -0.5003619510797586, -1.049451),
                                     (-2.096169, 0.442456, 0.330967),
                                     (0.1099180323027551, 0.49975249192431126, 0.12895004845413263),
                                     (0.0968305244601207, 1.541163426619638, -0.2957935733803621),
                                     (1.3528900161513775, -0.3071185242270663, -0.1597255565298214),
                                     (1.2893590489202413, -1.3465650489202414, -0.0030494266196379027),
                                     (2.237492573380362, 0.228642, -0.3672204266196379),
                                     (0.3964784208897406, 0.5475917474661557, 1.0661361467607242))}
        expected_xyz_1 = {'symbols': ('C', 'H', 'H', 'H', 'C', 'H', 'C', 'H', 'H', 'H'),
                          'isotopes': (12, 1, 1, 1, 12, 1, 12, 1, 1, 1),
                          'coords': ((-1.279906, -0.191149, -0.024558), (-1.269506, -1.096592, 0.5966835244601206),
                                     (-1.5349400489202414, -0.5204400489202414, -1.049451),
                                     (-2.096169, 0.442456, 0.330967),
                                     (-0.028646032302755094, 0.5343935080756888, -0.07889604845413262),
                                     (0.0867914755398793, 1.571280573380362, -0.2656764266196379),
                                     (1.2836079838486225, -0.2031954757729337, 0.08276155652982141),
                                     (1.2692809510797587, -1.3264869510797588, -0.033166573380362094),
                                     (2.2073754266196377, 0.228642, -0.3973375733803621),
                                     (1.2497975791102593, 0.035600252533844357, 1.0059018532392758))}
        self.assertEqual(xyzs[0], expected_xyz_0)
        self.assertEqual(xyzs[1], expected_xyz_1)
        reactive_bonds = [(4, 9), (6, 9)]
        reactive_bonds_diffs, report = nmd.get_bond_length_changes(bonds=reactive_bonds,
                                                                   xyzs=xyzs,
                                                                   weights=weights,
                                                                   amplitude=0.50,
                                                                   return_none_if_change_is_insignificant=True,
                                                                   considered_reacttive=True,
                                                                   )
        self.assertTrue(np.allclose(reactive_bonds_diffs, np.array([[1.43238515], [1.52941596]])))

        rxn = ARCReaction(r_species=[ARCSpecies(label='iC3H7', smiles='C[CH]C',
                                                xyz=os.path.join(ARC_PATH, 'arc', 'testing', 'composite', 'C3H7', 'iC3H7.gjf'))],
                          p_species=[ARCSpecies(label='nC3H7', smiles='[CH2]CC',
                                                xyz=os.path.join(ARC_PATH, 'arc', 'testing', 'composite', 'C3H7', 'nC3H7.log'))])
        rxn.ts_species = ARCSpecies(label='TS', is_ts=True, xyz=xyz_path)
        r_bonds, _ = rxn.get_bonds(r_bonds_only=True)
        non_reactive_bonds = list()
        for bond in r_bonds:
            if bond not in reactive_bonds:
                non_reactive_bonds.append(bond)
        baseline, std = nmd.get_bond_length_changes_baseline_and_std(non_reactive_bonds=non_reactive_bonds,
                                                                     xyzs=xyzs,
                                                                     weights=weights,
                                                                     )
        self.assertAlmostEqual(baseline[0], 0.10390012)
        self.assertAlmostEqual(std, 0.1217205)
        min_reactive_bond_diff = np.min(reactive_bonds_diffs)
        sigma = (min_reactive_bond_diff - baseline) / std
        self.assertAlmostEqual(float(sigma), 10.914223252)

    def test_get_displaced_xyzs(self):
        """Test the get_displaced_xyzs() function."""
        xyzs = nmd.get_displaced_xyzs(xyz=self.xyz_1,
                                      amplitude=0.25,
                                      normal_mode_disp=self.nmd_1,
                                      weights=self.weights_1,
                                      )
        self.assertEqual(xyzs[0], self.displaced_xyz_3[0])
        self.assertEqual(xyzs[1], self.displaced_xyz_3[1])

        normal_mode_disp = np.array([[-0.06, 0.04, 0.01], [-0., -0.03, -0.01], [0.08, 0., 0.02],
                                     [-0.02, 0.04, -0.02], [-0., -0.03, 0.01], [0.14, -0.05, 0.06],
                                     [0.04, -0.07, 0.01], [0.96, -0.16, 0.02], [0.04, -0.06, -0.01]])
        weights = np.array([[1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.]])
        xyzs = nmd.get_displaced_xyzs(xyz=self.ts_2_xyz,
                                      amplitude=0.3,
                                      normal_mode_disp=normal_mode_disp,
                                      weights=weights,
                                      )
        expected_xyzs = ({'symbols': ('O', 'O', 'H', 'N', 'N', 'H', 'H', 'H', 'H'),
                          'isotopes': (16, 16, 1, 14, 14, 1, 1, 1, 1),
                          'coords': ((-1.257006, -0.668458, -0.220271), (-1.463976, 0.65901, 0.247741),
                                     (-1.263724, 1.193859, -0.530436), (1.462296, 0.575219, -0.113103),
                                     (1.134258, -0.740491, 0.078665), (1.9110719999999999, 0.730591, -1.010228),
                                     (1.914641, 1.061462, 0.650661), (-0.29245499999999996, -0.800273, -0.189117),
                                     (1.130454, -0.896148, 1.0914249999999999))},
                         {'symbols': ('O', 'O', 'H', 'N', 'N', 'H', 'H', 'H', 'H'),
                          'isotopes': (16, 16, 1, 14, 14, 1, 1, 1, 1),
                          'coords': ((-1.293006, -0.644458, -0.214271), (-1.463976, 0.64101, 0.24174099999999998),
                                     (-1.215724, 1.193859, -0.518436), (1.450296, 0.5992190000000001, -0.125103),
                                     (1.134258, -0.758491, 0.084665), (1.995072, 0.700591, -0.974228),
                                     (1.938641, 1.019462, 0.656661), (0.283545, -0.8962730000000001, -0.177117),
                                     (1.154454, -0.932148, 1.085425))})
        self.assertEqual(xyzs[0], expected_xyzs[0])
        self.assertEqual(xyzs[1], expected_xyzs[1])

    def test_find_equivalent_atoms(self):
        """Test the find_equivalent_atoms() function."""
        r_eq_atoms, p_eq_atoms = nmd.find_equivalent_atoms(reaction=self.rxn_1, reactant_only=False)
        self.assertEqual(r_eq_atoms, [[1, 2, 3, 4]])
        self.assertEqual(p_eq_atoms, [[2, 3, 4], [6, 1]])

        r_eq_atoms, p_eq_atoms = nmd.find_equivalent_atoms(reaction=self.rxn_1, reactant_only=True)
        self.assertEqual(r_eq_atoms, [[1, 2, 3, 4]])
        self.assertEqual(p_eq_atoms, [])

        r_eq_atoms, _ = nmd.find_equivalent_atoms(reaction=self.rxn_2, reactant_only=True)
        self.assertEqual(r_eq_atoms, [[3, 4], [5, 6, 7, 8]])

    def test_identify_equivalent_atoms_in_molecule(self):
        """Test the identify_equivalent_atoms_in_molecule() function."""
        equivalent_atoms = nmd.identify_equivalent_atoms_in_molecule(molecule=Molecule(smiles='C'))
        self.assertEqual(equivalent_atoms, [[1, 2, 3, 4]])

        equivalent_atoms = nmd.identify_equivalent_atoms_in_molecule(molecule=Molecule(smiles='OO'))
        self.assertEqual(equivalent_atoms, [[0, 1], [2, 3]])

        equivalent_atoms = nmd.identify_equivalent_atoms_in_molecule(molecule=Molecule(smiles='CC'))
        self.assertEqual(equivalent_atoms, [[0, 1], [2, 3, 4, 5, 6, 7]])

        equivalent_atoms = nmd.identify_equivalent_atoms_in_molecule(molecule=Molecule(smiles='NCC(CN)CN'))
        self.assertEqual(equivalent_atoms, [[0, 4, 6], [1, 3, 5], [7, 8, 14, 15, 18, 19], [9, 10, 12, 13, 16, 17]])

        equivalent_atoms = nmd.identify_equivalent_atoms_in_molecule(molecule=Molecule(smiles='OO'), atom_map=[3, 2, 1, 0])
        self.assertEqual(equivalent_atoms, [[3, 2], [1, 0]])

    def test_fingerprint_atom(self):
        """Test the fingerprint_atom() function."""
        fp = nmd.fingerprint_atom(atom_index=0, molecule=Molecule(smiles='[H]'))
        self.assertEqual(fp, [1])

        fp = nmd.fingerprint_atom(atom_index=0, molecule=Molecule(smiles='[H][H]'))
        self.assertEqual(fp, [1, [1]])

        fp = nmd.fingerprint_atom(atom_index=0, molecule=ARCSpecies(label='CH4', smiles='C', xyz=self.ch4_xyz).mol)
        self.assertEqual(fp, [6, [1], [1], [1], [1]])

        fp = nmd.fingerprint_atom(atom_index=1, molecule=ARCSpecies(label='CH4', smiles='C', xyz=self.ch4_xyz).mol)
        self.assertEqual(fp, [1, [6, [1], [1], [1]]])

        fp = nmd.fingerprint_atom(atom_index=0, molecule=ARCSpecies(label='H2O2', smiles='OO', xyz=self.h2o2_xyz).mol)
        self.assertEqual(fp, [8, [1], [8, [1]]])

        fp = nmd.fingerprint_atom(atom_index=0, molecule=Molecule(smiles='CCCCCCCCCCCC'))
        self.assertEqual(fp, [6, [1], [1], [1], [6, [1], [1], [6, [1], [1], [6]]]])

        fp = nmd.fingerprint_atom(atom_index=1, molecule=Molecule(smiles='CCCCCCCCCCCC'))
        self.assertEqual(fp, [6, [1], [1], [6, [1], [1], [1]], [6, [1], [1], [6, [1], [1], [6]]]])

        fp = nmd.fingerprint_atom(atom_index=15, molecule=Molecule(smiles='CCCCCCCCCCCC'))
        self.assertEqual(fp, [1, [6, [1], [6, [1], [1], [1]], [6, [1], [1], [6]]]])

        c2h5no2_xyz = """O                  0.62193295    1.59121319   -0.58381518
                         N                  0.43574593    0.41740669    0.07732982
                         O                  1.34135576   -0.35713755    0.18815532
                         C                 -0.87783860    0.10001361    0.65582554
                         C                 -1.73002357   -0.64880063   -0.38564362
                         H                 -1.37248469    1.00642547    0.93625873
                         H                 -0.74723653   -0.51714586    1.52009245
                         H                 -1.23537748   -1.55521250   -0.66607681
                         H                 -2.68617014   -0.87982825    0.03543830
                         H                 -1.86062564   -0.03164117   -1.24991054"""
        c2h5no2_mol = ARCSpecies(label='S', smiles='CC[N+](=O)[O-]', xyz=c2h5no2_xyz).mol
        fp = nmd.fingerprint_atom(atom_index=0, molecule=c2h5no2_mol)
        self.assertEqual(fp, [8, [7, [6, [1], [1], [6]], [8]]])
        fp = nmd.fingerprint_atom(atom_index=2, molecule=c2h5no2_mol)
        self.assertEqual(fp, [8, [7, [6, [1], [1], [6]], [8]]])
        fp = nmd.fingerprint_atom(atom_index=2, molecule=c2h5no2_mol, depth=0)
        self.assertEqual(fp, [8])
        fp = nmd.fingerprint_atom(atom_index=2, molecule=c2h5no2_mol, depth=1)
        self.assertEqual(fp, [8, [7]])
        fp = nmd.fingerprint_atom(atom_index=2, molecule=c2h5no2_mol, depth=5)
        self.assertEqual(fp, [8, [7, [6, [1], [1], [6, [1], [1], [1]]], [8]]])

    @classmethod
    def tearDownClass(cls):
        """
        A function that is run ONCE after all unit tests in this class.
        Delete all project directories created during these unit tests
        """
        projects = ['tmp_nmd_project']
        for project in projects:
            project_directory = os.path.join(ARC_PATH, 'Projects', project)
            shutil.rmtree(project_directory, ignore_errors=True)
        file_paths = [os.path.join(ARC_PATH, 'arc', 'checks', 'nul'), os.path.join(ARC_PATH, 'arc', 'checks', 'run.out')]
        for file_path in file_paths:
            if os.path.isfile(file_path):
                os.remove(file_path)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
