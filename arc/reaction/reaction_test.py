#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.reaction.reaction module
"""

from itertools import permutations
import os
import shutil
import time
import unittest

from arc.common import ARC_PATH, almost_equal_lists, read_yaml_file
from arc.exceptions import ReactionError
from arc.main import ARC
from arc.reaction.reaction import ARCReaction, remove_dup_species
from arc.scheduler import Scheduler
from arc.species import ARCSpecies
from arc.mapping.engine import check_atom_map


class TestARCReaction(unittest.TestCase):
    """
    Contains unit tests for the ARCSpecies class
    """

    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None

        cls.h2_xyz = {'coords': ((0, 0, 0.3736550), (0, 0, -0.3736550)), 'isotopes': (1, 1), 'symbols': ('H', 'H')}
        cls.ho2_xyz = {'coords': ((0.0558910, -0.6204870, 0.0000000),
                                  (0.0558910, 0.7272050, 0.0000000),
                                  (-0.8942590, -0.8537420, 0.0000000)),
                       'isotopes': (16, 16, 1), 'symbols': ('O', 'O', 'H')}
        cls.nh_xyz = {'coords': ((0.509499983131626, 0.0, 0.0), (-0.509499983131626, 0.0, 0.0)),
                      'isotopes': (14, 1), 'symbols': ('N', 'H')}
        cls.ch4_xyz = """C      -0.00000000    0.00000000    0.00000000
                         H      -0.65055201   -0.77428020   -0.41251879
                         H      -0.34927558    0.98159583   -0.32768232
                         H      -0.02233792   -0.04887375    1.09087665
                         H       1.02216551   -0.15844188   -0.35067554"""
        cls.oh_xyz = """O       0.48890387    0.00000000    0.00000000
                        H      -0.48890387    0.00000000    0.00000000"""
        cls.ch3_xyz = """C       0.00000000    0.00000001   -0.00000000
                         H       1.06690511   -0.17519582    0.05416493
                         H      -0.68531716   -0.83753536   -0.02808565
                         H      -0.38158795    1.01273118   -0.02607927"""
        cls.h2o_xyz = """O      -0.00032832    0.39781490    0.00000000
                         H      -0.76330345   -0.19953755    0.00000000
                         H       0.76363177   -0.19827735    0.00000000"""
        cls.ch3chch3_xyz = """C                  0.50180491   -0.93942231   -0.57086745
                              C                  0.01278145    0.13148427    0.42191407
                              C                 -0.86874485    1.29377369   -0.07163907
                              H                  0.28549447    0.06799101    1.45462711
                              H                  1.44553946   -1.32386345   -0.24456986
                              H                  0.61096295   -0.50262210   -1.54153222
                              H                 -0.24653265    2.11136864   -0.37045418
                              H                 -0.21131163   -1.73585284   -0.61629002
                              H                 -1.51770930    1.60958621    0.71830245
                              H                 -1.45448167    0.96793094   -0.90568876"""
        cls.ch3ch2ch2_xyz = """C                  0.48818717   -0.94549701   -0.55196729
                               C                  0.35993708    0.29146456    0.35637075
                               C                 -0.91834764    1.06777042   -0.01096751
                               H                  0.30640232   -0.02058840    1.37845537
                               H                  1.37634603   -1.48487836   -0.29673876
                               H                  0.54172192   -0.63344406   -1.57405191
                               H                  1.21252186    0.92358349    0.22063264
                               H                 -0.36439762   -1.57761595   -0.41622918
                               H                 -1.43807526    1.62776079    0.73816131
                               H                 -1.28677889    1.04716138   -1.01532486"""
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
        cls.ch2choh_xyz = """C      -0.80601307   -0.11773769    0.32792128
                             C       0.23096883    0.47536513   -0.26437348
                             O       1.44620485   -0.11266560   -0.46339257
                             H      -1.74308628    0.41660480    0.45016601
                             H      -0.75733964   -1.13345488    0.70278513
                             H       0.21145717    1.48838416   -0.64841675
                             H       1.41780836   -1.01649567   -0.10468897"""
        cls.ch3cho_xyz = """C      -0.64851652   -0.03628781   -0.04007233
                            C       0.84413281    0.04088405    0.05352862
                            O       1.47323666   -0.23917853    1.06850992
                            H      -1.06033881    0.94648764   -0.28238370
                            H      -0.92134271   -0.74783968   -0.82281679
                            H      -1.04996634   -0.37234114    0.91874740
                            H       1.36260637    0.37153887   -0.86221771"""
        cls.rxn1 = ARCReaction(reactants=['CH4', 'OH'], products=['CH3', 'H2O'],
                               r_species=[ARCSpecies(label='CH4', smiles='C'),
                                          ARCSpecies(label='OH', smiles='[OH]')],
                               p_species=[ARCSpecies(label='CH3', smiles='[CH3]'),
                                          ARCSpecies(label='H2O', smiles='O')])
        cls.rxn_1_w_xyz = ARCReaction(r_species=[ARCSpecies(label='CH4', smiles='C', xyz=cls.ch4_xyz),
                                                 ARCSpecies(label='OH', smiles='[OH]', xyz=cls.oh_xyz)],
                                      p_species=[ARCSpecies(label='CH3', smiles='[CH3]', xyz=cls.ch3_xyz),
                                                 ARCSpecies(label='H2O', smiles='O', xyz=cls.h2o_xyz)])
        cls.rxn2 = ARCReaction(reactants=['C2H5', 'OH'], products=['C2H4', 'H2O'],
                               r_species=[ARCSpecies(label='C2H5', smiles='C[CH2]'),
                                          ARCSpecies(label='OH', smiles='[OH]')],
                               p_species=[ARCSpecies(label='C2H4', smiles='C=C'),
                                          ARCSpecies(label='H2O', smiles='O')])
        cls.rxn3 = ARCReaction(reactants=['CH3CH2NH'], products=['CH2CH2NH2'],
                               r_species=[ARCSpecies(label='CH3CH2NH', smiles='CC[NH]')],
                               p_species=[ARCSpecies(label='CH2CH2NH2', smiles='[CH2]CN')])
        cls.rxn4 = ARCReaction(reactants=['NH2', 'NH2NH'], products=['N', 'NH2N'],
                               r_species=[ARCSpecies(label='NH2', smiles='[NH2]'),
                                          ARCSpecies(label='NH2NH', smiles='N[NH]')],
                               p_species=[ARCSpecies(label='N', smiles='N'),
                                          ARCSpecies(label='NH2N', smiles='N[N]')])
        cls.rxn5 = ARCReaction(reactants=['NH2', 'NH2'], products=['NH', 'NH3'],
                               r_species=[ARCSpecies(label='NH2', smiles='[NH2]')],
                               p_species=[ARCSpecies(label='NH', smiles='[NH]'),
                                          ARCSpecies(label='NH3', smiles='N')])
        cls.rxn6 = ARCReaction(reactants=['NH2', 'N2H3'], products=['NH3', 'H2NN(S)'],
                               r_species=[ARCSpecies(label='NH2', smiles='[NH2]'),
                                          ARCSpecies(label='N2H3', smiles='N[NH]')],
                               p_species=[ARCSpecies(label='NH3', smiles='N'),
                                          ARCSpecies(label='H2NN(S)',
                                                     adjlist="""multiplicity 1
1 N u0 p0 c+1 {2,D} {3,S} {4,S}
2 N u0 p2 c-1 {1,D}
3 H u0 p0 c0 {1,S}
4 H u0 p0 c0 {1,S}""")])
        cls.rxn7 = ARCReaction(reactants=['NH2', 'N2H3'], products=['NH3', 'H2NN(T)'],
                               r_species=[ARCSpecies(label='NH2', smiles='[NH2]'),
                                          ARCSpecies(label='N2H3', smiles='N[NH]')],
                               p_species=[ARCSpecies(label='NH3', smiles='N'),
                                          ARCSpecies(label='H2NN(T)',
                                                     adjlist="""multiplicity 3
1 N u0 p1 c0 {2,S} {3,S} {4,S}
2 N u2 p1 c0 {1,S}
3 H u0 p0 c0 {1,S}
4 H u0 p0 c0 {1,S}""")])
        cls.rxn8 = ARCReaction(r_species=[ARCSpecies(label='CH4', smiles='C', xyz=cls.ch4_xyz),
                                          ARCSpecies(label='OH', smiles='[OH]', xyz=cls.oh_xyz)],
                               p_species=[ARCSpecies(label='CH3', smiles='[CH3]', xyz=cls.ch3_xyz),
                                          ARCSpecies(label='H2O', smiles='O', xyz=cls.h2o_xyz)])
        cls.rxn9 = ARCReaction(r_species=[ARCSpecies(label='NCOO', smiles='NCO[O]')],
                               p_species=[ARCSpecies(label='NHCH2', smiles='N=C'),
                                          ARCSpecies(label='HO2', smiles='O[O]')])
        cls.rxn10 = ARCReaction(r_species=[ARCSpecies(label='HNO', smiles='N=O'),
                                           ARCSpecies(label='NO2', smiles='O=[N+][O-]')],
                                p_species=[ARCSpecies(label='HNO2', smiles='[O-][NH+]=O'),
                                           ARCSpecies(label='NO', smiles='[N]=O')])
        cls.rxn11 = ARCReaction(r_species=[ARCSpecies(label='C[CH]C', smiles='C[CH]C', xyz=cls.ch3chch3_xyz)],
                                p_species=[ARCSpecies(label='[CH2]CC', smiles='[CH2]CC', xyz=cls.ch3ch2ch2_xyz)])
        cls.rxn12 = ARCReaction(r_species=[ARCSpecies(label='NH', smiles='[NH]'), ARCSpecies(label='N2H3', smiles='N[NH]')],
                                p_species=[ARCSpecies(label='NH2', smiles='[NH2]'), ARCSpecies(label='N2H2(T)', smiles='[NH][NH]')])
        cls.rxn_13_w_xyz = ARCReaction(r_species=[ARCSpecies(label='HO2', smiles='O[O]', xyz=cls.ho2_xyz),
                                                  ARCSpecies(label='N2H4', smiles='NN', xyz=cls.n2h4_xyz)],
                                       p_species=[ARCSpecies(label='H2O2', smiles='OO', xyz=cls.h2o2_xyz),
                                                  ARCSpecies(label='N2H3', smiles='N[NH]', xyz=cls.n2h3_xyz)])

    def test_str(self):
        """Test the string representation of the object"""
        str_representation = str(self.rxn1)
        self.assertEqual(self.rxn1.charge, 0)
        expected_representation = 'ARCReaction(label="CH4 + OH <=> CH3 + H2O", multiplicity=2, charge=0)'
        self.assertEqual(str_representation, expected_representation)

    def test_as_dict(self):
        """Test ARCReaction.as_dict()"""
        rxn_dict_1 = self.rxn1.as_dict()
        # mol.atoms are not tested since all id's (including connectivity) changes depending on how the test is run.
        expected_dict_1 = {'family': 'H_Abstraction',
                           'family_own_reverse': True,
                           'label': 'CH4 + OH <=> CH3 + H2O',
                           'multiplicity': 2,
                           'p_species': [{'bond_corrections': {'C-H': 3},
                                          'cheap_conformer': 'C       0.00000000    0.00000001   -0.00000000\n'
                                                             'H       1.06690511   -0.17519582    0.05416493\n'
                                                             'H      -0.68531716   -0.83753536   -0.02808565\n'
                                                             'H      -0.38158795    1.01273118   -0.02607927',
                                          'label': 'CH3',
                                          'long_thermo_description': "Bond corrections: {'C-H': 3}\n",
                                          'mol': {'atom_order': rxn_dict_1['p_species'][0]['mol']['atom_order'],
                                                  'atoms': rxn_dict_1['p_species'][0]['mol']['atoms'],
                                                  'multiplicity': 2,
                                                  'props': {}},
                                          'multiplicity': 2,
                                          'number_of_rotors': 0},
                                         {'bond_corrections': {'H-O': 2},
                                          'cheap_conformer': 'O      -0.00032832    0.39781490    0.00000000\n'
                                                             'H      -0.76330345   -0.19953755    0.00000000\n'
                                                             'H       0.76363177   -0.19827735    0.00000000',
                                          'label': 'H2O',
                                          'long_thermo_description': "Bond corrections: {'H-O': 2}\n",
                                          'mol': {'atom_order': rxn_dict_1['p_species'][1]['mol']['atom_order'],
                                                  'atoms': rxn_dict_1['p_species'][1]['mol']['atoms'],
                                                  'multiplicity': 1,
                                                  'props': {}},
                                          'multiplicity': 1,
                                          'number_of_rotors': 0}],
                           'products': ['CH3', 'H2O'],
                           'r_species': [{'bond_corrections': {'C-H': 4},
                                          'cheap_conformer': 'C      -0.00000000   -0.00000000    0.00000000\n'
                                                             'H      -0.63306457   -0.78034118   -0.42801448\n'
                                                             'H      -0.38919244    0.98049560   -0.28294367\n'
                                                             'H       0.00329661   -0.09013273    1.08846898\n'
                                                             'H       1.01896040   -0.11002169   -0.37751083',
                                          'label': 'CH4',
                                          'long_thermo_description': "Bond corrections: {'C-H': 4}\n",
                                          'mol': {'atom_order': rxn_dict_1['r_species'][0]['mol']['atom_order'],
                                                  'atoms': rxn_dict_1['r_species'][0]['mol']['atoms'],
                                                  'multiplicity': 1,
                                                  'props': {}},
                                          'multiplicity': 1,
                                          'number_of_rotors': 0},
                                         {'bond_corrections': {'H-O': 1},
                                          'cheap_conformer': 'O       0.00000000    0.00000000    0.61310000\n'
                                                             'H       0.00000000    0.00000000   -0.61310000',
                                          'label': 'OH',
                                          'long_thermo_description': "Bond corrections: {'H-O': 1}\n",
                                          'mol': {'atom_order': rxn_dict_1['r_species'][1]['mol']['atom_order'],
                                                  'atoms': rxn_dict_1['r_species'][1]['mol']['atoms'],
                                                  'multiplicity': 2,
                                                  'props': {}},
                                          'multiplicity': 2,
                                          'number_of_rotors': 0}],
                           'reactants': ['CH4', 'OH'],
                           }
        self.assertEqual(rxn_dict_1, expected_dict_1)

        rxn_dict_6 = self.rxn6.as_dict()
        # The ``long_thermo_description`` attribute isn't deterministic (order could change)
        expected_dict_6 = {'label': 'NH2 + N2H3 <=> NH3 + H2NN[S]',
                           'family': 'Disproportionation',
                           'multiplicity': 1,
                           'p_species': [{'bond_corrections': {'H-N': 3},
                                          'cheap_conformer': 'N       0.00064924   -0.00099698    0.29559292\n'
                                                             'H      -0.41786606    0.84210396   -0.09477452\n'
                                                             'H      -0.52039228   -0.78225292   -0.10002797\n'
                                                             'H       0.93760911   -0.05885406   -0.10079043',
                                          'label': 'NH3',
                                          'long_thermo_description': "Bond corrections: {'H-N': 3}\n",
                                          'mol': {'atom_order': rxn_dict_6['p_species'][0]['mol']['atom_order'],
                                                  'atoms': rxn_dict_6['p_species'][0]['mol']['atoms'],
                                                  'multiplicity': 1,
                                                  'props': {}},
                                          'multiplicity': 1,
                                          'number_of_rotors': 0},
                                         {'arkane_file': None,
                                          'adjlist': """multiplicity 1
1 N u0 p0 c+1 {2,D} {3,S} {4,S}
2 N u0 p2 c-1 {1,D}
3 H u0 p0 c0 {1,S}
4 H u0 p0 c0 {1,S}""",
                                          'bond_corrections': {'H-N': 2, 'N=N': 1},
                                          'cheap_conformer': 'N      -0.09766126    0.01379054    0.00058556\n'
                                                             'N       1.34147594   -0.18942713   -0.00804275\n'
                                                             'H      -0.74382022   -0.77560691    0.00534230\n'
                                                             'H      -0.49999445    0.95124349    0.00211489',
                                          'label': 'H2NN[S]',
                                          'long_thermo_description': rxn_dict_6['p_species'][1]['long_thermo_description'],
                                          'mol': {'atom_order': rxn_dict_6['p_species'][1]['mol']['atom_order'],
                                                  'atoms': rxn_dict_6['p_species'][1]['mol']['atoms'],
                                                  'multiplicity': 1,
                                                  'props': {}},
                                          'multiplicity': 1,
                                          'number_of_rotors': 0,
                                          'original_label': 'H2NN(S)'}],
                           'products': ['H2NN[S]', 'NH3'],
                           'r_species': [{'bond_corrections': {'H-N': 2},
                                          'cheap_conformer': 'N       0.00016375    0.40059499    0.00000000\n'
                                                             'H      -0.83170922   -0.19995756    0.00000000\n'
                                                             'H       0.83154548   -0.20063742    0.00000000',
                                          'label': 'NH2',
                                          'long_thermo_description': "Bond corrections: {'H-N': 2}\n",
                                          'mol': {'atom_order': rxn_dict_6['r_species'][0]['mol']['atom_order'],
                                                  'atoms': rxn_dict_6['r_species'][0]['mol']['atoms'],
                                                  'multiplicity': 2,
                                                  'props': {}},
                                          'multiplicity': 2,
                                          'number_of_rotors': 0},
                                         {'bond_corrections': {'H-N': 3, 'N-N': 1},
                                          'cheap_conformer': 'N      -0.46751749    0.03795671    0.31180026\n'
                                                             'N       0.79325823   -0.46038094   -0.24114357\n'
                                                             'H      -1.19307188   -0.63034971    0.05027053\n'
                                                             'H      -0.69753009    0.90231202   -0.17907452\n'
                                                             'H       1.56486123    0.15046192    0.05814730',
                                          'label': 'N2H3',
                                          'long_thermo_description': rxn_dict_6['r_species'][1]['long_thermo_description'],
                                          'mol': {'atom_order': rxn_dict_6['r_species'][1]['mol']['atom_order'],
                                                  'atoms': rxn_dict_6['r_species'][1]['mol']['atoms'],
                                                  'multiplicity': 2,
                                                  'props': {}},
                                          'multiplicity': 2,
                                          'mol_list': ['[NH]N', '[NH-][NH2+]'],
                                          'number_of_rotors': 0}],
                           'reactants': ['N2H3', 'NH2'],
                           }
        self.assertEqual(rxn_dict_6, expected_dict_6)

        rxn_7_dict = self.rxn7.as_dict()
        self.assertEqual(rxn_7_dict['p_species'][1]['adjlist'], """multiplicity 3
1 N u0 p1 c0 {2,S} {3,S} {4,S}
2 N u2 p1 c0 {1,S}
3 H u0 p0 c0 {1,S}
4 H u0 p0 c0 {1,S}""")

    def test_from_dict(self):
        """Test ARCReaction.from_dict()"""
        rxn_dict = self.rxn1.as_dict()
        rxn = ARCReaction(reaction_dict=rxn_dict)
        self.assertEqual(rxn.label, 'CH4 + OH <=> CH3 + H2O')
        self.assertEqual(rxn.multiplicity, 2)
        self.assertEqual(rxn.charge, 0)
        self.assertEqual(rxn.family, 'H_Abstraction')
        self.assertEqual(rxn.family_own_reverse, True)
        self.assertEqual(rxn.reactants, ['CH4', 'OH'])
        self.assertEqual(rxn.products, ['CH3', 'H2O'])
        self.assertIsNone(rxn.index)

    def test_copy(self):
        """Test the copy() method."""
        rxn_copy = self.rxn1.copy()
        self.assertIsNot(self.rxn1, rxn_copy)
        self.assertEqual(rxn_copy.multiplicity, self.rxn1.multiplicity)
        self.assertEqual(rxn_copy.label, self.rxn1.label)

        rxn_copy = self.rxn8.copy()
        self.assertIsNot(self.rxn8, rxn_copy)
        self.assertEqual(self.rxn8.label, rxn_copy.label)
        self.assertEqual(self.rxn8.multiplicity, rxn_copy.multiplicity)
        self.assertEqual(self.rxn8.charge, rxn_copy.charge)
        self.assertTrue(check_atom_map(self.rxn8))
        self.assertTrue(check_atom_map(rxn_copy))
        self.assertEqual(self.rxn8.atom_map, rxn_copy.atom_map)
        self.assertEqual(tuple(spc.label for spc in self.rxn8.r_species),
                         tuple(spc.label for spc in rxn_copy.r_species))
        self.assertEqual(tuple(spc.label for spc in self.rxn8.p_species),
                         tuple(spc.label for spc in rxn_copy.p_species))
        self.assertNotIn(self.rxn8.r_species[0].mol.atoms[0].id, [atom.id for atom in rxn_copy.r_species[0].mol.atoms
                                                                  + rxn_copy.p_species[0].mol.atoms])

    def test_flip_reaction(self):
        """Test the flip_reaction() method."""
        flipped_rxn = self.rxn1.flip_reaction()
        self.assertIsNot(self.rxn1, flipped_rxn)
        self.assertEqual(flipped_rxn.multiplicity, self.rxn1.multiplicity)
        self.assertNotEqual(flipped_rxn.label, self.rxn1.label)
        for s1, s2 in zip(self.rxn1.r_species, flipped_rxn.p_species):
            self.assertEqual(s1.label, s2.label)
            self.assertTrue(s1.mol.is_isomorphic(s2.mol))
        for s1, s2 in zip(self.rxn1.p_species, flipped_rxn.r_species):
            self.assertEqual(s1.label, s2.label)
            self.assertTrue(s1.mol.is_isomorphic(s2.mol))

        flipped_rxn = self.rxn8.flip_reaction()
        self.assertIsNot(self.rxn8, flipped_rxn)
        self.assertNotEqual(self.rxn8.label, flipped_rxn.label)
        self.assertEqual(self.rxn8.label.split('<=>')[0].strip(), flipped_rxn.label.split('<=>')[1].strip())
        self.assertEqual(self.rxn8.label.split('<=>')[1].strip(), flipped_rxn.label.split('<=>')[0].strip())
        self.assertEqual(self.rxn8.multiplicity, flipped_rxn.multiplicity)
        self.assertEqual(self.rxn8.charge, flipped_rxn.charge)
        self.assertTrue(check_atom_map(flipped_rxn))
        self.assertEqual(self.rxn8.atom_map[0], 0)
        self.assertEqual(self.rxn8.atom_map[5], 4)
        self.assertEqual(flipped_rxn.atom_map[0], 0)
        self.assertEqual(flipped_rxn.atom_map[4], 5)
        self.assertEqual(tuple(spc.label for spc in self.rxn8.r_species),
                         tuple(spc.label for spc in flipped_rxn.p_species))
        self.assertEqual(tuple(spc.label for spc in self.rxn8.p_species),
                         tuple(spc.label for spc in flipped_rxn.r_species))
        self.assertNotIn(self.rxn8.r_species[0].mol.atoms[0].id, [atom.id for atom in flipped_rxn.r_species[0].mol.atoms
                                                                  + flipped_rxn.p_species[0].mol.atoms])

    def test_is_isomerization(self):
        """Test the is_isomerization() method"""
        self.assertFalse(self.rxn1.is_isomerization())
        self.assertFalse(self.rxn2.is_isomerization())
        self.assertTrue(self.rxn3.is_isomerization())
        self.assertFalse(self.rxn4.is_isomerization())
        self.assertFalse(self.rxn5.is_isomerization())
        self.assertFalse(self.rxn6.is_isomerization())
        self.assertFalse(self.rxn7.is_isomerization())
        self.assertFalse(self.rxn8.is_isomerization())
        self.assertFalse(self.rxn9.is_isomerization())

    def test_rxn_family(self):
        """Test that ARC gets the correct RMG family for different reactions"""
        n2h3_xyz = {'coords': ((-0.470579649119187, 0.04999660847282449, 0.3054306465848634),
                               (0.7822241718336367, -0.48270144244781193, -0.23341826421899858),
                               (1.5677258653370059, 0.10203349472372605, 0.08145841384293159),
                               (-0.6670140616222734, 0.9245291856920813, -0.1819911659528955),
                               (-1.212356326429186, -0.5938578464408176, 0.028520369744099196)),
                    'isotopes': (14, 14, 1, 1, 1),
                    'symbols': ('N', 'N', 'H', 'H', 'H')}
        nh2_xyz = {'coords': ((0.0001637451536497341, 0.4005949879135532, 0.0),
                              (-0.8317092208339203, -0.19995756341639623, 0.0),
                              (0.8315454756802706, -0.20063742449715688, 0.0)),
                   'isotopes': (14, 1, 1),
                   'symbols': ('N', 'H', 'H')}
        n2h2_t_xyz = {'coords': ((0.5974274138372041, -0.41113104979405946, 0.08609839663782763),
                                 (-0.5974274348582206, 0.41113108883884353, -0.08609846602622732),
                                 (1.421955422639823, 0.19737093442024492, 0.02508578507394823),
                                 (-1.4219554016188147, -0.19737097346502322, -0.02508571568554942)),
                      'isotopes': (14, 1, 14, 1),
                      'symbols': ('N', 'N', 'H', 'H')}
        r_1 = ARCSpecies(label='NH', smiles='[NH]', xyz=self.nh_xyz)
        r_2 = ARCSpecies(label='N2H3', smiles='N[NH]', xyz=n2h3_xyz)
        p_1 = ARCSpecies(label='NH2', smiles='[NH2]', xyz=nh2_xyz)
        p_2 = ARCSpecies(label='N2H2(T)', smiles='[NH][NH]', xyz=n2h2_t_xyz, multiplicity=3)
        rxn = ARCReaction(reactants=['NH', 'N2H3'], products=['NH2', 'N2H2(T)'],
                          r_species=[r_1, r_2], p_species=[p_1, p_2])

        self.assertEqual(rxn.family, 'H_Abstraction')
        self.assertEqual(self.rxn1.family, 'H_Abstraction')
        self.assertTrue(self.rxn1.family_own_reverse)
        self.assertEqual(self.rxn2.family, 'Disproportionation')
        self.assertFalse(self.rxn2.family_own_reverse)
        self.assertEqual(self.rxn3.family, 'intra_H_migration')
        self.assertTrue(self.rxn3.family_own_reverse)
        self.assertEqual(self.rxn4.family, 'H_Abstraction')
        self.rxn5.check_attributes()
        self.assertEqual(self.rxn5.family, 'H_Abstraction')
        self.rxn9.check_attributes()
        self.assertEqual(self.rxn9.family, 'HO2_Elimination_from_PeroxyRadical')
        rxn_9_flipped = self.rxn9.flip_reaction()
        rxn_9_flipped.check_attributes()
        self.assertEqual(rxn_9_flipped.family, 'HO2_Elimination_from_PeroxyRadical')
        self.assertEqual(self.rxn10.family, 'H_Abstraction')
        rxn_1 = ARCReaction(r_species=[ARCSpecies(label='C2H6', smiles='CC'),
                                       ARCSpecies(label='CCOOj', smiles='CCO[O]')],
                            p_species=[ARCSpecies(label='CCOOH', smiles='CCOO'),
                                       ARCSpecies(label='C2H5', smiles='C[CH2]')])
        rxn_1.check_attributes()
        self.assertEqual(rxn_1.family, 'H_Abstraction')

        # Test identifying the reaction family for a zwitterions read from an Arkane YAML files
        base_path = os.path.join(ARC_PATH, 'arc', 'testing', 'yml_testing', 'HNNO+NH3O=H2NO+NH2NO')
        rxn2 = ARCReaction(r_species=[ARCSpecies(label='HNNO', smiles='[O-][N+]=N', yml_path=os.path.join(base_path, 'HNNO.yml')),
                                      ARCSpecies(label='NH3O', smiles='[O-][NH3+]', yml_path=os.path.join(base_path, 'NH3O.yml'))],
                           p_species=[ARCSpecies(label='H2NO', smiles='N[O]', yml_path=os.path.join(base_path, 'H2NO.yml')),
                                      ARCSpecies(label='NH2NO', smiles='NN=O', yml_path=os.path.join(base_path, 'NH2NO.yml'))])
        self.assertEqual(rxn2.family, 'H_Abstraction')
        self.assertEqual(self.rxn12.family, 'H_Abstraction')

    def test_charge_property(self):
        """Test determining charge"""
        self.assertEqual(self.rxn1.charge, 0)

    def test_multiplicity_property(self):
        """Test determining multiplicity"""
        self.assertEqual(self.rxn1.multiplicity, 2)
        self.assertEqual(self.rxn2.multiplicity, 1)
        self.assertEqual(self.rxn3.multiplicity, 2)
        self.assertEqual(self.rxn4.multiplicity, 3)
        self.assertEqual(self.rxn5.multiplicity, 3)
        self.assertEqual(self.rxn6.multiplicity, 1)
        self.assertEqual(self.rxn7.multiplicity, 3)

        # isomerization
        rxn_1 = ARCReaction(reactants=['nC3H7'], products=['iC3H7'],
                            r_species=[ARCSpecies(label='nC3H7', smiles='[CH2]CC')],
                            p_species=[ARCSpecies(label='iC3H7', smiles='C[CH]C')])
        self.assertEqual(rxn_1.multiplicity, 2)
        rxn_2 = ARCReaction(reactants=['CC=O'], products=['C=COH'],
                            r_species=[ARCSpecies(label='CC=O', smiles='CC=O')],
                            p_species=[ARCSpecies(label='C=COH', smiles='C=CO')])
        self.assertEqual(rxn_2.multiplicity, 1)

        # unimolecular
        rxn_3 = ARCReaction(reactants=['H', 'OH'], products=['H2O'],
                            r_species=[ARCSpecies(label='H', smiles='[H]'), ARCSpecies(label='OH', smiles='[OH]')],
                            p_species=[ARCSpecies(label='H2O', smiles='O')])
        self.assertEqual(rxn_3.multiplicity, 1)

        # Reactions for which the multiplicity was wrongly determined before code fixes:
        rxn_4 = ARCReaction(reactants=['H', 'HO2'], products=['OH', 'OH'],
                            r_species=[ARCSpecies(label='H', smiles='[H]'), ARCSpecies(label='HO2', smiles='O[O]')],
                            p_species=[ARCSpecies(label='OH', smiles='[OH]'), ARCSpecies(label='OH', smiles='[OH]')])
        self.assertEqual(rxn_4.multiplicity, 1)

        rxn_5 = ARCReaction(r_species=[ARCSpecies(label='N', smiles='[N]'), ARCSpecies(label='HNO', smiles='N=O')],
                            p_species=[ARCSpecies(label='NO', smiles='[N]=O'),
                                       ARCSpecies(label='NH', smiles='[NH]')])
        self.assertEqual(rxn_5.multiplicity, 4)
        rxn_5 = ARCReaction(r_species=[ARCSpecies(label='N', smiles='[N]'), ARCSpecies(label='HNO', smiles='N=O')],
                            p_species=[ARCSpecies(label='NO', smiles='[N]=O'), ARCSpecies(label='NH', smiles='[NH]')])
        self.assertEqual(rxn_5.multiplicity, 4)

    def test_check_atom_balance(self):
        """Test the Reaction check_atom_balance method"""
        # A normal reaction
        rxn1 = ARCReaction(reactants=['CH4', 'OH'], products=['CH3', 'H2O'],
                           r_species=[ARCSpecies(label='CH4', smiles='C'),
                                      ARCSpecies(label='OH', smiles='[OH]')],
                           p_species=[ARCSpecies(label='CH3', smiles='[CH3]'),
                                      ARCSpecies(label='H2O', smiles='O')])
        self.assertTrue(rxn1.check_atom_balance())

        # A reaction with the same species twice on one side
        rxn3 = ARCReaction(reactants=['CH4', 'OH', 'H2O'], products=['CH3', 'H2O', 'H2O'],
                           r_species=[ARCSpecies(label='CH4', smiles='C'),
                                      ARCSpecies(label='OH', smiles='[OH]'),
                                      ARCSpecies(label='H2O', smiles='O')],
                           p_species=[ARCSpecies(label='CH3', smiles='[CH3]'),
                                      ARCSpecies(label='H2O', smiles='O')])
        self.assertTrue(rxn3.check_atom_balance())

        # Another reaction with the same species twice on one side
        rxn4 = ARCReaction(reactants=['OH', 'OH'], products=['O', 'H2O'],
                           r_species=[ARCSpecies(label='OH', smiles='[OH]')],
                           p_species=[ARCSpecies(label='O', smiles='[O]'),
                                      ARCSpecies(label='H2O', smiles='O')])
        self.assertTrue(rxn4.check_atom_balance())

        # Another reaction with the same species twice on one side, reading from the reaction label
        nh2 = ARCSpecies(label='NH2', smiles='[NH2]')
        h = ARCSpecies(label='H', smiles='[H]')
        n2h3 = ARCSpecies(label='N2H3', smiles='[NH]N')
        rxn_label = 'NH2 + NH2 <=> H + N2H3'
        rxn5 = ARCReaction(reaction_dict={'label': rxn_label}, species_list=[nh2, h, n2h3])
        self.assertTrue(rxn5.check_atom_balance())
        rxn5_flipped = rxn5.flip_reaction()
        self.assertTrue(rxn5_flipped.check_atom_balance())

        # Legitimate reactions that previously failed in the atom balance test
        rxn6 = ARCReaction(reactants=['NH', '[O-][N+](=N)N'], products=['NH2', '[N-]=[N+]([O])N'],
                           r_species=[ARCSpecies(label='NH', smiles='[NH]'),
                                      ARCSpecies(label='[O-][N+](=N)N', smiles='[O-][N+](=N)N')],
                           p_species=[ARCSpecies(label='NH2', smiles='[NH2]'),
                                      ARCSpecies(label='[N-]=[N+]([O])N', smiles='[N-]=[N+]([O])N')])
        self.assertTrue(rxn6.check_atom_balance())

        rxn7 = ARCReaction(reactants=['N3O2', 'HON'], products=['NO', 'HN3O2'],
                           r_species=[ARCSpecies(label='N3O2', smiles='[N-]=[N+](N=O)[O]'),
                                      ARCSpecies(label='HON', smiles='[OH+]=[N-]')],
                           p_species=[ARCSpecies(label='NO', smiles='[N]=O'),
                                      ARCSpecies(label='HN3O2', smiles='[O-][N+](=N)N=O')])
        self.assertTrue(rxn7.check_atom_balance())

        # A reaction that involves charged species
        rxn8 = ARCReaction(reactants=['C6CNC3', 'OH'], products=['C6COH', 'NC3'],
                           r_species=[ARCSpecies(label='C6CNC3', smiles='c1ccccc1C[N+](C)(C)C', charge=1, multiplicity=1),
                                      ARCSpecies(label='OH', smiles='[OH-]', charge=-1, multiplicity=1)],
                           p_species=[ARCSpecies(label='C6COH', smiles='c1ccccc1CO'),
                                      ARCSpecies(label='NC3', smiles='CN(C)C')])
        self.assertTrue(rxn8.check_atom_balance())

        # A *non*-balanced reaction
        with self.assertRaises(ReactionError):
            ARCReaction(reactants=['CH4', 'OH'], products=['CH4', 'H2O'],
                        r_species=[ARCSpecies(label='CH4', smiles='C'),
                                   ARCSpecies(label='OH', smiles='[OH]')],
                        p_species=[ARCSpecies(label='CH4', smiles='C'),
                                   ARCSpecies(label='H2O', smiles='O')])

    def test_get_species_count(self):
        """Test the get_species_count() method"""
        rxn1 = ARCReaction(reactants=['CH4', 'OH', 'H2O'], products=['CH3', 'H2O', 'H2O'])
        spc1 = ARCSpecies(label='OH', smiles='[OH]')
        spc2 = ARCSpecies(label='H2O', smiles='O')
        # check by species
        self.assertEqual(rxn1.get_species_count(species=spc1, well=0), 1)
        self.assertEqual(rxn1.get_species_count(species=spc1, well=1), 0)
        self.assertEqual(rxn1.get_species_count(species=spc2, well=0), 1)
        self.assertEqual(rxn1.get_species_count(species=spc2, well=1), 2)
        # check by label
        self.assertEqual(rxn1.get_species_count(label=spc1.label, well=0), 1)
        self.assertEqual(rxn1.get_species_count(label=spc1.label, well=1), 0)
        self.assertEqual(rxn1.get_species_count(label=spc2.label, well=0), 1)
        self.assertEqual(rxn1.get_species_count(label=spc2.label, well=1), 2)

        h2nn = ARCSpecies(label='H2NN(T)', smiles='[N]N')
        n2h2 = ARCSpecies(label='N2H4', smiles='NN')
        n2h3 = ARCSpecies(label='N2H3', smiles='[NH]N')
        rxn2 = ARCReaction(r_species=[h2nn, n2h2], p_species=[n2h3, n2h3])
        self.assertEqual(rxn2.get_species_count(label=n2h3.label, well=1), 2)

    def test_get_reactants_and_products(self):
        """Test getting reactants and products"""
        self.rxn1.remove_dup_species()
        reactants, products = self.rxn1.get_reactants_and_products()
        for spc in reactants + products:
            self.assertIsInstance(spc, ARCSpecies)
        self.assertEqual(len(reactants), 2)
        self.assertEqual(len(products), 2)

        reactants, products = self.rxn5.get_reactants_and_products()
        for spc in reactants + products:
            self.assertIsInstance(spc, ARCSpecies)
        self.assertEqual(len(reactants), 2)
        self.assertEqual(len(products), 2)
        self.assertEqual(reactants[0].label, reactants[1].label)

        h2nn = ARCSpecies(label='H2NN(T)', smiles='[N]N')
        n2h2 = ARCSpecies(label='N2H4', smiles='NN')
        n2h3 = ARCSpecies(label='N2H3', smiles='[NH]N')
        rxn1 = ARCReaction(r_species=[h2nn, n2h2], p_species=[n2h3, n2h3])
        reactants, products = rxn1.get_reactants_and_products(return_copies=False)
        self.assertEqual(len(reactants), 2)
        self.assertEqual(len(products), 2)
        self.assertIs(products[0], products[1])
        reactants, products = rxn1.get_reactants_and_products(return_copies=True)
        self.assertEqual(len(reactants), 2)
        self.assertEqual(len(products), 2)
        self.assertIsNot(products[0], products[1])

    def test_get_expected_changing_bonds(self):
        """Test the get_expected_changing_bonds() method."""
        expected_breaking_bonds, expected_forming_bonds = self.rxn11.get_expected_changing_bonds(
            r_label_dict={'*1': 1, '*2': 2, '*3': 6})
        self.assertEqual(expected_breaking_bonds, [(2, 6)])
        self.assertEqual(expected_forming_bonds, [(1, 6)])

    def test_get_number_of_atoms_in_reaction_zone(self):
        """Test the get_number_of_atoms_in_reaction_zone() method."""
        self.assertEqual(self.rxn1.get_number_of_atoms_in_reaction_zone(), 3)  # H_Abstraction
        self.assertEqual(self.rxn2.get_number_of_atoms_in_reaction_zone(), 4)  # Disprop
        self.assertEqual(self.rxn3.get_number_of_atoms_in_reaction_zone(), 3)
        self.assertEqual(self.rxn4.get_number_of_atoms_in_reaction_zone(), 3)
        self.assertEqual(self.rxn5.get_number_of_atoms_in_reaction_zone(), 3)
        self.assertEqual(self.rxn6.get_number_of_atoms_in_reaction_zone(), 4)  # Disprop
        self.assertEqual(self.rxn7.get_number_of_atoms_in_reaction_zone(), 3)
        self.assertEqual(self.rxn8.get_number_of_atoms_in_reaction_zone(), 3)
        self.assertEqual(self.rxn9.get_number_of_atoms_in_reaction_zone(), 5)  # HO2_Elimination_from_PeroxyRadical
        self.assertEqual(self.rxn10.get_number_of_atoms_in_reaction_zone(), 3)
        self.assertEqual(self.rxn11.get_number_of_atoms_in_reaction_zone(), 3)

    def test_get_reactants_xyz(self):
        """Test getting a combined string/dict representation of the cartesian coordinates of all reactant species"""
        ch3nh2_xyz = {'coords': ((-0.5734111454228507, 0.0203516083213337, 0.03088703933770556),
                                 (0.8105595891860601, 0.00017446498908627427, -0.4077728757313545),
                                 (-1.1234549667791063, -0.8123899006368857, -0.41607711106038836),
                                 (-0.6332220120842996, -0.06381791823047896, 1.1196983583774054),
                                 (-1.053200912106195, 0.9539501896695028, -0.27567270246542575),
                                 (1.3186422395164141, 0.7623906284020254, 0.038976118645639976),
                                 (1.2540872076899663, -0.8606590725145833, -0.09003882710357966)),
                      'isotopes': (12, 14, 1, 1, 1, 1, 1),
                      'symbols': ('C', 'N', 'H', 'H', 'H', 'H', 'H')}
        ch2nh2_xyz = {'coords': ((0.6919493009211066, 0.054389375309083846, 0.02065422596281878),
                                 (1.3094508022837807, -0.830934909576592, 0.14456347719459348),
                                 (1.1649142139806816, 1.030396183273415, 0.08526955368597328),
                                 (-0.7278194451655412, -0.06628299353512612, -0.30657582460750543),
                                 (-1.2832757211903472, 0.7307667658607352, 0.00177732009031573),
                                 (-1.155219150829674, -0.9183344213315149, 0.05431124767380799)),
                      'isotopes': (12, 1, 1, 14, 1, 1),
                      'symbols': ('C', 'H', 'H', 'N', 'H', 'H')}
        r_1 = ARCSpecies(label='H', smiles='[H]', xyz={'coords': ((0, 0, 0),), 'isotopes': (1,), 'symbols': ('H',)})
        r_2 = ARCSpecies(label='CH3NH2', smiles='CN', xyz=ch3nh2_xyz)
        p_1 = ARCSpecies(label='H2', smiles='[H][H]', xyz=self.h2_xyz)
        p_2 = ARCSpecies(label='CH2NH2', smiles='[CH2]N', xyz=ch2nh2_xyz)
        rxn_1 = ARCReaction(reactants=['H', 'CH3NH2'], products=['H2', 'CH2NH2'],
                            r_species=[r_1, r_2], p_species=[p_1, p_2])
        reactants_xyz_str = rxn_1.get_reactants_xyz()
        reactants_xyz_dict = rxn_1.get_reactants_xyz(return_format='dict')
        expected_reactants_xyz_str = """H      -0.33027713    0.00000000    0.00000000
C      -0.69896689    0.01307198    0.19065646
N       0.68500384   -0.00710516   -0.24800345
H      -1.24901071   -0.81966953   -0.25630769
H      -0.75877776   -0.07109755    1.27946778
H      -1.17875666    0.94667056   -0.11590328
H       1.19308649    0.75511100    0.19874554
H       1.12853146   -0.86793870    0.06973060"""
        expected_reactants_xyz_dict = {'symbols': ('H', 'C', 'N', 'H', 'H', 'H', 'H', 'H'),
                                       'isotopes': (1, 12, 14, 1, 1, 1, 1, 1),
                                       'coords': ((-0.33027712709756135, 0.0, 0.0),
                                                  (-0.6989668914012912, 0.013071980537625375, 0.19065646408548478),
                                                  (0.6850038432076195, -0.00710516279462205, -0.2480034509835753),
                                                  (-1.249010712757547, -0.8196695284205939, -0.25630768631260914),
                                                  (-0.7587777580627402, -0.07109754601418727, 1.2794677831251846),
                                                  (-1.1787566580846356, 0.9466705618857946, -0.11590327771764654),
                                                  (1.1930864935379737, 0.7551110006183172, 0.19874554339341918),
                                                  (1.1285314617115256, -0.8679387002982916, 0.06973059764419955))}
        self.assertEqual(reactants_xyz_str, expected_reactants_xyz_str)
        self.assertEqual(reactants_xyz_dict, expected_reactants_xyz_dict)

        c2h5o3_xyz = {'coords': ((-1.3476727508427788, -0.49923624257482285, -0.3366372557370102),
                                 (-0.11626816111736853, 0.3110915299407186, 0.018860985632263887),
                                 (0.7531175607750088, 0.3366822240291409, -1.1050387236863213),
                                 (0.5228736844989644, -0.3049881931104616, 1.1366016759286774),
                                 (1.8270658637404131, 0.34102014147584997, 1.2684162942337813),
                                 (-2.039181700362481, -0.5535509846570477, 0.5100031541057821),
                                 (-1.865025875161301, -0.06806929272376178, -1.1994046923960628),
                                 (-1.0711960095793496, -1.5264629385419055, -0.6002175107608478),
                                 (-0.40133538695862053, 1.3357900487643664, 0.28224155088545305),
                                 (1.3942569570346546, 1.035594500292526, -0.8890721851777293)),
                      'isotopes': (12, 12, 16, 16, 16, 1, 1, 1, 1, 1),
                      'symbols': ('C', 'C', 'O', 'O', 'O', 'H', 'H', 'H', 'H', 'H')}
        c2h4o_xyz = {'coords': ((-0.6485165220711699, -0.036287809639473964, -0.040072327958319325),
                                (0.8441328059817381, 0.04088405476411104, 0.05352861712992162),
                                (1.4799812732494606, 1.0748679945888888, -0.1224478071645769),
                                (-1.0603388058764294, 0.9464876376852732, -0.28238370478893315),
                                (-0.9213427138232859, -0.7478396768473443, -0.8228167900899559),
                                (-1.0499663443190728, -0.37234114306362315, 0.9187474043028493),
                                (1.3560503068587568, -0.9057710574878411, 0.29544460856901716)),
                     'isotopes': (12, 12, 16, 1, 1, 1, 1),
                     'symbols': ('C', 'C', 'O', 'H', 'H', 'H', 'H')}
        ho2_xyz = {'coords': ((0.0558910, -0.6204870, 0.0000000),
                              (0.0558910, 0.7272050, 0.0000000),
                              (-0.8942590, -0.8537420, 0.0000000)),
                   'isotopes': (16, 16, 1),
                   'symbols': ('O', 'O', 'H')}
        r_1 = ARCSpecies(label='C2H5O3', smiles='CC(O)O[O]', xyz=c2h5o3_xyz)
        p_1 = ARCSpecies(label='C2H4O', smiles='CC=O', xyz=c2h4o_xyz)
        p_2 = ARCSpecies(label='HO2', smiles='O[O]', xyz=ho2_xyz)
        rxn = ARCReaction(r_species=[r_1], p_species=[p_1, p_2])
        self.assertIn(rxn.atom_map[0:5], [[0, 1, 2, 8, 7], [0, 1, 2, 7, 8]])
        for index in [5, 6, 7]:
            self.assertIn(rxn.atom_map[index], [3, 4, 5])
        self.assertEqual(rxn.atom_map[8], 6)
        self.assertEqual(rxn.atom_map[9], 9)
        self.assertTrue(check_atom_map(rxn))

    def test_get_single_mapped_product_xyz(self):
        """Test the Reaction get_single_mapped_product_xyz() method"""
        # Trivial unimolecular with an intentional mixed atom order: H2O <=> H2O
        h2o_xyz_1 = """O      -0.00032832    0.39781490    0.00000000
                       H      -0.76330345   -0.19953755    0.00000000
                       H       0.76363177   -0.19827735    0.00000000"""
        r_1 = ARCSpecies(label='H2O', smiles='O', xyz=h2o_xyz_1)
        h2o_xyz_2 = """H      -0.76330345   -0.19953755    0.00000000
                       H       0.76363177   -0.19827735    0.00000000
                       O      -0.00032832    0.39781490    0.00000000"""
        p_1 = ARCSpecies(label='H2O', smiles='O', xyz=h2o_xyz_2)
        rxn_1 = ARCReaction(reactants=['H2O'], products=['H2O'],
                            r_species=[r_1], p_species=[p_1])
        mapped_product = rxn_1.get_single_mapped_product_xyz()
        self.assertEqual(rxn_1.atom_map, [2, 0, 1])
        self.assertTrue(check_atom_map(rxn_1))
        expected_xyz = {'symbols': ('O', 'H', 'H'), 'isotopes': (16, 1, 1),
                        'coords': ((-0.00032832, 0.3978149, 0.0), (-0.76330345, -0.19953755, 0.0),
                                   (0.76363177, -0.19827735, 0.0))}
        self.assertEqual(mapped_product.get_xyz(), expected_xyz)

        reactant_xyz = """C  -1.3087    0.0068    0.0318
                          C   0.1715   -0.0344    0.0210
                          N   0.9054   -0.9001    0.6395
                          O   2.1683   -0.5483    0.3437
                          N   2.1499    0.5449   -0.4631
                          N   0.9613    0.8655   -0.6660
                          H  -1.6558    0.9505    0.4530
                          H  -1.6934   -0.0680   -0.9854
                          H  -1.6986   -0.8169    0.6255"""
        reactant = ARCSpecies(label='reactant', smiles='C([C]1=[N]O[N]=[N]1)', xyz=reactant_xyz)
        product_xyz = """C  -1.0108   -0.0114   -0.0610
                         C   0.4780    0.0191    0.0139
                         N   1.2974   -0.9930    0.4693
                         O   0.6928   -1.9845    0.8337
                         N   1.7456    1.9701   -0.6976
                         N   1.1642    1.0763   -0.3716
                         H  -1.4020    0.9134   -0.4821
                         H  -1.3327   -0.8499   -0.6803
                         H  -1.4329   -0.1554    0.9349"""
        product = ARCSpecies(label='product', smiles='[N-]=[N+]=C(N=O)C', xyz=product_xyz)
        rxn_2 = ARCReaction(r_species=[reactant], p_species=[product])
        self.assertTrue(check_atom_map(rxn_2))
        mapped_product = rxn_2.get_single_mapped_product_xyz()
        self.assertEqual(rxn_2.atom_map[:6], [0, 1, 2, 3, 4, 5])
        self.assertIn(rxn_2.atom_map[6], [6, 8])
        self.assertIn(rxn_2.atom_map[7], [6, 7])
        self.assertIn(rxn_2.atom_map[8], [7, 8])
        expected_xyz = {'symbols': ('C', 'C', 'N', 'O', 'N', 'N', 'H', 'H', 'H'),
                        'isotopes': (12, 12, 14, 16, 14, 14, 1, 1, 1),
                        'coords': ((-1.0108, -0.0114, -0.061), (0.478, 0.0191, 0.0139), (1.2974, -0.993, 0.4693),
                                   (0.6928, -1.9845, 0.8337), (1.7456, 1.9701, -0.6976), (1.1642, 1.0763, -0.3716),
                                   (-1.4329, -0.1554, 0.9349), (-1.402, 0.9134, -0.4821), (-1.3327, -0.8499, -0.6803))}
        self.assertEqual(mapped_product.get_xyz(), expected_xyz)

        reactant_xyz = """C  -1.3087    0.0068    0.0318
                          C   0.1715   -0.0344    0.0210
                          N   0.9054   -0.9001    0.6395
                          O   2.1683   -0.5483    0.3437
                          N   2.1499    0.5449   -0.4631
                          N   0.9613    0.8655   -0.6660
                          H  -1.6558    0.9505    0.4530
                          H  -1.6934   -0.0680   -0.9854
                          H  -1.6986   -0.8169    0.6255"""
        reactant = ARCSpecies(label='reactant', smiles='C([C]1=[N]O[N]=[N]1)', xyz=reactant_xyz)
        product_xyz = """C  -1.0108   -0.0114   -0.0610
                         C   0.4780    0.0191    0.0139
                         N   1.2974   -0.9930    0.4693
                         O   0.6928   -1.9845    0.8337
                         N   1.7456    1.9701   -0.6976
                         N   1.1642    1.0763   -0.3716
                         H  -1.4020    0.9134   -0.4821
                         H  -1.3327   -0.8499   -0.6803
                         H  -1.4329   -0.1554    0.9349"""
        product = ARCSpecies(label='product', smiles='[N-]=[N+]=C(N=O)C', xyz=product_xyz)
        rxn_2 = ARCReaction(r_species=[reactant], p_species=[product])
        self.assertTrue(check_atom_map(rxn_2))
        mapped_product = rxn_2.get_single_mapped_product_xyz()
        self.assertEqual(rxn_2.atom_map[:6], [0, 1, 2, 3, 4, 5])
        self.assertIn(rxn_2.atom_map[6], [6, 8])
        self.assertIn(rxn_2.atom_map[7], [6, 7])
        self.assertIn(rxn_2.atom_map[8], [7, 8])
        expected_xyz = {'symbols': ('C', 'C', 'N', 'O', 'N', 'N', 'H', 'H', 'H'),
                        'isotopes': (12, 12, 14, 16, 14, 14, 1, 1, 1),
                        'coords': ((-1.0108, -0.0114, -0.061), (0.478, 0.0191, 0.0139), (1.2974, -0.993, 0.4693),
                                   (0.6928, -1.9845, 0.8337), (1.7456, 1.9701, -0.6976), (1.1642, 1.0763, -0.3716),
                                   (-1.4329, -0.1554, 0.9349), (-1.402, 0.9134, -0.4821), (-1.3327, -0.8499, -0.6803))}
        self.assertEqual(mapped_product.get_xyz(), expected_xyz)

    def test_check_attributes(self):
        """Test checking the reaction attributes"""
        rxn_1 = ARCReaction(label='H + [O-][N+](=N)N=O <=> [N-]=[N+](N=O)[O] + H2',
                            r_species=[ARCSpecies(label='H', smiles='[H]'),
                                       ARCSpecies(label='[O-][N+](=N)N=O', smiles='[O-][N+](=N)N=O')],
                            p_species=[ARCSpecies(label='H2', smiles='[H][H]'),
                                       ARCSpecies(label='[N-]=[N+](N=O)[O]', smiles='[N-]=[N+](N=O)[O]')])
        rxn_1.check_attributes()
        self.assertEqual(rxn_1.reactants, ['H', '[O-][Np][=N]N=O'])
        self.assertEqual(rxn_1.products, ['H2', '[N-]=[Np][N=O][O]'])
        self.assertEqual(rxn_1.r_species[1].label, '[O-][Np][=N]N=O')

    def test_remove_dup_species(self):
        """Test the remove_dup_species function"""
        species_list = [ARCSpecies(label='OH', smiles='[OH]'),
                        ARCSpecies(label='OH', smiles='[OH]'),
                        ARCSpecies(label='H', smiles='[H]'),
                        ARCSpecies(label='H', smiles='[H]'),
                        ARCSpecies(label='H2O', smiles='O')]
        new_species_list = remove_dup_species(species_list=species_list)
        self.assertEqual(len(new_species_list), 3)

    def test_check_done_opt_r_n_p(self):
        """Test the check_done_opt_r_n_p() method"""
        c3_1_path = os.path.join(ARC_PATH, 'arc', 'testing', 'yml_testing', 'C3_1.yml')  # 1-propyl
        c3_2_path = os.path.join(ARC_PATH, 'arc', 'testing', 'yml_testing', 'C3_2.yml')  # 2-propyl
        c3_1_spc = ARCSpecies(yml_path=c3_1_path)
        c3_2_spc = ARCSpecies(yml_path=c3_2_path)
        rxn_1 = ARCReaction(r_species=[c3_1_spc], p_species=[c3_2_spc])
        self.assertIsNone(rxn_1.done_opt_r_n_p)
        rxn_1.check_done_opt_r_n_p()
        self.assertEqual(rxn_1.done_opt_r_n_p, True)

        rxn_2 = ARCReaction(r_species=[ARCSpecies(label='C1_3', smiles='[CH2]CC')],
                            p_species=[ARCSpecies(label='C3_2', smiles='C[CH]C')])
        rxn_2.check_done_opt_r_n_p()
        self.assertEqual(rxn_2.done_opt_r_n_p, False)

    def tests_white_space_in_reaction_label(self):
        """Test that an extra white space in the reaction label does not confuse ARC."""
        hno = ARCSpecies(label='HNO', smiles='N=O')
        h2no = ARCSpecies(label='H2NO', smiles='N[O]')
        no = ARCSpecies(label='NO', smiles='[N]=O')
        nh2oh = ARCSpecies(label='NH2OH', smiles='NO')
        rxn_1 = ARCReaction(label='HNO + H2NO <=> NO + NH2OH ',
                            r_species=[hno, h2no], p_species=[no, nh2oh])
        self.assertEqual(rxn_1.reactants, ['H2NO', 'HNO'])
        self.assertEqual(rxn_1.products, ['NH2OH', 'NO'])

        rxn_2 = ARCReaction(reaction_dict={'label': 'HNO + H2NO <=> NO + NH2OH '},
                            species_list=[hno, h2no, no, nh2oh])
        self.assertEqual(rxn_2.reactants, ['H2NO', 'HNO'])
        self.assertEqual(rxn_2.products, ['NH2OH', 'NO'])

    def tests_special_characters_in_reaction_label(self):
        """Test that the respective species objects can be identified from a reaction label with a special character."""
        no2 = ARCSpecies(label='NO2', smiles='[O]N=O')
        nh3o = ARCSpecies(label='[O-][NH3+]', smiles='[O-][NH3+]')
        hono = ARCSpecies(label='HONO', smiles='ON=O')
        h2no = ARCSpecies(label='H2NO', smiles='N[O]')
        rxn_1 = ARCReaction(label='NO2 + [O-][NH3+] <=> HONO + H2NO',
                            r_species=[no2, nh3o], p_species=[hono, h2no])
        self.assertEqual(rxn_1.reactants, ['NO2', '[O-][NH3p]'])
        self.assertEqual(rxn_1.products, ['H2NO', 'HONO'])

        rxn_2 = ARCReaction(reaction_dict={'label': 'NO2 + [O-][NH3+] <=> HONO + H2NO'},
                            species_list=[no2, nh3o, hono, h2no])
        self.assertEqual(rxn_2.reactants, ['NO2', '[O-][NH3p]'])
        self.assertEqual(rxn_2.products, ['H2NO', 'HONO'])

    def test_get_element_masses(self):
        """Test the get_element_masses() method."""
        self.assertTrue(almost_equal_lists(self.rxn1.get_element_mass(),
                                           [12.0, 1.00782503224, 1.00782503224, 1.00782503224,
                                            1.00782503224, 15.99491461957, 1.00782503224]))
        self.assertTrue(almost_equal_lists(self.rxn2.get_element_mass(),
                                           [12.0, 12.0, 1.00782503224, 1.00782503224, 1.00782503224, 1.00782503224,
                                            1.00782503224, 15.99491461957, 1.00782503224]))
        self.assertTrue(almost_equal_lists(self.rxn3.get_element_mass(),
                                           [12.0, 12.0, 14.00307400443, 1.00782503224, 1.00782503224, 1.00782503224,
                                            1.00782503224, 1.00782503224, 1.00782503224]))
        self.assertTrue(almost_equal_lists(self.rxn5.get_element_mass(),
                                           [14.00307400443, 1.00782503224, 1.00782503224,
                                            14.00307400443, 1.00782503224, 1.00782503224]))

    def test_get_bonds(self):
        """Test the get_bonds() method."""
        r_bonds, p_bonds = self.rxn_1_w_xyz.get_bonds()
        self.assertEqual(r_bonds, [(0, 1), (0, 2), (0, 3), (0, 4), (5, 6)])  # CH4 + OH
        self.assertEqual(p_bonds, [(0, 2), (0, 3), (0, 4), (5, 6), (1, 5)])  # CH3 + H2O

        r_bonds, p_bonds = self.rxn_13_w_xyz.get_bonds()
        self.assertEqual(r_bonds, [(0, 1), (1, 2), (3, 4), (3, 5), (3, 6), (4, 7), (4, 8)])  # HO2 + N2H4
        self.assertEqual(p_bonds, [(0, 1), (1, 2), (0, 5), (3, 4), (3, 6), (4, 7), (4, 8)])  # H2O2 + N2H3

        r_bonds, p_bonds = self.rxn_13_w_xyz.get_bonds(r_bonds_only=True)
        self.assertEqual(r_bonds, [(0, 1), (1, 2), (3, 4), (3, 5), (3, 6), (4, 7), (4, 8)])
        self.assertEqual(p_bonds, [])

    def test_get_formed_and_broken_bonds(self):
        """Test the get_formed_and_broken_bonds() function."""
        formed_bonds, broken_bonds = self.rxn_1_w_xyz.get_formed_and_broken_bonds()
        self.assertEqual(formed_bonds, [(1, 5)])
        self.assertEqual(broken_bonds, [(0, 1)])

        formed_bonds, broken_bonds = self.rxn_13_w_xyz.get_formed_and_broken_bonds()
        self.assertEqual(formed_bonds, [(0, 5)])
        self.assertEqual(broken_bonds, [(3, 5)])

    def test_get_changed_bonds(self):
        """Test the get_changed_bonds() function."""
        rxn_7 = ARCReaction(r_species=[ARCSpecies(label='C2H5NO2', smiles='[O-][N+](=O)CC',
                                                  xyz=""" O                  0.62193295    1.59121319   -0.58381518
                                                          N                  0.43574593    0.41740669    0.07732982
                                                          O                  1.34135576   -0.35713755    0.18815532
                                                          C                 -0.87783860    0.10001361    0.65582554
                                                          C                 -1.73002357   -0.64880063   -0.38564362
                                                          H                 -1.37248469    1.00642547    0.93625873
                                                          H                 -0.74723653   -0.51714586    1.52009245
                                                          H                 -1.23537748   -1.55521250   -0.66607681
                                                          H                 -2.68617014   -0.87982825    0.03543830
                                                          H                 -1.86062564   -0.03164117   -1.24991054""")],
                            p_species=[ARCSpecies(label='C2H5ONO', smiles='CCON=O',
                                                  xyz=""" O                  0.17295033    0.86074746   -0.26735563
                                                          N                  0.91672790   -0.24436868   -0.54142357
                                                          O                  1.79483464   -0.57413163    0.20189010
                                                          C                 -0.95986462    0.48910558    0.52227250
                                                          C                 -1.81700256   -0.52654731   -0.25577875
                                                          H                 -1.54504256    1.35857196    0.73789948
                                                          H                 -0.62677356    0.04723842    1.43808021
                                                          H                 -1.23182462   -1.39601368   -0.47140573
                                                          H                 -2.66463333   -0.80462899    0.33506188
                                                          H                 -2.15009363   -0.08468014   -1.17158646""")])
        changed_bonds = rxn_7.get_changed_bonds()
        self.assertEqual(changed_bonds, [(0, 1), (1, 2)])
        changed_bonds = self.rxn_1_w_xyz.get_changed_bonds()
        self.assertEqual(changed_bonds, [])

    def test_multi_reactants(self):
        """Test that a reaction can be defined with many (>3) reactants or products given ts_xyz_guess."""
        with self.assertRaises(ReactionError):
            ARCReaction(r_species=[ARCSpecies(label='C2H6', smiles='CC'),
                                   ARCSpecies(label='OH', smiles='[OH]'),
                                   ARCSpecies(label='H2O', smiles='O'),
                                   ARCSpecies(label='H2O', smiles='O')],
                        p_species=[ARCSpecies(label='C2H5', smiles='C[CH2]'),
                                   ARCSpecies(label='H2O', smiles='O'),
                                   ARCSpecies(label='H2O', smiles='O'),
                                   ARCSpecies(label='H2O', smiles='O')])

        ARCReaction(r_species=[ARCSpecies(label='C2H6', smiles='CC'),
                               ARCSpecies(label='OH', smiles='[OH]'),
                               ARCSpecies(label='H2O', smiles='O'),
                               ARCSpecies(label='H2O', smiles='O')],
                    p_species=[ARCSpecies(label='C2H5', smiles='C[CH2]'),
                               ARCSpecies(label='H2O', smiles='O'),
                               ARCSpecies(label='H2O', smiles='O'),
                               ARCSpecies(label='H2O', smiles='O')],
                    ts_xyz_guess=["""C      -3.61598200   -0.14868700   -0.24629000
                                    C      -2.87249000   -0.46501800   -1.37357200
                                    H      -1.49356900   -0.33047800   -1.35181200
                                    H      -0.84385200    0.11582900   -0.20370300
                                    H      -1.59912100    0.44823500    0.91958000
                                    H      -2.97672300    0.31399300    0.89602200
                                    H      -4.69390100   -0.24965500   -0.26210800
                                    H      -3.36776700   -0.80869200   -2.27293600
                                    O      -0.91216500   -0.55783000   -2.23774700
                                    H      -1.09587000    0.85059200    1.79098900
                                    O      -3.55685000    0.58245300    1.76982000
                                    H       0.61957500    0.23804900   -0.17583200
                                    H       1.21746400   -1.55405900    0.34558600
                                    O       2.65817800   -1.34488900    0.51012400
                                    H       3.16323600   -2.29177100    0.72530400
                                    H       3.06938800   -0.89699200   -0.39503300"""])

    def test_load_ts_xyz_user_guess_from_files(self):
        """Test various loading a reaction and populating the TS ARCSpecies with user xyz guesses from files"""
        project_directory = os.path.join(ARC_PATH, 'arc', 'testing', 'reactions', 'methanoate_hydrolysis')
        input_dict = read_yaml_file(path=os.path.join(project_directory, 'input_1.yml'))
        input_dict['project_directory'] = project_directory
        arc_object = ARC(**input_dict)
        Scheduler(project=arc_object.project,
                  species_list=arc_object.species,
                  rxn_list=arc_object.reactions,
                  conformer_opt_level=arc_object.conformer_opt_level,
                  opt_level=arc_object.opt_level,
                  freq_level=arc_object.freq_level,
                  sp_level=arc_object.sp_level,
                  scan_level=arc_object.scan_level,
                  ts_guess_level=arc_object.ts_guess_level,
                  ess_settings=arc_object.ess_settings,
                  job_types=arc_object.job_types,
                  project_directory=arc_object.project_directory,
                  ts_adapters=arc_object.ts_adapters,
                  testing=True,
                  )
        self.assertEqual(len(arc_object.reactions[0].ts_species.ts_guesses[0].initial_xyz['symbols']), 19)
        self.assertEqual(len(arc_object.reactions[0].ts_species.ts_guesses[1].initial_xyz['symbols']), 19)

        input_dict = read_yaml_file(path=os.path.join(project_directory, 'input_2.yml'))
        input_dict['project_directory'] = project_directory
        arc_object = ARC(**input_dict)
        Scheduler(project=arc_object.project,
                  species_list=arc_object.species,
                  rxn_list=arc_object.reactions,
                  conformer_opt_level=arc_object.conformer_opt_level,
                  opt_level=arc_object.opt_level,
                  freq_level=arc_object.freq_level,
                  sp_level=arc_object.sp_level,
                  scan_level=arc_object.scan_level,
                  ts_guess_level=arc_object.ts_guess_level,
                  ess_settings=arc_object.ess_settings,
                  job_types=arc_object.job_types,
                  project_directory=arc_object.project_directory,
                  ts_adapters=arc_object.ts_adapters,
                  testing=True,
                  )
        self.assertEqual(len(arc_object.reactions[0].ts_species.ts_guesses), 2)
        self.assertEqual(len(arc_object.reactions[0].ts_species.ts_guesses[1].initial_xyz['symbols']), 19)

    def test_get_rxn_smiles(self):
        """Tests the get_rxn_smiles method"""
        self.assertEqual(self.rxn1.get_rxn_smiles(), "C.[OH]>>[CH3].O")
        self.assertEqual(self.rxn2.get_rxn_smiles(), "C[CH2].[OH]>>C=C.O")
        self.assertEqual(self.rxn3.get_rxn_smiles(), "CC[NH]>>[CH2]CN")
        self.assertEqual(self.rxn4.get_rxn_smiles(), "[NH2].[NH]N>>N.[N]N")
        self.assertEqual(self.rxn5.get_rxn_smiles(), "[NH2].[NH2]>>[NH].N")
        self.assertEqual(self.rxn6.get_rxn_smiles(), "[NH2].[NH]N>>N.[N-]=[NH2+]")
        self.assertEqual(self.rxn7.get_rxn_smiles(), "[NH2].[NH]N>>N.[N]N")
        self.assertEqual(self.rxn8.get_rxn_smiles(), "C.[OH]>>[CH3].O")
        self.assertEqual(self.rxn9.get_rxn_smiles(), "NCO[O]>>C=N.[O]O")
        self.assertEqual(self.rxn10.get_rxn_smiles(), "N=O.[O-][N+]=O>>[O-][NH+]=O.[N]=O")
        self.assertEqual(self.rxn11.get_rxn_smiles(), "C[CH]C>>[CH2]CC")

    def test_atom_map_property(self):
        """Test that the atom map is saved in the reaction object, and that it is quick to restore it."""
        r_1 = ARCSpecies(label='CH2CHOH', smiles='C=CO', xyz=self.ch2choh_xyz)
        p_1 = ARCSpecies(label='CH3CHO', smiles='CC=O', xyz=self.ch3cho_xyz)
        rxn = ARCReaction(r_species=[r_1], p_species=[p_1])
        atom_map = rxn.atom_map
        self.assertTrue(check_atom_map(rxn))
        self.assertTrue(atom_map[:3], [0, 1, 2])
        self.assertIn(tuple(atom_map[3:5]+[atom_map[-1]]), permutations([3, 4, 5]))
        self.assertEqual(atom_map[5], 6)
        t0 = time.time()
        atom_map = rxn.atom_map
        t1 = time.time() - t0
        self.assertLess(t1, 0.1)
        self.assertTrue(atom_map[:3], [0, 1, 2])

    @classmethod
    def tearDownClass(cls):
        """A function that is run ONCE after all unit tests in this class."""
        project_directory = os.path.join(ARC_PATH, 'arc', 'testing', 'reactions', 'methanoate_hydrolysis')
        sub_folders = ['log_and_restart_archive', 'output']
        files_to_remove = ['arc.log']
        for sub_folder in sub_folders:
            shutil.rmtree(os.path.join(project_directory, sub_folder), ignore_errors=True)
        for file_path in files_to_remove:
            full_file_path = os.path.join(project_directory, file_path)
            if os.path.isfile(full_file_path):
                os.remove(full_file_path)
        file_paths = [os.path.join(ARC_PATH, 'arc', 'reaction', 'nul'), os.path.join(ARC_PATH, 'arc', 'reaction', 'run.out')]
        for file_path in file_paths:
            if os.path.isfile(file_path):
                os.remove(file_path)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
