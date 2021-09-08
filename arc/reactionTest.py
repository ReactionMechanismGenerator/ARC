#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.reaction module
"""

import os
import unittest

from rmgpy.reaction import Reaction
from rmgpy.species import Species

import arc.rmgdb as rmgdb
from arc.common import ARC_PATH
from arc.exceptions import ReactionError
from arc.reaction import ARCReaction, remove_dup_species
from arc.species import ARCSpecies


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
        cls.rmgdb = rmgdb.make_rmg_database_object()
        rmgdb.load_families_only(cls.rmgdb)
        cls.rxn1 = ARCReaction(reactants=['CH4', 'OH'], products=['CH3', 'H2O'],
                               rmg_reaction=Reaction(reactants=[Species(label='CH4', smiles='C'),
                                                                Species(label='OH', smiles='[OH]')],
                                                     products=[Species(label='CH3', smiles='[CH3]'),
                                                               Species(label='H2O', smiles='O')]))
        cls.rxn2 = ARCReaction(reactants=['C2H5', 'OH'], products=['C2H4', 'H2O'],
                               rmg_reaction=Reaction(reactants=[Species(label='C2H5', smiles='C[CH2]'),
                                                                Species(label='OH', smiles='[OH]')],
                                                     products=[Species(label='C2H4', smiles='C=C'),
                                                               Species(label='H2O', smiles='O')]))
        cls.rxn3 = ARCReaction(reactants=['CH3CH2NH'], products=['CH2CH2NH2'],
                               rmg_reaction=Reaction(reactants=[Species(label='CH3CH2NH', smiles='CC[NH]')],
                                                     products=[Species(label='CH2CH2NH2', smiles='[CH2]CN')]))
        cls.rxn4 = ARCReaction(reactants=['NH2', 'NH2NH'], products=['N', 'NH2N'],
                               rmg_reaction=Reaction(reactants=[Species(label='NH2', smiles='[NH2]'),
                                                                Species(label='NH2NH', smiles='N[NH]')],
                                                     products=[Species(label='N', smiles='N'),
                                                               Species(label='NH2N', smiles='N[N]')]))
        cls.rxn5 = ARCReaction(reactants=['NH2', 'NH2'], products=['NH', 'NH3'],
                               r_species=[ARCSpecies(label='NH2', smiles='[NH2]')],
                               p_species=[ARCSpecies(label='NH', smiles='[NH]'),
                                          ARCSpecies(label='NH3', smiles='N')])
        cls.rxn6 = ARCReaction(reactants=['NH2', 'N2H3'], products=['NH3', 'H2NN(S)'],
                               r_species=[ARCSpecies(label='NH2', smiles='[NH2]'),
                                          ARCSpecies(label='N2H3', smiles='N[NH]')],
                               p_species=[ARCSpecies(label='NH3', smiles='N'),
                                          ARCSpecies(label='H2NN(S)', adjlist="""multiplicity 1
                                                                                 1 N u0 p0 c+1 {2,D} {3,S} {4,S}
                                                                                 2 N u0 p2 c-1 {1,D}
                                                                                 3 H u0 p0 c0 {1,S}
                                                                                 4 H u0 p0 c0 {1,S}""")])
        cls.rxn7 = ARCReaction(reactants=['NH2', 'N2H3'], products=['NH3', 'H2NN(T)'],
                               r_species=[ARCSpecies(label='NH2', smiles='[NH2]'),
                                          ARCSpecies(label='N2H3', smiles='N[NH]')],
                               p_species=[ARCSpecies(label='NH3', smiles='N'),
                                          ARCSpecies(label='H2NN(T)', adjlist="""multiplicity 3
                                                                                 1 N u0 p1 c0 {2,S} {3,S} {4,S}
                                                                                 2 N u2 p1 c0 {1,S}
                                                                                 3 H u0 p0 c0 {1,S}
                                                                                 4 H u0 p0 c0 {1,S}""")])

    def test_str(self):
        """Test the string representation of the object"""
        str_representation = str(self.rxn1)
        self.assertEqual(self.rxn1.charge, 0)
        expected_representation = 'ARCReaction(label="CH4 + OH <=> CH3 + H2O", ' \
                                  'rmg_reaction="CH4 + OH <=> CH3 + H2O", ' \
                                  'multiplicity=2, charge=0)'
        self.assertEqual(str_representation, expected_representation)

    def test_as_dict(self):
        """Test ARCReaction.as_dict()"""
        self.rxn1.determine_family(self.rmgdb)
        rxn_dict_1 = self.rxn1.as_dict()
        # mol.atoms are not tested since all id's (including connectivity) changes depending on how the test is run.
        expected_dict_1 = {'charge': 0,
                           'family': 'H_Abstraction',
                           'family_own_reverse': True,
                           'index': None,
                           'label': 'CH4 + OH <=> CH3 + H2O',
                           'long_kinetic_description': '',
                           'multiplicity': 2,
                           'p_species': [{'arkane_file': None,
                                          'bond_corrections': {'C-H': 3},
                                          'charge': 0,
                                          'cheap_conformer': 'C       0.00000000    0.00000001   -0.00000000\n'
                                                             'H       1.06690511   -0.17519582    0.05416493\n'
                                                             'H      -0.68531716   -0.83753536   -0.02808565\n'
                                                             'H      -0.38158795    1.01273118   -0.02607927',
                                          'compute_thermo': True,
                                          'consider_all_diastereomers': True,
                                          'force_field': 'MMFF94s',
                                          'is_ts': False,
                                          'label': 'CH3',
                                          'long_thermo_description': "Bond corrections: {'C-H': 3}\n",
                                          'mol': {'atoms': rxn_dict_1['p_species'][0]['mol']['atoms'],
                                                  'multiplicity': 2,
                                                  'props': {}},
                                          'multiplicity': 2,
                                          'number_of_rotors': 0},
                                         {'arkane_file': None,
                                          'bond_corrections': {'H-O': 2},
                                          'charge': 0,
                                          'cheap_conformer': 'O      -0.00032832    0.39781490    0.00000000\n'
                                                             'H      -0.76330345   -0.19953755    0.00000000\n'
                                                             'H       0.76363177   -0.19827735    0.00000000',
                                          'compute_thermo': True,
                                          'consider_all_diastereomers': True,
                                          'force_field': 'MMFF94s',
                                          'is_ts': False,
                                          'label': 'H2O',
                                          'long_thermo_description': "Bond corrections: {'H-O': 2}\n",
                                          'mol': {'atoms': rxn_dict_1['p_species'][1]['mol']['atoms'],
                                                  'multiplicity': 1,
                                                  'props': {}},
                                          'multiplicity': 1,
                                          'number_of_rotors': 0}],
                           'products': ['CH3', 'H2O'],
                           'r_species': [{'arkane_file': None,
                                          'bond_corrections': {'C-H': 4},
                                          'charge': 0,
                                          'cheap_conformer': 'C      -0.00000000   -0.00000000    0.00000000\n'
                                                             'H      -0.63306457   -0.78034118   -0.42801448\n'
                                                             'H      -0.38919244    0.98049560   -0.28294367\n'
                                                             'H       0.00329661   -0.09013273    1.08846898\n'
                                                             'H       1.01896040   -0.11002169   -0.37751083',
                                          'compute_thermo': True,
                                          'consider_all_diastereomers': True,
                                          'force_field': 'MMFF94s',
                                          'is_ts': False,
                                          'label': 'CH4',
                                          'long_thermo_description': "Bond corrections: {'C-H': 4}\n",
                                          'mol': {'atoms': rxn_dict_1['r_species'][0]['mol']['atoms'],
                                                  'multiplicity': 1,
                                                  'props': {}},
                                          'multiplicity': 1,
                                          'number_of_rotors': 0},
                                         {'arkane_file': None,
                                          'bond_corrections': {'H-O': 1},
                                          'charge': 0,
                                          'cheap_conformer': 'O       0.48890387    0.00000000    0.00000000\n'
                                                             'H      -0.48890387    0.00000000    0.00000000',
                                          'compute_thermo': True,
                                          'consider_all_diastereomers': True,
                                          'force_field': 'MMFF94s',
                                          'is_ts': False,
                                          'label': 'OH',
                                          'long_thermo_description': "Bond corrections: {'H-O': 1}\n",
                                          'mol': {'atoms': rxn_dict_1['r_species'][1]['mol']['atoms'],
                                                  'multiplicity': 2,
                                                  'props': {}},
                                          'multiplicity': 2,
                                          'number_of_rotors': 0}],
                           'reactants': ['CH4', 'OH'],
                           'ts_label': None,
                           'ts_xyz_guess': []}
        self.assertEqual(rxn_dict_1, expected_dict_1)

        rxn_dict_6 = self.rxn6.as_dict()
        # The ``long_thermo_description`` attribute isn't deterministic (order could change)
        expected_dict_6 = {'charge': 0,
                           'family_own_reverse': 0,
                           'index': None,
                           'label': 'NH2 + N2H3 <=> NH3 + H2NN[S]',
                           'long_kinetic_description': '',
                           'multiplicity': 1,
                           'p_species': [{'arkane_file': None,
                                          'bond_corrections': {'H-N': 3},
                                          'charge': 0,
                                          'cheap_conformer': 'N       0.00064924   -0.00099698    0.29559292\n'
                                                             'H      -0.41786606    0.84210396   -0.09477452\n'
                                                             'H      -0.52039228   -0.78225292   -0.10002797\n'
                                                             'H       0.93760911   -0.05885406   -0.10079043',
                                          'compute_thermo': True,
                                          'consider_all_diastereomers': True,
                                          'force_field': 'MMFF94s',
                                          'is_ts': False,
                                          'label': 'NH3',
                                          'long_thermo_description': "Bond corrections: {'H-N': 3}\n",
                                          'mol': {'atoms': rxn_dict_6['p_species'][0]['mol']['atoms'],
                                                  'multiplicity': 1,
                                                  'props': {}},
                                          'multiplicity': 1,
                                          'number_of_rotors': 0},
                                         {'arkane_file': None,
                                          'bond_corrections': {'H-N': 2, 'N=N': 1},
                                          'charge': 0,
                                          'cheap_conformer': 'N      -0.08201544    0.01567102    0.28740725\n'
                                                             'N       1.12656450   -0.21525765   -0.48621674\n'
                                                             'H      -0.50742562   -0.72901556    0.83982059\n'
                                                             'H      -0.53712345    0.92860218    0.29862267',
                                          'compute_thermo': True,
                                          'consider_all_diastereomers': True,
                                          'force_field': 'MMFF94s',
                                          'is_ts': False,
                                          'label': 'H2NN[S]',
                                          'long_thermo_description': rxn_dict_6['p_species'][1]['long_thermo_description'],
                                          'mol': {'atoms': rxn_dict_6['p_species'][1]['mol']['atoms'],
                                                  'multiplicity': 1,
                                                  'props': {}},
                                          'multiplicity': 1,
                                          'number_of_rotors': 0,
                                          'original_label': 'H2NN(S)'}],
                           'products': ['H2NN[S]', 'NH3'],
                           'r_species': [{'arkane_file': None,
                                          'bond_corrections': {'H-N': 2},
                                          'charge': 0,
                                          'cheap_conformer': 'N       0.00016375    0.40059499    0.00000000\n'
                                                             'H      -0.83170922   -0.19995756    0.00000000\n'
                                                             'H       0.83154548   -0.20063742    0.00000000',
                                          'compute_thermo': True,
                                          'consider_all_diastereomers': True,
                                          'force_field': 'MMFF94s',
                                          'is_ts': False,
                                          'label': 'NH2',
                                          'long_thermo_description': "Bond corrections: {'H-N': 2}\n",
                                          'mol': {'atoms': rxn_dict_6['r_species'][0]['mol']['atoms'],
                                                  'multiplicity': 2,
                                                  'props': {}},
                                          'multiplicity': 2,
                                          'number_of_rotors': 0},
                                         {'arkane_file': None,
                                          'bond_corrections': {'H-N': 3, 'N-N': 1},
                                          'charge': 0,
                                          'cheap_conformer': 'N      -0.46751749    0.03795671    0.31180026\n'
                                                             'N       0.79325823   -0.46038094   -0.24114357\n'
                                                             'H      -1.19307188   -0.63034971    0.05027053\n'
                                                             'H      -0.69753009    0.90231202   -0.17907452\n'
                                                             'H       1.56486123    0.15046192    0.05814730',
                                          'compute_thermo': True,
                                          'consider_all_diastereomers': True,
                                          'force_field': 'MMFF94s',
                                          'is_ts': False,
                                          'label': 'N2H3',
                                          'long_thermo_description': rxn_dict_6['r_species'][1]['long_thermo_description'],
                                          'mol': {'atoms': rxn_dict_6['r_species'][1]['mol']['atoms'],
                                                  'multiplicity': 2,
                                                  'props': {}},
                                          'multiplicity': 2,
                                          'number_of_rotors': 0}],
                           'reactants': ['N2H3', 'NH2'],
                           'ts_label': None,
                           'ts_xyz_guess': []}
        self.assertEqual(rxn_dict_6, expected_dict_6)

    def test_from_dict(self):
        """Test ARCReaction.from_dict()"""
        rxn_dict = self.rxn1.as_dict()
        rxn = ARCReaction(reaction_dict=rxn_dict)
        self.assertEqual(rxn.label, 'CH4 + OH <=> CH3 + H2O')
        self.assertEqual(rxn.ts_methods, [tsm.lower() for tsm in default_ts_methods])

    def test_is_isomerization(self):
        """Test the is_isomerization() method"""
        self.assertFalse(self.rxn1.is_isomerization())
        self.assertFalse(self.rxn2.is_isomerization())
        self.assertTrue(self.rxn3.is_isomerization())
        self.assertFalse(self.rxn4.is_isomerization())
        self.assertFalse(self.rxn5.is_isomerization())
        self.assertFalse(self.rxn6.is_isomerization())
        self.assertFalse(self.rxn7.is_isomerization())

    def test_from_rmg_reaction(self):
        """Test setting up an ARCReaction from an RMG Reaction"""
        rmg_rxn_1 = Reaction(reactants=[Species(label='nC3H7', smiles='[CH2]CC')],
                             products=[Species(label='iC3H7', smiles='C[CH]C')])
        rxn_1 = ARCReaction(rmg_reaction=rmg_rxn_1)
        self.assertEqual(rxn_1.label, 'nC3H7 <=> iC3H7')

        rmg_rxn_2 = Reaction(reactants=[Species(label='OH', smiles='[OH]'), Species(label='OH', smiles='[OH]')],
                             products=[Species(label='O', smiles='[O]'), Species(label='H2O', smiles='O')])
        rxn_2 = ARCReaction(rmg_reaction=rmg_rxn_2)
        self.assertEqual(rxn_2.label, 'OH + OH <=> O + H2O')

    def test_from_rmg_reaction(self):
        """Test setting up an ARCReaction from an RMG Reaction"""
        rmg_rxn_1 = Reaction(reactants=[Species(label='nC3H7', smiles='[CH2]CC')],
                             products=[Species(label='iC3H7', smiles='C[CH]C')])
        rxn_1 = ARCReaction(rmg_reaction=rmg_rxn_1)
        self.assertEqual(rxn_1.label, 'nC3H7 <=> iC3H7')

        rmg_rxn_2 = Reaction(reactants=[Species(label='OH', smiles='[OH]'), Species(label='OH', smiles='[OH]')],
                             products=[Species(label='O', smiles='[O]'), Species(label='H2O', smiles='O')])
        rxn_2 = ARCReaction(rmg_reaction=rmg_rxn_2)
        self.assertEqual(rxn_2.label, 'OH + OH <=> O + H2O')

    def test_rmg_reaction_to_str(self):
        """Test the rmg_reaction_to_str() method and the reaction label generated"""
        spc1 = Species().from_smiles('CON=O')
        spc1.label = 'CONO'
        spc2 = Species().from_smiles('C[N+](=O)[O-]')
        spc2.label = 'CNO2'
        rmg_reaction = Reaction(reactants=[spc1], products=[spc2])
        rxn = ARCReaction(rmg_reaction=rmg_reaction)
        rxn_str = rxn.rmg_reaction_to_str()
        self.assertEqual(rxn_str, 'CON=O <=> [O-][N+](=O)C')
        self.assertEqual(rxn.label, 'CONO <=> CNO2')

    def test_rxn_family(self):
        """Test that ARC gets the correct RMG family for different reactions"""
        self.rxn1.determine_family(rmg_database=self.rmgdb)
        self.assertEqual(self.rxn1.family.label, 'H_Abstraction')
        self.assertTrue(self.rxn1.family_own_reverse)
        self.rxn2.determine_family(rmg_database=self.rmgdb)
        self.assertEqual(self.rxn2.family.label, 'Disproportionation')
        self.assertFalse(self.rxn2.family_own_reverse)
        self.rxn3.determine_family(rmg_database=self.rmgdb)
        self.assertEqual(self.rxn3.family.label, 'intra_H_migration')
        self.assertTrue(self.rxn3.family_own_reverse)
        self.rxn4.determine_family(rmg_database=self.rmgdb)
        self.assertEqual(self.rxn4.family.label, 'H_Abstraction')
        self.rxn5.rmg_reaction_from_arc_species()
        self.rxn5.check_attributes()
        self.rxn5.determine_family(rmg_database=self.rmgdb)
        self.assertEqual(self.rxn5.family.label, 'H_Abstraction')
        rxn_1 = ARCReaction(r_species=[ARCSpecies(label='C2H6', smiles='CC'),
                                       ARCSpecies(label='CCOOj', smiles='CCO[O]')],
                            p_species=[ARCSpecies(label='CCOOH', smiles='CCOO'),
                                       ARCSpecies(label='C2H5', smiles='C[CH2]')])
        rxn_1.check_attributes()
        rxn_1.determine_family(rmg_database=self.rmgdb)
        self.assertEqual(rxn_1.family.label, 'H_Abstraction')

    def test_charge_property(self):
        """Test determining charge"""
        self.assertEqual(self.rxn1.charge, 0)

    def test_multiplicity_property(self):
        """Test determining multiplicity"""
        self.assertEqual(self.rxn1.multiplicity, 2)
        self.rxn2.arc_species_from_rmg_reaction()
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

        # Legitimate reactions that previously failed in the atom balance test
        rxn5 = ARCReaction(reactants=['NH', '[O-][N+](=N)N'], products=['NH2', '[N-]=[N+]([O])N'],
                           r_species=[ARCSpecies(label='NH', smiles='[NH]'),
                                      ARCSpecies(label='[O-][N+](=N)N', smiles='[O-][N+](=N)N')],
                           p_species=[ARCSpecies(label='NH2', smiles='[NH2]'),
                                      ARCSpecies(label='[N-]=[N+]([O])N', smiles='[N-]=[N+]([O])N')])
        self.assertTrue(rxn5.check_atom_balance())
        rxn6 = ARCReaction(reactants=['N3O2', 'HON'], products=['NO', 'HN3O2'],
                           r_species=[ARCSpecies(label='N3O2', smiles='[N-]=[N+](N=O)[O]'),
                                      ARCSpecies(label='HON', smiles='[OH+]=[N-]')],
                           p_species=[ARCSpecies(label='NO', smiles='[N]=O'),
                                      ARCSpecies(label='HN3O2', smiles='[O-][N+](=N)N=O')])
        self.assertTrue(rxn6.check_atom_balance())

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

    def test_get_reactants_and_products(self):
        """Test getting reactants and products"""
        self.rxn1.arc_species_from_rmg_reaction()
        self.rxn1.remove_dup_species()
        reactants, products = self.rxn1.get_reactants_and_products(arc=True)
        for spc in reactants + products:
            self.assertIsInstance(spc, ARCSpecies)
        self.assertEqual(len(reactants), 2)
        self.assertEqual(len(products), 2)

        reactants, products = self.rxn1.get_reactants_and_products(arc=False)
        for spc in reactants + products:
            self.assertIsInstance(spc, Species)
        self.assertEqual(len(reactants), 2)
        self.assertEqual(len(products), 2)

        reactants, products = self.rxn5.get_reactants_and_products(arc=True)
        for spc in reactants + products:
            self.assertIsInstance(spc, ARCSpecies)
        self.assertEqual(len(reactants), 2)
        self.assertEqual(len(products), 2)
        self.assertEqual(reactants[0].label, reactants[1].label)

        reactants, products = self.rxn5.get_reactants_and_products(arc=False)
        for spc in reactants + products:
            self.assertIsInstance(spc, Species)
        self.assertEqual(len(reactants), 2)
        self.assertEqual(len(products), 2)
        self.assertNotEqual(products[0].label, products[1].label)

    def test_get_atom_map(self):
        """Test getting an atom map for a reaction"""

        # 1. trivial unimolecular: H2O <=> H2O
        h2o_xyz_1 = {'symbols': ('O', 'H', 'H'),
                     'isotopes': (16, 1, 1),
                     'coords': ((-0.0003283189391273643, 0.39781490416473486, 0.0),
                                (-0.7633034507689803, -0.19953755103743254, 0.0),
                                (0.7636317697081081, -0.19827735312730177, 0.0))}
        r_1 = ARCSpecies(label='H2O', smiles='O', xyz=h2o_xyz_1)
        p_1 = ARCSpecies(label='H2O', smiles='O', xyz=h2o_xyz_1)
        rxn_1 = ARCReaction(reactants=['H2O'], products=['H2O'], r_species=[r_1], p_species=[p_1])
        self.assertEqual(rxn_1.atom_map, [0, 1, 2])
        self.assertTrue(check_atom_map(rxn_1))

        # 2. trivial unimolecular with an intentional mixed atom order: H2O <=> H2O
        h2o_xyz_2 = {'symbols': ('H', 'H', 'O'),
                     'isotopes': (1, 1, 16),
                     'coords': ((0.39781, 0.0, -0.00032),
                                (-0.19953, 0.0, -0.76330),
                                (-0.19827, 0.0, 0.76363))}
        p_1 = ARCSpecies(label='H2O', smiles='O', xyz=h2o_xyz_2)
        rxn_2 = ARCReaction(reactants=['H2O'], products=['H2O'], r_species=[r_1], p_species=[p_1])
        self.assertEqual(rxn_2.atom_map, [2, 0, 1])
        self.assertTrue(check_atom_map(rxn_2))

        # 3. trivial bimolecular: H + CH3NH2 <=> H2 + CH2NH2
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
        h2_xyz = {'coords': ((0, 0, 0.3736550),
                             (0, 0, -0.3736550)),
                  'isotopes': (1, 1),
                  'symbols': ('H', 'H')}
        r_1 = ARCSpecies(label='H', smiles='[H]', xyz={'coords': ((0, 0, 0),), 'isotopes': (1,), 'symbols': ('H',)})
        r_2 = ARCSpecies(label='CH3NH2', smiles='CN', xyz=ch3nh2_xyz)
        p_1 = ARCSpecies(label='H2', smiles='[H][H]', xyz=h2_xyz)
        p_2 = ARCSpecies(label='CH2NH2', smiles='[CH2]N', xyz=ch2nh2_xyz)
        rxn_3 = ARCReaction(reactants=['H', 'CH3NH2'], products=['H2', 'CH2NH2'],
                            r_species=[r_1, r_2], p_species=[p_1, p_2])
        self.assertEqual(rxn_3.atom_map, [0, 2, 5, 6, 1, 7, 3, 4])
        self.assertTrue(check_atom_map(rxn_3))

        # 4. trivial bimolecular in reverse order: H + CH3NH2 <=> CH2NH2 + H2
        rxn_4 = ARCReaction(reactants=['H', 'CH3NH2'], products=['CH2NH2', 'H2'],
                            r_species=[r_1, r_2], p_species=[p_2, p_1])
        self.assertEqual(rxn_4.atom_map, [6, 0, 3, 4, 7, 5, 1, 2])
        self.assertTrue(check_atom_map(rxn_4))

        # 5. representative reactions from RMG families
        # 1+2_Cycloaddition: CH2 + C2H4 <=> C3H6
        ch2_xyz = {'coords': ((-1.3519460059345912e-10, -5.04203763365717e-10, 0.0),
                              (-1.064874800478917, -0.016329711355091817, 0.0),
                              (1.0648748006141107, 0.016329711859301474, 0.0)),
                   'isotopes': (12, 1, 1),
                   'symbols': ('C', 'H', 'H')}
        c2h4_xyz = {'coords': ((0.6664040429179742, 0.044298334171779405, -0.0050238049104911735),
                               (-0.6664040438461246, -0.04429833352898575, 0.00502380522486473),
                               (1.1686968388986039, 0.8743086488169786, -0.4919298928897832),
                               (1.2813853343929593, -0.7114426553520238, 0.4734595111827543),
                               (-1.2813853352424778, 0.7114426574294024, -0.4734595076873365),
                               (-1.1686968371212578, -0.8743086515369692, 0.49192988907998186)),
                    'isotopes': (12, 12, 1, 1, 1, 1),
                    'symbols': ('C', 'C', 'H', 'H', 'H', 'H')}
        c_c3h6_xyz = {'coords': ((0.7868661913782324, -0.3644249639827158, -0.016337299842911886),
                                 (-0.07793785747147405, 0.8603229755261934, 0.07746513362297117),
                                 (-0.708928275400647, -0.4958980792223481, -0.06112784358024908),
                                 (1.339749295484817, -0.5278616711993785, -0.9341881111902739),
                                 (1.3001119953298585, -0.6947493102195698, 0.8793780279658545),
                                 (-0.15055582331881673, 1.3597070015370083, 1.0367271647162946),
                                 (-0.11091839380255, 1.5265948517709569, -0.7768389650606503),
                                 (-1.1693748373792934, -0.7484015319217499, -1.0093221066790388),
                                 (-1.2090122948201234, -0.9152892722884018, 0.8042440000480116)),
                      'isotopes': (12, 12, 12, 1, 1, 1, 1, 1, 1),
                      'symbols': ('C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H')}

        r_1 = ARCSpecies(label='CH2', xyz=ch2_xyz, adjlist="""1 C u0 p1 c0 {2,S} {3,S}
                                                              2 H u0 p0 c0 {1,S}
                                                              3 H u0 p0 c0 {1,S}""")
        r_2 = ARCSpecies(label='C2H4', smiles='C=C', xyz=c2h4_xyz)
        p_1 = ARCSpecies(label='cC3H6', smiles='C1CC1', xyz=c_c3h6_xyz)
        rxn = ARCReaction(reactants=['CH2', 'C2H4'], products=['cC3H6'],
                          r_species=[r_1, r_2], p_species=[p_1])
        self.assertEqual(rxn.atom_map, [0, 7, 6, 1, 2, 5, 4, 8, 3])
        self.assertTrue(check_atom_map(rxn))

        # 1,2-Birad_to_alkene: SO2(T) => SO2(S)
        so2_t_xyz = {'coords': ((0.02724478716956233, 0.6093829407458188, 0.0),
                                (-1.3946381818031768, -0.24294788636871906, 0.0),
                                (1.3673933946336125, -0.36643505437710233, 0.0)),
                     'isotopes': (32, 16, 16),
                     'symbols': ('S', 'O', 'O')}
        so2_s_xyz = {'coords': ((-1.3554230894998571, -0.4084942756329785, 0.0),
                                (-0.04605352293144468, 0.6082507106551855, 0.0),
                                (1.4014766124312934, -0.19975643502220325, 0.0)),
                     'isotopes': (16, 32, 16),
                     'symbols': ('O', 'S', 'O')}

        r_1 = ARCSpecies(label='SO2(T)', smiles='O=[S][O]', multiplicity=3, xyz=so2_t_xyz)
        p_1 = ARCSpecies(label='SO2(S)', smiles='O=S=O', multiplicity=1, xyz=so2_s_xyz)
        rxn = ARCReaction(reactants=['SO2(T)'], products=['SO2(S)'],
                          r_species=[r_1], p_species=[p_1])
        self.assertEqual(rxn.atom_map, [1, 0, 2])
        self.assertTrue(check_atom_map(rxn))

        # 1,2_Insertion_CO: C4H10 + CO <=> C5H10O
        c4h10_xyz = {'coords': ((-0.5828455298013108, 1.3281531294599287, -0.04960015063595639),
                                (0.20452033859928953, 0.05503751610159247, -0.351590668388836),
                                (1.2187217734495472, -0.22435034939324036, 0.7553438935018645),
                                (-0.7402757883531311, -1.131897259046642, -0.526270047908048),
                                (-1.149632334529979, 1.2345299096044358, 0.8830543278319224),
                                (-1.2910247691071444, 1.5474495198220646, -0.8556442099189145),
                                (0.08958996004802251, 2.187049294072444, 0.047578963870699015),
                                (0.7510696374695547, 0.20211678856476709, -1.2911649516059494),
                                (1.9161788635733445, 0.6129834282608764, 0.8637033961259424),
                                (0.723393227383255, -0.37955365746174813, 1.7199258030015812),
                                (1.8052293751859985, -1.1207509229675587, 0.5277678765569422),
                                (-1.4506401201091412, -0.9467671747910582, -1.3389353480864132),
                                (-1.31330819789714, -1.3230974306153704, 0.3874767468986707),
                                (-0.18097643591114793, -2.04090279161046, -0.7716456312435797)),
                     'isotopes': (12, 12, 12, 12, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
                     'symbols': ('C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H')}
        co_xyz = {'coords': ((0, 0, -0.6748240),
                             (0, 0, 0.5061180)),
                  'isotopes': (12, 16),
                  'symbols': ('C', 'O')}
        c5h10o_xyz = {'coords': ((1.4311352287218408, -0.1713595727440808, -0.4215888483848517),
                                 (-0.007186117613478591, 0.06984820110647515, 0.04712543561838732),
                                 (-0.9581449869575146, 0.0768516496023853, -1.153820345745391),
                                 (-0.42459441572492335, -1.0196556708425513, 1.0398144790596706),
                                 (-0.06395445555126768, 1.4459669990683603, 0.6988370467311186),
                                 (-0.39842691952831133, 1.6544415349370807, 1.860895997103657),
                                 (1.7538538399565853, 0.5988164487250668, -1.1317944597170102),
                                 (1.5308761668570723, -1.1450780226312873, -0.9137377478255552),
                                 (2.130467943093651, -0.145756780679422, 0.4221764324976206),
                                 (-1.9882342251557934, 0.2821166362714845, -0.8400630940054319),
                                 (-0.6807867076715277, 0.8517398665867646, -1.8779276281234922),
                                 (-0.9490513003000888, -0.8874499123119038, -1.6737493906621435),
                                 (0.23329847490706446, -1.0315570674753483, 1.9164599735169805),
                                 (-0.3863240121264062, -2.0126378831961222, 0.578337115559457),
                                 (-1.4463966539332702, -0.8570614833514035, 1.4016914821743647),
                                 (0.22346814102625032, 2.2907750569345855, 0.04734355220249537)),
                      'isotopes': (12, 12, 12, 12, 12, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
                      'symbols': ('C', 'C', 'C', 'C', 'C', 'O', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H')}
        r_1 = ARCSpecies(label='C4H10', smiles='CC(C)C', xyz=c4h10_xyz)
        r_2 = ARCSpecies(label='CO', smiles='[C-]#[O+]', xyz=co_xyz)
        p_1 = ARCSpecies(label='C5H10O', smiles='CC(C)(C)C=O', xyz=c5h10o_xyz)
        rxn = ARCReaction(reactants=['C4H10', 'CO'], products=['C5H10O'],
                          r_species=[r_1, r_2], p_species=[p_1])
        self.assertEqual(rxn.atom_map, [2, 1, 3, 0, 10, 9, 15, 11, 14, 13, 12, 6, 7, 8, 4, 5])
        self.assertTrue(check_atom_map(rxn))
        # same reaction in reverse:
        rxn_rev = ARCReaction(reactants=['C4H10', 'CO'], products=['C5H10O'],
                              r_species=[p_1], p_species=[r_1, r_2])
        rxn.atom_map = None  # Reset the ._atom_map property so it'll be recalculated.
        self.assertEqual(rxn_rev.atom_map, [3, 1, 0, 2, 14, 15, 11, 12, 13, 5, 4, 7, 10, 9, 8, 6])
        self.assertTrue(check_atom_map(rxn_rev))

        # 1,2_Insertion_carbene: CH2 + CH3CHCH2 <=> CH2C(CH3)CH3
        ch3chch2_xyz = {'coords': ((1.1254127400230443, -0.3017844766611556, -0.7510291174036663),
                                   (0.28418579724689313, 0.4695373959408027, -0.051603271988589404),
                                   (-0.9765320571742121, -0.022861128064798977, 0.5796557618407167),
                                   (2.0283411338639996, 0.11473658845658036, -1.1869837855667502),
                                   (0.9363880582389235, -1.3596076643472066, -0.9052806971855372),
                                   (0.515236677022072, 1.5247207956403894, 0.0767856648621589),
                                   (-1.834273253628283, 0.5215277899266343, 0.1730882558342602),
                                   (-0.9426369634676213, 0.14532401512967796, 1.660369021940393),
                                   (-1.1361221321267199, -1.0915933160227609, 0.40499816767425895)),
                        'isotopes': (12, 12, 12, 1, 1, 1, 1, 1, 1),
                        'symbols': ('C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H')}
        ch2c_ch3_ch3_xyz = {'coords': ((1.311440657629446, -0.8232491279762378, -0.014011949069650012),
                                       (0.17465327546657444, -0.10963752839647688, -0.0018661407331225198),
                                       (0.15987152622943782, 1.3695567386476604, -0.2588365812127651),
                                       (-1.1598795338142822, -0.7418076224229717, 0.26952094648817027),
                                       (1.3132561268001393, -1.892944418795293, 0.17288836586596953),
                                       (2.2726500205330566, -0.3580870043824404, -0.21120172695017245),
                                       (-0.24975928963985833, 1.9011183874772648, 0.606181008492715),
                                       (1.163527517827743, 1.764347378625805, -0.44883027337458375),
                                       (-0.4592780349942656, 1.5969487568042877, -1.1326541018396823),
                                       (-1.620301775899056, -0.29150503408945216, 1.1548738224107944),
                                       (-1.0763585161942282, -1.8190672015429195, 0.4478982868219457),
                                       (-1.829821973944695, -0.5956733239492041, -0.5839616568996387)),
                            'isotopes': (12, 12, 12, 12, 1, 1, 1, 1, 1, 1, 1, 1),
                            'symbols': ('C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H')}

        r_1 = ARCSpecies(label='CH2', xyz=ch2_xyz, adjlist="""1 C u0 p1 c0 {2,S} {3,S}
                                                              2    H u0 p0 c0 {1,S}
                                                              3    H u0 p0 c0 {1,S}""")
        r_2 = ARCSpecies(label='CH3CHCH2', smiles='C=CC', xyz=ch3chch2_xyz)
        p_1 = ARCSpecies(label='CH2C(CH3)CH3', smiles='C=C(C)C', xyz=ch2c_ch3_ch3_xyz)
        rxn = ARCReaction(reactants=['CH3CHCH2', 'CH2'], products=['CH2C(CH3)CH3'],
                          r_species=[r_1, r_2], p_species=[p_1])
        self.assertEqual(rxn.atom_map, [1, 8, 6, 2, 0, 3, 5, 4, 7, 11, 10, 9])
        self.assertTrue(check_atom_map(rxn))

        # 1,2_NH3_elimination: NCC <=> C2H4 + NH3
        ncc_xyz = {'coords': ((1.1517341397735719, -0.37601689454792764, -0.5230788502681245),
                              (0.2893395715754821, 0.449973844025586, 0.3114935868175311),
                              (-1.1415136758153028, -0.05605900830417449, 0.25915656466172177),
                              (1.1385906595862103, -1.3375972344465683, -0.18540144244632334),
                              (2.115146540825731, -0.05549033543399281, -0.4352422172888292),
                              (0.6517228987651973, 0.4341829477365257, 1.3446618712379401),
                              (0.32794656354609036, 1.4855039198343405, -0.04141588729000556),
                              (-1.2132836539673237, -1.083868352883045, 0.6307611987658565),
                              (-1.7869541982465666, 0.5726121409749625, 0.8809463351815471),
                              (-1.5327288460430881, -0.03324102695569548, -0.763616120234416)),
                   'isotopes': (14, 12, 12, 1, 1, 1, 1, 1, 1, 1),
                   'symbols': ('N', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H')}
        c2h4_xyz = {'coords': ((0.6664040429179742, 0.044298334171779405, -0.0050238049104911735),
                               (-0.6664040438461246, -0.04429833352898575, 0.00502380522486473),
                               (1.1686968388986039, 0.8743086488169786, -0.4919298928897832),
                               (1.2813853343929593, -0.7114426553520238, 0.4734595111827543),
                               (-1.2813853352424778, 0.7114426574294024, -0.4734595076873365),
                               (-1.1686968371212578, -0.8743086515369692, 0.49192988907998186)),
                    'isotopes': (12, 12, 1, 1, 1, 1),
                    'symbols': ('C', 'C', 'H', 'H', 'H', 'H')}
        nh3_xyz = {'coords': ((0.0006492354002636227, -0.0009969784288894215, 0.2955929244020652),
                              (-0.4178660616416419, 0.842103963871788, -0.09477452075659776),
                              (-0.5203922802597125, -0.7822529247012627, -0.10002797449860866),
                              (0.9376091065010891, -0.05885406074163403, -0.10079042914685925)),
                   'isotopes': (14, 1, 1, 1),
                   'symbols': ('N', 'H', 'H', 'H')}

        r_1 = ARCSpecies(label='NCC', smiles='NCC', xyz=ncc_xyz)
        p_1 = ARCSpecies(label='C2H4', smiles='C=C', xyz=c2h4_xyz)
        p_2 = ARCSpecies(label='NH3', smiles='N', xyz=nh3_xyz)
        rxn = ARCReaction(reactants=['NCC'], products=['C2H4', 'NH3'], r_species=[r_1], p_species=[p_1, p_2])
        self.assertEqual(rxn.atom_map, [6, 0, 1, 3, 2, 9, 5, 8, 4, 7])
        self.assertTrue(check_atom_map(rxn))

        # Cyclopentadiene_scission: C6H6 <=> C6H6_2
        c6h6_a_xyz = {'coords': ((1.465264096022479, 0.3555098886638667, 0.15268159347190322),
                                 (0.4583546746026421, 1.1352991023740606, -0.26555553330413073),
                                 (-0.7550043760214846, 0.35970165318809594, -0.5698935045151712),
                                 (-1.485327813119871, -0.35660657095915016, 0.46119177830578917),
                                 (-0.3414477960946828, -1.060779229397218, -0.11686056681841692),
                                 (0.9879417277856641, -1.006839916409751, 0.12489717473407935),
                                 (2.4630837864551887, 0.6629994259328668, 0.4197578798464181),
                                 (0.5110882588097015, 2.2100951208919897, -0.3734820378556644),
                                 (-1.1192886361027838, 0.384286081689225, -1.5897813181530946),
                                 (-2.453224961870327, -0.7758708758357847, 0.2158838729688473),
                                 (-1.3859013659398718, -0.054382091828296085, 1.4971154213962072),
                                 (1.6544624054733257, -1.8534125883098933, 0.0440452399232336)),
                      'isotopes': (12, 12, 12, 12, 12, 12, 1, 1, 1, 1, 1, 1),
                      'symbols': ('C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H')}
        c6h6_b_xyz = {'coords': ((-1.474267041853848, 0.27665693719971857, -0.31815898666696507),
                                 (-0.25527025747758825, 1.1936776717612125, -0.2432148642540069),
                                 (0.9917471212521393, 0.7578589393970138, 0.059037260524552534),
                                 (1.2911962562420976, -0.6524103892231805, 0.34598643264742923),
                                 (0.321535921890914, -1.5867102018006056, 0.32000545365633654),
                                 (-0.9417407846918554, -1.043897260224426, -0.002820356559266387),
                                 (-2.2262364004658077, 0.5956762298613206, 0.40890113659975075),
                                 (-1.90597332290244, 0.31143075666839354, -1.3222845692785703),
                                 (-0.4221153027089989, 2.2469871640348815, -0.4470234892644997),
                                 (1.824518548011024, 1.4543788790156666, 0.0987362566117616),
                                 (2.3174577767359237, -0.9162726684959432, 0.5791638390925197),
                                 (0.4791474859684761, -2.637376058194065, 0.5216718868909702)),
                      'isotopes': (12, 12, 12, 12, 12, 12, 1, 1, 1, 1, 1, 1),
                      'symbols': ('C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H')}
        r_1 = ARCSpecies(label='C6H6_1', smiles='C1=CC2CC2=C1', xyz=c6h6_a_xyz)
        p_1 = ARCSpecies(label='C6H6_b', xyz=c6h6_b_xyz, adjlist="""multiplicity 1
                                                                    1  C u0 p0 c0 {2,S} {6,S} {7,S} {8,S}
                                                                    2  C u0 p0 c0 {1,S} {3,D} {9,S}
                                                                    3  C u0 p0 c0 {2,D} {4,S} {10,S}
                                                                    4  C u0 p0 c0 {3,S} {5,D} {11,S}
                                                                    5  C u0 p0 c0 {4,D} {6,S} {12,S}
                                                                    6  C u0 p1 c0 {1,S} {5,S}
                                                                    7  H u0 p0 c0 {1,S}
                                                                    8  H u0 p0 c0 {1,S}
                                                                    9  H u0 p0 c0 {2,S}
                                                                    10 H u0 p0 c0 {3,S}
                                                                    11 H u0 p0 c0 {4,S}
                                                                    12 H u0 p0 c0 {5,S}""")
        rxn = ARCReaction(reactants=['C6H6_1'], products=['C6H6_b'], r_species=[r_1], p_species=[p_1])
        self.assertEqual(rxn.atom_map, [1, 4, 2, 0, 5, 3, 10, 9, 8, 7, 6, 11])
        self.assertTrue(check_atom_map(rxn))

        # Diels_alder_addition: C5H8 + C6H10 <=> C11H18
        c5h8_xyz = {'coords': ((2.388426506127341, -0.6020682478448856, -0.8986239521455471),
                               (1.396815470095451, 0.2559764141247285, -0.632876393172657),
                               (0.15313289103802616, -0.14573699483201027, -0.021031021618524288),
                               (-0.8389550397179193, 0.7169723970589436, 0.24404493146763848),
                               (-2.134416163598417, 0.3239681447826502, 0.876570575233924),
                               (2.3152942675091053, -1.659966892807757, -0.6677097540505594),
                               (3.307483005086718, -0.2549641760041707, -1.3602353900805908),
                               (1.5244650427331894, 1.3064325129702357, -0.885889345476673),
                               (0.03466763777311284, -1.1983886005211812, 0.22843510909849465),
                               (-0.7292165776910954, 1.7707854265126919, -0.0018778749819206306),
                               (-2.2727957408233483, 0.8723614467877541, 1.8133393569164875),
                               (-2.964923876317021, 0.5697550629604972, 0.20779128725191479),
                               (-2.179977422215191, -0.7469982986693162, 1.0980624715582143)),
                    'isotopes': (12, 12, 12, 12, 12, 1, 1, 1, 1, 1, 1, 1, 1),
                    'symbols': ('C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H')}
        c6h10_xyz = {'coords': ((-3.1580683741393027, 0.12435732050743648, 0.062246914825226686),
                                (-1.7707067002277326, -0.16620360178113405, 0.5349691635238483),
                                (-0.6661431116320429, 0.14509293595419076, -0.15922278773395507),
                                (0.6661430871150528, -0.14509454337155672, 0.31675535258860427),
                                (1.7707067221399997, 0.1662031310249801, -0.37743578397831457),
                                (3.158068300764984, -0.1243566042028104, 0.0952878105284522),
                                (-3.168673040679144, 0.6195109523415628, -0.9138744940397917),
                                (-3.725598607890039, -0.8074661408951581, -0.02161915764945937),
                                (-3.667502866324593, 0.7743593207314634, 0.7801554985274884),
                                (-1.689979340654408, -0.658941962521633, 1.5012451075034898),
                                (-0.7566435422183935, 0.638020444531764, -1.1251628620848497),
                                (0.7566435677358615, -0.6380221416323665, 1.2826952112274577),
                                (1.6899798722782449, 0.6589420540549689, -1.3437113386443575),
                                (3.667504053415929, -0.7743586338884942, -0.6226199722740314),
                                (3.168672421569792, -0.6195094502282033, 1.0714098656973499),
                                (3.7255975587458146, 0.8074669193750107, 0.17915450597834431)),
                     'isotopes': (12, 12, 12, 12, 12, 12, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
                     'symbols': ('C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H')}
        c11h18_xyz = {'coords': ((-2.703705731925332, -1.6725650155987697, -0.8086740370494231),
                                 (-1.5787486719767332, -0.9567028627407828, -0.9374280026568313),
                                 (-1.0380952176862677, -0.018956959638684944, 0.13044184610143217),
                                 (-0.9559942521379576, 1.4339345435700377, -0.4211901270741127),
                                 (-1.0332585693389826, 2.4699679588763424, 0.7057994083773913),
                                 (0.29330767863777807, 1.6799654965887798, -1.231863983890581),
                                 (1.3587533124739988, 0.8659977820300574, -1.2254134762392683),
                                 (1.4646089005142253, -0.38325835216422677, -0.38135979446817597),
                                 (2.8541328762948206, -0.44094297285873296, 0.26496359509221906),
                                 (0.34006414566475085, -0.47702332337899384, 0.6874327625078416),
                                 (0.25971352010871435, -1.877767868723041, 1.3110012371667645),
                                 (-3.0417827308140666, -2.322041441265777, -1.6103599735778764),
                                 (-3.3090907216942607, -1.6304164329807997, 0.09119562710875757),
                                 (-1.0140304257321437, -1.0355853116087828, -1.8648599111730466),
                                 (-1.7519605248870382, -0.02119386222025415, 0.9668436940879309),
                                 (-1.81873965484405, 1.6086866163315763, -1.078721360123664),
                                 (-1.0435582627465962, 3.4878991458476953, 0.2994048809648787),
                                 (-1.951192645793586, 2.339835712359188, 1.2889715104672264),
                                 (-0.1829564138869207, 2.3964873273832725, 1.392011686068868),
                                 (0.30323343644376083, 2.563802985828772, -1.8662760923115096),
                                 (2.19957127963344, 1.1019579368769516, -1.875167073357893),
                                 (1.38566629304311, -1.2380356901050797, -1.0661633096552972),
                                 (3.643719857910847, -0.3537387754651392, -0.49036436481236706),
                                 (2.9928305557668415, 0.3709281222195243, 0.9878133651866298),
                                 (3.0131256633168944, -1.3901032201204848, 0.7853911163504533),
                                 (0.5952865875235361, 0.20718050090958492, 1.5092530947735325),
                                 (-0.5854224393429074, -1.9456691543733, 2.004496412992174),
                                 (1.163930543826011, -2.107144483058995, 1.883011771518183),
                                 (0.1405916116476466, -2.6554984025201795, 0.5498094976254133)),
                      'isotopes': (12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
                                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
                      'symbols': ('C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'H',
                                  'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H')}
        r_1 = ARCSpecies(label='C5H8', smiles='C=CC=CC', xyz=c5h8_xyz)
        r_2 = ARCSpecies(label='C6H10', smiles='CC=CC=CC', xyz=c6h10_xyz)
        p_1 = ARCSpecies(label='C11H18', smiles='C=CC1C(C)C=CC(C)C1C', xyz=c11h18_xyz)
        rxn = ARCReaction(reactants=['C5H8', 'C6H10'], products=['C11H18'], r_species=[r_1, r_2], p_species=[p_1])
        self.assertEqual(rxn.atom_map, [4, 9, 2, 5, 10, 20, 15, 27, 18, 16, 17, 14, 25, 0, 1,
                                        3, 6, 7, 8, 22, 19, 26, 21, 24, 23, 28, 13, 12, 11])
        self.assertTrue(check_atom_map(rxn))

        # Disproportionation: C4H7 + O2 <=> HO2 + C4H6
        c4h7_xyz = {'coords': ((-2.040921404503424, -0.12903384637698798, 0.1559892045303822),
                               (-0.7546540332943176, -0.4098957103161423, -0.07681407943731554),
                               (0.3137517227573887, 0.47379064315829633, 0.303025839828397),
                               (0.09502978026667419, 1.3942096052269417, 0.834199535314798),
                               (1.734012668678617, 0.08135386277553083, 0.0906583073064836),
                               (-2.352694765027891, 0.7875623110661275, 0.6465408489803196),
                               (-2.8196259554078997, -0.8225020395029484, -0.14562281062524043),
                               (-0.49241522029670126, -1.3423933663677394, -0.5719234920796793),
                               (2.1384679029944533, -0.37774268314938586, 0.9976275308860051),
                               (2.3331746195641365, 0.9677350359207318, -0.13953736233285224),
                               (1.8458746842689961, -0.623083812434417, -0.7393520512375593)),
                    'isotopes': (12, 12, 12, 1, 12, 1, 1, 1, 1, 1, 1),
                    'symbols': ('C', 'C', 'C', 'H', 'C', 'H', 'H', 'H', 'H', 'H', 'H')}
        o2_xyz = {'coords': ((0, 0, 0.6487420),
                             (0, 0, -0.6487420)),
                  'isotopes': (16, 16),
                  'symbols': ('O', 'O')}
        ho2_xyz = {'coords': ((0.0558910, -0.6204870, 0.0000000),
                              (0.0558910, 0.7272050, 0.0000000),
                              (-0.8942590, -0.8537420, 0.0000000)),
                   'isotopes': (16, 16, 1),
                   'symbols': ('O', 'O', 'H')}
        c4h6_xyz = {'coords': ((-1.1313721520581368, 0.4375787725187425, 1.3741095482244203),
                               (-0.5236696446754213, -0.27046339876338915, 0.4152401808417905),
                               (0.5236696150303143, 0.2704633473040529, -0.41524017130113694),
                               (1.1313721685204072, -0.4375787650279751, -1.3741095524658273),
                               (-0.8696512779281117, 1.4694838181320669, 1.5851480041034802),
                               (-1.915706463982211, -0.010750118295295768, 1.9758596362701513),
                               (-0.8263303869083625, -1.301920739528746, 0.24674332317151054),
                               (0.8263303006084768, 1.3019207019374226, -0.24674330995607902),
                               (1.9157064555415753, 0.010750214228214268, -1.9758596165946158),
                               (0.8696513858514865, -1.469483832503432, -1.5851480422945032)),
                    'isotopes': (12, 12, 12, 12, 1, 1, 1, 1, 1, 1),
                    'symbols': ('C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H')}
        r_1 = ARCSpecies(label='C4H7', smiles='C=C[CH]C', xyz=c4h7_xyz)
        r_2 = ARCSpecies(label='O2', smiles='[O][O]', xyz=o2_xyz)
        p_1 = ARCSpecies(label='HO2', smiles='O[O]', xyz=ho2_xyz)
        p_2 = ARCSpecies(label='C4H6', smiles='C=CC=C', xyz=c4h6_xyz)
        rxn = ARCReaction(reactants=['C4H7', 'O2'], products=['HO2', 'C4H6'],
                          r_species=[r_1, r_2], p_species=[p_1, p_2])
        self.assertEqual(rxn.atom_map, [3, 4, 5, 10, 6, 8, 11, 7, 9, 12, 2, 0, 1])
        self.assertTrue(check_atom_map(rxn))

        # Disproportionation: HO2 + NHOH <=> NH2OH + O2
        nhoh_xyz = {'coords': ((0.5055094877826753, 0.03248552573561613, -0.443416250587286),
                               (1.392367115364475, -0.021750569314658803, 0.07321920788090872),
                               (-0.570163178752975, -0.035696714715839996, 0.48914535186936214),
                               (-1.3277134243941644, 0.024961758294888944, -0.11894830916297996)),
                    'isotopes': (14, 1, 16, 1),
                    'symbols': ('N', 'H', 'O', 'H')}
        nh2oh_xyz = {'coords': ((-0.442357984214193, 0.12755746178283767, -0.283450834226086),
                                (0.8066044298181865, -0.19499391813986608, 0.38695057103192726),
                                (-0.9953709942529645, -0.7170738803381369, -0.11579136415085267),
                                (-0.8349557675364339, 0.8418549600381088, 0.33540420712720587),
                                (1.4660803161854115, -0.05734462334294334, -0.32311257978218827)),
                     'isotopes': (14, 16, 1, 1, 1),
                     'symbols': ('N', 'O', 'H', 'H', 'H')}
        r_1 = ARCSpecies(label='NHOH', smiles='[NH]O', xyz=nhoh_xyz)
        r_2 = ARCSpecies(label='HO2', smiles='O[O]', xyz=ho2_xyz)
        p_1 = ARCSpecies(label='O2', smiles='[O][O]', xyz=o2_xyz)
        p_2 = ARCSpecies(label='NH2OH', smiles='NO', xyz=nh2oh_xyz)
        rxn = ARCReaction(reactants=['NHOH', 'HO2'], products=['O2', 'NH2OH'],
                          r_species=[r_1, r_2], p_species=[p_1, p_2])
        self.assertEqual(rxn.atom_map, [2, 6, 0, 5, 3, 1, 4])
        self.assertTrue(check_atom_map(rxn))

        # HO2_Elimination_from_PeroxyRadical: C2H5O3 <=> C2H4O + HO2
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
        r_1 = ARCSpecies(label='C2H5O3', smiles='CC(O)O[O]', xyz=c2h5o3_xyz)
        p_1 = ARCSpecies(label='C2H4O', smiles='CC=O', xyz=c2h4o_xyz)
        p_2 = ARCSpecies(label='HO2', smiles='O[O]', xyz=ho2_xyz)
        rxn = ARCReaction(reactants=['C2H5O3'], products=['HO2', 'C2H4O'],
                          r_species=[r_1], p_species=[p_1, p_2])
        self.assertEqual(rxn.atom_map, [0, 1, 2, 8, 7, 4, 9, 5, 3, 6])
        self.assertTrue(check_atom_map(rxn))

        # H_Abstraction: C3H6O + C4H9O <=> C3H5O + C4H10O
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
        self.assertEqual(rxn.atom_map, [12, 1, 11, 13, 2, 19, 6, 7, 14, 21, 9, 15,
                                        8, 10, 0, 3, 4, 20, 17, 16, 5, 23, 18, 22])
        self.assertTrue(check_atom_map(rxn))

        # H_Abstraction: NH + N2H3 <=> NH2 + N2H2(T)
        nh_xyz = {'coords': ((0.509499983131626, 0.0, 0.0), (-0.509499983131626, 0.0, 0.0)),
                  'isotopes': (14, 1),
                  'symbols': ('N', 'H')}
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
        n2h3_t_xyz = {'coords': ((0.5974274138372041, -0.41113104979405946, 0.08609839663782763),
                                 (1.421955422639823, 0.19737093442024492, 0.02508578507394823),
                                 (-0.5974274348582206, 0.41113108883884353, -0.08609846602622732),
                                 (-1.4219554016188147, -0.19737097346502322, -0.02508571568554942)),
                      'isotopes': (14, 1, 14, 1),
                      'symbols': ('N', 'H', 'N', 'H')}
        r_1 = ARCSpecies(label='NH', smiles='[NH]', xyz=nh_xyz)
        r_2 = ARCSpecies(label='N2H3', smiles='N[NH]', xyz=n2h3_xyz)
        p_1 = ARCSpecies(label='NH2', smiles='[NH2]', xyz=nh2_xyz)
        p_2 = ARCSpecies(label='N2H2(T)', smiles='[NH][NH]', xyz=n2h3_t_xyz)
        rxn = ARCReaction(reactants=['NH', 'N2H3'], products=['NH2', 'N2H2(T)'],
                          r_species=[r_1, r_2], p_species=[p_1, p_2])
        self.assertEqual(rxn.atom_map, [0, 1, 3, 5, 4, 6, 2])
        self.assertTrue(check_atom_map(rxn))

        # Intra_Disproportionation: C10H10_a <=> C10H10_b
        c10h10_a_xyz = {'coords': ((3.1623638230700997, 0.39331289450005563, -0.031839117414963584),
                                   (1.8784852381397288, 0.037685951926618944, -0.13659028131444134),
                                   (0.9737380560194014, 0.5278617594060281, -1.1526858375270472),
                                   (1.2607098516126556, 1.1809007875206383, -1.9621017164412065),
                                   (-0.36396095305912823, -0.13214785064139675, -1.0200667625809143),
                                   (-1.5172464644867296, 0.8364138939810618, -1.0669384323486588),
                                   (-2.4922101649968655, 0.8316551483126366, -0.14124720277902958),
                                   (-2.462598061982958, -0.09755474191953761, 0.9703503187569243),
                                   (-1.4080417204047313, -0.8976377310686736, 1.1927020968566089),
                                   (-0.27981087345916755, -0.8670643393461046, 0.29587765657632165),
                                   (1.1395623815572733, -0.9147118621123697, 0.771368745020215),
                                   (3.7901243915692864, -0.006544237180536178, 0.7580206603561134),
                                   (3.6186251824572455, 1.0920401631166292, -0.725695658374561),
                                   (-0.4799044636709365, -0.8577283498506146, -1.8345168113636874),
                                   (-1.5704890060131314, 1.527002009812866, -1.902575985299536),
                                   (-3.3260277144990296, 1.5238536460491903, -0.20338465526703625),
                                   (-3.311126364299293, -0.09969554359088921, 1.6478137927333953),
                                   (-1.3707042898204835, -1.549541647625315, 2.0589774409040964),
                                   (1.5338362221707007, -1.9310023570889727, 0.6663504223502944),
                                   (1.2246749300961473, -0.5970975942012858, 1.816181327157103)),
                        'isotopes': (12, 12, 12, 1, 12, 12, 12, 12, 12, 12, 12, 1, 1, 1, 1, 1, 1, 1, 1, 1),
                        'symbols': ('C', 'C', 'C', 'H', 'C', 'C', 'C', 'C', 'C', 'C', 'C',
                                    'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H')}
        c10h10_b_xyz = {'coords': ((3.247237794328524, -0.13719671162966918, 0.19555833918937052),
                                   (1.9094861712282774, -0.08067655688828143, 0.14898941432495702),
                                   (0.9973729357914858, -1.2703386896415134, -0.09322848415119056),
                                   (-0.37904449715218924, -0.6747166782032148, -0.049044448345326556),
                                   (-0.32906812544096026, 0.704634441388649, 0.189424753183012),
                                   (-1.4900181263846768, 1.4572613706024167, 0.2695747550348709),
                                   (-2.715200996994148, 0.8069241052920498, 0.10660938013945513),
                                   (-2.765284083663716, -0.5753713833636181, -0.13236922431004927),
                                   (-1.5909002849280705, -1.3270914347507115, -0.21179882275795825),
                                   (1.0862366144301145, 1.1823049698313937, 0.33079658088902575),
                                   (3.8424769924852367, 0.7530758608805569, 0.37314678191170336),
                                   (3.7762437608797406, -1.0749685445597326, 0.05710603017340202),
                                   (1.1128196175313243, -2.0170485762246773, 0.6986324476157837),
                                   (1.187449599052061, -1.7129398667445945, -1.0760419644685346),
                                   (-1.453108430051206, 2.525963604437891, 0.45426129138400156),
                                   (-3.639988653002051, 1.3756767310587803, 0.16518163487425436),
                                   (-3.7283956370857467, -1.0643593255501977, -0.2566648708585298),
                                   (-1.631427244782937, -2.3956407728893367, -0.3966116183664473),
                                   (1.3188711462571718, 1.9143096670969255, -0.4489453399950017),
                                   (1.2442414475018486, 1.6101977898569013, 1.3257284397785851)),
                        'isotopes': (12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
                        'symbols': ('C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C',
                                    'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H')}
        r_1 = ARCSpecies(label='C10H10_a', smiles='C=C1[CH]C2C=CC=C[C]2C1', xyz=c10h10_a_xyz, multiplicity=3)
        p_1 = ARCSpecies(label='C10H10_b', smiles='C=C1CC2=C(C=CC=C2)C1', xyz=c10h10_b_xyz)
        rxn = ARCReaction(reactants=['C10H10_a'], products=['C10H10_b'],
                          r_species=[r_1], p_species=[p_1])
        self.assertEqual(rxn.atom_map, [0, 1, 8, 13, 3, 2, 7, 6, 5, 4, 9, 10, 17, 12, 11, 16, 15, 14, 19, 18])
        self.assertTrue(check_atom_map(rxn))

        # Intra_R_Add_Endocyclic: C9H15_a <=> C9H15_b
        c9h15_a_xyz = {'coords': ((3.2994642637411093, -0.9763218631003405, -0.6681519125224107),
                                  (2.092867397835492, -0.585345209944081, -1.094234941414971),
                                  (1.1613654936979811, 0.23706312530825574, -0.350374400298155),
                                  (1.4994958941811034, 0.8089206946686178, 0.9907635052181555),
                                  (-0.2167854131709981, 0.47662057541684727, -0.9114032766325476),
                                  (-1.2860618884154418, -0.32193095884739475, -0.1923953058559322),
                                  (-1.2894939032313453, -1.8000285883092857, -0.4999462913608906),
                                  (-2.2130886752718024, 0.1935684936507141, 0.6410238159619941),
                                  (-2.383413365750594, 1.6157044776486373, 1.0712910920067213),
                                  (3.696965023185511, -0.7025114561770845, 0.3028336703297904),
                                  (3.9271105560154953, -1.5992344159626835, -1.2980759074189403),
                                  (1.7682570069194234, -0.916244398055435, -2.0798789408635727),
                                  (2.487309562171708, 1.280610628494466, 0.9837303428781683),
                                  (1.4864744914143402, 0.025765724669991667, 1.7553223060895524),
                                  (0.7820729499500115, 1.5805317186266579, 1.2867175051786177),
                                  (-0.4230089341260823, 1.5513883408081797, -0.8834461827090913),
                                  (-0.2525088519499385, 0.22261243999961292, -1.9790204993055305),
                                  (-1.3684387790718693, -1.963416003052446, -1.5797964159431177),
                                  (-2.1302956103647683, -2.3198259338415648, -0.028168861405248807),
                                  (-0.3695265066561803, -2.2717068331186607, -0.14091188769329688),
                                  (-2.9423489352590817, -0.48429745146049047, 1.0846035398328122),
                                  (-1.6122780147641311, 2.2876041556921556, 0.691039744143378),
                                  (-3.355397325714956, 1.9889012668068031, 0.7341417908661508),
                                  (-2.358736435364993, 1.6715714700786672, 2.1643375109183345)),
                       'isotopes': (12, 12, 12, 12, 12, 12, 12, 12, 12, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
                       'symbols': ('C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H',
                                   'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H')}
        c9h15_b_xyz = {'coords': ((-2.5844283571790947, -1.0321735590817163, 0.6979015062994665),
                                  (-1.7475791961733325, -0.06626031223098465, 1.0969062210689708),
                                  (-0.6595890997866395, 0.44535980204779074, 0.19328352629720955),
                                  (-1.2282980470691727, 1.3553131311426074, -0.9035809147486576),
                                  (0.5116680348286848, 1.1207407424824027, 0.9673543663680071),
                                  (1.4515588336620284, 0.3272984970368279, 0.15908291396837426),
                                  (2.790359263734144, -0.07794911282678396, 0.6402051576162312),
                                  (0.39868762752392606, -0.6378444769932384, -0.22420607008722174),
                                  (0.4638588895257082, -1.0921494216349954, -1.67271225417489),
                                  (-3.3702037268621954, -1.3885510696474426, 1.3571613514582264),
                                  (-2.514756820219243, -1.4880585713706647, -0.2847299130808078),
                                  (-1.8525509876151718, 0.37468306283449787, 2.084629086686036),
                                  (-1.799839887685193, 2.183429924439949, -0.4685383552676342),
                                  (-0.4353196261713602, 1.7968012774608193, -1.5175814720438623),
                                  (-1.8935148835851774, 0.8013551257352, -1.5751801294655083),
                                  (0.5782060423798034, 0.8980837541669382, 2.0389365697505366),
                                  (0.6148811498579635, 2.2012663914014428, 0.8270631269396428),
                                  (3.3879992933324807, 0.8037689675842231, 0.889705284982396),
                                  (2.714060940989492, -0.70739197884347, 1.5320182978132968),
                                  (3.3202001915969395, -0.6394149338748517, -0.13488149851161066),
                                  (0.41364799755952236, -1.5167424973440258, 0.43811504056239386),
                                  (0.5615949640204292, -0.25426187677410833, -2.3707422632792787),
                                  (-0.44004933907211424, -1.649313877420301, -1.9393981990380054),
                                  (1.3194067424075275, -1.7579889882901385, -1.8308113801134083)),
                       'isotopes': (12, 12, 12, 12, 12, 12, 12, 12, 12, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
                       'symbols': ('C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H',
                                   'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H')}
        r_1 = ARCSpecies(label='C9H15_a', smiles='C=C[C](C)CC(C)=CC', xyz=c9h15_a_xyz)
        p_1 = ARCSpecies(label='C9H15_b', smiles='C=CC1(C)C[C](C)C1C', xyz=c9h15_b_xyz)
        rxn = ARCReaction(reactants=['C9H15_a'], products=['C9H15_b'],
                          r_species=[r_1], p_species=[p_1])
        self.assertEqual(rxn.atom_map, [0, 4, 2, 3, 5, 7, 8, 1, 6, 11, 9, 10, 16, 22,
                                        20, 21, 14, 23, 12, 15, 19, 13, 17, 18])
        self.assertTrue(check_atom_map(rxn))

        # R_Addition_COm: C6H5 + CO <=> C7H5O
        c6h5_xyz = {'coords': ((0.1817676212163122, -1.6072341699404684, -0.014610043584505971),
                               (1.3027386938520413, -0.7802649159703986, -0.0076490984025043415),
                               (1.1475642728457944, 0.6058877336062989, 0.0049505291900821605),
                               (-0.13153090866637432, 1.1630287566213553, 0.010572132054881396),
                               (-1.2538945539777469, 0.33431084106618875, 0.0035960198829159064),
                               (-1.0955796250821246, -1.0514866492107922, -0.009001872474708414),
                               (2.2945290976411314, -1.222259069017827, -0.0120983109779029),
                               (2.0221784174097133, 1.2509576921755168, 0.010380274196135802),
                               (-0.25367929671488426, 2.243094989267151, 0.020390170037011494),
                               (-2.250483228275848, 0.767769623706613, 0.007970374795096042),
                               (-1.9636104902480103, -1.7038048323036503, -0.014500174716503693)),
                    'isotopes': (12, 12, 12, 12, 12, 12, 1, 1, 1, 1, 1),
                    'symbols': ('C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H')}
        c7h5o_xyz = {'coords': ((3.6046073677554133, -0.5344883784336433, -0.4686416595112313),
                                (2.450376918247941, -0.36333993603429526, -0.3185783842705891),
                                (1.0452654391924683, -0.1549911825981301, -0.13589722616572933),
                                (0.17928460434727675, -1.2499432760163112, -0.05142261213479038),
                                (-1.1903765403115871, -1.0453466445053263, 0.1266843926255701),
                                (-1.6956393825954146, 0.2514281313346147, 0.22045344560169716),
                                (-0.83456152850501, 1.3456029168509482, 0.13658183163326776),
                                (0.5355376113902255, 1.1439499192771787, -0.04151298496012599),
                                (0.5636535723209614, -2.2654604192056556, -0.12342236998627074),
                                (-1.863668502126327, -1.8967496276402949, 0.19236058875963039),
                                (-2.762672022813616, 0.4096470069241553, 0.3591803505729345),
                                (-1.2308445290855283, 2.355601612430931, 0.2099632744051463),
                                (1.1990369921831652, 2.0040898776158196, -0.1057486465694942)),
                     'isotopes': (16, 12, 12, 12, 12, 12, 12, 12, 1, 1, 1, 1, 1),
                     'symbols': ('O', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H')}
        r_1 = ARCSpecies(label='C6H5', smiles='[c]1ccccc1', xyz=c6h5_xyz)
        r_2 = ARCSpecies(label='CO', smiles='[C-]#[O+]', xyz=co_xyz)
        p_1 = ARCSpecies(label='C7H5O', smiles='O=[C]c1ccccc1', xyz=c7h5o_xyz)
        rxn = ARCReaction(reactants=['C6H5', 'CO'], products=['C7H5O'],
                          r_species=[r_1, r_2], p_species=[p_1])
        self.assertEqual(rxn.atom_map, [1, 5, 6, 7, 3, 4, 12, 11, 10, 9, 8, 2, 0])
        self.assertTrue(check_atom_map(rxn))

        # intra_NO2_ONO_conversion: C2H5NO2 <=> C2H5ONO
        c6h5_xyz = {'coords': ((1.8953828083622057, 0.8695975650550358, 0.6461465212661076),
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
        c7h5o_xyz = {'coords': ((-1.3334725178745668, 0.2849178019354427, 0.4149005134933577),
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
        r_1 = ARCSpecies(label='C2H5NO2', smiles='[O-][N+](=O)CC', xyz=c6h5_xyz)
        p_1 = ARCSpecies(label='C2H5ONO', smiles='CCON=O', xyz=c7h5o_xyz)
        rxn = ARCReaction(reactants=['C2H5NO2'], products=['C2H5ONO'],
                          r_species=[r_1], p_species=[p_1])
        self.assertEqual(rxn.atom_map, [4, 3, 2, 1, 0, 8, 9, 6, 5, 7])
        self.assertTrue(check_atom_map(rxn))

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
        h2_xyz = {'coords': ((0, 0, 0.3736550),
                             (0, 0, -0.3736550)),
                  'isotopes': (1, 1),
                  'symbols': ('H', 'H')}
        r_1 = ARCSpecies(label='H', smiles='[H]', xyz={'coords': ((0, 0, 0),), 'isotopes': (1,), 'symbols': ('H',)})
        r_2 = ARCSpecies(label='CH3NH2', smiles='CN', xyz=ch3nh2_xyz)
        p_1 = ARCSpecies(label='H2', smiles='[H][H]', xyz=h2_xyz)
        p_2 = ARCSpecies(label='CH2NH2', smiles='[CH2]N', xyz=ch2nh2_xyz)
        rxn_1 = ARCReaction(reactants=['H', 'CH3NH2'], products=['H2', 'CH2NH2'],
                            r_species=[r_1, r_2], p_species=[p_1, p_2])
        reactants_xyz_str = rxn_1.get_reactants_xyz()
        reactants_xyz_dict = rxn_1.get_reactants_xyz(return_format='dict')
        expected_reactants_xyz_str = """H       0.00000000    0.00000000    0.00000000
C      -0.57341115    0.02035161    0.03088704
N       0.81055959    0.00017446   -0.40777288
H      -1.12345497   -0.81238990   -0.41607711
H      -0.63322201   -0.06381792    1.11969836
H      -1.05320091    0.95395019   -0.27567270
H       1.31864224    0.76239063    0.03897612
H       1.25408721   -0.86065907   -0.09003883"""
        expected_reactants_xyz_dict = {'symbols': ('H', 'C', 'N', 'H', 'H', 'H', 'H', 'H'),
                                       'isotopes': (1, 12, 14, 1, 1, 1, 1, 1),
                                       'coords': ((0, 0, 0),
                                                  (-0.5734111454228507, 0.0203516083213337, 0.03088703933770556),
                                                  (0.8105595891860601, 0.00017446498908627427, -0.4077728757313545),
                                                  (-1.1234549667791063, -0.8123899006368857, -0.41607711106038836),
                                                  (-0.6332220120842996, -0.06381791823047896, 1.1196983583774054),
                                                  (-1.053200912106195, 0.9539501896695028, -0.27567270246542575),
                                                  (1.3186422395164141, 0.7623906284020254, 0.038976118645639976),
                                                  (1.2540872076899663, -0.8606590725145833, -0.09003882710357966))}
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
                   'symbols': ('O', 'O', 'H')}  # 3, 4, 9
        r_1 = ARCSpecies(label='C2H5O3', smiles='CC(O)O[O]', xyz=c2h5o3_xyz)
        p_1 = ARCSpecies(label='C2H4O', smiles='CC=O', xyz=c2h4o_xyz)
        p_2 = ARCSpecies(label='HO2', smiles='O[O]', xyz=ho2_xyz)
        rxn = ARCReaction(r_species=[r_1], p_species=[p_1, p_2])
        self.assertEqual(rxn.atom_map, [0, 1, 2, 8, 7, 3, 4, 5, 6, 9])  # [3, 4, 5] are identical in 2D
        self.assertTrue(check_atom_map(rxn))
        rxn = ARCReaction(r_species=[p_1, p_2], p_species=[r_1])
        self.assertEqual(rxn.atom_map, [0, 1, 2, 8, 7, 3, 4, 5, 6, 9])  # [3, 4, 5] are identical in 2D
        self.assertTrue(check_atom_map(rxn))

        # H_Abstraction: C3H6O + C4H9O <=> C3H5O + C4H10O
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
        rxn = ARCReaction(reactants=['C3H6O', 'C4H9O'], products=['C3H5O', 'C4H10O'])
        rxn.r_species = [r_1, r_2]
        rxn.p_species = [p_1, p_2]
        self.assertEqual(rxn.atom_map, [0, 1, 3, 5, 6, 7, 2, 5+9, 8, 0+9, 6+9, 7+9, 1+9, 2+9, 3+9, 4+9, 8+9, 9+9,
                                        10+9, 11+9, 12+9, 13+9])
        self.assertTrue(check_atom_map(rxn))

        # H_Abstraction: NH + N2H3 <=> NH2 + N2H2(T)
        nh_xyz = {'coords': ((0.509499983131626, 0.0, 0.0), (-0.509499983131626, 0.0, 0.0)),
                  'isotopes': (14, 1),
                  'symbols': ('N', 'H')}
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
        n2h3_t_xyz = {'coords': ((0.5974274138372041, -0.41113104979405946, 0.08609839663782763),
                                 (1.421955422639823, 0.19737093442024492, 0.02508578507394823),
                                 (-0.5974274348582206, 0.41113108883884353, -0.08609846602622732),
                                 (-1.4219554016188147, -0.19737097346502322, -0.02508571568554942)),
                      'isotopes': (14, 1, 14, 1),
                      'symbols': ('N', 'H', 'N', 'H')}
        r_1 = ARCSpecies(label='NH', smiles='[NH]', xyz=nh_xyz)
        r_2 = ARCSpecies(label='N2H3', smiles='N[NH]', xyz=n2h3_xyz)
        p_1 = ARCSpecies(label='NH2', smiles='[NH2]', xyz=nh2_xyz)
        p_2 = ARCSpecies(label='N2H2(T)', smiles='[NH][NH]', xyz=n2h3_t_xyz)
        rxn = ARCReaction(reactants=['NH', 'N2H3'], products=['NH2', 'N2H2(T)'],
                          r_species=[r_1, r_2], p_species=[p_1, p_2])
        self.assertEqual(rxn.atom_map, [0, 1, 3, 5, 4, 6, 2])
        self.assertTrue(check_atom_map(rxn))

        # Intra_Disproportionation: C10H10_a <=> C10H10_b
        c10h10_a_xyz = {'coords': ((3.1623638230700997, 0.39331289450005563, -0.031839117414963584),
                                   (1.8784852381397288, 0.037685951926618944, -0.13659028131444134),
                                   (0.9737380560194014, 0.5278617594060281, -1.1526858375270472),
                                   (1.2607098516126556, 1.1809007875206383, -1.9621017164412065),
                                   (-0.36396095305912823, -0.13214785064139675, -1.0200667625809143),
                                   (-1.5172464644867296, 0.8364138939810618, -1.0669384323486588),
                                   (-2.4922101649968655, 0.8316551483126366, -0.14124720277902958),
                                   (-2.462598061982958, -0.09755474191953761, 0.9703503187569243),
                                   (-1.4080417204047313, -0.8976377310686736, 1.1927020968566089),
                                   (-0.27981087345916755, -0.8670643393461046, 0.29587765657632165),
                                   (1.1395623815572733, -0.9147118621123697, 0.771368745020215),
                                   (3.7901243915692864, -0.006544237180536178, 0.7580206603561134),
                                   (3.6186251824572455, 1.0920401631166292, -0.725695658374561),
                                   (-0.4799044636709365, -0.8577283498506146, -1.8345168113636874),
                                   (-1.5704890060131314, 1.527002009812866, -1.902575985299536),
                                   (-3.3260277144990296, 1.5238536460491903, -0.20338465526703625),
                                   (-3.311126364299293, -0.09969554359088921, 1.6478137927333953),
                                   (-1.3707042898204835, -1.549541647625315, 2.0589774409040964),
                                   (1.5338362221707007, -1.9310023570889727, 0.6663504223502944),
                                   (1.2246749300961473, -0.5970975942012858, 1.816181327157103)),
                        'isotopes': (12, 12, 12, 1, 12, 12, 12, 12, 12, 12, 12, 1, 1, 1, 1, 1, 1, 1, 1, 1),
                        'symbols': ('C', 'C', 'C', 'H', 'C', 'C', 'C', 'C', 'C', 'C', 'C',
                                    'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H')}
        c10h10_b_xyz = {'coords': ((3.247237794328524, -0.13719671162966918, 0.19555833918937052),
                                   (1.9094861712282774, -0.08067655688828143, 0.14898941432495702),
                                   (0.9973729357914858, -1.2703386896415134, -0.09322848415119056),
                                   (-0.37904449715218924, -0.6747166782032148, -0.049044448345326556),
                                   (-0.32906812544096026, 0.704634441388649, 0.189424753183012),
                                   (-1.4900181263846768, 1.4572613706024167, 0.2695747550348709),
                                   (-2.715200996994148, 0.8069241052920498, 0.10660938013945513),
                                   (-2.765284083663716, -0.5753713833636181, -0.13236922431004927),
                                   (-1.5909002849280705, -1.3270914347507115, -0.21179882275795825),
                                   (1.0862366144301145, 1.1823049698313937, 0.33079658088902575),
                                   (3.8424769924852367, 0.7530758608805569, 0.37314678191170336),
                                   (3.7762437608797406, -1.0749685445597326, 0.05710603017340202),
                                   (1.1128196175313243, -2.0170485762246773, 0.6986324476157837),
                                   (1.187449599052061, -1.7129398667445945, -1.0760419644685346),
                                   (-1.453108430051206, 2.525963604437891, 0.45426129138400156),
                                   (-3.639988653002051, 1.3756767310587803, 0.16518163487425436),
                                   (-3.7283956370857467, -1.0643593255501977, -0.2566648708585298),
                                   (-1.631427244782937, -2.3956407728893367, -0.3966116183664473),
                                   (1.3188711462571718, 1.9143096670969255, -0.4489453399950017),
                                   (1.2442414475018486, 1.6101977898569013, 1.3257284397785851)),
                        'isotopes': (12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
                        'symbols': ('C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C',
                                    'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H')}
        r_1 = ARCSpecies(label='C10H10_a', smiles='C=C1[CH]C2C=CC=C[C]2C1', xyz=c10h10_a_xyz, multiplicity=3)
        p_1 = ARCSpecies(label='C10H10_b', smiles='C=C1CC2=C(C=CC=C2)C1', xyz=c10h10_b_xyz)
        rxn = ARCReaction(r_species=[r_1], p_species=[p_1])
        self.assertEqual(rxn.atom_map, [0, 1, 8, 13, 3, 2, 7, 6, 5, 4, 9, 10, 17, 12, 11, 16, 15, 14, 19, 18])
        self.assertTrue(check_atom_map(rxn))

        # Intra_R_Add_Endocyclic: C9H15_a <=> C9H15_b
        c9h15_a_xyz = {'coords': ((3.2994642637411093, -0.9763218631003405, -0.6681519125224107),
                                  (2.092867397835492, -0.585345209944081, -1.094234941414971),
                                  (1.1613654936979811, 0.23706312530825574, -0.350374400298155),
                                  (1.4994958941811034, 0.8089206946686178, 0.9907635052181555),
                                  (-0.2167854131709981, 0.47662057541684727, -0.9114032766325476),
                                  (-1.2860618884154418, -0.32193095884739475, -0.1923953058559322),
                                  (-1.2894939032313453, -1.8000285883092857, -0.4999462913608906),
                                  (-2.2130886752718024, 0.1935684936507141, 0.6410238159619941),
                                  (-2.383413365750594, 1.6157044776486373, 1.0712910920067213),
                                  (3.696965023185511, -0.7025114561770845, 0.3028336703297904),
                                  (3.9271105560154953, -1.5992344159626835, -1.2980759074189403),
                                  (1.7682570069194234, -0.916244398055435, -2.0798789408635727),
                                  (2.487309562171708, 1.280610628494466, 0.9837303428781683),
                                  (1.4864744914143402, 0.025765724669991667, 1.7553223060895524),
                                  (0.7820729499500115, 1.5805317186266579, 1.2867175051786177),
                                  (-0.4230089341260823, 1.5513883408081797, -0.8834461827090913),
                                  (-0.2525088519499385, 0.22261243999961292, -1.9790204993055305),
                                  (-1.3684387790718693, -1.963416003052446, -1.5797964159431177),
                                  (-2.1302956103647683, -2.3198259338415648, -0.028168861405248807),
                                  (-0.3695265066561803, -2.2717068331186607, -0.14091188769329688),
                                  (-2.9423489352590817, -0.48429745146049047, 1.0846035398328122),
                                  (-1.6122780147641311, 2.2876041556921556, 0.691039744143378),
                                  (-3.355397325714956, 1.9889012668068031, 0.7341417908661508),
                                  (-2.358736435364993, 1.6715714700786672, 2.1643375109183345)),
                       'isotopes': (12, 12, 12, 12, 12, 12, 12, 12, 12, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
                       'symbols': ('C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H',
                                   'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H')}
        c9h15_b_xyz = {'coords': ((-2.5844283571790947, -1.0321735590817163, 0.6979015062994665),
                                  (-1.7475791961733325, -0.06626031223098465, 1.0969062210689708),
                                  (-0.6595890997866395, 0.44535980204779074, 0.19328352629720955),
                                  (-1.2282980470691727, 1.3553131311426074, -0.9035809147486576),
                                  (0.5116680348286848, 1.1207407424824027, 0.9673543663680071),
                                  (1.4515588336620284, 0.3272984970368279, 0.15908291396837426),
                                  (2.790359263734144, -0.07794911282678396, 0.6402051576162312),
                                  (0.39868762752392606, -0.6378444769932384, -0.22420607008722174),
                                  (0.4638588895257082, -1.0921494216349954, -1.67271225417489),
                                  (-3.3702037268621954, -1.3885510696474426, 1.3571613514582264),
                                  (-2.514756820219243, -1.4880585713706647, -0.2847299130808078),
                                  (-1.8525509876151718, 0.37468306283449787, 2.084629086686036),
                                  (-1.799839887685193, 2.183429924439949, -0.4685383552676342),
                                  (-0.4353196261713602, 1.7968012774608193, -1.5175814720438623),
                                  (-1.8935148835851774, 0.8013551257352, -1.5751801294655083),
                                  (0.5782060423798034, 0.8980837541669382, 2.0389365697505366),
                                  (0.6148811498579635, 2.2012663914014428, 0.8270631269396428),
                                  (3.3879992933324807, 0.8037689675842231, 0.889705284982396),
                                  (2.714060940989492, -0.70739197884347, 1.5320182978132968),
                                  (3.3202001915969395, -0.6394149338748517, -0.13488149851161066),
                                  (0.41364799755952236, -1.5167424973440258, 0.43811504056239386),
                                  (0.5615949640204292, -0.25426187677410833, -2.3707422632792787),
                                  (-0.44004933907211424, -1.649313877420301, -1.9393981990380054),
                                  (1.3194067424075275, -1.7579889882901385, -1.8308113801134083)),
                       'isotopes': (12, 12, 12, 12, 12, 12, 12, 12, 12, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
                       'symbols': ('C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H',
                                   'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H')}
        r_1 = ARCSpecies(label='C9H15_a', smiles='C=C[C](C)CC(C)=CC', xyz=c9h15_a_xyz)
        p_1 = ARCSpecies(label='C9H15_b', smiles='C=CC1(C)C[C](C)C1C', xyz=c9h15_b_xyz)
        rxn = ARCReaction(r_species=[r_1], p_species=[p_1])
        self.assertEqual(rxn.atom_map, [0, 4, 2, 3, 5, 7, 8, 1, 6, 11, 9, 10, 16, 22,
                                        20, 21, 14, 23, 12, 15, 19, 13, 17, 18])
        self.assertTrue(check_atom_map(rxn))

        # R_Addition_COm: C6H5 + CO <=> C7H5O
        c6h5_xyz = {'coords': ((0.1817676212163122, -1.6072341699404684, -0.014610043584505971),
                               (1.3027386938520413, -0.7802649159703986, -0.0076490984025043415),
                               (1.1475642728457944, 0.6058877336062989, 0.0049505291900821605),
                               (-0.13153090866637432, 1.1630287566213553, 0.010572132054881396),
                               (-1.2538945539777469, 0.33431084106618875, 0.0035960198829159064),
                               (-1.0955796250821246, -1.0514866492107922, -0.009001872474708414),
                               (2.2945290976411314, -1.222259069017827, -0.0120983109779029),
                               (2.0221784174097133, 1.2509576921755168, 0.010380274196135802),
                               (-0.25367929671488426, 2.243094989267151, 0.020390170037011494),
                               (-2.250483228275848, 0.767769623706613, 0.007970374795096042),
                               (-1.9636104902480103, -1.7038048323036503, -0.014500174716503693)),
                    'isotopes': (12, 12, 12, 12, 12, 12, 1, 1, 1, 1, 1),
                    'symbols': ('C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H')}
        c7h5o_xyz = {'coords': ((3.6046073677554133, -0.5344883784336433, -0.4686416595112313),
                                (2.450376918247941, -0.36333993603429526, -0.3185783842705891),
                                (1.0452654391924683, -0.1549911825981301, -0.13589722616572933),
                                (0.17928460434727675, -1.2499432760163112, -0.05142261213479038),
                                (-1.1903765403115871, -1.0453466445053263, 0.1266843926255701),
                                (-1.6956393825954146, 0.2514281313346147, 0.22045344560169716),
                                (-0.83456152850501, 1.3456029168509482, 0.13658183163326776),
                                (0.5355376113902255, 1.1439499192771787, -0.04151298496012599),
                                (0.5636535723209614, -2.2654604192056556, -0.12342236998627074),
                                (-1.863668502126327, -1.8967496276402949, 0.19236058875963039),
                                (-2.762672022813616, 0.4096470069241553, 0.3591803505729345),
                                (-1.2308445290855283, 2.355601612430931, 0.2099632744051463),
                                (1.1990369921831652, 2.0040898776158196, -0.1057486465694942)),
                     'isotopes': (16, 12, 12, 12, 12, 12, 12, 12, 1, 1, 1, 1, 1),
                     'symbols': ('O', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H')}
        co_xyz = {'coords': ((0.0, 0.0, -0.6552100),
                             (0.0, 0.0, 0.4914080)),
                  'isotopes': (12, 16),
                  'symbols': ('C', 'O')}
        r_1 = ARCSpecies(label='C6H5', smiles='[c]1ccccc1', xyz=c6h5_xyz)
        r_2 = ARCSpecies(label='CO', smiles='[C-]#[O+]', xyz=co_xyz)
        p_1 = ARCSpecies(label='C7H5O', smiles='O=[C]c1ccccc1', xyz=c7h5o_xyz)
        rxn = ARCReaction(r_species=[r_1, r_2], p_species=[p_1])
        self.assertEqual(rxn.atom_map, [1, 5, 6, 7, 3, 4, 12, 11, 10, 9, 8, 2, 0])
        self.assertTrue(check_atom_map(rxn))

        # intra_NO2_ONO_conversion: C2H5NO2 <=> C2H5ONO
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
        r_1 = ARCSpecies(label='C2H5NO2', smiles='[O-][N+](=O)CC', xyz=c2h5no2_xyz)
        p_1 = ARCSpecies(label='C2H5ONO', smiles='CCON=O', xyz=c2h5ono_xyz)
        rxn = ARCReaction(r_species=[r_1], p_species=[p_1])
        self.assertEqual(rxn.atom_map, [4, 3, 2, 1, 0, 8, 9, 6, 5, 7])
        self.assertTrue(check_atom_map(rxn))

    def test_get_mapped_product_xyz(self):
        """Test the Reaction get_mapped_product_xyz method"""
        # trivial unimolecular with an intentional mixed atom order: H2O <=> H2O
        h2o_xyz_1 = {'symbols': ('O', 'H', 'H'),
                     'isotopes': (16, 1, 1),
                     'coords': ((-0.19827, 0.0, 0.76363),
                                (0.39781, 0.0, -0.00032),
                                (-0.19953, 0.0, -0.76330))}
        r_1 = ARCSpecies(label='H2O', smiles='O', xyz=h2o_xyz_1)

        h2o_xyz_2 = {'symbols': ('H', 'H', 'O'),
                     'isotopes': (1, 1, 16),
                     'coords': ((0.39781, 0.0, -0.00032),
                                (-0.19953, 0.0, -0.76330),
                                (-0.19827, 0.0, 0.76363))}
        p_1 = ARCSpecies(label='H2O', smiles='O', xyz=h2o_xyz_2)

        rxn_1 = ARCReaction(reactants=['H2O'], products=['H2O'],
                            r_species=[r_1], p_species=[p_1])
        _, mapped_product = rxn_1.get_mapped_product_xyz()
        self.assertEqual(rxn_1.atom_map, [2, 0, 1])
        self.assertTrue(check_atom_map(rxn_1))
        self.assertTrue(mapped_product.get_xyz(), h2o_xyz_1)

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
        mapped_product = rxn_2.get_mapped_product_xyz()[1]
        self.assertEqual(rxn_2.atom_map, [2, 0, 1])
        self.assertTrue(check_atom_map(rxn_2))
        self.assertTrue(mapped_product.get_xyz(), h2o_xyz_1)

    def test_check_attributes(self):
        """Test checking the reaction attributes"""
        rxn_1 = ARCReaction(label='H + [O-][N+](=N)N=O <=> [N-]=[N+](N=O)[O] + H2',
                            r_species=[ARCSpecies(label='H', smiles='[H]'),
                                       ARCSpecies(label='[O-][N+](=N)N=O', smiles='[O-][N+](=N)N=O')],
                            p_species=[ARCSpecies(label='H2', smiles='[H][H]'),
                                       ARCSpecies(label='[N-]=[N+](N=O)[O]', smiles='[N-]=[N+](N=O)[O]')],
                            )
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
                        ARCSpecies(label='H2O', smiles='O'),
                        ]
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


def check_atom_map(rxn: ARCReaction) -> bool:
    """
    A helper function for testing a reaction atom map.
    Tests that element symbols are ordered correctly.
    Note: This is a necessary but not a sufficient condition.

    Args:
        rxn (ARCReaction): The reaction to examine.

    Returns: bool
        Whether the atom mapping makes sense.
    """
    r_elements, p_elements = list(), list()
    for r_species in rxn.r_species:
        r_elements.extend(list(r_species.get_xyz()['symbols']))
    for p_species in rxn.p_species:
        p_elements.extend(list(p_species.get_xyz()['symbols']))
    for i, map_i in enumerate(rxn.atom_map):
        if r_elements[i] != p_elements[map_i]:
            break
    else:
        # Did not break, the mapping makes sense.
        return True
    return False


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
