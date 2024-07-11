#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.reaction.family module
"""

import os
import unittest

from rmgpy.molecule import Group, Molecule

from arc.common import generate_resonance_structures
from arc.reaction.family import (ReactionFamily,
                                 ARC_FAMILIES_PATH,
                                 RMG_DB_PATH,
                                 add_labels_to_molecule,
                                 descent_complex_group,
                                 determine_possible_reaction_products_from_family,
                                 get_reaction_family_products,
                                 get_all_families,
                                 get_entries,
                                 get_group_adjlist,
                                 get_initial_reactant_labels_from_template,
                                 get_isomorphic_subgraph,
                                 get_product_num,
                                 get_reactant_groups_from_template,
                                 get_recipe_actions,
                                 get_rmg_recommended_family_sets,
                                 is_own_reverse,
                                 is_reversible,
                                 )
from arc.reaction.reaction import ARCReaction
from arc.species.species import ARCSpecies


class TestReactionFamily(unittest.TestCase):
    """
    Contains unit tests for the arc reaction family module
    """

    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None

    def test_rmgdb_path(self):
        """Test finding the RMG-database path"""
        self.assertIn('RMG-database', RMG_DB_PATH)
        self.assertTrue(os.path.isdir(RMG_DB_PATH))

    def test_arc_families_path(self):
        """Test finding the ARC families folder path"""
        self.assertIn('ARC', ARC_FAMILIES_PATH)
        self.assertIn('data', ARC_FAMILIES_PATH)
        self.assertIn('families', ARC_FAMILIES_PATH)
        self.assertTrue(os.path.isdir(ARC_FAMILIES_PATH))

    def test_get_reaction_family_products(self):
        """Test determining the reaction family using product dicts"""
        rxn_0a = ARCReaction(r_species=[ARCSpecies(label='CH4', smiles='C'), ARCSpecies(label='O2', smiles='[O][O]')],
                             p_species=[ARCSpecies(label='CH3', smiles='[CH3]'),
                                        ARCSpecies(label='HO2', smiles='O[O]')])
        products = get_reaction_family_products(rxn_0a)
        expected_products = [{'discovered_in_reverse': False,
                              'family': 'H_Abstraction',
                              'group_labels': ('X_H', 'Y_rad'),
                              'label_map': {'*1': 0, '*2': 1, '*3': 5},
                              'own_reverse': True,
                              'products': [Molecule(smiles="[O]O"), Molecule(smiles="[CH3]")]},
                             {'discovered_in_reverse': False,
                              'family': 'H_Abstraction',
                              'group_labels': ('X_H', 'Y_rad'),
                              'label_map': {'*1': 0, '*2': 1, '*3': 6},
                              'own_reverse': True,
                              'products': [Molecule(smiles="[O]O"), Molecule(smiles="[CH3]")]},
                             {'discovered_in_reverse': False,
                              'family': 'H_Abstraction',
                              'group_labels': ('X_H', 'Y_rad'),
                              'label_map': {'*1': 0, '*2': 2, '*3': 5},
                              'own_reverse': True,
                              'products': [Molecule(smiles="[O]O"), Molecule(smiles="[CH3]")]},
                             {'discovered_in_reverse': False,
                              'family': 'H_Abstraction',
                              'group_labels': ('X_H', 'Y_rad'),
                              'label_map': {'*1': 0, '*2': 2, '*3': 6},
                              'own_reverse': True,
                              'products': [Molecule(smiles="[O]O"), Molecule(smiles="[CH3]")]},
                             {'discovered_in_reverse': False,
                              'family': 'H_Abstraction',
                              'group_labels': ('X_H', 'Y_rad'),
                              'label_map': {'*1': 0, '*2': 3, '*3': 5},
                              'own_reverse': True,
                              'products': [Molecule(smiles="[O]O"), Molecule(smiles="[CH3]")]},
                             {'discovered_in_reverse': False,
                              'family': 'H_Abstraction',
                              'group_labels': ('X_H', 'Y_rad'),
                              'label_map': {'*1': 0, '*2': 3, '*3': 6},
                              'own_reverse': True,
                              'products': [Molecule(smiles="[O]O"), Molecule(smiles="[CH3]")]},
                             {'discovered_in_reverse': False,
                              'family': 'H_Abstraction',
                              'group_labels': ('X_H', 'Y_rad'),
                              'label_map': {'*1': 0, '*2': 4, '*3': 5},
                              'own_reverse': True,
                              'products': [Molecule(smiles="[O]O"), Molecule(smiles="[CH3]")]},
                             {'discovered_in_reverse': False,
                              'family': 'H_Abstraction',
                              'group_labels': ('X_H', 'Y_rad'),
                              'label_map': {'*1': 0, '*2': 4, '*3': 6},
                              'own_reverse': True,
                              'products': [Molecule(smiles="[O]O"), Molecule(smiles="[CH3]")]}]
        self.assertEqual(products, expected_products)

        ch4_xyz = """C      -0.00000000   -0.00000000    0.00000000
H      -0.87497771   -0.55943190   -0.33815595
H      -0.04050904    1.01567250   -0.39958464
H       0.00816153    0.03824434    1.09149909
H       0.90732523   -0.49448494   -0.35375850"""
        o2_xyz = {'symbols': ('O', 'O'), 'isotopes': (16, 16), 'coords': ((0.0, 0.0, 0.6029), (0.0, 0.0, -0.6029))}
        ch3_xyz = """C      -0.00000000   -0.00000000   -0.00000001
H       1.04110758   -0.29553525    0.02584268
H      -0.77665903   -0.75407986   -0.00884379
H      -0.26444854    1.04961512   -0.01699888"""
        ho2_xyz = """O      -0.15635718    0.45208323    0.00000000
O       0.99456866   -0.18605915    0.00000000
H      -0.83821148   -0.26602407    0.00000000"""

        rxn_0b = ARCReaction(r_species=[ARCSpecies(label='CH4', xyz=ch4_xyz), ARCSpecies(label='O2', xyz=o2_xyz, multiplicity=3)],
                             p_species=[ARCSpecies(label='CH3', xyz=ch3_xyz), ARCSpecies(label='HO2', xyz=ho2_xyz)])
        products = get_reaction_family_products(rxn_0b)
        expected_products = [{'discovered_in_reverse': True,
                              'family': 'Disproportionation',  # Todo: should be H_abs after merging Calvin's PR
                              'group_labels': 'Root',
                              'label_map': {'*1': 0, '*2': 4, '*3': 5, '*4': 6},
                              'own_reverse': False,
                              'products': [Molecule(smiles="C"), Molecule(smiles="O=O")]}]
        print(products)
        self.assertEqual(products, expected_products)


        rxn_1 = ARCReaction(reactants=['NH2', 'NH2'], products=['NH', 'NH3'],
                            r_species=[ARCSpecies(label='NH2', smiles='[NH2]')],
                            p_species=[ARCSpecies(label='NH', smiles='[NH]'), ARCSpecies(label='NH3', smiles='N')])
        products = get_reaction_family_products(rxn_1)
        expected_products = [{'discovered_in_reverse': False,
                              'family': 'H_Abstraction',
                              'group_labels': ('Xrad_H', 'Y_rad'),
                              'label_map': {'*1': 0, '*2': 1, '*3': 3},
                              'own_reverse': True,
                              'products': [Molecule(smiles="N"), Molecule(smiles="[NH]")]},
                             {'discovered_in_reverse': False,
                              'family': 'H_Abstraction',
                              'group_labels': ('Xrad_H', 'Y_rad'),
                              'label_map': {'*1': 0, '*2': 2, '*3': 3},
                              'own_reverse': True,
                              'products': [Molecule(smiles="N"), Molecule(smiles="[NH]")]},
                             {'discovered_in_reverse': False,
                              'family': 'H_Abstraction',
                              'group_labels': ('Y_rad', 'Xrad_H'),
                              'label_map': {'*1': 3, '*2': 4, '*3': 0},
                              'own_reverse': True,
                              'products': [Molecule(smiles="[NH]"), Molecule(smiles="N")]},
                             {'discovered_in_reverse': False,
                              'family': 'H_Abstraction',
                              'group_labels': ('Y_rad', 'Xrad_H'),
                              'label_map': {'*1': 3, '*2': 5, '*3': 0},
                              'own_reverse': True,
                              'products': [Molecule(smiles="N"), Molecule(smiles="[NH]")]}]
        self.assertEqual(products, expected_products)

        rxn_2f = ARCReaction(r_species=[ARCSpecies(label='C2H3O3', smiles='[CH2]C(=O)OO')],
                             p_species=[ARCSpecies(label='C2H2O2', smiles='O=C1CO1'),
                                        ARCSpecies(label='OH', smiles='[OH]')])
        products = get_reaction_family_products(rxn_2f)
        expected_products = [{'discovered_in_reverse': False,
                              'family': 'Cyclic_Ether_Formation',
                              'group_labels': 'R2OO',
                              'label_map': {'*1': 0, '*2': 3, '*3': 4, '*4': 1},
                              'own_reverse': False,
                              'products': [Molecule(smiles="[OH]"), Molecule(smiles="O=C1CO1")]},
                             {'discovered_in_reverse': False,
                              'family': 'Cyclic_Ether_Formation',
                              'group_labels': 'R2OO',
                              'label_map': {'*1': 0, '*2': 3, '*3': 4, '*4': 1},
                              'own_reverse': False,
                              'products': [Molecule(smiles="[OH]"), Molecule(smiles="O=C1CO1")]}]
        self.assertEqual(products, expected_products)

        rxn_2_r = ARCReaction(p_species=[ARCSpecies(label='C2H3O3', smiles='[CH2]C(=O)OO')],
                              r_species=[ARCSpecies(label='C2H2O2', smiles='O=C1CO1'),
                                         ARCSpecies(label='OH', smiles='[OH]')])
        products = get_reaction_family_products(rxn_2_r)
        expected_products = [{'discovered_in_reverse': True,
                              'family': 'Cyclic_Ether_Formation',
                              'group_labels': 'R2OO',
                              'label_map': {'*1': 0, '*2': 3, '*3': 4, '*4': 1},
                              'own_reverse': False,
                              'products': [Molecule(smiles="[OH]"), Molecule(smiles="O=C1CO1")]},
                             {'discovered_in_reverse': True,
                              'family': 'Cyclic_Ether_Formation',
                              'group_labels': 'R2OO',
                              'label_map': {'*1': 0, '*2': 3, '*3': 4, '*4': 1},
                              'own_reverse': False,
                              'products': [Molecule(smiles="[OH]"), Molecule(smiles="O=C1CO1")]}]
        self.assertEqual(products, expected_products)

        rxn_3 = ARCReaction(r_species=[ARCSpecies(label='CCC[O]', smiles='CCC[O]')],
                            p_species=[ARCSpecies(label='C[CH]CO', smiles='C[CH]CO')])
        products = get_reaction_family_products(rxn_3)
        expected_products = [{'discovered_in_reverse': False,
                              'family': 'intra_H_migration',
                              'group_labels': 'R3Hall',
                              'label_map': {'*1': 3, '*2': 1, '*3': 7, '*4': 2},
                              'own_reverse': True,
                              'products': [Molecule(smiles="C[CH]CO")]},
                             {'discovered_in_reverse': False,
                              'family': 'intra_H_migration',
                              'group_labels': 'R3Hall',
                              'label_map': {'*1': 3, '*2': 1, '*3': 8, '*4': 2},
                              'own_reverse': True,
                              'products': [Molecule(smiles="C[CH]CO")]}]
        self.assertEqual(products, expected_products)

        rxn_4f = ARCReaction(r_species=[ARCSpecies(label='HOCCOH', smiles='OC=CO')],
                             p_species=[ARCSpecies(label='HOCCO', smiles='OCC=O')])
        products = get_reaction_family_products(rxn_4f)
        expected_products = [{'discovered_in_reverse': False,
                              'family': 'Ketoenol',
                              'group_labels': 'Root',
                              'label_map': {'*1': 2, '*2': 1, '*3': 0, '*4': 4},
                              'own_reverse': False,
                              'products': [Molecule(smiles="O=CCO")]},
                             {'discovered_in_reverse': False,
                              'family': 'Ketoenol',
                              'group_labels': 'Root',
                              'label_map': {'*1': 1, '*2': 2, '*3': 3, '*4': 7},
                              'own_reverse': False,
                              'products': [Molecule(smiles="O=CCO")]}]
        self.assertEqual(products, expected_products)

        rxn_4r = ARCReaction(r_species=[ARCSpecies(label='HOCCO', smiles='OCC=O')],
                             p_species=[ARCSpecies(label='HOCCOH', smiles='OC=CO')])
        products = get_reaction_family_products(rxn_4r)
        expected_products = [{'discovered_in_reverse': True,
                              'family': 'Ketoenol',
                              'group_labels': 'Root',
                              'label_map': {'*1': 2, '*2': 1, '*3': 0, '*4': 4},
                              'own_reverse': False,
                              'products': [Molecule(smiles="O=CCO")]},
                             {'discovered_in_reverse': True,
                              'family': 'Ketoenol',
                              'group_labels': 'Root',
                              'label_map': {'*1': 1, '*2': 2, '*3': 3, '*4': 7},
                              'own_reverse': False,
                              'products': [Molecule(smiles="O=CCO")]}]
        self.assertEqual(products, expected_products)

        # Test a family that is bimolecular, but the two reactants are defined as a single group in the recipe:
        rxn_5f = ARCReaction(r_species=[ARCSpecies(label='CO2', smiles='O=C=O'),
                                        ARCSpecies(label='H2', smiles='[H][H]')],
                             p_species=[ARCSpecies(label='CHOOH', smiles='O=CO')])
        products = get_reaction_family_products(rxn_5f)
        expected_products = [{'discovered_in_reverse': False,
                              'family': '1,3_Insertion_CO2',
                              'group_labels': 'Root',
                              'label_map': {'*1': 1, '*2': 0, '*3': 4, '*4': 3},
                              'own_reverse': False,
                              'products': [Molecule(smiles="O=CO")]},
                             {'discovered_in_reverse': False,
                              'family': '1,3_Insertion_CO2',
                              'group_labels': 'Root',
                              'label_map': {'*1': 1, '*2': 0, '*3': 3, '*4': 4},
                              'own_reverse': False,
                              'products': [Molecule(smiles="O=CO")]},
                             {'discovered_in_reverse': False,
                              'family': '1,3_Insertion_CO2',
                              'group_labels': 'Root',
                              'label_map': {'*1': 1, '*2': 2, '*3': 4, '*4': 3},
                              'own_reverse': False,
                              'products': [Molecule(smiles="O=CO")]},
                             {'discovered_in_reverse': False,
                              'family': '1,3_Insertion_CO2',
                              'group_labels': 'Root',
                              'label_map': {'*1': 1, '*2': 2, '*3': 3, '*4': 4},
                              'own_reverse': False,
                              'products': [Molecule(smiles="O=CO")]}]
        self.assertEqual(products, expected_products)

        rxn_5r = ARCReaction(r_species=[ARCSpecies(label='CHOOH', smiles='O=CO')],
                             p_species=[ARCSpecies(label='CO2', smiles='O=C=O'),
                                        ARCSpecies(label='H2', smiles='[H][H]')])
        products = get_reaction_family_products(rxn_5r)
        expected_products = [{'discovered_in_reverse': True,
                              'family': '1,3_Insertion_CO2',
                              'group_labels': 'Root',
                              'label_map': {'*1': 1, '*2': 0, '*3': 4, '*4': 3},
                              'own_reverse': False,
                              'products': [Molecule(smiles="O=CO")]},
                             {'discovered_in_reverse': True,
                              'family': '1,3_Insertion_CO2',
                              'group_labels': 'Root',
                              'label_map': {'*1': 1, '*2': 0, '*3': 3, '*4': 4},
                              'own_reverse': False,
                              'products': [Molecule(smiles="O=CO")]},
                             {'discovered_in_reverse': True,
                              'family': '1,3_Insertion_CO2',
                              'group_labels': 'Root',
                              'label_map': {'*1': 1, '*2': 2, '*3': 4, '*4': 3},
                              'own_reverse': False,
                              'products': [Molecule(smiles="O=CO")]},
                             {'discovered_in_reverse': True,
                              'family': '1,3_Insertion_CO2',
                              'group_labels': 'Root',
                              'label_map': {'*1': 1, '*2': 2, '*3': 3, '*4': 4},
                              'own_reverse': False,
                              'products': [Molecule(smiles="O=CO")]}]
        self.assertEqual(products, expected_products)

        rxn_6 = ARCReaction(r_species=[ARCSpecies(label='C2H3O3', smiles='CC(=O)O[O]')],
                            p_species=[ARCSpecies(label='C2H2O', smiles='C=C=O'),
                                       ARCSpecies(label='HO2', smiles='O[O]')])
        products = get_reaction_family_products(rxn_6)
        expected_products = [{'discovered_in_reverse': False,
                              'family': 'HO2_Elimination_from_PeroxyRadical',
                              'group_labels': 'R2OO',
                              'label_map': {'*1': 0, '*2': 1, '*3': 3, '*4': 4, '*5': 5},
                              'own_reverse': False,
                              'products': [Molecule(smiles="C=C=O"), Molecule(smiles="[O]O")]},
                             {'discovered_in_reverse': False,
                              'family': 'HO2_Elimination_from_PeroxyRadical',
                              'group_labels': 'R2OO',
                              'label_map': {'*1': 0, '*2': 1, '*3': 3, '*4': 4, '*5': 6},
                              'own_reverse': False,
                              'products': [Molecule(smiles="C=C=O"), Molecule(smiles="[O]O")]},
                             {'discovered_in_reverse': False,
                              'family': 'HO2_Elimination_from_PeroxyRadical',
                              'group_labels': 'R2OO',
                              'label_map': {'*1': 0, '*2': 1, '*3': 3, '*4': 4, '*5': 7},
                              'own_reverse': False,
                              'products': [Molecule(smiles="[O]O"), Molecule(smiles="C=C=O")]}]
        self.assertEqual(products, expected_products)

        spc_1 = ARCSpecies(label='C2H3O3', smiles='[CH2]C(=O)OO')
        spc_1.mol_list = generate_resonance_structures(object_=spc_1.mol, keep_isomorphic=True)
        rxn_7 = ARCReaction(r_species=[spc_1],
                            p_species=[ARCSpecies(label='C2H2O', smiles='C=C=O'),
                                       ARCSpecies(label='HO2', smiles='O[O]')])
        products = get_reaction_family_products(rxn_7)
        expected_products = [{'discovered_in_reverse': True,
                              'family': 'R_Addition_MultipleBond',
                              'group_labels': ('R_R', 'OJ_sec'),
                              'label_map': {'*1': 1, '*2': 0, '*3': 6},
                              'own_reverse': False,
                              'products': [Molecule(smiles="[CH2]C(=O)OO")]}]
        self.assertEqual(products, expected_products)

        rxn_8 = ARCReaction(r_species=[ARCSpecies(label='C3H7O2a', smiles='C[CH]COO')],
                            p_species=[ARCSpecies(label='C3H7O2b', smiles='CC(O)C[O]')])
        products = get_reaction_family_products(rxn_8)
        expected_products = [{'discovered_in_reverse': False,
                              'family': 'intra_OH_migration',
                              'group_labels': 'R2OOH',
                              'label_map': {'*1': 1, '*2': 3, '*3': 4, '*4': 2},
                              'own_reverse': False,
                              'products': [Molecule(smiles="CC(O)C[O]")]}]
        self.assertEqual(products, expected_products)

        rxn_9 = ARCReaction(r_species=[ARCSpecies(label='R', smiles='C[CH]CCCC')],  # iso-teleological paths
                            p_species=[ARCSpecies(label='P', smiles='[CH2]CCCCC')])
        products = get_reaction_family_products(rxn_9)
        expected_products = [{'discovered_in_reverse': False,
                              'family': 'intra_H_migration',
                              'group_labels': 'R5Hall',
                              'label_map': {'*1': 1, '*2': 5, '*3': 16, '*4': 2, '*5': 4, '*6': 3},
                              'own_reverse': True,
                              'products': [Molecule(smiles="[CH2]CCCCC")]},
                             {'discovered_in_reverse': False,
                              'family': 'intra_H_migration',
                              'group_labels': 'R5Hall',
                              'label_map': {'*1': 1, '*2': 5, '*3': 17, '*4': 2, '*5': 4, '*6': 3},
                              'own_reverse': True,
                              'products': [Molecule(smiles="[CH2]CCCCC")]},
                             {'discovered_in_reverse': False,
                              'family': 'intra_H_migration',
                              'group_labels': 'R5Hall',
                              'label_map': {'*1': 1, '*2': 5, '*3': 18, '*4': 2, '*5': 4, '*6': 3},
                              'own_reverse': True,
                              'products': [Molecule(smiles="[CH2]CCCCC")]},
                             {'discovered_in_reverse': False,
                              'family': 'intra_H_migration',
                              'group_labels': 'R2H',
                              'label_map': {'*1': 1, '*2': 0, '*3': 6},
                              'own_reverse': True,
                              'products': [Molecule(smiles="[CH2]CCCCC")]},
                             {'discovered_in_reverse': False,
                              'family': 'intra_H_migration',
                              'group_labels': 'R2H',
                              'label_map': {'*1': 1, '*2': 0, '*3': 7},
                              'own_reverse': True,
                              'products': [Molecule(smiles="[CH2]CCCCC")]},
                             {'discovered_in_reverse': False,
                              'family': 'intra_H_migration',
                              'group_labels': 'R2H',
                              'label_map': {'*1': 1, '*2': 0, '*3': 8},
                              'own_reverse': True,
                              'products': [Molecule(smiles="[CH2]CCCCC")]}]
        self.assertEqual(products, expected_products)

    def test_determine_possible_reaction_products_from_family(self):
        """Test determining the possible reaction products from a family"""
        rxn_1 = ARCReaction(r_species=[ARCSpecies(label='nC3H7', smiles='[CH2]CC')],
                            p_species=[ARCSpecies(label='iC3H7', smiles='C[CH]C')])
        product_dicts = determine_possible_reaction_products_from_family(rxn_1, family_label='intra_H_migration')
        expected_product_dicts = [{'discovered_in_reverse': False,
                                   'family': 'intra_H_migration',
                                   'group_labels': 'R3Hall',
                                   'label_map': {'*1': 0, '*4': 1, '*2': 2, '*3': 7},
                                   'own_reverse': True,
                                   'products': [Molecule(smiles="[CH2]CC")]},
                                  {'discovered_in_reverse': False,
                                   'family': 'intra_H_migration',
                                   'group_labels': 'R3Hall',
                                   'label_map': {'*1': 0, '*4': 1, '*2': 2, '*3': 8},
                                   'own_reverse': True,
                                   'products': [Molecule(smiles="[CH2]CC")]},
                                  {'discovered_in_reverse': False,
                                   'family': 'intra_H_migration',
                                   'group_labels': 'R3Hall',
                                   'label_map': {'*1': 0, '*4': 1, '*2': 2, '*3': 9},
                                   'own_reverse': True,
                                   'products': [Molecule(smiles="[CH2]CC")]},
                                  {'discovered_in_reverse': False,
                                   'family': 'intra_H_migration',
                                   'group_labels': 'R2H',
                                   'label_map': {'*1': 0, '*2': 1, '*3': 5},
                                   'own_reverse': True,
                                   'products': [Molecule(smiles="C[CH]C")]},
                                  {'discovered_in_reverse': False,
                                   'family': 'intra_H_migration',
                                   'group_labels': 'R2H',
                                   'label_map': {'*1': 0, '*2': 1, '*3': 6},
                                   'own_reverse': True,
                                   'products': [Molecule(smiles="C[CH]C")]}]
        self.assertEqual(product_dicts, expected_product_dicts)

    def test_get_all_families(self):
        """Test getting all families from RMG and/or ARC"""
        families = get_all_families(consider_arc_families=False)
        self.assertIsInstance(families, list)
        self.assertIn('1,3_Insertion_ROR', families)
        self.assertIn('6_membered_central_C-C_shift', families)
        self.assertIn('H_Abstraction', families)
        self.assertIn('intra_H_migration', families)
        self.assertIn('Retroene', families)
        self.assertIn('Ketoenol', families)
        self.assertIn('intra_NO2_ONO_conversion', families)
        self.assertIn('intra_OH_migration', families)
        families = get_all_families(consider_rmg_families=False)
        self.assertIsInstance(families, list)
        self.assertIn('hydrolysis', families)

    def test_get_rmg_recommended_family_sets(self):
        """Test getting RMG recommended family sets"""
        recommended_families = get_rmg_recommended_family_sets()
        self.assertIn('default', recommended_families)
        self.assertIn('ch_pyrolysis', recommended_families)
        self.assertIn('liquid_peroxide', recommended_families)

    def test_load(self):
        """Test loading a reaction family from the RMG database"""
        fam_1 = ReactionFamily('1,3_NH3_elimination')
        self.assertEqual(fam_1.label, '1,3_NH3_elimination')
        self.assertTrue(fam_1.reversible)
        self.assertFalse(fam_1.own_reverse)
        self.assertEqual(fam_1.reactants, [['Root']])
        self.assertEqual(fam_1.product_num, 2)
        self.assertEqual(fam_1.entries, {'Root': """1 *1 N   u0 p1 c0 {2,S} {4,S} {5,S}
2 *2 R!H u0 c0 {1,S} {3,[S,D]}
3 *3 R!H u0 c0 {2,[S,D]} {6,S}
4    H   u0 p0 c0 {1,S}
5    H   u0 p0 c0 {1,S}
6 *4 H   u0 p0 c0 {3,S}"""})
        self.assertEqual(fam_1.actions, [['FORM_BOND', '*1', 1, '*4'],
                                         ['BREAK_BOND', '*1', 1, '*2'],
                                         ['BREAK_BOND', '*3', 1, '*4'],
                                         ['CHANGE_BOND', '*2', 1, '*3']])

        fam_2 = ReactionFamily('Retroene')
        self.assertEqual(fam_2.label, 'Retroene')
        self.assertTrue(fam_2.reversible)
        self.assertFalse(fam_2.own_reverse)
        self.assertEqual(fam_2.reactants, [['Root']])
        self.assertEqual(fam_2.product_num, 2)
        self.assertEqual(fam_2.entries, {'Root': """1 *3 R!H u0 {2,S} {3,[S,D]}
2 *4 R!H u0 {1,S} {4,[S,D]}
3 *2 R!H u0 {1,[S,D]} {5,[D,T,B]}
4 *5 R!H u0 {2,[S,D]} {6,S}
5 *1 R!H u0 {3,[D,T,B]}
6 *6 H   u0 {4,S}"""})
        self.assertEqual(fam_2.actions, [['CHANGE_BOND', '*1', -1, '*2'],
                                         ['BREAK_BOND', '*5', 1, '*6'],
                                         ['BREAK_BOND', '*3', 1, '*4'],
                                         ['FORM_BOND', '*1', 1, '*6'],
                                         ['CHANGE_BOND', '*2', 1, '*3'],
                                         ['CHANGE_BOND', '*4', 1, '*5']])

        fam_3 = ReactionFamily('H_Abstraction')
        self.assertEqual(fam_3.label, 'H_Abstraction')
        self.assertTrue(fam_3.reversible)
        self.assertTrue(fam_3.own_reverse)
        self.assertEqual(fam_3.reactants, [['Xrad_H', 'X_H', 'C_quartet_H', 'C_doublet_H', 'CH2_triplet_H',
                                            'CH2_singlet_H', 'NH_triplet_H', 'NH_singlet_H'],
                                           ['Y_rad', 'Y_1centerbirad', 'N_atom_quartet', 'N_atom_doublet',
                                            'CH_quartet', 'CH_doublet', 'C_quintet', 'C_triplet']])
        self.assertEqual(fam_3.product_num, 2)
        self.assertEqual(fam_3.entries, {'CH2_singlet_H': '1 *1 C u0 p1 {2,S} {3,S}\n2 *2 H u0 {1,S}\n3    H u0 {1,S}',
                                         'CH2_triplet_H': '1 *1 Cs u2 {2,S} {3,S}\n2 *2 H  u0 {1,S}\n3    H  u0 {1,S}',
                                         'CH_doublet': '1 *3 C u1 p1 {2,S}\n2    H u0 {1,S}',
                                         'CH_quartet': '1 *3 C u3 p0 {2,S}\n2    H u0 p0 {1,S}',
                                         'C_doublet_H': '1 *1 C u1 p1 {2,S}\n2 *2 H u0 p0 {1,S}',
                                         'C_quartet_H': '1 *1 C u3 p0 {2,S}\n2 *2 H u0 p0 {1,S}',
                                         'C_quintet': '1 *3 C u4 p0',
                                         'C_triplet': '1 *3 C u2 p1',
                                         'NH_singlet_H': '1 *1 N u0 p2 {2,S}\n2 *2 H u0 {1,S}',
                                         'NH_triplet_H': '1 *1 N u2 p1 {2,S}\n2 *2 H u0 {1,S}',
                                         'N_atom_doublet': '1 *3 N u1 p2',
                                         'N_atom_quartet': '1 *3 N u3 p1',
                                         'X_H': '1 *1 R u0 {2,S}\n2 *2 H u0 {1,S}',
                                         'Xrad_H': '1 *1 R!H u1 {2,S}\n2 *2 H   u0 {1,S}',
                                         'Y_1centerbirad': '1 *3 [Cs,Cd,CO,CS,O,S,N] u2',
                                         'Y_rad': '1 *3 R u1'})
        self.assertEqual(fam_3.actions, [['BREAK_BOND', '*1', 1, '*2'],
                                         ['FORM_BOND', '*2', 1, '*3'],
                                         ['GAIN_RADICAL', '*1', '1'],
                                         ['LOSE_RADICAL', '*3', '1']])

    def test_generate_products(self):
        """Test generating products from a family reaction"""
        nc3h7_xyz = """C       1.37804814    0.27791806   -0.19510872
C       0.17556863   -0.34036372    0.43264370
C      -0.83187347    0.70417963    0.88324345
H       2.32472321   -0.25030138   -0.17788282
H       1.28333786    1.14668473   -0.83693739
H      -0.29365347   -1.02041871   -0.28598479
H       0.48921065   -0.93758132    1.29559484
H      -1.19281046    1.29833383    0.03681883
H      -1.69637465    0.21982206    1.34848803
H      -0.39179140    1.38837612    1.61666966"""  # C_0 is the radical site that is labeled *1 in intra_H_migration
        reactants = [ARCSpecies(label='nC3H7', smiles='[CH2]CC', xyz=nc3h7_xyz)]
        fam_1 = ReactionFamily('intra_H_migration')
        products = fam_1.generate_products(reactants=reactants)
        # {'R2H': [[Molecule(smiles="C[CH]C")], [Molecule(smiles="C[CH]C")]],
        #  'R3Hall': [[Molecule(smiles="[CH2]CC")], [Molecule(smiles="[CH2]CC")], [Molecule(smiles="[CH2]CC")]]}
        self.assertEqual(list(products.keys()), ['R3Hall', 'R2H'])
        self.assertEqual(len(products['R2H']), 2)
        self.assertEqual(len(products['R3Hall']), 3)
        mol_2 = Molecule(smiles='C[CH]C')
        mol_3 = Molecule(smiles='CC[CH2]')
        for i, (mol, isomorphic_subgraph) in enumerate(products['R2H']):
            self.assertTrue(mol[0].is_isomorphic(mol_2))
            self.assertEqual(isomorphic_subgraph, {1: '*2', 0: '*1', 5 + i: '*3'})
        for i, (mol, isomorphic_subgraph) in enumerate(products['R3Hall']):
            self.assertTrue(mol[0].is_isomorphic(mol_3))
            self.assertEqual(isomorphic_subgraph, {1: '*4', 2: '*2', 0: '*1', 7 + i: '*3'})

        reactants = [ARCSpecies(label='OH', smiles='[OH]'), ARCSpecies(label='NCC', smiles='NCC')]
        fam_2 = ReactionFamily('H_Abstraction')
        products = fam_2.generate_products(reactants=reactants)
        # {('Y_rad', 'X_H'): [[Molecule(smiles="CC[NH]"), Molecule(smiles="O")],
        #                     [Molecule(smiles="CC[NH]"), Molecule(smiles="O")],
        #                     [Molecule(smiles="C[CH]N"), Molecule(smiles="O")],
        #                     [Molecule(smiles="C[CH]N"), Molecule(smiles="O")],
        #                     [Molecule(smiles="[CH2]CN"), Molecule(smiles="O")],
        #                     [Molecule(smiles="[CH2]CN"), Molecule(smiles="O")],
        #                     [Molecule(smiles="O"), Molecule(smiles="[CH2]CN")]]}
        self.assertEqual(list(products.keys()), [('Y_rad', 'X_H')])
        self.assertEqual(len(products[('Y_rad', 'X_H')]), 7)
        water = Molecule(smiles='O')
        mol_1 = Molecule(smiles='CC[NH]')
        mol_2 = Molecule(smiles='C[CH]N')
        mol_3 = Molecule(smiles='[CH2]CN')
        isomorphism_count = {1: 0, 2: 0, 3: 0}
        for i, prod_list in enumerate(products[('Y_rad', 'X_H')]):
            self.assertEqual(len(prod_list), 2)
            self.assertEqual(len(prod_list[0]), 2)
            self.assertTrue(any([mol.is_isomorphic(water) for mol in prod_list[0]]))  # continue fixing tests, your on thwe right path, then implement this atom label map in functions
            if any([mol.is_isomorphic(mol_1) for mol in prod_list[0]]):
                isomorphism_count[1] += 1
            if any([mol.is_isomorphic(mol_2) for mol in prod_list[0]]):
                isomorphism_count[2] += 1
            if any([mol.is_isomorphic(mol_3) for mol in prod_list[0]]):
                isomorphism_count[3] += 1
            self.assertEqual(prod_list[1][5 + i], '*2')
        self.assertEqual(isomorphism_count, {1: 2, 2: 2, 3: 3})

        reactants = [ARCSpecies(label='CCNO2', smiles='CC[N+](=O)[O-]')]
        fam_3 = ReactionFamily('intra_NO2_ONO_conversion')
        products = fam_3.generate_products(reactants=reactants)
        self.assertEqual(list(products.keys()), ['RNO2'])
        self.assertEqual(len(products['RNO2']), 1)

    def test_apply_recipe(self):
        """Test applying a recipe to a reaction"""
        enol_xyz = """N      -1.08230641   -0.01219735   -0.71533019
C      -0.39232442   -0.36252334    0.40095597
C       0.86960167   -0.07311243    0.72761693
O       1.72832531    0.66102764   -0.04056470
H      -0.67544356    0.53998289   -1.45605881
H      -2.04604227   -0.31325832   -0.81723278
H      -0.98464818   -0.95797120    1.09502934
H       1.34031259   -0.39779001    1.64713425
H       1.24252625    0.91583948   -0.84155142"""
        fam_1 = ReactionFamily('Ketoenol')
        reactant = ARCSpecies(label='enol', smiles='NC=CO', xyz=enol_xyz).mol
        isomorphic_subgraph = {2: '*2', 3: '*3', 1: '*1', 8: '*4'}
        updated_structures = fam_1.apply_recipe(mols=[reactant], isomorphic_subgraph=isomorphic_subgraph)
        self.assertEqual(len(updated_structures), 1)
        product = Molecule(smiles='NCC=O')
        self.assertTrue(updated_structures[0].is_isomorphic(product))

    def test_get_reactant_num(self):
        """Test getting the number of reactants from a family"""
        fam_1 = ReactionFamily('intra_H_migration')
        self.assertEqual(fam_1.get_reactant_num(), 1)
        fam_2 = ReactionFamily('1,3_Insertion_CO2')
        self.assertEqual(fam_2.get_reactant_num(), 2)
        fam_3 = ReactionFamily('H_Abstraction')
        self.assertEqual(fam_3.get_reactant_num(), 2)

    def test_add_labels_to_molecule(self):
        """Test adding labels to a molecule"""
        mol = Molecule().from_smiles('C')
        labels = {0: '*1', 2: '*2'}
        mol = add_labels_to_molecule(mol, labels)
        self.assertEqual(mol.to_adjacency_list(), """1 *1 C u0 p0 c0 {2,S} {3,S} {4,S} {5,S}
2    H u0 p0 c0 {1,S}
3 *2 H u0 p0 c0 {1,S}
4    H u0 p0 c0 {1,S}
5    H u0 p0 c0 {1,S}
""")

    def test_is_reversible(self):
        """Test checking if a reaction family is reversible"""
        groups_as_lines = ['template(reactants=["Root"], products=["ketone"], ownReverse=False)',
                           'reverse = "Ketone_To_Enol"', 'reversible = True']
        self.assertTrue(is_reversible(groups_as_lines))
        groups_as_lines = ['Template(reactants=["singlet"], products=["triplet"], ownReverse=False)',
                           'reverse = "None"', 'reversible = False']
        self.assertFalse(is_reversible(groups_as_lines))

    def test_is_own_reverse(self):
        """Test checking if a reaction family is its own reverse"""
        groups_as_lines = ['template(reactants=["Root"], products=["ketone"], ownReverse=False)',
                           'reverse = "Ketone_To_Enol"', 'reversible = True']
        self.assertFalse(is_own_reverse(groups_as_lines))
        groups_as_lines = ['template(reactants=["X_H_or_Xrad_H_Xbirad_H_Xtrirad_H", "Y_rad_birad_trirad_quadrad"], ' +
                           'products=["X_H_or_Xrad_H_Xbirad_H_Xtrirad_H", "Y_rad_birad_trirad_quadrad"], ' +
                           'ownReverse=True)', 'reversible = True']
        self.assertTrue(is_own_reverse(groups_as_lines))

    def test_get_product_num(self):
        """Test getting the number of products from a family"""
        groups_as_lines = ['template(reactants=["Root"], products=["ketone"], ownReverse=False)',
                           'reverse = "Ketone_To_Enol"', 'reversible = True']
        self.assertEqual(get_product_num(groups_as_lines), 1)
        groups_as_lines = ['template(reactants=["X_H_or_Xrad_H_Xbirad_H_Xtrirad_H", "Y_rad_birad_trirad_quadrad"], ' +
                           'products=["X_H_or_Xrad_H_Xbirad_H_Xtrirad_H", "Y_rad_birad_trirad_quadrad"], ' +
                           'ownReverse=True)', 'reversible = True']
        self.assertEqual(get_product_num(groups_as_lines), 2)

    def test_get_reactant_groups_from_template(self):
        """Test getting reactant groups from a template"""
        fam_1 = ReactionFamily('6_membered_central_C-C_shift')
        groups_path = os.path.join(RMG_DB_PATH, 'input', 'kinetics', 'families', fam_1.label, 'groups.py')
        with open(groups_path, 'r') as f:
            groups = f.readlines()
        reactants_1 = get_reactant_groups_from_template(groups)
        self.assertEqual(reactants_1, [['1_5_unsaturated_hexane']])

        fam_2 = ReactionFamily('H_Abstraction')
        groups_path = os.path.join(RMG_DB_PATH, 'input', 'kinetics', 'families', fam_2.label, 'groups.py')
        with open(groups_path, 'r') as f:
            groups = f.readlines()
        reactants_2 = get_reactant_groups_from_template(groups)
        expected_reactants = [['Xrad_H', 'X_H', 'C_quartet_H', 'C_doublet_H', 'CH2_triplet_H', 'CH2_singlet_H', 'NH_triplet_H', 'NH_singlet_H'],
                              ['Y_rad', 'Y_1centerbirad', 'N_atom_quartet', 'N_atom_doublet', 'CH_quartet', 'CH_doublet', 'C_quintet', 'C_triplet']]
        self.assertEqual(reactants_2, expected_reactants)

        fam_3 = ReactionFamily('1,2-Birad_to_alkene')
        groups_path = os.path.join(RMG_DB_PATH, 'input', 'kinetics', 'families', fam_3.label, 'groups.py')
        with open(groups_path, 'r') as f:
            groups = f.readlines()
        reactants_3 = get_reactant_groups_from_template(groups)
        expected_reactants = [['Y_12_13', 'Y_12_31', 'NOS', 'Y_12_00', 'Y_12_10', 'Y_12_20a', 'Y_12_20b', 'Y_12_30',
                               'Y_12_40', 'Y_12_01', 'Y_12_02a', 'Y_12_02b', 'Y_12_03', 'Y_12_04', 'Y_12_11a',
                               'Y_12_11b', 'Y_12_12a', 'Y_12_12b', 'Y_12_21a', 'Y_12_21b', 'Y_12_22a', 'Y_12_22b']]
        self.assertEqual(reactants_3, expected_reactants)

    def test_descent_complex_group(self):
        """Test getting the specific groups of a complex group"""
        self.assertEqual(descent_complex_group('N5c'), ['N5c'])
        self.assertEqual(descent_complex_group("OR{Xtrirad_H, Xbirad_H, Xrad_H, X_H}"),
                         ['Xtrirad_H', 'Xbirad_H', 'Xrad_H', 'X_H'])

    def test_get_initial_reactant_labels_from_template(self):
        """Test getting initial reactant labels from a template"""
        content = ['template(reactants=["RnH"], products=["RnH"], ownReverse=True)']
        reactants = get_initial_reactant_labels_from_template(content)
        self.assertEqual(reactants, ['RnH'])
        products = get_initial_reactant_labels_from_template(content, products=True)
        self.assertEqual(products, ['RnH'])

        content = ['template(reactants=["Root"], products=["RR", "NH3"], ownReverse=False)']
        reactants = get_initial_reactant_labels_from_template(content)
        self.assertEqual(reactants, ['Root'])
        products = get_initial_reactant_labels_from_template(content, products=True)
        self.assertEqual(products, ['RR', 'NH3'])

        content = ['template(reactants=["X_H_or_Xrad_H_Xbirad_H_Xtrirad_H", "Y_rad_birad_trirad_quadrad"], '
                   'products=["X_H_or_Xrad_H_Xbirad_H_Xtrirad_H", "Y_rad_birad_trirad_quadrad"], ownReverse=True)']
        reactants = get_initial_reactant_labels_from_template(content)
        self.assertEqual(reactants, ['X_H_or_Xrad_H_Xbirad_H_Xtrirad_H', 'Y_rad_birad_trirad_quadrad'])
        products = get_initial_reactant_labels_from_template(content, products=True)
        self.assertEqual(products, ['X_H_or_Xrad_H_Xbirad_H_Xtrirad_H', 'Y_rad_birad_trirad_quadrad'])

    def test_get_recipe_actions(self):
        """Test getting recipe actions"""
        recipe_lines = [
            "recipe(actions=[",
            "['BREAK_BOND', '*1', 1, '*2'],",
            "['FORM_BOND', '*2', 1, '*3'],",
            "['GAIN_RADICAL', '*1', '1'],",
            "['LOSE_RADICAL', '*3', '1'],",
            "])"]
        actions = get_recipe_actions(recipe_lines)
        self.assertEqual(actions, [['BREAK_BOND', '*1', 1, '*2'],
                                   ['FORM_BOND', '*2', 1, '*3'],
                                   ['GAIN_RADICAL', '*1', '1'],
                                   ['LOSE_RADICAL', '*3', '1']])

        fam_1 = ReactionFamily('6_membered_central_C-C_shift')
        groups_path = os.path.join(RMG_DB_PATH, 'input', 'kinetics', 'families', fam_1.label, 'groups.py')
        with open(groups_path, 'r') as f:
            groups = f.readlines()
        actions = get_recipe_actions(groups)
        self.assertEqual(actions, [['BREAK_BOND', '*3', 1, '*4'],
                                   ['CHANGE_BOND', '*1', -1, '*2'],
                                   ['CHANGE_BOND', '*5', -1, '*6'],
                                   ['CHANGE_BOND', '*2', 1, '*3'],
                                   ['CHANGE_BOND', '*4', 1, '*5'],
                                   ['FORM_BOND', '*1', 1, '*6']])

    def test_get_entries(self):
        """Test getting entries from a family"""
        fam_1 = ReactionFamily('1,3_Insertion_ROR')
        groups_as_lines = fam_1.get_groups_file_as_lines()
        entries = get_entries(groups_as_lines=groups_as_lines, entry_labels=['doublebond', 'cco_2H'])
        self.assertEqual(entries, {'doublebond': 'OR{Cd_Cdd, Cdd_Cd, Cd_Cd, Sd_Cd, N1dc_N5ddc, N3d_Cd}',
                                   'cco_2H': """1 *1 Cd        u0 {2,D} {3,S} {4,S}
2 *2 Cdd       u0 {1,D} {5,D}
3    H         u0 {1,S}
4    H         u0 {1,S}
5    [O2d,S2d] u0 {2,D}"""})

    def test_get_isomorphic_subgraph(self):
        """Test getting the isomorphic subgraph"""
        oh_xyz = {'symbols': ('O', 'H'), 'isotopes': (16, 1), 'coords': ((0.0, 0.0, 0.6131), (0.0, 0.0, -0.6131))}
        ncc_xyz = {'symbols': ('N', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H'),
                   'isotopes': (14, 12, 12, 1, 1, 1, 1, 1, 1, 1),
                   'coords': ((1.2562343261659277, 0.055282951474465256, -0.5803566480287667),
                              (0.2433806820538858, -0.2998022593250662, 0.4048026179081362),
                              (-1.1419715208061372, 0.05031794898903049, -0.10936163768735493),
                              (0.30010122891852403, -1.3732777777516794, 0.6116620393221122),
                              (0.4378838143353429, 0.23284176746008836, 1.3415344409291001),
                              (-1.3668225970409684, -0.4847381186432614, -1.0382269930378532),
                              (-1.2346536276957667, 1.1246102699428049, -0.30155738107540364),
                              (-1.899948780117477, -0.22595560919578353, 0.6306294743856438),
                              (1.224390390624704, 1.0579934139157359, -0.7603389078698333),
                              (2.1814060835619262, -0.13727258686634466, -0.1987870048457775))}
        spc_1 = ARCSpecies(label='OH', smiles='[OH]', xyz=oh_xyz)
        spc_2 = ARCSpecies(label='NCC', smiles='NCC', xyz=ncc_xyz)
        groups_as_lines = ReactionFamily('H_Abstraction').groups_as_lines
        group_1 = Group().from_adjacency_list(get_group_adjlist(groups_as_lines=groups_as_lines, entry_label='Y_rad'))
        group_2 = Group().from_adjacency_list(get_group_adjlist(groups_as_lines=groups_as_lines, entry_label='X_H'))
        isomorphic_subgraphs_1 = spc_1.mol.find_subgraph_isomorphisms(other=group_1, save_order=True)
        isomorphic_subgraphs_2 = spc_2.mol.find_subgraph_isomorphisms(other=group_2, save_order=True)
        self.assertEqual(len(isomorphic_subgraphs_1), 1)
        self.assertEqual(len(isomorphic_subgraphs_2), 7)
        for key, val in isomorphic_subgraphs_1[0].items():  # [{<Atom 'O.'>: <GroupAtom [*3 'R']>}]
            self.assertEqual(key.atomtype.label, 'O2s')
            self.assertEqual(val.label, '*3')
        for isomorphic_subgraph in isomorphic_subgraphs_2:
            # [{<Atom 'C'>: <GroupAtom [*1 'R']>, <Atom 'H'>: <GroupAtom [*2 'H']>}, {<Atom 'C'>: <GroupAtom [*1 'R']>,
            # <Atom 'H'>: <GroupAtom [*2 'H']>}, {<Atom 'C'>: <GroupAtom [*1 'R']>, <Atom 'H'>: <GroupAtom [*2 'H']>},
            # {<Atom 'C'>: <GroupAtom [*1 'R']>, <Atom 'H'>: <GroupAtom [*2 'H']>}, {<Atom 'C'>: <GroupAtom [*1 'R']>,
            # <Atom 'H'>: <GroupAtom [*2 'H']>}, {<Atom 'N'>: <GroupAtom [*1 'R']>, <Atom 'H'>: <GroupAtom [*2 'H']>},
            # {<Atom 'N'>: <GroupAtom [*1 'R']>, <Atom 'H'>: <GroupAtom [*2 'H']>}]
            for key, val in isomorphic_subgraph.items():
                if key.is_hydrogen():
                    self.assertEqual(val.label, '*2')
                else:
                    self.assertEqual(val.label, '*1')
        isomorphic_subgraph = get_isomorphic_subgraph(isomorphic_subgraph_1=isomorphic_subgraphs_1[0],
                                                      isomorphic_subgraph_2=isomorphic_subgraphs_2[0],
                                                      mol_1=spc_1.mol,
                                                      mol_2=spc_2.mol,
                                                      )
        self.assertEqual(isomorphic_subgraph, {0: '*3', 3: '*1', 5: '*2'})
        isomorphic_subgraph = get_isomorphic_subgraph(isomorphic_subgraph_1=isomorphic_subgraphs_1[0],
                                                      isomorphic_subgraph_2=isomorphic_subgraphs_2[2],
                                                      mol_1=spc_1.mol,
                                                      mol_2=spc_2.mol,
                                                      )
        self.assertEqual(isomorphic_subgraph, {0: '*3', 4: '*1', 7: '*2'})


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
