#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the
arc.job.adapters.ts.linear_utils.math_zmat module
"""

import unittest

from arc.species import ARCSpecies
from arc.reaction import ARCReaction

from arc.job.adapters.ts.linear_utils.math_zmat import (
    BASE_WEIGHT_GRID,
    HAMMOND_DELTA,
    _clip01,
    _get_all_referenced_atoms,
    _get_all_zmat_rows,
    average_zmat_params,
    get_r_constraints,
    get_rxn_weight,
    get_weight_grid,
    interp_dihedral_deg,
)


def _make_spc(label: str, smiles: str, e0=None, e_elect=None) -> ARCSpecies:
    """Create an ARCSpecies and set energy attributes."""
    spc = ARCSpecies(label=label, smiles=smiles)
    if e0 is not None:
        spc.e0 = e0
    if e_elect is not None:
        spc.e_elect = e_elect
    return spc


class TestMathZmat(unittest.TestCase):
    """Contains unit tests for Z-matrix math utilities."""

    @classmethod
    def setUpClass(cls):
        """A method that is run before all unit tests in this class."""
        cls.maxDiff = None

    # ------------------------------------------------------------------
    # _get_all_zmat_rows
    # ------------------------------------------------------------------
    def test_get_all_zmat_rows_simple_bond(self):
        """Standard bond-length variable encodes one row."""
        self.assertEqual(_get_all_zmat_rows('R_1_0'), [1])

    def test_get_all_zmat_rows_consolidated(self):
        """Consolidated variable encodes multiple rows via pipe separator."""
        self.assertEqual(_get_all_zmat_rows('R_2|4_0|0'), [2, 4])

    def test_get_all_zmat_rows_angle(self):
        """Angle variable has a single row."""
        self.assertEqual(_get_all_zmat_rows('A_3_1_0'), [3])

    def test_get_all_zmat_rows_dihedral(self):
        """Dihedral variable has a single row."""
        self.assertEqual(_get_all_zmat_rows('D_4_1_0_2'), [4])

    def test_get_all_zmat_rows_dummy_dihedral(self):
        """Dummy-atom dihedral has a single row."""
        self.assertEqual(_get_all_zmat_rows('DX_5_1_0_2'), [5])

    def test_get_all_zmat_rows_unparseable(self):
        """Unparseable variable name returns empty list."""
        self.assertEqual(_get_all_zmat_rows('garbage'), [])

    def test_get_all_zmat_rows_empty_string(self):
        """Empty string returns empty list."""
        self.assertEqual(_get_all_zmat_rows(''), [])

    def test_get_all_zmat_rows_non_numeric(self):
        """Non-numeric row index returns empty list."""
        self.assertEqual(_get_all_zmat_rows('R_abc_0'), [])

    # ------------------------------------------------------------------
    # _get_all_referenced_atoms
    # ------------------------------------------------------------------
    def test_get_all_referenced_atoms_bond(self):
        """Bond variable references two atoms."""
        self.assertEqual(_get_all_referenced_atoms('R_1_0'), [1, 0])

    def test_get_all_referenced_atoms_consolidated(self):
        """Consolidated variable references all packed indices."""
        self.assertEqual(_get_all_referenced_atoms('R_2|4_0|0'), [2, 4, 0, 0])

    def test_get_all_referenced_atoms_angle(self):
        """Angle variable references three atoms."""
        self.assertEqual(_get_all_referenced_atoms('A_3_1_0'), [3, 1, 0])

    def test_get_all_referenced_atoms_dihedral(self):
        """Dihedral variable references four atoms."""
        self.assertEqual(_get_all_referenced_atoms('D_4_3_0_2'), [4, 3, 0, 2])

    def test_get_all_referenced_atoms_dummy_dihedral(self):
        """DX prefix is stripped, remaining indices are parsed."""
        result = _get_all_referenced_atoms('DX_5_1_0_2')
        self.assertEqual(result, [5, 1, 0, 2])

    def test_get_all_referenced_atoms_unparseable(self):
        """Unparseable returns empty list."""
        self.assertEqual(_get_all_referenced_atoms('x'), [])

    # ------------------------------------------------------------------
    # _clip01
    # ------------------------------------------------------------------
    def test_clip01_in_range(self):
        """Value in [0, 1] is returned unchanged."""
        self.assertEqual(_clip01(0.5), 0.5)

    def test_clip01_below_zero(self):
        """Negative value is clipped to 0."""
        self.assertEqual(_clip01(-0.1), 0.0)

    def test_clip01_above_one(self):
        """Value above 1 is clipped to 1."""
        self.assertEqual(_clip01(1.5), 1.0)

    def test_clip01_boundary_zero(self):
        """Exact 0 passes through."""
        self.assertEqual(_clip01(0.0), 0.0)

    def test_clip01_boundary_one(self):
        """Exact 1 passes through."""
        self.assertEqual(_clip01(1.0), 1.0)

    # ------------------------------------------------------------------
    # interp_dihedral_deg
    # ------------------------------------------------------------------
    def test_interp_dihedral_w0_returns_a(self):
        """Weight 0 returns the first angle."""
        self.assertAlmostEqual(interp_dihedral_deg(30.0, 90.0, w=0.0), 30.0)

    def test_interp_dihedral_w1_returns_b(self):
        """Weight 1 returns the second angle."""
        self.assertAlmostEqual(interp_dihedral_deg(30.0, 90.0, w=1.0), 90.0)

    def test_interp_dihedral_midpoint(self):
        """Weight 0.5 returns the midpoint of two angles on the same side."""
        self.assertAlmostEqual(interp_dihedral_deg(30.0, 90.0, w=0.5), 60.0)

    def test_interp_dihedral_wraparound(self):
        """Interpolation wraps correctly across the +-180 boundary."""
        result = interp_dihedral_deg(-179.0, 179.0, w=0.5)
        self.assertAlmostEqual(abs(result), 180.0, places=5)

    def test_interp_dihedral_wraparound_quarter(self):
        """Wraparound at w=0.25: from -170 toward 170 should go through 180."""
        result = interp_dihedral_deg(-170.0, 170.0, w=0.5)
        self.assertAlmostEqual(abs(result), 180.0, places=5)

    def test_interp_dihedral_same_angle(self):
        """Interpolating between identical angles returns that angle."""
        self.assertAlmostEqual(interp_dihedral_deg(45.0, 45.0, w=0.5), 45.0)

    def test_interp_dihedral_large_input_normalized(self):
        """Input angles outside [-180, 180] are normalized before interpolation."""
        result = interp_dihedral_deg(350.0, 370.0, w=0.5)
        self.assertGreaterEqual(result, -180.0)
        self.assertLessEqual(result, 180.0)

    # ------------------------------------------------------------------
    # get_r_constraints
    # ------------------------------------------------------------------
    def test_get_r_constraints_empty_bonds(self):
        """No bonds produces empty constraints."""
        result = get_r_constraints([], [])
        self.assertEqual(result, {'R_atom': []})

    def test_get_r_constraints_single_breaking_bond(self):
        """A single breaking bond is included in R_atom constraints."""
        result = get_r_constraints([(0, 1)], [])
        self.assertEqual(len(result['R_atom']), 1)
        self.assertIn(result['R_atom'][0], [(0, 1), (1, 0)])

    def test_get_r_constraints_breaking_and_forming(self):
        """Atoms appearing in both breaking and forming bonds are prioritized."""
        result = get_r_constraints([(0, 1)], [(1, 2)])
        constraints = result['R_atom']
        self.assertTrue(len(constraints) >= 1)
        all_atoms = set()
        for bond in constraints:
            all_atoms.update(bond)
        self.assertIn(1, all_atoms)

    def test_get_r_constraints_frequency_sorting(self):
        """Atom with highest frequency in bond changes appears first."""
        result = get_r_constraints([(0, 1), (1, 2)], [(1, 3)])
        constraints = result['R_atom']
        self.assertTrue(constraints[0][0] == 1)

    def test_get_r_constraints_deterministic(self):
        """Same inputs produce identical output regardless of call order."""
        r1 = get_r_constraints([(2, 5), (3, 5)], [(5, 7)])
        r2 = get_r_constraints([(2, 5), (3, 5)], [(5, 7)])
        self.assertEqual(r1, r2)

    # ------------------------------------------------------------------
    # average_zmat_params
    # ------------------------------------------------------------------
    def test_average_zmat_params_weight_out_of_range(self):
        """Weight outside [0, 1] returns None."""
        zmat = {'vars': {'R_1_0': 1.5}, 'coords': [('R_1_0', '', '')]}
        self.assertIsNone(average_zmat_params(zmat, zmat, weight=-0.1))
        self.assertIsNone(average_zmat_params(zmat, zmat, weight=1.5))

    def test_average_zmat_params_missing_vars(self):
        """Missing 'vars' key returns None."""
        zmat_no_vars = {'coords': [('R_1_0', '', '')]}
        zmat_ok = {'vars': {'R_1_0': 1.5}, 'coords': [('R_1_0', '', '')]}
        self.assertIsNone(average_zmat_params(zmat_no_vars, zmat_ok))

    def test_average_zmat_params_missing_coords(self):
        """Missing 'coords' key returns None."""
        zmat_no_coords = {'vars': {'R_1_0': 1.5}}
        zmat_ok = {'vars': {'R_1_0': 1.5}, 'coords': [('R_1_0', '', '')]}
        self.assertIsNone(average_zmat_params(zmat_no_coords, zmat_ok))

    def test_average_zmat_params_mismatched_keys(self):
        """Mismatched variable keys returns None."""
        zmat_1 = {'vars': {'R_1_0': 1.5}, 'coords': [('R_1_0', '', '')],
                  'symbols': ('C', 'H'), 'isotopes': (12, 1), 'map': {0: 0, 1: 1}}
        zmat_2 = {'vars': {'R_2_0': 2.0}, 'coords': [('R_2_0', '', '')],
                  'symbols': ('C', 'H'), 'isotopes': (12, 1), 'map': {0: 0, 1: 1}}
        self.assertIsNone(average_zmat_params(zmat_1, zmat_2))

    def test_average_zmat_params_bond_length_interpolation(self):
        """Bond lengths are linearly interpolated."""
        zmat_1 = {'vars': {'R_1_0': 1.0}, 'coords': [('R_1_0', '', '')],
                  'symbols': ('C', 'H'), 'isotopes': (12, 1), 'map': {0: 0, 1: 1}}
        zmat_2 = {'vars': {'R_1_0': 2.0}, 'coords': [('R_1_0', '', '')],
                  'symbols': ('C', 'H'), 'isotopes': (12, 1), 'map': {0: 0, 1: 1}}
        result = average_zmat_params(zmat_1, zmat_2, weight=0.5)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result['vars']['R_1_0'], 1.5)

    def test_average_zmat_params_bond_length_clamping(self):
        """Bond lengths below 0.4 A are clamped."""
        zmat_1 = {'vars': {'R_1_0': 0.5}, 'coords': [('R_1_0', '', '')],
                  'symbols': ('C', 'H'), 'isotopes': (12, 1), 'map': {0: 0, 1: 1}}
        zmat_2 = {'vars': {'R_1_0': 0.1}, 'coords': [('R_1_0', '', '')],
                  'symbols': ('C', 'H'), 'isotopes': (12, 1), 'map': {0: 0, 1: 1}}
        result = average_zmat_params(zmat_1, zmat_2, weight=1.0)
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result['vars']['R_1_0'], 0.4)

    def test_average_zmat_params_angle_interpolation(self):
        """Angles are linearly interpolated."""
        zmat_1 = {'vars': {'A_2_1_0': 90.0}, 'coords': [('', '', ''), ('', '', ''), ('', 'A_2_1_0', '')],
                  'symbols': ('C', 'C', 'H'), 'isotopes': (12, 12, 1), 'map': {0: 0, 1: 1, 2: 2}}
        zmat_2 = {'vars': {'A_2_1_0': 120.0}, 'coords': [('', '', ''), ('', '', ''), ('', 'A_2_1_0', '')],
                  'symbols': ('C', 'C', 'H'), 'isotopes': (12, 12, 1), 'map': {0: 0, 1: 1, 2: 2}}
        result = average_zmat_params(zmat_1, zmat_2, weight=0.5)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result['vars']['A_2_1_0'], 105.0)

    def test_average_zmat_params_angle_clamping_low(self):
        """Angles below 1 degree are clamped to singularity floor."""
        zmat_1 = {'vars': {'A_2_1_0': 2.0}, 'coords': [('', '', ''), ('', '', ''), ('', 'A_2_1_0', '')],
                  'symbols': ('C', 'C', 'H'), 'isotopes': (12, 12, 1), 'map': {0: 0, 1: 1, 2: 2}}
        zmat_2 = {'vars': {'A_2_1_0': -2.0}, 'coords': [('', '', ''), ('', '', ''), ('', 'A_2_1_0', '')],
                  'symbols': ('C', 'C', 'H'), 'isotopes': (12, 12, 1), 'map': {0: 0, 1: 1, 2: 2}}
        result = average_zmat_params(zmat_1, zmat_2, weight=1.0)
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result['vars']['A_2_1_0'], 1.0)

    def test_average_zmat_params_angle_clamping_high(self):
        """Angles above 179 degrees are clamped to singularity ceiling."""
        zmat_1 = {'vars': {'A_2_1_0': 178.0}, 'coords': [('', '', ''), ('', '', ''), ('', 'A_2_1_0', '')],
                  'symbols': ('C', 'C', 'H'), 'isotopes': (12, 12, 1), 'map': {0: 0, 1: 1, 2: 2}}
        zmat_2 = {'vars': {'A_2_1_0': 185.0}, 'coords': [('', '', ''), ('', '', ''), ('', 'A_2_1_0', '')],
                  'symbols': ('C', 'C', 'H'), 'isotopes': (12, 12, 1), 'map': {0: 0, 1: 1, 2: 2}}
        result = average_zmat_params(zmat_1, zmat_2, weight=1.0)
        self.assertIsNotNone(result)
        self.assertLessEqual(result['vars']['A_2_1_0'], 179.0)

    def test_average_zmat_params_dihedral_interpolation(self):
        """Dihedrals use circular shortest-path interpolation."""
        zmat_1 = {'vars': {'D_3_2_1_0': -170.0},
                  'coords': [('', '', ''), ('', '', ''), ('', '', ''),
                             ('', '', 'D_3_2_1_0')],
                  'symbols': ('C', 'C', 'C', 'H'), 'isotopes': (12, 12, 12, 1),
                  'map': {0: 0, 1: 1, 2: 2, 3: 3}}
        zmat_2 = {'vars': {'D_3_2_1_0': 170.0},
                  'coords': [('', '', ''), ('', '', ''), ('', '', ''),
                             ('', '', 'D_3_2_1_0')],
                  'symbols': ('C', 'C', 'C', 'H'), 'isotopes': (12, 12, 12, 1),
                  'map': {0: 0, 1: 1, 2: 2, 3: 3}}
        result = average_zmat_params(zmat_1, zmat_2, weight=0.5)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(abs(result['vars']['D_3_2_1_0']), 180.0, places=3)

    def test_average_zmat_params_weight_zero_returns_zmat1(self):
        """Weight 0 returns zmat_1 values."""
        zmat_1 = {'vars': {'R_1_0': 1.5}, 'coords': [('R_1_0', '', '')],
                  'symbols': ('C', 'H'), 'isotopes': (12, 1), 'map': {0: 0, 1: 1}}
        zmat_2 = {'vars': {'R_1_0': 2.5}, 'coords': [('R_1_0', '', '')],
                  'symbols': ('C', 'H'), 'isotopes': (12, 1), 'map': {0: 0, 1: 1}}
        result = average_zmat_params(zmat_1, zmat_2, weight=0.0)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result['vars']['R_1_0'], 1.5)

    def test_average_zmat_params_weight_one_returns_zmat2(self):
        """Weight 1 returns zmat_2 values."""
        zmat_1 = {'vars': {'R_1_0': 1.5}, 'coords': [('R_1_0', '', '')],
                  'symbols': ('C', 'H'), 'isotopes': (12, 1), 'map': {0: 0, 1: 1}}
        zmat_2 = {'vars': {'R_1_0': 2.5}, 'coords': [('R_1_0', '', '')],
                  'symbols': ('C', 'H'), 'isotopes': (12, 1), 'map': {0: 0, 1: 1}}
        result = average_zmat_params(zmat_1, zmat_2, weight=1.0)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result['vars']['R_1_0'], 2.5)

    def test_average_zmat_params_reactive_filtering(self):
        """With reactive_xyz_indices, only variables referencing reactive atoms are interpolated."""
        zmat_1 = {
            'vars': {'R_1_0': 1.0, 'R_2_1': 1.5},
            'coords': [('R_1_0', '', ''), ('R_2_1', '', '')],
            'symbols': ('C', 'C', 'H'),
            'isotopes': (12, 12, 1),
            'map': {0: 0, 1: 1, 2: 2},
        }
        zmat_2 = {
            'vars': {'R_1_0': 2.0, 'R_2_1': 3.0},
            'coords': [('R_1_0', '', ''), ('R_2_1', '', '')],
            'symbols': ('C', 'C', 'H'),
            'isotopes': (12, 12, 1),
            'map': {0: 0, 1: 1, 2: 2},
        }
        result = average_zmat_params(zmat_1, zmat_2, weight=0.5,
                                     reactive_xyz_indices={0, 1})
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result['vars']['R_1_0'], 1.5)
        # R_2_1 references atom 1 (reactive) → INTERPOLATED.
        self.assertAlmostEqual(result['vars']['R_2_1'], 2.25)

    # ------------------------------------------------------------------
    # get_rxn_weight
    # ------------------------------------------------------------------
    def test_get_rxn_weight_symmetric(self):
        """Thermoneutral reaction returns 0.5."""
        spc_r = _make_spc('R', '[H][H]', e0=0.0)
        spc_p = _make_spc('P', '[H][H]', e0=0.0)
        rxn = ARCReaction(reactants=['R'], products=['P'],
                          r_species=[spc_r], p_species=[spc_p])
        w = get_rxn_weight(rxn)
        self.assertAlmostEqual(w, 0.5)

    def test_get_rxn_weight_endothermic(self):
        """Endothermic reaction gives weight > 0.5 (product-like TS)."""
        spc_r = _make_spc('R', '[H][H]', e0=0.0)
        spc_p = _make_spc('P', '[H][H]', e0=100.0)
        rxn = ARCReaction(reactants=['R'], products=['P'],
                          r_species=[spc_r], p_species=[spc_p])
        w = get_rxn_weight(rxn)
        self.assertIsNotNone(w)
        self.assertGreater(w, 0.5)
        self.assertLessEqual(w, 0.7)

    def test_get_rxn_weight_exothermic(self):
        """Exothermic reaction gives weight < 0.5 (reactant-like TS)."""
        spc_r = _make_spc('R', '[H][H]', e0=100.0)
        spc_p = _make_spc('P', '[H][H]', e0=0.0)
        rxn = ARCReaction(reactants=['R'], products=['P'],
                          r_species=[spc_r], p_species=[spc_p])
        w = get_rxn_weight(rxn)
        self.assertIsNotNone(w)
        self.assertLess(w, 0.5)
        self.assertGreaterEqual(w, 0.3)

    def test_get_rxn_weight_no_energy_returns_none(self):
        """If no energies are available, returns None."""
        spc_r = _make_spc('R', '[H][H]')
        spc_p = _make_spc('P', '[H][H]')
        spc_r.e0 = None
        spc_r.e_elect = None
        spc_p.e0 = None
        spc_p.e_elect = None
        rxn = ARCReaction(reactants=['R'], products=['P'],
                          r_species=[spc_r], p_species=[spc_p])
        w = get_rxn_weight(rxn)
        self.assertIsNone(w)

    def test_get_rxn_weight_swapped_bounds(self):
        """Swapped w_min/w_max are automatically corrected."""
        spc_r = _make_spc('R', '[H][H]', e0=0.0)
        spc_p = _make_spc('P', '[H][H]', e0=50.0)
        rxn = ARCReaction(reactants=['R'], products=['P'],
                          r_species=[spc_r], p_species=[spc_p])
        w = get_rxn_weight(rxn, w_min=0.7, w_max=0.3)
        self.assertIsNotNone(w)
        self.assertGreaterEqual(w, 0.3)
        self.assertLessEqual(w, 0.7)

    def test_get_rxn_weight_invalid_bounds(self):
        """Invalid bounds (w_min > 0.5) raise ValueError."""
        spc_r = _make_spc('R', '[H][H]', e0=0.0)
        spc_p = _make_spc('P', '[H][H]', e0=50.0)
        rxn = ARCReaction(reactants=['R'], products=['P'],
                          r_species=[spc_r], p_species=[spc_p])
        with self.assertRaises(ValueError):
            get_rxn_weight(rxn, w_min=0.6, w_max=0.8)

    def test_get_rxn_weight_negative_delta_e_sat(self):
        """Negative delta_e_sat raises ValueError."""
        spc_r = _make_spc('R', '[H][H]', e0=0.0)
        spc_p = _make_spc('P', '[H][H]', e0=50.0)
        rxn = ARCReaction(reactants=['R'], products=['P'],
                          r_species=[spc_r], p_species=[spc_p])
        with self.assertRaises(ValueError):
            get_rxn_weight(rxn, delta_e_sat=-10.0)

    def test_get_rxn_weight_clamped_to_bounds(self):
        """Very large delta_E gets clamped to w_max."""
        spc_r = _make_spc('R', '[H][H]', e0=0.0)
        spc_p = _make_spc('P', '[H][H]', e0=1000.0)
        rxn = ARCReaction(reactants=['R'], products=['P'],
                          r_species=[spc_r], p_species=[spc_p])
        w = get_rxn_weight(rxn)
        self.assertIsNotNone(w)
        self.assertEqual(w, 0.7)

    def test_get_rxn_weight_e_elect_fallback(self):
        """Falls back to e_elect when e0 is unavailable."""
        spc_r = _make_spc('R', '[H][H]', e_elect=0.0)
        spc_p = _make_spc('P', '[H][H]', e_elect=50.0)
        spc_r.e0 = None
        spc_p.e0 = None
        rxn = ARCReaction(reactants=['R'], products=['P'],
                          r_species=[spc_r], p_species=[spc_p])
        w = get_rxn_weight(rxn)
        self.assertIsNotNone(w)
        self.assertGreater(w, 0.5)

    def test_get_rxn_weight_custom_reorg_energy_float(self):
        """Custom reorganization energy as a float."""
        spc_r = _make_spc('R', '[H][H]', e0=0.0)
        spc_p = _make_spc('P', '[H][H]', e0=50.0)
        rxn = ARCReaction(reactants=['R'], products=['P'],
                          r_species=[spc_r], p_species=[spc_p])
        w = get_rxn_weight(rxn, reorg_energy=200.0)
        self.assertIsNotNone(w)
        expected = 0.5 + 50.0 / (2.0 * 200.0)
        self.assertAlmostEqual(w, expected, places=4)

    def test_get_rxn_weight_custom_reorg_energy_tuple(self):
        """Custom reorganization energy as a (lambda_exo, lambda_endo) tuple."""
        spc_r = _make_spc('R', '[H][H]', e0=0.0)
        spc_p = _make_spc('P', '[H][H]', e0=50.0)
        rxn = ARCReaction(reactants=['R'], products=['P'],
                          r_species=[spc_r], p_species=[spc_p])
        w = get_rxn_weight(rxn, reorg_energy=(200.0, 300.0))
        self.assertIsNotNone(w)
        expected = 0.5 + 50.0 / (2.0 * 300.0)
        self.assertAlmostEqual(w, expected, places=4)

    def test_get_rxn_weight_invalid_reorg_tuple_length(self):
        """Reorg energy tuple of wrong length raises ValueError."""
        spc_r = _make_spc('R', '[H][H]', e0=0.0)
        spc_p = _make_spc('P', '[H][H]', e0=50.0)
        rxn = ARCReaction(reactants=['R'], products=['P'],
                          r_species=[spc_r], p_species=[spc_p])
        with self.assertRaises(ValueError):
            get_rxn_weight(rxn, reorg_energy=(100.0, 200.0, 300.0))

    def test_get_rxn_weight_zero_reorg_energy(self):
        """Zero reorganization energy raises ValueError."""
        spc_r = _make_spc('R', '[H][H]', e0=0.0)
        spc_p = _make_spc('P', '[H][H]', e0=50.0)
        rxn = ARCReaction(reactants=['R'], products=['P'],
                          r_species=[spc_r], p_species=[spc_p])
        with self.assertRaises(ValueError):
            get_rxn_weight(rxn, reorg_energy=0.0)

    # ------------------------------------------------------------------
    # get_weight_grid
    # ------------------------------------------------------------------
    def test_get_weight_grid_no_hammond(self):
        """Without Hammond, returns just the base grid (sorted, unique)."""
        spc_r = _make_spc('R', '[H][H]')
        spc_p = _make_spc('P', '[H][H]')
        spc_r.e0 = None
        spc_r.e_elect = None
        spc_p.e0 = None
        spc_p.e_elect = None
        rxn = ARCReaction(reactants=['R'], products=['P'],
                          r_species=[spc_r], p_species=[spc_p])
        grid = get_weight_grid(rxn, include_hammond=False)
        self.assertEqual(grid, sorted(BASE_WEIGHT_GRID))

    def test_get_weight_grid_with_hammond(self):
        """With energies available, Hammond points are added to the grid."""
        spc_r = _make_spc('R', '[H][H]', e0=0.0)
        spc_p = _make_spc('P', '[H][H]', e0=80.0)
        rxn = ARCReaction(reactants=['R'], products=['P'],
                          r_species=[spc_r], p_species=[spc_p])
        grid = get_weight_grid(rxn)
        self.assertTrue(len(grid) >= len(BASE_WEIGHT_GRID))
        self.assertEqual(grid, sorted(grid))
        for w in grid:
            self.assertGreaterEqual(w, 0.0)
            self.assertLessEqual(w, 1.0)

    def test_get_weight_grid_sorted_unique(self):
        """Grid values are sorted and unique."""
        spc_r = _make_spc('R', '[H][H]', e0=0.0)
        spc_p = _make_spc('P', '[H][H]', e0=0.0)
        rxn = ARCReaction(reactants=['R'], products=['P'],
                          r_species=[spc_r], p_species=[spc_p])
        grid = get_weight_grid(rxn)
        self.assertEqual(grid, sorted(set(grid)))

    def test_get_weight_grid_no_energies_still_has_base(self):
        """When energies unavailable, Hammond skipped but base grid returned."""
        spc_r = _make_spc('R', '[H][H]')
        spc_p = _make_spc('P', '[H][H]')
        spc_r.e0 = None
        spc_r.e_elect = None
        spc_p.e0 = None
        spc_p.e_elect = None
        rxn = ARCReaction(reactants=['R'], products=['P'],
                          r_species=[spc_r], p_species=[spc_p])
        grid = get_weight_grid(rxn, include_hammond=True)
        self.assertTrue(len(grid) >= len(BASE_WEIGHT_GRID))


if __name__ == '__main__':
    unittest.main()
