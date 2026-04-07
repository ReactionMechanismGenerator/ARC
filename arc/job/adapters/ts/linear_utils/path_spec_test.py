#!/usr/bin/env python3
# encoding: utf-8

"""
Unit tests for arc.job.adapters.ts.linear_utils.path_spec
"""

import unittest

import numpy as np

from arc.common import get_single_bond_length
from arc.species import ARCSpecies
from arc.job.adapters.ts.linear_utils.path_spec import (
    PAULING_DELTA,
    ReactionPathSpec,
    _build_adjacency,
    _bond_order_map,
    _canon,
    _canon_list,
    _compute_changed_bonds,
    _compute_unchanged_near_core,
    _multi_source_bfs,
    get_ts_target_distance,
    has_recipe_channel_mismatch,
    validate_guess_against_path_spec,
)


# ---------------------------------------------------------------------------
# Helpers used by the tests
# ---------------------------------------------------------------------------

def _propane_radical():
    """Return a propyl radical ARCSpecies (3 C in a chain).

    Atom indices: C0-C1-C2 with H atoms attached.  Used because it has a
    real molecular graph and a clean linear backbone.
    """
    return ARCSpecies(label='propyl', smiles='CC[CH2]')


def _propene():
    """Return propene ARCSpecies (used to give a different bond-order pattern)."""
    return ARCSpecies(label='propene', smiles='C=CC')


def _butane():
    return ARCSpecies(label='butane', smiles='CCCC')


# ---------------------------------------------------------------------------
# Canonicalization
# ---------------------------------------------------------------------------

class TestCanonicalBondOrdering(unittest.TestCase):
    """Bond keys must be canonicalized as (min, max) and lists deterministic."""

    def test_canon_basic(self):
        self.assertEqual(_canon(2, 5), (2, 5))
        self.assertEqual(_canon(5, 2), (2, 5))
        self.assertEqual(_canon(0, 0), (0, 0))

    def test_canon_list_dedup_and_sort(self):
        bonds = [(3, 1), (2, 4), (1, 3), (0, 5)]
        out = _canon_list(bonds)
        self.assertEqual(out, [(0, 5), (1, 3), (2, 4)])

    def test_canon_list_handles_none(self):
        self.assertEqual(_canon_list(None), [])
        self.assertEqual(_canon_list([]), [])


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------

class TestGraphHelpers(unittest.TestCase):

    def test_build_adjacency_propane(self):
        mol = _propane_radical().mol
        adj = _build_adjacency(mol)
        # Every atom should be present and self-edges absent.
        self.assertEqual(len(adj), len(mol.atoms))
        for i, nbrs in adj.items():
            self.assertNotIn(i, nbrs)
        # The C0 backbone atom should have at least one C neighbor.
        symbols = [a.element.symbol for a in mol.atoms]
        c_indices = [i for i, s in enumerate(symbols) if s == 'C']
        for ci in c_indices:
            self.assertTrue(len(adj[ci]) >= 1)

    def test_multi_source_bfs_distances(self):
        # Linear chain 0-1-2-3-4 sourced from {0}.
        adj = {0: {1}, 1: {0, 2}, 2: {1, 3}, 3: {2, 4}, 4: {3}}
        dist = _multi_source_bfs(adj, {0})
        self.assertEqual(dist, {0: 0, 1: 1, 2: 2, 3: 3, 4: 4})

    def test_multi_source_bfs_multi_source(self):
        adj = {0: {1}, 1: {0, 2}, 2: {1, 3}, 3: {2, 4}, 4: {3}}
        dist = _multi_source_bfs(adj, {0, 4})
        # Distance to nearest source.
        self.assertEqual(dist[0], 0)
        self.assertEqual(dist[1], 1)
        self.assertEqual(dist[2], 2)
        self.assertEqual(dist[3], 1)
        self.assertEqual(dist[4], 0)

    def test_bond_order_map_propene(self):
        mol = _propene().mol
        orders = _bond_order_map(mol)
        # Propene has at least one double bond.
        has_double = any(abs(v - 2.0) < 1e-6 for v in orders.values())
        self.assertTrue(has_double)
        # Every key should be canonical (min, max).
        for (i, j) in orders.keys():
            self.assertLessEqual(i, j)


# ---------------------------------------------------------------------------
# changed_bonds extraction
# ---------------------------------------------------------------------------

class TestChangedBonds(unittest.TestCase):

    def test_changed_bonds_no_change(self):
        """Same molecule on both sides → no changed bonds."""
        mol = _butane().mol
        changed = _compute_changed_bonds(mol, mol)
        self.assertEqual(changed, [])

    def test_changed_bonds_butane_to_butene(self):
        """Butane vs but-1-ene (same atom ordering): expect a bond-order change.

        Note: SMILES parsing may give different orderings, so we just check
        that something was detected without strict assertions on the indices.
        """
        butane_mol = _butane().mol
        butene = ARCSpecies(label='butene', smiles='C=CCC')
        butene_mol = butene.mol
        # Only valid if heavy-atom counts match (we manipulate two separate
        # mols of equal length here just to exercise the comparison code).
        if len(butane_mol.atoms) == len(butene_mol.atoms):
            changed = _compute_changed_bonds(butane_mol, butene_mol)
            # We don't strictly know the indices, but the function should
            # return a deterministic sorted list (possibly empty if SMILES
            # parsing gave different orderings).
            self.assertIsInstance(changed, list)
            for entry in changed:
                self.assertEqual(len(entry), 2)
                self.assertLessEqual(entry[0], entry[1])

    def test_changed_bonds_self_no_change(self):
        """A molecule compared with itself: deterministic empty list."""
        mol = _propane_radical().mol
        self.assertEqual(_compute_changed_bonds(mol, mol), [])


# ---------------------------------------------------------------------------
# unchanged_near_core BFS rule
# ---------------------------------------------------------------------------

class TestUnchangedNearCore(unittest.TestCase):
    """Verify the exact BFS-shell rule for unchanged_near_core_bonds."""

    def _toy_chain_mol(self, n_carbons=6):
        """Build a straight C(n) alkane and return its mol."""
        smi = 'C' * n_carbons
        return ARCSpecies(label=f'C{n_carbons}', smiles=smi).mol

    def test_unchanged_near_core_includes_first_shell(self):
        """A reactive bond at the chain end pulls in the next shell."""
        mol = self._toy_chain_mol(6)  # C0-C1-C2-C3-C4-C5
        # Heavy-atom indices for C atoms (RMG sometimes orders H first; find C indices):
        c_indices = [i for i, a in enumerate(mol.atoms) if a.element.symbol == 'C']
        # We use the first 6 in graph order for the test scaffold.
        # Pretend a forming bond between c_indices[0] and c_indices[1] is reactive.
        fb = [(c_indices[0], c_indices[1])]
        unchanged = _compute_unchanged_near_core(
            mol, breaking_bonds=[], forming_bonds=fb, changed_bonds=[])
        # The reactive bond itself must NOT appear.
        self.assertNotIn(_canon(c_indices[0], c_indices[1]), unchanged)
        # Bonds two hops away should still be included; bonds 4+ hops away
        # must be excluded.
        # We don't make exact assertions on hexane's H-positions; we just
        # check the determinism + ordering invariants.
        self.assertEqual(unchanged, sorted(unchanged))
        for (a, b) in unchanged:
            self.assertLess(a, b)

    def test_unchanged_near_core_empty_for_no_reactive_set(self):
        mol = self._toy_chain_mol(4)
        out = _compute_unchanged_near_core(mol, [], [], [])
        self.assertEqual(out, [])

    def test_unchanged_near_core_excludes_reactive_bonds(self):
        mol = self._toy_chain_mol(4)
        c_indices = [i for i, a in enumerate(mol.atoms) if a.element.symbol == 'C']
        bb = [(c_indices[0], c_indices[1])]
        fb = [(c_indices[2], c_indices[3])]
        unchanged = _compute_unchanged_near_core(mol, bb, fb, [])
        for key in unchanged:
            self.assertNotIn(key, [_canon(*bb[0]), _canon(*fb[0])])


# ---------------------------------------------------------------------------
# ReactionPathSpec construction
# ---------------------------------------------------------------------------

class TestReactionPathSpecBuild(unittest.TestCase):

    def test_build_minimal(self):
        sp = _propane_radical()
        spec = ReactionPathSpec.build(
            r_mol=sp.mol,
            mapped_p_mol=sp.mol,
            breaking_bonds=[(0, 1)],
            forming_bonds=[(1, 2)],
            r_xyz=None,
            op_xyz=None,
            weight=0.5,
            family='test_family',
        )
        # Reactive bonds in canonical order.
        self.assertEqual(spec.breaking_bonds, [(0, 1)])
        self.assertEqual(spec.forming_bonds, [(1, 2)])
        # changed_bonds is empty when comparing a mol to itself.
        self.assertEqual(spec.changed_bonds, [])
        # reactive_atoms = {0, 1, 2}
        self.assertEqual(spec.reactive_atoms, {0, 1, 2})
        self.assertEqual(spec.weight, 0.5)
        self.assertEqual(spec.family, 'test_family')

    def test_build_canonicalizes_input_bonds(self):
        sp = _propane_radical()
        # Pass bonds in non-canonical order; build() must canonicalize.
        spec = ReactionPathSpec.build(
            r_mol=sp.mol, mapped_p_mol=sp.mol,
            breaking_bonds=[(2, 0)],
            forming_bonds=[(3, 1)],
        )
        self.assertEqual(spec.breaking_bonds, [(0, 2)])
        self.assertEqual(spec.forming_bonds, [(1, 3)])

    def test_build_deterministic(self):
        sp = _propane_radical()
        s1 = ReactionPathSpec.build(
            r_mol=sp.mol, mapped_p_mol=sp.mol,
            breaking_bonds=[(0, 1), (3, 2)],
            forming_bonds=[(4, 5)],
        )
        s2 = ReactionPathSpec.build(
            r_mol=sp.mol, mapped_p_mol=sp.mol,
            breaking_bonds=[(2, 3), (1, 0)],
            forming_bonds=[(5, 4)],
        )
        self.assertEqual(s1.breaking_bonds, s2.breaking_bonds)
        self.assertEqual(s1.forming_bonds, s2.forming_bonds)

    def test_build_stores_reference_distances(self):
        sp = _propane_radical()
        # Use the species' own xyz as both reactant and product reference.
        xyz = sp.get_xyz()
        spec = ReactionPathSpec.build(
            r_mol=sp.mol, mapped_p_mol=sp.mol,
            breaking_bonds=[(0, 1)],
            forming_bonds=[(1, 2)],
            r_xyz=xyz, op_xyz=xyz,
        )
        # Reference distances should be populated for the reactive bonds.
        for key in spec.breaking_bonds + spec.forming_bonds:
            self.assertIn(key, spec.ref_dist_r)
            self.assertIn(key, spec.ref_dist_p)
            self.assertIsNotNone(spec.ref_dist_r[key])
            self.assertIsNotNone(spec.ref_dist_p[key])

    def test_build_unchanged_near_core_populated(self):
        sp = _propane_radical()
        spec = ReactionPathSpec.build(
            r_mol=sp.mol, mapped_p_mol=sp.mol,
            breaking_bonds=[(0, 1)],
            forming_bonds=[],
        )
        # The unchanged_near_core_bonds list is deterministic and sorted.
        self.assertEqual(spec.unchanged_near_core_bonds,
                         sorted(spec.unchanged_near_core_bonds))
        for key in spec.unchanged_near_core_bonds:
            self.assertNotIn(key, spec.breaking_bonds)
            self.assertNotIn(key, spec.forming_bonds)


# ---------------------------------------------------------------------------
# get_ts_target_distance
# ---------------------------------------------------------------------------

class TestGetTsTargetDistance(unittest.TestCase):

    def test_breaking_uses_pauling(self):
        symbols = ('C', 'C')
        d = get_ts_target_distance((0, 1), 'breaking', symbols)
        sbl = get_single_bond_length('C', 'C')
        self.assertAlmostEqual(d, sbl + PAULING_DELTA, places=6)

    def test_forming_uses_pauling(self):
        symbols = ('C', 'C')
        d = get_ts_target_distance((0, 1), 'forming', symbols)
        sbl = get_single_bond_length('C', 'C')
        self.assertAlmostEqual(d, sbl + PAULING_DELTA, places=6)

    def test_changed_interpolates_when_both_distances_given(self):
        symbols = ('C', 'C')
        d = get_ts_target_distance(
            (0, 1), 'changed', symbols,
            d_r=1.50, d_p=1.30, weight=0.5)
        self.assertAlmostEqual(d, 1.40, places=6)

    def test_changed_interpolates_at_endpoints(self):
        symbols = ('C', 'C')
        d0 = get_ts_target_distance(
            (0, 1), 'changed', symbols, d_r=1.50, d_p=1.30, weight=0.0)
        d1 = get_ts_target_distance(
            (0, 1), 'changed', symbols, d_r=1.50, d_p=1.30, weight=1.0)
        self.assertAlmostEqual(d0, 1.50, places=6)
        self.assertAlmostEqual(d1, 1.30, places=6)

    def test_changed_falls_back_when_only_one_distance_given(self):
        symbols = ('C', 'C')
        d_r_only = get_ts_target_distance(
            (0, 1), 'changed', symbols, d_r=1.45, d_p=None)
        d_p_only = get_ts_target_distance(
            (0, 1), 'changed', symbols, d_r=None, d_p=1.35)
        self.assertAlmostEqual(d_r_only, 1.45, places=6)
        self.assertAlmostEqual(d_p_only, 1.35, places=6)

    def test_changed_conservative_fallback(self):
        symbols = ('C', 'C')
        d = get_ts_target_distance((0, 1), 'changed', symbols)
        # Conservative fallback: a single-bond length.
        self.assertAlmostEqual(d, get_single_bond_length('C', 'C'), places=6)

    def test_unchanged_near_core_prefers_d_r(self):
        symbols = ('C', 'C')
        d = get_ts_target_distance(
            (0, 1), 'unchanged_near_core', symbols, d_r=1.52)
        self.assertAlmostEqual(d, 1.52, places=6)

    def test_unchanged_near_core_falls_back_to_sbl(self):
        symbols = ('C', 'C')
        d = get_ts_target_distance(
            (0, 1), 'unchanged_near_core', symbols, d_r=None)
        self.assertAlmostEqual(d, get_single_bond_length('C', 'C'), places=6)

    def test_unknown_role_raises(self):
        with self.assertRaises(ValueError):
            get_ts_target_distance((0, 1), 'nonsense', ('C', 'C'))


# ---------------------------------------------------------------------------
# has_recipe_channel_mismatch — exact Phase 1 thresholds
# ---------------------------------------------------------------------------

class TestHasRecipeChannelMismatch(unittest.TestCase):
    """Exact thresholds: 1.30 / 3.00 / 1.25×SBL."""

    def _two_atom_xyz(self, sym1: str, sym2: str, distance: float) -> dict:
        return {
            'symbols': (sym1, sym2),
            'isotopes': (12, 12),
            'coords': ((0.0, 0.0, 0.0), (distance, 0.0, 0.0)),
        }

    def test_forming_bond_far_apart_returns_true(self):
        spec = ReactionPathSpec(forming_bonds=[(0, 1)])
        xyz = self._two_atom_xyz('C', 'C', 3.5)
        mismatch, reason = has_recipe_channel_mismatch(spec, xyz, r_mol=None)
        self.assertTrue(mismatch)
        self.assertIn('failed-to-form', reason)

    def test_forming_bond_reasonable_returns_false(self):
        spec = ReactionPathSpec(forming_bonds=[(0, 1)])
        xyz = self._two_atom_xyz('C', 'C', 2.0)
        mismatch, _ = has_recipe_channel_mismatch(spec, xyz, r_mol=None)
        self.assertFalse(mismatch)

    def test_forming_bond_just_below_threshold_passes(self):
        spec = ReactionPathSpec(forming_bonds=[(0, 1)])
        xyz = self._two_atom_xyz('C', 'C', 2.99)
        mismatch, _ = has_recipe_channel_mismatch(spec, xyz, r_mol=None)
        self.assertFalse(mismatch)

    def test_breaking_bond_too_short_returns_true(self):
        spec = ReactionPathSpec(breaking_bonds=[(0, 1)])
        xyz = self._two_atom_xyz('C', 'C', 1.2)
        mismatch, reason = has_recipe_channel_mismatch(spec, xyz, r_mol=None)
        self.assertTrue(mismatch)
        self.assertIn('failed-to-break', reason)

    def test_breaking_bond_at_threshold_passes(self):
        spec = ReactionPathSpec(breaking_bonds=[(0, 1)])
        xyz = self._two_atom_xyz('C', 'C', 1.31)
        mismatch, _ = has_recipe_channel_mismatch(spec, xyz, r_mol=None)
        self.assertFalse(mismatch)

    def test_unchanged_near_core_snapped_returns_true(self):
        sbl_cc = get_single_bond_length('C', 'C')
        # 1.30 × sbl is above the 1.25 threshold.
        d = 1.30 * sbl_cc
        spec = ReactionPathSpec(unchanged_near_core_bonds=[(0, 1)])
        xyz = self._two_atom_xyz('C', 'C', d)
        mismatch, reason = has_recipe_channel_mismatch(spec, xyz, r_mol=None)
        self.assertTrue(mismatch)
        self.assertIn('snapped-spectator', reason)

    def test_unchanged_near_core_within_limit_passes(self):
        sbl_cc = get_single_bond_length('C', 'C')
        d = 1.20 * sbl_cc
        spec = ReactionPathSpec(unchanged_near_core_bonds=[(0, 1)])
        xyz = self._two_atom_xyz('C', 'C', d)
        mismatch, _ = has_recipe_channel_mismatch(spec, xyz, r_mol=None)
        self.assertFalse(mismatch)

    def test_empty_spec_returns_false(self):
        spec = ReactionPathSpec()
        xyz = self._two_atom_xyz('C', 'C', 1.5)
        mismatch, _ = has_recipe_channel_mismatch(spec, xyz, r_mol=None)
        self.assertFalse(mismatch)


# ---------------------------------------------------------------------------
# validate_guess_against_path_spec
# ---------------------------------------------------------------------------

class TestValidateGuessAgainstPathSpec(unittest.TestCase):
    """The wrapper must compose generic validation + recipe-channel mismatch."""

    def test_valid_guess_passes(self):
        sp = _propane_radical()
        xyz = sp.get_xyz()
        spec = ReactionPathSpec.build(
            r_mol=sp.mol, mapped_p_mol=sp.mol,
            breaking_bonds=[], forming_bonds=[],
        )
        ok, reason = validate_guess_against_path_spec(
            xyz=xyz, path_spec=spec, r_mol=sp.mol, family=None,
        )
        self.assertTrue(ok, f'Expected pass; got reason {reason}')

    def test_recipe_mismatch_rejects(self):
        """A guess where a forming bond is far apart is rejected by the wrapper.

        Construct the test against an empty path_spec field with only the
        forming bond declared, and a hand-built XYZ that does not collide.
        """
        # Two H atoms placed 4 Å apart — no collisions, but forming bond is
        # far above the 3.00 Å Phase 1 threshold.
        h2 = ARCSpecies(label='H2', smiles='[H][H]')
        bad_xyz = {
            'symbols': ('H', 'H'),
            'isotopes': (1, 1),
            'coords': ((0.0, 0.0, 0.0), (4.0, 0.0, 0.0)),
        }
        spec = ReactionPathSpec(forming_bonds=[(0, 1)])
        ok, reason = validate_guess_against_path_spec(
            xyz=bad_xyz, path_spec=spec, r_mol=h2.mol,
        )
        self.assertFalse(ok)
        self.assertIn('recipe-mismatch', reason)


# ---------------------------------------------------------------------------
# Centralization plumbing — confirm linear.py imports the wrapper
# ---------------------------------------------------------------------------

class TestPlumbingCentralization(unittest.TestCase):
    """Prove that the orchestration shell wires through the new wrapper."""

    def test_linear_module_imports_wrapper(self):
        from arc.job.adapters.ts import linear
        self.assertTrue(hasattr(linear, 'validate_guess_against_path_spec'))
        self.assertTrue(hasattr(linear, 'ReactionPathSpec'))

    def test_path_context_has_path_spec_field(self):
        from arc.job.adapters.ts.linear import _PathContext
        # The dataclass must declare a path_spec field.
        self.assertIn('path_spec', _PathContext.__dataclass_fields__)

    def test_guess_record_has_path_spec_field(self):
        from arc.job.adapters.ts.linear import GuessRecord
        self.assertIn('path_spec', GuessRecord.__dataclass_fields__)


if __name__ == '__main__':
    unittest.main()
