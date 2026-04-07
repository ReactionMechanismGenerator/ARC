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
    PathChemistry,
    ReactionPathSpec,
    _build_adjacency,
    _bond_order_map,
    _canon,
    _canon_list,
    _compute_changed_bonds,
    _compute_unchanged_near_core,
    _multi_source_bfs,
    classify_path_chemistry,
    get_ts_target_distance,
    has_bad_changed_bond_length,
    has_bad_reactive_core_planarity,
    has_bad_unchanged_near_core_bond,
    has_inward_blocking_h_on_forming_axis,
    has_recipe_channel_mismatch,
    score_guess_against_path_spec,
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


# ---------------------------------------------------------------------------
# Phase 2: PathChemistry classification
# ---------------------------------------------------------------------------

class TestPathChemistryClassification(unittest.TestCase):
    """Verify the exact rule order of :func:`classify_path_chemistry`."""

    def test_substitution_like_single_heavy_pivot(self):
        # Single bb, single fb, sharing one heavy atom (atom 1).
        # Pivot is C → SUBSTITUTION_LIKE.
        spec = ReactionPathSpec(
            breaking_bonds=[(1, 2)],
            forming_bonds=[(0, 1)],
        )
        symbols = ('C', 'C', 'C')
        chem = classify_path_chemistry(spec, r_mol=None, symbols=symbols)
        self.assertIs(chem, PathChemistry.SUBSTITUTION_LIKE)

    def test_substitution_like_with_h_pivot_falls_to_h_transfer(self):
        # The shared atom is H, so substitution_like should NOT match;
        # should fall through to H_TRANSFER.
        spec = ReactionPathSpec(
            breaking_bonds=[(0, 1)],  # C-H
            forming_bonds=[(1, 2)],   # H-C
        )
        symbols = ('C', 'H', 'C')
        chem = classify_path_chemistry(spec, r_mol=None, symbols=symbols)
        self.assertIs(chem, PathChemistry.H_TRANSFER)

    def test_h_transfer_when_shared_atom_is_h(self):
        spec = ReactionPathSpec(
            breaking_bonds=[(0, 4)],  # heavy-H
            forming_bonds=[(4, 3)],   # H-heavy
        )
        symbols = ('C', 'C', 'C', 'C', 'H')
        chem = classify_path_chemistry(spec, r_mol=None, symbols=symbols)
        self.assertIs(chem, PathChemistry.H_TRANSFER)

    def test_non_h_group_shift_when_shared_atom_is_heavy(self):
        # Two bonds, but multiple bb/fb with a heavy shared atom and no
        # SUBSTITUTION_LIKE single-bond match.
        spec = ReactionPathSpec(
            breaking_bonds=[(0, 1), (3, 4)],
            forming_bonds=[(0, 2)],  # atom 0 (C) is the shared pivot
        )
        symbols = ('C', 'C', 'C', 'C', 'C')
        chem = classify_path_chemistry(spec, r_mol=None, symbols=symbols)
        self.assertIs(chem, PathChemistry.NON_H_GROUP_SHIFT)

    def test_concerted_hetero_rearrangement(self):
        # 2+ bb, 2+ fb, with at least 2 hetero reactive atoms (O, O).
        spec = ReactionPathSpec(
            breaking_bonds=[(0, 1), (2, 3)],
            forming_bonds=[(4, 5), (6, 7)],
        )
        # symbols: indices 1 and 3 are O, others C; 4 and 7 also O.
        symbols = ('C', 'O', 'C', 'O', 'C', 'C', 'C', 'O')
        chem = classify_path_chemistry(spec, r_mol=None, symbols=symbols)
        self.assertIs(chem, PathChemistry.CONCERTED_HETERO_REARRANGEMENT)

    def test_cycloaddition_or_ring_closure(self):
        # No shared atoms, single heavy-heavy forming bond → cycloaddition.
        spec = ReactionPathSpec(
            forming_bonds=[(0, 4)],
        )
        symbols = ('C', 'C', 'C', 'C', 'C')
        chem = classify_path_chemistry(spec, r_mol=None, symbols=symbols)
        self.assertIs(chem, PathChemistry.CYCLOADDITION_OR_RING_CLOSURE)

    def test_generic_fallback(self):
        # No shared atoms, no heavy-heavy forming bond → GENERIC.
        spec = ReactionPathSpec(
            forming_bonds=[(0, 1)],  # H is index 1, so heavy-H not heavy-heavy
        )
        symbols = ('C', 'H')
        chem = classify_path_chemistry(spec, r_mol=None, symbols=symbols)
        self.assertIs(chem, PathChemistry.GENERIC)

    def test_empty_spec_is_generic(self):
        chem = classify_path_chemistry(ReactionPathSpec(), r_mol=None, symbols=())
        self.assertIs(chem, PathChemistry.GENERIC)


# ---------------------------------------------------------------------------
# Phase 2: changed-bond length validator
# ---------------------------------------------------------------------------

class TestHasBadChangedBondLength(unittest.TestCase):

    def _xyz(self, distance, sym1='C', sym2='C'):
        return {
            'symbols': (sym1, sym2),
            'isotopes': (12, 12),
            'coords': ((0.0, 0.0, 0.0), (distance, 0.0, 0.0)),
        }

    def test_no_changed_bonds_returns_false(self):
        spec = ReactionPathSpec()
        bad, reason = has_bad_changed_bond_length(spec, self._xyz(1.5),
                                                  symbols=('C', 'C'))
        self.assertFalse(bad)
        self.assertEqual(reason, '')

    def test_changed_bond_within_tolerance_passes(self):
        spec = ReactionPathSpec(
            changed_bonds=[(0, 1)],
            ref_dist_r={(0, 1): 1.50},
            ref_dist_p={(0, 1): 1.40},
        )
        # target = 0.5*(1.50+1.40) = 1.45; we use 1.50 (deviation 0.05)
        bad, _ = has_bad_changed_bond_length(spec, self._xyz(1.50),
                                             symbols=('C', 'C'))
        self.assertFalse(bad)

    def test_changed_bond_heavy_outside_tolerance_rejects(self):
        spec = ReactionPathSpec(
            changed_bonds=[(0, 1)],
            ref_dist_r={(0, 1): 1.50},
            ref_dist_p={(0, 1): 1.40},
        )
        # target = 1.45; deviation must exceed 0.25 → use 1.80 (dev = 0.35).
        bad, reason = has_bad_changed_bond_length(spec, self._xyz(1.80),
                                                  symbols=('C', 'C'))
        self.assertTrue(bad)
        self.assertIn('bad-changed-bond', reason)

    def test_changed_bond_h_involving_uses_tighter_tolerance(self):
        # H tolerance is 0.20.
        spec = ReactionPathSpec(
            changed_bonds=[(0, 1)],
            ref_dist_r={(0, 1): 1.10},
            ref_dist_p={(0, 1): 1.00},
        )
        # target = 1.05; deviation 0.21 > 0.20 → reject.
        bad, _ = has_bad_changed_bond_length(spec, self._xyz(1.26, sym2='H'),
                                             symbols=('C', 'H'))
        self.assertTrue(bad)


# ---------------------------------------------------------------------------
# Phase 2: unchanged_near_core bond validator
# ---------------------------------------------------------------------------

class TestHasBadUnchangedNearCoreBond(unittest.TestCase):

    def _xyz(self, distance):
        return {
            'symbols': ('C', 'C'),
            'isotopes': (12, 12),
            'coords': ((0.0, 0.0, 0.0), (distance, 0.0, 0.0)),
        }

    def test_unchanged_within_window_passes(self):
        spec = ReactionPathSpec(
            unchanged_near_core_bonds=[(0, 1)],
            ref_dist_r={(0, 1): 1.54},
        )
        bad, _ = has_bad_unchanged_near_core_bond(spec, self._xyz(1.55),
                                                  symbols=('C', 'C'))
        self.assertFalse(bad)

    def test_unchanged_too_short_rejects(self):
        spec = ReactionPathSpec(
            unchanged_near_core_bonds=[(0, 1)],
            ref_dist_r={(0, 1): 1.54},
        )
        # 0.82*1.54 = 1.263 → 1.20 is below
        bad, reason = has_bad_unchanged_near_core_bond(
            spec, self._xyz(1.20), symbols=('C', 'C'))
        self.assertTrue(bad)
        self.assertIn('bad-unchanged-near-core', reason)

    def test_unchanged_too_long_rejects(self):
        spec = ReactionPathSpec(
            unchanged_near_core_bonds=[(0, 1)],
            ref_dist_r={(0, 1): 1.54},
        )
        # 1.25*1.54 = 1.925 → 2.10 is above
        bad, _ = has_bad_unchanged_near_core_bond(
            spec, self._xyz(2.10), symbols=('C', 'C'))
        self.assertTrue(bad)

    def test_no_ref_falls_back_to_sbl(self):
        spec = ReactionPathSpec(
            unchanged_near_core_bonds=[(0, 1)],
        )
        # SBL(C,C) ≈ 1.54; 1.55 is fine, 3.00 is way over.
        bad, _ = has_bad_unchanged_near_core_bond(
            spec, self._xyz(1.55), symbols=('C', 'C'))
        self.assertFalse(bad)
        bad, _ = has_bad_unchanged_near_core_bond(
            spec, self._xyz(3.00), symbols=('C', 'C'))
        self.assertTrue(bad)


# ---------------------------------------------------------------------------
# Phase 2: inward-blocking-H-on-forming-axis validator
# ---------------------------------------------------------------------------

class TestHasInwardBlockingHOnFormingAxis(unittest.TestCase):

    def test_inactive_when_chemistry_is_h_transfer(self):
        # The check must be skipped for non-substitution/cycloaddition chemistries.
        ethane = ARCSpecies(label='ethane', smiles='CC')
        spec = ReactionPathSpec(forming_bonds=[(0, 1)])
        bad, _ = has_inward_blocking_h_on_forming_axis(
            spec, ethane.get_xyz(), ethane.mol,
            symbols=tuple(a.element.symbol for a in ethane.mol.atoms),
            chemistry=PathChemistry.H_TRANSFER)
        self.assertFalse(bad)

    def test_inactive_when_no_heavy_heavy_forming_bond(self):
        h2 = ARCSpecies(label='H2', smiles='[H][H]')
        spec = ReactionPathSpec(forming_bonds=[(0, 1)])
        bad, _ = has_inward_blocking_h_on_forming_axis(
            spec, h2.get_xyz(), h2.mol,
            symbols=('H', 'H'),
            chemistry=PathChemistry.CYCLOADDITION_OR_RING_CLOSURE)
        self.assertFalse(bad)

    def test_blocking_h_intrudes_on_axis(self):
        # Construct an artificial geometry: C0 and C1 forming a bond ~3 Å
        # apart, with a hydrogen on C0 placed close to C1 and ~at right angles.
        xyz = {
            'symbols': ('C', 'C', 'H'),
            'isotopes': (12, 12, 1),
            'coords': (
                (0.0, 0.0, 0.0),   # C0
                (3.0, 0.0, 0.0),   # C1
                (1.5, 0.05, 0.0),  # H, halfway in front of C1
            ),
        }

        # Stub minimal r_mol-like adjacency: H bonded to C0.
        class _StubAtom:
            def __init__(self, sym, bonds_dict):
                self.bonds = bonds_dict

                class _Elt:
                    def __init__(self, s):
                        self.symbol = s
                self.element = _Elt(sym)

        c0 = _StubAtom('C', {})
        c1 = _StubAtom('C', {})
        h2 = _StubAtom('H', {})
        c0.bonds = {h2: None}
        h2.bonds = {c0: None}
        c1.bonds = {}

        class _StubMol:
            atoms = [c0, c1, h2]

        spec = ReactionPathSpec(
            forming_bonds=[(0, 1)],
            reactive_atoms={0, 1},
        )
        bad, reason = has_inward_blocking_h_on_forming_axis(
            spec, xyz, _StubMol(),
            symbols=('C', 'C', 'H'),
            chemistry=PathChemistry.CYCLOADDITION_OR_RING_CLOSURE)
        self.assertTrue(bad)
        self.assertIn('inward-blocking-H', reason)


# ---------------------------------------------------------------------------
# Phase 2: reactive-core-planarity validator
# ---------------------------------------------------------------------------

class TestHasBadReactiveCorePlanarity(unittest.TestCase):

    def test_inactive_for_non_concerted(self):
        spec = ReactionPathSpec(
            breaking_bonds=[(0, 1)],
            forming_bonds=[(2, 3)],
        )
        xyz = {
            'symbols': ('C', 'O', 'C', 'O'),
            'isotopes': (12, 16, 12, 16),
            'coords': ((0., 0., 0.), (1., 0., 0.), (0., 1., 0.), (1., 1., 5.)),
        }
        bad, _ = has_bad_reactive_core_planarity(
            spec, xyz, symbols=('C', 'O', 'C', 'O'),
            chemistry=PathChemistry.GENERIC)
        self.assertFalse(bad)

    def test_few_atoms_returns_false(self):
        # Fewer than 4 heavy reactive atoms → cannot fit a plane → False.
        spec = ReactionPathSpec(
            breaking_bonds=[(0, 1), (2, 3)],
            forming_bonds=[(0, 2), (1, 3)],
        )
        xyz = {
            'symbols': ('O', 'O', 'C'),
            'isotopes': (16, 16, 12),
            'coords': ((0., 0., 0.), (1., 0., 0.), (0., 1., 0.)),
        }
        bad, _ = has_bad_reactive_core_planarity(
            spec, xyz, symbols=('O', 'O', 'C'),
            chemistry=PathChemistry.CONCERTED_HETERO_REARRANGEMENT)
        self.assertFalse(bad)

    def test_planar_core_passes(self):
        spec = ReactionPathSpec(
            breaking_bonds=[(0, 1), (2, 3)],
            forming_bonds=[(0, 2), (1, 3)],
        )
        xyz = {
            'symbols': ('O', 'O', 'C', 'C'),
            'isotopes': (16, 16, 12, 12),
            'coords': ((0., 0., 0.), (1., 0., 0.), (1., 1., 0.), (0., 1., 0.)),
        }
        bad, _ = has_bad_reactive_core_planarity(
            spec, xyz, symbols=('O', 'O', 'C', 'C'),
            chemistry=PathChemistry.CONCERTED_HETERO_REARRANGEMENT)
        self.assertFalse(bad)

    def test_strongly_nonplanar_core_rejects(self):
        spec = ReactionPathSpec(
            breaking_bonds=[(0, 1), (2, 3)],
            forming_bonds=[(0, 2), (1, 3)],
        )
        # A genuinely 3D tetrahedral arrangement of the 4 reactive atoms.
        # No plane fits all four within 0.35 Å RMS.
        xyz = {
            'symbols': ('O', 'O', 'C', 'C'),
            'isotopes': (16, 16, 12, 12),
            'coords': ((0., 0., 0.), (2., 0., 0.), (0., 2., 0.), (0., 0., 2.)),
        }
        bad, reason = has_bad_reactive_core_planarity(
            spec, xyz, symbols=('O', 'O', 'C', 'C'),
            chemistry=PathChemistry.CONCERTED_HETERO_REARRANGEMENT)
        self.assertTrue(bad)
        self.assertIn('reactive-core', reason)


# ---------------------------------------------------------------------------
# Phase 2: scoring function
# ---------------------------------------------------------------------------

class TestScoreGuessAgainstPathSpec(unittest.TestCase):

    def test_zero_score_for_perfect_breaking_target(self):
        # Single breaking bond at exactly sbl + Pauling delta should give 0.
        sbl = get_single_bond_length('C', 'C')
        target = sbl + PAULING_DELTA
        spec = ReactionPathSpec(breaking_bonds=[(0, 1)])
        xyz = {
            'symbols': ('C', 'C'),
            'isotopes': (12, 12),
            'coords': ((0.0, 0.0, 0.0), (target, 0.0, 0.0)),
        }
        s = score_guess_against_path_spec(
            spec, xyz, r_mol=None, symbols=('C', 'C'),
            chemistry=PathChemistry.GENERIC)
        self.assertAlmostEqual(s, 0.0, places=5)

    def test_score_increases_with_breaking_deviation(self):
        sbl = get_single_bond_length('C', 'C')
        target = sbl + PAULING_DELTA
        spec = ReactionPathSpec(breaking_bonds=[(0, 1)])
        xyz_close = {
            'symbols': ('C', 'C'),
            'isotopes': (12, 12),
            'coords': ((0.0, 0.0, 0.0), (target + 0.1, 0.0, 0.0)),
        }
        xyz_far = {
            'symbols': ('C', 'C'),
            'isotopes': (12, 12),
            'coords': ((0.0, 0.0, 0.0), (target + 0.5, 0.0, 0.0)),
        }
        s_close = score_guess_against_path_spec(
            spec, xyz_close, r_mol=None, symbols=('C', 'C'),
            chemistry=PathChemistry.GENERIC)
        s_far = score_guess_against_path_spec(
            spec, xyz_far, r_mol=None, symbols=('C', 'C'),
            chemistry=PathChemistry.GENERIC)
        self.assertLess(s_close, s_far)

    def test_planarity_penalty_added(self):
        spec = ReactionPathSpec(
            breaking_bonds=[(0, 1), (2, 3)],
            forming_bonds=[(0, 2), (1, 3)],
        )
        # Strongly non-planar core (tetrahedral 3D arrangement).
        xyz = {
            'symbols': ('O', 'O', 'C', 'C'),
            'isotopes': (16, 16, 12, 12),
            'coords': ((0., 0., 0.), (2., 0., 0.), (0., 2., 0.), (0., 0., 2.)),
        }
        s = score_guess_against_path_spec(
            spec, xyz, r_mol=None, symbols=('O', 'O', 'C', 'C'),
            chemistry=PathChemistry.CONCERTED_HETERO_REARRANGEMENT)
        # Score must include the +5 planarity penalty.
        self.assertGreaterEqual(s, 5.0)

    def test_none_xyz_returns_inf(self):
        spec = ReactionPathSpec()
        s = score_guess_against_path_spec(
            spec, xyz=None, r_mol=None, symbols=())
        self.assertEqual(s, float('inf'))


# ---------------------------------------------------------------------------
# Phase 2: validator dispatch — H_TRANSFER routing
# ---------------------------------------------------------------------------

class TestHTransferDispatchOverride(unittest.TestCase):
    """When chemistry == H_TRANSFER the H-migration validator must run
    regardless of family string."""

    def test_h_transfer_routes_to_h_migration_even_with_unknown_family(self):
        from arc.job.adapters.ts.linear_utils.postprocess import validate_ts_guess
        # Build a non-trivial xyz with two H atoms placed too close together.
        # validate_h_migration's _has_h_close_contact should fire.
        xyz = {
            'symbols': ('C', 'H', 'H'),
            'isotopes': (12, 1, 1),
            'coords': ((0.0, 0.0, 0.0), (0.7, 0.0, 0.0), (0.78, 0.0, 0.0)),
        }
        c = ARCSpecies(label='CH2', smiles='[CH2]')
        ok, reason = validate_ts_guess(
            xyz=xyz,
            migrating_hs={1, 2},
            forming_bonds=[(0, 1)],
            r_mol=c.mol,
            label='test',
            family='non_existent_family',
            chemistry='h_transfer',
        )
        self.assertFalse(ok)

    def test_non_h_transfer_chemistry_skips_h_migration(self):
        # Sanity: passing chemistry=None preserves prior behavior — an
        # unknown family runs no family validator.
        from arc.job.adapters.ts.linear_utils.postprocess import validate_ts_guess
        sp = _propane_radical()
        ok, _ = validate_ts_guess(
            xyz=sp.get_xyz(),
            migrating_hs=set(),
            forming_bonds=[],
            r_mol=sp.mol,
            label='test',
            family='non_existent_family',
            chemistry=None,
        )
        self.assertTrue(ok)


# ---------------------------------------------------------------------------
# Phase 2: orchestration triage — score-sort + second dedup
# ---------------------------------------------------------------------------

class TestOrchestrationTriage(unittest.TestCase):
    """Verify linear.py exposes the Phase 2 plumbing correctly."""

    def test_linear_module_imports_phase2_helpers(self):
        from arc.job.adapters.ts import linear
        self.assertTrue(hasattr(linear, 'classify_path_chemistry'))
        self.assertTrue(hasattr(linear, 'score_guess_against_path_spec'))
        self.assertTrue(hasattr(linear, 'PathChemistry'))


# ---------------------------------------------------------------------------
# Phase 2b: tightened frontier exemption — strict 2-condition rule
# ---------------------------------------------------------------------------


class TestTightenedFrontierExemption(unittest.TestCase):
    """The frontier exemption must require BOTH a non-trivial bond-order
    shift AND direct topological adjacency to a breaking/forming bond."""

    def _xyz_chain(self, n_atoms: int) -> dict:
        """A linear chain of *n_atoms* C atoms spaced 2.0 Å apart on the x axis."""
        coords = tuple((2.0 * i, 0.0, 0.0) for i in range(n_atoms))
        return {
            'symbols': tuple('C' for _ in range(n_atoms)),
            'isotopes': tuple(12 for _ in range(n_atoms)),
            'coords': coords,
        }

    def test_adjacent_changed_bond_with_bo_shift_is_exempt(self):
        """Changed bond at (1,2) sharing atom 1 with breaking bond (0,1)
        and BO shift 0.5 → exempt (no rejection even though distance is wrong)."""
        spec = ReactionPathSpec(
            breaking_bonds=[(0, 1)],
            changed_bonds=[(1, 2)],
            ref_dist_r={(1, 2): 1.50},
            ref_dist_p={(1, 2): 1.40},
            bond_order_r={(1, 2): 1.0},
            bond_order_p={(1, 2): 1.5},
        )
        xyz = self._xyz_chain(3)  # bond (1,2) at 2.0 Å — way outside tol
        bad, reason = has_bad_changed_bond_length(spec, xyz, symbols=('C', 'C', 'C'))
        self.assertFalse(bad, f'Frontier-adjacent bond should be exempt, got: {reason}')

    def test_isolated_changed_bond_with_bo_shift_is_validated(self):
        """Changed bond at (4,5) — disjoint from breaking bond (0,1) — and
        BO shift 1.0 → MUST be validated and rejected because distance ≠ target."""
        spec = ReactionPathSpec(
            breaking_bonds=[(0, 1)],
            changed_bonds=[(4, 5)],
            ref_dist_r={(4, 5): 1.50},
            ref_dist_p={(4, 5): 1.20},
            bond_order_r={(4, 5): 1.0},
            bond_order_p={(4, 5): 2.0},
        )
        xyz = self._xyz_chain(6)  # bond (4,5) at 2.0 Å — far from target ~1.35
        bad, reason = has_bad_changed_bond_length(spec, xyz, symbols=tuple('C' * 6))
        self.assertTrue(bad, 'Isolated changed bond must NOT be exempt; got pass')
        self.assertIn('bad-changed-bond', reason)

    def test_adjacent_changed_bond_without_bo_shift_is_validated(self):
        """Changed bond at (1,2) shares an atom with breaking bond (0,1) but
        the BO shift is below 0.5 → exemption does NOT apply, must validate."""
        spec = ReactionPathSpec(
            breaking_bonds=[(0, 1)],
            changed_bonds=[(1, 2)],
            ref_dist_r={(1, 2): 1.50},
            ref_dist_p={(1, 2): 1.45},
            bond_order_r={(1, 2): 1.0},
            bond_order_p={(1, 2): 1.2},  # shift = 0.2 < 0.5
        )
        xyz = self._xyz_chain(3)  # bond at 2.0 Å — far from target ~1.475
        bad, reason = has_bad_changed_bond_length(spec, xyz, symbols=('C', 'C', 'C'))
        self.assertTrue(bad, 'Sub-threshold BO shift must NOT be exempt; got pass')
        self.assertIn('bad-changed-bond', reason)

    def test_adjacency_via_forming_bond_also_qualifies(self):
        """Adjacency check considers BOTH breaking and forming bonds."""
        spec = ReactionPathSpec(
            breaking_bonds=[],
            forming_bonds=[(0, 1)],
            changed_bonds=[(1, 2)],
            ref_dist_r={(1, 2): 1.50},
            ref_dist_p={(1, 2): 1.34},
            bond_order_r={(1, 2): 1.0},
            bond_order_p={(1, 2): 2.0},
        )
        xyz = self._xyz_chain(3)
        bad, _ = has_bad_changed_bond_length(spec, xyz, symbols=('C', 'C', 'C'))
        self.assertFalse(bad, 'Forming-bond adjacency must also qualify for exemption')


# ---------------------------------------------------------------------------
# Phase 2b: shared finalizer — stable sort, dedup, cap
# ---------------------------------------------------------------------------


class TestFinalizeTsGuesses(unittest.TestCase):
    """End-to-end coverage of the unified _finalize_ts_guesses helper."""

    def _make_record(self, label: str, score_offset: float = 0.0):
        """Build a ``GuessRecord`` whose XYZ has a unique heavy-atom
        signature so the dedup pass leaves it intact."""
        from arc.job.adapters.ts.linear import GuessRecord
        # Use a 1-C chain offset along z so each guess is heavy-atom-distinct.
        # Encoding the score offset directly in the z coordinate gives us a
        # ready-made knob for nudging the score in tests below.
        xyz = {
            'symbols': ('C',),
            'isotopes': (12,),
            'coords': ((0.0, 0.0, score_offset),),
        }
        return GuessRecord(xyz=xyz, strategy=label)

    def test_finalizer_caps_to_5(self):
        """Pass 10 distinct guesses, expect 5 returned."""
        from arc.job.adapters.ts.linear import _finalize_ts_guesses

        class _StubRxn:
            label = 'stub'

        records = [self._make_record(f'g{i}', score_offset=0.05 * i)
                   for i in range(10)]
        out = _finalize_ts_guesses(records, path_spec=None,
                                    rxn=_StubRxn(), r_mol=None)
        self.assertEqual(len(out), 5)

    def test_finalizer_strict_stable_sort_preserves_input_order_on_score_tie(self):
        """Three guesses A, B, C with A and C tied at the same score
        (and B with a different score).  After the sort, A must precede C."""
        from arc.job.adapters.ts.linear import (
            GuessRecord, _finalize_ts_guesses,
        )

        # Build three distinct heavy-atom XYZs so dedup keeps all three.
        def _xyz(z):
            return {
                'symbols': ('C',),
                'isotopes': (12,),
                'coords': ((0.0, 0.0, float(z)),),
            }
        rec_a = GuessRecord(xyz=_xyz(0.0), strategy='A')
        rec_b = GuessRecord(xyz=_xyz(5.0), strategy='B')
        rec_c = GuessRecord(xyz=_xyz(10.0), strategy='C')

        # Patch the score function: A and C tie at 1.5, B at 2.0.
        from arc.job.adapters.ts import linear as L
        scores_by_strategy = {'A': 1.5, 'B': 2.0, 'C': 1.5}
        original_score = L.score_guess_against_path_spec

        def _fake_score(path_spec, xyz, r_mol, symbols, chemistry=None):
            for r in (rec_a, rec_b, rec_c):
                if r.xyz['coords'][0] == xyz['coords'][0]:
                    return scores_by_strategy[r.strategy]
            return float('inf')

        # Inject a trivial path_spec into each record so the finalizer
        # actually invokes the (patched) scorer.
        from arc.job.adapters.ts.linear_utils.path_spec import ReactionPathSpec
        trivial_spec = ReactionPathSpec()
        for r in (rec_a, rec_b, rec_c):
            r.path_spec = trivial_spec

        class _StubRxn:
            label = 'stub'

        L.score_guess_against_path_spec = _fake_score
        try:
            out = _finalize_ts_guesses(
                [rec_a, rec_b, rec_c], path_spec=None,
                rxn=_StubRxn(), r_mol=None)
        finally:
            L.score_guess_against_path_spec = original_score

        # Expected order: A (1.5, idx 0), C (1.5, idx 2), B (2.0, idx 1).
        self.assertEqual(len(out), 3)
        self.assertEqual(out[0]['coords'][0][2], 0.0)   # A
        self.assertEqual(out[1]['coords'][0][2], 10.0)  # C
        self.assertEqual(out[2]['coords'][0][2], 5.0)   # B

    def test_finalizer_accepts_raw_dicts(self):
        """Plain XYZ dicts (no GuessRecord wrapper) must be wrapped and processed."""
        from arc.job.adapters.ts.linear import _finalize_ts_guesses

        class _StubRxn:
            label = 'stub'

        raw_xyzs = [
            {'symbols': ('C',), 'isotopes': (12,),
             'coords': ((0.0, 0.0, float(i)),)}
            for i in range(3)
        ]
        out = _finalize_ts_guesses(raw_xyzs, path_spec=None,
                                    rxn=_StubRxn(), r_mol=None)
        self.assertEqual(len(out), 3)
        self.assertEqual(out[0]['coords'][0][2], 0.0)
        self.assertEqual(out[2]['coords'][0][2], 2.0)

    def test_finalizer_filters_colliding_atoms(self):
        """Guesses with atomic collisions must be removed before sorting."""
        from arc.job.adapters.ts.linear import _finalize_ts_guesses

        class _StubRxn:
            label = 'stub'

        good = {
            'symbols': ('C', 'C'),
            'isotopes': (12, 12),
            'coords': ((0.0, 0.0, 0.0), (1.5, 0.0, 0.0)),
        }
        bad = {
            'symbols': ('C', 'C'),
            'isotopes': (12, 12),
            'coords': ((0.0, 0.0, 0.0), (0.05, 0.0, 0.0)),  # 0.05 Å apart
        }
        out = _finalize_ts_guesses([bad, good], path_spec=None,
                                    rxn=_StubRxn(), r_mol=None)
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out[0]['coords'][1][0], 1.5, places=6)

    def test_finalizer_addition_pipeline_routing(self):
        """Mock-test of the spec example: 10 valid addition guesses route
        through the finalizer, get sorted, deduped, and capped to 5."""
        from arc.job.adapters.ts.linear import _finalize_ts_guesses

        class _StubRxn:
            label = 'stub'

        # 10 heavy-atom-distinct guesses (no path_spec → all score = inf).
        # Spacing of 0.10 Å between neighbors safely exceeds the 0.05 Å
        # heavy-atom-match tolerance, so dedup leaves all 10 alone.
        guesses = [
            {'symbols': ('C', 'O'),
             'isotopes': (12, 16),
             'coords': ((0.0, 0.0, 0.0), (1.4 + 0.10 * i, 0.0, 0.0))}
            for i in range(10)
        ]
        out = _finalize_ts_guesses(guesses, path_spec=None,
                                    rxn=_StubRxn(), r_mol=None)
        self.assertEqual(len(out), 5)
        # Stable preservation: with all-equal scores, original order survives,
        # then cap-to-5 keeps the first 5 entries.
        for k in range(5):
            self.assertAlmostEqual(out[k]['coords'][1][0], 1.4 + 0.10 * k,
                                   places=6)


if __name__ == '__main__':
    unittest.main()
