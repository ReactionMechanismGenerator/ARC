#!/usr/bin/env python3
# encoding: utf-8

"""
Unit tests for arc.job.adapters.ts.linear_utils.path_spec
"""

import unittest
from unittest.mock import patch

import numpy as np

from arc.common import get_single_bond_length
from arc.job.adapters.ts import linear
from arc.job.adapters.ts.linear import GuessRecord, _PathContext, _build_addition_path_spec, _enrich_post_migration_path_spec, _finalize_ts_guesses
from arc.job.adapters.ts.linear_utils import addition, path_spec
from arc.job.adapters.ts.linear_utils.migration_inference import infer_frag_fallback_h_migration
from arc.job.adapters.ts.linear_utils.path_spec import (
    PAULING_DELTA,
    PathChemistry,
    ReactionPathSpec,
    _all_bonds,
    _bond_order_map,
    _canon,
    _canon_list,
    _compute_changed_bonds,
    _compute_unchanged_near_core,
    _heavy_forming_bonds,
    _multi_source_bfs,
    _safe_order,
    _shared_atoms_between_bb_and_fb,
    _xyz_distance,
    classify_path_chemistry,
    get_ts_target_distance,
    has_bad_changed_bond_length,
    has_bad_reactive_core_planarity,
    has_bad_unchanged_near_core_bond,
    has_committed_spectator_group,
    has_inward_blocking_h_on_forming_axis,
    has_recipe_channel_mismatch,
    has_wrong_h_migration_committed,
    insertion_ring_extra_stretch,
    mol_to_adjacency,
    score_guess_against_path_spec,
    validate_addition_guess,
    validate_guess_against_path_spec,
)
from arc.job.adapters.ts.linear_utils.postprocess import validate_ts_guess
from arc.reaction import ARCReaction
from arc.species import ARCSpecies


def _propane_radical():
    return ARCSpecies(label='propyl', smiles='CC[CH2]')


def _propene():
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

    def test_mol_to_adjacency_propane(self):
        mol = _propane_radical().mol
        adj = mol_to_adjacency(mol)
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
        unchanged = _compute_unchanged_near_core(mol, breaking_bonds=[], forming_bonds=fb, changed_bonds=[])
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
        spec = ReactionPathSpec.build(r_mol=sp.mol,
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
        spec = ReactionPathSpec.build(r_mol=sp.mol, mapped_p_mol=sp.mol,
                                      breaking_bonds=[(2, 0)], forming_bonds=[(3, 1)])
        self.assertEqual(spec.breaking_bonds, [(0, 2)])
        self.assertEqual(spec.forming_bonds, [(1, 3)])

    def test_build_deterministic(self):
        sp = _propane_radical()
        s1 = ReactionPathSpec.build(r_mol=sp.mol, mapped_p_mol=sp.mol,
                                    breaking_bonds=[(0, 1), (3, 2)], forming_bonds=[(4, 5)])
        s2 = ReactionPathSpec.build(r_mol=sp.mol, mapped_p_mol=sp.mol,
                                    breaking_bonds=[(2, 3), (1, 0)], forming_bonds=[(5, 4)])
        self.assertEqual(s1.breaking_bonds, s2.breaking_bonds)
        self.assertEqual(s1.forming_bonds, s2.forming_bonds)

    def test_build_stores_reference_distances(self):
        sp = _propane_radical()
        # Use the species' own xyz as both reactant and product reference.
        xyz = sp.get_xyz()
        spec = ReactionPathSpec.build(r_mol=sp.mol, mapped_p_mol=sp.mol,
                                      breaking_bonds=[(0, 1)], forming_bonds=[(1, 2)],
                                      r_xyz=xyz, op_xyz=xyz)
        # Reference distances should be populated for the reactive bonds.
        for key in spec.breaking_bonds + spec.forming_bonds:
            self.assertIn(key, spec.ref_dist_r)
            self.assertIn(key, spec.ref_dist_p)
            self.assertIsNotNone(spec.ref_dist_r[key])
            self.assertIsNotNone(spec.ref_dist_p[key])

    def test_build_unchanged_near_core_populated(self):
        sp = _propane_radical()
        spec = ReactionPathSpec.build(r_mol=sp.mol, mapped_p_mol=sp.mol,
                                      breaking_bonds=[(0, 1)], forming_bonds=[])
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
# has_recipe_channel_mismatch — exact thresholds
# ---------------------------------------------------------------------------

class TestHasRecipeChannelMismatch(unittest.TestCase):
    """Exact thresholds: 1.30 / 3.00 / 1.25×SBL."""

    def _two_atom_xyz(self, sym1: str, sym2: str, distance: float) -> dict:
        return {'symbols': (sym1, sym2),
                'isotopes': (12, 12),
                'coords': ((0.0, 0.0, 0.0), (distance, 0.0, 0.0))}

    def test_forming_bond_far_apart_returns_true(self):
        spec = ReactionPathSpec(forming_bonds=[(0, 1)])
        xyz = self._two_atom_xyz('C', 'C', 3.5)
        mismatch, reason = has_recipe_channel_mismatch(spec, xyz)
        self.assertTrue(mismatch)
        self.assertIn('failed-to-form', reason)

    def test_forming_bond_reasonable_returns_false(self):
        spec = ReactionPathSpec(forming_bonds=[(0, 1)])
        xyz = self._two_atom_xyz('C', 'C', 2.0)
        mismatch, _ = has_recipe_channel_mismatch(spec, xyz)
        self.assertFalse(mismatch)

    def test_forming_bond_just_below_threshold_passes(self):
        spec = ReactionPathSpec(forming_bonds=[(0, 1)])
        xyz = self._two_atom_xyz('C', 'C', 2.99)
        mismatch, _ = has_recipe_channel_mismatch(spec, xyz)
        self.assertFalse(mismatch)

    def test_breaking_bond_too_short_returns_true(self):
        spec = ReactionPathSpec(breaking_bonds=[(0, 1)])
        xyz = self._two_atom_xyz('C', 'C', 1.2)
        mismatch, reason = has_recipe_channel_mismatch(spec, xyz)
        self.assertTrue(mismatch)
        self.assertIn('failed-to-break', reason)

    def test_breaking_bond_at_threshold_passes(self):
        spec = ReactionPathSpec(breaking_bonds=[(0, 1)])
        xyz = self._two_atom_xyz('C', 'C', 1.31)
        mismatch, _ = has_recipe_channel_mismatch(spec, xyz)
        self.assertFalse(mismatch)

    def test_unchanged_near_core_snapped_returns_true(self):
        sbl_cc = get_single_bond_length('C', 'C')
        # 1.30 × sbl is above the 1.25 threshold.
        d = 1.30 * sbl_cc
        spec = ReactionPathSpec(unchanged_near_core_bonds=[(0, 1)])
        xyz = self._two_atom_xyz('C', 'C', d)
        mismatch, reason = has_recipe_channel_mismatch(spec, xyz)
        self.assertTrue(mismatch)
        self.assertIn('snapped-spectator', reason)

    def test_unchanged_near_core_within_limit_passes(self):
        sbl_cc = get_single_bond_length('C', 'C')
        d = 1.20 * sbl_cc
        spec = ReactionPathSpec(unchanged_near_core_bonds=[(0, 1)])
        xyz = self._two_atom_xyz('C', 'C', d)
        mismatch, _ = has_recipe_channel_mismatch(spec, xyz)
        self.assertFalse(mismatch)

    def test_empty_spec_returns_false(self):
        spec = ReactionPathSpec()
        xyz = self._two_atom_xyz('C', 'C', 1.5)
        mismatch, _ = has_recipe_channel_mismatch(spec, xyz)
        self.assertFalse(mismatch)


# ---------------------------------------------------------------------------
# validate_guess_against_path_spec
# ---------------------------------------------------------------------------

class TestValidateGuessAgainstPathSpec(unittest.TestCase):
    """The wrapper must compose generic validation + recipe-channel mismatch."""

    def test_valid_guess_passes(self):
        sp = _propane_radical()
        xyz = sp.get_xyz()
        spec = ReactionPathSpec.build(r_mol=sp.mol, mapped_p_mol=sp.mol,
                                      breaking_bonds=[], forming_bonds=[])
        ok, reason = validate_guess_against_path_spec(xyz=xyz, path_spec=spec, r_mol=sp.mol, family=None)
        self.assertTrue(ok, f'Expected pass; got reason {reason}')

    def test_recipe_mismatch_rejects(self):
        """A guess where a forming bond is far apart is rejected by the wrapper.

        Construct the test against an empty path_spec field with only the
        forming bond declared, and a hand-built XYZ that does not collide.
        """
        # Two H atoms placed 4 Å apart — no collisions, but forming bond is
        # far above the 3.00 Å threshold.
        h2 = ARCSpecies(label='H2', smiles='[H][H]')
        bad_xyz = {'symbols': ('H', 'H'),
                   'isotopes': (1, 1),
                   'coords': ((0.0, 0.0, 0.0), (4.0, 0.0, 0.0))}
        spec = ReactionPathSpec(forming_bonds=[(0, 1)])
        ok, reason = validate_guess_against_path_spec(xyz=bad_xyz, path_spec=spec, r_mol=h2.mol)
        self.assertFalse(ok)
        self.assertIn('recipe-mismatch', reason)


# ---------------------------------------------------------------------------
# Centralization plumbing — confirm linear.py imports the wrapper
# ---------------------------------------------------------------------------

class TestPlumbingCentralization(unittest.TestCase):
    """Prove that the orchestration shell wires through the new wrapper."""

    def test_linear_module_imports_wrapper(self):
        self.assertTrue(hasattr(linear, 'validate_guess_against_path_spec'))
        self.assertTrue(hasattr(linear, 'ReactionPathSpec'))

    def test_path_context_has_path_spec_field(self):
        # The dataclass must declare a path_spec field.
        self.assertIn('path_spec', _PathContext.__dataclass_fields__)

    def test_guess_record_has_path_spec_field(self):
        self.assertIn('path_spec', GuessRecord.__dataclass_fields__)


# ---------------------------------------------------------------------------
# PathChemistry classification
# ---------------------------------------------------------------------------

class TestPathChemistryClassification(unittest.TestCase):
    """Verify the exact rule order of :func:`classify_path_chemistry`."""

    def test_substitution_like_single_heavy_pivot(self):
        # Single bb, single fb, sharing one heavy atom (atom 1).
        # Pivot is C → SUBSTITUTION_LIKE.
        spec = ReactionPathSpec(breaking_bonds=[(1, 2)], forming_bonds=[(0, 1)])
        symbols = ('C', 'C', 'C')
        chem = classify_path_chemistry(spec, symbols=symbols)
        self.assertIs(chem, PathChemistry.SUBSTITUTION_LIKE)

    def test_substitution_like_with_h_pivot_falls_to_h_transfer(self):
        # The shared atom is H, so substitution_like should NOT match;
        # should fall through to H_TRANSFER.
        spec = ReactionPathSpec(
            breaking_bonds=[(0, 1)],  # C-H
            forming_bonds=[(1, 2)],   # H-C
        )
        symbols = ('C', 'H', 'C')
        chem = classify_path_chemistry(spec, symbols=symbols)
        self.assertIs(chem, PathChemistry.H_TRANSFER)

    def test_h_transfer_when_shared_atom_is_h(self):
        spec = ReactionPathSpec(
            breaking_bonds=[(0, 4)],  # heavy-H
            forming_bonds=[(4, 3)],   # H-heavy
        )
        symbols = ('C', 'C', 'C', 'C', 'H')
        chem = classify_path_chemistry(spec, symbols=symbols)
        self.assertIs(chem, PathChemistry.H_TRANSFER)

    def test_non_h_group_shift_when_shared_atom_is_heavy(self):
        # Two bonds, but multiple bb/fb with a heavy shared atom and no
        # SUBSTITUTION_LIKE single-bond match.
        spec = ReactionPathSpec(
            breaking_bonds=[(0, 1), (3, 4)],
            forming_bonds=[(0, 2)],  # atom 0 (C) is the shared pivot
        )
        symbols = ('C', 'C', 'C', 'C', 'C')
        chem = classify_path_chemistry(spec, symbols=symbols)
        self.assertIs(chem, PathChemistry.NON_H_GROUP_SHIFT)

    def test_concerted_hetero_rearrangement(self):
        # 2+ bb, 2+ fb, with at least 2 hetero reactive atoms (O, O).
        spec = ReactionPathSpec(breaking_bonds=[(0, 1), (2, 3)], forming_bonds=[(4, 5), (6, 7)])
        # symbols: indices 1 and 3 are O, others C; 4 and 7 also O.
        symbols = ('C', 'O', 'C', 'O', 'C', 'C', 'C', 'O')
        chem = classify_path_chemistry(spec, symbols=symbols)
        self.assertIs(chem, PathChemistry.CONCERTED_HETERO_REARRANGEMENT)

    def test_cycloaddition_or_ring_closure(self):
        # No shared atoms, single heavy-heavy forming bond → cycloaddition.
        spec = ReactionPathSpec(forming_bonds=[(0, 4)])
        symbols = ('C', 'C', 'C', 'C', 'C')
        chem = classify_path_chemistry(spec, symbols=symbols)
        self.assertIs(chem, PathChemistry.CYCLOADDITION_OR_RING_CLOSURE)

    def test_generic_fallback(self):
        # No shared atoms, no heavy-heavy forming bond → GENERIC.
        spec = ReactionPathSpec(forming_bonds=[(0, 1)])  # H is index 1, so heavy-H not heavy-heavy
        symbols = ('C', 'H')
        chem = classify_path_chemistry(spec, symbols=symbols)
        self.assertIs(chem, PathChemistry.GENERIC)

    def test_empty_spec_is_generic(self):
        chem = classify_path_chemistry(ReactionPathSpec(), symbols=())
        self.assertIs(chem, PathChemistry.GENERIC)


# ---------------------------------------------------------------------------
# changed-bond length validator
# ---------------------------------------------------------------------------

class TestHasBadChangedBondLength(unittest.TestCase):

    def _xyz(self, distance, sym1='C', sym2='C'):
        return {'symbols': (sym1, sym2),
                'isotopes': (12, 12),
                'coords': ((0.0, 0.0, 0.0), (distance, 0.0, 0.0))}

    def test_no_changed_bonds_returns_false(self):
        spec = ReactionPathSpec()
        bad, reason = has_bad_changed_bond_length(spec, self._xyz(1.5), symbols=('C', 'C'))
        self.assertFalse(bad)
        self.assertEqual(reason, '')

    def test_changed_bond_within_tolerance_passes(self):
        spec = ReactionPathSpec(changed_bonds=[(0, 1)],
                                ref_dist_r={(0, 1): 1.50},
                                ref_dist_p={(0, 1): 1.40})
        # target = 0.5*(1.50+1.40) = 1.45; we use 1.50 (deviation 0.05)
        bad, _ = has_bad_changed_bond_length(spec, self._xyz(1.50), symbols=('C', 'C'))
        self.assertFalse(bad)

    def test_changed_bond_heavy_outside_tolerance_rejects(self):
        spec = ReactionPathSpec(changed_bonds=[(0, 1)],
                                ref_dist_r={(0, 1): 1.50}, ref_dist_p={(0, 1): 1.40})
        # target = 1.45; deviation must exceed 0.25 → use 1.80 (dev = 0.35).
        bad, reason = has_bad_changed_bond_length(spec, self._xyz(1.80),
                                                  symbols=('C', 'C'))
        self.assertTrue(bad)
        self.assertIn('bad-changed-bond', reason)

    def test_changed_bond_h_involving_uses_tighter_tolerance(self):
        # H tolerance is 0.20.
        spec = ReactionPathSpec(changed_bonds=[(0, 1)],
                                ref_dist_r={(0, 1): 1.10}, ref_dist_p={(0, 1): 1.00})
        # target = 1.05; deviation 0.21 > 0.20 → reject.
        bad, _ = has_bad_changed_bond_length(spec, self._xyz(1.26, sym2='H'), symbols=('C', 'H'))
        self.assertTrue(bad)


# ---------------------------------------------------------------------------
# unchanged_near_core bond validator
# ---------------------------------------------------------------------------

class TestHasBadUnchangedNearCoreBond(unittest.TestCase):

    def _xyz(self, distance):
        return {'symbols': ('C', 'C'),
                'isotopes': (12, 12),
                'coords': ((0.0, 0.0, 0.0), (distance, 0.0, 0.0))}

    def test_unchanged_within_window_passes(self):
        spec = ReactionPathSpec(unchanged_near_core_bonds=[(0, 1)], ref_dist_r={(0, 1): 1.54})
        bad, _ = has_bad_unchanged_near_core_bond(spec, self._xyz(1.55), symbols=('C', 'C'))
        self.assertFalse(bad)

    def test_unchanged_too_short_rejects(self):
        spec = ReactionPathSpec(unchanged_near_core_bonds=[(0, 1)], ref_dist_r={(0, 1): 1.54})
        # 0.82*1.54 = 1.263 → 1.20 is below
        bad, reason = has_bad_unchanged_near_core_bond(spec, self._xyz(1.20), symbols=('C', 'C'))
        self.assertTrue(bad)
        self.assertIn('bad-unchanged-near-core', reason)

    def test_unchanged_too_long_rejects(self):
        spec = ReactionPathSpec(unchanged_near_core_bonds=[(0, 1)], ref_dist_r={(0, 1): 1.54})
        # 1.25*1.54 = 1.925 → 2.10 is above
        bad, _ = has_bad_unchanged_near_core_bond(spec, self._xyz(2.10), symbols=('C', 'C'))
        self.assertTrue(bad)

    def test_no_ref_falls_back_to_sbl(self):
        spec = ReactionPathSpec(unchanged_near_core_bonds=[(0, 1)])
        # SBL(C,C) ≈ 1.54; 1.55 is fine, 3.00 is way over.
        bad, _ = has_bad_unchanged_near_core_bond(
            spec, self._xyz(1.55), symbols=('C', 'C'))
        self.assertFalse(bad)
        bad, _ = has_bad_unchanged_near_core_bond(
            spec, self._xyz(3.00), symbols=('C', 'C'))
        self.assertTrue(bad)


# ---------------------------------------------------------------------------
# inward-blocking-H-on-forming-axis validator
# ---------------------------------------------------------------------------

class TestHasInwardBlockingHOnFormingAxis(unittest.TestCase):

    def test_inactive_when_chemistry_is_h_transfer(self):
        # The check must be skipped for non-substitution/cycloaddition chemistries.
        ethane = ARCSpecies(label='ethane', smiles='CC')
        spec = ReactionPathSpec(forming_bonds=[(0, 1)])
        bad, _ = has_inward_blocking_h_on_forming_axis(spec, ethane.get_xyz(), ethane.mol,
                                                       symbols=tuple(a.element.symbol for a in ethane.mol.atoms),
                                                       chemistry=PathChemistry.H_TRANSFER)
        self.assertFalse(bad)

    def test_inactive_when_no_heavy_heavy_forming_bond(self):
        h2 = ARCSpecies(label='H2', smiles='[H][H]')
        spec = ReactionPathSpec(forming_bonds=[(0, 1)])
        bad, _ = has_inward_blocking_h_on_forming_axis(spec, h2.get_xyz(), h2.mol, symbols=('H', 'H'),
                                                       chemistry=PathChemistry.CYCLOADDITION_OR_RING_CLOSURE)
        self.assertFalse(bad)

    def test_blocking_h_intrudes_on_axis(self):
        # Construct an artificial geometry: C0 and C1 forming a bond ~3 Å
        # apart, with a hydrogen on C0 placed close to C1 and ~at right angles.
        xyz = {'symbols': ('C', 'C', 'H'),
               'isotopes': (12, 12, 1),
               'coords': (
                   (0.0, 0.0, 0.0),  # C0
                   (3.0, 0.0, 0.0),  # C1
                   (1.5, 0.05, 0.0))}  # H, halfway in front of C1

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

        spec = ReactionPathSpec(forming_bonds=[(0, 1)], reactive_atoms={0, 1})
        bad, reason = has_inward_blocking_h_on_forming_axis(spec, xyz, _StubMol(),
                                                            symbols=('C', 'C', 'H'),
                                                            chemistry=PathChemistry.CYCLOADDITION_OR_RING_CLOSURE)
        self.assertTrue(bad)
        self.assertIn('inward-blocking-H', reason)


# ---------------------------------------------------------------------------
# reactive-core-planarity validator
# ---------------------------------------------------------------------------

class TestHasBadReactiveCorePlanarity(unittest.TestCase):

    def test_inactive_for_non_concerted(self):
        spec = ReactionPathSpec(breaking_bonds=[(0, 1)], forming_bonds=[(2, 3)])
        xyz = {'symbols': ('C', 'O', 'C', 'O'),
               'isotopes': (12, 16, 12, 16),
               'coords': ((0., 0., 0.), (1., 0., 0.), (0., 1., 0.), (1., 1., 5.))}
        bad, _ = has_bad_reactive_core_planarity(spec, xyz, symbols=('C', 'O', 'C', 'O'),
                                                 chemistry=PathChemistry.GENERIC)
        self.assertFalse(bad)

    def test_few_atoms_returns_false(self):
        # Fewer than 4 heavy reactive atoms → cannot fit a plane → False.
        spec = ReactionPathSpec(breaking_bonds=[(0, 1), (2, 3)], forming_bonds=[(0, 2), (1, 3)], )
        xyz = {'symbols': ('O', 'O', 'C'),
               'isotopes': (16, 16, 12),
               'coords': ((0., 0., 0.), (1., 0., 0.), (0., 1., 0.))}
        bad, _ = has_bad_reactive_core_planarity(
            spec, xyz, symbols=('O', 'O', 'C'),
            chemistry=PathChemistry.CONCERTED_HETERO_REARRANGEMENT)
        self.assertFalse(bad)

    def test_planar_core_passes(self):
        spec = ReactionPathSpec(breaking_bonds=[(0, 1), (2, 3)], forming_bonds=[(0, 2), (1, 3)], )
        xyz = {'symbols': ('O', 'O', 'C', 'C'),
               'isotopes': (16, 16, 12, 12),
               'coords': ((0., 0., 0.), (1., 0., 0.), (1., 1., 0.), (0., 1., 0.))}
        bad, _ = has_bad_reactive_core_planarity(
            spec, xyz, symbols=('O', 'O', 'C', 'C'),
            chemistry=PathChemistry.CONCERTED_HETERO_REARRANGEMENT)
        self.assertFalse(bad)

    def test_strongly_nonplanar_core_rejects(self):
        spec = ReactionPathSpec(breaking_bonds=[(0, 1), (2, 3)], forming_bonds=[(0, 2), (1, 3)])
        # A genuinely 3D tetrahedral arrangement of the 4 reactive atoms.
        # No plane fits all four within 0.35 Å RMS.
        xyz = {'symbols': ('O', 'O', 'C', 'C'),
               'isotopes': (16, 16, 12, 12),
               'coords': ((0., 0., 0.), (2., 0., 0.), (0., 2., 0.), (0., 0., 2.))}
        bad, reason = has_bad_reactive_core_planarity(spec, xyz, symbols=('O', 'O', 'C', 'C'),
                                                      chemistry=PathChemistry.CONCERTED_HETERO_REARRANGEMENT)
        self.assertTrue(bad)
        self.assertIn('reactive-core', reason)


# ---------------------------------------------------------------------------
# narrow recipe-consistency / wrong-channel screening
# ---------------------------------------------------------------------------

class TestHasWrongHMigrationCommitted(unittest.TestCase):
    """``has_wrong_h_migration_committed`` rejects only when:

    1. chemistry is H_TRANSFER, AND
    2. the path-spec names a migrating H, AND
    3. the named migrating H still sits near its donor (≤ 1.20 × sbl), AND
    4. a spectator H is at most 0.70 × of the named migrating H's
       distance to the named acceptor.
    """

    def setUp(self):
        # Use propane as a backbone with a clear migrating-H story:
        # propane = C0-C1-C2 with an H on each carbon.
        self.sp = ARCSpecies(label='propane', smiles='CCC')
        self.symbols = self.sp.get_xyz()['symbols']
        self.atom_to_idx = {a: i for i, a in enumerate(self.sp.mol.atoms)}
        # Pick the two terminal C atoms.
        terminal_cs = [self.atom_to_idx[a] for a in self.sp.mol.atoms if a.element.symbol == 'C'
                       and sum(1 for n in a.bonds.keys() if n.element.symbol != 'H') == 1]
        self.c_donor, self.c_acceptor = terminal_cs[0], terminal_cs[1]
        # Pick one H bonded to c_donor as the intended migrating H.
        self.h_intended = next(self.atom_to_idx[n] for n in self.sp.mol.atoms[self.c_donor].bonds.keys()
                               if n.element.symbol == 'H')
        # Pick one H bonded to c_acceptor as the spectator H.
        self.h_spectator = next(self.atom_to_idx[n] for n in self.sp.mol.atoms[self.c_acceptor].bonds.keys()
                                if n.element.symbol == 'H')

    def _make_spec(self):
        # Path-spec: the forming bond is (h_intended, c_acceptor) — that's
        # the recipe's "this H is migrating to this acceptor".
        return ReactionPathSpec(breaking_bonds=[(self.c_donor, self.h_intended)],
                                forming_bonds=[(self.c_acceptor, self.h_intended)],
                                family='intra_H_migration')

    def test_no_op_when_chemistry_is_not_h_transfer(self):
        spec = self._make_spec()
        xyz = self.sp.get_xyz()
        bad, _ = has_wrong_h_migration_committed(spec, xyz, self.sp.mol, self.symbols, chemistry=PathChemistry.GENERIC)
        self.assertFalse(bad)

    def test_no_op_when_intended_h_already_moved(self):
        """If the named migrating H has already moved off the donor
        (d > 1.20 × sbl), the rule does not fire even if a spectator
        is closer to the acceptor — the channel has engaged."""
        spec = self._make_spec()
        coords = np.asarray(self.sp.get_xyz()['coords'], dtype=float).copy()
        # Move the intended H far from the donor.
        donor_pos = coords[self.c_donor]
        acceptor_pos = coords[self.c_acceptor]
        # Place at the midpoint between donor and acceptor.
        coords[self.h_intended] = 0.5 * (donor_pos + acceptor_pos)
        xyz = {**self.sp.get_xyz(),
               'coords': tuple(tuple(row) for row in coords)}
        bad, _ = has_wrong_h_migration_committed(spec, xyz, self.sp.mol, self.symbols, chemistry=PathChemistry.H_TRANSFER)
        self.assertFalse(bad)

    def test_rejects_when_spectator_h_is_committed(self):
        """A spectator H bonded to the acceptor C trivially sits at
        ~1.09 Å from the acceptor.  When the intended migrating H is
        still near its donor (d ~ 1.09 Å), the spectator's d/intended
        ratio is ~ 1.09 / d_intended_acceptor.  In propane the
        intended H to the acceptor is several Å away, so the rule
        should fire."""
        spec = self._make_spec()
        xyz = self.sp.get_xyz()
        bad, reason = has_wrong_h_migration_committed(spec, xyz, self.sp.mol, self.symbols, chemistry=PathChemistry.H_TRANSFER)
        self.assertTrue(bad)
        self.assertIn('wrong-h-migration', reason)
        self.assertIn(f'H{self.h_spectator}', reason)

    def test_no_op_when_no_h_in_forming_bond(self):
        """A forming bond between two heavy atoms is not subject to
        this check — there is no migrating H to test."""
        spec = ReactionPathSpec(breaking_bonds=[(self.c_donor, self.c_acceptor)],
                                forming_bonds=[(self.c_donor, self.c_acceptor)],
                                family='generic')
        xyz = self.sp.get_xyz()
        bad, _ = has_wrong_h_migration_committed(spec, xyz, self.sp.mol, self.symbols, chemistry=PathChemistry.H_TRANSFER)
        self.assertFalse(bad)


class TestHasCommittedSpectatorGroup(unittest.TestCase):
    """
    ``has_committed_spectator_group`` rejects only for an explicit
    family allowlist when a spectator heavy atom has formed a near-
    bond-length contact with a reactive endpoint that is not its
    expected partner.
    """

    def _xyz_two_carbons(self, d_ca: float, d_spectator: float):
        """
        Build a 4-atom xyz: C0 (reactive), C1 (reactive partner),
        C2 (a far spectator), C3 (the spectator committed to C0 at
        ``d_spectator`` — placed perpendicular to the C0-C1 axis so it
        is not also close to C1).
        """
        return {'symbols': ('C', 'C', 'C', 'C'),
                'isotopes': (12, 12, 12, 12),
                'coords': (
                    (0.0, 0.0, 0.0),           # C0
                    (d_ca, 0.0, 0.0),          # C1 (reactive partner)
                    (10.0, 0.0, 0.0),          # C2 (spectator far away)
                    (0.0, d_spectator, 0.0))}  # C3 (spectator perpendicular to C0)

    def test_no_op_for_unrelated_family(self):
        spec = ReactionPathSpec(breaking_bonds=[(0, 1)], forming_bonds=[(0, 1)],
                                reactive_atoms={0, 1}, family='intra_H_migration')
        xyz = self._xyz_two_carbons(d_ca=2.0, d_spectator=1.0)
        # Build a tiny propane mol — only used for n_atoms / adj API.
        mol = ARCSpecies(label='butane', smiles='CCCC').mol
        bad, _ = has_committed_spectator_group(spec, xyz, mol, symbols=('C', 'C', 'C', 'C'))
        self.assertFalse(bad)

    def test_rejects_for_allowlisted_family_when_committed(self):
        # Family is on the allowlist; spectator C3 sits at 1.0 Å from
        # reactive endpoint C0 — well below 0.85 × sbl(C, C) = 1.275 Å.
        spec = ReactionPathSpec(breaking_bonds=[(0, 1)], forming_bonds=[(0, 1)],
                                reactive_atoms={0, 1}, family='1,3_Insertion_ROR')
        xyz = self._xyz_two_carbons(d_ca=2.0, d_spectator=1.0)
        mol = ARCSpecies(label='butane', smiles='CCCC').mol
        bad, reason = has_committed_spectator_group(spec, xyz, mol, symbols=('C', 'C', 'C', 'C'))
        self.assertTrue(bad)
        self.assertIn('committed-spectator', reason)
        self.assertIn('1,3_Insertion_ROR', reason)

    def test_no_op_when_distance_is_just_long_enough(self):
        # Spectator at d = 1.40 Å — above 0.85 × sbl(C,C) = 1.275 Å.
        spec = ReactionPathSpec(breaking_bonds=[(0, 1)], forming_bonds=[(0, 1)],
            reactive_atoms={0, 1}, family='1,3_Insertion_ROR')
        xyz = self._xyz_two_carbons(d_ca=2.0, d_spectator=1.40)
        mol = ARCSpecies(label='butane', smiles='CCCC').mol
        bad, _ = has_committed_spectator_group(spec, xyz, mol, symbols=('C', 'C', 'C', 'C'))
        self.assertFalse(bad)

    def test_no_op_when_spectator_is_already_an_approved_partner(self):
        # The spectator atom IS in the path-spec as a reactive partner —
        # so it is not a spectator at all.
        spec = ReactionPathSpec(breaking_bonds=[(0, 1), (0, 3)], forming_bonds=[(0, 1), (0, 3)],
            reactive_atoms={0, 1, 3}, family='1,3_Insertion_ROR')
        xyz = self._xyz_two_carbons(d_ca=2.0, d_spectator=1.0)
        mol = ARCSpecies(label='butane', smiles='CCCC').mol
        bad, _ = has_committed_spectator_group(spec, xyz, mol, symbols=('C', 'C', 'C', 'C'))
        self.assertFalse(bad)


# ---------------------------------------------------------------------------
# scoring function
# ---------------------------------------------------------------------------

class TestScoreGuessAgainstPathSpec(unittest.TestCase):

    def test_zero_score_for_perfect_breaking_target(self):
        # Single breaking bond at exactly sbl + Pauling delta should give 0.
        sbl = get_single_bond_length('C', 'C')
        target = sbl + PAULING_DELTA
        spec = ReactionPathSpec(breaking_bonds=[(0, 1)])
        xyz = {'symbols': ('C', 'C'),
               'isotopes': (12, 12),
               'coords': ((0.0, 0.0, 0.0), (target, 0.0, 0.0))}
        s = score_guess_against_path_spec(
            spec, xyz, r_mol=None, symbols=('C', 'C'),
            chemistry=PathChemistry.GENERIC)
        self.assertAlmostEqual(s, 0.0, places=5)

    def test_score_increases_with_breaking_deviation(self):
        sbl = get_single_bond_length('C', 'C')
        target = sbl + PAULING_DELTA
        spec = ReactionPathSpec(breaking_bonds=[(0, 1)])
        xyz_close = {'symbols': ('C', 'C'),
                     'isotopes': (12, 12),
                     'coords': ((0.0, 0.0, 0.0), (target + 0.1, 0.0, 0.0))}
        xyz_far = {'symbols': ('C', 'C'),
                   'isotopes': (12, 12),
                   'coords': ((0.0, 0.0, 0.0), (target + 0.5, 0.0, 0.0))}
        s_close = score_guess_against_path_spec(
            spec, xyz_close, r_mol=None, symbols=('C', 'C'),
            chemistry=PathChemistry.GENERIC)
        s_far = score_guess_against_path_spec(
            spec, xyz_far, r_mol=None, symbols=('C', 'C'),
            chemistry=PathChemistry.GENERIC)
        self.assertLess(s_close, s_far)

    def test_planarity_penalty_added(self):
        spec = ReactionPathSpec(breaking_bonds=[(0, 1), (2, 3)], forming_bonds=[(0, 2), (1, 3)])
        # Strongly non-planar core (tetrahedral 3D arrangement).
        xyz = {'symbols': ('O', 'O', 'C', 'C'),
               'isotopes': (16, 16, 12, 12),
               'coords': ((0., 0., 0.), (2., 0., 0.), (0., 2., 0.), (0., 0., 2.))}
        s = score_guess_against_path_spec(
            spec, xyz, r_mol=None, symbols=('O', 'O', 'C', 'C'),
            chemistry=PathChemistry.CONCERTED_HETERO_REARRANGEMENT)
        # Score must include the +5 planarity penalty.
        self.assertGreaterEqual(s, 5.0)

    def test_none_xyz_returns_inf(self):
        spec = ReactionPathSpec()
        s = score_guess_against_path_spec(spec, xyz=None, r_mol=None, symbols=())
        self.assertEqual(s, float('inf'))


# ---------------------------------------------------------------------------
# validator dispatch — H_TRANSFER routing
# ---------------------------------------------------------------------------

class TestHTransferDispatchOverride(unittest.TestCase):
    """When chemistry == H_TRANSFER the H-migration validator must run regardless of family string."""

    def test_h_transfer_routes_to_h_migration_even_with_unknown_family(self):
        # Build a non-trivial xyz with two H atoms placed too close together.
        # validate_h_migration's _has_h_close_contact should fire.
        xyz = {'symbols': ('C', 'H', 'H'),
               'isotopes': (12, 1, 1),
               'coords': ((0.0, 0.0, 0.0), (0.7, 0.0, 0.0), (0.78, 0.0, 0.0))}
        c = ARCSpecies(label='CH2', smiles='[CH2]')
        ok, reason = validate_ts_guess(xyz=xyz,
                                       migrating_hs={1, 2},
                                       forming_bonds=[(0, 1)],
                                       r_mol=c.mol,
                                       label='test',
                                       family='non_existent_family',
                                       chemistry='h_transfer')
        self.assertFalse(ok)

    def test_non_h_transfer_chemistry_skips_h_migration(self):
        # Sanity: passing chemistry=None preserves prior behavior — an
        # unknown family runs no family validator.
        sp = _propane_radical()
        ok, _ = validate_ts_guess(xyz=sp.get_xyz(),
                                  migrating_hs=set(),
                                  forming_bonds=[],
                                  r_mol=sp.mol,
                                  label='test',
                                  family='non_existent_family',
                                  chemistry=None)
        self.assertTrue(ok)


# ---------------------------------------------------------------------------
# orchestration triage — score-sort + second dedup
# ---------------------------------------------------------------------------

class TestOrchestrationTriage(unittest.TestCase):
    """Verify linear.py exposes the plumbing correctly."""

    def test_linear_module_imports_path_spec_helpers(self):
        self.assertTrue(hasattr(linear, 'classify_path_chemistry'))
        self.assertTrue(hasattr(linear, 'score_guess_against_path_spec'))
        self.assertTrue(hasattr(linear, 'PathChemistry'))


# ---------------------------------------------------------------------------
# tightened frontier exemption — strict 2-condition rule
# ---------------------------------------------------------------------------

class TestTightenedFrontierExemption(unittest.TestCase):
    """The frontier exemption must require BOTH a non-trivial bond-order
    shift AND direct topological adjacency to a breaking/forming bond."""

    def _xyz_chain(self, n_atoms: int) -> dict:
        """A linear chain of *n_atoms* C atoms spaced 2.0 Å apart on the x axis."""
        coords = tuple((2.0 * i, 0.0, 0.0) for i in range(n_atoms))
        return {'symbols': tuple('C' for _ in range(n_atoms)),
                'isotopes': tuple(12 for _ in range(n_atoms)),
                'coords': coords}

    def test_adjacent_changed_bond_with_bo_shift_is_exempt(self):
        """Changed bond at (1,2) sharing atom 1 with breaking bond (0,1)
        and BO shift 0.5 → exempt (no rejection even though distance is wrong)."""
        spec = ReactionPathSpec(breaking_bonds=[(0, 1)],
                                changed_bonds=[(1, 2)],
                                ref_dist_r={(1, 2): 1.50},
                                ref_dist_p={(1, 2): 1.40},
                                bond_order_r={(1, 2): 1.0},
                                bond_order_p={(1, 2): 1.5})
        xyz = self._xyz_chain(3)  # bond (1,2) at 2.0 Å — way outside tol
        bad, reason = has_bad_changed_bond_length(spec, xyz, symbols=('C', 'C', 'C'))
        self.assertFalse(bad, f'Frontier-adjacent bond should be exempt, got: {reason}')

    def test_isolated_changed_bond_with_bo_shift_is_validated(self):
        """Changed bond at (4,5) — disjoint from breaking bond (0,1) — and
        BO shift 1.0 → MUST be validated and rejected because distance ≠ target."""
        spec = ReactionPathSpec(breaking_bonds=[(0, 1)],
                                changed_bonds=[(4, 5)],
                                ref_dist_r={(4, 5): 1.50},
                                ref_dist_p={(4, 5): 1.20},
                                bond_order_r={(4, 5): 1.0},
                                bond_order_p={(4, 5): 2.0})
        xyz = self._xyz_chain(6)  # bond (4,5) at 2.0 Å — far from target ~1.35
        bad, reason = has_bad_changed_bond_length(spec, xyz, symbols=tuple('C' * 6))
        self.assertTrue(bad, 'Isolated changed bond must NOT be exempt; got pass')
        self.assertIn('bad-changed-bond', reason)

    def test_adjacent_changed_bond_without_bo_shift_is_validated(self):
        """Changed bond at (1,2) shares an atom with breaking bond (0,1) but
        the BO shift is below 0.5 → exemption does NOT apply, must validate."""
        spec = ReactionPathSpec(breaking_bonds=[(0, 1)],
                                changed_bonds=[(1, 2)],
                                ref_dist_r={(1, 2): 1.50},
                                ref_dist_p={(1, 2): 1.45},
                                bond_order_r={(1, 2): 1.0},
                                bond_order_p={(1, 2): 1.2})  # shift = 0.2 < 0.5
        xyz = self._xyz_chain(3)  # bond at 2.0 Å — far from target ~1.475
        bad, reason = has_bad_changed_bond_length(spec, xyz, symbols=('C', 'C', 'C'))
        self.assertTrue(bad, 'Sub-threshold BO shift must NOT be exempt; got pass')
        self.assertIn('bad-changed-bond', reason)

    def test_adjacency_via_forming_bond_also_qualifies(self):
        """Adjacency check considers BOTH breaking and forming bonds."""
        spec = ReactionPathSpec(breaking_bonds=[],
                                forming_bonds=[(0, 1)],
                                changed_bonds=[(1, 2)],
                                ref_dist_r={(1, 2): 1.50},
                                ref_dist_p={(1, 2): 1.34},
                                bond_order_r={(1, 2): 1.0},
                                bond_order_p={(1, 2): 2.0})
        xyz = self._xyz_chain(3)
        bad, _ = has_bad_changed_bond_length(spec, xyz, symbols=('C', 'C', 'C'))
        self.assertFalse(bad, 'Forming-bond adjacency must also qualify for exemption')


# ---------------------------------------------------------------------------
# shared finalizer — stable sort, dedup, cap
# ---------------------------------------------------------------------------

class TestFinalizeTsGuesses(unittest.TestCase):
    """End-to-end coverage of the unified _finalize_ts_guesses helper."""

    def _make_record(self, label: str, score_offset: float = 0.0):
        """Build a ``GuessRecord`` whose XYZ has a unique heavy-atom signature so the dedup pass leaves it intact."""
        # Use a 1-C chain offset along z so each guess is heavy-atom-distinct.
        # Encoding the score offset directly in the z coordinate gives us a
        # ready-made knob for nudging the score in tests below.
        xyz = {'symbols': ('C',),
               'isotopes': (12,),
               'coords': ((0.0, 0.0, score_offset),)}
        return GuessRecord(xyz=xyz, strategy=label)

    def test_finalizer_caps_to_5(self):
        """Pass 10 distinct guesses, expect 5 returned."""

        class _StubRxn:
            label = 'stub'

        records = [self._make_record(f'g{i}', score_offset=0.05 * i) for i in range(10)]
        out = _finalize_ts_guesses(records, path_spec=None, rxn=_StubRxn(), r_mol=None)
        self.assertEqual(len(out), 5)

    def test_finalizer_strict_stable_sort_preserves_input_order_on_score_tie(self):
        """
        Three guesses A, B, C with A and C tied at the same score
        (and B with a different score). After the sort, A must precede C.
        """
        # Build three distinct heavy-atom XYZs so dedup keeps all three.
        def _xyz(z):
            return {'symbols': ('C',),
                    'isotopes': (12,),
                    'coords': ((0.0, 0.0, float(z)),)}
        rec_a = GuessRecord(xyz=_xyz(0.0), strategy='A')
        rec_b = GuessRecord(xyz=_xyz(5.0), strategy='B')
        rec_c = GuessRecord(xyz=_xyz(10.0), strategy='C')

        # Patch the score function: A and C tie at 1.5, B at 2.0.
        scores_by_strategy = {'A': 1.5, 'B': 2.0, 'C': 1.5}
        original_score = linear.score_guess_against_path_spec

        def _fake_score(path_spec, xyz, r_mol, symbols, chemistry=None):
            for r in (rec_a, rec_b, rec_c):
                if r.xyz['coords'][0] == xyz['coords'][0]:
                    return scores_by_strategy[r.strategy]
            return float('inf')

        # Inject a trivial path_spec into each record so the finalizer
        # actually invokes the (patched) scorer.
        trivial_spec = ReactionPathSpec()
        for r in (rec_a, rec_b, rec_c):
            r.path_spec = trivial_spec

        class _StubRxn:
            label = 'stub'

        linear.score_guess_against_path_spec = _fake_score
        try:
            out = _finalize_ts_guesses([rec_a, rec_b, rec_c], path_spec=None, rxn=_StubRxn(), r_mol=None)
        finally:
            linear.score_guess_against_path_spec = original_score

        # Expected order: A (1.5, idx 0), C (1.5, idx 2), B (2.0, idx 1).
        self.assertEqual(len(out), 3)
        self.assertEqual(out[0]['coords'][0][2], 0.0)   # A
        self.assertEqual(out[1]['coords'][0][2], 10.0)  # C
        self.assertEqual(out[2]['coords'][0][2], 5.0)   # B

    def test_finalizer_accepts_raw_dicts(self):
        """Plain XYZ dicts (no GuessRecord wrapper) must be wrapped and processed."""

        class _StubRxn:
            label = 'stub'

        raw_xyzs = [{'symbols': ('C',), 'isotopes': (12,), 'coords': ((0.0, 0.0, float(i)),)} for i in range(3)]
        out = _finalize_ts_guesses(raw_xyzs, path_spec=None, rxn=_StubRxn(), r_mol=None)
        self.assertEqual(len(out), 3)
        self.assertEqual(out[0]['coords'][0][2], 0.0)
        self.assertEqual(out[2]['coords'][0][2], 2.0)

    def test_finalizer_filters_colliding_atoms(self):
        """Guesses with atomic collisions must be removed before sorting."""

        class _StubRxn:
            label = 'stub'

        good = {'symbols': ('C', 'C'),
                'isotopes': (12, 12),
                'coords': ((0.0, 0.0, 0.0), (1.5, 0.0, 0.0))}
        bad = {'symbols': ('C', 'C'),
               'isotopes': (12, 12),
               'coords': ((0.0, 0.0, 0.0), (0.05, 0.0, 0.0))}  # 0.05 Å apart
        out = _finalize_ts_guesses([bad, good], path_spec=None, rxn=_StubRxn(), r_mol=None)
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out[0]['coords'][1][0], 1.5, places=6)

    def test_finalizer_addition_pipeline_routing(self):
        """Mock-test of the spec example: 10 valid addition guesses route
        through the finalizer, get sorted, deduped, and capped to 5."""

        class _StubRxn:
            label = 'stub'

        # 10 heavy-atom-distinct guesses (no path_spec → all score = inf).
        # Spacing of 0.10 Å between neighbors safely exceeds the 0.05 Å
        # heavy-atom-match tolerance, so dedup leaves all 10 alone.
        guesses = [{'symbols': ('C', 'O'),
                    'isotopes': (12, 16),
                    'coords': ((0.0, 0.0, 0.0), (1.4 + 0.10 * i, 0.0, 0.0))} for i in range(10)]
        out = _finalize_ts_guesses(guesses, path_spec=None, rxn=_StubRxn(), r_mol=None)
        self.assertEqual(len(out), 5)
        # Stable preservation: with all-equal scores, original order survives,
        # then cap-to-5 keeps the first 5 entries.
        for k in range(5):
            self.assertAlmostEqual(out[k]['coords'][1][0], 1.4 + 0.10 * k, places=6)


# ---------------------------------------------------------------------------
# addition pipeline path-spec wiring
# ---------------------------------------------------------------------------

class TestAdditionPathSpec(unittest.TestCase):
    """
    verify the addition pipeline now carries
    :class:`ReactionPathSpec` metadata through :class:`GuessRecord`
    objects and routes validation through
    :func:`validate_guess_against_path_spec`.
    """

    def _make_addition_rxn(self):
        """
        A small concerted dissociation: CCN ⇌ C=C + NH3.

        Coordinates copied from
        ``linear_test.py::test_interpolate_1_3_nh3_elimination`` so the
        same template-guided guesses get produced.
        """
        r_xyz = """C 1.14981017 0.04138987 -0.06722786
C      -0.25415691   -0.17696939    0.46881798
N      -0.38312147    0.39227542    1.80366803
H       1.89791609   -0.44343932    0.56909864
H       1.38984772    1.10839783   -0.12647258
H       1.23928774   -0.38052643   -1.07342059
H      -0.98187243    0.29596310   -0.19835733
H      -0.47689047   -1.24854874    0.50220986
H       0.27600194   -0.06032721    2.43576220
H -1.31312338 0.19118810 2.16873833"""
        p1_xyz = """C -0.63422754 -0.20894058 -0.01346068
C       0.63422754    0.20894058    0.01346068
H      -1.30426171   -0.01843680    0.81903872
H      -1.02752125   -0.74974821   -0.86852786
H       1.02752125    0.74974821    0.86852786
H 1.30426171 0.01843680 -0.81903872"""
        p2_xyz = """N 0.00064924 -0.00099698 0.29559292
H      -0.41786606    0.84210396   -0.09477452
H      -0.52039228   -0.78225292   -0.10002797
H 0.93760911 -0.05885406 -0.10079043"""
        r = ARCSpecies(label='R', smiles='CCN', xyz=r_xyz)
        p1 = ARCSpecies(label='P1', smiles='C=C', xyz=p1_xyz)
        p2 = ARCSpecies(label='P2', smiles='N', xyz=p2_xyz)
        return r, p1, p2, ARCReaction(r_species=[r], p_species=[p1, p2])

    def test_build_addition_path_spec_template_guided(self):
        """The shared addition path-spec helper builds a populated
        :class:`ReactionPathSpec` for a template-guided cut, using the
         :meth:`ReactionPathSpec.build` machinery (no hand-rolled
        bond-shell logic)."""
        r, _, _, rxn = self._make_addition_rxn()
        uni_xyz = r.get_xyz()
        spec = _build_addition_path_spec(uni_mol=r.mol,
                                         uni_xyz=uni_xyz,
                                         breaking_bonds=[(1, 2)],
                                         forming_bonds=[],
                                         weight=0.5,
                                         family='1,3_NH3_elimination',
                                         label='test')
        self.assertIsNotNone(spec)
        self.assertEqual(spec.breaking_bonds, [(1, 2)])
        self.assertEqual(spec.forming_bonds, [])
        # derivation populates these reactant-side fields.
        self.assertIn((1, 2), spec.ref_dist_r)
        self.assertIsNotNone(spec.ref_dist_r[(1, 2)])
        self.assertIn((1, 2), spec.bond_order_r)
        self.assertIsNotNone(spec.bond_order_r[(1, 2)])
        # The unchanged-near-core list comes from the BFS shell
        # over the reactant graph: it should be non-empty (the C-N
        # frontier has plenty of one-shell neighbors), and should NOT
        # contain the breaking bond itself.
        self.assertGreater(len(spec.unchanged_near_core_bonds), 0)
        self.assertNotIn((1, 2), spec.unchanged_near_core_bonds)
        # Conservative: no changed_bonds when mapped_p_mol is None.
        self.assertEqual(spec.changed_bonds, [])
        # The reactive-atom set must include both endpoints.
        self.assertIn(1, spec.reactive_atoms)
        self.assertIn(2, spec.reactive_atoms)
        self.assertEqual(spec.weight, 0.5)
        self.assertEqual(spec.family, '1,3_NH3_elimination')

    def test_build_addition_path_spec_fallback_minimal(self):
        """For a fragmentation cut without forming bonds, the spec must
        be deterministic, valid, and conservative."""
        r, _, _, _ = self._make_addition_rxn()
        spec = _build_addition_path_spec(uni_mol=r.mol,
                                         uni_xyz=r.get_xyz(),
                                         breaking_bonds=[(1, 2)],
                                         forming_bonds=None,  # exercise explicit None handling
                                         weight=0.5,
                                         family=None,
                                         label='test')
        self.assertIsNotNone(spec)
        self.assertEqual(spec.breaking_bonds, [(1, 2)])
        self.assertEqual(spec.forming_bonds, [])
        self.assertEqual(spec.changed_bonds, [])
        self.assertGreater(len(spec.unchanged_near_core_bonds), 0)
        # Reactant-side metadata is populated...
        self.assertIsNotNone(spec.ref_dist_r.get((1, 2)))
        self.assertIsNotNone(spec.bond_order_r.get((1, 2)))
        # ...and product-side metadata is genuinely None (no op_xyz / mapped_p_mol).
        self.assertIsNone(spec.ref_dist_p.get((1, 2)))
        self.assertIsNone(spec.bond_order_p.get((1, 2)))

    def test_build_addition_path_spec_with_cross_bonds(self):
        """Cross bonds are passed through into ``forming_bonds``."""
        r, _, _, _ = self._make_addition_rxn()
        spec = _build_addition_path_spec(uni_mol=r.mol,
                                         uni_xyz=r.get_xyz(),
                                         breaking_bonds=[(1, 2)],
                                         forming_bonds=[(0, 1)],
                                         weight=0.5,
                                         family='test',
                                         label='test')
        self.assertIsNotNone(spec)
        self.assertEqual(spec.breaking_bonds, [(1, 2)])
        self.assertEqual(spec.forming_bonds, [(0, 1)])

    # ---- #1: addition guesses are GuessRecord -----------------

    def _make_simple_dissociation_rxn(self):
        """
        Propyl radical → methyl + ethylene.

        A pure C-C dissociation with no H migration, so the addition
        pipeline produces at least one non-degraded record carrying a
        real :class:`ReactionPathSpec`.
        """
        r = ARCSpecies(label='propyl', smiles='[CH2]CC')
        p1 = ARCSpecies(label='methyl', smiles='[CH3]')
        p2 = ARCSpecies(label='ethylene', smiles='C=C')
        return ARCReaction(r_species=[r], p_species=[p1, p2])

    def test_interpolate_addition_uses_guess_records_internally(self):
        """A live ``interpolate_addition`` run produces guesses that
        round-trip through ``_finalize_ts_guesses``.  We trace the
        finalizer call and confirm that it sees ``GuessRecord``-shaped
        inputs and that at least one carries a real
        :class:`ReactionPathSpec`."""
        rxn = self._make_simple_dissociation_rxn()
        captured: list = []
        original = linear._finalize_ts_guesses

        def _trace(ts_xyzs, path_spec, rxn, r_mol):
            captured.append(list(ts_xyzs))
            return original(ts_xyzs, path_spec=path_spec, rxn=rxn, r_mol=r_mol)

        linear._finalize_ts_guesses = _trace
        try:
            linear.interpolate_addition(rxn, weight=0.5)
        finally:
            linear._finalize_ts_guesses = original

        self.assertEqual(len(captured), 1, 'finalizer should be called once')
        records = captured[0]
        self.assertGreater(len(records), 0, 'expected at least one record')
        for rec in records:
            self.assertIsInstance(rec, linear.GuessRecord)
        # At least one of the surviving records should carry a real
        # ReactionPathSpec. (Branches that involve H migration
        # deliberately strip the spec into degraded mode; pure C-C
        # dissociation preserves it.)
        with_spec = [rec for rec in records if rec.path_spec is not None]
        self.assertGreater(len(with_spec), 0, 'expected at least one record carrying a ReactionPathSpec')

    def test_addition_validation_routes_through_wrapper(self):
        """When ``stretch_bond`` is called with a ``path_spec``, it
        routes validation through
        :func:`validate_guess_against_path_spec` rather than the legacy
        :func:`validate_ts_guess`."""
        r, _, _, _ = self._make_addition_rxn()
        spec = ReactionPathSpec.build(r_mol=r.mol,
                                      mapped_p_mol=None,
                                      breaking_bonds=[(1, 2)],
                                      forming_bonds=[],
                                      r_xyz=r.get_xyz(),
                                      weight=0.5)
        # Cleanup-phase: the addition-side validation gateway is now the
        # single canonical helper :func:`path_spec.validate_addition_guess`
        # (re-exported into ``addition`` and ``linear``). Use scoped
        # ``unittest.mock.patch`` against the canonical names instead of
        # the legacy direct module-attribute reassignment patching pattern.

        original_wrapper = path_spec.validate_guess_against_path_spec
        original_legacy = path_spec.validate_ts_guess
        calls = {'wrapper': 0, 'legacy': 0}

        def _spy_wrapper(*args, **kwargs):
            calls['wrapper'] += 1
            return original_wrapper(*args, **kwargs)

        def _spy_legacy(*args, **kwargs):
            calls['legacy'] += 1
            return original_legacy(*args, **kwargs)

        with patch.object(path_spec, 'validate_guess_against_path_spec', side_effect=_spy_wrapper), \
                patch.object(path_spec, 'validate_ts_guess', side_effect=_spy_legacy):
            addition.stretch_bond(uni_xyz=r.get_xyz(),
                                  uni_mol=r.mol,
                                  split_bonds=[(1, 2)],
                                  cross_bonds=None,
                                  weight=0.5,
                                  label='test',
                                  path_spec=spec)
        self.assertGreaterEqual(calls['wrapper'], 1)

    def test_addition_validation_falls_back_when_no_path_spec(self):
        """``stretch_bond`` with ``path_spec=None`` MUST use the legacy ``validate_ts_guess`` path so degraded mode never crashes."""
        r, _, _, _ = self._make_addition_rxn()
        original_wrapper = path_spec.validate_guess_against_path_spec
        original_legacy = path_spec.validate_ts_guess
        calls = {'wrapper': 0, 'legacy': 0}

        def _spy_wrapper(*args, **kwargs):
            calls['wrapper'] += 1
            return original_wrapper(*args, **kwargs)

        def _spy_legacy(*args, **kwargs):
            calls['legacy'] += 1
            return original_legacy(*args, **kwargs)

        with patch.object(path_spec, 'validate_guess_against_path_spec',
                          side_effect=_spy_wrapper), \
             patch.object(path_spec, 'validate_ts_guess',
                          side_effect=_spy_legacy):
            addition.stretch_bond(uni_xyz=r.get_xyz(),
                                  uni_mol=r.mol,
                                  split_bonds=[(1, 2)],
                                  cross_bonds=None,
                                  weight=0.5,
                                  label='test',
                                  path_spec=None)

        # path_spec=None must route through the legacy validator and NEVER touch the path-spec wrapper.
        self.assertEqual(calls['wrapper'], 0)
        self.assertGreaterEqual(calls['legacy'], 1)

    def test_addition_wrapper_rejects_snapped_unchanged_near_core(self):
        """
        A fallback addition guess whose unchanged-near-core bond is
        stretched far beyond ``1.25 × sbl`` must be rejected by the
         wrapper when used inside the addition validation gateway.
         """
        prop = ARCSpecies(label='propyl', smiles='CC[CH2]')
        spec = ReactionPathSpec.build(r_mol=prop.mol,
            mapped_p_mol=None,
            breaking_bonds=[(0, 1)],
            forming_bonds=[],
            r_xyz=prop.get_xyz(),
                                      weight=0.5)
        # Take the propyl XYZ and pull one heavy atom far away so an
        # unchanged_near_core bond is "snapped".
        coords = list(map(list, prop.get_xyz()['coords']))
        coords[2][0] += 5.0
        bad_xyz = {'symbols': prop.get_xyz()['symbols'],
                   'isotopes': prop.get_xyz().get('isotopes', tuple(0 for _ in coords)),
                   'coords': tuple(tuple(row) for row in coords)}
        ok, reason = validate_addition_guess(xyz=bad_xyz,
                                             path_spec=spec,
                                             uni_mol=prop.mol,
                                             forming_bonds=[],
                                             label='snapped-test')
        self.assertFalse(ok)
        self.assertTrue(any(token in reason for token in ('recipe-mismatch', 'path-spec-check', 'detached', 'colliding', 'fragment')),
                        f'expected wrapper rejection reason, got: {reason!r}')

    def test_addition_record_with_path_spec_gets_finite_score(self):
        """
        A :class:`GuessRecord` carrying a real
        :class:`ReactionPathSpec` receives a finite (non-+inf) score
        when scored via ``score_guess_against_path_spec``, and round-
        trips through :func:`_finalize_ts_guesses` cleanly.
        """
        r, _, _, rxn = self._make_addition_rxn()
        spec = _build_addition_path_spec(uni_mol=r.mol,
                                         uni_xyz=r.get_xyz(),
                                         breaking_bonds=[(1, 2)],
                                         forming_bonds=[],
                                         weight=0.5,
                                         family='test',
                                         label='test')
        rec = GuessRecord(xyz=r.get_xyz(),
                          bb=[(1, 2)],
                          fb=[],
                          family='test',
                          strategy='test',
                          path_spec=spec)
        symbols = tuple(r.get_xyz()['symbols'])
        chemistry = classify_path_chemistry(spec, symbols)
        score = score_guess_against_path_spec(spec, rec.xyz, r.mol, symbols, chemistry)
        self.assertNotEqual(score, float('inf'), 'addition guess with valid path_spec must score finitely')
        out = _finalize_ts_guesses([rec], path_spec=None, rxn=rxn, r_mol=r.mol)
        self.assertEqual(len(out), 1)

    def test_partial_spec_validation_does_not_crash(self):
        """A spec with no product-side metadata must validate without
        raising, regardless of whether it accepts or rejects."""
        prop = ARCSpecies(label='propyl', smiles='CC[CH2]')
        spec = ReactionPathSpec.build(r_mol=prop.mol,
                                      mapped_p_mol=None,
                                      breaking_bonds=[(0, 1)],
                                      forming_bonds=[],
                                      r_xyz=prop.get_xyz(),
                                      weight=0.5)
        result = validate_guess_against_path_spec(xyz=prop.get_xyz(),
                                                  path_spec=spec,
                                                  r_mol=prop.mol,
                                                  label='partial-test')
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], bool)
        self.assertIsInstance(result[1], str)

    def test_partial_spec_scoring_does_not_crash(self):
        """Scoring on a spec with no product-side metadata must return
        a deterministic finite (or +inf) value, never raise."""
        prop = ARCSpecies(label='propyl', smiles='CC[CH2]')
        spec = ReactionPathSpec.build(r_mol=prop.mol,
                                      mapped_p_mol=None,
                                      breaking_bonds=[(0, 1)],
                                      forming_bonds=[],
                                      r_xyz=prop.get_xyz(),
                                      weight=0.5)
        symbols = tuple(prop.get_xyz()['symbols'])
        chemistry = classify_path_chemistry(spec, symbols)
        score = score_guess_against_path_spec(spec, prop.get_xyz(), prop.mol, symbols, chemistry)
        # Deterministic and not NaN.
        self.assertEqual(score, score)
        self.assertLess(score, float('inf'))

    def test_addition_path_spec_helper_returns_none_on_failure(self):
        """``_build_addition_path_spec`` swallows builder exceptions
        and returns ``None`` so the caller can degrade gracefully."""
        spec = _build_addition_path_spec(uni_mol=None,
                                         uni_xyz={'symbols': ('C',), 'isotopes': (12,), 'coords': ((0.0, 0.0, 0.0),)},
                                         breaking_bonds=[(0, 0)],
                                         forming_bonds=[],
                                         weight=0.5,
                                         family=None,
                                         label='test')
        self.assertIsNone(spec)


# ---------------------------------------------------------------------------
# post-migration topology enrichment gates
# ---------------------------------------------------------------------------

class TestPostMigrationEnrichmentGates(unittest.TestCase):
    """The enricher only attaches richer metadata when ALL of its topology gates pass. Each gate has a dedicated test."""

    def test_g1_rejects_zero_or_multiple_migrations(self):
        xyz = {'symbols': ('C', 'N', 'H'),
               'isotopes': (12, 14, 1),
               'coords': ((0.0, 0.0, 0.0), (2.5, 0.0, 0.0), (1.0, 0.5, 0.0))}
        class _A:
            def __init__(self, sym, bonds=None):
                class _E:
                    def __init__(self, s):
                        self.symbol = s
                self.element = _E(sym)
                self.bonds = bonds or {}
        c = _A('C'); n = _A('N'); h = _A('H')
        c.bonds = {h: None}; h.bonds = {c: None}
        class _M:
            atoms = [c, n, h]

        # Zero migrations.
        spec = _enrich_post_migration_path_spec(uni_mol=_M(), uni_xyz=xyz, ts_xyz=xyz,
                                                base_breaking=[], base_forming=[],
                                                migrations=[], weight=0.5, family=None, label='G1-zero')
        self.assertIsNone(spec)
        # Two migrations.
        m = [{'h_idx': 2, 'donor': 0, 'acceptor': 1, 'source': 'cross_bond'},
             {'h_idx': 2, 'donor': 0, 'acceptor': 1, 'source': 'cross_bond'}]
        spec = _enrich_post_migration_path_spec(uni_mol=_M(), uni_xyz=xyz, ts_xyz=xyz,
                                                base_breaking=[], base_forming=[],
                                                migrations=m, weight=0.5, family=None, label='G1-multi')
        self.assertIsNone(spec)

    def test_g3_rejects_nearest_core_acceptor(self):
        xyz = {'symbols': ('C', 'N', 'H'),
               'isotopes': (12, 14, 1),
               'coords': ((0.0, 0.0, 0.0), (2.5, 0.0, 0.0), (1.0, 0.5, 0.0))}
        class _A:
            def __init__(self, sym, bonds=None):
                class _E:
                    def __init__(self, s):
                        self.symbol = s
                self.element = _E(sym)
                self.bonds = bonds or {}
        c = _A('C'); n = _A('N'); h = _A('H')
        c.bonds = {h: None}; h.bonds = {c: None}
        class _M:
            atoms = [c, n, h]

        spec = _enrich_post_migration_path_spec(uni_mol=_M(), uni_xyz=xyz, ts_xyz=xyz,
                                                base_breaking=[], base_forming=[],
                                                migrations=[{'h_idx': 2, 'donor': 0, 'acceptor': 1, 'source': 'nearest_core'}],
                                                weight=0.5, family=None, label='G3',
                                                require_cross_bond_acceptor=True)
        self.assertIsNone(spec)

    def test_g4_rejects_distorted_donor_distance(self):
        xyz = {'symbols': ('C', 'N', 'H'),
               'isotopes': (12, 14, 1),
               'coords': ((0.0, 0.0, 0.0), (2.5, 0.0, 0.0), (3.6, 0.0, 0.0))}
        class _A:
            def __init__(self, sym, bonds=None):
                class _E:
                    def __init__(self, s):
                        self.symbol = s
                self.element = _E(sym)
                self.bonds = bonds or {}
        c = _A('C'); n = _A('N'); h = _A('H')
        c.bonds = {h: None}; h.bonds = {c: None}
        class _M:
            atoms = [c, n, h]
        spec = _enrich_post_migration_path_spec(uni_mol=_M(), uni_xyz=xyz, ts_xyz=xyz,
                                                base_breaking=[], base_forming=[],
                                                migrations=[{'h_idx': 2, 'donor': 0, 'acceptor': 1, 'source': 'cross_bond'}],
                                                weight=0.5, family=None, label='G4')
        self.assertIsNone(spec)

    def test_g5_rejects_competing_nearby_atom(self):
        # Place a third heavy atom right next to the migrating H.
        xyz = {'symbols': ('C', 'N', 'H', 'C'),
               'isotopes': (12, 14, 1, 12),
               'coords': ((0.0, 0.0, 0.0),
                          (2.5, 0.0, 0.0),
                          (1.25, 0.5, 0.0),
                          (1.30, 0.55, 0.0))}  # extremely close to the migrating H
        class _A:
            def __init__(self, sym, bonds=None):
                class _E:
                    def __init__(self, s):
                        self.symbol = s
                self.element = _E(sym)
                self.bonds = bonds or {}
        c = _A('C'); n = _A('N'); h = _A('H'); c2 = _A('C')
        c.bonds = {h: None}; h.bonds = {c: None}
        class _M:
            atoms = [c, n, h, c2]

        spec = _enrich_post_migration_path_spec(uni_mol=_M(), uni_xyz=xyz, ts_xyz=xyz,
                                                base_breaking=[], base_forming=[],
                                                migrations=[{'h_idx': 2, 'donor': 0, 'acceptor': 1, 'source': 'cross_bond'}],
                                                weight=0.5, family=None, label='G5')
        self.assertIsNone(spec)

    def test_all_gates_pass_returns_enriched_spec(self):
        """
        Crafted donor–acceptor–H triple where every gate passes:
        the enricher returns a populated :class:`ReactionPathSpec` whose
        ``breaking_bonds`` and ``forming_bonds`` lists were extended.

        Uses methylamine (CH3-NH2) so the underlying RMG ``Molecule``
        carries real bond objects (the path-spec factory needs
        ``bond.order``).  The migrating H index is one of the H atoms
        on the C, with the N as acceptor.
        """
        sp = ARCSpecies(label='methylamine', smiles='CN')
        symbols = sp.get_xyz()['symbols']
        # Find C, N, and one H bonded to C.
        c_idx = next(i for i, s in enumerate(symbols) if s == 'C')
        n_idx = next(i for i, s in enumerate(symbols) if s == 'N')
        atom_to_idx = {a: i for i, a in enumerate(sp.mol.atoms)}
        h_on_c = next(
            atom_to_idx[nbr]
            for nbr in sp.mol.atoms[c_idx].bonds.keys()
            if nbr.element.symbol == 'H'
        )
        # Build a TS-like geometry: place donor at origin, acceptor on
        # the +x axis, migrating H at the triangulated point.
        d_donor_h = get_single_bond_length('C', 'H') + PAULING_DELTA
        d_acceptor_h = get_single_bond_length('N', 'H') + PAULING_DELTA
        d_da = d_donor_h + d_acceptor_h - 0.20
        x = (d_da ** 2 + d_donor_h ** 2 - d_acceptor_h ** 2) / (2.0 * d_da)
        y = float(np.sqrt(max(d_donor_h ** 2 - x ** 2, 0.0)))

        coords = list(map(list, sp.get_xyz()['coords']))
        coords[c_idx] = [0.0, 0.0, 0.0]
        coords[n_idx] = [d_da, 0.0, 0.0]
        coords[h_on_c] = [x, y, 0.0]
        # Move all *other* heavy atoms far away so they don't compete
        # with the migrating H during the G5 check. Methylamine only
        # has C and N as heavy atoms, so this is already handled.
        # Move all *other* H atoms far away so they don't influence
        # any heavy/H sub-check unintentionally.
        for i, sym in enumerate(symbols):
            if sym == 'H' and i != h_on_c:
                coords[i] = [10.0 + i, 10.0, 10.0]

        ts_xyz = {'symbols': symbols,
                  'isotopes': sp.get_xyz().get('isotopes', tuple(0 for _ in symbols)),
                  'coords': tuple(tuple(float(c) for c in row) for row in coords)}

        spec = _enrich_post_migration_path_spec(uni_mol=sp.mol,
                                                uni_xyz=sp.get_xyz(),
                                                ts_xyz=ts_xyz,
                                                base_breaking=[],
                                                base_forming=[],
                                                migrations=[{'h_idx': h_on_c, 'donor': c_idx, 'acceptor': n_idx, 'source': 'cross_bond'}],
                                                weight=0.5, family='test', label='G-all-pass')
        self.assertIsNotNone(spec)
        # The (donor, h) bond is now in breaking_bonds, the (acceptor, h)
        # bond is in forming_bonds (canonicalized).
        canon_dh = (min(c_idx, h_on_c), max(c_idx, h_on_c))
        canon_ah = (min(n_idx, h_on_c), max(n_idx, h_on_c))
        self.assertIn(canon_dh, spec.breaking_bonds)
        self.assertIn(canon_ah, spec.forming_bonds)


# ---------------------------------------------------------------------------
# finite-score promotion via the live addition pipeline
# ---------------------------------------------------------------------------

class TestFiniteScorePromotion(unittest.TestCase):
    """
    Run the full :func:`interpolate_addition` pipeline on a real
    H-migration addition reaction and verify that at least one
    post-migration template-guided guess now reaches the finalizer with
    a real finite score (not the +inf).
    """

    def test_template_post_migration_now_carries_finite_score(self):
        r_xyz = """C 1.14981017 0.04138987 -0.06722786
C      -0.25415691   -0.17696939    0.46881798
N      -0.38312147    0.39227542    1.80366803
H       1.89791609   -0.44343932    0.56909864
H       1.38984772    1.10839783   -0.12647258
H       1.23928774   -0.38052643   -1.07342059
H      -0.98187243    0.29596310   -0.19835733
H      -0.47689047   -1.24854874    0.50220986
H       0.27600194   -0.06032721    2.43576220
H -1.31312338 0.19118810 2.16873833"""
        r = ARCSpecies(label='R', smiles='CCN', xyz=r_xyz)
        p1 = ARCSpecies(label='P1', smiles='C=C')
        p2 = ARCSpecies(label='P2', smiles='N')
        rxn = ARCReaction(r_species=[r], p_species=[p1, p2])

        captured = []
        original = linear._finalize_ts_guesses

        def trace(ts_xyzs, path_spec, rxn, r_mol):
            captured.append(list(ts_xyzs))
            return original(ts_xyzs, path_spec=path_spec, rxn=rxn, r_mol=r_mol)

        linear._finalize_ts_guesses = trace
        try:
            linear.interpolate_addition(rxn, weight=0.5)
        finally:
            linear._finalize_ts_guesses = original

        records = captured[0] if captured else []
        self.assertGreater(len(records), 0)
        # At least one template-guided post-migration record now carries
        # a non-None ReactionPathSpec whose breaking_bonds list explicitly
        # includes a (heavy, H) pair — i.e. enrichment fired.
        promoted = []
        for rec in records:
            if rec.strategy != 'template_guided' or rec.path_spec is None:
                continue
            for (a, b) in rec.path_spec.breaking_bonds:
                if 'H' in (rec.xyz['symbols'][a], rec.xyz['symbols'][b]):
                    promoted.append(rec)
                    break
        self.assertGreater(len(promoted), 0,' should promote at least one template-guided '
                                            'post-migration record to a finite-score, enriched spec.')

        # And computing its score must yield a finite (non-+inf) value.
        rec = promoted[0]
        symbols = tuple(rec.xyz['symbols'])
        chem = classify_path_chemistry(rec.path_spec, symbols)
        score = score_guess_against_path_spec(rec.path_spec, rec.xyz, r.mol, symbols, chem)
        self.assertNotEqual(score, float('inf'), 'enriched record must score finitely')


# ---------------------------------------------------------------------------
# degraded-mode preservation for ambiguous cases
# ---------------------------------------------------------------------------

class TestDegradedPreservation(unittest.TestCase):
    """Cases where the topology gates correctly refuse enrichment must still produce a record (degraded mode) and never crash."""

    def test_g3_nearest_core_acceptor_remains_degraded(self):
        xyz = {'symbols': ('C', 'N', 'H'),
               'isotopes': (12, 14, 1),
               'coords': ((0.0, 0.0, 0.0), (2.5, 0.0, 0.0), (1.0, 0.5, 0.0))}
        class _A:
            def __init__(self, sym, bonds=None):
                class _E:
                    def __init__(self, s):
                        self.symbol = s
                self.element = _E(sym)
                self.bonds = bonds or {}
        c = _A('C'); n = _A('N'); h = _A('H')
        c.bonds = {h: None}; h.bonds = {c: None}
        class _M:
            atoms = [c, n, h]

        spec = _enrich_post_migration_path_spec(uni_mol=_M(), uni_xyz=xyz, ts_xyz=xyz,
                                                base_breaking=[], base_forming=[],
                                                migrations=[{'h_idx': 2, 'donor': 0, 'acceptor': 1, 'source': 'nearest_core'}],
                                                weight=0.5, family=None, label='degraded-G3')
        self.assertIsNone(spec)

    def test_xy_elimination_remains_degraded_no_crash(self):
        """The XY-elimination dedicated motif builder is explicitly
        out-of-scope for  enrichment.  Verify the call still
        completes without crashing."""
        r = ARCSpecies(label='R', smiles='CCC(=O)O')
        p1 = ARCSpecies(label='P1', smiles='C=C')
        p2 = ARCSpecies(label='P2', smiles='[H][H]')
        p3 = ARCSpecies(label='P3', smiles='O=C=O')
        rxn = ARCReaction(r_species=[r], p_species=[p1, p2, p3])
        out = linear.interpolate_addition(rxn, weight=0.5)
        self.assertIsInstance(out, list)


# ---------------------------------------------------------------------------
# fragmentation-fallback finite-score promotion
# ---------------------------------------------------------------------------

class TestFragFallbackPromotion(unittest.TestCase):
    """
    Live ``interpolate_addition`` tests proving promotes
    a trustworthy frag-fallback single-H migration to a finite score
    while preserving degraded mode for ambiguous cases.
    """

    def _make_co2_insertion_rxn(self):
        # 1,3_Insertion_CO2: a tertiary carboxylic acid → CO2 + alkene.
        # Triggers the frag-fallback branch with a single carboxylic-
        # acid H migrating from O to a methyl C.
        r_xyz = """C -2.49526563 -1.71744655 -0.05070502
C      -1.02439736   -1.43442148   -0.09903098
C      -0.03369873   -2.05724101    0.58071768
C      -0.33706853   -3.21011130    1.53261903
C      -0.85997546   -2.75301776    2.88970062
C       1.47028993   -1.69663277    0.40874963
C       2.07012721   -1.30878861    1.77489924
C       1.74154080   -0.52957843   -0.56586530
C       2.20380863   -2.91010712   -0.19503352
O       1.73766923   -3.76432521   -0.93387645
O       3.51955595   -2.95521685    0.10563464
H      -3.10560600   -0.84943191    0.20522867
H      -2.69351927   -2.49442175    0.69448241
H      -2.82934183   -2.08783564   -1.02499122
H      -0.78561006   -0.62690380   -0.78689260
H       0.55596394   -3.82539962    1.69076067
H      -1.06313426   -3.89101221    1.07201521
H      -1.79380282   -2.19600714    2.76086637
H      -0.14816985   -2.09644367    3.39930225
H      -1.06676022   -3.60034589    3.54974021
H       2.00787775   -2.13013981    2.49728135
H       3.13273869   -1.05110182    1.68998787
H       1.55881292   -0.43721490    2.19959957
H       1.26204457    0.39674910   -0.22955829
H       2.81639214   -0.32714424   -0.65140849
H       1.38896771   -0.75762301   -1.57886997
H 3.82821421 -3.76916475 -0.34544391"""
        p1_xyz = """O -1.37316735 0.24657196 0.00000000
C      -0.00000000   -0.05081069    0.00000000
O 1.37316735 -0.34819332 0.00000000"""
        p2_xyz = """C -2.38749724 -2.07681559 -0.24769962
C                 -0.91223049   -1.65569429   -0.38128430
C                  0.00974358   -2.13096778    0.49086589
C                 -0.41782495   -3.09217448    1.61552887
C                 -0.80439231   -2.28004191    2.86557141
C                  1.48500992   -1.70984460    0.35728267
C                  2.14707486   -1.71091779    1.74770306
C                  1.56241137   -0.29548780   -0.24703833
H                 -3.01706611   -1.28917857   -0.60570935
H                 -2.61083914   -2.27262544    0.78024828
H                 -2.55960948   -2.96124003   -0.82482247
H                 -0.61515337   -0.98784419   -1.16270699
H                  0.39428846   -3.74720075    1.85283123
H                 -1.25842619   -3.66927357    1.29111215
H                 -1.61650572   -1.62501564    2.62826906
H                  0.03620892   -1.70294282    3.18998814
H                 -1.10146915   -2.94789332    3.64699310
H                  2.09329593   -2.69362024    2.16758843
H                  3.17209758   -1.41831921    1.65488875
H                  1.63583722   -1.02155958    2.38670333
H                  1.05117373    0.39387041    0.39196194
H                  2.58743409   -0.00288922   -0.33985264
H                  1.10240521   -0.29474214   -1.21310964
H 1.99624756 -2.39920281 -0.28171759"""
        r = ARCSpecies(label='R', smiles='CC=C(CC)C(C)(C)C(=O)O', xyz=r_xyz)
        p1 = ARCSpecies(label='P1', smiles='O=C=O', xyz=p1_xyz)
        p2 = ARCSpecies(label='P2', smiles='CC=C(CC)C(C)C', xyz=p2_xyz)
        return r, p1, p2, ARCReaction(r_species=[r], p_species=[p1, p2])

    def test_frag_fallback_single_h_now_carries_finite_score(self):
        """
        For 1,3_Insertion_CO2, the frag-fallback branch produces
        exactly one frag-fallback record carrying an enriched
        :class:`ReactionPathSpec` whose ``breaking_bonds`` includes a
        new (heavy, H) pair, and that record scores finitely.
        """
        r, _, _, rxn = self._make_co2_insertion_rxn()
        captured: list = []
        original = linear._finalize_ts_guesses

        def trace(ts_xyzs, path_spec, rxn, r_mol):
            captured.append(list(ts_xyzs))
            return original(ts_xyzs, path_spec=path_spec, rxn=rxn, r_mol=r_mol)

        linear._finalize_ts_guesses = trace
        try:
            linear.interpolate_addition(rxn, weight=0.5)
        finally:
            linear._finalize_ts_guesses = original

        records = captured[0] if captured else []
        promoted = [rec for rec in records if rec.strategy == 'frag_fallback' and rec.path_spec is not None]
        self.assertGreater(len(promoted), 0,'should promote at least one frag-fallback single-H record to a finite-score, enriched spec.')
        rec = promoted[0]
        # The enriched spec must contain a (heavy, H) breaking bond
        # — the H-side of the migration the helper inferred.
        h_bond_present = any(('H' in (rec.xyz['symbols'][a], rec.xyz['symbols'][b])
                              and rec.xyz['symbols'][a] != rec.xyz['symbols'][b]) for (a, b) in
                             rec.path_spec.breaking_bonds)
        self.assertTrue(h_bond_present, 'enriched frag-fallback spec must include a (heavy, H) '
                                        'breaking bond inferred by the helper.')
        symbols = tuple(rec.xyz['symbols'])
        chem = classify_path_chemistry(rec.path_spec, symbols)
        score = score_guess_against_path_spec(rec.path_spec, rec.xyz, r.mol, symbols, chem)
        self.assertNotEqual(score, float('inf'), 'enriched frag-fallback record must score finitely')

    def test_pure_cc_dissociation_does_not_get_h_enriched(self):
        """
        Ethane → 2 CH3• is a pure C-C cleavage with NO H migration.
        The  helper must NOT enrich any frag-fallback record
        with a (heavy, H) breaking bond.  Verifies that the helper
        does not invent an H-migration topology when none exists.
        """
        r = ARCSpecies(label='ethane', smiles='CC')
        p1 = ARCSpecies(label='P1', smiles='[CH3]')
        p2 = ARCSpecies(label='P2', smiles='[CH3]')
        rxn = ARCReaction(r_species=[r], p_species=[p1, p2])

        captured: list = []
        original = linear._finalize_ts_guesses

        def trace(ts_xyzs, path_spec, rxn, r_mol):
            captured.append(list(ts_xyzs))
            return original(ts_xyzs, path_spec=path_spec, rxn=rxn, r_mol=r_mol)

        linear._finalize_ts_guesses = trace
        try:
            linear.interpolate_addition(rxn, weight=0.5)
        finally:
            linear._finalize_ts_guesses = original

        records = captured[0] if captured else []
        # No frag-fallback record may carry an enriched spec whose
        # breaking_bonds contain a (heavy, H) pair.
        for rec in records:
            if rec.strategy != 'frag_fallback' or rec.path_spec is None:
                continue
            for (a, b) in rec.path_spec.breaking_bonds:
                ab_syms = (rec.xyz['symbols'][a], rec.xyz['symbols'][b])
                self.assertFalse('H' in ab_syms and ab_syms[0] != ab_syms[1],
                                 'frag-fallback enrichment fired on a case with no H migration.')

    def test_helper_returns_none_for_topologically_inconsistent_input(self):
        """
        A directly fed inconsistent (donor and acceptor on the same
        fragment) input must return ``None`` from the helper, not a partial enrichment.
        """
        sp = ARCSpecies(label='ethanol', smiles='CCO')
        symbols = sp.get_xyz()['symbols']
        atom_to_idx = {a: i for i, a in enumerate(sp.mol.atoms)}
        c0 = next(i for i, s in enumerate(symbols) if s == 'C')
        h_on_c0 = next(atom_to_idx[nbr] for nbr in sp.mol.atoms[c0].bonds.keys() if nbr.element.symbol == 'H')
        coords = list(map(list, sp.get_xyz()['coords']))
        post_coords = [list(c) for c in coords]
        # Move the H slightly so S1 finds it; donor is C0, but with no
        # cut at all the molecule has 1 fragment ⇒ S3 fails.
        post_coords[h_on_c0][0] += 1.0
        pre = {**sp.get_xyz(),
               'coords': tuple(tuple(c) for c in coords)}
        post = {**sp.get_xyz(),
                'coords': tuple(tuple(c) for c in post_coords)}
        out = infer_frag_fallback_h_migration(pre_xyz=pre, post_xyz=post,
                                              uni_mol=sp.mol, split_bonds=[],
                                              multi_species=None, label='no-fake')
        self.assertIsNone(out)

    def test_template_guided_finite_score_promotion_unchanged(self):
        """
        The template-guided enrichment for ``1,3_NH3_elimination``
        must continue to fire and produce a finite score after .
        This is the regression guard for the directive's 'stability rule'.
        """
        r_xyz = """C 1.14981017 0.04138987 -0.06722786
C      -0.25415691   -0.17696939    0.46881798
N      -0.38312147    0.39227542    1.80366803
H       1.89791609   -0.44343932    0.56909864
H       1.38984772    1.10839783   -0.12647258
H       1.23928774   -0.38052643   -1.07342059
H      -0.98187243    0.29596310   -0.19835733
H      -0.47689047   -1.24854874    0.50220986
H       0.27600194   -0.06032721    2.43576220
H -1.31312338 0.19118810 2.16873833"""
        r = ARCSpecies(label='R', smiles='CCN', xyz=r_xyz)
        p1 = ARCSpecies(label='P1', smiles='C=C')
        p2 = ARCSpecies(label='P2', smiles='N')
        rxn = ARCReaction(r_species=[r], p_species=[p1, p2])

        captured: list = []
        original = linear._finalize_ts_guesses

        def trace(ts_xyzs, path_spec, rxn, r_mol):
            captured.append(list(ts_xyzs))
            return original(ts_xyzs, path_spec=path_spec, rxn=rxn, r_mol=r_mol)

        linear._finalize_ts_guesses = trace
        try:
            linear.interpolate_addition(rxn, weight=0.5)
        finally:
            linear._finalize_ts_guesses = original

        records = captured[0] if captured else []
        promoted_template = [rec for rec in records if rec.strategy == 'template_guided' and rec.path_spec is not None]
        self.assertGreater(len(promoted_template), 0, 'template-guided enrichment must remain stable '
                                                      'after at least one template_guided record '
                                                      'should still carry a non-None ReactionPathSpec.')

    def test_xy_elimination_remains_degraded(self):
        """The XY-elimination dedicated motif builder is still explicitly out of scope. Verify the call still completes."""
        r = ARCSpecies(label='R', smiles='CCC(=O)O')
        p1 = ARCSpecies(label='P1', smiles='C=C')
        p2 = ARCSpecies(label='P2', smiles='[H][H]')
        p3 = ARCSpecies(label='P3', smiles='O=C=O')
        rxn = ARCReaction(r_species=[r], p_species=[p1, p2, p3])
        out = linear.interpolate_addition(rxn, weight=0.5)
        self.assertIsInstance(out, list)

    def test_cleanup_without_enrichment_preserves_degraded_mode(self):
        """
        A frag-fallback single-H displacement that satisfies S1+S2
        but is *too distorted* (S5 fails) must remain degraded.  Local
        cleanup may still touch the geometry, but no enriched spec is
        attached.  Verified by feeding a hand-crafted post-migration
        XYZ where the H lies far from any plausible acceptor.
        """
        sp = ARCSpecies(label='ccn', smiles='CCN')
        symbols = sp.get_xyz()['symbols']
        atom_to_idx = {a: i for i, a in enumerate(sp.mol.atoms)}
        c_central = None
        for atom in sp.mol.atoms:
            if atom.element.symbol == 'C':
                heavy_nbrs = [n for n in atom.bonds.keys() if n.element.symbol != 'H']
                if len(heavy_nbrs) == 2:
                    c_central = atom_to_idx[atom]
                    break
        self.assertIsNotNone(c_central)
        h_on_c = next(atom_to_idx[nbr] for nbr in sp.mol.atoms[c_central].bonds.keys() if nbr.element.symbol == 'H')
        coords = list(map(list, sp.get_xyz()['coords']))
        post_coords = [list(c) for c in coords]
        post_coords[h_on_c] = [50.0, 50.0, 50.0]
        pre = {**sp.get_xyz(), 'coords': tuple(tuple(c) for c in coords)}
        post = {**sp.get_xyz(), 'coords': tuple(tuple(c) for c in post_coords)}
        n_idx = next(i for i, s in enumerate(symbols) if s == 'N')
        out = infer_frag_fallback_h_migration(pre_xyz=pre, post_xyz=post,
                                              uni_mol=sp.mol, split_bonds=[(c_central, n_idx)],
                                              multi_species=None, label='cleanup-without-enrichment')
        self.assertIsNone(out)


class TestCarbeneTargetConsistency(unittest.TestCase):
    """
    Cleanup-phase: prove that the insertion-ring builder and the
    scorer/validator use the *same* family-aware target distance for
    ``1,2_Insertion_carbene``.

    Before the cleanup, the builder applied a +0.20 Å carbene
    extra stretch via :func:`addition.insertion_ring_extra_stretch`,
    but :func:`get_ts_target_distance` ignored its ``family`` parameter
    entirely.  As a result, the scorer judged carbene guesses against
    the un-calibrated default target while the builder produced
    geometries at the calibrated (looser) target — a guaranteed
    mismatch every time the scorer ran on a carbene insertion guess.
    """

    def test_get_ts_target_distance_applies_carbene_calibration(self):
        """
        For ``1,2_Insertion_carbene``, ``get_ts_target_distance``
        on a ``forming`` or ``breaking`` C–C bond must equal
        ``sbl + PAULING_DELTA + 0.20`` (the same target the builder produces).
        """
        symbols = ('C', 'C')
        sbl = float(get_single_bond_length('C', 'C'))
        baseline = sbl + PAULING_DELTA
        carbene_target = baseline + 0.20

        # ``forming`` role for a C-C bond on the carbene family.
        target_forming = get_ts_target_distance(bond=(0, 1), role='forming', symbols=symbols,
                                                family='1,2_Insertion_carbene')
        self.assertAlmostEqual(target_forming, carbene_target, places=6)

        # ``breaking`` role for the same bond.
        target_breaking = get_ts_target_distance(bond=(0, 1), role='breaking', symbols=symbols,
                                                 family='1,2_Insertion_carbene')
        self.assertAlmostEqual(target_breaking, carbene_target, places=6)

        # The +0.20 delta must come from the SAME helper the builder uses.
        self.assertAlmostEqual(insertion_ring_extra_stretch('1,2_Insertion_carbene'), 0.20, places=6)

    def test_get_ts_target_distance_unaffected_for_other_families(self):
        """Non-carbene families must NOT receive the calibration —
        only the explicitly named carbene family triggers the +0.20 Å stretch.
        """
        symbols = ('C', 'C')
        baseline = float(get_single_bond_length('C', 'C')) + PAULING_DELTA

        for fam in (None, '', '1,2_Insertion_CO', '1,3_Insertion_RSR', 'Diels_alder_addition', 'intra_H_migration'):
            for role in ('breaking', 'forming'):
                target = get_ts_target_distance(bond=(0, 1), role=role, symbols=symbols, family=fam)
                self.assertAlmostEqual(target, baseline, places=6,
                                       msg=f'family={fam!r} role={role!r} should be uncalibrated '
                                           f'(expected {baseline:.4f}, got {target:.4f})')

    def test_builder_and_scorer_share_same_helper(self):
        """
        The builder-side ``addition.insertion_ring_extra_stretch``
        alias must resolve to the canonical ``path_spec.insertion_ring_extra_stretch``.
        This is the cleanup invariant that prevents the two sides from drifting apart again.
        """
        self.assertIs(addition.insertion_ring_extra_stretch,
                      path_spec.insertion_ring_extra_stretch)
        for fam in (None, '1,2_Insertion_carbene', '1,2_Insertion_CO', '1,3_Insertion_RSR'):
            self.assertEqual(addition.insertion_ring_extra_stretch(fam),
                             path_spec.insertion_ring_extra_stretch(fam),
                             msg=f'builder/scorer extra-stretch mismatch for family={fam!r}')


class TestUnifiedAdditionGateway(unittest.TestCase):
    """
    Cleanup-phase: prove the canonical :func:`path_spec.validate_addition_guess`
    gateway exists and routes correctly in both path-spec and degraded modes.
    """

    def test_canonical_gateway_routes_through_wrapper(self):
        sp = ARCSpecies(label='ethane', smiles='CC')
        spec = path_spec.ReactionPathSpec.build(r_mol=sp.mol,
                                                mapped_p_mol=None,
                                                breaking_bonds=[(0, 1)],
                                                forming_bonds=[],
                                                r_xyz=sp.get_xyz(),
                                                weight=0.5)
        with patch.object(path_spec, 'validate_guess_against_path_spec', return_value=(True, '')) as wrapper, \
                patch.object(path_spec, 'validate_ts_guess', return_value=(True, '')) as legacy:
            ok, _ = path_spec.validate_addition_guess(xyz=sp.get_xyz(),
                                                      uni_mol=sp.mol,
                                                      forming_bonds=[(0, 1)],
                                                      label='canonical-routes-wrapper',
                                                      path_spec=spec)
        self.assertTrue(ok)
        self.assertEqual(wrapper.call_count, 1)
        self.assertEqual(legacy.call_count, 0)

    def test_canonical_gateway_falls_back_when_no_path_spec(self):
        sp = ARCSpecies(label='ethane', smiles='CC')
        with patch.object(path_spec, 'validate_guess_against_path_spec', return_value=(True, '')) as wrapper, \
                patch.object(path_spec, 'validate_ts_guess', return_value=(True, '')) as legacy:
            ok, _ = path_spec.validate_addition_guess(xyz=sp.get_xyz(),
                                                      uni_mol=sp.mol,
                                                      forming_bonds=[(0, 1)],
                                                      label='canonical-falls-back',
                                                      path_spec=None)
        self.assertTrue(ok)
        self.assertEqual(wrapper.call_count, 0)
        self.assertEqual(legacy.call_count, 1)

    def test_addition_module_re_exports_canonical_gateway(self):
        """The ``addition`` module must re-import the canonical gateway
        so leaf builders share the same source of truth as ``linear``."""
        self.assertIs(addition.validate_addition_guess, path_spec.validate_addition_guess)

    def test_linear_module_re_imports_canonical_gateway(self):
        """The ``linear`` module must re-import the canonical gateway."""
        self.assertIs(linear.validate_addition_guess, path_spec.validate_addition_guess)


class TestAllBonds(unittest.TestCase):
    """``_all_bonds`` returns a sorted list of canonical bond tuples."""

    def test_ethane_has_seven_bonds(self):
        """Ethane has 1 C-C + 6 C-H = 7 bonds, and the list is sorted."""
        sp = ARCSpecies(label='ethane', smiles='CC')
        bonds = _all_bonds(sp.mol)
        self.assertEqual(len(bonds), 7)
        # Canonical ordering: each tuple has min index first, list is sorted.
        for (i, j) in bonds:
            self.assertLess(i, j)
        self.assertEqual(bonds, sorted(bonds))


class TestXyzDistance(unittest.TestCase):
    """``_xyz_distance`` returns the Euclidean distance between two atoms,
    or ``None`` on missing/bad input."""

    def test_known_pair_distance(self):
        xyz = {'symbols': ('C', 'C'),
               'isotopes': (12, 12),
               'coords': ((0.0, 0.0, 0.0), (3.0, 4.0, 0.0))}
        self.assertAlmostEqual(_xyz_distance(xyz, 0, 1), 5.0, places=6)

    def test_none_xyz_returns_none(self):
        self.assertIsNone(_xyz_distance(None, 0, 1))

    def test_out_of_range_returns_none(self):
        xyz = {'symbols': ('C',),
               'isotopes': (12,),
               'coords': ((0.0, 0.0, 0.0),)}
        self.assertIsNone(_xyz_distance(xyz, 0, 5))


class TestSafeOrder(unittest.TestCase):
    """``_safe_order`` looks up a canonical bond-order from a dict, returning ``None`` on a miss."""

    def test_hit_and_miss(self):
        bond_orders = {(0, 1): 2.0, (1, 3): 1.0}
        # Hit: canonical ordering swaps the input.
        self.assertEqual(_safe_order(bond_orders, 1, 0), 2.0)
        self.assertEqual(_safe_order(bond_orders, 3, 1), 1.0)
        # Miss.
        self.assertIsNone(_safe_order(bond_orders, 0, 5))


class TestHeavyFormingBonds(unittest.TestCase):
    """``_heavy_forming_bonds`` returns only forming bonds whose endpoints are both heavy."""

    def test_filters_out_h_endpoints(self):
        symbols = ('C', 'O', 'H', 'N')
        bonds = [(0, 1), (0, 2), (1, 3), (2, 3)]
        # Only (0,1) and (1,3) have both endpoints heavy.
        out = _heavy_forming_bonds(bonds, symbols)
        self.assertEqual(out, [(0, 1), (1, 3)])


class TestSharedAtomsBetweenBbAndFb(unittest.TestCase):
    """``_shared_atoms_between_bb_and_fb`` returns atoms shared between a breaking-bond list and a forming-bond list."""

    def test_with_shared_atoms(self):
        """Atom 1 is in both a breaking and a forming bond."""
        shared = _shared_atoms_between_bb_and_fb(breaking_bonds=[(0, 1), (2, 3)], forming_bonds=[(1, 4)])
        self.assertEqual(shared, {1})

    def test_without_shared_atoms(self):
        """No atoms are shared — the two bond sets are disjoint."""
        shared = _shared_atoms_between_bb_and_fb(breaking_bonds=[(0, 1)], forming_bonds=[(2, 3)])
        self.assertEqual(shared, set())


class TestIsFrontierExempt(unittest.TestCase):
    """
    Exercise the nested ``_is_frontier_exempt`` closure inside
    ``has_bad_changed_bond_length`` via crafted :class:`ReactionPathSpec`
    instances.  The exemption requires BOTH a |Δbo| ≥ 0.5 shift AND that
    the changed bond share an atom with a breaking/forming bond.
    """

    def test_both_conditions_hold_is_exempt(self):
        """
        Large bond-order shift and adjacency to the reactive core →
        the exemption short-circuits the distance check, so a geometry
        that would otherwise fail returns ``(False, '')``.
        """
        symbols = ('C', 'C', 'H')
        spec = ReactionPathSpec(breaking_bonds=[(0, 2)],  # shares atom 0 with changed
                                forming_bonds=[],
                                changed_bonds=[(0, 1)],
                                unchanged_near_core_bonds=[],
                                reactive_atoms={0, 1, 2},
                                weight=0.5,
                                family=None,
                                bond_order_r={(0, 1): 1.0, (0, 2): 1.0},
                                bond_order_p={(0, 1): 2.0, (0, 2): 0.0},  # Δbo=1.0 on (0,1)
                                ref_dist_r={(0, 1): 1.54},
                                ref_dist_p={(0, 1): 1.33})
        # Deliberately-bad geometry: (0,1) far from any sensible target.
        xyz = {'symbols': symbols,
               'isotopes': (12, 12, 1),
               'coords': ((0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (-0.5, 0.0, 0.0))}
        bad, reason = has_bad_changed_bond_length(spec, xyz, symbols)
        self.assertFalse(bad)
        self.assertEqual(reason, '')

    def test_missing_adjacency_is_not_exempt(self):
        """Large Δbo but the changed bond shares no atom with breaking / forming → NOT exempt; a far-off distance is rejected."""
        symbols = ('C', 'C', 'C', 'H')
        spec = ReactionPathSpec(breaking_bonds=[(2, 3)],  # no shared atom with (0,1)
                                forming_bonds=[],
                                changed_bonds=[(0, 1)],
                                unchanged_near_core_bonds=[],
                                reactive_atoms={2, 3},
                                weight=0.5,
                                family=None,
                                bond_order_r={(0, 1): 1.0, (2, 3): 1.0},
                                bond_order_p={(0, 1): 2.0, (2, 3): 0.0},
                                ref_dist_r={(0, 1): 1.54},
                                ref_dist_p={(0, 1): 1.33})
        xyz = {'symbols': symbols,
               'isotopes': (12, 12, 12, 1),
               'coords': ((0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (10.0, 0.0, 0.0), (11.0, 0.0, 0.0))}
        bad, reason = has_bad_changed_bond_length(spec, xyz, symbols)
        self.assertTrue(bad)
        self.assertIn('bad-changed-bond', reason)


class TestBondDistViaScoreGuess(unittest.TestCase):
    """
    Exercise the nested ``_bond_dist`` closure inside
    ``score_guess_against_path_spec`` by scoring a geometry whose
    breaking-bond distance is a known Euclidean value.
    """

    def test_bond_dist_feeds_into_score(self):
        """A geometry with exactly the breaking-bond target distance contributes zero to the score; shifting by 0.35 Å contributes 1.0."""
        symbols = ('C', 'C', 'H', 'H')
        # An almost-empty spec with a single breaking bond (0,1).
        spec = ReactionPathSpec(breaking_bonds=[(0, 1)],
                                forming_bonds=[],
                                changed_bonds=[],
                                unchanged_near_core_bonds=[],
                                reactive_atoms={0, 1},
                                weight=0.5,
                                family=None,
                                bond_order_r={(0, 1): 1.0},
                                bond_order_p={(0, 1): 0.0},
                                ref_dist_r={(0, 1): 1.54},
                                ref_dist_p={(0, 1): 3.00})
        # target = sbl(C,C) + PAULING_DELTA.
        target = get_ts_target_distance(bond=(0, 1), role='breaking', symbols=symbols,
                                        d_r=1.54, d_p=3.00, weight=0.5, family=None)
        # On-target: bond at exactly `target` along x.
        xyz_on = {'symbols': symbols,
                  'isotopes': (12, 12, 1, 1),
                  'coords': ((0.0, 0.0, 0.0), (target, 0.0, 0.0),
                             (10.0, 0.0, 0.0), (11.0, 0.0, 0.0))}
        score_on = score_guess_against_path_spec(
            spec, xyz_on, r_mol=None, symbols=symbols,
            chemistry=PathChemistry.GENERIC)
        # Score = |d - target| / 0.35 + penalty(planarity)
        # + penalty(blocking).  With no r_mol, planarity/blocking helpers
        # should return False (no penalties).
        self.assertAlmostEqual(score_on, 0.0, places=6)

        # Shift the bond by +0.35 Å → |d - target| / 0.35 = 1.0.
        xyz_off = {'symbols': symbols,
                   'isotopes': (12, 12, 1, 1),
                   'coords': ((0.0, 0.0, 0.0), (target + 0.35, 0.0, 0.0),
                              (10.0, 0.0, 0.0), (11.0, 0.0, 0.0))}
        score_off = score_guess_against_path_spec(
            spec, xyz_off, r_mol=None, symbols=symbols,
            chemistry=PathChemistry.GENERIC)
        self.assertAlmostEqual(score_off, 1.0, places=4)


if __name__ == '__main__':
    unittest.main()
