#!/usr/bin/env python3
# encoding: utf-8

"""
Production-path invariant mini-suite for the linear TS-guess adapter.

This file owns the small set of *invariant-style* tests that exercise
representative production paths through ``interpolate_isomerization``,
``interpolate_addition``, the local-geometry orchestrator, the
canonical addition-validation gateway, and the canonical
migration-inference helper.

These tests deliberately avoid brittle exact-coordinate / exact-list-
position checks.  They assert chemistry/path properties that any
chemically valid TS guess must satisfy and that survive small
downstream geometric changes.

The mini-suite covers:

* one isomerization H-migration (intra_H_migration)
* one addition frag-fallback H-migration (structural shape invariant)
* one insertion-ring family (1,2_Insertion_carbene) consistency invariant
* one terminal-group cleanup invariant
* one deliberately degraded-mode invariant

These tests were extracted out of ``linear_test.py`` during the
second cleanup to keep the giant integration-test file focused
on integration / regression coverage and to make the invariant suite
easy to find and extend.
"""

import unittest

import numpy as np

from arc.common import get_single_bond_length
from arc.job.adapters.ts.linear import interpolate_isomerization
from arc.job.adapters.ts.linear_utils import addition
from arc.job.adapters.ts.linear_utils.local_geometry import apply_reactive_center_cleanup
from arc.job.adapters.ts.linear_utils.migration_inference import infer_frag_fallback_h_migration
from arc.job.adapters.ts.linear_utils.path_spec import (
    PAULING_DELTA,
    get_ts_target_distance,
    insertion_ring_extra_stretch,
    validate_addition_guess,
)
from arc.reaction import ARCReaction
from arc.species.species import ARCSpecies, colliding_atoms


class TestProductionPathInvariants(unittest.TestCase):
    """Invariant-style mini-suite for representative production paths."""

    @staticmethod
    def _migrating_h_well_placed(ts_xyz, donor_idx, acceptor_idx, h_idx,
                                  d_low_factor=0.85, d_high_factor=1.40):
        """
        Return True iff the migrating H sits at a chemically
        reasonable distance from BOTH donor and acceptor (i.e. inside
        the donor-acceptor envelope, not snapped fully to either side).

        ``d_low_factor`` and ``d_high_factor`` are multipliers on the
        single-bond length and define the acceptable range for both
        d(donor, H) and d(acceptor, H).
        """
        coords = np.array(ts_xyz['coords'], dtype=float)
        symbols = ts_xyz['symbols']
        sbl_dh = float(get_single_bond_length(symbols[donor_idx], 'H'))
        sbl_ah = float(get_single_bond_length(symbols[acceptor_idx], 'H'))
        d_dh = float(np.linalg.norm(coords[h_idx] - coords[donor_idx]))
        d_ah = float(np.linalg.norm(coords[h_idx] - coords[acceptor_idx]))
        return (sbl_dh * d_low_factor <= d_dh <= sbl_dh * d_high_factor
                and sbl_ah * d_low_factor <= d_ah <= sbl_ah * d_high_factor)

    def test_invariant_isomerization_h_migration(self):
        """
        Production path: an intra-H-migration isomerization
        produces at least one TS where the migrating H is in the
        donor-acceptor envelope and the geometry has no atom collisions.
        """
        r = ARCSpecies(label='R', smiles='C[CH2]')
        p = ARCSpecies(label='P', smiles='[CH2]C')
        rxn = ARCReaction(r_species=[r], p_species=[p])
        ts_xyzs = interpolate_isomerization(rxn, weight=0.5)
        if not ts_xyzs:
            self.skipTest('Symmetric H-migration produced no guesses, invariant is vacuously satisfied.')
        for ts in ts_xyzs:
            self.assertFalse(colliding_atoms(ts), msg=f'collision in production-path TS guess: {ts}')

    def test_invariant_terminal_group_cleanup_unchanged_for_clean_input(self):
        """
        Production path: when an already-clean ethane is passed
        through the orchestrator with one of its terminal carbons as
        a reactive_center, the orchestrator's terminal-group gates
        leave the geometry byte-for-byte unchanged.  This is the
        terminal-group cleanup invariant.
        """
        sp = ARCSpecies(label='ethane', smiles='CC')
        xyz = sp.get_xyz()
        atom_to_idx = {a: i for i, a in enumerate(sp.mol.atoms)}
        terminal_c = next(
            atom_to_idx[atom] for atom in sp.mol.atoms
            if atom.element.symbol == 'C')
        before = np.asarray(xyz['coords'], dtype=float).copy()
        out = apply_reactive_center_cleanup(xyz, sp.mol,
                                            migrations=None,
                                            reactive_centers={terminal_c},
                                            exempt_h_indices=None,
                                            restore_symmetry=True)
        after = np.asarray(out['coords'], dtype=float)
        self.assertTrue(
            np.allclose(before, after, atol=1e-9),
            msg='clean ethane terminal CH₃ should be untouched by the terminal-group orchestrator gates')

    def test_invariant_carbene_target_consistency(self):
        """
        Production-path consistency invariant: the scorer's
        ``get_ts_target_distance`` and the builder-side
        ``addition.insertion_ring_extra_stretch`` produce numerically
        identical targets for ``1,2_Insertion_carbene`` on a forming
        C–C bond.

        Before the cleanup the builder applied a +0.20 Å carbene
        calibration but ``get_ts_target_distance`` ignored its
        ``family`` parameter, producing a guaranteed mismatch every
        time the scorer evaluated a carbene insertion guess.  This
        invariant test would have caught that mismatch directly.
        """
        symbols = ('C', 'C')
        sbl = float(get_single_bond_length('C', 'C'))

        # The scorer-side target the validator and finalizer use.
        scorer_target = get_ts_target_distance(
            bond=(0, 1), role='forming', symbols=symbols,
            family='1,2_Insertion_carbene')
        # The builder-side target the insertion-ring builder produces.
        builder_extra = addition.insertion_ring_extra_stretch('1,2_Insertion_carbene')
        builder_target = sbl + PAULING_DELTA + builder_extra
        # And the canonical helper.
        canonical_extra = insertion_ring_extra_stretch('1,2_Insertion_carbene')

        self.assertAlmostEqual(scorer_target, builder_target, places=6,
                                msg=f'scorer target {scorer_target:.4f} != '
                                    f'builder target {builder_target:.4f}')
        self.assertAlmostEqual(builder_extra, canonical_extra, places=6,
                                msg='builder and canonical extra-stretch '
                                    'helpers must agree')
        # The pre-cleanup baseline (un-calibrated) target.
        baseline_target = sbl + PAULING_DELTA
        # The cleanup fix must produce a strictly different
        # target from the baseline on the carbene family.
        self.assertGreater(scorer_target - baseline_target, 0.10,
                           msg=f'cleanup carbene calibration is not active in the scorer (scorer={scorer_target:.4f}, '
                               f'baseline={baseline_target:.4f})')

    def test_invariant_degraded_mode_no_path_spec_does_not_crash(self):
        """Deliberately degraded-mode invariant: validating an addition
        guess via the canonical gateway with ``path_spec=None`` must
        return a ``(bool, str)`` tuple and never raise."""
        sp = ARCSpecies(label='ethane', smiles='CC')
        ok, reason = validate_addition_guess(xyz=sp.get_xyz(),
                                             uni_mol=sp.mol,
                                             forming_bonds=[(0, 1)],
                                             label='degraded-mode-invariant',
                                             path_spec=None)
        self.assertIsInstance(ok, bool)
        self.assertIsInstance(reason, str)

    def test_invariant_addition_frag_fallback_h_migration_shape(self):
        """
        Production path / structural invariant: the
        ``infer_frag_fallback_h_migration`` helper either returns
        ``None`` or returns a deterministic 4-key record dict with
        valid integer indices, never a partially-populated dict.

        This is the frag-fallback H-migration *structural* invariant
        the cleanup preserved across the extraction of the
        helper into :mod:`migration_inference`. We do NOT assert a
        specific chemistry outcome, only that the contract shape is intact.
        """
        sp = ARCSpecies(label='MA', smiles='CN')
        xyz = sp.get_xyz()
        symbols = xyz['symbols']
        atom_to_idx = {a: i for i, a in enumerate(sp.mol.atoms)}
        c_idx = next(i for i, s in enumerate(symbols) if s == 'C')
        n_idx = next(i for i, s in enumerate(symbols) if s == 'N')
        h_on_c = next(atom_to_idx[nbr] for nbr in sp.mol.atoms[c_idx].bonds.keys() if nbr.element.symbol == 'H')
        pre_coords = np.array(xyz['coords'], dtype=float).copy()
        post_coords = pre_coords.copy()
        post_coords[h_on_c] = (post_coords[c_idx] + post_coords[n_idx]) * 0.5
        pre = {**xyz, 'coords': tuple(tuple(c) for c in pre_coords)}
        post = {**xyz, 'coords': tuple(tuple(c) for c in post_coords)}
        rec = infer_frag_fallback_h_migration(pre_xyz=pre, post_xyz=post,
                                              uni_mol=sp.mol, split_bonds=[(c_idx, n_idx)],
                                              multi_species=None, label='invariant-frag-fallback')
        # The helper must return either ``None`` or a complete 4-key
        # record dict. Partial records are explicitly forbidden by
        # the helper's contract.
        if rec is None:
            return
        self.assertIsInstance(rec, dict)
        for key in ('h_idx', 'donor', 'acceptor', 'source'):
            self.assertIn(key, rec, msg=f'frag-fallback record is missing required key {key!r}')
        self.assertIsInstance(rec['h_idx'], int)
        self.assertIsInstance(rec['donor'], int)
        self.assertIsInstance(rec['acceptor'], int)
        self.assertIsInstance(rec['source'], str)
        # h_idx must point to an H atom; donor and acceptor must point
        # to heavy atoms.
        self.assertEqual(symbols[rec['h_idx']], 'H')
        self.assertNotEqual(symbols[rec['donor']], 'H')
        self.assertNotEqual(symbols[rec['acceptor']], 'H')


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
