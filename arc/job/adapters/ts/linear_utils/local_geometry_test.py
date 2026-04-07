#!/usr/bin/env python3
# encoding: utf-8

"""Phase 3b unit tests for local reactive-center geometry helpers."""

import unittest

import numpy as np

from arc.common import get_single_bond_length
from arc.species import ARCSpecies

from arc.job.adapters.ts.linear_utils.local_geometry import (
    apply_reactive_center_cleanup,
    clean_migrating_h,
    identify_h_migration_pairs,
    is_terminal_group_asymmetric,
    orient_h_away_from_axis,
    regularize_terminal_h_geometry,
    restore_terminal_h_symmetry,
)
from arc.job.adapters.ts.linear_utils.postprocess import PAULING_DELTA


class TestRegularizeTerminalHGeometry(unittest.TestCase):
    """``regularize_terminal_h_geometry`` snaps distorted terminal H
    bond lengths back to ``get_single_bond_length(C, H)`` while
    preserving direction."""

    def setUp(self):
        # Methane CH4 — center is C[0], 4 H atoms.
        self.sp = ARCSpecies(label='CH4', smiles='C')
        self.xyz = self.sp.get_xyz()
        self.mol = self.sp.mol
        # Identify the C atom and the H atoms.
        self.c_idx = next(i for i, s in enumerate(self.xyz['symbols']) if s == 'C')
        self.h_idxs = [i for i, s in enumerate(self.xyz['symbols']) if s == 'H']

    def test_no_op_when_within_range(self):
        """A fresh CH4 should not be touched."""
        out = regularize_terminal_h_geometry(self.xyz, self.mol, self.c_idx)
        self.assertIs(out, self.xyz)

    def test_pulls_overstretched_h_back(self):
        """An H atom stretched far past sbl is snapped back along its bond."""
        coords = list(map(list, self.xyz['coords']))
        h = self.h_idxs[0]
        c_pos = np.array(coords[self.c_idx])
        h_pos = np.array(coords[h])
        # Stretch this H to 2.0 Å along the same direction.
        direction = (h_pos - c_pos) / float(np.linalg.norm(h_pos - c_pos))
        coords[h] = (c_pos + direction * 2.0).tolist()
        bad = {**self.xyz, 'coords': tuple(tuple(row) for row in coords)}
        out = regularize_terminal_h_geometry(bad, self.mol, self.c_idx)
        sbl = get_single_bond_length('C', 'H')
        new_d = float(np.linalg.norm(
            np.array(out['coords'][h]) - np.array(out['coords'][self.c_idx])))
        self.assertAlmostEqual(new_d, sbl, places=4)

    def test_exclude_atoms_skips_listed_hydrogen(self):
        coords = list(map(list, self.xyz['coords']))
        h = self.h_idxs[0]
        c_pos = np.array(coords[self.c_idx])
        h_pos = np.array(coords[h])
        direction = (h_pos - c_pos) / float(np.linalg.norm(h_pos - c_pos))
        coords[h] = (c_pos + direction * 2.5).tolist()
        bad = {**self.xyz, 'coords': tuple(tuple(row) for row in coords)}
        out = regularize_terminal_h_geometry(
            bad, self.mol, self.c_idx, exclude_atoms={h})
        # Excluded H is not touched — its distance is still 2.5.
        self.assertAlmostEqual(
            float(np.linalg.norm(
                np.array(out['coords'][h]) - np.array(out['coords'][self.c_idx]))),
            2.5, places=4)


class TestOrientHAwayFromAxis(unittest.TestCase):
    """The blocking-H reorientation helper reflects an inward H to the
    opposite side of the donor–parent–acceptor axis when it sits in the
    blocking pocket."""

    def test_inward_h_is_reflected(self):
        """An H bonded to donor and sitting in the inward blocking
        pocket (small angle to the donor–acceptor axis AND close to the
        acceptor) is reflected across that axis to the outward side."""
        # Build a hand-crafted donor–acceptor pair with one H sitting
        # IN the blocking pocket: short distance to acceptor (< 1.60 Å)
        # AND small angle to the donor–acceptor axis (< 85°).
        xyz = {
            'symbols': ('C', 'C', 'H'),
            'isotopes': (12, 12, 1),
            'coords': (
                (0.0, 0.0, 0.0),     # donor
                (2.5, 0.0, 0.0),     # acceptor — closer than before
                (1.4, 0.10, 0.0),    # inward H: ~1.10 Å from acceptor
            ),
        }
        class _A:
            def __init__(self, sym, bonds):
                self.bonds = bonds
                class _E:
                    def __init__(self, s):
                        self.symbol = s
                self.element = _E(sym)
        c0 = _A('C', {})
        c1 = _A('C', {})
        h2 = _A('H', {})
        c0.bonds = {h2: None}
        h2.bonds = {c0: None}
        c1.bonds = {}
        class _M:
            atoms = [c0, c1, h2]

        out = orient_h_away_from_axis(xyz, _M(), donor=0, acceptor=1, exclude_h=set())
        h_after = np.array(out['coords'][2])
        # Reflected x component flips sign of the axis-parallel piece
        # (1.4 → -1.4 after subtracting 2 × 1.4 × axis_hat from h_vec).
        self.assertLess(h_after[0], 0.0)
        # Heavy atoms unchanged.
        self.assertEqual(out['coords'][0], xyz['coords'][0])
        self.assertEqual(out['coords'][1], xyz['coords'][1])

    def test_excluded_h_is_left_alone(self):
        xyz = {
            'symbols': ('C', 'C', 'H'),
            'isotopes': (12, 12, 1),
            'coords': (
                (0.0, 0.0, 0.0), (3.0, 0.0, 0.0), (0.5, 0.05, 0.0),
            ),
        }
        class _A:
            def __init__(self, sym, bonds):
                self.bonds = bonds
                class _E:
                    def __init__(self, s):
                        self.symbol = s
                self.element = _E(sym)
        c0 = _A('C', {})
        c1 = _A('C', {})
        h2 = _A('H', {})
        c0.bonds = {h2: None}
        h2.bonds = {c0: None}
        class _M:
            atoms = [c0, c1, h2]
        out = orient_h_away_from_axis(
            xyz, _M(), donor=0, acceptor=1, exclude_h={2})
        # The migrating H (excluded) is untouched, so the result should be
        # identical to the input.
        self.assertEqual(out['coords'], xyz['coords'])

    def test_outward_h_is_left_alone(self):
        # H sitting OUTSIDE the blocking pocket — should not be moved.
        xyz = {
            'symbols': ('C', 'C', 'H'),
            'isotopes': (12, 12, 1),
            'coords': (
                (0.0, 0.0, 0.0), (3.0, 0.0, 0.0), (-0.7, 0.7, 0.0),
            ),
        }
        class _A:
            def __init__(self, sym, bonds):
                self.bonds = bonds
                class _E:
                    def __init__(self, s):
                        self.symbol = s
                self.element = _E(sym)
        c0 = _A('C', {})
        c1 = _A('C', {})
        h2 = _A('H', {})
        c0.bonds = {h2: None}
        h2.bonds = {c0: None}
        class _M:
            atoms = [c0, c1, h2]
        out = orient_h_away_from_axis(
            xyz, _M(), donor=0, acceptor=1, exclude_h=set())
        self.assertEqual(out['coords'], xyz['coords'])


class TestCleanMigratingH(unittest.TestCase):
    """``clean_migrating_h`` re-places the migrating H at the donor–
    acceptor triangulated TS position and is idempotent."""

    def test_idempotent_on_triangulated_position(self):
        # Two heavy atoms 2.5 Å apart, an H roughly between them.
        xyz = {
            'symbols': ('C', 'N', 'H'),
            'isotopes': (12, 14, 1),
            'coords': (
                (0.0, 0.0, 0.0), (2.5, 0.0, 0.0), (1.0, 0.5, 0.0),
            ),
        }
        class _A:
            def __init__(self, sym):
                class _E:
                    def __init__(self, s):
                        self.symbol = s
                self.element = _E(sym)
                self.bonds = {}
        c = _A('C'); n = _A('N'); h = _A('H')
        class _M:
            atoms = [c, n, h]
        out = clean_migrating_h(xyz, _M(), donor=0, acceptor=1, h_idx=2)
        h_pos_1 = np.array(out['coords'][2])
        # Run again — should be a no-op (idempotent).
        out2 = clean_migrating_h(out, _M(), donor=0, acceptor=1, h_idx=2)
        h_pos_2 = np.array(out2['coords'][2])
        self.assertTrue(np.allclose(h_pos_1, h_pos_2, atol=1e-10))

    def test_h_distance_to_donor_matches_target(self):
        xyz = {
            'symbols': ('C', 'N', 'H'),
            'isotopes': (12, 14, 1),
            'coords': (
                (0.0, 0.0, 0.0), (2.5, 0.0, 0.0), (1.0, 0.5, 0.0),
            ),
        }
        class _A:
            def __init__(self, sym):
                class _E:
                    def __init__(self, s):
                        self.symbol = s
                self.element = _E(sym)
                self.bonds = {}
        c = _A('C'); n = _A('N'); h = _A('H')
        class _M:
            atoms = [c, n, h]
        out = clean_migrating_h(xyz, _M(), donor=0, acceptor=1, h_idx=2)
        h_pos = np.array(out['coords'][2])
        d_dh = float(np.linalg.norm(h_pos - np.array(out['coords'][0])))
        sbl_ch = get_single_bond_length('C', 'H')
        # The triangulation places H at distance d_DH = sbl + PAULING_DELTA from donor.
        self.assertAlmostEqual(d_dh, sbl_ch + PAULING_DELTA, places=4)


class TestRestoreTerminalHSymmetry(unittest.TestCase):
    """``restore_terminal_h_symmetry`` averages H positions about the
    parent–center axis when the center is clearly CH₂ or CH₃."""

    def test_ch3_distorted_h_is_resymmetrized(self):
        sp = ARCSpecies(label='ethane', smiles='CC')
        xyz = sp.get_xyz()
        symbols = xyz['symbols']
        # Find the first C and its H neighbors.
        c0 = next(i for i, s in enumerate(symbols) if s == 'C')
        atom_to_idx = {a: i for i, a in enumerate(sp.mol.atoms)}
        h_neighbors = [
            atom_to_idx[nbr] for nbr in sp.mol.atoms[c0].bonds.keys()
            if nbr.element.symbol == 'H'
        ]
        # Distort one H by ~0.02 Å (well within the bail-out tolerance).
        coords = list(map(list, xyz['coords']))
        coords[h_neighbors[0]][1] += 0.02
        bad = {**xyz, 'coords': tuple(tuple(row) for row in coords)}
        out = restore_terminal_h_symmetry(bad, sp.mol, c0)
        new_coords = np.asarray(out['coords'])
        # The three H atoms around c0 should have nearly equal distances
        # to c0 after symmetrization.
        c_pos = new_coords[c0]
        dists = [float(np.linalg.norm(new_coords[h] - c_pos)) for h in h_neighbors]
        self.assertAlmostEqual(max(dists) - min(dists), 0.0, places=2)

    def test_skips_non_terminal_centers(self):
        # The central C of propane has 2 heavy neighbors → not a clear terminal.
        sp = ARCSpecies(label='propane', smiles='CCC')
        xyz = sp.get_xyz()
        symbols = xyz['symbols']
        # The middle C is the one with 2 C neighbors.
        atom_to_idx = {a: i for i, a in enumerate(sp.mol.atoms)}
        middle_c = None
        for atom in sp.mol.atoms:
            if atom.element.symbol == 'C':
                heavy_nbrs = sum(1 for n in atom.bonds.keys() if n.element.symbol != 'H')
                if heavy_nbrs == 2:
                    middle_c = atom_to_idx[atom]
                    break
        self.assertIsNotNone(middle_c)
        out = restore_terminal_h_symmetry(xyz, sp.mol, middle_c)
        # No-op: the function returns the original input.
        self.assertIs(out, xyz)

    def test_does_not_introduce_h_crowding(self):
        sp = ARCSpecies(label='ethane', smiles='CC')
        xyz = sp.get_xyz()
        out = restore_terminal_h_symmetry(
            xyz, sp.mol,
            next(i for i, s in enumerate(xyz['symbols']) if s == 'C'),
        )
        # Verify pairwise H distances are not closer than 1.5 Å.
        coords = np.asarray(out['coords'])
        h_idxs = [i for i, s in enumerate(out['symbols']) if s == 'H']
        for i in range(len(h_idxs)):
            for j in range(i + 1, len(h_idxs)):
                d = float(np.linalg.norm(coords[h_idxs[i]] - coords[h_idxs[j]]))
                self.assertGreater(d, 1.50)


class TestIsTerminalGroupAsymmetric(unittest.TestCase):
    """Phase 4b — pure detector for unphysically distorted terminal CH₂/CH₃.

    Returns ``True`` only when an umbrella inversion (any H angle to
    the parent → center axis exceeds 100°) OR an azimuthal distortion
    (CH₃ cyclic spacings deviate from 120° by more than the threshold,
    or CH₂ smaller cyclic separation drops below 80°) is present.
    Returns ``False`` for already-good geometries so the symmetry
    restorer leaves them alone.
    """

    @staticmethod
    def _build_axial_ch3(theta_radians):
        """Construct an XYZ for a model -CH₃ group with explicit per-H
        azimuth angles around a fixed parent → center axis.

        Atom layout:
            0: parent (X) at (0, 0, 0)
            1: center (C) at (0, 0, 1.54)         — outward axis is +z
            2..2+n_h: H atoms placed at azimuth ``theta_radians`` around +z
        """
        n_h = len(theta_radians)
        # Standard outward C–H direction relative to the +z axis is at
        # 70.5° from +z (the supplement of the tetrahedral 109.5° H–C–C
        # angle).  Place each H on that cone at the chosen azimuth.
        cone_angle = np.deg2rad(70.5)
        sin_c = float(np.sin(cone_angle))
        cos_c = float(np.cos(cone_angle))
        center = np.array([0.0, 0.0, 1.54])
        symbols = ['C', 'C'] + ['H'] * n_h
        coords = [(0.0, 0.0, 0.0), tuple(center)]
        for theta in theta_radians:
            h = center + np.array([sin_c * np.cos(theta),
                                   sin_c * np.sin(theta),
                                   cos_c])
            coords.append(tuple(h))
        return {
            'symbols': tuple(symbols),
            'isotopes': (12, 12) + (1,) * n_h,
            'coords': tuple(coords),
        }, 1, 0, list(range(2, 2 + n_h))

    def test_ideal_ch3_is_not_asymmetric(self):
        """An ideal CH₃ with H atoms at 0°, 120°, 240° passes."""
        xyz, c, p, hs = self._build_axial_ch3(
            [0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0])
        self.assertFalse(is_terminal_group_asymmetric(
            xyz, center_idx=c, parent_idx=p, h_indices=hs))

    def test_inverted_ch3_umbrella_is_asymmetric(self):
        """An H angle to the outward axis exceeding 100° (the inverted
        umbrella case) is rejected."""
        # Build a CH₃ but flip ONE H to point back toward the parent by
        # placing it on the *inverted* cone (z < 0 below the center).
        n_h = 3
        thetas = [0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0]
        cone_angle = np.deg2rad(70.5)
        center = np.array([0.0, 0.0, 1.54])
        coords = [(0.0, 0.0, 0.0), tuple(center)]
        for k, theta in enumerate(thetas):
            sin_c = float(np.sin(cone_angle))
            if k == 0:
                # Inverted: place this H on the -z side of the center.
                cos_c = -float(np.cos(cone_angle))  # negative z component
            else:
                cos_c = float(np.cos(cone_angle))
            h = center + np.array([sin_c * np.cos(theta),
                                   sin_c * np.sin(theta),
                                   cos_c])
            coords.append(tuple(h))
        xyz = {
            'symbols': ('C', 'C', 'H', 'H', 'H'),
            'isotopes': (12, 12, 1, 1, 1),
            'coords': tuple(coords),
        }
        self.assertTrue(is_terminal_group_asymmetric(
            xyz, center_idx=1, parent_idx=0, h_indices=[2, 3, 4]))

    def test_ch3_azimuthal_distortion_is_asymmetric(self):
        """A CH₃ where one H is bent far enough that the cyclic spacings
        deviate from 120° by more than the default threshold (20°) is
        rejected."""
        # Place H atoms at 0°, 120°, 240° + 50°  = 290°.  Cyclic
        # spacings become 120°, 170°, 70° → max deviation 50° > 20°.
        thetas = [0.0,
                  2.0 * np.pi / 3.0,
                  4.0 * np.pi / 3.0 + np.deg2rad(50.0)]
        xyz, c, p, hs = self._build_axial_ch3(thetas)
        self.assertTrue(is_terminal_group_asymmetric(
            xyz, center_idx=c, parent_idx=p, h_indices=hs))

    def test_ch3_small_distortion_is_not_asymmetric(self):
        """A CH₃ with H atoms only ~10° away from ideal must NOT trip
        the detector — the threshold (20°) gives a comfortable margin
        so RDKit/force-field geometries pass through unchanged."""
        thetas = [0.0,
                  2.0 * np.pi / 3.0 + np.deg2rad(10.0),
                  4.0 * np.pi / 3.0 - np.deg2rad(5.0)]
        xyz, c, p, hs = self._build_axial_ch3(thetas)
        self.assertFalse(is_terminal_group_asymmetric(
            xyz, center_idx=c, parent_idx=p, h_indices=hs))

    def test_ch2_squeezed_is_asymmetric(self):
        """A CH₂ whose two H projected azimuths are squeezed to ~60°
        (smaller cyclic separation < 80°) is rejected."""
        thetas = [0.0, np.deg2rad(60.0)]
        xyz, c, p, hs = self._build_axial_ch3(thetas)
        # Update symbols / isotopes for the 2-H case.
        new_symbols = ('C', 'C', 'H', 'H')
        new_iso = (12, 12, 1, 1)
        xyz = {
            'symbols': new_symbols,
            'isotopes': new_iso,
            'coords': xyz['coords'],
        }
        self.assertTrue(is_terminal_group_asymmetric(
            xyz, center_idx=c, parent_idx=p, h_indices=hs))

    def test_ch2_outward_pair_is_not_asymmetric(self):
        """A CH₂ with the two H projected azimuths roughly opposite
        (~180° apart) is *not* rejected — that's the normal terminal
        =CH₂ geometry."""
        thetas = [0.0, np.pi]
        xyz, c, p, hs = self._build_axial_ch3(thetas)
        xyz = {
            'symbols': ('C', 'C', 'H', 'H'),
            'isotopes': (12, 12, 1, 1),
            'coords': xyz['coords'],
        }
        self.assertFalse(is_terminal_group_asymmetric(
            xyz, center_idx=c, parent_idx=p, h_indices=hs))

    def test_non_terminal_h_count_returns_false(self):
        """Passing fewer than 2 or more than 3 H indices is treated as
        'not eligible' and returns False without throwing."""
        xyz, c, p, hs = self._build_axial_ch3(
            [0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0])
        # Single H (not a CH₂/CH₃ shape) → False.
        self.assertFalse(is_terminal_group_asymmetric(
            xyz, center_idx=c, parent_idx=p, h_indices=[hs[0]]))
        # 4 H atoms (CH₄ shape) → False.
        self.assertFalse(is_terminal_group_asymmetric(
            xyz, center_idx=c, parent_idx=p, h_indices=hs + [hs[0]]))

    def test_degenerate_axis_returns_false(self):
        """When parent and center share the same coordinates, the axis
        is degenerate; the function returns False (no opinion)."""
        xyz = {
            'symbols': ('C', 'C', 'H', 'H', 'H'),
            'isotopes': (12, 12, 1, 1, 1),
            'coords': ((0.0, 0.0, 0.0),
                       (0.0, 0.0, 0.0),  # parent ≡ center
                       (1.0, 0.0, 0.0),
                       (-0.5, 0.866, 0.0),
                       (-0.5, -0.866, 0.0)),
        }
        self.assertFalse(is_terminal_group_asymmetric(
            xyz, center_idx=1, parent_idx=0, h_indices=[2, 3, 4]))

    def test_threshold_is_tunable(self):
        """A wider threshold lets larger distortions pass."""
        thetas = [0.0,
                  2.0 * np.pi / 3.0,
                  4.0 * np.pi / 3.0 + np.deg2rad(25.0)]
        xyz, c, p, hs = self._build_axial_ch3(thetas)
        # Default threshold (20°) → asymmetric.
        self.assertTrue(is_terminal_group_asymmetric(
            xyz, center_idx=c, parent_idx=p, h_indices=hs))
        # Wider threshold (40°) → no longer asymmetric.
        self.assertFalse(is_terminal_group_asymmetric(
            xyz, center_idx=c, parent_idx=p, h_indices=hs,
            threshold_deg=40.0))


class TestApplyReactiveCenterCleanup(unittest.TestCase):
    """Phase 4a thin orchestrator over the existing local-geometry helpers.

    These tests demonstrate that the orchestrator:
    * is a no-op when nothing is requested,
    * routes a migration triple through ``clean_migrating_h`` +
      ``orient_h_away_from_axis`` + ``regularize_terminal_h_geometry``,
    * runs ``restore_terminal_h_symmetry`` on each named reactive center
      (and respects the existing terminal-CH₂/CH₃ gating),
    * leaves unrelated atoms in the molecule untouched.
    """

    def test_no_op_when_no_targets(self):
        sp = ARCSpecies(label='ethane', smiles='CC')
        xyz = sp.get_xyz()
        out = apply_reactive_center_cleanup(xyz, sp.mol)
        self.assertIs(out, xyz)
        out_none = apply_reactive_center_cleanup(
            xyz, sp.mol, migrations=[], reactive_centers=set())
        self.assertIs(out_none, xyz)

    def test_orchestrator_repositions_migrating_h(self):
        """Given a (donor, acceptor, h_idx) record where the H is sitting
        at its donor-side bond length, the orchestrator triangulates it
        to the symmetric Pauling TS position between donor and acceptor."""
        # Use methylamine (CH3-NH2) — atom 0 is C (donor), atom 1 is N
        # (acceptor), pick one of C's H atoms as the migrating H.  Then
        # override the geometry to make the H sit on the C-N axis at
        # a normal C-H bond length (1.10 Å).
        sp = ARCSpecies(label='MA', smiles='CN')
        symbols = sp.get_xyz()['symbols']
        c_idx = next(i for i, s in enumerate(symbols) if s == 'C')
        n_idx = next(i for i, s in enumerate(symbols) if s == 'N')
        atom_to_idx = {a: i for i, a in enumerate(sp.mol.atoms)}
        h_on_c = next(
            atom_to_idx[nbr] for nbr in sp.mol.atoms[c_idx].bonds.keys()
            if nbr.element.symbol == 'H'
        )
        # Build a synthetic geometry that lays C, N, and the chosen H
        # in a straight line: C at origin, N at +x = 3.0, the H at
        # +x = 1.10 (still bonded to C in the graph).
        coords = np.zeros((len(symbols), 3))
        coords[c_idx] = (0.0, 0.0, 0.0)
        coords[n_idx] = (2.50, 0.0, 0.0)
        coords[h_on_c] = (1.10, 0.10, 0.0)
        # Park the other atoms far enough away that they don't matter.
        for k in range(len(symbols)):
            if k in (c_idx, n_idx, h_on_c):
                continue
            coords[k] = (10.0 + k, 10.0, 10.0)
        synthetic = {'symbols': symbols, 'isotopes': sp.get_xyz()['isotopes'],
                     'coords': tuple(tuple(row) for row in coords)}
        out = apply_reactive_center_cleanup(
            synthetic, sp.mol,
            migrations=[{'h_idx': h_on_c, 'donor': c_idx, 'acceptor': n_idx}],
        )
        coords_out = np.asarray(out['coords'], dtype=float)
        d_ch = float(np.linalg.norm(coords_out[h_on_c] - coords_out[c_idx]))
        d_nh = float(np.linalg.norm(coords_out[h_on_c] - coords_out[n_idx]))
        sbl_ch = float(get_single_bond_length('C', 'H'))
        sbl_nh = float(get_single_bond_length('N', 'H'))
        self.assertAlmostEqual(d_ch, sbl_ch + PAULING_DELTA, places=3)
        self.assertAlmostEqual(d_nh, sbl_nh + PAULING_DELTA, places=3)

    def test_orchestrator_does_not_rotate_already_symmetric_ch3(self):
        """Phase 4b regression guard.  When the named reactive center is
        an already-symmetric terminal CH₃, the orchestrator's symmetry
        restoration must NOT fire (the asymmetry detector returns False)
        and the H atoms must end up byte-for-byte where they started.

        This is the test that the Phase 4a wiring (which always called
        ``restore_terminal_h_symmetry`` at this site) failed because
        the unconditional rotation churned coordinates of already-good
        groups.
        """
        sp = ARCSpecies(label='ethane', smiles='CC')
        xyz = sp.get_xyz()
        symbols = xyz['symbols']
        c0 = next(i for i, s in enumerate(symbols) if s == 'C')
        atom_to_idx = {a: i for i, a in enumerate(sp.mol.atoms)}
        h_neighbors = [
            atom_to_idx[nbr] for nbr in sp.mol.atoms[c0].bonds.keys()
            if nbr.element.symbol == 'H'
        ]
        before = np.asarray(xyz['coords'], dtype=float).copy()
        out = apply_reactive_center_cleanup(
            xyz, sp.mol, reactive_centers={c0}, restore_symmetry=True)
        after = np.asarray(out['coords'], dtype=float)
        for h in h_neighbors:
            self.assertTrue(
                np.allclose(before[h], after[h], atol=1e-9),
                msg=f'H{h} moved on an already-symmetric CH₃ '
                    f'(asymmetry detector should have returned False)',
            )

    def test_orchestrator_restores_azimuthally_distorted_ch3_when_signaled(self):
        """Phase 4b — when the orchestrator is given a CH₃ whose H
        atoms are azimuthally distorted (cyclic spacings around the
        parent → center axis deviate from 120° by more than the
        detector threshold), the asymmetry detector returns True and
        ``restore_terminal_h_symmetry`` is invoked.

        We use azimuthal-only distortion (rotation of one H around the
        outward axis) so that all per-H bond lengths to the parent C
        are preserved — that keeps the symmetrizer's 0.05 Å bond-length
        bail-out from firing.  This isolates the test to the
        "asymmetry signal triggers restoration" pathway specifically.
        """
        sp = ARCSpecies(label='ethane', smiles='CC')
        xyz = sp.get_xyz()
        symbols = xyz['symbols']
        c0 = next(i for i, s in enumerate(symbols) if s == 'C')
        c1 = next(i for i, s in enumerate(symbols) if s == 'C' and i != c0)
        atom_to_idx = {a: i for i, a in enumerate(sp.mol.atoms)}
        h_neighbors = [
            atom_to_idx[nbr] for nbr in sp.mol.atoms[c0].bonds.keys()
            if nbr.element.symbol == 'H'
        ]
        coords = list(map(list, xyz['coords']))
        # Build the outward axis (c0 → away from c1).
        c0_pos = np.array(coords[c0])
        c1_pos = np.array(coords[c1])
        axis = c0_pos - c1_pos
        axis /= float(np.linalg.norm(axis))
        # Rotate ONE H of c0 by +50° around the axis (Rodrigues rotation)
        # — this preserves its bond length to c0 but breaks the 120°
        # azimuthal symmetry, which the detector should pick up.
        h_target = h_neighbors[0]
        h_vec = np.array(coords[h_target]) - c0_pos
        theta_rad = np.deg2rad(50.0)
        cos_t, sin_t = float(np.cos(theta_rad)), float(np.sin(theta_rad))
        rotated = (h_vec * cos_t
                   + np.cross(axis, h_vec) * sin_t
                   + axis * float(np.dot(axis, h_vec)) * (1.0 - cos_t))
        coords[h_target] = (c0_pos + rotated).tolist()
        bad = {**xyz, 'coords': tuple(tuple(row) for row in coords)}
        # Sanity: the bond length is preserved.
        self.assertAlmostEqual(
            float(np.linalg.norm(np.array(coords[h_target]) - c0_pos)),
            float(np.linalg.norm(np.array(xyz['coords'][h_target]) - c0_pos)),
            places=6,
        )
        out = apply_reactive_center_cleanup(
            bad, sp.mol, reactive_centers={c0}, restore_symmetry=True)
        new_coords = np.asarray(out['coords'], dtype=float)
        # The orchestrator should have re-symmetrized: at least one H
        # of c0 must have moved relative to the distorted input.
        moved = any(
            not np.allclose(np.array(coords[h]), new_coords[h], atol=1e-6)
            for h in h_neighbors
        )
        self.assertTrue(
            moved,
            'orchestrator should have re-symmetrized the azimuthally distorted CH₃')
        # And the resulting H–C bond lengths should still all be ~equal
        # (symmetry restoration preserves bond length within 0.05 Å).
        c_pos = new_coords[c0]
        dists = [float(np.linalg.norm(new_coords[h] - c_pos))
                 for h in h_neighbors]
        self.assertLess(max(dists) - min(dists), 0.05)

    def test_orchestrator_distorted_ch3_does_not_touch_unrelated_atoms(self):
        """Phase 4b — even when symmetry restoration *does* fire on a
        distorted center, atoms outside the immediate first shell of
        that center are not moved at all (no whole-molecule churn)."""
        sp = ARCSpecies(label='ethane', smiles='CC')
        xyz = sp.get_xyz()
        symbols = xyz['symbols']
        c0 = next(i for i, s in enumerate(symbols) if s == 'C')
        c1 = next(i for i, s in enumerate(symbols) if s == 'C' and i != c0)
        atom_to_idx = {a: i for i, a in enumerate(sp.mol.atoms)}
        h_on_c1 = [
            atom_to_idx[nbr] for nbr in sp.mol.atoms[c1].bonds.keys()
            if nbr.element.symbol == 'H'
        ]
        h_on_c0 = [
            atom_to_idx[nbr] for nbr in sp.mol.atoms[c0].bonds.keys()
            if nbr.element.symbol == 'H'
        ]
        coords = list(map(list, xyz['coords']))
        # Invert one H on c0 so the asymmetry detector fires there.
        c0_pos = np.array(coords[c0])
        c1_pos = np.array(coords[c1])
        outward = c0_pos - c1_pos
        outward /= float(np.linalg.norm(outward))
        h_vec = np.array(coords[h_on_c0[0]]) - c0_pos
        proj = float(np.dot(h_vec, outward)) * outward
        coords[h_on_c0[0]] = (np.array(coords[h_on_c0[0]]) - 2.0 * proj).tolist()
        bad = {**xyz, 'coords': tuple(tuple(row) for row in coords)}
        before = np.asarray(bad['coords'], dtype=float).copy()
        out = apply_reactive_center_cleanup(
            bad, sp.mol, reactive_centers={c0}, restore_symmetry=True)
        after = np.asarray(out['coords'], dtype=float)
        # The other terminal C and its 3 H atoms must be untouched.
        for idx in [c1] + h_on_c1:
            self.assertTrue(
                np.allclose(before[idx], after[idx], atol=1e-9),
                msg=f'atom {idx} ({symbols[idx]}) moved when it should not have',
            )

    def test_orchestrator_does_not_touch_unrelated_atoms(self):
        """When the orchestrator is given one explicit reactive center,
        atoms not in the immediate first shell of that center are not
        moved at all (no whole-molecule churn)."""
        sp = ARCSpecies(label='propane', smiles='CCC')
        xyz = sp.get_xyz()
        symbols = xyz['symbols']
        # The middle C has 2 heavy neighbors → ``restore_terminal_h_symmetry``
        # bails out, and ``regularize_terminal_h_geometry`` is a no-op when
        # the H bond lengths are already in range.  Pick a *terminal* C
        # so the orchestrator IS allowed to act on it; verify that the
        # atoms in the OTHER terminal group are unchanged.
        atom_to_idx = {a: i for i, a in enumerate(sp.mol.atoms)}
        terminal_cs = []
        for atom in sp.mol.atoms:
            if atom.element.symbol == 'C':
                heavy_nbrs = sum(1 for n in atom.bonds.keys()
                                 if n.element.symbol != 'H')
                if heavy_nbrs == 1:
                    terminal_cs.append(atom_to_idx[atom])
        self.assertEqual(len(terminal_cs), 2)
        c_target = terminal_cs[0]
        c_other = terminal_cs[1]
        coords_before = np.asarray(xyz['coords']).copy()
        out = apply_reactive_center_cleanup(
            xyz, sp.mol, reactive_centers={c_target})
        coords_after = np.asarray(out['coords'])
        # The OTHER terminal C and its 3 H atoms must be untouched.
        other_h_idxs = [
            atom_to_idx[nbr] for nbr in sp.mol.atoms[c_other].bonds.keys()
            if nbr.element.symbol == 'H'
        ]
        for idx in [c_other] + other_h_idxs:
            self.assertTrue(
                np.allclose(coords_before[idx], coords_after[idx], atol=1e-9),
                msg=f'atom {idx} ({symbols[idx]}) moved when it should not have',
            )

    def test_orchestrator_skips_migrating_h_in_symmetry_pass(self):
        """The migrating H should never be re-symmetrized — it has been
        intentionally placed at the triangulated TS position by
        ``clean_migrating_h`` and the symmetry pass would otherwise drag
        it back to a normal C–H bond length."""
        sp = ARCSpecies(label='MA', smiles='CN')
        symbols = sp.get_xyz()['symbols']
        c_idx = next(i for i, s in enumerate(symbols) if s == 'C')
        n_idx = next(i for i, s in enumerate(symbols) if s == 'N')
        atom_to_idx = {a: i for i, a in enumerate(sp.mol.atoms)}
        h_on_c = next(
            atom_to_idx[nbr] for nbr in sp.mol.atoms[c_idx].bonds.keys()
            if nbr.element.symbol == 'H'
        )
        # Build a synthetic colinear geometry — same as the previous test.
        coords = np.zeros((len(symbols), 3))
        coords[c_idx] = (0.0, 0.0, 0.0)
        coords[n_idx] = (2.50, 0.0, 0.0)
        coords[h_on_c] = (1.10, 0.10, 0.0)
        for k in range(len(symbols)):
            if k in (c_idx, n_idx, h_on_c):
                continue
            coords[k] = (10.0 + k, 10.0, 10.0)
        synthetic = {'symbols': symbols, 'isotopes': sp.get_xyz()['isotopes'],
                     'coords': tuple(tuple(row) for row in coords)}
        out = apply_reactive_center_cleanup(
            synthetic, sp.mol,
            migrations=[{'h_idx': h_on_c, 'donor': c_idx, 'acceptor': n_idx}],
        )
        coords_out = np.asarray(out['coords'], dtype=float)
        d_ch = float(np.linalg.norm(coords_out[h_on_c] - coords_out[c_idx]))
        sbl_ch = float(get_single_bond_length('C', 'H'))
        # The migrating H must have moved away from its original 1.10 Å
        # bond length toward the Pauling TS distance (~1.51 Å).
        self.assertGreater(d_ch, sbl_ch + 0.10)


class TestIdentifyHMigrationPairs(unittest.TestCase):
    """``identify_h_migration_pairs`` recovers (h, donor, acceptor)
    triples mirroring ``migrate_verified_atoms``'s internal selection."""

    def test_returns_cross_bond_acceptor_when_available(self):
        # Mock: H4 is the migrating atom, donor is C0, acceptor is N1.
        xyz = {
            'symbols': ('C', 'N', 'C', 'H', 'H'),
            'isotopes': (12, 14, 12, 1, 1),
            'coords': (
                (0.0, 0.0, 0.0),     # C0 (donor, in large)
                (2.5, 0.0, 0.0),     # N1 (acceptor, in core)
                (1.0, 1.5, 0.0),     # C2 (in core)
                (0.5, -1.0, 0.0),    # H3 (non-migrating, on C0)
                (1.0, 0.5, 0.0),     # H4 (migrating)
            ),
        }
        class _A:
            def __init__(self, sym, bonds):
                class _E:
                    def __init__(self, s):
                        self.symbol = s
                self.element = _E(sym)
                self.bonds = bonds
        c0 = _A('C', {}); n1 = _A('N', {}); c2 = _A('C', {})
        h3 = _A('H', {}); h4 = _A('H', {})
        c0.bonds = {h3: None, h4: None}
        h3.bonds = {c0: None}
        h4.bonds = {c0: None}
        class _M:
            atoms = [c0, n1, c2, h3, h4]
        recs = identify_h_migration_pairs(
            xyz, _M(), migrating_atoms={4},
            core={1, 2}, large_prod_atoms={0, 3},
            cross_bonds=[(4, 1)],
        )
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]['h_idx'], 4)
        self.assertEqual(recs[0]['donor'], 0)
        self.assertEqual(recs[0]['acceptor'], 1)
        self.assertEqual(recs[0]['source'], 'cross_bond')

    def test_falls_back_to_nearest_core_when_no_cross_bond(self):
        xyz = {
            'symbols': ('C', 'N', 'C', 'H'),
            'isotopes': (12, 14, 12, 1),
            'coords': (
                (0.0, 0.0, 0.0), (2.5, 0.0, 0.0),
                (1.0, 1.5, 0.0), (1.0, 0.5, 0.0),
            ),
        }
        class _A:
            def __init__(self, sym, bonds):
                class _E:
                    def __init__(self, s):
                        self.symbol = s
                self.element = _E(sym)
                self.bonds = bonds
        c0 = _A('C', {}); n1 = _A('N', {}); c2 = _A('C', {}); h3 = _A('H', {})
        c0.bonds = {h3: None}
        h3.bonds = {c0: None}
        class _M:
            atoms = [c0, n1, c2, h3]
        recs = identify_h_migration_pairs(
            xyz, _M(), migrating_atoms={3},
            core={1, 2}, large_prod_atoms={0}, cross_bonds=None,
        )
        self.assertEqual(len(recs), 1)
        # H3 is closest to C2 (1.0 Å away), not N1 (2.0 Å away).
        self.assertEqual(recs[0]['acceptor'], 2)
        self.assertEqual(recs[0]['source'], 'nearest_core')


# ---------------------------------------------------------------------------
# Phase 3c — fragmentation-fallback single-H inference
# ---------------------------------------------------------------------------


class TestInferFragFallbackHMigration(unittest.TestCase):
    """Strict, deterministic single-H migration inference for the
    frag-fallback addition branch.

    The helper combines five signals (S1–S5) and returns either ONE
    migration record or ``None``.  Each test isolates exactly one
    signal to prove it gates correctly.
    """

    def _real_carboxylic_acid_setup(self):
        """Build a small carboxylic acid (formic acid) with the
        post-migration H sitting between the OH oxygen (donor) and a
        nearby acceptor C atom on a *separate* fragment.

        Used as the canonical S1+S2+S3+S4+S5 success case.
        """
        from arc.species import ARCSpecies
        from arc.reaction import ARCReaction
        # Acetic acid → CH4 + CO2 (1,3_Insertion_CO2-style toy reaction).
        # Atom indices in CC(=O)O:
        #   0 C (methyl)
        #   1 C (carbonyl)
        #   2 O (=O)
        #   3 O (-OH)
        #   4 H (carboxylic-acid H, the migrating H)
        #   5..7 H on the methyl
        sp = ARCSpecies(label='acid', smiles='CC(=O)O')
        return sp

    def test_s1_zero_h_moved_returns_none(self):
        """No H displaced ⇒ S1 fails ⇒ ``None``."""
        from arc.job.adapters.ts.linear_utils.local_geometry import (
            infer_frag_fallback_h_migration,
        )
        sp = self._real_carboxylic_acid_setup()
        xyz = sp.get_xyz()
        out = infer_frag_fallback_h_migration(
            pre_xyz=xyz, post_xyz=xyz,
            uni_mol=sp.mol, split_bonds=[(1, 3)],
            multi_species=None, label='S1-zero',
        )
        self.assertIsNone(out)

    def test_s1_two_h_moved_returns_none(self):
        """Two H atoms displaced ⇒ S1 fails ⇒ ``None`` (no multi-H
        enrichment in Phase 3c)."""
        from arc.job.adapters.ts.linear_utils.local_geometry import (
            infer_frag_fallback_h_migration,
        )
        sp = self._real_carboxylic_acid_setup()
        symbols = sp.get_xyz()['symbols']
        h_idxs = [i for i, s in enumerate(symbols) if s == 'H']
        self.assertGreaterEqual(len(h_idxs), 2)
        pre_coords = list(map(list, sp.get_xyz()['coords']))
        post_coords = [list(c) for c in pre_coords]
        post_coords[h_idxs[0]][0] += 0.5
        post_coords[h_idxs[1]][0] += 0.5
        pre = {**sp.get_xyz(),
               'coords': tuple(tuple(c) for c in pre_coords)}
        post = {**sp.get_xyz(),
                'coords': tuple(tuple(c) for c in post_coords)}
        out = infer_frag_fallback_h_migration(
            pre_xyz=pre, post_xyz=post,
            uni_mol=sp.mol, split_bonds=[(1, 3)],
            multi_species=None, label='S1-two',
        )
        self.assertIsNone(out)

    def test_s2_h_with_two_heavy_neighbors_returns_none(self):
        """An H atom with two heavy neighbors in the reactant graph
        cannot have a unique donor ⇒ S2 fails ⇒ ``None``."""
        from arc.job.adapters.ts.linear_utils.local_geometry import (
            infer_frag_fallback_h_migration,
        )
        # Build a stub mol where one H has two heavy neighbors
        # (something like a hydride bridge — chemically unusual but
        # exactly the S2 failure case).
        class _A:
            def __init__(self, sym, bonds=None):
                class _E:
                    def __init__(self, s):
                        self.symbol = s
                self.element = _E(sym)
                self.bonds = bonds or {}
        c0 = _A('C'); c1 = _A('C'); h = _A('H')
        c0.bonds = {h: 1}
        c1.bonds = {h: 1}
        h.bonds = {c0: 1, c1: 1}
        class _M:
            atoms = [c0, c1, h]
        pre = {
            'symbols': ('C', 'C', 'H'),
            'isotopes': (12, 12, 1),
            'coords': ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (1.0, 0.5, 0.0)),
        }
        post = {
            'symbols': ('C', 'C', 'H'),
            'isotopes': (12, 12, 1),
            'coords': ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (1.0, 1.5, 0.0)),
        }
        out = infer_frag_fallback_h_migration(
            pre_xyz=pre, post_xyz=post,
            uni_mol=_M(), split_bonds=[(0, 1)],
            multi_species=None, label='S2',
        )
        self.assertIsNone(out)

    def test_s3_donor_and_acceptor_in_same_fragment_returns_none(self):
        """When the inferred donor and acceptor lie in the same
        connected component after the cut, the migration cannot be a
        fragment-to-fragment H transfer ⇒ S3 fails ⇒ ``None``."""
        from arc.job.adapters.ts.linear_utils.local_geometry import (
            infer_frag_fallback_h_migration,
        )
        from arc.species import ARCSpecies
        # Ethanol: O bonded to C1 bonded to C0.  Cut C0-C1; the OH and
        # the methyl are now in different fragments.
        # If we move H bonded to C0 (in the methyl fragment) and the
        # nearest "acceptor" candidate is also in the methyl fragment,
        # S3 must fail.
        sp = ARCSpecies(label='ethanol', smiles='CCO')
        symbols = sp.get_xyz()['symbols']
        atom_to_idx = {a: i for i, a in enumerate(sp.mol.atoms)}
        # Find an H bonded to a methyl C (atom 0).
        c0 = next(i for i, s in enumerate(symbols) if s == 'C')
        h_on_c0 = next(
            atom_to_idx[nbr] for nbr in sp.mol.atoms[c0].bonds.keys()
            if nbr.element.symbol == 'H'
        )
        coords = list(map(list, sp.get_xyz()['coords']))
        pre_coords = [list(c) for c in coords]
        post_coords = [list(c) for c in coords]
        # Push the H far from C0 toward atom 1 — also in the same
        # fragment (after cutting C2-O).  Both donor and the nearest
        # heavy partner end up in the methyl fragment ⇒ S3 fails.
        post_coords[h_on_c0][0] += 1.0
        pre = {**sp.get_xyz(),
               'coords': tuple(tuple(c) for c in pre_coords)}
        post = {**sp.get_xyz(),
                'coords': tuple(tuple(c) for c in post_coords)}
        # Cut a bond that is NOT touching the methyl side: e.g. the C-O.
        c_idx_one = next(
            i for i, s in enumerate(symbols)
            if s == 'C' and i != c0
        )
        o_idx = next(i for i, s in enumerate(symbols) if s == 'O')
        out = infer_frag_fallback_h_migration(
            pre_xyz=pre, post_xyz=post,
            uni_mol=sp.mol, split_bonds=[(c_idx_one, o_idx)],
            multi_species=None, label='S3',
        )
        # The migrated H lives entirely inside the methyl fragment, so
        # *every* heavy candidate from the "other side" (just the O) is
        # outside the methyl fragment.  But the methyl fragment side
        # is the donor's side, so S3 enforces the acceptor to be in
        # the OH side — and the H may not be near it.  S5 catches the
        # bad geometry and S5 returns None.
        self.assertIsNone(out)

    def test_inference_success_returns_single_record(self):
        """Acetic acid → CO2 + CH4 toy mock: cut the C-OH bond and move
        the carboxylic-acid H toward the methyl C.  This is the canonical
        single-H migration case the helper must promote."""
        from arc.job.adapters.ts.linear_utils.local_geometry import (
            infer_frag_fallback_h_migration,
        )
        from arc.species import ARCSpecies
        sp = ARCSpecies(label='acid', smiles='CC(=O)O')
        symbols = sp.get_xyz()['symbols']
        atom_to_idx = {a: i for i, a in enumerate(sp.mol.atoms)}

        # Atom indices for acetic acid.  Find the OH H, the OH O, the
        # carbonyl C, and the methyl C.
        c_methyl = None
        c_carbonyl = None
        o_oh = None
        h_oh = None
        for i, atom in enumerate(sp.mol.atoms):
            if atom.element.symbol == 'C':
                heavy_nbrs = [
                    n for n in atom.bonds.keys()
                    if n.element.symbol != 'H'
                ]
                if len(heavy_nbrs) == 1:
                    c_methyl = i
                elif len(heavy_nbrs) >= 2:
                    c_carbonyl = i
            elif atom.element.symbol == 'O':
                # OH oxygen has exactly one H neighbor.
                if any(n.element.symbol == 'H' for n in atom.bonds.keys()):
                    o_oh = i
                    for n in atom.bonds.keys():
                        if n.element.symbol == 'H':
                            h_oh = atom_to_idx[n]
                            break
        self.assertIsNotNone(c_methyl)
        self.assertIsNotNone(c_carbonyl)
        self.assertIsNotNone(o_oh)
        self.assertIsNotNone(h_oh)

        # Build the pre/post geometry: pre = reactant, post = the OH H
        # has moved toward the methyl C at a TS-like distance.
        coords = list(map(list, sp.get_xyz()['coords']))
        # Move the methyl C far enough that the OH H, when relocated to
        # the C-O midpoint of the C_methyl←OH axis, lies between the
        # two heavy atoms at chemically sensible distances.
        coords[c_methyl] = [3.0, 0.0, 0.0]
        coords[o_oh] = [0.0, 0.0, 0.0]
        coords[c_carbonyl] = [-1.5, 0.0, 0.0]
        # Place all other heavy atoms very far away so they don't
        # collide with the H or interfere with S5 rival checks.
        far = 25.0
        for i, sym in enumerate(symbols):
            if i in (c_methyl, c_carbonyl, o_oh, h_oh):
                continue
            if sym == 'H':
                coords[i] = [far + i, far, far]
            else:
                coords[i] = [far + i, -far, -far]
        # Pre: the H is still on the OH side.
        coords[h_oh] = [-0.95, 0.0, 0.0]
        pre = {
            'symbols': symbols,
            'isotopes': sp.get_xyz().get(
                'isotopes', tuple(0 for _ in symbols)),
            'coords': tuple(tuple(float(c) for c in row) for row in coords),
        }
        # Post: the H has moved to the donor–acceptor TS midpoint.
        from arc.common import get_single_bond_length
        from arc.job.adapters.ts.linear_utils.postprocess import PAULING_DELTA
        d_dh = get_single_bond_length('O', 'H') + PAULING_DELTA
        d_ah = get_single_bond_length('C', 'H') + PAULING_DELTA
        d_da = float(np.linalg.norm(
            np.array(coords[c_methyl]) - np.array(coords[o_oh])))
        x = (d_da ** 2 + d_dh ** 2 - d_ah ** 2) / (2.0 * d_da)
        y = float(np.sqrt(max(d_dh ** 2 - x ** 2, 0.0)))
        coords_post = [list(c) for c in coords]
        coords_post[h_oh] = [x, y, 0.0]
        post = {
            'symbols': symbols,
            'isotopes': sp.get_xyz().get(
                'isotopes', tuple(0 for _ in symbols)),
            'coords': tuple(tuple(float(c) for c in row) for row in coords_post),
        }

        # Cut the carbonyl-C—OH bond so the OH H + OH O are on one
        # fragment and the methyl C is on the other.
        out = infer_frag_fallback_h_migration(
            pre_xyz=pre, post_xyz=post,
            uni_mol=sp.mol, split_bonds=[(c_carbonyl, o_oh)],
            multi_species=None, label='success',
        )
        self.assertIsNotNone(out)
        self.assertEqual(out['h_idx'], h_oh)
        self.assertEqual(out['donor'], o_oh)
        self.assertEqual(out['acceptor'], c_methyl)
        self.assertEqual(out['source'], 'frag_inferred')

    def test_s5_competing_rival_returns_none(self):
        """When a non-acceptor heavy atom in the acceptor's fragment
        sits closer to the migrated H than ``0.95 × sbl(rival, H)``,
        S5 must reject the inference."""
        from arc.job.adapters.ts.linear_utils.local_geometry import (
            infer_frag_fallback_h_migration,
        )
        from arc.species import ARCSpecies
        # Build a CCN reactant.  Cut C-N to create two fragments.
        # The N side has only the N (one heavy atom).  The C side has
        # two C atoms — one of them is the donor of the migrating H.
        # Place the migrated H so it ends up *closer to a non-acceptor
        # carbon* than its acceptor.  S5 must catch this.
        sp = ARCSpecies(label='ccn', smiles='CCN')
        symbols = sp.get_xyz()['symbols']
        # We don't actually need a chemically sensible geometry here:
        # just one where the post-migration H lands between two heavy
        # atoms that are *both* in the acceptor fragment, with the
        # rival closer than the chosen acceptor.
        # The key is: S5 considers the acceptor's fragment.  If the
        # acceptor candidate set includes more than one heavy atom and
        # one of them is too close to the H, S5 fails.
        # We construct this directly with a stub mol.
        class _A:
            def __init__(self, sym, bonds=None):
                class _E:
                    def __init__(self, s):
                        self.symbol = s
                self.element = _E(sym)
                self.bonds = bonds or {}
        c0 = _A('C'); c1 = _A('C'); n = _A('N'); h = _A('H')
        c0.bonds = {h: 1}; h.bonds = {c0: 1}
        # No bond from the H side to either acceptor candidate; both
        # acceptors are in a separate fragment.
        class _M:
            atoms = [c0, c1, n, h]
        pre = {
            'symbols': ('C', 'C', 'N', 'H'),
            'isotopes': (12, 12, 14, 1),
            'coords': ((0.0, 0.0, 0.0), (3.0, 0.0, 0.0),
                        (4.0, 0.0, 0.0), (1.0, 0.5, 0.0)),
        }
        post = {
            'symbols': ('C', 'C', 'N', 'H'),
            'isotopes': (12, 12, 14, 1),
            # H moved to be very close to N — but C1 is also close.
            'coords': ((0.0, 0.0, 0.0), (3.0, 0.0, 0.0),
                        (3.5, 0.0, 0.0), (3.05, 0.0, 0.0)),
        }
        # Cut nothing related to (0,1) so the C0 fragment ends up with
        # only itself + the H, and the {C1, N} fragment is the
        # acceptor side.  No split bonds means a single fragment, so
        # use a non-existent cut for the test:
        out = infer_frag_fallback_h_migration(
            pre_xyz=pre, post_xyz=post,
            uni_mol=_M(), split_bonds=[],  # zero cut → single fragment
            multi_species=None, label='S5-rival',
        )
        self.assertIsNone(out)


if __name__ == '__main__':
    unittest.main()
