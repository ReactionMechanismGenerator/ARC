#!/usr/bin/env python3
# encoding: utf-8

"""
Unit tests for Cartesian-Hessian parsing in the Gaussian and Orca ESS adapters
(``parse_cartesian_hessian_lower_triangle``).

The parsers return the packed lower triangle (including the diagonal),
row-major, in **native atomic units (hartree/bohr²)** — no SI conversion is
applied. These tests assert the triangle length (``3N(3N+1)/2``), native-unit
sanity (diagonal force constants are O(0.1-1.5), not ~1e3 which would signal an
SI J/m² leak), and the graceful-absence path. ARC's parser tests deliberately
assert only ARC-native scientific values and do not depend on a downstream
consumer's schema package.
"""

import os
import unittest

from arc.common import ARC_PATH
from arc.parser.adapters.gaussian import GaussianParser
from arc.parser.adapters.orca import OrcaParser

ARC_TESTING_PATH = os.path.join(ARC_PATH, 'arc', 'testing')


def _diagonal(triangle):
    """Extract the diagonal from a packed row-major lower triangle."""
    # In the packed lower triangle, H[i][i] sits at index i*(i+1)//2 + i.
    n = 0
    while (n + 1) * (n + 2) // 2 <= len(triangle):
        n += 1
    return [triangle[i * (i + 1) // 2 + i] for i in range(n)]


class TestGaussianCartesianHessian(unittest.TestCase):
    """Gaussian ``Force constants in Cartesian coordinates:`` block parsing."""

    def test_parse_two_atom_hessian(self):
        """A real 2-atom (NH) Gaussian freq log yields a 21-entry triangle."""
        path = os.path.join(ARC_TESTING_PATH, 'restart', '2_restart_rate',
                            'calcs', 'Species', 'NH_freq.out')
        triangle = GaussianParser(path).parse_cartesian_hessian_lower_triangle()
        self.assertIsNotNone(triangle)
        n_atoms = 2
        self.assertEqual(len(triangle), (3 * n_atoms) * (3 * n_atoms + 1) // 2)
        self.assertEqual(len(triangle), 21)
        # Native hartree/bohr²: the N-H stretch diagonal is ~0.39, the
        # transverse components ~6e-5 — all O(1) or smaller. If Arkane's SI
        # conversion had leaked in, the stretch would be ~1e3 (J/m²).
        diag = _diagonal(triangle)
        self.assertAlmostEqual(diag[2], 0.389752, places=5)
        self.assertLess(max(abs(v) for v in triangle), 5.0)

    def test_parse_three_atom_hessian(self):
        """A real 3-atom (CHO) Gaussian freq log yields a 45-entry triangle."""
        path = os.path.join(ARC_TESTING_PATH, 'freq', 'CHO_neg_freq.out')
        triangle = GaussianParser(path).parse_cartesian_hessian_lower_triangle()
        self.assertIsNotNone(triangle)
        self.assertEqual(len(triangle), 45)  # 3N=9 -> 9*10/2
        # Native units: diagonal force constants are O(0.1-1.5), never ~1e3.
        self.assertLess(max(abs(v) for v in triangle), 5.0)
        self.assertGreater(max(_diagonal(triangle)), 0.1)

    def test_absent_block_returns_none(self):
        """A Gaussian log without the FC block (no IOp(7/33=1)) returns None."""
        # ``CH3OH_freq.out`` is a plain freq log; if it lacks the block the
        # parser must decline gracefully rather than raise. Any log without
        # the block exercises the same path — use one with only frequencies.
        path = os.path.join(ARC_TESTING_PATH, 'freq', 'C2H6_freq_QChem.out')
        # A Q-Chem log has no Gaussian FC block; GaussianParser must return
        # None rather than raise.
        triangle = GaussianParser(path).parse_cartesian_hessian_lower_triangle()
        self.assertIsNone(triangle)

class TestOrcaCartesianHessian(unittest.TestCase):
    """Orca sibling ``.hess`` file (``$hessian`` block) parsing."""

    def setUp(self):
        self.orca_dir = os.path.join(ARC_TESTING_PATH, 'freq', 'orca_hessian_h2o')
        self.log_path = os.path.join(self.orca_dir, 'output.out')

    def test_locate_sibling_hess(self):
        """The parser finds the ``input.hess`` sibling next to the log."""
        located = OrcaParser(self.log_path)._locate_hess_file()
        self.assertIsNotNone(located)
        self.assertTrue(located.endswith('input.hess'))

    def test_parse_three_atom_hessian(self):
        """The 3-atom (H2O) Orca .hess yields a 45-entry native-unit triangle."""
        triangle = OrcaParser(self.log_path).parse_cartesian_hessian_lower_triangle()
        self.assertIsNotNone(triangle)
        self.assertEqual(len(triangle), 45)  # 3N=9 -> 9*10/2
        # Native hartree/bohr²: diagonals ~0.3-0.6, never ~1e3.
        diag = _diagonal(triangle)
        self.assertTrue(all(0.1 < v < 1.0 for v in diag))
        self.assertLess(max(abs(v) for v in triangle), 5.0)

    def test_missing_hess_returns_none(self):
        """No sibling .hess (a plain Orca log) returns None, never raises."""
        path = os.path.join(ARC_TESTING_PATH, 'freq', 'orca_example_freq.log')
        triangle = OrcaParser(path).parse_cartesian_hessian_lower_triangle()
        self.assertIsNone(triangle)

if __name__ == '__main__':
    unittest.main()
