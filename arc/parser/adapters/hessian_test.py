#!/usr/bin/env python3
# encoding: utf-8

"""
Unit tests for Cartesian-Hessian parsing in the Gaussian and Orca ESS adapters
(``parse_cartesian_hessian_lower_triangle`` and its frame-matched geometry
partner ``parse_cartesian_hessian_geometry``).

The parsers return the packed lower triangle (including the diagonal),
row-major, in **native atomic units (hartree/bohr²)** — no SI conversion is
applied. Beyond shape and native-unit sanity, these tests assert a *physical*
invariant: mass-weighting the parsed triangle against the parsed geometry and
projecting out translation and rotation must reproduce the frequencies the ESS
itself reported. That single check catches a transpose, a unit slip, a
mis-paged column block, and a frame mismatch — none of which change the
triangle's length. ARC's parser tests deliberately assert only ARC-native
scientific values and do not depend on a downstream consumer's schema package.
"""

import math
import os
import unittest

import numpy as np

from arc.common import ARC_PATH
from arc.constants import E_h, a0, amu, c
from arc.parser.adapters.gaussian import GaussianParser
from arc.parser.adapters.orca import OrcaParser
from arc.species.converter import get_element_mass_from_xyz

ARC_TESTING_PATH = os.path.join(ARC_PATH, 'arc', 'testing')


def _diagonal(triangle):
    """Extract the diagonal from a packed row-major lower triangle."""
    # In the packed lower triangle, H[i][i] sits at index i*(i+1)//2 + i.
    n = 0
    while (n + 1) * (n + 2) // 2 <= len(triangle):
        n += 1
    return [triangle[i * (i + 1) // 2 + i] for i in range(n)]


def _unpack(triangle: list[float]) -> np.ndarray:
    """Rebuild the full symmetric matrix from a packed row-major lower triangle."""
    dimension = int(round((math.isqrt(8 * len(triangle) + 1) - 1) / 2))
    matrix = np.zeros((dimension, dimension), dtype=np.float64)
    index = 0
    for row in range(dimension):
        for col in range(row + 1):
            matrix[row, col] = matrix[col, row] = triangle[index]
            index += 1
    return matrix


def vibrational_frequencies_cm1(triangle: list[float], xyz: dict, n_expected: int) -> np.ndarray:
    """
    Reconstruct harmonic frequencies from a parsed Hessian and its geometry.

    Mass-weights the Hessian, projects out the six (five for a linear species)
    translational and rotational directions, and converts the remaining
    eigenvalues to wavenumbers. Imaginary modes come back negative, matching how
    ESSs report them.

    Args:
        triangle (list): The packed lower triangle in hartree/bohr².
        xyz (dict): The geometry, in the *same* Cartesian frame as ``triangle``.
        n_expected (int): How many vibrational modes to return.

    Returns:
        np.ndarray: ``n_expected`` frequencies in cm^-1, ascending.
    """
    hessian = _unpack(triangle) * E_h / a0 ** 2                  # J/m^2
    masses = np.array(get_element_mass_from_xyz(xyz)) * amu      # kg
    coords = np.array(xyz['coords']) * 1e-10                     # m
    root_mass = np.repeat(np.sqrt(masses), 3)
    weighted = hessian / np.outer(root_mass, root_mass)          # s^-2

    dimension = 3 * len(masses)
    centered = coords - (masses[:, None] * coords).sum(axis=0) / masses.sum()
    rigid = np.zeros((dimension, 6))
    for axis in range(3):
        vector = np.zeros(dimension)
        vector[axis::3] = 1.0
        rigid[:, axis] = vector * root_mass
    for index, (p, q) in enumerate(((1, 2), (2, 0), (0, 1))):
        vector = np.zeros(dimension)
        vector[p::3] = centered[:, q]
        vector[q::3] = -centered[:, p]
        rigid[:, 3 + index] = vector * root_mass
    basis, _ = np.linalg.qr(rigid)
    projector = np.eye(dimension) - basis @ basis.T

    eigenvalues = np.linalg.eigvalsh(projector @ weighted @ projector)
    frequencies = np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues)) / (2 * np.pi * c) / 100.0
    vibrational = frequencies[np.argsort(np.abs(frequencies))[-n_expected:]]
    return np.sort(vibrational)


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
        # A Q-Chem log has no Gaussian FC block; GaussianParser must return
        # None rather than raise.
        path = os.path.join(ARC_TESTING_PATH, 'freq', 'C2H6_freq_QChem.out')
        triangle = GaussianParser(path).parse_cartesian_hessian_lower_triangle()
        self.assertIsNone(triangle)

    def test_hessian_geometry_is_the_input_orientation(self):
        """The frame-matched geometry differs from ``parse_geometry``'s standard orientation."""
        path = os.path.join(ARC_TESTING_PATH, 'restart', '3_restart_bde', 'calcs', 'Species',
                            'anilino_radical', 'freq_a4345', 'output.out')
        parser = GaussianParser(path)
        frame_xyz, frame = parser.parse_cartesian_hessian_geometry()
        self.assertEqual(frame, 'gaussian_input_orientation')
        standard = parser.parse_geometry()
        self.assertEqual(frame_xyz['symbols'], standard['symbols'])
        # Same molecule, different frame: internal distances agree, Cartesian
        # coordinates do not. This is exactly the difference that silently
        # corrupts a reconstructed spectrum.
        frame_coords, standard_coords = np.array(frame_xyz['coords']), np.array(standard['coords'])
        self.assertGreater(np.abs(frame_coords - standard_coords).max(), 1.0)

        def distances(coords):
            return np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)

        self.assertLess(np.abs(distances(frame_coords) - distances(standard_coords)).max(), 1e-5)

    def test_hessian_reproduces_reported_frequencies(self):
        """Mass-weighting the triangle against its frame-matched geometry reproduces the log's frequencies."""
        path = os.path.join(ARC_TESTING_PATH, 'restart', '3_restart_bde', 'calcs', 'Species',
                            'anilino_radical', 'freq_a4345', 'output.out')
        parser = GaussianParser(path)
        triangle = parser.parse_cartesian_hessian_lower_triangle()
        frame_xyz, _ = parser.parse_cartesian_hessian_geometry()
        reported = np.sort(np.array(parser.parse_frequencies()))
        reconstructed = vibrational_frequencies_cm1(triangle, frame_xyz, len(reported))
        self.assertLess(np.abs(reconstructed - reported).max(), 1.0)

    def test_standard_orientation_does_not_reproduce_frequencies(self):
        """Pin the defect the frame-matched geometry exists to avoid.

        Pairing the Hessian with ``parse_geometry``'s standard orientation is a
        pure rotation away and reconstructs a materially different spectrum.
        Nothing about the triangle's length or finiteness changes, so only a
        physical check like this one can detect it.
        """
        path = os.path.join(ARC_TESTING_PATH, 'restart', '3_restart_bde', 'calcs', 'Species',
                            'anilino_radical', 'freq_a4345', 'output.out')
        parser = GaussianParser(path)
        triangle = parser.parse_cartesian_hessian_lower_triangle()
        reported = np.sort(np.array(parser.parse_frequencies()))
        wrong_frame = vibrational_frequencies_cm1(triangle, parser.parse_geometry(), len(reported))
        self.assertGreater(np.abs(wrong_frame - reported).max(), 50.0)


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
        # Native hartree/bohr²: never ~1e3, which would signal an SI J/m² leak.
        self.assertLess(max(abs(v) for v in triangle), 5.0)
        self.assertGreater(max(_diagonal(triangle)), 0.1)

    def test_hessian_geometry_comes_from_the_hess_atoms_block(self):
        """The frame-matched geometry is the ``.hess`` ``$atoms`` block, converted from Bohr."""
        xyz, frame = OrcaParser(self.log_path).parse_cartesian_hessian_geometry()
        self.assertEqual(frame, 'orca_hess_atoms')
        self.assertEqual(xyz['symbols'], ('O', 'H', 'H'))
        # $atoms holds Bohr; the parser must return Angstrom. The O-H bond is
        # ~0.96 A (~1.81 bohr), so an unconverted value would be far too long.
        coords = np.array(xyz['coords'])
        self.assertAlmostEqual(float(np.linalg.norm(coords[1] - coords[0])), 0.9578, places=3)

    def test_hessian_reproduces_reported_frequencies(self):
        """The .hess triangle and its $atoms geometry reproduce the log's reported frequencies."""
        parser = OrcaParser(self.log_path)
        triangle = parser.parse_cartesian_hessian_lower_triangle()
        xyz, _ = parser.parse_cartesian_hessian_geometry()
        reported = np.sort(np.array(parser.parse_frequencies()))
        self.assertEqual(len(reported), 3)
        reconstructed = vibrational_frequencies_cm1(triangle, xyz, len(reported))
        self.assertLess(np.abs(reconstructed - reported).max(), 1.0)

    def test_dimension_is_bounded_by_the_declared_atom_count(self):
        """A ``$hessian`` dimension inconsistent with ``$atoms`` is declined, not allocated.

        ``n_rows`` is file content that sizes two O(n^2) allocations, so a
        truncated or corrupt header must not drive a multi-terabyte request.
        """
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'input.hess')
            with open(path, 'w') as f:
                f.write('$orca_hessian_file\n\n$hessian\n2000000\n\n'
                        '$atoms\n3\n O 15.99491 0.0 0.0 0.0\n'
                        ' H 1.00783 1.8 0.0 0.0\n H 1.00783 0.0 1.8 0.0\n\n$end\n')
            with open(os.path.join(tmp, 'output.out'), 'w') as f:
                f.write('* O   R   C   A *\n')
            triangle = OrcaParser(os.path.join(tmp, 'output.out')).parse_cartesian_hessian_lower_triangle()
            self.assertIsNone(triangle)

    def test_missing_hess_returns_none(self):
        """No sibling .hess (a plain Orca log) returns None, never raises."""
        path = os.path.join(ARC_TESTING_PATH, 'freq', 'orca_example_freq.log')
        triangle = OrcaParser(path).parse_cartesian_hessian_lower_triangle()
        self.assertIsNone(triangle)


if __name__ == '__main__':
    unittest.main()
