"""Tests for the Gaussian + ORCA constraint parsers."""

import os
import tempfile
import unittest

from arc.parser.adapters.gaussian import (
    _gaussian_letter_to_tckdb_kind,
    parse_gaussian_constraints,
)
from arc.parser.adapters.orca import parse_orca_constraints


def _write(text: str) -> str:
    """Write ``text`` to a NamedTemporaryFile and return the path.

    The caller is responsible for cleanup; tests use ``addCleanup``.
    """
    fd, path = tempfile.mkstemp(suffix='.gjf')
    os.close(fd)
    with open(path, 'w') as f:
        f.write(text)
    return path


class TestGaussianLetterClassifier(unittest.TestCase):
    """Direct tests of the letter → constraint_kind helper."""

    def test_known_letters_with_correct_arity(self):
        self.assertEqual(_gaussian_letter_to_tckdb_kind('X', 1), 'cartesian_atom')
        self.assertEqual(_gaussian_letter_to_tckdb_kind('B', 2), 'bond')
        self.assertEqual(_gaussian_letter_to_tckdb_kind('A', 3), 'angle')
        self.assertEqual(_gaussian_letter_to_tckdb_kind('D', 4), 'dihedral')

    def test_letters_with_wrong_arity_return_none(self):
        self.assertIsNone(_gaussian_letter_to_tckdb_kind('B', 3))
        self.assertIsNone(_gaussian_letter_to_tckdb_kind('A', 2))
        self.assertIsNone(_gaussian_letter_to_tckdb_kind('D', 3))
        self.assertIsNone(_gaussian_letter_to_tckdb_kind('X', 2))

    def test_non_constraint_letters_return_none(self):
        self.assertIsNone(_gaussian_letter_to_tckdb_kind('L', 4))
        self.assertIsNone(_gaussian_letter_to_tckdb_kind('O', 3))

    def test_unknown_letter_returns_none(self):
        self.assertIsNone(_gaussian_letter_to_tckdb_kind('Q', 2))
        self.assertIsNone(_gaussian_letter_to_tckdb_kind('', 2))


class TestGaussianInputDeckConstraints(unittest.TestCase):
    """Parsing held-fixed constraints from Gaussian input decks."""

    def setUp(self):
        self._paths: list[str] = []

    def tearDown(self):
        for p in self._paths:
            try:
                os.remove(p)
            except FileNotFoundError:
                # Test may have already removed the file; any other
                # OSError (permission, busy) should surface, not be hidden.
                pass

    def _deck(self, body: str) -> str:
        path = _write(body)
        self._paths.append(path)
        return path

    def test_bond_freeze(self):
        path = self._deck("B 1 2 F\n")
        result = parse_gaussian_constraints(path)
        self.assertEqual(result, [
            {'constraint_kind': 'bond', 'atoms': [1, 2], 'target_value': None},
        ])

    def test_angle_freeze(self):
        path = self._deck("A 1 2 3 F\n")
        result = parse_gaussian_constraints(path)
        self.assertEqual(result, [
            {'constraint_kind': 'angle', 'atoms': [1, 2, 3], 'target_value': None},
        ])

    def test_dihedral_freeze(self):
        path = self._deck("D 1 2 3 4 F\n")
        result = parse_gaussian_constraints(path)
        self.assertEqual(result, [
            {'constraint_kind': 'dihedral', 'atoms': [1, 2, 3, 4], 'target_value': None},
        ])

    def test_cartesian_atom_freeze(self):
        path = self._deck("X 7 F\n")
        result = parse_gaussian_constraints(path)
        self.assertEqual(result, [
            {'constraint_kind': 'cartesian_atom', 'atoms': [7], 'target_value': None},
        ])

    def test_target_value_with_equals_sign(self):
        path = self._deck("D 1 2 3 4 = 180.0 F\n")
        result = parse_gaussian_constraints(path)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0]['target_value'], 180.0)
        self.assertEqual(result[0]['constraint_kind'], 'dihedral')

    def test_target_value_bare(self):
        path = self._deck("B 1 2 1.45 F\n")
        result = parse_gaussian_constraints(path)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0]['target_value'], 1.45)

    def test_implicit_freeze_no_action_code(self):
        # ARC's ``_load_scan_specs`` treats no-action-code lines as F.
        path = self._deck("B 1 2\n")
        result = parse_gaussian_constraints(path)
        self.assertEqual(result, [
            {'constraint_kind': 'bond', 'atoms': [1, 2], 'target_value': None},
        ])

    def test_scan_line_is_not_a_constraint(self):
        # An ``S`` line is the active scan coordinate — must NOT appear in
        # the constraint list. It belongs in scan_result.coordinates[].
        path = self._deck("D 1 2 3 4 S 36 10.0\n")
        self.assertEqual(parse_gaussian_constraints(path), [])

    def test_mixed_scan_and_freeze(self):
        path = self._deck(
            "D 1 2 3 4 S 36 10.0\n"
            "B 5 6 F\n"
            "A 7 8 9 F\n"
        )
        result = parse_gaussian_constraints(path)
        self.assertEqual(result, [
            {'constraint_kind': 'bond', 'atoms': [5, 6], 'target_value': None},
            {'constraint_kind': 'angle', 'atoms': [7, 8, 9], 'target_value': None},
        ])

    def test_linear_bend_letter_skipped(self):
        path = self._deck(
            "L 1 2 3 4 F\n"
            "B 5 6 F\n"
        )
        result = parse_gaussian_constraints(path)
        self.assertEqual(result, [
            {'constraint_kind': 'bond', 'atoms': [5, 6], 'target_value': None},
        ])

    def test_malformed_line_skipped(self):
        # Letter B but only one atom token; must be skipped, not raise.
        path = self._deck(
            "B 1 F\n"
            "A 1 2 3 F\n"
        )
        result = parse_gaussian_constraints(path)
        self.assertEqual(result, [
            {'constraint_kind': 'angle', 'atoms': [1, 2, 3], 'target_value': None},
        ])

    def test_unknown_letter_skipped(self):
        path = self._deck(
            "Q 1 2 F\n"
            "B 3 4 F\n"
        )
        result = parse_gaussian_constraints(path)
        self.assertEqual(result, [
            {'constraint_kind': 'bond', 'atoms': [3, 4], 'target_value': None},
        ])

    def test_missing_file_returns_empty_list(self):
        self.assertEqual(parse_gaussian_constraints('/nonexistent/path.gjf'), [])


class TestGaussianLogConstraints(unittest.TestCase):
    """Parsing held-fixed constraints from Gaussian log files."""

    def test_modredundant_block_in_log(self):
        log_body = (
            "  Some preamble\n"
            "  ...\n"
            " The following ModRedundant input section has been read:\n"
            " D       4       1       2       5 S  36 10.000\n"
            " B       1       2 F\n"
            " A       3       4       5 F\n"
            "\n"
            " Isotopes and Nuclear Properties:\n"
        )
        fd, path = tempfile.mkstemp(suffix='.log')
        os.close(fd)
        self.addCleanup(os.remove, path)
        with open(path, 'w') as f:
            f.write(log_body)

        result = parse_gaussian_constraints(path)
        self.assertEqual(result, [
            {'constraint_kind': 'bond', 'atoms': [1, 2], 'target_value': None},
            {'constraint_kind': 'angle', 'atoms': [3, 4, 5], 'target_value': None},
        ])


class TestOrcaConstraints(unittest.TestCase):
    """Defensive ORCA constraint parsing — atom indices converted to 1-based."""

    def setUp(self):
        self._paths: list[str] = []

    def tearDown(self):
        for p in self._paths:
            try:
                os.remove(p)
            except FileNotFoundError:
                # Test may have already removed the file; any other
                # OSError (permission, busy) should surface, not be hidden.
                pass

    def _deck(self, body: str) -> str:
        fd, path = tempfile.mkstemp(suffix='.in')
        os.close(fd)
        self._paths.append(path)
        with open(path, 'w') as f:
            f.write(body)
        return path

    def test_full_constraints_block(self):
        path = self._deck(
            "! B3LYP def2-SVP Opt\n"
            "%geom Constraints\n"
            "  { B 0 1 1.4 C }\n"
            "  { A 0 1 2 90.0 C }\n"
            "  { D 0 1 2 3 180.0 C }\n"
            "  { C 4 C }\n"
            "end\n"
            "* xyz 0 1\n"
        )
        result = parse_orca_constraints(path)
        # ORCA is 0-based; TCKDB is 1-based.
        self.assertEqual(result, [
            {'constraint_kind': 'bond', 'atoms': [1, 2], 'target_value': 1.4},
            {'constraint_kind': 'angle', 'atoms': [1, 2, 3], 'target_value': 90.0},
            {'constraint_kind': 'dihedral', 'atoms': [1, 2, 3, 4], 'target_value': 180.0},
            {'constraint_kind': 'cartesian_atom', 'atoms': [5], 'target_value': None},
        ])

    def test_no_constraints_block(self):
        path = self._deck("! B3LYP def2-SVP Opt\n* xyz 0 1\n")
        self.assertEqual(parse_orca_constraints(path), [])

    def test_missing_file(self):
        self.assertEqual(parse_orca_constraints('/nonexistent/path.in'), [])


if __name__ == '__main__':
    unittest.main()
