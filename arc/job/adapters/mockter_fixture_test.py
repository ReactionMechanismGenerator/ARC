"""
Tests for the mockter fixture loader.
"""

import os
import tempfile
import unittest

from arc.common import save_yaml_file
from arc.job.adapters.mockter_fixture import (
    Fixture,
    FixtureEntry,
    FixtureError,
    SUPPORTED_SCHEMA_VERSIONS,
    VALID_RAISE_KINDS,
)


FIXTURES_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'testing', 'mockter_fixtures')


class TestFixtureLoad(unittest.TestCase):
    """Schema validation at load time."""

    def test_missing_file_raises(self):
        """Loading a nonexistent path raises FixtureError."""
        with self.assertRaises(FixtureError):
            Fixture.load('/nonexistent/path/to/fixture.yml')

    def test_unsupported_schema_version_raises(self):
        """Wrong schema_version is rejected even if everything else is valid."""
        with tempfile.NamedTemporaryFile('w', suffix='.yml', delete=False) as f:
            path = f.name
        try:
            save_yaml_file(path=path, content={'schema_version': 999, 'species': {}})
            with self.assertRaises(FixtureError):
                Fixture.load(path)
        finally:
            os.unlink(path)

    def test_missing_species_key_raises(self):
        """Top-level 'species' is required."""
        with tempfile.NamedTemporaryFile('w', suffix='.yml', delete=False) as f:
            path = f.name
        try:
            save_yaml_file(path=path, content={'schema_version': 1})
            with self.assertRaises(FixtureError):
                Fixture.load(path)
        finally:
            os.unlink(path)

    def test_supported_versions_set_includes_one(self):
        """Sanity: schema v1 is in the supported set."""
        self.assertIn(1, SUPPORTED_SCHEMA_VERSIONS)


class TestFixtureLookup(unittest.TestCase):
    """Lookup resolution against a real committed fixture."""

    @classmethod
    def setUpClass(cls):
        """Load mockter1 (n-butane, multi-conformer + scans)."""
        path = os.path.join(FIXTURES_DIR, 'mockter1.yml')
        if not os.path.isfile(path):
            raise unittest.SkipTest('mockter1.yml not present')
        cls.fixture = Fixture.load(path)

    def test_unknown_label_returns_none(self):
        """Lookup with a label that doesn't exist returns None."""
        self.assertIsNone(self.fixture.lookup('not_a_species', 'opt'))

    def test_opt_lookup_returns_payload(self):
        """opt lookup returns a FixtureEntry with xyz and e_elect."""
        entry = self.fixture.lookup('n_butane', 'opt')
        self.assertIsNotNone(entry)
        self.assertIsInstance(entry, FixtureEntry)
        self.assertIn('xyz', entry.payload)
        self.assertIn('e_elect', entry.payload)

    def test_fine_opt_distinct_from_opt(self):
        """fine=True selects fine_opt; fine=False selects opt."""
        coarse = self.fixture.lookup('n_butane', 'opt', fine=False)
        fine = self.fixture.lookup('n_butane', 'opt', fine=True)
        self.assertIsNotNone(coarse)
        self.assertIsNotNone(fine)
        self.assertEqual(coarse.source_path[-1], 'opt')
        self.assertEqual(fine.source_path[-1], 'fine_opt')

    def test_freq_lookup_carries_freqs_zpe_hessian(self):
        """freq entry exposes freqs, zpe, hessian_block."""
        entry = self.fixture.lookup('n_butane', 'freq')
        self.assertIsNotNone(entry)
        self.assertIn('freqs', entry.payload)
        self.assertIn('zpe', entry.payload)
        self.assertIn('hessian_block', entry.payload)

    def test_sp_lookup(self):
        """sp entry has e_elect and (optionally) t1_diagnostic."""
        entry = self.fixture.lookup('n_butane', 'sp')
        self.assertIsNotNone(entry)
        self.assertIn('e_elect', entry.payload)

    def test_conformer_lookup_by_index(self):
        """conf_opt with conformer=N returns the Nth conformer entry."""
        entry = self.fixture.lookup('n_butane', 'conf_opt', conformer=0)
        self.assertIsNotNone(entry)
        self.assertEqual(entry.source_path, ('n_butane', 'conformers', 0))
        self.assertIn('xyz', entry.payload)

    def test_conformer_index_out_of_range_returns_none(self):
        """conformer index beyond available conformers returns None."""
        entry = self.fixture.lookup('n_butane', 'conf_opt', conformer=999)
        self.assertIsNone(entry)

    def test_scan_lookup_by_torsions_matches_canonically(self):
        """scan torsions match regardless of input order."""
        scans = self.fixture.species['n_butane'].get('scans') or []
        if not scans:
            self.skipTest('n_butane has no scans in fixture')
        torsions = scans[0]['torsions']
        entry = self.fixture.lookup('n_butane', 'scan', torsions=torsions)
        self.assertIsNotNone(entry)
        self.assertEqual(entry.payload['energies'], scans[0]['energies'])

    def test_scan_lookup_no_torsions_returns_first(self):
        """scan without torsions argument returns the first scan."""
        scans = self.fixture.species['n_butane'].get('scans') or []
        if not scans:
            self.skipTest('n_butane has no scans in fixture')
        entry = self.fixture.lookup('n_butane', 'scan')
        self.assertIsNotNone(entry)

    def test_unknown_job_type_returns_none(self):
        """Unsupported job_type returns None rather than raising."""
        self.assertIsNone(self.fixture.lookup('n_butane', 'orbitals'))

    def test_composite_lookup_present_in_s3(self):
        """mockter3 has a composite block on methanol."""
        path = os.path.join(FIXTURES_DIR, 'mockter3.yml')
        if not os.path.isfile(path):
            self.skipTest('mockter3.yml not present')
        f3 = Fixture.load(path)
        entry = f3.lookup('methanol', 'composite')
        self.assertIsNotNone(entry)
        self.assertIn('e_elect', entry.payload)
        self.assertIn('hessian_block', entry.payload)

    def test_ts_lookup_in_s5(self):
        """mockter5 has a TS — looked up via is_ts=True."""
        path = os.path.join(FIXTURES_DIR, 'mockter5.yml')
        if not os.path.isfile(path):
            self.skipTest('mockter5.yml not present')
        f5 = Fixture.load(path)
        entry = f5.lookup('TS0', 'opt', is_ts=True)
        self.assertIsNotNone(entry)
        self.assertIn('xyz', entry.payload)


class TestFixtureRaiseEntries(unittest.TestCase):
    """``raise:`` clause semantics."""

    def setUp(self):
        """Build an in-memory fixture with one raise entry per supported kind."""
        self.fd, self.path = tempfile.mkstemp(suffix='.yml')
        os.close(self.fd)
        save_yaml_file(path=self.path, content={
            'schema_version': 1,
            'species': {
                'X': {
                    'conformers': [
                        {'xyz': 'H 0 0 0', 'e_elect': -0.5},
                        {'raise': {'type': 'sigterm', 'message': 'simulated walltime'}},
                        {'raise': {'type': 'oom'}},
                        {'raise': {'type': 'totally_made_up'}},
                    ],
                },
            },
        })

    def tearDown(self):
        os.unlink(self.path)

    def test_normal_entry_is_not_raise(self):
        """A normal payload entry returns is_raise() == False."""
        f = Fixture.load(self.path)
        entry = f.lookup('X', 'conf_opt', conformer=0)
        self.assertFalse(entry.is_raise())

    def test_raise_entry_kind_extracted(self):
        """A raise entry returns the requested kind via raise_kind()."""
        f = Fixture.load(self.path)
        entry = f.lookup('X', 'conf_opt', conformer=1)
        self.assertTrue(entry.is_raise())
        self.assertEqual(entry.raise_kind(), 'sigterm')
        self.assertEqual(entry.raise_message(), 'simulated walltime')

    def test_raise_entry_no_message_returns_none(self):
        """raise_message() returns None when no message is set."""
        f = Fixture.load(self.path)
        entry = f.lookup('X', 'conf_opt', conformer=2)
        self.assertEqual(entry.raise_kind(), 'oom')
        self.assertIsNone(entry.raise_message())

    def test_invalid_raise_kind_rejected(self):
        """raise_kind() raises FixtureError for unrecognized values."""
        f = Fixture.load(self.path)
        entry = f.lookup('X', 'conf_opt', conformer=3)
        with self.assertRaises(FixtureError):
            entry.raise_kind()

    def test_valid_raise_kinds_complete(self):
        """The kinds enumerated in the module match the schema's contract."""
        self.assertEqual(
            VALID_RAISE_KINDS,
            {'crash', 'oom', 'timeout', 'scf_nonconvergence', 'sigterm'},
        )


if __name__ == '__main__':
    unittest.main()
