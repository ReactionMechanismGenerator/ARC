#!/usr/bin/env python3
# encoding: utf-8

"""
Unit tests for arc.job.adapters.ts.crest
"""

import os
import tempfile
import unittest

from arc.species.converter import str_to_xyz


class TestCrestAdapter(unittest.TestCase):
    """
    Tests for CREST input generation.
    """

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_creates_valid_input_files(self):
        """
        Ensure CREST inputs are written with expected content/format.
        """
        from arc.job.adapters.ts import crest as crest_mod

        xyz = str_to_xyz(
            """O 0.0 0.0 0.0
               H 0.0 0.0 0.96
               H 0.9 0.0 0.0"""
        )

        backups = {
            "settings": crest_mod.settings,
            "submit_scripts": crest_mod.submit_scripts,
            "CREST_PATH": crest_mod.CREST_PATH,
            "CREST_ENV_PATH": crest_mod.CREST_ENV_PATH,
            "SERVERS": crest_mod.SERVERS,
        }

        try:
            crest_mod.settings = {"submit_filenames": {"PBS": "submit.sh"}}
            crest_mod.submit_scripts = {
                "local": {
                    "crest": (
                        "#PBS -q {queue}\n"
                        "#PBS -N {name}\n"
                        "#PBS -l select=1:ncpus={cpus}:mem={memory}gb\n"
                    ),
                    "crest_job": "{activation_line}\ncd {path}\n{commands}\n",
                }
            }
            crest_mod.CREST_PATH = "/usr/bin/crest"
            crest_mod.CREST_ENV_PATH = ""
            crest_mod.SERVERS = {
                "local": {"cluster_soft": "pbs", "cpus": 4, "memory": 8, "queue": "testq"}
            }

            crest_dir = crest_mod.crest_ts_conformer_search(
                xyz_guess=xyz, a_atom=0, h_atom=1, b_atom=2, path=self.tmpdir.name, xyz_crest_int=0
            )

            coords_path = os.path.join(crest_dir, "coords.ref")
            constraints_path = os.path.join(crest_dir, "constraints.inp")
            submit_path = os.path.join(crest_dir, "submit.sh")

            self.assertTrue(os.path.exists(coords_path))
            self.assertTrue(os.path.exists(constraints_path))
            self.assertTrue(os.path.exists(submit_path))

            with open(coords_path) as f:
                coords = f.read().strip().splitlines()
            self.assertEqual(coords[0].strip(), "$coord")
            self.assertEqual(coords[-1].strip(), "$end")
            self.assertEqual(len(coords) - 2, len(xyz["symbols"]))

            with open(constraints_path) as f:
                constraints = f.read()
            self.assertIn("atoms: 1, 2, 3", constraints)
            self.assertIn("force constant: 0.5", constraints)
            self.assertIn("reference=coords.ref", constraints)
            self.assertIn("distance: 1, 2, auto", constraints)
            self.assertIn("distance: 2, 3, auto", constraints)
            self.assertIn("$metadyn", constraints)
            self.assertTrue(constraints.strip().endswith("$end"))
        finally:
            crest_mod.settings = backups["settings"]
            crest_mod.submit_scripts = backups["submit_scripts"]
            crest_mod.CREST_PATH = backups["CREST_PATH"]
            crest_mod.CREST_ENV_PATH = backups["CREST_ENV_PATH"]
            crest_mod.SERVERS = backups["SERVERS"]


if __name__ == "__main__":
    unittest.main()
