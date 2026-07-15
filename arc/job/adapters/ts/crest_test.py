#!/usr/bin/env python3
# encoding: utf-8

"""
Unit tests for arc.job.adapters.ts.crest
"""

import os
import tempfile
import unittest

from arc.species.converter import str_to_xyz, xyz_to_str


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

    def test_creates_submit_file_without_crest_templates(self):
        """
        Ensure fallback submit template generation works when submit.py has no CREST templates.
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
            crest_mod.submit_scripts = {"local": {}}
            crest_mod.CREST_PATH = "/usr/bin/crest"
            crest_mod.CREST_ENV_PATH = ""
            crest_mod.SERVERS = {
                "local": {"cluster_soft": "pbs", "cpus": 4, "memory": 8, "queue": "testq"}
            }

            crest_dir = crest_mod.crest_ts_conformer_search(
                xyz_guess=xyz, a_atom=0, h_atom=1, b_atom=2, path=self.tmpdir.name, xyz_crest_int=1
            )

            submit_path = os.path.join(crest_dir, "submit.sh")
            self.assertTrue(os.path.exists(submit_path))
            with open(submit_path) as f:
                submit_text = f.read()
            self.assertIn("#PBS -q testq", submit_text)
            self.assertIn("coords.ref --cinp constraints.inp --noreftopo -T 4", submit_text)
        finally:
            crest_mod.settings = backups["settings"]
            crest_mod.submit_scripts = backups["submit_scripts"]
            crest_mod.CREST_PATH = backups["CREST_PATH"]
            crest_mod.CREST_ENV_PATH = backups["CREST_ENV_PATH"]
            crest_mod.SERVERS = backups["SERVERS"]

    def test_process_completed_jobs_rejects_dissociated_reactive_triad(self):
        """Do not accept a crest_best.xyz whose acceptor has separated from the transferring H."""
        from arc.job.adapters.ts import crest as crest_mod

        reference_xyz = str_to_xyz("""O 0.00000000 -0.02752832 -1.20590500
                                      H 0.00000000 -0.02752832 -0.03383145
                                      O 0.00000000 -0.02752832  1.12142787
                                      H 0.00000000  0.90131726  1.37454478""")
        dissociated_xyz = str_to_xyz("""O -1.1644 0.0000 0.0000
                                        H  0.0000 0.0000 0.0000
                                        O  4.9000 0.0000 0.0000
                                        H  5.8703 0.0000 0.0000""")
        zeus_bad_xyz = str_to_xyz("""O -0.71236464  0.03765902 -0.02937463
                                      H -0.60136223 -0.77534746  0.43444583
                                      O  0.69301187  0.05895500  0.02917997
                                      H  0.90856791 -0.75830305 -0.43135588""")
        crest_path = os.path.join(self.tmpdir.name, 'crest_0')
        os.makedirs(crest_path)
        crest_best_path = os.path.join(crest_path, 'crest_best.xyz')
        with open(crest_best_path, 'w') as f:
            f.write(f"4\nCREST geometry\n{xyz_to_str(dissociated_xyz)}\n")

        jobs = {'123': {'path': crest_path, 'status': 'done'}}
        references = {
            crest_path: {
                'xyz': reference_xyz,
                'reactive_atoms': {'A': 0, 'H': 1, 'B': 2},
            },
        }
        self.assertEqual(crest_mod.process_completed_jobs(jobs, crest_references={}), [])
        self.assertEqual(crest_mod.process_completed_jobs(jobs, crest_references=references), [])

        with open(crest_best_path, 'w') as f:
            f.write(f"4\nCREST geometry\n{xyz_to_str(zeus_bad_xyz)}\n")
        self.assertEqual(crest_mod.process_completed_jobs(jobs, crest_references=references), [])

        with open(crest_best_path, 'w') as f:
            f.write(f"4\nCREST geometry\n{xyz_to_str(reference_xyz)}\n")
        self.assertEqual(
            crest_mod.process_completed_jobs(jobs, crest_references=references),
            [reference_xyz],
        )


if __name__ == "__main__":
    unittest.main()
