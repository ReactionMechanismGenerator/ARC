#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.job.adapters.ts.autotst_ts module
"""

import os
import shutil
import unittest

from arc.common import ARC_TESTING_PATH
import arc.job.adapters.common as common
from arc.job.adapters.ts.autotst_ts import AutoTSTAdapter
from arc.reaction import ARCReaction
from arc.species import ARCSpecies


class TestAutoTSTAdapter(unittest.TestCase):
    """
    Contains unit tests for the AutoTSTAdapter class.
    """

    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None
        cls.project_dir = os.path.join(ARC_TESTING_PATH, 'test_AutoTST')

    def setUp(self):
        """
        A method that is run before each unit test in this class.
        """
        self.rxn_1 = ARCReaction(reactants=['CC[O]'], products=['[CH2]CO'],
                                 r_species=[ARCSpecies(label='CC[O]', smiles='CC[O]')],
                                 p_species=[ARCSpecies(label='[CH2]CO', smiles='[CH2]CO')])

    def _remove_test_dir(self, path: str):
        """A helper function to remove a single test's project directory (and the shared
        parent directory if it is empty). Tests may run in parallel (pytest-xdist), so
        each test must only ever remove its own subdirectory."""
        shutil.rmtree(path, ignore_errors=True)
        try:
            os.rmdir(self.project_dir)
        except OSError:
            pass

    def get_adapter(self, dir_name: str) -> AutoTSTAdapter:
        """A helper function to instantiate an AutoTSTAdapter instance."""
        project_directory = os.path.join(self.project_dir, dir_name)
        self.addCleanup(self._remove_test_dir, project_directory)
        return AutoTSTAdapter(job_type='tsg',
                              reactions=[self.rxn_1],
                              testing=True,
                              project='test',
                              project_directory=project_directory,
                              )

    def test_supported_families(self):
        """Test that the AutoTST adapter advertises the expected RMG families (gate 1)."""
        adapter = self.get_adapter(dir_name='tst_supported_families')
        for family in ['intra_H_migration', 'H_Abstraction', 'R_Addition_MultipleBond', 'Disproportionation']:
            self.assertIn(family, adapter.supported_families)

    def test_disproportionation_passes_both_gates(self):
        """Test that a Disproportionation reaction is admitted through BOTH TS-adapter gates for autotst.

        Gate 1: the adapter's own ``supported_families`` includes 'Disproportionation'.
        Gate 2: ``ts_adapters_by_rmg_family`` maps 'Disproportionation' to a list that includes 'autotst'.
        """
        # Gate 2: the RMG-family -> adapters registry (checked directly, no RMG classification needed).
        self.assertIn('Disproportionation', common.ts_adapters_by_rmg_family)
        self.assertIn('autotst', common.ts_adapters_by_rmg_family['Disproportionation'])

        # Gate 1: the adapter advertises Disproportionation as supported.
        adapter = self.get_adapter(dir_name='tst_disprop_gates')
        self.assertIn('Disproportionation', adapter.supported_families)

    def test_intra_h_migration_still_supported(self):
        """Test that enabling Disproportionation did not disturb the pre-existing intra_H_migration gate."""
        adapter = self.get_adapter(dir_name='tst_intra_h')
        self.assertIn('intra_H_migration', adapter.supported_families)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
