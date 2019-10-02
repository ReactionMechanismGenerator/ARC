#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.job.trsh module
"""

import os
import unittest

import arc.job.trsh as trsh
from arc.settings import arc_path, supported_ess


class TestTrsh(unittest.TestCase):
    """
    Contains unit tests for the job.trsh module
    """

    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None
        path = os.path.join(arc_path, 'arc', 'testing', 'trsh')
        cls.base_path = {ess: os.path.join(path, ess) for ess in supported_ess}

    def test_determine_ess_status(self):
        """Test the determine_ess_status() function"""

        # Gaussian

        path = os.path.join(self.base_path['gaussian'], 'converged.out')
        status, keywords, error, line = trsh.determine_ess_status(
            output_path=path, species_label='OH', job_type='opt')
        self.assertEqual(status, 'done')
        self.assertEqual(keywords, list())
        self.assertEqual(error, '')
        self.assertEqual(line, '')

        path = os.path.join(self.base_path['gaussian'], 'l913.out')
        status, keywords, error, line = trsh.determine_ess_status(
            output_path=path, species_label='tst', job_type='composite')
        self.assertEqual(status, 'errored')
        self.assertEqual(keywords, ['MaxOptCycles', 'GL913'])
        self.assertEqual(error, 'Maximum optimization cycles reached.')
        self.assertIn('Error termination via Lnk1e', line)
        self.assertIn('g09/l913.exe', line)

        path = os.path.join(self.base_path['gaussian'], 'l301.out')
        status, keywords, error, line = trsh.determine_ess_status(
            output_path=path, species_label='Zr2O4H', job_type='opt')
        self.assertEqual(status, 'errored')
        self.assertEqual(keywords, ['GL301', 'InputError'])
        self.assertEqual(error, 'Either charge, multiplicity, or basis set was not specified correctly. '
                                'Alternatively, a specified atom does not match any standard atomic symbol.')
        self.assertIn('Error termination via Lnk1e', line)
        self.assertIn('g16/l301.exe', line)

        path = os.path.join(self.base_path['gaussian'], 'l9999.out')
        status, keywords, error, line = trsh.determine_ess_status(
            output_path=path, species_label='Zr2O4H', job_type='opt')
        self.assertEqual(status, 'errored')
        self.assertEqual(keywords, ['Unconverged', 'GL9999'])
        self.assertEqual(error, 'Unconverged')
        self.assertIn('Error termination via Lnk1e', line)
        self.assertIn('g16/l9999.exe', line)

        path = os.path.join(self.base_path['gaussian'], 'syntax.out')
        status, keywords, error, line = trsh.determine_ess_status(
            output_path=path, species_label='Zr2O4H', job_type='opt')
        self.assertEqual(status, 'errored')
        self.assertEqual(keywords, ['Syntax'])
        self.assertEqual(error, 'There was a syntax error in the Gaussian input file. Check your Gaussian input '
                                'file template under arc/job/inputs.py. Alternatively, perhaps the level of theory '
                                'is not supported by Gaussian in the format it was given.')
        self.assertFalse(line)

        # QChem

        path = os.path.join(self.base_path['qchem'], 'H2_opt.out')
        status, keywords, error, line = trsh.determine_ess_status(
            output_path=path, species_label='H2', job_type='opt')
        self.assertEqual(status, 'done')
        self.assertEqual(keywords, list())
        self.assertEqual(error, '')
        self.assertEqual(line, '')


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
