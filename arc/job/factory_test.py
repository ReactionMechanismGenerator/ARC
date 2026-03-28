#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests for ARC's factories
"""

import os
import shutil
import unittest
from unittest.mock import patch

from arc.common import ARC_TESTING_PATH
from arc.exceptions import JobError
from arc.job.factory import job_factory, register_job_adapter
from arc.job.adapters.xtb_adapter import xTBAdapter
from arc.parser.factory import ess_factory, register_ess_adapter
from arc.parser.adapters.xtb import XTBParser
from arc.species import ARCSpecies


class TestFactories(unittest.TestCase):
    """
    Contains unit tests for job and parser factories.
    """

    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.project_dir = os.path.join(ARC_TESTING_PATH, 'factory_tests_delete')
        os.makedirs(cls.project_dir, exist_ok=True)

    def test_job_factory_unregistered(self):
        """Test that job_factory raises ValueError for unregistered adapters"""
        with self.assertRaises(ValueError):
            job_factory('non_existent', 'project', self.project_dir)

    def test_job_factory_missing_species_and_reactions(self):
        """Test that job_factory raises JobError if both species and reactions are missing"""
        with self.assertRaises(JobError):
            job_factory('xtb', 'project', self.project_dir)

    def test_job_factory_invalid_species(self):
        """Test that job_factory raises JobError for invalid species type"""
        with self.assertRaises(JobError):
            job_factory('xtb', 'project', self.project_dir, species=['not_a_species'])

    def test_register_job_adapter_invalid_class(self):
        """Test that register_job_adapter raises TypeError for invalid class"""
        with self.assertRaises(TypeError):
            register_job_adapter('gaussian', object)

    def test_ess_factory_unregistered(self):
        """Test that ess_factory raises ValueError for unregistered adapters"""
        with self.assertRaises(ValueError):
            ess_factory('path', 'non_existent')

    def test_ess_factory_invalid_type(self):
        """Test that ess_factory raises TypeError for non-string adapter name"""
        with self.assertRaises(TypeError):
            ess_factory('path', 123)

    def test_register_ess_adapter_invalid_class(self):
        """Test that register_ess_adapter raises TypeError for invalid class"""
        with self.assertRaises(TypeError):
            register_ess_adapter('gaussian', object)

    def test_job_factory_success(self):
        """Test successful instantiation via job_factory"""
        spc = ARCSpecies(label='H', smiles='[H]')
        job = job_factory('xtb', 'project', self.project_dir, species=[spc], job_type='opt')
        self.assertIsInstance(job, xTBAdapter)

    def test_ess_factory_success(self):
        """Test successful instantiation via ess_factory"""
        with patch('arc.parser.adapter.ESSAdapter.check_logfile_exists'):
            ess = ess_factory('path', 'xtb')
        self.assertIsInstance(ess, XTBParser)

    @classmethod
    def tearDownClass(cls):
        """
        A function that is run ONCE after all unit tests in this class.
        """
        shutil.rmtree(cls.project_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
