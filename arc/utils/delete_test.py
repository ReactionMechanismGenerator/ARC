#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests for ARC's arc.utils.delete module
"""

import unittest
from unittest.mock import patch, MagicMock
from arc.utils.delete import parse_command_line_arguments, main
from arc.exceptions import InputError


class TestDelete(unittest.TestCase):
    """
    Contains unit tests for the delete utility.
    Mocks are used to isolate the main deletion logic from the actual system environment,
    preventing unintended file deletion or execution of command-line parsing during tests.
    - mock_parse: Simulates command-line arguments (sys.argv) parsing.
    - mock_local: Simulates the deletion of local ARC jobs.
    - mock_remote: Simulates the deletion of remote ARC jobs via SSH.
    - mock_isfile: Simulates the presence or absence of the initiated_jobs.csv database file.
    """

    def test_parse_command_line_arguments(self):
        """Test parsing command line arguments"""
        # Test project flag
        args = parse_command_line_arguments(['-p', 'test_project'])
        self.assertEqual(args.project, 'test_project')
        
        # Test job flag
        args = parse_command_line_arguments(['-j', 'a1234'])
        self.assertEqual(args.job, '1234')
        
        args = parse_command_line_arguments(['--job', '5678'])
        self.assertEqual(args.job, '5678')
        
        # Test all flag
        args = parse_command_line_arguments(['-a'])
        self.assertTrue(args.all)

    @patch('arc.utils.delete.delete_all_arc_jobs')
    @patch('arc.utils.delete.delete_all_local_arc_jobs')
    @patch('arc.utils.delete.parse_command_line_arguments')
    def test_main_no_args(self, mock_parse, mock_local, mock_remote):
        """Test main raises InputError if no arguments are provided"""
        mock_parse.return_value = MagicMock(all=False, project='', job='', server='')
        with self.assertRaises(InputError):
            main()

    @patch('arc.utils.delete.delete_all_arc_jobs')
    @patch('arc.utils.delete.delete_all_local_arc_jobs')
    @patch('arc.utils.delete.parse_command_line_arguments')
    @patch('arc.utils.delete.os.path.isfile')
    def test_main_all(self, mock_isfile, mock_parse, mock_local, mock_remote):
        """Test main with the --all flag"""
        mock_parse.return_value = MagicMock(all=True, project='', job='', server=['local'])
        mock_isfile.return_value = False
        main()
        mock_local.assert_called_with(jobs=None)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
