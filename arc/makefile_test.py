#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests checking that ARC's Makefile only references devtools scripts that exist
"""

import os
import re
import unittest


ARC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAKEFILE_PATH = os.path.join(ARC_ROOT, 'Makefile')


def get_devtools_dir(text: str) -> str | None:
    """
    Get the value assigned to DEVTOOLS_DIR in a Makefile.

    Args:
        text (str): The Makefile content.

    Returns:
        str | None: The DEVTOOLS_DIR value, or ``None`` if the Makefile does not define it.
    """
    match = re.search(r'^DEVTOOLS_DIR\s*:?=\s*(\S+)\s*$', text, re.MULTILINE)
    return match.group(1) if match is not None else None


def get_makefile_devtools_scripts(text: str) -> set[str]:
    """
    Get the repo-relative paths of all devtools shell scripts a Makefile references,
    either through ``$(DEVTOOLS_DIR)`` or as a literal path.

    Args:
        text (str): The Makefile content.

    Returns:
        set[str]: Script paths relative to the repository root, e.g. ``'devtools/install_rmg.sh'``.
    """
    devtools_dir = get_devtools_dir(text)
    scripts = set()
    for prefix, name in re.findall(r'(\$\(DEVTOOLS_DIR\)|\bdevtools)/([\w.-]+\.sh)\b', text):
        directory = devtools_dir if prefix == '$(DEVTOOLS_DIR)' else prefix
        scripts.add(f'{directory}/{name}')
    return scripts


class TestMakefileDevtoolsScripts(unittest.TestCase):
    """
    Contains unit tests for the devtools scripts referenced in ARC's Makefile.
    """

    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        with open(MAKEFILE_PATH, 'r') as f:
            cls.text = f.read()

    def test_devtools_dir_is_defined(self):
        """Test that the Makefile defines DEVTOOLS_DIR, so script references can be resolved"""
        self.assertEqual(get_devtools_dir(self.text), 'devtools')

    def test_referenced_scripts_exist(self):
        """Test that every devtools script the Makefile references exists"""
        scripts = get_makefile_devtools_scripts(self.text)
        self.assertGreater(len(scripts), 0)
        self.assertIn('devtools/install_rmg.sh', scripts)
        missing = sorted(s for s in scripts if not os.path.isfile(os.path.join(ARC_ROOT, s)))
        self.assertEqual(missing, [], f'The Makefile references missing devtools scripts: {missing}')

    def test_get_makefile_devtools_scripts_synthetic(self):
        """Test resolving both $(DEVTOOLS_DIR) and literal devtools references in a synthetic Makefile"""
        text = 'DEVTOOLS_DIR := tools\n\na:\n\tbash $(DEVTOOLS_DIR)/a.sh --x\nb:\n\tbash devtools/b-1.sh\n'
        self.assertEqual(get_makefile_devtools_scripts(text), {'tools/a.sh', 'devtools/b-1.sh'})

    def test_get_makefile_devtools_scripts_extremes(self):
        """Test the parser on Makefiles with no devtools references or no DEVTOOLS_DIR definition"""
        self.assertEqual(get_makefile_devtools_scripts(''), set())
        self.assertEqual(get_makefile_devtools_scripts('all:\n\techo hi.sh\n'), set())
        self.assertIsNone(get_devtools_dir('all:\n\techo hi\n'))
        self.assertEqual(get_makefile_devtools_scripts('a:\n\tbash $(DEVTOOLS_DIR)/a.sh\n'), {'None/a.sh'})


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
