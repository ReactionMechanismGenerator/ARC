#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests checking that ARC's Makefile only references devtools scripts that exist,
and what its compile targets build
"""

import os
import re
import shutil
import subprocess
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


def dry_run(target: str) -> str:
    """
    Print the commands a Makefile target would run, without running them.

    Args:
        target (str): The Makefile target.

    Returns:
        str: The output of ``make -n <target>`` in the repository root.
    """
    return subprocess.run(['make', '-n', '-C', ARC_ROOT, '--no-print-directory', target],
                          check=True, capture_output=True, text=True, timeout=60).stdout


@unittest.skipIf(shutil.which('make') is None, 'make is not on PATH')
class TestMakefileCompileTargets(unittest.TestCase):
    """
    Contains unit tests for the commands ARC's compile and kernels targets run.
    """

    def test_kernels_builds_only_the_c_kernels(self):
        """Test that make kernels compiles the C kernels and not the Cython extensions"""
        commands = dry_run('kernels')
        self.assertIn('-o arc/species/_zmat_c_kernels.so', commands)
        self.assertIn('utilities.py check-python', commands)
        self.assertNotIn('build_ext', commands)

    def test_compile_builds_cython_extensions_and_c_kernels(self):
        """Test that make compile builds the Cython extensions in place and the C kernels"""
        commands = dry_run('compile')
        self.assertIn('python setup.py build_ext main --inplace', commands)
        self.assertIn('-o arc/species/_zmat_c_kernels.so', commands)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
