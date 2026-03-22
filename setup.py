#!/usr/bin/env python3
# encoding: utf-8

import sys
import os
from collections import OrderedDict

try:
    from setuptools import setup, find_packages
    from setuptools.extension import Extension
except ImportError:
    print('The setuptools package is required to compile the molecule module in ARC.')
    raise

try:
    from Cython.Build import cythonize
    from Cython.Compiler import Options
except ImportError:
    print('Cython is required to compile the molecule module in ARC.')
    raise

try:
    import numpy
except ImportError:
    print('NumPy is required to compile the molecule module in ARC.')
    raise

Options.annotate = False  # HTML annotation

# Check if we are building for a coverage report
BUILD_FOR_COVERAGE = os.environ.get("ARC_COVERAGE") == "1"

directives = {
    'language_level': 3,
    'linetrace': BUILD_FOR_COVERAGE,
}

macros = [('CYTHON_TRACE', '1'), ('CYTHON_TRACE_NOGIL', '1')] if BUILD_FOR_COVERAGE else []

ext_modules = [
    Extension('arc.molecule.atomtype', ['arc/molecule/atomtype.py'], include_dirs=['.'], define_macros=macros),
    Extension('arc.constants', ['arc/constants.py'], include_dirs=['.'], define_macros=macros),
    Extension('arc.molecule.element', ['arc/molecule/element.py'], include_dirs=['.'], define_macros=macros),
    Extension('arc.molecule.graph', ['arc/molecule/graph.pyx'], include_dirs=['.'], define_macros=macros),
    Extension('arc.molecule.group', ['arc/molecule/group.py'], include_dirs=['.'], define_macros=macros),
    Extension('arc.molecule.molecule', ['arc/molecule/molecule.py'], include_dirs=['.'], define_macros=macros),
    Extension('arc.molecule.symmetry', ['arc/molecule/symmetry.py'], include_dirs=['.'], define_macros=macros),
    Extension('arc.molecule.vf2', ['arc/molecule/vf2.pyx'], include_dirs=['.'], define_macros=macros),
    Extension('arc.molecule.converter', ['arc/molecule/converter.py'], include_dirs=['.'], define_macros=macros),
    Extension('arc.molecule.translator', ['arc/molecule/translator.py'], include_dirs=['.'], define_macros=macros),
    Extension('arc.molecule.util', ['arc/molecule/util.py'], include_dirs=['.'], define_macros=macros),
    Extension('arc.molecule.inchi', ['arc/molecule/inchi.py'], include_dirs=['.'], define_macros=macros),
    Extension('arc.molecule.resonance', ['arc/molecule/resonance.py'], include_dirs=['.'], define_macros=macros),
    Extension('arc.molecule.pathfinder', ['arc/molecule/pathfinder.py'], include_dirs=['.'], define_macros=macros),
    Extension('arc.molecule.kekulize', ['arc/molecule/kekulize.pyx'], include_dirs=['.'], define_macros=macros),
]

if 'main' in sys.argv:
    # This is for `python setup.py build_ext main`
    sys.argv.remove('main')
    ext_modules.extend(ext_modules)

ext_modules = list(OrderedDict.fromkeys(ext_modules))


setup(
    name='ARC-Molecule',
    version='1.0.0',
    description='ARC Molecule Cython components',
    author='ARC Developers',
    packages=find_packages(include=['arc.molecule', 'arc.molecule.*']),
    ext_modules=cythonize(ext_modules, build_dir='build', compiler_directives=directives),
    include_dirs=['.', numpy.get_include()],
)
