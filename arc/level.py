"""
A module for working with levels of theory.

Note:
    ARC cannot directly make use of the LevelOfTheory class defined in Arkane.modelchem
    since it standardizing the arguments, e.g.:
    'cbs-qb3' -> 'cbsqb3', 'wb97x-d' -> 'wb97xd', 'def2-tzvp' -> 'def2tzvp'.
    This name standardization is desired from a theoretical pin tof view,
    where `wb97xd` and `wb97x-d3` might be considered similar.
    However, in ARC we must preserve the exact level of theory as inputted,
    since it is later being forwarded to the ESS.
"""

from __future__ import annotations

import os
from typing import Iterable, List, Optional, Union

from arkane.modelchem import LevelOfTheory

from arc.common import get_ordered_intersection_of_two_lists, read_yaml_file
from arc.settings import arc_path, levels_ess, supported_ess


class Level(object):
    """
    Uniquely defines the settings used for a quantum calculation.

    Args:
        repr (str, dict, optional): A dictionary or a simple string representation of the level of theory,
                                    e.g. "wb97xd/def2-tzvp", or {'method': 'b3lyp', 'basis': '6-31g'}.
        method (str, optional): Quantum chemistry method.
        basis (str, optional): Basis set.
        auxiliary_basis (str, optional): Auxiliary basis set for correlated methods.
        cabs (str, optional): Complementary auxiliary basis set for F12 calculations.
        method_type (str, optional): The level of theory method type (DFT, wavefunction, force field, semi-empirical, or composite).
        software (str, optional): Quantum chemistry software.
        software_version (Union[int, float, str], optional): Quantum chemistry software version.
        solvation_method (str, optional): Solvation method.
        solvent (str, optional): Solvent.
        solvation_scheme_level (Level, optional): A Level class representing the level of theory to calculate a
                                                  solvation energy correction at.
        args (str, optional): Tuple of additional arguments provided to the software.
    """

    def __init__(self,
                 repr: Optional[Union[str, dict]] = None,
                 method: Optional[str] = None,
                 basis: Optional[str] = None,
                 auxiliary_basis: Optional[str] = None,
                 cabs: Optional[str] = None,
                 method_type: Optional[str] = None,
                 software: Optional[str] = None,
                 software_version: Optional[Union[int, float, str]] = None,
                 solvation_method: Optional[str] = None,
                 solvent: Optional[str] = None,
                 solvation_scheme_level: Optional[Level] = None,
                 args: Optional[Union[str, Iterable[str]]] = None,
                 ):
        self.repr = repr
        self.method = method
        if self.repr is not None and self.method is not None:
            raise ValueError(f'Either repr or method must be specified, not both.\n'
                             f'Got: "{self.repr}" and "{self.method}".')
        if self.repr is None and self.method is None:
            raise ValueError(f'Either repr or method must be specified, got neither.')

        self.basis = basis
        self.auxiliary_basis = auxiliary_basis
        self.cabs = cabs
        self.method_type = method_type
        self.software = software
        self.software_version = software_version
        self.solvation_method = solvation_method
        self.solvent = solvent
        if isinstance(solvation_scheme_level, (dict, str)):
            solvation_scheme_level = Level(repr=solvation_scheme_level)
        if solvation_scheme_level is not None \
                and (solvation_scheme_level.solvent is not None
                     or solvation_scheme_level.solvation_method is not None
                     or solvation_scheme_level.solvation_scheme_level is not None):
            raise ValueError(f'Cannot represent a solvation_scheme_level which itself has solvation attributes.')
        self.solvation_scheme_level = solvation_scheme_level
        if self.solvation_method is not None and self.solvent is None:
            raise ValueError(f'Cannot represent a level of theory with a solvation method ("{self.solvation_method}") '
                             f'that lacks a solvent.')

        if isinstance(args, (list, tuple)):
            for arg in args:
                if not isinstance(arg, str):
                    raise ValueError(f'All entries in the args argument must be strings.\n'
                                     f'Got {arg} which is a {type(arg)} in {args}.')
        elif args is not None and not isinstance(args, str):
            raise ValueError(f'The args argument must be either a string or an iterable.\n'
                             f'Got {args} which is a {type(args)}.')
        self.args = args

        if self.repr is not None:
            self.build()

        self.lower()

        if self.method_type is None:
            self.deduce_method_type()
        self.compatible_ess = list()
        if self.software is None:
            self.deduce_software()

    def __str__(self) -> str:
        """
        Return a humane-readable string representation of the object.

        Returns:
            str: The level of theory string representation.
        """
        str_ = self.method
        if self.basis is not None:
            str_ += f'/{self.basis}'
        if self.auxiliary_basis is not None:
            str_ += f', auxiliary_basis: {self.auxiliary_basis}'
        if self.cabs is not None:
            str_ += f', cabs: {self.cabs}'
        if self.solvation_method is not None:
            str_ += f', solvation_method: {self.solvation_method}'
            if self.solvent is not None:
                str_ += f', solvent: {self.solvent}'
            if self.solvation_scheme_level is not None:
                str_ += f", solvation_scheme_level: '{str(self.solvation_scheme_level)}'"
        if self.software is not None:
            str_ += f', software: {self.software}'
            if self.software_version is not None:
                str_ += f', software_version: {self.software_version}'
        if self.args is not None:
            str_ += f', args: {self.args}'
        if self.method_type is not None:
            str_ += f' ({self.method_type})'
        return str_

    def lower(self):
        """
        Set arguments to lowercase.
        """
        self.method = self.method.lower()
        if self.basis is not None:
            self.basis = self.basis.lower()
        if self.auxiliary_basis is not None:
            self.auxiliary_basis = self.auxiliary_basis.lower()
        if self.cabs is not None:
            self.cabs = self.cabs.lower()
        if self.method_type is not None:
            self.method_type = self.method_type.lower()
        if self.software is not None:
            self.software = self.software.lower()
        if isinstance(self.software_version, str):
            self.software_version = self.software_version.lower()
        if self.solvation_method is not None:
            self.solvation_method = self.solvation_method.lower()
        if self.solvent is not None:
            self.solvent = self.solvent.lower()
        self.args = (self.args.lower(),) if isinstance(self.args, str) \
            else tuple([arg.lower() for arg in self.args]) if self.args is not None else None

    def build(self):
        """
        Assign object attributes from a dictionary representation of the object or a simple string ("method/basis").
        """
        level_dict = {'method': '',
                      'basis': None,
                      'auxiliary_basis': None,
                      'cabs': None,
                      'method_type': None,
                      'software': None,
                      'software_version': None,
                      'solvation_method': None,
                      'solvent': None,
                      'solvation_scheme_level': None,
                      'args': None}
        allowed_keys = list(level_dict.keys())

        if isinstance(self.repr, str):
            if ' ' in self.repr:
                # illegal inputs like 'dlpno-ccsd(t)/def2-svp def2-svp/c' or 'b3 lyp'
                raise ValueError(f'{self.repr} has empty spaces. Please use a dictionary format '
                                 f'to clearly specify method, basis, auxiliary basis, and dispersion in this case. '
                                 f'See documentation for more details.')
            if self.repr.count('/') >= 2:
                # illegal inputs like 'dlpno-ccsd(t)/def2-svp/def2-svp/c'
                raise ValueError(f'{self.repr} has multiple slashes. Please use a dictionary format '
                                 f'to specify method, basis, auxiliary basis, and dispersion in this case. '
                                 f'See documentation for more details.')
            if '/' not in self.repr:
                # e.g., 'AM1', 'XTB', 'CBS-QB3'
                # Notice that this function is not designed to distinguish composite methods and
                # semi-empirical methods. If such differentiation is needed elsewhere in the codebase, please use
                # `determine_model_chemistry_type` in common.py
                level_dict['method'] = self.repr
            else:
                splits = self.repr.split('/')
                level_dict['method'] = splits[0]
                level_dict['basis'] = splits[1]

        elif isinstance(self.repr, dict):
            # also treats representations of LevelOfTheory.as_dict from a restart file
            if 'method' not in self.repr.keys():
                raise ValueError(f'The repr dictionary argument must at least have a "method" key, got:\n{self.repr}')
            for key, value in self.repr.items():
                if key in allowed_keys and value:
                    level_dict[key] = value
                elif key not in allowed_keys:
                    raise ValueError(f'Got an illegal key "{key}" in level of theory dictionary representation'
                                     f'\n{self.repr}')

        else:
            raise ValueError(f'The repr argument must be either a string or a dictionary. '
                             f'Got {self.repr} which is a {type(self.repr)}.')

        self.__init__(**level_dict)

    def to_arkane_level_of_theory(self) -> LevelOfTheory:
        """
        Convert to an Arkane LevelOfTheory instance.

        Returns:
            LevelOfTheory: An Arkae LevelOfTheory instance
        """
        kwargs = self.__dict__
        del kwargs['solvation_scheme_level']
        del kwargs['method_type']
        del kwargs['repr']
        del kwargs['compatible_ess']
        return LevelOfTheory(**kwargs)

    def deduce_method_type(self):
        """
        Determine the type of a model chemistry:
        DFT, wavefunction, force field, semi-empirical, or composite
        """
        wave_function_methods = ['hf', 'cc', 'ci', 'mp2', 'mp3', 'cp', 'cep', 'nevpt', 'dmrg', 'ri', 'cas', 'ic', 'mr',
                                 'bd', 'mbpt']
        semiempirical_methods = ['am', 'pm', 'zindo', 'mndo', 'xtb', 'nddo']
        force_field_methods = ['amber', 'mmff', 'dreiding', 'uff', 'qmdff', 'gfn', 'gaff', 'ghemical', 'charmm', 'ani']
        # all composite methods supported by Gaussian
        composite_methods = ['cbs-4m', 'cbs-qb3', 'rocbs-qb3', 'cbs-apno', 'w1u', 'w1ro', 'w1bd', 'g1', 'g2', 'g3',
                             'g4', 'g2mp2', 'g3mp2', 'g3b3', 'g3mp2b3', 'g4mp2']
        # Composite methods
        if self.method in composite_methods:
            self.method_type = 'composite'
        # Special cases
        elif self.method in ['m06hf', 'm06-hf']:
            self.method_type = 'dft'
        # General cases
        elif any(wf_method in self.method for wf_method in wave_function_methods):
            self.method_type = 'wavefunction'
        elif any(sm_method in self.method for sm_method in semiempirical_methods):
            self.method_type = 'semiempirical'
        elif any(ff_method in self.method for ff_method in force_field_methods):
            self.method_type = 'force_field'
        else:
            # assume DFT
            self.method_type = 'dft'

    def deduce_software(self,
                        job_type: Optional[str] = None):
        """
        Deduce the ESS to be used for a given level of theory.
        Populates the .software attribute.

        Args:
            job_type (str, optional): An ARC job type, assists in determining the software.
        """
        self.software = None  # reset the attribute, this method might be called for different job types

        # OneDMin
        if self.software is None and job_type == 'onedmin':
            if 'onedmin' not in supported_ess:
                raise ValueError(f'Could not find the OneDMin software to compute Lennard-Jones parameters.\n'
                                 f'levels_ess is:\n{levels_ess}')
            self.software = 'onedmin'

        # Gromacs
        if self.software is None and job_type == 'gromacs':
            if 'gromacs' not in supported_ess:
                raise ValueError(f'Could not find the Gromacs software to run the MD job {self.method}.\n'
                                 f'levels_ess is:\n{levels_ess}')
            self.software = 'gromacs'

        # QChem
        if self.software is None and job_type == 'orbitals':
            # currently we only have a script to print orbitals on QChem,
            # could/should be elaborated to additional ESS
            if 'qchem' not in supported_ess:
                raise ValueError(f'Could not find the QChem software to compute molecular orbitals.\n'
                                 f'levels_ess is:\n{levels_ess}')
            self.software = 'qchem'

        # Orca
        if self.software is None and 'dlpno' in self.method:
            if 'orca' not in supported_ess:
                raise ValueError(f'Could not find Orca to run a DLPNO job.\nlevels_ess is:\n{levels_ess}')
            self.software = 'orca'

        # Gaussian
        if self.software is None \
                and (self.method_type == 'composite' or job_type == 'ff_param_fit' or job_type == 'irc'):
            if 'gaussian' not in supported_ess:
                raise ValueError(f'Could not find Gaussian to run the {self.method}.\n'
                                 f'levels_ess is:\n{levels_ess}')
            self.software = 'gaussian'

        # User phrases from settings (levels_ess)
        if self.software is None:
            for ess, phrase_list in levels_ess.items():
                for phrase in phrase_list:
                    if self.software is None and \
                            (phrase in self.method or self.basis is not None and phrase in self.basis):
                        self.software = ess.lower()

        if self.software is None:
            preferred_ess_order = ['gaussian', 'qchem', 'orca', 'molpro', 'terachem']
            # compatible_software = determine_ess_based_on_method(self.method)
            # model_chem_type = determine_model_chemistry_type(self.method)

            if self.method_type in ['force_field', 'semiempirical']:
                preferred_ess_order = ['gaussian', 'qchem', 'orca', 'molpro', 'terachem']
            elif self.method_type in ['wavefunction']:
                preferred_ess_order = ['molpro', 'gaussian', 'orca', 'qchem']
            elif self.method_type in ['composite']:
                preferred_ess_order = ['gaussian']
            elif self.method_type in ['dft']:
                preferred_ess_order = ['gaussian', 'qchem', 'terachem', 'orca']

            self.determine_compatible_ess()
            relevant_software = get_ordered_intersection_of_two_lists(self.compatible_ess, supported_ess)
            self.software = get_ordered_intersection_of_two_lists(preferred_ess_order, relevant_software)[0] \
                if relevant_software else None

    def determine_compatible_ess(self):
        """
        Determine compatible ESS.
        """
        ess_methods = read_yaml_file(path=os.path.join(arc_path, 'data', 'ess_methods.yml'))
        ess_methods = {ess: [method.lower() for method in methods] for ess, methods in ess_methods.items()}
        if self.method in ess_methods['gaussian']:
            self.compatible_ess.append('gaussian')
        if self.method in ess_methods['orca']:
            self.compatible_ess.append('orca')
        if self.method in ess_methods['qchem']:
            self.compatible_ess.append('qchem')
        if self.method in ess_methods['terachem']:
            self.compatible_ess.append('terachem')
        if self.method in ess_methods['molpro']:
            self.compatible_ess.append('molpro')
