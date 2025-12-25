"""
A module for working with levels of theory.
"""

import os
from typing import Dict, Iterable, List, Optional, Union

from arc.common import ARC_PATH, get_logger, get_ordered_intersection_of_two_lists, read_yaml_file
from arc.imports import settings


logger = get_logger()

levels_ess, supported_ess = settings['levels_ess'], settings['supported_ess']


class Level(object):
    """
    Uniquely defines the settings used for a quantum calculation level of theory.
    Either ``repr`` or ``method`` must be specified.

    Args:
        repr (str, dict, Level optional): A dictionary or a simple string representation of the level of theory,
                                          e.g. "wb97xd/def2-tzvp", or {'method': 'b3lyp', 'basis': '6-31g'}.
                                          Not in ``LevelOfTheory``.
        method (str, optional): Quantum chemistry method.
        basis (str, optional): Basis set.
        auxiliary_basis (str, optional): Auxiliary basis set for correlated methods.
        dispersion (str, optional): The DFT dispersion info (if not already included in method).
        cabs (str, optional): Complementary auxiliary basis set for F12 calculations.
        method_type (str, optional): The level of theory method type (DFT, wavefunction, force field, semi-empirical,
                                     or composite). Not in ``LevelOfTheory``.
        software (str, optional): Quantum chemistry software.
        software_version (Union[int, float, str], optional): Quantum chemistry software version.
        solvation_method (str, optional): Solvation method.
        solvent (str, optional): The solvent. Values are strings of "known" solvents, see https://gaussian.com/scrf/.
        solvation_scheme_level (Level, optional): A Level class representing the level of theory to calculate a
                                                  solvation energy correction at. Not in ``LevelOfTheory``.
        args (Dict[Dict[str, str]], optional): Additional arguments provided to the software.
                                               Different than the ``args`` in ``LevelOfTheory``.
        compatible_ess (list, optional): Entries are names of compatible ESS. Not in ``LevelOfTheory``.
    """

    def __init__(self,
                 repr: Optional[Union[str, dict, 'Level']] = None,
                 method: Optional[str] = None,
                 basis: Optional[str] = None,
                 auxiliary_basis: Optional[str] = None,
                 dispersion: Optional[str] = None,
                 cabs: Optional[str] = None,
                 method_type: Optional[str] = None,
                 software: Optional[str] = None,
                 software_version: Optional[Union[int, float, str]] = None,
                 compatible_ess: Optional[List[str]] = None,
                 solvation_method: Optional[str] = None,
                 solvent: Optional[str] = None,
                 solvation_scheme_level: Optional['Level'] = None,
                 args: Optional[Union[Dict[str, str], Iterable, str]] = None,
                 ):
        self.repr = repr
        self.method = method
        if self.repr is not None and self.method is not None:
            raise ValueError(f'Either repr or method must be specified, not both.\n'
                             f'Got: "{self.repr}" and "{self.method}".')
        if self.repr is None and self.method is None:
            raise ValueError('Either repr or method must be specified, got neither.')

        self.basis = basis
        self.auxiliary_basis = auxiliary_basis
        self.dispersion = dispersion
        self.cabs = cabs
        self.method_type = method_type
        self.software = software
        self.software_version = software_version
        self.compatible_ess = compatible_ess
        self.solvation_method = solvation_method
        self.solvent = solvent
        if isinstance(solvation_scheme_level, (dict, str)):
            solvation_scheme_level = Level(repr=solvation_scheme_level)
        if solvation_scheme_level is not None \
                and (solvation_scheme_level.solvent is not None
                     or solvation_scheme_level.solvation_method is not None
                     or solvation_scheme_level.solvation_scheme_level is not None):
            raise ValueError('Cannot represent a solvation_scheme_level which itself has solvation attributes.')
        self.solvation_scheme_level = solvation_scheme_level
        if (self.solvation_method is None) != (self.solvent is None):
            raise ValueError(
                'Both solvation method and solvent must be defined together, or both must be None. '
                f'Got solvation method = "{self.solvation_method}", solvent = "{self.solvent}".'
            )
        self.args = args or {'keyword': dict(), 'block': dict()}

        if self.repr is not None:
            self.build()

        self.lower()

        if self.method_type is None:
            self.deduce_method_type()
        if self.dispersion is not None and self.method_type not in ['dft', 'composite']:
            raise ValueError(f'Dispersion is only allowed for DFT (or composite) methods, got {self.dispersion} '
                             f'for {self.method} which is a {self.method_type}')
        if self.software is None:
            # it wasn't set by the user, try determining it
            self.deduce_software()

    def __eq__(self, other: 'Level') -> bool:
        """
        Determine equality between Level object instances.
        """
        if isinstance(other, Level):
            return str(self) == str(other)
        return False

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
        if self.dispersion is not None:
            str_ += f', dispersion: {self.dispersion}'
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
        if self.args is not None and self.args and all([val for val in self.args.values()]):
            if any([key == 'keyword' for key in self.args.keys()]):
                str_ += ', keyword args:'
                for key, arg in self.args.items():
                    if key == 'keyword':
                        str_ += f' {arg}'
        return str_

    def copy(self):
        """
        A method to create a copy of the object.

        Returns:
            Level: A copy of the object.
        """
        return Level(repr=self.as_dict())

    def simple(self) -> str:
        """
        Return a simple humane-readable string representation of the object.

        Returns:
            str: The simple level of theory string representation.
        """
        str_ = self.method
        if self.basis is not None:
            str_ += f'/{self.basis}'
        return str_

    def as_dict(self) -> dict:
        """
        Returns a minimal dictionary representation from which the object can be reconstructed.
        Useful for ARC restart files.
        """
        original_dict = self.__dict__
        clean_dict = {}
        for key, val in original_dict.items():
            if val is not None and key != 'args' or key == 'args' and all([v for v in self.args.values()]):
                clean_dict[key] = val
        return clean_dict

    def build(self):
        """
        Assign object attributes from a dictionary representation of the object or a simple string ("method/basis").
        Useful for ARC restart files.
        """
        level_dict = {'method': '',
                      'basis': None,
                      'auxiliary_basis': None,
                      'dispersion': None,
                      'cabs': None,
                      'method_type': None,
                      'software': None,
                      'software_version': None,
                      'compatible_ess': None,
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
                # Note that this function is not designed to distinguish between composite and semi-empirical methods.
                level_dict['method'] = self.repr
            else:
                splits = self.repr.split('/')
                level_dict['method'] = splits[0]
                if len (splits) > 1:
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

        elif isinstance(self.repr, Level):
            level_dict = self.repr.as_dict()

        else:
            raise ValueError(f'The repr argument must be either a string, a dictionary or a Level type.\n'
                             f'Got {self.repr} which is a {type(self.repr)}.')

        self.repr = None  # reset
        self.__init__(**level_dict)

    def lower(self):
        """
        Set arguments to lowercase.
        """
        self.method = self.method.lower()
        if self.basis is not None:
            self.basis = self.basis.lower()
        if self.auxiliary_basis is not None:
            self.auxiliary_basis = self.auxiliary_basis.lower()
        if self.dispersion is not None:
            self.dispersion = self.dispersion.lower()
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

        args = {'keyword': dict(), 'block': dict()}

        # 1st level dict, set self.args in place
        if isinstance(self.args, (list, tuple)):
            for arg in self.args:
                if not isinstance(arg, str):
                    raise ValueError(f'All entries in the args argument must be strings.\n'
                                     f'Got {arg} which is a {type(arg)} in {self.args}.')
            self.args = ' '.join([arg.lower() for arg in self.args])
        if isinstance(self.args, str):
            self.args = {'keyword': {'general': args.lower()}, 'block': dict()}
        elif self.args is not None and not isinstance(args, dict):
            raise ValueError(f'The args argument must be either a string, an iterable or a dictionary.\n'
                             f'Got {self.args} which is a {type(self.args)}.')

        # 2nd level dict, set in args, then transfer to self.args
        for key1, val1 in self.args.items():
            args[key1.lower()] = dict()
            if isinstance(val1, dict):
                for key2, val2 in val1.items():
                    new_val2 = str(val2) if isinstance(val2, (int, float)) else val2
                    if not isinstance(new_val2, str):
                        raise ValueError(f'All entries in the args argument must be str, int, or float types.\n'
                                         f'Got {new_val2} which is a {type(new_val2)} in {self.args}.')
                    args[key1.lower()][key2.lower()] = new_val2.lower()
            elif isinstance(val1, str):
                args[key1.lower()]['general'] = val1.lower()
            elif isinstance(val1, (list, tuple)):
                for v1 in val1:
                    if not isinstance(v1, str):
                        raise ValueError(f'All entries in the args argument must be strings.\n'
                                         f'Got {v1} which is a {type(v1)} in {self.args}.')
                args['keyword']['general'] = ' '.join([v1.lower() for v1 in val1])
            else:
                raise ValueError(f'Values of the args dictionary must be either dictionaries, strings, or lists, '
                                 f'got {val1} which is a {type(val1)}.')

        self.args = args

    def deduce_method_type(self):
        """
        Determine the model chemistry type:
        DFT, wavefunction, force field, semi-empirical, or composite
        """
        wave_function_methods = ['hf', 'cc', 'ci', 'mp2', 'mp3', 'cp', 'cep', 'nevpt', 'dmrg', 'ri', 'cas', 'ic', 'mr',
                                 'bd', 'mbpt']
        semiempirical_methods = ['am', 'pm', 'zindo', 'mndo', 'xtb', 'nddo']
        force_field_methods = ['amber', 'mmff', 'dreiding', 'uff', 'qmdff', 'gfn', 'gaff', 'ghemical', 'charmm', 'ani']
        # all composite methods supported by Gaussian
        composite_methods = ['cbs-4m', 'cbs-qb3', 'cbs-qb3-paraskevas', 'rocbs-qb3', 'cbs-apno', 'w1u', 'w1ro', 'w1bd',
                             'g1', 'g2', 'g3', 'g4', 'g2mp2', 'g3mp2', 'g3b3', 'g3mp2b3', 'g4mp2']
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

        # OneDMin
        if job_type == 'onedmin':
            if 'onedmin' not in supported_ess:
                raise ValueError(f'Could not find the OneDMin software to compute Lennard-Jones parameters.\n'
                                 f'levels_ess is:\n{levels_ess}')
            self.software = 'onedmin'

        # QChem
        if job_type == 'orbitals':
            # currently we only have a script to print orbitals on QChem,
            # could/should be elaborated to additional ESS
            if 'qchem' not in supported_ess:
                raise ValueError(f'Could not find the QChem software to compute molecular orbitals.\n'
                                 f'levels_ess is:\n{levels_ess}')
            self.software = 'qchem'

        # Orca
        if 'dlpno' in self.method:
            if 'orca' not in supported_ess:
                raise ValueError(f'Could not find Orca to run a DLPNO job.\nlevels_ess is:\n{levels_ess}')
            self.software = 'orca'

        # Gaussian
        if self.method_type == 'composite' or job_type == 'composite' or job_type == 'irc' \
                or any([sum(['iop' in value.lower() for value in subdict.values()]) for subdict in self.args.values()]):
            if 'gaussian' not in supported_ess:
                raise ValueError(f'Could not find Gaussian to run the {self.method}.\n'
                                 f'levels_ess is:\n{levels_ess}')
            self.software = 'gaussian'

        # TorchANI
        if 'torchani' in self.method:
            self.software = 'torchani'

        # xTB
        if 'xtb' in self.method or 'gfn' in self.method:
            self.software = 'xtb'

        # User phrases from settings (levels_ess)
        if self.software is None:
            for ess, phrase_list in levels_ess.items():
                for phrase in phrase_list:
                    if self.software is None and \
                            (phrase in self.method or self.basis is not None and phrase in self.basis):
                        self.software = ess.lower()

        if self.software is None:
            preferred_ess_order = ['gaussian', 'qchem', 'orca', 'molpro', 'terachem', 'cfour']

            if self.method_type in ['force_field', 'semiempirical']:
                preferred_ess_order = ['gaussian', 'qchem', 'orca', 'molpro', 'terachem']
            elif self.method_type in ['wavefunction']:
                preferred_ess_order = ['molpro', 'gaussian', 'orca', 'cfour', 'qchem']
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
        if self.compatible_ess is None:
            # Don't append if the user already specified a restricted list.
            self.compatible_ess = list()
            ess_methods = read_yaml_file(path=os.path.join(ARC_PATH, 'data', 'ess_methods.yml'))
            ess_methods = {ess: [method.lower() for method in methods] for ess, methods in ess_methods.items()}
            for ess in supported_ess:
                if ess in ess_methods and self.method in ess_methods[ess]:
                    self.compatible_ess.append(ess)


def assign_frequency_scale_factor(level: Union[str, Level]) -> Optional[int]:
    """
    Assign a frequency scaling factor to a level of theory.

    Args:
        level (Union[str, Level]): The level of theory.

    Returns:
        Optional[int]: The frequency scaling factor.
    """
    freq_scale_factors = read_yaml_file(os.path.join(ARC_PATH, 'data', 'freq_scale_factors.yml'))['freq_scale_factors']
    if isinstance(level, str):
        if level in freq_scale_factors:
            return freq_scale_factors[level]
        level = Level(repr=level)
    level_str = str(level)
    if level_str in freq_scale_factors:
        return freq_scale_factors[level_str]
    return None
