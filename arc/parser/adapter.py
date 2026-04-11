"""
A module for the abstract ESSAdapter class
"""

import os
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING

from arc.common import get_logger

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

logger = get_logger()

class ESSEnum(str, Enum):
    """
    The supported electronic structure software (ESS) adapters.
    The available adapters are a finite set.
    """
    cfour = 'cfour'
    gaussian = 'gaussian'
    molpro = 'molpro'
    orca = 'orca'
    psi4 = 'psi4'
    qchem = 'qchem'
    terachem = 'terachem'
    xtb = 'xtb'
    yaml = 'yaml'

class ESSAdapter(ABC):
    """
    An abstract class for ESS adapters.
    """
    def __init__(self, log_file_path: str):
        """
        Initialize the ESSAdapter with the path to the log file, and check if the log file exists.

        Args:
            log_file_path (str): The path to the log file to be parsed.
        """
        self.log_file_path = log_file_path
        self.check_logfile_exists()

    @abstractmethod
    def logfile_contains_errors(self) -> bool:
        """
        Check if the ESS log file contains any errors.

        Returns: bool
            True if the log file contains errors, False otherwise.
        """
        pass

    @abstractmethod
    def parse_geometry(self) -> dict[str, tuple] | None:
        """
        Parse the xyz geometry from an ESS log file.

        Returns: dict[str, tuple] | None
            The cartesian geometry.
        """
        pass

    @abstractmethod
    def parse_frequencies(self) -> 'np.ndarray' | None:
        """
        Parse the frequencies from a freq job output file.

        Returns: np.ndarray | None
            The parsed frequencies (in cm^-1).
        """
        pass

    @abstractmethod
    def parse_normal_mode_displacement(self) -> tuple['np.ndarray' | None, 'np.ndarray' | None]:
        """
        Parse frequencies and normal mode displacement.

        Returns: tuple['np.ndarray' | None, 'np.ndarray' | None]
            The frequencies (in cm^-1) and the normal mode displacements.
        """
        pass

    @abstractmethod
    def parse_t1(self) -> float | None:
        """
        Parse the T1 parameter from a CC calculation.

        Returns: float | None
            The T1 parameter.
        """
        pass

    @abstractmethod
    def parse_e_elect(self) -> float | None:
        """
        Parse the electronic energy from an sp job output file.

        Returns: float | None
            The electronic energy in kJ/mol.
        """
        pass

    @abstractmethod
    def parse_zpe_correction(self) -> float | None:
        """
        Determine the calculated ZPE correction (E0 - e_elect) from a frequency output file.

        Returns: float | None
            The calculated zero point energy in kJ/mol.
        """
        pass

    @abstractmethod
    def parse_1d_scan_energies(self) -> tuple[list[float] | None, list[float] | None]:
        """
        Parse the 1D torsion scan energies from an ESS log file.

        Returns: tuple[list[float] | None, list[float] | None]
            The electronic energy in kJ/mol and the dihedral scan angle in degrees.
        """
        pass

    @abstractmethod
    def parse_1d_scan_coords(self) -> list[dict[str, tuple]] | None:
        """
        Parse the 1D torsion scan coordinates from an ESS log file.

        Returns: list[dict[str, tuple]]
            The Cartesian coordinates.
        """
        pass

    @abstractmethod
    def parse_irc_traj(self) -> list[dict[str, tuple]] | None:
        """
        Parse the IRC trajectory coordinates from an ESS log file.

        Returns: list[dict[str, tuple]]
            The Cartesian coordinates.
        """
        pass

    @abstractmethod
    def parse_scan_conformers(self) -> 'pd.DataFrame' | None:
        """
        Parse all internal coordinates of scan conformers into a DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing internal coordinates for all conformers
        """
        pass

    @abstractmethod
    def parse_nd_scan_energies(self) -> Dict | None:
        """
        Parse the ND torsion scan energies from an ESS log file.

        Returns: Dict | None
            The "results" dictionary, which has the following structure::

                  results = {'directed_scan_type': <str, used for the fig name>,
                             'scans': <list, entries are lists of torsion indices>,
                             'directed_scan': <dict, keys are tuples of '{0:.2f}' formatted dihedrals,
                                               values are dictionaries with the following keys and values:
                                               {'energy': <float, energy in kJ/mol>,  * only this is used here
                                                'xyz': <dict>,
                                                'is_isomorphic': <bool>,
                                                'trsh': <list, job.ess_trsh_methods>}>
                             }
        """
        pass

    @abstractmethod
    def parse_dipole_moment(self) -> float | None:
        """
        Parse the dipole moment in Debye from an opt job output file.

        Returns: float | None
            The dipole moment in Debye.
        """
        pass

    @abstractmethod
    def parse_polarizability(self) -> float | None:
        """
        Parse the polarizability from a freq job output file, returns the value in Angstrom^3.

        Returns: float | None
            The polarizability in Angstrom^3.
        """
        pass

    def parse_opt_steps(self) -> int | None:
        """
        Parse the number of geometry optimization steps from an opt job output file.

        Returns: int | None
            The number of optimization cycles, or ``None`` if not an opt job or not parseable.
        """
        return None

    def parse_ess_version(self) -> str | None:
        """
        Parse the ESS software version string from the log file header.

        Returns: str | None
            A version string like ``'Gaussian 16, Revision C.01'`` or ``'ORCA 5.0.4'``, or ``None``.
        """
        return None

    def check_logfile_exists(self,
                             counter: int = 5,
                             sleep_time: int = 5,
                             ) -> None:
        """
        Check if the log file exists, and raise an error if it does not.

        Args:
            counter (int): The number of times to check for the log file.
            sleep_time (int): The time in seconds to wait between checks.
        """
        for _ in range(counter):
            if os.path.isfile(self.log_file_path):
                return None
            logger.debug(f'Log file {self.log_file_path} does not exist. '
                         f'Waiting {sleep_time} seconds before checking again.')
            time.sleep(sleep_time)
        raise FileNotFoundError(f'The log file {self.log_file_path} does not exist after {counter} attempts.')
