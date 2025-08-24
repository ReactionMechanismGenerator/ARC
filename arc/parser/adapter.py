"""
A module for the abstract ESSAdapter class
"""

import os
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

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
    def parse_geometry(self) -> Optional[Dict[str, tuple]]:
        """
        Parse the xyz geometry from an ESS log file.

        Returns: Optional[Dict[str, tuple]]
            The cartesian geometry.
        """
        pass

    @abstractmethod
    def parse_frequencies(self) -> Optional['np.ndarray']:
        """
        Parse the frequencies from a freq job output file.

        Returns: Optional[np.ndarray]
            The parsed frequencies (in cm^-1).
        """
        pass

    @abstractmethod
    def parse_normal_mode_displacement(self) -> Tuple[Optional['np.ndarray'], Optional['np.ndarray']]:
        """
        Parse frequencies and normal mode displacement.

        Returns: Tuple[Optional['np.ndarray'], Optional['np.ndarray']]
            The frequencies (in cm^-1) and the normal mode displacements.
        """
        pass

    @abstractmethod
    def parse_t1(self) -> Optional[float]:
        """
        Parse the T1 parameter from a CC calculation.

        Returns: Optional[float]
            The T1 parameter.
        """
        pass

    @abstractmethod
    def parse_e_elect(self) -> Optional[float]:
        """
        Parse the electronic energy from an sp job output file.

        Returns: Optional[float]
            The electronic energy in kJ/mol.
        """
        pass

    @abstractmethod
    def parse_zpe_correction(self) -> Optional[float]:
        """
        Determine the calculated ZPE correction (E0 - e_elect) from a frequency output file.

        Returns: Optional[float]
            The calculated zero point energy in kJ/mol.
        """
        pass

    @abstractmethod
    def parse_1d_scan_energies(self) -> Tuple[Optional[List[float]], Optional[List[float]]]:
        """
        Parse the 1D torsion scan energies from an ESS log file.

        Returns: Tuple[Optional[List[float]], Optional[List[float]]]
            The electronic energy in kJ/mol and the dihedral scan angle in degrees.
        """
        pass

    @abstractmethod
    def parse_1d_scan_coords(self) -> Optional[List[Dict[str, tuple]]]:
        """
        Parse the 1D torsion scan coordinates from an ESS log file.

        Returns: List[Dict[str, tuple]]
            The Cartesian coordinates.
        """
        pass

    @abstractmethod
    def parse_irc_traj(self) -> Optional[List[Dict[str, tuple]]]:
        """
        Parse the IRC trajectory coordinates from an ESS log file.

        Returns: List[Dict[str, tuple]]
            The Cartesian coordinates.
        """
        pass

    @abstractmethod
    def parse_scan_conformers(self) -> Optional['pd.DataFrame']:
        """
        Parse all internal coordinates of scan conformers into a DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing internal coordinates for all conformers
        """
        pass

    @abstractmethod
    def parse_nd_scan_energies(self) -> Optional[Dict]:
        """
        Parse the ND torsion scan energies from an ESS log file.

        Returns: Optional[Dict]
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
    def parse_dipole_moment(self) -> Optional[float]:
        """
        Parse the dipole moment in Debye from an opt job output file.

        Returns: Optional[float]
            The dipole moment in Debye.
        """
        pass

    @abstractmethod
    def parse_polarizability(self) -> Optional[float]:
        """
        Parse the polarizability from a freq job output file, returns the value in Angstrom^3.

        Returns: Optional[float]
            The polarizability in Angstrom^3.
        """
        pass

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
