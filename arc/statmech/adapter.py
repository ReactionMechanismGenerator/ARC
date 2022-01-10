"""
A module for the abstract StatmechAdapter class
"""

from abc import ABC, abstractmethod


class StatmechAdapter(ABC):
    """
    An abstract class for statmech adapters.
    """

    @abstractmethod
    def compute_thermo(self,
                       kinetics_flag: bool = True,
                       e0_only: bool = False,
                       skip_rotors: bool = False,
                       ) -> None:
        """
        Generate thermodynamic data for a species.

        Args:
            kinetics_flag (bool, optional): Whether this call is used for generating species statmech
                                            for a rate coefficient calculation.
            e0_only (bool, optional): Whether to only run statmech (w/o thermo) to compute E0.
            skip_rotors (bool, optional): Whether to skip internal rotor consideration. Default: ``False``.
        """
        pass

    @abstractmethod
    def compute_high_p_rate_coefficient(self,
                                        skip_rotors: bool = False,
                                        verbose: bool = True,
                                        ) -> None:
        """
        Generate a high pressure rate coefficient for a reaction.
        Populates the reaction.kinetics attribute.

        Args:
            skip_rotors (bool, optional): Whether to skip internal rotor consideration. Default: ``False``.
            verbose (bool, optional): Whether to log messages. Default: ``True``.
        """
        pass
