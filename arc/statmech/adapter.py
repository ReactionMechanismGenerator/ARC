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
                       e0_only: bool = False,
                       skip_rotors: bool = False,
                       ) -> None:
        """
        Generate thermodynamic data for a species.

        Args:
            e0_only (bool, optional): Whether to only run statmech (w/o thermo) to compute E0.
            skip_rotors (bool, optional): Whether to skip internal rotor consideration. Default: ``False``.
        """
        pass

    @abstractmethod
    def compute_high_p_rate_coefficient(self,
                                        skip_rotors: bool = False,
                                        estimate_dh_rxn: bool = False,
                                        require_ts_convergence: bool = True,
                                        verbose: bool = True,
                                        ) -> None:
        """
        Generate a high pressure rate coefficient for a reaction.
        Populates the reaction.kinetics attribute.

        Args:
            skip_rotors (bool, optional): Whether to skip internal rotor consideration. Default: ``False``.
            estimate_dh_rxn (bool, optional): Whether to estimate DH reaction instead of computing it. Default: ``False``.
                                              Useful for checking that the reaction could in principle be computed even
                                              when thermodynamic properties of reactants and products were still not computed.
            require_ts_convergence (bool, optional): Whether to attempt computing a rate only for converged TS species.
            verbose (bool, optional): Whether to log messages. Default: ``True``.
        """
        pass
