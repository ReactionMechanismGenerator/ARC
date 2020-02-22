#!/usr/bin/env python3
# encoding: utf-8

"""
A module for the abstract StatmechAdapter class
"""

from abc import ABC, abstractmethod


class StatmechAdapter(ABC):
    """
    An abstract class for statmech adapters.
    """

    @abstractmethod
    def compute_thermo(self, kinetics_flag: bool = True) -> None:
        """
        Generate thermodynamic data for a species.

        Args:
            kinetics_flag (bool, optional): Whether this call is used for generating species statmech
                                            for a rate coefficient calculation.
        """
        pass

    @abstractmethod
    def compute_high_p_rate_coefficient(self) -> None:
        """
        Generate a high pressure rate coefficient for a reaction.
        """
        pass
