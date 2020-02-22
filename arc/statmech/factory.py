#!/usr/bin/env python3
# encoding: utf-8

"""
A module for generating statmech adapters.
"""

from typing import Type

from arc.reaction import ARCReaction
from arc.species.species import ARCSpecies
from arc.statmech.adapter import StatmechAdapter


_registered_statmech_adapters = {}


def register_statmech_adapter(statmech_adapter_label: str,
                              statmech_adapter_class: Type[StatmechAdapter],
                              ) -> None:
    """
    A register for statmech adapters.

    Args:
        statmech_adapter_label (StatmechEnum): A string representation for a statmech adapter.
        statmech_adapter_class (typing.Type[StatmechAdapter]): The statmech adapter class (a child of StatmechAdapter).

    Raises:
        TypeError: If statmech_class is not a StatmechAdapter instance.
    """
    if not issubclass(statmech_adapter_class, StatmechAdapter):
        raise TypeError(f'Statmech adapter class {statmech_adapter_class} is not a StatmechAdapter type.')
    _registered_statmech_adapters[statmech_adapter_label] = statmech_adapter_class


def statmech_factory(statmech_adapter_label: str,  # add everything that goes into the adapter class init
                     output_directory: str,
                     output_dict: dict,
                     use_bac: bool,
                     sp_level: str = '',
                     freq_scale_factor: float = 1.0,
                     species: Type[ARCSpecies] = None,
                     reaction: Type[ARCReaction] = None,
                     species_dict: dict = None,
                     T_min: tuple = None,
                     T_max: tuple = None,
                     T_count: int = 50,
                     ) -> Type[StatmechAdapter]:
    """
    A factory generating a statmech adapter corresponding to ``statmech_adapter``.

    Args:
        statmech_adapter_label (StatmechEnum): A string representation for a statmech adapter.
        output_directory (str): The path to the ARC project output directory.
        output_dict (dict): Keys are labels, values are output file paths.
                            See Scheduler for a description of this dictionary.
        use_bac (bool): Whether or not to use bond additivity corrections (BACs) for thermo calculations.
        sp_level (str, optional): The level of theory used for the single point energy calculation
                                  (could be a composite method), used for determining energy corrections.
        freq_scale_factor (float, optional): The harmonic frequencies scaling factor.
        species (ARCSpecies, optional): The species object.
        reaction (list, optional): The reaction object.
        species_dict (dict, optional): Keys are labels, values are ARCSpecies objects.
        T_min (tuple, optional): The minimum temperature for kinetics computations, e.g., (500, 'K').
        T_max (tuple, optional): The maximum temperature for kinetics computations, e.g., (3000, 'K').
        T_count (int, optional): The number of temperature points between t_min and t_max for kinetics computations.

    Returns:
        StatmechAdapter: The requested StatmechAdapter instance, initialized with the respective arguments,
    """
    statmech_adapter_class = _registered_statmech_adapters[statmech_adapter_label](output_directory=output_directory,
                                                                                   output_dict=output_dict,
                                                                                   use_bac=use_bac,
                                                                                   sp_level=sp_level,
                                                                                   freq_scale_factor=freq_scale_factor,
                                                                                   species=species,
                                                                                   reaction=reaction,
                                                                                   species_dict=species_dict,
                                                                                   T_min=T_min,
                                                                                   T_max=T_max,
                                                                                   T_count=T_count,
                                                                                   )
    return statmech_adapter_class
