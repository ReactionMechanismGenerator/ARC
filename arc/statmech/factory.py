"""
A module for generating statmech adapters.
"""

from typing import TYPE_CHECKING, List, Optional, Type

from arc.statmech.adapter import StatmechAdapter

if TYPE_CHECKING:
    from arc.level import Level
    from arc.reaction import ARCReaction
    from arc.species.species import ARCSpecies


_registered_statmech_adapters = {}


def register_statmech_adapter(statmech_adapter_label: str,
                              statmech_adapter_class: Type[StatmechAdapter],
                              ) -> None:
    """
    A register for statmech adapters.

    Args:
        statmech_adapter_label (StatmechEnum): A string representation for a statmech adapter.
        statmech_adapter_class (Type[StatmechAdapter]): The statmech adapter class (a child of StatmechAdapter).

    Raises:
        TypeError: If statmech_class is not a subclass of StatmechAdapter.
    """
    if not issubclass(statmech_adapter_class, StatmechAdapter):
        raise TypeError(f'Statmech adapter class {statmech_adapter_class} is not a subclass of StatmechAdapter.')
    _registered_statmech_adapters[statmech_adapter_label] = statmech_adapter_class


def statmech_factory(statmech_adapter_label: str,  # add everything that goes into the adapter class init
                     output_directory: str,
                     calcs_directory: str,
                     output_dict: dict,
                     species: List['ARCSpecies'],
                     reactions: Optional[List['ARCReaction']] = None,
                     bac_type: Optional[str] = 'p',
                     sp_level: Optional['Level'] = None,
                     freq_level: Optional['Level'] = None,
                     freq_scale_factor: float = 1.0,
                     skip_nmd: bool = False,
                     species_dict: dict = None,
                     T_min: Optional[tuple] = None,
                     T_max: Optional[tuple] = None,
                     T_count: int = 50,
                     ) -> StatmechAdapter:
    """
    A factory generating a statmech adapter corresponding to ``statmech_adapter_label``.

    Args:
        statmech_adapter_label (StatmechEnum): A string representation for a statmech adapter.
        output_directory (str): The path to the ARC project output directory.
        calcs_directory (str): The path to the ARC project calculations directory.
        output_dict (dict): Keys are labels, values are output file paths.
                            See Scheduler for a description of this dictionary.
        bac_type (Optional[str], optional): The bond additivity correction type. 'p' for Petersson- or 'm' for Melius-type BAC.
                                            ``None`` to not use BAC.
        species (List[ARCSpecies]): A list of ARCSpecies objects to compute thermodynamic properties for.
        reactions (Optional[List[ARCReaction]]): A list of ARCReaction objects to compute kinetics for.
        sp_level (Level, optional): The level of theory used for energy corrections.
        freq_level (Level, optional): The level of theory used for frequency calculations.
        freq_scale_factor (float, optional): The harmonic frequencies scaling factor.
        skip_nmd (bool, optional): Whether to skip the normal mode displacement check analysis.
        species_dict (dict, optional): Keys are labels, values are ARCSpecies objects.
        T_min (tuple, optional): The minimum temperature for kinetics computations, e.g., (500, 'K').
        T_max (tuple, optional): The maximum temperature for kinetics computations, e.g., (3000, 'K').
        T_count (int, optional): The number of temperature points between t_min and t_max for kinetics computations.

    Returns: StatmechAdapter
        The requested StatmechAdapter subclass, initialized with the respective arguments.
    """
    statmech_adapter_class = \
        _registered_statmech_adapters[statmech_adapter_label](output_directory=output_directory,
                                                              output_dict=output_dict,
                                                              calcs_directory=calcs_directory,
                                                              species=species,
                                                              reactions=reactions,
                                                              bac_type=bac_type,
                                                              sp_level=sp_level,
                                                              freq_level=freq_level,
                                                              freq_scale_factor=freq_scale_factor,
                                                              skip_nmd=skip_nmd,
                                                              species_dict=species_dict,
                                                              T_min=T_min,
                                                              T_max=T_max,
                                                              T_count=T_count,
                                                              )
    return statmech_adapter_class
