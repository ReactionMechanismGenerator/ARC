"""
A factory for parsing ESS log files.
"""

from typing import Optional, Type

from arc.parser.adapter import ESSAdapter, ESSEnum

_registered_ess_adapters = {}  # keys are ESSEnum, values are ESSAdapter subclasses


def register_ess_adapter(ess_adapter_label: str,
                         ess_adapter_class: Type[ESSAdapter],
                         ) -> None:
    """
    A register for ess adapters.

    Args:
        ess_adapter_label (str): A string representation for an ESS adapter.
        ess_adapter_class (JobAdapter): The ESS adapter class (a child of ESSAdapter).

    Raises:
        TypeError: If ess_adapter_class is not a subclass of ESSAdapter.
    """
    if not issubclass(ess_adapter_class, ESSAdapter):
        raise TypeError(
            f'ESS adapter class {ess_adapter_class} is not a subclass ESSAdapter.')
    _registered_ess_adapters[ESSEnum(ess_adapter_label.lower())] = ess_adapter_class


def ess_factory(log_file_path: str,
                ess_adapter: str,
                ) -> ESSAdapter:
    """
    A factory generating an ESS adapter corresponding to ``ess_adapter``.

    Args:
        log_file_path (str): The path to the log file to be parsed.
        ess_adapter (str): The string representation of the ESS adapter, validated against ``ESSEnum``.

    Returns: ESSAdapter
        The requested ESSAdapter subclass, initialized with the ESS log file path.
    Raises:
        TypeError: If ess_adapter is not a string.
        ValueError: If ess_adapter is not registered in _registered_ess_adapters.
    """
    if not isinstance(ess_adapter, str):
        raise TypeError(f'Expected "ess_adapter" to be a string, got {type(ess_adapter)} instead.')

    ess_enum = ESSEnum(ess_adapter.lower())
    
    if ess_enum not in _registered_ess_adapters.keys():
        raise ValueError(
            f'The "ess_adapter" argument of {ess_adapter} was not present in the keys for the '
            f'_registered_ess_adapters dictionary: {list(_registered_ess_adapters.keys())}'
            f'\nPlease check that the job adapter was registered properly.'
        )

    return _registered_ess_adapters[ess_enum](log_file_path=log_file_path)
