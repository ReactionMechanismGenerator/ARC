"""
A factory for parsing ESS log files.
"""

from typing import Optional, Type

from arc.parser.adapter import ESSAdapter, ESSEnum
from arc.parser.parser import determine_ess

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
        raise TypeError(f'ESS adapter class {ess_adapter_class} is not a subclass ESSAdapter.')
    _registered_ess_adapters[ESSEnum(ess_adapter_label.lower())] = ess_adapter_class


def ess_factory(log_file_path: str,
                ess_adapter: Optional[str] = None,
                ) -> ESSAdapter:
    """
    A factory generating an ESS adapter corresponding to ``ess_adapter``.

    Args:
        log_file_path (str): The path to the log file to be parsed.
        ess_adapter (str): The string representation of the ESS adapter, validated against ``ESSEnum``.

    Returns: ESSAdapter
        The requested ESSAdapter subclass, initialized with the ESS log file path.
    """
    ess_adapter = ess_adapter or determine_ess(log_file_path=log_file_path)
    if ess_adapter not in _registered_ess_adapters.keys():
        raise ValueError(f'The "ess_adapter" argument of {ess_adapter} was not present in the keys for the '
                         f'_registered_ess_adapters dictionary: {list(_registered_ess_adapters.keys())}'
                         f'\nPlease check that the job adapter was registered properly.')

    ess_adapter = ESSEnum(ess_adapter.lower())

    ess_adapter_class = _registered_ess_adapters[ess_adapter](log_file_path=log_file_path)
    return ess_adapter_class
