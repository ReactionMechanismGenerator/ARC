"""
This module contains classes which extend Exception for usage in ARC.
"""


class ActionError(Exception):
    """
    An exception raised when generating conformers.
    """
    pass


class AtomTypeError(Exception):
    """
    An exception raised when generating conformers.
    """
    pass


class ConformerError(Exception):
    """
    An exception raised when generating conformers.
    """
    pass


class ConverterError(Exception):
    """
    An exception raised when converting molecular representations.
    """
    pass


class DependencyError(Exception):
    """
    An exception raised when converting molecular representations.
    """
    pass


class ElementError(Exception):
    """
    An exception raised when converting molecular representations.
    """
    pass


class ILPSolutionError(Exception):
    """
    An exception raised when parsing an input file for any module.
    """
    pass


class ImplicitBenzeneError(Exception):
    """
    An exception raised when parsing an input file for any module.
    """
    pass


class InchiException(Exception):
    """
    An exception raised when parsing an input file for any module.
    """
    pass


class InputError(Exception):
    """
    An exception raised when parsing an input file for any module.
    """
    pass


class InvalidAdjacencyListError(Exception):
    """
    An exception raised when parsing an input file for any module.
    """
    pass


class JobError(Exception):
    """
    An exception class for exceptional behavior that occurs while working with jobs.
    """
    pass


class KekulizationError(Exception):
    """
    An exception class for exceptional behavior that occurs while working with jobs.
    """
    pass


class ParserError(Exception):
    """
    This exception is raised whenever an error occurs while parsing files.
    """
    pass


class ReactionError(Exception):
    """
    An exception class for exceptional behavior that occurs while working with reactions.
    """
    pass


class ResonanceError(Exception):
    """
    An exception class for exceptional behavior that occurs while working with reactions.
    """
    pass


class RotorError(Exception):
    """
    An exception class for exceptional behavior that occurs while working with rotors.
    """
    pass


class SanitizationError(Exception):
    """
    Exception class to handle errors during SMILES perception.
    """
    pass


class SchedulerError(Exception):
    """
    An exception class for exceptional behavior that occurs while working with the scheduler.
    """
    pass


class ServerError(Exception):
    """
    An exception class for exceptional behavior that occurs while working with servers.
    """
    pass


class SettingsError(Exception):
    """
    An exception raised when dealing with settings.
    """
    pass


class SpeciesError(Exception):
    """
    An exception class for exceptional behavior that occurs while working with chemical species.
    """
    pass


class TrshError(Exception):
    """
    An exception class for exceptional behavior that occurs while troubleshooting ESS jobs.
    """
    pass


class TSError(Exception):
    """
    An exception class for exceptional behavior that occurs while working with transition states.
    """
    pass


class UnexpectedChargeError(Exception):
    """
    An exception class for exceptional behavior that occurs while working with transition states.
    """
    pass


class VectorsError(Exception):
    """
    An exception class for exceptional behavior that occurs while working with vectors.
    """
    pass


class VF2Error(Exception):
    """
    An exception class for exceptional behavior that occurs while working with vectors.
    """
    pass


class ZMatError(Exception):
    """
    An exception class for exceptional behavior that occurs while working with z matrices.
    """
    pass
