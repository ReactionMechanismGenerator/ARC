#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains classes which extend Exception for usage in the RMG module.
"""


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


class InputError(Exception):
    """
    An exception raised when parsing an input file for any module.
    """
    pass


class JobError(Exception):
    """
    An exception class for exceptional behavior that occurs while working with jobs.
    """
    pass


class OutputError(Exception):
    """
    This exception is raised whenever an error occurs while saving output information.
    """
    pass


class ParserError(Exception):
    """
    This exception is raised whenever an error occurs while parsing files.
    """
    pass


class ProcessorError(Exception):
    """
    This exception is raised whenever an error occurs while processing thermo and kinetics.
    """
    pass


class QAError(Exception):
    """
    This exception is raised whenever an error occurs while checking a Job's quality.
    """
    pass


class ReactionError(Exception):
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


class VectorsError(Exception):
    """
    An exception class for exceptional behavior that occurs while working with vectors.
    """
    pass


class ZMatError(Exception):
    """
    An exception class for exceptional behavior that occurs while working with z matrices.
    """
    pass
