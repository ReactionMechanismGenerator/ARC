#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module contains classes which extend Exception for usage in the RMG module
"""


class ConformerError(Exception):
    """
    An exception raised when generating conformers
    """
    pass


class InputError(Exception):
    """
    An exception raised when parsing an input file for any module
    """
    pass


class JobError(Exception):
    """
    An exception class for exceptional behavior that occurs while working with jobs
    """
    pass


class OutputError(Exception):
    """
    This exception is raised whenever an error occurs while saving output information
    """
    pass


class ParserError(Exception):
    """
    This exception is raised whenever an error occurs while parsing files
    """
    pass


class ReactionError(Exception):
    """
    An exception class for exceptional behavior that occurs while working with reactions
    """
    pass


class RotorError(Exception):
    """
    An exception class for exceptional behavior that occurs while working with rotors
    """
    pass


class SanitizationError(Exception):
    """
    Exception class to handle errors during SMILES perception.
    """
    pass


class SchedulerError(Exception):
    """
    An exception class for exceptional behavior that occurs while working with the scheduler
    """
    pass


class ServerError(Exception):
    """
    An exception class for exceptional behavior that occurs while working with servers
    """
    pass


class SettingsError(Exception):
    """
    An exception raised when dealing with settings
    """
    pass


class SpeciesError(Exception):
    """
    An exception class for exceptional behavior that occurs while working with chemical species
    """
    pass


class TSError(Exception):
    """
    An exception class for exceptional behavior that occurs while working with transition states
    """
    pass
