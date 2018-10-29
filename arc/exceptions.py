#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module contains classes which extend Exception for usage in the RMG module
"""

class InputError(Exception):
    """
    An exception raised when parsing an input file for any module.
    Pass a string describing the error.
    """
    pass

class OutputError(Exception):
    """
    This exception is raised whenever an error occurs while saving output
    information. Pass a string describing the circumstances of the exceptional
    behavior.
    """
    pass

class SettingsError(Exception):
    """
    An exception raised when dealing with settings.
    """
    pass

class SpeciesError(Exception):
    """
    An exception class for exceptional behavior that occurs while working with
    chemical species. Pass a string describing the circumstances that caused the
    exceptional behavior.
    """
    pass
