#!/usr/bin/env python
# encoding: utf-8

from __future__ import (absolute_import, division, print_function, unicode_literals)
import os

from arc.rmgdb import determine_reaction_family
from arc.arc_exceptions import TSError, InputError


def h_abs_brute_force(rmgdb, rxn):
    """
    Generate TS guesses for H Abstraction reactions
    `rxn` is an ARCReaction object
    returns a list of xyz guesses
    """
    if rxn.family is None:
        rxn.family, _ = determine_reaction_family(rmgdb, rxn)
    if rxn.family != 'H_Abstraction':
        raise InputError('The brute force H Abstraction TS guess method accepts H Abstraction reactions only.'
                         ' Got: {0}'.format(rxn.family))







