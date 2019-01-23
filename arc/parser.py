#!/usr/bin/env python
# encoding: utf-8

from __future__ import (absolute_import, division, print_function, unicode_literals)
import logging
import numpy as np
import os

"""
Various ESS parsing tools
"""

##################################################################


def parse_frequencies(path, software):
    if not os.path.isfile(path):
        raise ValueError('Could not find file {0}'.format(path))
    freqs = np.array([], np.float64)
    if software.lower() == 'qchem':
        with open(path, 'rb') as f:
            for line in f:
                if ' Frequency:' in line:
                    items = line.split()
                    for i, item in enumerate(items):
                        if i:
                            freqs = np.append(freqs, [(float(item))])
    elif software.lower() == 'gaussian':
        with open(path, 'rb') as f:
            line = f.readline()
            while line != '':
                if 'Frequencies --' in line:
                    freqs = np.append(freqs, [float(frq) for frq in line.split()[2:]])
                line = f.readline()
    else:
        raise ValueError('parse_frequencies() can curtrently only parse QChem and gaussian files,'
                         ' got {0}'.format(software))
    logging.debug('Using parser.parse_frequencies. Determined frequencies are: {0}'.format(freqs))
    return freqs


def parse_t1(path):
    """
    Parse the T1 parameter from a Molpro coupled cluster calculation
    """
    if not os.path.isfile(path):
        raise ValueError('Could not find file {0}'.format(path))
    t1 = None
    with open(path, 'rb') as f:
        for line in f:
            if 'T1 diagnostic:' in line:
                t1 = float(line.split()[-1])
    return t1
