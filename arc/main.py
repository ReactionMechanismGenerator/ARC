#!/usr/bin/env python
# encoding: utf-8

import logging
import sys
import os
import time

from rmgpy.species import Species
from rmgpy.reaction import Reaction

from arc.settings import arc_path
from arc.scheduler import Scheduler
from arc.exceptions import InputError
from arc.species import ARCSpecies

##################################################################


class ARC(object):
    """
    Main ARC object.
    The software should be run on a local computer, sending commands to one or more servers.
    """
    def __init__(self, project, species_list, rxn_list, level_of_theory, freq_level='', scan_level='',
                 verbose=logging.INFO):
        self.project = project
        self.species_list = species_list
        self.rxn_list = rxn_list
        self.ts_list = rxn_list  # TODO: derive TS from rxns
        self.level_of_theory = level_of_theory.lower()
        self.freq_level = freq_level.lower()
        self.scan_level = scan_level.lower()
        if not ('cbs' in self.level_of_theory or '//' in self.level_of_theory):
            raise InputError('Level of theory should either be a composite method (like CBS-QB3) or be of the'
                             'form sp//geometry, e.g., CCSD(T)-F12/avtz//wB97x-D3/6-311++g**')
        self.verbose = verbose
        self.output_directory = os.path.join(arc_path, 'Projects', self.project)
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
        self.initialize_log(self.verbose, os.path.join(self.output_directory, 'arc.log'))
        self.execute()

    def execute(self):
        # TODO: also allow xyz as input for both species and TSs
        logging.info('Starting project {0}\n\n'.format(self.project))
        for species in self.species_list:
            if not isinstance(species, ARCSpecies):
                raise ValueError('All species in species_list must be ARCSpecies objects.'
                                 ' Got {0}'.format(type(species)))
            logging.info('Considering species: {0}'.format(species.label))
        self.species_list = self.species_list
        for rxn in self.rxn_list:
            if not isinstance(rxn, Reaction):
                logging.error('`rxn_list` must be a list of RMG.Reaction objects. Got {0}'.format(type(rxn)))
                raise ValueError()
            logging.info('Considering reacrion {0}'.format(rxn))
        Scheduler(project=self.project, species_list=self.species_list,  level_of_theory=self.level_of_theory,
                  freq_level=self.freq_level, scan_level=self.scan_level)
        self.log_footer()

    def initialize_log(self, verbose=logging.INFO, log_file=None):
        """
        Set up a logger for ARC to use to print output to stdout.
        The `verbose` parameter is an integer specifying the amount of log text seen
        at the console; the levels correspond to those of the :data:`logging` module.
        """
        # Create logger
        logger = logging.getLogger()
        # logger.setLevel(verbose)
        logger.setLevel(logging.DEBUG)

        # Use custom level names for cleaner log output
        logging.addLevelName(logging.CRITICAL, 'Critical: ')
        logging.addLevelName(logging.ERROR, 'Error: ')
        logging.addLevelName(logging.WARNING, 'Warning: ')
        logging.addLevelName(logging.INFO, '')
        logging.addLevelName(logging.DEBUG, '')
        logging.addLevelName(0, '')

        # Create formatter and add to handlers
        formatter = logging.Formatter('%(levelname)s%(message)s')

        # Remove old handlers before adding ours
        while logger.handlers:
            logger.removeHandler(logger.handlers[0])

        # Create console handler; send everything to stdout rather than stderr
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(verbose)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # Create file handler; always be at least verbose in the file
        if log_file:
            fh = logging.FileHandler(filename=log_file)
            fh.setLevel(min(logging.DEBUG,verbose))
            fh.setFormatter(formatter)
            logger.addHandler(fh)
            self.log_header()

    def log_header(self, level=logging.INFO):
        """
        Output a header containing identifying information about CanTherm to the log.
        """
        logging.log(level, 'ARC execution initiated at {0}'.format(time.asctime()))
        logging.log(level, '')
        logging.log(level, '###############################################################')
        logging.log(level, '#                                                             #')
        logging.log(level, '#                            ARC                              #')
        logging.log(level, '#                                                             #')
        logging.log(level, '#   Version: 0.1                                              #')
        logging.log(level, '#                                                             #')
        logging.log(level, '###############################################################')
        logging.log(level, '')

    def log_footer(self, level=logging.INFO):
        """
        Output a footer to the log.
        """
        logging.log(level, '')
        logging.log(level, 'ARC execution terminated at {0}'.format(time.asctime()))

# TODO: MRCI, determine occ
# TODO: sucsessive opt (B3LYP, CCSD, MRCI)
# also of course possible to just provide a "known" TS xyz guess...
# need to know optical isomers and external symmetry (could also be read from QM, but not always right) for thermo

