#!/usr/bin/env python
# encoding: utf-8

import logging
import sys
import os
import time

from rmgpy.species import Species
from rmgpy.reaction import Reaction

from arc.settings import arc_path, default_levels_of_theory
from arc.scheduler import Scheduler
from arc.exceptions import InputError
from arc.species import ARCSpecies
from arc.processor import Processor

##################################################################


class ARC(object):
    """
    Main ARC object.
    The software is currently configured to run on a local computer, sending jobs / commands to one or more servers.

    The attributes are:

    ====================== =================== =========================================================================
    Attribute              Type                Description
    ====================== =================== =========================================================================
    `project`              ``str``             The project's name. Used for naming the working directory.
    'rmg_species_list'     ''list''            A list RMG Species objects. Species must have a non-empty label attribute
                                                 and are assumed to be stab;e wells (not TSs)
    `arc_species_list`     ``list``            A list of ARCSpecies objects (each entry represent either a stable well
                                                 or a TS)
    'rxn_list'             ``list``            A list of RMG Reaction objects. Will (hopefully) be converted into TSs
    'conformer_level'      ``str``             Level of theory for conformer searches
    'composite_method'     ``str``             Composite method
    'opt_level'            ``str``             Level of theory for geometry optimization
    'freq_level'           ``str``             Level of theory for frequency calculations
    'sp_level'             ``str``             Level of theory for single point calculations
    'scan_level'           ``str``             Level of theory for rotor scans
    'output'               ``dict``            Output dictionary with status and final QM files for all species
    'fine'                 ``bool``            Whether or not to use a fine grid for opt jobs (spawns an additional job)
    ====================== =================== =========================================================================

    `level_of_theory` is a string representing either sp//geometry levels or a composite method, e.g. 'CBS-QB3',
                                                 'CCSD(T)-F12a/aug-cc-pVTZ//B3LYP/6-311++G(3df,3pd)'...
    """
    def __init__(self, project, rmg_species_list=list(), arc_species_list=list(), rxn_list=list(),
                 level_of_theory='', conformer_level='', composite_method='', opt_level='', freq_level='', sp_level='',
                 scan_level='', fine=True, generate_conformers=True, scan_rotors=True, verbose=logging.INFO):

        self.project = project
        self.output_directory = os.path.join(arc_path, 'Projects', self.project)
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
        self.fine = fine
        self.generate_conformers = generate_conformers
        self.scan_rotors = scan_rotors
        if not self.fine:
            logging.info('\n')
            logging.warning('Not using a fine grid for geometry optimization jobs')
            logging.info('\n')
        self.output = dict()
        self.verbose = verbose
        self.initialize_log(verbose=self.verbose, log_file=os.path.join(self.output_directory, 'arc.log'))

        logging.info('Starting project {0}\n\n'.format(self.project))

        if level_of_theory.count('//') > 1:
            raise InputError('Level of theory seems wrong. It should either be a composite method (like CBS-QB3)'
                             ' or be of the form sp//geometry, e.g., CCSD(T)-F12/avtz//wB97x-D3/6-311++g**.'
                             ' Got: {0}'.format(level_of_theory))

        if conformer_level:
            self.conformer_level = conformer_level.lower()
            logging.info('Using {0} for refined conformer searches (after filtering via force fields)'.format(
                conformer_level))
        else:
            # logging.info('Using B97-D3/def2-SVP for refined conformer searches (after filtering via force fields)')
            # self.conformer_level = 'b97-d3/def2-msvp'  # use B97-D3/def2-mSVP as default for conformer search
            logging.info('Using {0} for refined conformer searches (after filtering via force fields)'.format(
                default_levels_of_theory['conformer']))
            self.conformer_level = default_levels_of_theory['conformer']

        self.composite_method = composite_method

        if level_of_theory:
            if '/' not in level_of_theory:
                # assume this is a composite method
                self.composite_method = level_of_theory.lower()
                logging.info('Using composite method {0}'.format(level_of_theory))
            elif '//' in level_of_theory:
                self.opt_level = level_of_theory.lower().split('//')[1]
                self.freq_level = level_of_theory.lower().split('//')[1]
                self.sp_level = level_of_theory.lower().split('//')[0]
                logging.info('Using {0} for geometry optimizations'.format(level_of_theory.split('//')[1]))
                logging.info('Using {0} for frequency calculations'.format(level_of_theory.split('//')[1]))
                logging.info('Using {0} for single point calculations'.format(level_of_theory.split('//')[0]))
            elif '/' in level_of_theory and '//' not in level_of_theory:
                # assume this is not a composite method, and the user meant to run opt, freq and sp at this level.
                # running an sp after opt at the same level is meaningless, but doesn't matter much also
                # The '//' combination will later assist in differentiating between composite to non-composite methods
                self.opt_level = level_of_theory.lower()
                self.freq_level = level_of_theory.lower()
                self.sp_level = level_of_theory.lower()
                logging.info('Using {0} for geometry optimizations'.format(level_of_theory))
                logging.info('Using {0} for frequency calculations'.format(level_of_theory))
                logging.info('Using {0} for single point calculations'.format(level_of_theory))
        else:
            if opt_level and not self.composite_method:
                self.opt_level = opt_level.lower()
                logging.info('Using {0} for geometry optimizations'.format(opt_level))
            else:
                # self.opt_level = 'wb97x-d3/def2-tzvpd'
                # logging.info('Using wB97x-D3/def2-TZVPD for geometry optimizations')
                self.opt_level = default_levels_of_theory['opt']
                logging.info('Using {0} for geometry optimizations'.format(default_levels_of_theory['opt']))
            if freq_level:
                self.freq_level = freq_level.lower()
                logging.info('Using {0} for frequency calculations'.format(freq_level))
            else:
                # self.freq_level = 'wb97x-d3/def2-tzvpd'
                # logging.info('Using wB97x-D3/def2-TZVPD for frequency calculations')
                self.freq_level = default_levels_of_theory['freq']
                logging.info('Using {0} for frequency calculations'.format(default_levels_of_theory['freq']))
            if sp_level:
                self.sp_level = sp_level.lower()
                logging.info('Using {0} for single point calculations'.format(sp_level))
            else:
                logging.info('Using {0} for single point calculations'.format(default_levels_of_theory['sp']))
                self.sp_level = default_levels_of_theory['sp']
        self.composite_method = composite_method.lower()
        if self.composite_method:
            logging.info('Using composite method {0}'.format(composite_method))
        if scan_level:
            self.scan_level = scan_level.lower()
            logging.info('Using {0} for rotor scans'.format(scan_level))
        else:
            # self.scan_level = 'b3lyp/6-311+g(d,p)'
            self.scan_level = default_levels_of_theory['scan']
            # logging.info('Using B3LYP/6-311+G(d,p) for rotor scans')
            logging.info('Using {0} for rotor scans'.format(default_levels_of_theory['scan']))

        self.rmg_species_list = rmg_species_list
        self.arc_species_list = arc_species_list
        if self.rmg_species_list:
            for rmg_spc in self.rmg_species_list:
                if not isinstance(rmg_spc, Species):
                    raise InputError('All entries of rmg_species_list have to be RMG Species objects.'
                                     ' Got: {0}'.format(type(rmg_spc)))
                if not rmg_spc.label:
                    raise InputError('Missing label on RMG Species object {0}'.format(rmg_spc))
                arc_spc = ARCSpecies(is_ts=False, rmg_species=rmg_spc)  # assuming an RMG Species is not a TS
                self.arc_species_list.append(arc_spc)

        self.rxn_list = rxn_list

        self.scheduler = None

        self.execute()

    def execute(self):
        logging.info('\n\n')
        for species in self.arc_species_list:
            if not isinstance(species, ARCSpecies):
                raise ValueError('All species in species_list must be ARCSpecies objects.'
                                 ' Got {0}'.format(type(species)))
            logging.info('Considering species: {0}'.format(species.label))
        for rxn in self.rxn_list:
            if not isinstance(rxn, Reaction):
                logging.error('`rxn_list` must be a list of RMG.Reaction objects. Got {0}'.format(type(rxn)))
                raise ValueError()
            logging.info('Considering reacrion {0}'.format(rxn))
        self.scheduler = Scheduler(project=self.project, species_list=self.arc_species_list,
                                   composite_method=self.composite_method, conformer_level=self.conformer_level,
                                   opt_level=self.opt_level, freq_level=self.freq_level, sp_level=self.sp_level,
                                   scan_level=self.scan_level, fine=self.fine,
                                   generate_conformers=self.generate_conformers, scan_rotors=self.scan_rotors)
        Processor(self.project, self.scheduler.species_dict, self.scheduler.output)
        self.summary()
        self.log_footer()

    def summary(self):
        """
        Report status and data of all species / reactions
        """
        logging.info('\n\n\nAll jobs terminated. Project summary:\n')
        for label, output in self.scheduler.output.iteritems():
            if output['status'] == 'converged':
                logging.info('Species {0} converged successfully'.format(label))
            else:
                logging.info('Species {0} failed with message:\n  {1}'.format(label, output['status']))

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

        # Create file handler
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
# TODO: sucsessive opt (B3LYP, CCSD, CISD(T), MRCI)
# TODO: need to know optical isomers and external symmetry (could also be read from QM, but not always right) for thermo
# TODO: calc thermo and rates
# TODO: mongodb?  https://github.com/PACChem/QTC/blob/master/qtc/dbtools.py
# TODO: make visuallization files
# TODO: MRCI input file and auto-occ/closed/frozed...
# TODO: eventually log all levels of theory used for a species. Could be in YAML
# TODO: make it run on the server
# TODO: what if a species has an imaginary freq? wait for rotor results, it could improve via the dihedral correction. But if not?
# TODO: solve the problem w/ molpro running from ARC
# TODO: find where to chack status and call  job.troubleshoot_server()
# TODO: submit jobs in a job list (Colin)
# TODO: Py3 proof (__future__)


