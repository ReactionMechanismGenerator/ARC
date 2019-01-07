#!/usr/bin/env python
# encoding: utf-8

from __future__ import (absolute_import, division, print_function, unicode_literals)
import os
import logging

from arkane.input import species as arkane_species
from arkane.statmech import StatMechJob, assign_frequency_scale_factor, Log
from arkane.thermo import ThermoJob

from rmgpy import settings
from rmgpy.species import Species
from rmgpy.data.rmg import RMGDatabase
from rmgpy.exceptions import AtomTypeError

from arc.settings import arc_path
from arc.job.inputs import input_files
from arc import plotter
from arc.exceptions import SchedulerError, InputError

##################################################################


class Processor(object):
    """
    ARC Processor class. Post processes results in Arkane. The attributes are:

    ================ =========== ===============================================================================
    Attribute        Type        Description
    ================ =========== ===============================================================================
    `project`         ``str``    The project's name. Used for naming the directory.
    `species_dict`    ``dict``   Keys are labels, values are ARCSpecies objects
    `output`          ``dict``   Keys are labels, values are output file paths
    'use_bac'         ``bool``   Whether or not to use bond additivity corrections for thermo calculations
    'model_chemistry' ``list``   The model chemistry in Arkane for energy corrections (AE, BAC).
                                   This can be usually determined automatically.
    ================ =========== ===============================================================================
    """
    def __init__(self, project, species_dict, output, use_bac, model_chemistry):
        self.project = project
        self.species_dict = species_dict
        self.output = output
        self.use_bac = use_bac
        self.model_chemistry = model_chemistry
        self.database = None
        if any([species.generate_thermo for species in self.species_dict.values()]):
            self.load_rmg_database()

    def load_rmg_database(self, thermo_libraries=None, reaction_libraries=None, kinetics_families='default'):
        """
        A helper function for loading the RMG database
        """
        db_path = os.path.join(settings['database.directory'])
        thermo_libraries = thermo_libraries or []
        reaction_libraries = reaction_libraries or []
        if isinstance(thermo_libraries, str):
            thermo_libraries = [thermo_libraries]
        if isinstance(reaction_libraries, str):
            reaction_libraries = [reaction_libraries]
        if kinetics_families not in ('default', 'all', 'none'):
            if not isinstance(kinetics_families, list):
                raise InputError(
                    "kineticsFamilies should be either 'default', 'all', 'none', or a list of names eg."
                    " ['H_Abstraction','R_Recombination'] or ['!Intra_Disproportionation'].")

        if not thermo_libraries:
            thermo_libraries = []
            thermo_path = os.path.join(db_path, 'thermo', 'libraries')
            for thermo_library_path in os.listdir(thermo_path):
                thermo_library, _ = os.path.splitext(os.path.basename(thermo_library_path))
                thermo_libraries.append(thermo_library)
        # prioritize libraries
        thermo_priority = ['BurkeH2O2', 'thermo_DFT_CCSDTF12_BAC', 'DFT_QCI_thermo', 'Klippenstein_Glarborg2016',
                           'primaryThermoLibrary', 'primaryNS', 'NitrogenCurran', 'NOx2018', 'FFCM1(-)',
                           'SulfurLibrary', 'SulfurGlarborgH2S']
        indices_to_pop = []
        for i, lib in enumerate(thermo_libraries):
            if lib in thermo_priority:
                indices_to_pop.append(i)
        for i in reversed(range(len(thermo_libraries))):  # pop starting from the end, so other indices won't change
            if i in indices_to_pop:
                thermo_libraries.pop(i)
        thermo_libraries = thermo_priority + thermo_libraries

        # set library to be represented by a string rather than a unicode,
        # this might not be needed after a full migration to Py3
        old_thermo_libraries = thermo_libraries
        thermo_libraries = []
        for lib in old_thermo_libraries:
            thermo_libraries.append(str(lib))

        logging.info('\n\nLoading the RMG database...')
        self.database = RMGDatabase()
        self.database.load(
            path=db_path,
            thermoLibraries=thermo_libraries,
            transportLibraries='none',
            # reactionLibraries=reaction_libraries,
            seedMechanisms=[],
            kineticsFamilies='none',
            # kineticsDepositories=['training'],
            depository=False,
        )
        logging.info('\n\n')
        # for family in database.kinetics.families.values():  # load training
        #     family.addKineticsRulesFromTrainingSet(thermoDatabase=database.thermo)
        # for family in database.kinetics.families.values():
        #     family.fillKineticsRulesByAveragingUp(verbose=True)

    def process(self):
        species_list_for_plotting = []
        for species in self.species_dict.values():
            if self.output[species.label]['status'] == 'converged' and species.generate_thermo:
                linear = species.is_linear()
                if linear:
                    logging.info('Determined {0} to be a linear molecule'.format(species.label))
                species.determine_symmetry()
                multiplicity = species.multiplicity
                try:
                    sp_path = self.output[species.label]['composite']
                except KeyError:
                    try:
                        sp_path = self.output[species.label]['sp']
                    except KeyError:
                        raise SchedulerError('Could not find path to sp calculation for species {0}'.format(
                            species.label))
                if species.number_of_atoms == 1:
                    freq_path = sp_path
                    opt_path = sp_path
                else:
                    freq_path = self.output[species.label]['freq']
                    opt_path = self.output[species.label]['freq']
                rotors = ''
                for i in range(species.number_of_rotors):
                    if species.rotors_dict[i]['success']:
                        if not rotors:
                            rotors = '\n\nrotors = ['
                        if rotors[-1] == ',':
                            rotors += '\n'
                        rotor_path = species.rotors_dict[i]['scan_path']
                        pivots = str(species.rotors_dict[i]['pivots'])
                        top = str(species.rotors_dict[i]['top'])
                        rotor_symmetry = determine_rotor_symmetry(rotor_path, species.label, pivots)
                        rotors += input_files['arkane_rotor'].format(rotor_path=rotor_path, pivots=pivots, top=top,
                                                                     symmetry=rotor_symmetry)
                        if i < species.number_of_rotors - 1:
                            rotors += ',\n          '
                if rotors:
                    rotors += ']'
                # write the Arkane species input file
                if species.is_ts:
                    folder_name = 'TSs'
                else:
                    folder_name = 'Species'
                input_file_path = os.path.join(arc_path, 'Projects', self.project, folder_name, species.label,
                                               '{0}_arkane_input.py'.format(species.label))
                output_dir = os.path.join(arc_path, 'Projects', self.project, 'output', folder_name, species.label)
                if not os.path.isdir(output_dir):
                    os.makedirs(output_dir)
                output_file_path = os.path.join(output_dir, species.label + '_arkane_output.py')
                input_file = input_files['arkane_species']
                input_file= input_file.format(linear=linear, symmetry=species.external_symmetry,
                                              multiplicity=multiplicity, optical=species.optical_isomers,
                                              model_chemistry=self.model_chemistry, sp_path=sp_path, opt_path=opt_path,
                                              freq_path=freq_path, rotors=rotors)
                with open(input_file_path, 'wb') as f:
                    f.write(input_file)
                spec = arkane_species(species.label, input_file_path)
                stat_mech_job = StatMechJob(spec, input_file_path)
                stat_mech_job.applyBondEnergyCorrections = self.use_bac
                if self.use_bac:
                    logging.info('Using the following BAC for {0}: {1}'.format(species.label, species.bond_corrections))
                    stat_mech_job.bonds = species.bond_corrections
                stat_mech_job.modelChemistry = self.model_chemistry
                stat_mech_job.frequencyScaleFactor = assign_frequency_scale_factor(self.model_chemistry)
                stat_mech_job.execute(outputFile=output_file_path, plot=False)
                thermo_job = ThermoJob(spec, 'NASA')
                thermo_job.execute(outputFile=output_file_path, plot=False)
                species.thermo = spec.thermo  # copy the thermo from the Arkane species into the ARCSpecies
                plotter.log_thermo(species.label, path=output_file_path)

                species.rmg_species = Species(molecule=[species.mol])
                # species.rmg_species.label = str(species.label)
                try:
                    species.rmg_species.generate_resonance_structures(keep_isomorphic=False, filter_structures=True)
                except AtomTypeError:
                    pass
                species.rmg_thermo = self.database.thermo.getThermoData(species.rmg_species)
                species_list_for_plotting.append(species)
        if species_list_for_plotting:
            plotter.draw_thermo_parity_plot(species_list_for_plotting,
                                            path=os.path.join(arc_path, 'Projects', self.project, 'output'))


def determine_rotor_symmetry(rotor_path, label, pivots):
    """
    **  This is a temporary function, will soon be incorporated in Arkane instead**
    
    Determine the rotor symmetry number from the potential scan given in :list:`energies` in J/mol units
    Assumes the list represents a 360 degree scan
    str:`label` is the species name, used for logging and error messages
    list:`pivots` are the rotor's pivots, used for logging and error messages
    The *worst* resolution for each peak and valley is determined.
    The first criterion for a symmetric rotor is that the highest peak and the lowest peak must be within the
    worst peak resolution (and the same is checked for valleys).
    A second criterion for a symmetric rotor is that the highest and lowest peaks must be within 10% of
    the highest peak value. This is only applied if the highest peak is above 2 kJ/mol.
    """
    log = Log(path='')
    log.determine_qm_software(fullpath=rotor_path)
    energies, angle = log.software_log.loadScanEnergies()

    symmetry = None
    max_e = max(energies)
    if max_e > 2000:
        tol = 0.10 * max_e  # tolerance for the second criterion
    else:
        tol = max_e
    peaks, valleys = list(), [energies[0]]  # the peaks and valleys of the scan
    worst_peak_resolution, worst_valley_resolution = 0, max(energies[1] - energies[0], energies[-2] - energies[-1])
    for i, e in enumerate(energies):
        # identify peaks and valleys, and determine worst resolutions in the scan
        if i != 0 and i != len(energies) - 1:
            # this is an intermediate point in the scan
            if e > energies[i - 1] and e > energies[i + 1]:
                # this is a local peak
                if any([diff > worst_peak_resolution for diff in [e - energies[i - 1], e - energies[i + 1]]]):
                    worst_peak_resolution = max(e - energies[i - 1], e - energies[i + 1])
                peaks.append(e)
            elif e < energies[i - 1] and e < energies[i + 1]:
                # this is a local valley
                if any([diff > worst_valley_resolution for diff in [energies[i - 1] - e, energies[i + 1] - e]]):
                    worst_valley_resolution = max(energies[i - 1] - e, energies[i + 1] - e)
                valleys.append(e)
    # The number of peaks and valley must always be the same (what goes up must come down), if it isn't then there's
    # something seriously wrong with the scan
    if len(peaks) != len(valleys):
        raise InputError('Rotor of species {0} between pivots {1} does not have the same number'
                         ' of peaks and valleys.'.format(label, pivots))
    min_peak = min(peaks)
    max_peak = max(peaks)
    min_valley = min(valleys)
    max_valley = max(valleys)
    # Criterion 1: worst resolution
    if max_peak - min_peak > worst_peak_resolution:
        # The rotor cannot be symmetric
        symmetry = 1
        reason = 'worst peak resolution criterion'
    elif max_valley - min_valley > worst_valley_resolution:
        # The rotor cannot be symmetric
        symmetry = 1
        reason = 'worst valley resolution criterion'
    # Criterion 2: 10% * max_peak
    elif max_peak - min_peak > tol:
        # The rotor cannot be symmetric
        symmetry = 1
        reason = '10% of the maximum peak criterion'
    else:
        # We declare this rotor as symmetric and the symmetry number in the number of peaks (and valleys)
        symmetry = len(peaks)
        reason = 'number of peaks and valleys, all within the determined resolution criteria'
    if symmetry not in [1, 2, 3]:
        logging.info('Determined symmetry number {0} for rotor of species {1} between pivots {2};'
                     ' you should make sure this makes sense'.format(symmetry, label, pivots))
    else:
        logging.info('Determined a symmetry number of {0} for rotor of species {1} between pivots {2}'
                     ' based on the {3}.'.format(symmetry, label, pivots, reason))
    return symmetry
