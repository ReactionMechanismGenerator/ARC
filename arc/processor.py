#!/usr/bin/env python
# encoding: utf-8

from __future__ import (absolute_import, division, print_function, unicode_literals)
import os
import logging

from arkane.input import species as arkane_species
from arkane.statmech import StatMechJob, assign_frequency_scale_factor
from arkane.thermo import ThermoJob

from rmgpy import settings
from rmgpy.species import Species
from rmgpy.data.rmg import RMGDatabase
from rmgpy.exceptions import AtomTypeError

from arc.job.inputs import input_files
from arc import plotter
from arc.exceptions import SchedulerError, InputError
from arc.species import determine_rotor_symmetry

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
    `lib_long_desc`   ``str``    A multiline description of levels of theory for tthe outputted RMG libraries
    ================ =========== ===============================================================================
    """
    def __init__(self, project, species_dict, output, use_bac, model_chemistry, lib_long_desc):
        self.project = project
        self.species_dict = species_dict
        self.output = output
        self.use_bac = use_bac
        self.model_chemistry = model_chemistry
        self.database = None
        self.lib_long_desc = lib_long_desc
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

    def process(self, project_directory):
        species_list_for_thermo = []
        for species in self.species_dict.values():
            if self.output[species.label]['status'] == 'converged' and species.generate_thermo and not species.is_ts:
                linear = species.is_linear()
                if linear:
                    logging.info('Determined {0} to be a linear molecule'.format(species.label))
                    species.long_thermo_description += 'Treated as a linear species\n'
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
                pivots_for_description = 'Pivots of considered rotors: '
                for i in range(species.number_of_rotors):
                    if species.rotors_dict[i]['success']:
                        if not rotors:
                            rotors = '\n\nrotors = ['
                        if rotors[-1] == ',':
                            rotors += '\n'
                        rotor_path = species.rotors_dict[i]['scan_path']
                        pivots = str(species.rotors_dict[i]['pivots'])
                        pivots_for_description += str(species.rotors_dict[i]['pivots']) + ' ,'
                        top = str(species.rotors_dict[i]['top'])
                        rotor_symmetry = determine_rotor_symmetry(rotor_path, species.label, pivots)
                        rotors += input_files['arkane_rotor'].format(rotor_path=rotor_path, pivots=pivots, top=top,
                                                                     symmetry=rotor_symmetry)
                        if i < species.number_of_rotors - 1:
                            rotors += ',\n          '
                if rotors:
                    rotors += ']'
                    species.long_thermo_description = pivots_for_description[:-2] + '\n'
                # write the Arkane species input file
                folder_name = 'TSs' if species.is_ts else 'Species'
                input_file_path = os.path.join(project_directory, 'calcs', folder_name, species.label,
                                               '{0}_arkane_input.py'.format(species.label))
                output_dir = os.path.join(project_directory, 'output', folder_name, species.label)
                if not os.path.isdir(output_dir):
                    os.makedirs(output_dir)
                output_file_path = os.path.join(output_dir, species.label + '_arkane_output.py')
                input_file = input_files['arkane_species']
                if self.use_bac:
                    logging.info('Using the following BAC for {0}: {1}'.format(species.label, species.bond_corrections))
                    bonds = '\n\nbonds = {0}'.format(species.bond_corrections)
                else:
                    logging.debug('NOT using BAC for {0}'.format(species.label))
                    bonds = ''
                input_file= input_file.format(linear=linear, bonds=bonds, symmetry=species.external_symmetry,
                                              multiplicity=multiplicity, optical=species.optical_isomers,
                                              model_chemistry=self.model_chemistry, sp_path=sp_path, opt_path=opt_path,
                                              freq_path=freq_path, rotors=rotors)
                with open(input_file_path, 'wb') as f:
                    f.write(input_file)
                arkane_spc = arkane_species(species.label, input_file_path)
                if species.mol_list:
                    arkane_spc.molecule = species.mol_list
                stat_mech_job = StatMechJob(arkane_spc, input_file_path)
                stat_mech_job.applyBondEnergyCorrections = self.use_bac
                stat_mech_job.modelChemistry = self.model_chemistry
                stat_mech_job.frequencyScaleFactor = assign_frequency_scale_factor(self.model_chemistry)
                stat_mech_job.execute(outputFile=output_file_path, plot=False)
                thermo_job = ThermoJob(arkane_spc, 'NASA')
                thermo_job.execute(outputFile=output_file_path, plot=False)
                species.thermo = arkane_spc.thermo  # copy the thermo from the Arkane species into the ARCSpecies
                plotter.log_thermo(species.label, path=output_file_path)

                species.rmg_species = Species(molecule=[species.mol])
                species.rmg_species.reactive = True
                if species.mol_list:
                    species.rmg_species.molecule = species.mol_list  # add resonance structures for thermo determination
                try:
                    species.rmg_thermo = self.database.thermo.getThermoData(species.rmg_species)
                except ValueError:
                    logging.info('Could not retrieve RMG thermo for species {0}, possibly due to missing 2D structure '
                                 '(bond orders). Not including this species in the parity plots.'.format(species.label))
                else:
                    species_list_for_thermo.append(species)
        if species_list_for_thermo:
            output_dir = os.path.join(project_directory, 'output')
            plotter.draw_thermo_parity_plots(species_list_for_thermo, path=output_dir)
            libraries_path = os.path.join(output_dir, 'RMG libraries')
            plotter.save_thermo_lib(species_list_for_thermo, path=libraries_path,
                                    name=self.project, lib_long_desc=self.lib_long_desc)
