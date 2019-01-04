#!/usr/bin/env python
# encoding: utf-8

from __future__ import (absolute_import, division, print_function, unicode_literals)
import os
import logging

from arkane.input import species as arkane_species
from arkane.statmech import StatMechJob, assign_frequency_scale_factor
from arkane.thermo import ThermoJob

from rmgpy.data.rmg import RMGDatabase
from arc.settings import arc_path
from arc.job.inputs import input_files
from arc import plotter
from arc.exceptions import SchedulerError

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
                        rotor_symmetry = 1  # TODO
                        rotors += input_files['arkane_rotor'].format(rotor_path=rotor_path, pivots=pivots, top=top,
                                                                     symmetry=rotor_symmetry)
                        if i < species.number_of_rotors - 1:
                            rotors += ',\n          '
                if rotors:
                    rotors += ']'
                # write the Arkane species input file
                input_file_path = os.path.join(arc_path, 'Projects', self.project, species.label,
                                               '{0}_arkane_input.py'.format(species.label))
                output_dir = os.path.join(arc_path, 'Projects', self.project, 'output', species.label)
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

