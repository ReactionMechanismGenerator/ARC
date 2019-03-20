#!/usr/bin/env python
# encoding: utf-8

from __future__ import (absolute_import, division, print_function, unicode_literals)
import os
import shutil
import logging
from random import randint

from arkane.input import species as arkane_species, transitionState as arkane_transition_state,\
    reaction as arkane_reaction
from arkane.statmech import StatMechJob, assign_frequency_scale_factor
from arkane.thermo import ThermoJob
from arkane.kinetics import KineticsJob

from rmgpy.species import Species

import arc.rmgdb as rmgdb
from arc.job.inputs import input_files
from arc import plotter
from arc.arc_exceptions import SchedulerError, RotorError
from arc.species.species import determine_rotor_symmetry, determine_rotor_type

##################################################################


class Processor(object):
    """
    ARC Processor class. Post processes results in Arkane. The attributes are:

    ================ =========== ===============================================================================
    Attribute        Type        Description
    ================ =========== ===============================================================================
    `project`         ``str``    The project's name. Used for naming the directory.
    `species_dict`    ``dict``   Keys are labels, values are ARCSpecies objects
    `rxn_list`        ``list``   List of ARCReaction objects
    `output`          ``dict``   Keys are labels, values are output file paths
    `use_bac`         ``bool``   Whether or not to use bond additivity corrections for thermo calculations
    `model_chemistry` ``list``   The model chemistry in Arkane for energy corrections (AE, BAC).
                                   This can be usually determined automatically.
    `lib_long_desc`   ``str``    A multiline description of levels of theory for the outputted RMG libraries
    `project_directory` ``str``  The path of the ARC project directory
    `t_min`           ``tuple``  The minimum temperature for kinetics computations, e.g., (500, str('K'))
    `t_max`           ``tuple``  The maximum temperature for kinetics computations, e.g., (3000, str('K'))
    `t_count`         ``int``    The number of temperature points between t_min and t_max for kinetics computations
    ================ =========== ===============================================================================
    """
    def __init__(self, project, project_directory, species_dict, rxn_list, output, use_bac, model_chemistry,
                 lib_long_desc, rmgdatabase, t_min=None, t_max=None, t_count=None):
        self.rmgdb = rmgdatabase
        self.project = project
        self.project_directory = project_directory
        self.species_dict = species_dict
        self.rxn_list = rxn_list
        self.output = output
        self.use_bac = use_bac
        self.model_chemistry = model_chemistry
        self.lib_long_desc = lib_long_desc
        load_thermo_libs, load_kinetic_libs = True, True
        if not any([species.is_ts and species.final_xyz for species in self.species_dict.values()]):
            load_kinetic_libs = False  # don't load reaction libraries, not TS has converged
        if not any([species.generate_thermo for species in self.species_dict.values()]):
            load_thermo_libs = False  # don't load thermo libraries, not thermo requested
        rmgdb.load_rmg_database(rmgdb=self.rmgdb, load_thermo_libs=load_thermo_libs,
                                load_kinetic_libs=load_kinetic_libs)
        t_min = t_min if t_min is not None else (300, 'K')
        t_max = t_max if t_max is not None else (3000, 'K')
        if isinstance(t_min, (int, float)):
            t_min = (t_min, 'K')
        if isinstance(t_max, (int, float)):
            t_max = (t_max, 'K')
        self.t_min = (t_min[0], str(t_min[1]))
        self.t_max = (t_max[0], str(t_max[1]))
        self.t_count = t_count if t_count is not None else 50

    def _generate_arkane_species_file(self, species):
        """
        A helper function for generating the Arkane species file
        Assigns the input file path to species.arkane_file and returns a desired path for the output file
        """
        folder_name = 'rxns' if species.is_ts else 'Species'
        output_dir = os.path.join(self.project_directory, 'output', folder_name, species.label)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        output_file_path = os.path.join(output_dir, species.label + '_arkane_output.py')
        if os.path.isfile(os.path.join(output_dir, 'species_dictionary.txt')):
            os.remove(os.path.join(output_dir, 'species_dictionary.txt'))

        if species.yml_path is not None:
            species.arkane_file = species.yml_path
            return output_file_path

        if not species.is_ts:
            linear = species.is_linear()
            if linear:
                logging.info('Determined {0} to be a linear molecule'.format(species.label))
                species.long_thermo_description += 'Treated as a linear species\n'
        else:
            linear = False
        species.determine_symmetry()
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
        rotors, rotors_description = '', ''
        if any([i_r_dict['success'] for i_r_dict in species.rotors_dict.values()]):
            rotors = '\n\nrotors = ['
            rotors_description = '1D rotors:\n'
            for i in range(species.number_of_rotors):
                pivots = str(species.rotors_dict[i]['pivots'])
                scan = str(species.rotors_dict[i]['scan'])
                if species.rotors_dict[i]['success']:
                    rotor_path = species.rotors_dict[i]['scan_path']
                    rotor_type = determine_rotor_type(rotor_path)
                    top = str(species.rotors_dict[i]['top'])
                    try:
                        rotor_symmetry, max_e = determine_rotor_symmetry(rotor_path, species.label, pivots)
                    except RotorError:
                        logging.error('Could not determine rotor symmetry for species {0} between pivots {1}.'
                                      ' Setting the rotor symmetry to 1, which is probably WRONG.'.format(
                                       species.label, pivots))
                        rotor_symmetry = 1
                        max_e = None
                    max_e = ', max scan energy: {0:.2f} kJ/mol'.format(max_e) if max_e is not None else ''
                    free = ' (set as a FreeRotor)' if rotor_type == 'FreeRotor' else ''
                    rotors_description += 'pivots: ' + str(pivots) + ', dihedral: ' + str(scan) +\
                        ', rotor symmetry: ' + str(rotor_symmetry) + max_e + free + '\n'
                    if rotor_type == 'HinderedRotor':
                        rotors += input_files['arkane_hindered_rotor'].format(rotor_path=rotor_path, pivots=pivots,
                                                                              top=top, symmetry=rotor_symmetry)
                    elif rotor_type == 'FreeRotor':
                        rotors += input_files['arkane_free_rotor'].format(rotor_path=rotor_path, pivots=pivots,
                                                                          top=top, symmetry=rotor_symmetry)
                    if i < species.number_of_rotors - 1:
                        rotors += ',\n          '
                else:
                    rotors_description += '* Invalidated! pivots: ' + str(pivots) + ', dihedral: ' + str(scan) +\
                                          ', invalidation reason: ' + species.rotors_dict[i]['invalidation_reason'] +\
                                          '\n'

            rotors += ']'
            species.long_thermo_description += rotors_description + '\n'
        # write the Arkane species input file
        input_file_path = os.path.join(self.project_directory, 'output', folder_name, species.label,
                                       '{0}_arkane_input.py'.format(species.label))
        input_file = input_files['arkane_species']
        if self.use_bac and not species.is_ts:
            logging.info('Using the following BAC for {0}: {1}'.format(species.label, species.bond_corrections))
            bonds = '\n\nbonds = {0}'.format(species.bond_corrections)
        else:
            logging.debug('NOT using BAC for {0}'.format(species.label))
            bonds = ''
        input_file = input_file.format(linear=linear, bonds=bonds, symmetry=species.external_symmetry,
                                       multiplicity=species.multiplicity, optical=species.optical_isomers,
                                       model_chemistry=self.model_chemistry, sp_path=sp_path, opt_path=opt_path,
                                       freq_path=freq_path, rotors=rotors)
        with open(input_file_path, 'wb') as f:
            f.write(input_file)
        species.arkane_file = input_file_path
        return output_file_path

    def process(self):
        """Process ARC outputs and generate thermo and kinetics"""
        # Thermo:
        species_list_for_thermo_parity = list()
        species_for_thermo_lib = list()
        for species in self.species_dict.values():
            if not species.is_ts and 'ALL converged' in self.output[species.label]['status']:
                species_for_thermo_lib.append(species)
                output_file_path = self._generate_arkane_species_file(species)
                unique_arkane_species_label = False
                while not unique_arkane_species_label:
                    try:
                        arkane_spc = arkane_species(str(species.label), species.arkane_file)
                    except ValueError:
                        species.label += '_(' + str(randint(0, 999)) + ')'
                    else:
                        unique_arkane_species_label = True
                if species.mol_list:
                    arkane_spc.molecule = species.mol_list
                stat_mech_job = StatMechJob(arkane_spc, species.arkane_file)
                stat_mech_job.applyBondEnergyCorrections = self.use_bac
                stat_mech_job.modelChemistry = self.model_chemistry
                stat_mech_job.frequencyScaleFactor = assign_frequency_scale_factor(self.model_chemistry)
                stat_mech_job.execute(outputFile=output_file_path, plot=False)
                if species.generate_thermo:
                    thermo_job = ThermoJob(arkane_spc, 'NASA')
                    thermo_job.execute(outputFile=output_file_path, plot=False)
                    species.thermo = arkane_spc.getThermoData()
                    plotter.log_thermo(species.label, path=output_file_path)

                species.rmg_species = Species(molecule=[species.mol])
                species.rmg_species.reactive = True
                if species.mol_list:
                    species.rmg_species.molecule = species.mol_list  # add resonance structures for thermo determination
                try:
                    species.rmg_thermo = self.rmgdb.thermo.getThermoData(species.rmg_species)
                except ValueError:
                    logging.info('Could not retrieve RMG thermo for species {0}, possibly due to missing 2D structure '
                                 '(bond orders). Not including this species in the parity plots.'.format(species.label))
                else:
                    if species.generate_thermo:
                        species_list_for_thermo_parity.append(species)
        # Kinetics:
        rxn_list_for_kinetics_plots = list()
        arkane_spc_dict = dict()  # a dictionary with all species and the TSs
        for rxn in self.rxn_list:
            logging.info('\n\n')
            species = self.species_dict[rxn.ts_label]  # The TS
            if 'ALL converged' in self.output[species.label]['status'] and rxn.check_ts():
                self.copy_freq_output_for_ts(species.label)
                success = True
                rxn_list_for_kinetics_plots.append(rxn)
                output_file_path = self._generate_arkane_species_file(species)
                arkane_ts = arkane_transition_state(str(species.label), species.arkane_file)
                arkane_spc_dict[species.label] = arkane_ts
                stat_mech_job = StatMechJob(arkane_ts, species.arkane_file)
                stat_mech_job.applyBondEnergyCorrections = False
                if not self.model_chemistry:
                    stat_mech_job.modelChemistry = self.model_chemistry
                else:
                    stat_mech_job.applyAtomEnergyCorrections = False
                stat_mech_job.frequencyScaleFactor = assign_frequency_scale_factor(self.model_chemistry)
                stat_mech_job.execute(outputFile=None, plot=False)
                for spc in rxn.r_species + rxn.p_species:
                    if spc.label not in arkane_spc_dict.keys():
                        # add an extra character to the arkane_species label to distinguish between species calculated
                        #  for thermo and species calculated for kinetics (where we don't want to use BAC)
                        arkane_spc = arkane_species(str(spc.label + '_'), spc.arkane_file)
                        stat_mech_job = StatMechJob(arkane_spc, spc.arkane_file)
                        arkane_spc_dict[spc.label] = arkane_spc
                        stat_mech_job.applyBondEnergyCorrections = False
                        if not self.model_chemistry:
                            stat_mech_job.modelChemistry = self.model_chemistry
                        else:
                            stat_mech_job.applyAtomEnergyCorrections = False
                        stat_mech_job.frequencyScaleFactor = assign_frequency_scale_factor(self.model_chemistry)
                        stat_mech_job.execute(outputFile=None, plot=False)
                        # thermo_job = ThermoJob(arkane_spc, 'NASA')
                        # thermo_job.execute(outputFile=None, plot=False)
                        # arkane_spc.thermo = arkane_spc.getThermoData()
                rxn.dh_rxn298 = sum([product.thermo.getEnthalpy(298) for product in arkane_spc_dict.values()
                                     if product.label in rxn.products])\
                                - sum([reactant.thermo.getEnthalpy(298) for reactant in arkane_spc_dict.values()
                                       if reactant.label in rxn.reactants])
                arkane_rxn = arkane_reaction(label=str(rxn.label),
                                             reactants=[str(label + '_') for label in arkane_spc_dict.keys()
                                                        if label in rxn.reactants],
                                             products=[str(label + '_') for label in arkane_spc_dict.keys()
                                                       if label in rxn.products],
                                             transitionState=rxn.ts_label, tunneling='Eckart')
                kinetics_job = KineticsJob(reaction=arkane_rxn, Tmin=self.t_min, Tmax=self.t_max, Tcount=self.t_count)
                logging.info('Calculating rate for reaction {0}'.format(rxn.label))
                try:
                    kinetics_job.execute(outputFile=output_file_path, plot=False)
                except ValueError as e:
                    """
                    ValueError: One or both of the barrier heights of -9.35259 and 62.6834 kJ/mol encountered in Eckart
                    method are invalid.
                    """
                    logging.error('Failed to generate kinetics for {0} with message:\n{1}'.format(rxn.label, e))
                    success = False
                if success:
                    rxn.kinetics = kinetics_job.reaction.kinetics
                    plotter.log_kinetics(species.label, path=output_file_path)
                    rxn.rmg_reactions = rmgdb.determine_rmg_kinetics(rmgdb=self.rmgdb, reaction=rxn.rmg_reaction,
                                                                     dh_rxn298=rxn.dh_rxn298)

        logging.info('\n\n')
        output_dir = os.path.join(self.project_directory, 'output')

        if species_list_for_thermo_parity:
            plotter.draw_thermo_parity_plots(species_list_for_thermo_parity, path=output_dir)
            libraries_path = os.path.join(output_dir, 'RMG libraries')
            # species_list = [spc for spc in self.species_dict.values()]
            plotter.save_thermo_lib(species_for_thermo_lib, path=libraries_path,
                                    name=self.project, lib_long_desc=self.lib_long_desc)
        if rxn_list_for_kinetics_plots:
            plotter.draw_kinetics_plots(rxn_list_for_kinetics_plots, path=output_dir,
                                        t_min=self.t_min, t_max=self.t_max, t_count=self.t_count)
            libraries_path = os.path.join(output_dir, 'RMG libraries')
            plotter.save_kinetics_lib(rxn_list=rxn_list_for_kinetics_plots, path=libraries_path,
                                      name=self.project, lib_long_desc=self.lib_long_desc)

        self.clean_output_directory()

    def clean_output_directory(self):
        """
        A helper function to organize the output directory
        - remove redundant rotor.txt files (from kinetics jobs)
        - move remaining rotor files to the rotor directory
        - move the Arkane YAML file from the `species` directory to the base directory, and delete `species`
        """
        for base_folder in ['Species', 'rxns']:
            base_path = os.path.join(self.project_directory, 'output', base_folder)
            dir_names = list()
            for (_, dirs, _) in os.walk(base_path):
                dir_names.extend(dirs)
                break  # don't continue to explore subdirectories
            for species_label in dir_names:
                species_path = os.path.join(base_path, species_label)
                file_names = list()
                for (_, _, files) in os.walk(species_path):
                    file_names.extend(files)
                    break  # don't continue to explore subdirectories
                if any(['rotor' in file_name for file_name in file_names])\
                        and not os.path.exists(os.path.join(species_path, 'rotors')):
                    os.makedirs(os.path.join(species_path, 'rotors'))
                for file_name in file_names:
                    if '__rotor' in file_name:  # this is a duplicate `species__rotor.txt` file (with two underscores)
                        os.remove(os.path.join(species_path, file_name))
                    elif '_rotor' in file_name:  # move to the rotor directory
                        shutil.move(src=os.path.join(species_path, file_name),
                                    dst=os.path.join(species_path, 'rotors', file_name))
                if os.path.exists(os.path.join(species_path, 'species')):  # This is where Arkane saves the YAML file
                    species_yaml_files = os.listdir(os.path.join(species_path, 'species'))
                    if species_yaml_files:
                        species_yaml_file = species_yaml_files[0]
                        shutil.move(src=os.path.join(species_path, 'species', species_yaml_file),
                                    dst=os.path.join(species_path, species_yaml_file))
                    shutil.rmtree(os.path.join(species_path, 'species'))

    def copy_freq_output_for_ts(self, label):
        """
        Copy the frequency job output file into the TS geometry folder
        """
        calc_path = os.path.join(self.output[label]['freq'])
        output_path = os.path.join(self.project_directory, 'output', 'rxns', label, 'geometry', 'frequency.out')
        shutil.copyfile(calc_path, output_path)
