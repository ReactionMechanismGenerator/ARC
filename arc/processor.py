#!/usr/bin/env python
# encoding: utf-8

"""
Processor module for outputting thermoproperties and rates
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)
import os
import shutil
from random import randint

from arkane.input import species as arkane_input_species, transitionState as arkane_transition_state,\
    reaction as arkane_reaction, process_model_chemistry
from arkane.statmech import StatMechJob, assign_frequency_scale_factor
from arkane.thermo import ThermoJob
from arkane.kinetics import KineticsJob

from rmgpy.species import Species

from arc.common import get_logger
import arc.rmgdb as rmgdb
from arc.job.inputs import input_files
from arc import plotter
from arc.arc_exceptions import SchedulerError, RotorError
from arc.species.species import determine_rotor_symmetry, determine_rotor_type

##################################################################

logger = get_logger()


class Processor(object):
    """
    ARC Processor class. Post processes results in Arkane. The attributes are:

    ================== =========== =====================================================================================
    Attribute          Type        Description
    ================== =========== =====================================================================================
    `project`           ``str``    The project's name. Used for naming the directory.
    `species_dict`      ``dict``   Keys are labels, values are ARCSpecies objects
    `rxn_list`          ``list``   List of ARCReaction objects
    `output`            ``dict``   Keys are labels, values are output file paths
    `use_bac`           ``bool``   Whether or not to use bond additivity corrections for thermo calculations
    `sp_level`          ``str``    The single point level of theory, used for atom and bond corrections in Arkane
    `freq_level`        ``str``    The frequency level of theory, used for the frequency scaling factor in Arkane
                                     if `freq_scale_factor` was not given.
    `freq_scale_factor` ``float``  The harmonic frequencies scaling factor. Could be automatically determined
                                     if not available in Arkane and not provided by the user.
    `lib_long_desc`     ``str``    A multiline description of levels of theory for the outputted RMG libraries
    `project_directory` ``str``    The path of the ARC project directory
    `t_min`             ``tuple``  The minimum temperature for kinetics computations, e.g., (500, str('K'))
    `t_max`             ``tuple``  The maximum temperature for kinetics computations, e.g., (3000, str('K'))
    `t_count`           ``int``    The number of temperature points between t_min and t_max for kinetics computations
    ================== =========== =====================================================================================
    """
    def __init__(self, project, project_directory, species_dict, rxn_list, output, use_bac, model_chemistry,
                 lib_long_desc, rmgdatabase, t_min=None, t_max=None, t_count=None, freq_scale_factor=None):
        self.rmgdb = rmgdatabase
        self.project = project
        self.project_directory = project_directory
        self.species_dict = species_dict
        self.rxn_list = rxn_list
        self.output = output
        self.use_bac = use_bac
        self.sp_level, self.freq_level = process_model_chemistry(model_chemistry)
        self.freq_scale_factor = freq_scale_factor
        self.lib_long_desc = lib_long_desc
        load_thermo_libs, load_kinetic_libs = False, False
        if any([species.is_ts and species.final_xyz for species in self.species_dict.values()])\
                and any(['ALL converged' in out['status'] for out in output.values()]):
            load_kinetic_libs = True
        if any([species.generate_thermo for species in self.species_dict.values()])\
                and any(['ALL converged' in out['status'] for out in output.values()]):
            load_thermo_libs = True
        if any(['ALL converged' in spc_output_dict['status'] for spc_output_dict in output.values()]):
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
        A helper function for generating the Arkane species file.
        Assigns the input file path to species.arkane_file.

        Args:
            species (ARCSpecies): The species to process.

        Returns:
            str, unicode: The Arkane output path.
        """
        folder_name = 'rxns' if species.is_ts else 'Species'
        output_path = os.path.join(self.project_directory, 'output', folder_name, species.label, 'arkane')
        if not os.path.isdir(output_path):
            os.makedirs(output_path)

        if species.yml_path is not None:
            species.arkane_file = species.yml_path
            return output_path

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
                        logger.error('Could not determine rotor symmetry for species {0} between pivots {1}.'
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
        input_file = input_files['arkane_input_species']
        if self.use_bac and not species.is_ts:
            logger.info('Using the following BAC for {0}: {1}'.format(species.label, species.bond_corrections))
            bonds = 'bonds = {0}\n\n'.format(species.bond_corrections)
        else:
            logger.debug('NOT using BAC for {0}'.format(species.label))
            bonds = ''
        input_file = input_file.format(bonds=bonds, symmetry=species.external_symmetry,
                                       multiplicity=species.multiplicity, optical=species.optical_isomers,
                                       sp_level=self.sp_level, sp_path=sp_path, opt_path=opt_path,
                                       freq_path=freq_path, rotors=rotors)
        with open(input_file_path, 'wb') as f:
            f.write(input_file)
        species.arkane_file = input_file_path
        return output_path

    def process(self):
        """Process ARC outputs and generate thermo and kinetics"""
        # Thermo:
        species_list_for_thermo_parity = list()
        species_for_thermo_lib = list()
        species_for_transport_lib = list()
        unconverged_species = list()
        for species in self.species_dict.values():
            if not species.is_ts and 'ALL converged' in self.output[species.label]['status']:
                output_path = self._generate_arkane_species_file(species)
                unique_arkane_species_label = False
                while not unique_arkane_species_label:
                    try:
                        arkane_spc = arkane_input_species(str(species.label), species.arkane_file)
                    except ValueError:
                        species.label += '_' + str(randint(0, 999))
                    else:
                        unique_arkane_species_label = True
                species.rmg_species = Species(molecule=[species.mol])
                species.rmg_species.reactive = True
                if species.mol_list:
                    arkane_spc.molecule = species.mol_list
                    species.rmg_species.molecule = species.mol_list  # add resonance structures for thermo determination
                statmech_success = self._run_statmech(arkane_spc, species.arkane_file, output_path,
                                                      use_bac=self.use_bac)
                if not statmech_success:
                    continue

                if species.generate_thermo:
                    thermo_job = ThermoJob(arkane_spc, 'NASA')
                    thermo_job.execute(output_directory=output_path, plot=False)
                    species.thermo = arkane_spc.getThermoData()
                    plotter.log_thermo(species.label, path=output_path)
                    species_for_thermo_lib.append(species)
                if self.use_bac and self.sp_level:
                    # If BAC was used, save another Arkane YAML file of this species with no BAC, so it can be used
                    # for further rate calculations if needed (where the conformer.E0 has no BAC)
                    statmech_success = self._run_statmech(arkane_spc, species.arkane_file, output_path,
                                                          use_bac=False)
                    # if statmech_success:
                    #     arkane_spc.label += str('_no_BAC')
                    #     arkane_spc.thermo = None  # otherwise thermo won't be calculated, although we don't really care
                    #     thermo_job = ThermoJob(arkane_spc, 'NASA')
                    #     thermo_job.execute(output_directory=output_path, plot=False)
                try:
                    species.rmg_thermo = self.rmgdb.thermo.getThermoData(species.rmg_species)
                except (ValueError, AttributeError) as e:
                    logger.info('Could not retrieve RMG thermo for species {0}, possibly due to missing 2D structure '
                                '(bond orders). Not including this species in the parity plots.'
                                '\nGot: {1}'.format(species.label, e.message))
                else:
                    if species.generate_thermo:
                        species_list_for_thermo_parity.append(species)
                if 'onedmin converged' in self.output[species.label]['status'].lower():
                    species_for_transport_lib.append(species)
            elif 'ALL converged' not in self.output[species.label]['status']:
                unconverged_species.append(species)
        # Kinetics:
        rxn_list_for_kinetics_plots = list()
        arkane_spc_dict = dict()  # a dictionary with all species and the TSs
        for rxn in self.rxn_list:
            logger.info('\n\n')
            species = self.species_dict[rxn.ts_label]  # The TS
            if 'ALL converged' in self.output[species.label]['status'] and rxn.check_ts():
                self.copy_freq_output_for_ts(species.label)
                success = True
                rxn_list_for_kinetics_plots.append(rxn)
                output_path = self._generate_arkane_species_file(species)
                arkane_ts = arkane_transition_state(str(species.label), species.arkane_file)
                arkane_spc_dict[species.label] = arkane_ts
                self._run_statmech(arkane_ts, species.arkane_file, kinetics=True)
                for spc in rxn.r_species + rxn.p_species:
                    if spc.label not in arkane_spc_dict.keys():
                        # add an extra character to the arkane_species label to distinguish between species calculated
                        #  for thermo and species calculated for kinetics (where we don't want to use BAC)
                        arkane_spc = arkane_input_species(str(spc.label + '_'), spc.arkane_file)
                        self._run_statmech(arkane_spc, spc.arkane_file, kinetics=True)
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
                logger.info('Calculating rate for reaction {0}'.format(rxn.label))
                try:
                    kinetics_job.execute(output_directory=output_path, plot=False)
                except (ValueError, OverflowError) as e:
                    # ValueError: One or both of the barrier heights of -9.3526 and 62.683 kJ/mol encountered in Eckart
                    # method are invalid.
                    #
                    #   File "/home/alongd/Code/RMG-Py/arkane/kinetics.py", line 136, in execute
                    #     self.generateKinetics(self.Tlist.value_si)
                    #   File "/home/alongd/Code/RMG-Py/arkane/kinetics.py", line 179, in generateKinetics
                    #     klist[i] = self.reaction.calculateTSTRateCoefficient(Tlist[i])
                    #   File "rmgpy/reaction.py", line 818, in rmgpy.reaction.Reaction.calculateTSTRateCoefficient
                    #   File "rmgpy/reaction.py", line 844, in rmgpy.reaction.Reaction.calculateTSTRateCoefficient
                    # OverflowError: math range error
                    logger.error('Failed to generate kinetics for {0} with message:\n{1}'.format(rxn.label, e))
                    success = False
                if success:
                    rxn.kinetics = kinetics_job.reaction.kinetics
                    plotter.log_kinetics(species.label, path=output_path)
                    rxn.rmg_reactions = rmgdb.determine_rmg_kinetics(rmgdb=self.rmgdb, reaction=rxn.rmg_reaction,
                                                                     dh_rxn298=rxn.dh_rxn298)

        logger.info('\n\n')
        output_dir = os.path.join(self.project_directory, 'output')
        libraries_path = os.path.join(output_dir, 'RMG libraries')

        if species_list_for_thermo_parity:
            plotter.draw_thermo_parity_plots(species_list_for_thermo_parity, path=output_dir)
            plotter.save_thermo_lib(species_for_thermo_lib, path=libraries_path,
                                    name=self.project, lib_long_desc=self.lib_long_desc)

        if species_for_transport_lib:
            plotter.save_transport_lib(species_for_thermo_lib, path=libraries_path, name=self.project)

        if rxn_list_for_kinetics_plots:
            plotter.draw_kinetics_plots(rxn_list_for_kinetics_plots, path=output_dir,
                                        t_min=self.t_min, t_max=self.t_max, t_count=self.t_count)
            plotter.save_kinetics_lib(rxn_list=rxn_list_for_kinetics_plots, path=libraries_path,
                                      name=self.project, lib_long_desc=self.lib_long_desc)

        self._clean_output_directory()
        if unconverged_species:
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)
            with open(os.path.join(output_dir, 'unconverged_species.log'), 'w') as f:
                for spc in unconverged_species:
                    f.write(spc.label)
                    if spc.is_ts:
                        f.write(str(' rxn: {0}'.format(spc.rxn_label)))
                    elif spc.mol is not None:
                        f.write(str(' SMILES: {0}'.format(spc.mol.toSMILES())))
                    f.write(str('\n'))

    def _run_statmech(self, arkane_spc, arkane_file, output_path=None, use_bac=False, kinetics=False, plot=False):
        """A helper function for running an Arkane statmech job

        Args:
            arkane_spc (str, unicode): An Arkane species() function representor.
            arkane_file (str, unicode): The path to the Arkane species file (either in .py or YAML form).
            output_path (str, unicode): The path to the folder containing the Arkane output.py file.
            use_bac (bool): A flag indicating whether or not to use bond additivity corrections (True to use).
            kinetics (bool) A flag indicating whether this specie is part of a kinetics job.
            plot (bool): A flag indicating whether to plot a PDF of the calculated thermo properties (True to plot)

        Returns:
            bool: Whether the job was successful (True for successful).
        """
        success = True
        stat_mech_job = StatMechJob(arkane_spc, arkane_file)
        stat_mech_job.applyBondEnergyCorrections = use_bac and not kinetics and self.sp_level
        if not kinetics or (kinetics and self.sp_level):
            # currently we have to use a model chemistry for thermo
            stat_mech_job.modelChemistry = self.sp_level
        else:
            # if this is a kinetics computation and we don't have a valid model chemistry, don't bother about it
            stat_mech_job.applyAtomEnergyCorrections = False
        # Use the scaling factor if given, else try determining it from Arkane
        # (defaults to 1 and prints a warning if not found)
        stat_mech_job.frequencyScaleFactor = self.freq_scale_factor or assign_frequency_scale_factor(self.freq_level)
        try:
            stat_mech_job.execute(output_directory=os.path.join(output_path, 'output.py'), plot=plot)
        except Exception:
            success = False
        return success

    def _clean_output_directory(self):
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

                # Arkane YAML files:
                species_yaml_base_path = os.path.join(species_path, 'arkane', 'species')
                if os.path.exists(species_yaml_base_path):  # The Arkane YAML file is here
                    species_yaml_files = os.listdir(species_yaml_base_path)
                    if species_yaml_files:
                        for yml_file in species_yaml_files:
                            shutil.move(src=os.path.join(species_yaml_base_path, yml_file),
                                        dst=os.path.join(species_path, 'arkane', yml_file))
                    shutil.rmtree(species_yaml_base_path)

    def copy_freq_output_for_ts(self, label):
        """
        Copy the frequency job output file into the TS geometry folder
        """
        calc_path = os.path.join(self.output[label]['freq'])
        output_path = os.path.join(self.project_directory, 'output', 'rxns', label, 'geometry', 'frequency.out')
        shutil.copyfile(calc_path, output_path)
