#!/usr/bin/env python3
# encoding: utf-8

"""
An adapter for executing Arkane.
"""

import os
import shutil
from typing import Type

from rmgpy.species import Species

import arkane.input
from arkane.input import (reaction as arkane_reaction,
                          species as arkane_input_species,
                          transitionState as arkane_transition_state,
                          )
from arkane.kinetics import KineticsJob
from arkane.statmech import StatMechJob
from arkane.thermo import ThermoJob

import arc.plotter as plotter
from arc.common import get_logger
from arc.exceptions import InputError, RotorError
from arc.job.inputs import input_files
from arc.reaction import ARCReaction
from arc.species.species import ARCSpecies, determine_rotor_symmetry, determine_rotor_type
from arc.statmech.adapter import StatmechAdapter
from arc.statmech.factory import register_statmech_adapter


logger = get_logger()


class ArkaneAdapter(StatmechAdapter):
    """
    A class for working with the Arkane statmech software.

    Args:
        output_directory (str): The path to the ARC project output directory.
        output_dict (dict): Keys are labels, values are output file paths.
                            See Scheduler for a description of this dictionary.
        use_bac (bool): Whether or not to use bond additivity corrections (BACs) for thermo calculations.
        sp_level (str, optional): The level of theory used for the single point energy calculation
                                  (could be a composite method), used for determining energy corrections.
        freq_scale_factor (float, optional): The harmonic frequencies scaling factor.
        species (ARCSpecies, optional): The species object.
        reaction (ARCReaction, optional): The reaction object.
        species_dict (dict, optional): Keys are labels, values are ARCSpecies objects.
        T_min (tuple, optional): The minimum temperature for kinetics computations, e.g., (500, 'K').
        T_max (tuple, optional): The maximum temperature for kinetics computations, e.g., (3000, 'K').
        T_count (int, optional): The number of temperature points between t_min and t_max for kinetics computations.
    """

    def __init__(self,
                 output_directory: str,
                 output_dict: dict,
                 use_bac: bool,
                 sp_level: str = '',
                 freq_scale_factor: float = 1.0,
                 species: Type[ARCSpecies] = None,
                 reaction: Type[ARCReaction] = None,
                 species_dict: dict = None,
                 T_min: tuple = None,
                 T_max: tuple = None,
                 T_count: int = 50):
        self.output_directory = output_directory
        self.output_dict = output_dict
        self.use_bac = use_bac
        self.sp_level = sp_level
        self.freq_scale_factor = freq_scale_factor
        self.species = species
        self.reaction = reaction
        self.species_dict = species_dict
        self.T_min = T_min
        self.T_max = T_max
        self.T_count = T_count

        if not self.output_directory:
            raise InputError('A project directory was not provided.')

    def __str__(self) -> str:
        """
        A short representation of the current ArkaneAdapter.

        Returns:
            str: The desired string representation.
        """
        str_representation = 'ArkaneAdapter('
        str_representation += f'output_directory={self.output_directory}, '
        str_representation += f'use_bac={self.use_bac}, '
        str_representation += f'sp_level={self.sp_level}, '
        str_representation += f'freq_scale_factor={self.freq_scale_factor}, '
        str_representation += f'species={self.species}, '
        str_representation += f'reaction={self.reaction}, '
        str_representation += f'T_min={self.T_min}, '
        str_representation += f'T_max={self.T_max}, '
        str_representation += f'T_count={self.T_count})'
        return str_representation

    def compute_thermo(self,
                       kinetics_flag: bool = False,
                       e0_only: bool = False,
                       ) -> None:
        """
        Generate thermodynamic data for a species.
        Populates the species.thermo attribute.

        Args:
            kinetics_flag (bool, optional): Whether this call is used for generating species statmech
                                            for a rate coefficient calculation.
            e0_only (bool, optional): Whether to only run statmech (w/o thermo) to compute E0.
        """
        if not kinetics_flag:
            # initialize the Arkane species_dict so that species for which thermo is calculated won't interfere
            # with species used for a rate coefficient calculation.
            arkane.input.species_dict = dict()

        if self.species is None:
            raise InputError('Cannot not compute thermo without a species object.')

        arkane_output_path = self.generate_arkane_species_file(species=self.species,
                                                               use_bac=self.use_bac)

        if arkane_output_path is not None:
            arkane_species = arkane_input_species(self.species.label, self.species.arkane_file)
            self.species.rmg_species = Species(molecule=[self.species.mol])
            self.species.rmg_species.reactive = True
            if self.species.mol_list:
                # add resonance structures for thermo determination
                arkane_species.molecule = self.species.mol_list
                self.species.rmg_species.molecule = self.species.mol_list
            statmech_success = self.run_statmech(arkane_species=arkane_species,
                                                 arkane_file_path=self.species.arkane_file,
                                                 arkane_output_path=arkane_output_path,
                                                 use_bac=self.use_bac,
                                                 sp_level=self.sp_level,
                                                 plot=False)
            if statmech_success:
                self.species.e0 = arkane_species.conformer.E0.value_si * 0.001  # convert to kJ/mol
                logger.debug(f'Assigned E0 to {self.species.label}: {self.species.e0:.2f} kJ/mol')
                if not e0_only:
                    thermo_job = ThermoJob(arkane_species, 'NASA')
                    thermo_job.execute(output_directory=arkane_output_path, plot=False)
                    self.species.thermo = arkane_species.get_thermo_data()
                    if not kinetics_flag:
                        plotter.log_thermo(self.species.label, path=arkane_output_path)
            else:
                logger.error(f'Could not run statmech job for species {self.species.label}')
        clean_output_directory(species_path=os.path.join(self.output_directory, 'Species', self.species.label))

    def compute_high_p_rate_coefficient(self) -> None:
        """
        Generate a high pressure rate coefficient for a reaction.
        Populates the reaction.kinetics attribute.
        """
        ts_species = self.species_dict[self.reaction.ts_label]
        if self.output_dict[ts_species.label]['convergence'] and self.reaction.check_ts():
            success = True
            arkane_output_path = self.generate_arkane_species_file(species=ts_species,
                                                                   use_bac=False)
            arkane_ts_species = arkane_transition_state(ts_species.label, ts_species.arkane_file)
            statmech_success = self.run_statmech(arkane_species=arkane_ts_species,
                                                 arkane_file_path=ts_species.arkane_file,
                                                 arkane_output_path=arkane_output_path,
                                                 use_bac=False,
                                                 sp_level=self.sp_level,
                                                 plot=False)
            if not statmech_success:
                logger.error(f'Could not run statmech job for TS species {ts_species.label} '
                             f'of reaction {self.reaction.label}')
            else:
                ts_species.e0 = arkane_ts_species.conformer.E0.value_si * 0.001  # convert to kJ/mol
                self.reaction.dh_rxn298 = \
                    sum([product.thermo.get_enthalpy(298) for product in self.reaction.p_species]) \
                    - sum([reactant.thermo.get_enthalpy(298) for reactant in self.reaction.r_species])
                arkane_rxn = arkane_reaction(label=str(self.reaction.label),
                                             reactants=[species.label for species in self.reaction.r_species],
                                             products=[species.label for species in self.reaction.p_species],
                                             transitionState=self.reaction.ts_label,
                                             tunneling='Eckart')
                kinetics_job = KineticsJob(reaction=arkane_rxn, Tmin=self.T_min, Tmax=self.T_max, Tcount=self.T_count)
                logger.info(f'Calculating rate for reaction {self.reaction.label}')
                try:
                    kinetics_job.execute(output_directory=arkane_output_path, plot=False)
                except (ValueError, OverflowError) as e:
                    # ValueError: One or both of the barrier heights of -9.3526 and 62.683 kJ/mol encountered in Eckart
                    # method are invalid.
                    #
                    #   File "RMG-Py/arkane/kinetics.py", line 136, in execute
                    #     generateKinetics(Tlist.value_si)
                    #   File "RMG-Py/arkane/kinetics.py", line 179, in generateKinetics
                    #     klist[i] = reaction.calculateTSTRateCoefficient(Tlist[i])
                    #   File "rmgpy/reaction.py", line 818, in rmgpy.reaction.Reaction.calculateTSTRateCoefficient
                    #   File "rmgpy/reaction.py", line 844, in rmgpy.reaction.Reaction.calculateTSTRateCoefficient
                    # OverflowError: math range error
                    logger.error(f'Failed to generate kinetics for {self.reaction.label}, got:\n{e}')
                    success = False
                if success:
                    self.reaction.kinetics = kinetics_job.reaction.kinetics
                    plotter.log_kinetics(ts_species.label, path=arkane_output_path)

        # initialize the Arkane species_dict in case another reaction uses the same species
        arkane.input.species_dict = dict()
        clean_output_directory(species_path=os.path.join(self.output_directory, 'rxns', ts_species.label))

    def run_statmech(self,
                     arkane_species: Type[Species],
                     arkane_file_path: str,
                     arkane_output_path: str = None,
                     use_bac: bool = False,
                     sp_level: str = '',
                     plot: bool = False,
                     ) -> bool:
        """
        A helper function for running an Arkane statmech job.

        Args:
            arkane_species (arkane_input_species): An instance of an Arkane species() object.
            arkane_file_path (str): The path to the Arkane species file (either in .py or YAML form).
            arkane_output_path (str): The path to the folder in which the Arkane output.py file will be saved.
            use_bac (bool): A flag indicating whether or not to use bond additivity corrections (True to use).
            sp_level (str, optional): The level of theory used for the single point energy calculation
                                      (could be a composite method), used for determining energy corrections.
            plot (bool): A flag indicating whether to plot a PDF of the calculated thermo properties (True to plot)

        Returns:
            bool: Whether the statmech job was successful.
        """
        success = True
        stat_mech_job = StatMechJob(arkane_species, arkane_file_path)
        stat_mech_job.applyBondEnergyCorrections = use_bac and sp_level
        stat_mech_job.modelChemistry = sp_level
        if not sp_level:
            # if this is a kinetics computation and we don't have a valid model chemistry, don't bother about it
            stat_mech_job.applyAtomEnergyCorrections = False
        stat_mech_job.frequencyScaleFactor = self.freq_scale_factor
        try:
            stat_mech_job.execute(output_directory=arkane_output_path, plot=plot)
        except Exception as e:
            logger.error(f'Arkane statmech job for species {arkane_species.label} failed with the error message:\n{e}')
            if stat_mech_job.applyBondEnergyCorrections \
                    and 'missing' in str(e).lower() and 'bac parameters for model chemistry' in str(e).lower():
                # try executing Arkane w/o BACs
                logger.warning('Trying to run Arkane without BACs')
                stat_mech_job.applyBondEnergyCorrections = False
                try:
                    stat_mech_job.execute(output_directory=arkane_output_path, plot=plot)
                except Exception as e:
                    logger.error(f'Arkane statmech job for {arkane_species.label} failed with the error message:\n{e}')
                    success = False
            else:
                success = False
        return success

    def generate_arkane_species_file(self,
                                     species: Type[ARCSpecies],
                                     use_bac: bool,
                                     ) -> str:
        """
        A helper function for generating an Arkane Python species file.
        Assigns the path of the generated file to the species.arkane_file attribute.

        Args:
            species (ARCSpecies): The species to process.
            use_bac (bool): Whether or not to use bond additivity corrections (BACs) for thermo calculations.

        Returns:
            str: The path to the species arkane folder (Arkane's default output folder).
        """
        folder_name = 'rxns' if species.is_ts else 'Species'
        species_folder_path = os.path.join(self.output_directory, folder_name, species.label)
        arkane_output_path = os.path.join(species_folder_path, 'arkane')
        if not os.path.isdir(arkane_output_path):
            os.makedirs(arkane_output_path)

        if species.yml_path is not None:
            species.arkane_file = species.yml_path
            return arkane_output_path

        species.determine_symmetry()

        sp_path = self.output_dict[species.label]['paths']['composite'] \
            or self.output_dict[species.label]['paths']['sp']
        if species.number_of_atoms == 1:
            freq_path = sp_path
            opt_path = sp_path
        else:
            freq_path = self.output_dict[species.label]['paths']['freq']
            opt_path = self.output_dict[species.label]['paths']['freq']

        return_none_text = None
        if not sp_path:
            return_none_text = 'path to the sp calculation'
        if not freq_path:
            return_none_text = 'path to the freq calculation'
        if not os.path.isfile(freq_path):
            return_none_text = f'the freq file in path {freq_path}'
        if not os.path.isfile(sp_path):
            return_none_text = f'the freq file in path {sp_path}'
        if return_none_text is not None:
            logger.error(f'Could not find {return_none_text} for species {species.label}. Not calculating properties.')
            return None

        rotors, rotors_description = '', ''
        if species.rotors_dict is not None and any([i_r_dict['pivots'] for i_r_dict in species.rotors_dict.values()]):
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
                        rotor_symmetry, max_e = determine_rotor_symmetry(species.label, pivots, rotor_path)
                    except RotorError:
                        logger.error(f'Could not determine rotor symmetry for species {species.label} between '
                                     f'pivots {pivots}. Setting the rotor symmetry to 1, '
                                     f'this could very well be WRONG.')
                        rotor_symmetry = 1
                        max_e = None
                    max_e = f', max scan energy: {max_e:.2f} kJ/mol' if max_e is not None else ''
                    free = ' (set as a FreeRotor)' if rotor_type == 'FreeRotor' else ''
                    rotors_description += f'pivots: {pivots}, dihedral: {scan}, ' \
                                          f'rotor symmetry: {rotor_symmetry}{max_e}{free}\n'
                    if rotor_type == 'HinderedRotor':
                        rotors += input_files['arkane_hindered_rotor'].format(rotor_path=rotor_path,
                                                                              pivots=pivots,
                                                                              top=top,
                                                                              symmetry=rotor_symmetry)
                    elif rotor_type == 'FreeRotor':
                        rotors += input_files['arkane_free_rotor'].format(rotor_path=rotor_path,
                                                                          pivots=pivots,
                                                                          top=top,
                                                                          symmetry=rotor_symmetry)
                    if i < species.number_of_rotors - 1:
                        rotors += ',\n          '
                else:
                    rotors_description += f'* Invalidated! pivots: {pivots}, dihedral: {scan}, ' \
                                          f'invalidation reason: {species.rotors_dict[i]["invalidation_reason"]}\n'

            rotors += ']'
            if 'rotors' not in species.long_thermo_description:
                species.long_thermo_description += rotors_description + '\n'

        # write the Arkane species input file
        bac_txt = '' if use_bac else '_no_BAC'
        input_file_path = os.path.join(species_folder_path, f'{species.label}_arkane_input{bac_txt}.py')
        input_file = input_files['arkane_input_species']
        if use_bac and not species.is_ts:
            logger.info(f'Using the following BAC for {species.label}: {species.bond_corrections}')
            bonds = f'bonds = {species.bond_corrections}\n\n'
        else:
            logger.debug(f'NOT using BAC for {species.label}')
            bonds = ''
        input_file = input_file.format(bonds=bonds,
                                       symmetry=species.external_symmetry,
                                       multiplicity=species.multiplicity,
                                       optical=bool(species.chiral_centers) + 1,  # Arkane accepts 1 or 2 here
                                       sp_level=self.sp_level,
                                       sp_path=sp_path,
                                       opt_path=opt_path,
                                       freq_path=freq_path,
                                       rotors=rotors)

        if freq_path:
            with open(input_file_path, 'w') as f:
                f.write(input_file)
            species.arkane_file = input_file_path
        else:
            species.arkane_file = None

        return arkane_output_path


def clean_output_directory(species_path: str) -> None:
    """
    Relocate Arkane's YAML files.

    Args:
        species_path (str): THe path to the species folder.
    """
    species_yaml_base_path = os.path.join(species_path, 'arkane', 'species')
    if os.path.exists(species_yaml_base_path):
        species_yaml_files = os.listdir(species_yaml_base_path)
        if species_yaml_files:
            for yml_file in species_yaml_files:
                if '.yml' in yml_file:
                    shutil.move(src=os.path.join(species_yaml_base_path, yml_file),
                                dst=os.path.join(species_path, yml_file))
        shutil.rmtree(species_yaml_base_path)


register_statmech_adapter('arkane', ArkaneAdapter)
