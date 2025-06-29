"""
An adapter for executing Arkane.
"""

import os
import re
import shutil
from abc import ABC
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

from mako.template import Template
import numpy as np

import arc.plotter as plotter
from arc.checks.ts import check_ts, ts_passed_checks
from arc.common import ARC_PATH, get_logger, read_yaml_file
import arc.constants as constants
from arc.exceptions import InputError, RotorError
from arc.imports import input_files, settings
from arc.job.local import execute_command
from arc.parser.parser import parse_1d_scan_energies, parse_frequencies
from arc.species.converter import xyz_to_coords_and_element_numbers
from arc.species.species import determine_rotor_symmetry, determine_rotor_type
from arc.statmech.adapter import StatmechAdapter
from arc.statmech.factory import register_statmech_adapter

if TYPE_CHECKING:
    from arc.level import Level
    from arc.reaction import ARCReaction
    from arc.species.species import ARCSpecies


RMG_DB_PATH = settings['RMG_DB_PATH']
logger = get_logger()


main_input_template = """#!/usr/bin/env python
# -*- coding: utf-8 -*-

title = '${title}'
description = \"\"\"
${description}
\"\"\"
${model_chemistry}${atom_energies}${freq_scale_factor}
useHinderedRotors = ${use_hindered_rotors}
useBondCorrections = ${use_bac}
bondCorrectionType = '${bac_type}'

% for spc in species_list:
species(${spc['label']}, ${spc['path']}${spc['pdep_data'] if 'pdep_data' in spc else ''},
        structure=SMILES('${spc['smiles']}'))
% endfor

% for ts in ts_list:
transitionState(${ts['label']}, ${ts['path']})
% endfor

% for rxn in reaction_list:
reaction(
    label = '${rxn['label']}',
    reactants = '${rxn['reactants']}',
    products = '${rxn['products']}',
    transitionState = '${rxn['ts_label']}',
    tunneling = '${rxn['tunneling']}',
)
% endfor

% if len(reaction_list):
% for rxn in reaction_list:
kinetics(label='${rxn['label']}',
         Tmin=(${t_min or 300}, 'K'), Tmax=(${t_max or 3000}, 'K'), Tcount=${t_count or 25})
% endfor
% endif

% if ${compute_thermo}:
% for spc in species_list:
thermo('${spc['label']}', 'NASA')
% endfor
% endif

"""

species_input_template = """#!/usr/bin/env python
# -*- coding: utf-8 -*-

linear = ${linear}
spinMultiplicity = ${spin_multiplicity}

energy = ${sp_path}
geometry = ${freq_path}
frequencies = ${freq_path}

%if ${use_hindered_rotors}:
rotors = [
% for rotor in rotors:
    HinderedRotor(
        scanLog=Log('${rotor['scan_path']}'),
        pivots = ${rotor['pivots']},
        top = ${rotor['top']},
        symmetry = ${rotor['symmetry']},
        fit='best',
    ),
% endfor
]
%endif

"""


class ArkaneAdapter(StatmechAdapter, ABC):
    """
    A class for working with the Arkane statmech software.

    Args:
        output_directory (str): The path to the ARC project output directory.
        calcs_directory (str): The path to the ARC project calculations directory.
        output_dict (dict): Keys are labels, values are output file paths.
                            See Scheduler for a description of this dictionary.
        bac_type (str): The bond additivity correction type. 'p' for Petersson- or 'm' for Melius-type BAC.
                        ``None`` to not use BAC.
        species (List[ARCSpecies]): A list of ARCSpecies objects to compute thermodynamic properties for.
        reactions (Optional[List[ARCReaction]]): A list of ARCReaction objects to compute kinetics for.
        sp_level (Level, optional): The level of theory used for energy corrections.
        freq_scale_factor (float, optional): The harmonic frequencies scaling factor.
        skip_nmd (bool, optional): Whether to skip the normal mode displacement check. ``True`` to skip.
        species_dict (dict, optional): Keys are labels, values are ARCSpecies objects.
        T_min (tuple, optional): The minimum temperature for kinetics computations, e.g., (500, 'K').
        T_max (tuple, optional): The maximum temperature for kinetics computations, e.g., (3000, 'K').
        T_count (int, optional): The number of temperature points between t_min and t_max for kinetics computations.
    """
    def __init__(self,
                 output_directory: str,
                 calcs_directory: str,
                 output_dict: dict,
                 species: List['ARCSpecies'],
                 reactions: Optional[List['ARCReaction']] = None,
                 bac_type: str  ='p',
                 sp_level: Optional['Level'] = None,
                 freq_level: Optional['Level'] = None,
                 freq_scale_factor: float = 1.0,
                 skip_nmd: bool = False,
                 species_dict: dict = None,
                 T_min: tuple = None,
                 T_max: tuple = None,
                 T_count: int = 50,
                 ):
        self.output_directory = output_directory
        self.calcs_directory = calcs_directory
        self.output_dict = output_dict
        self.bac_type = bac_type
        self.species = species
        self.reactions = reactions
        self.sp_level = sp_level
        self.freq_level = freq_level
        self.freq_scale_factor = freq_scale_factor
        self.skip_nmd = skip_nmd
        self.species_dict = species_dict
        self.T_min = T_min
        self.T_max = T_max
        self.T_count = T_count
        if not self.output_directory or not self.calcs_directory:
            raise InputError(f'Output and calcs directories must be given, got: {self.output_directory}, {self.calcs_directory}')

    def __str__(self) -> str:
        """
        A short representation of the current ArkaneAdapter.

        Returns:
            str: The desired string representation.
        """
        str_ = 'ArkaneAdapter('
        str_ += f'output_directory={self.output_directory}, '
        str_ += f'calcs_directory={self.calcs_directory}, '
        str_ += f'bac_type={self.bac_type}, '
        if self.sp_level is not None:
            str_ += f'sp_level={self.sp_level.simple()}, '
        if self.freq_level is not None:
            str_ += f'freq_level={self.freq_level.simple()}, '
        str_ += f'freq_scale_factor={self.freq_scale_factor}, '
        str_ += f'species={[s.label for s in self.species]}, '
        if self.reactions is not None:
            str_ += f'reactions={[r.label for r in self.reactions]}, '
        str_ += f'T_min={self.T_min}, '
        str_ += f'T_max={self.T_max}, '
        str_ += f'T_count={self.T_count})'
        return str_

    def compute_thermo(self,
                       e0_only: bool = False,
                       skip_rotors: bool = False,
                       ) -> None:
        """
        Generate thermodynamic data for a species. Populates the species.thermo attribute.

        Args:
            e0_only (bool, optional): Whether to only run statmech (w/o thermo) to compute E0.
            skip_rotors (bool, optional): Whether to skip internal rotor consideration. Default: ``False``.
        """
        statmech_dir = create_statmech_dir(calcs_directory=self.calcs_directory,
                                           subdir='thermo',
                                           delete_existing_subdir=True)
        self.generate_arkane_input(statmech_dir=statmech_dir, skip_rotors=skip_rotors)
        self.generate_species_files(statmech_dir, skip_rotors, check_compute_thermo=True)
        run_arkane(statmech_dir)
        self.parse_arkane_thermo_output(statmech_dir)

    def compute_high_p_rate_coefficient(self,
                                        skip_rotors: bool = False,
                                        estimate_dh_rxn: bool = False,
                                        require_ts_convergence: bool = True,
                                        verbose: bool = False,
                                        ) -> None:
        """
        Generate a high pressure rate coefficient for a reaction.
        Populates the reaction.kinetics attribute.

        Args:
            skip_rotors (bool, optional): Whether to skip internal rotor consideration. Default: ``False``.
            estimate_dh_rxn (bool, optional): Whether to estimate DH reaction instead of computing it. Default: ``False``.
                                              Useful for checking that the reaction could in principle be computed even
                                              when thermodynamic properties of reactants and products were still not computed.
            require_ts_convergence (bool, optional): Whether to attempt computing a rate only for converged TS species.
            verbose (bool, optional): Whether to log messages.
        """
        self.filter_out_unconverged_reactions(require_ts_convergence=require_ts_convergence)
        statmech_dir = create_statmech_dir(calcs_directory=self.calcs_directory,
                                           subdir='kinetics',
                                           delete_existing_subdir=True)
        self.generate_arkane_input(statmech_dir=statmech_dir, skip_rotors=skip_rotors)
        self.generate_species_files(statmech_dir, skip_rotors, check_compute_thermo=False)
        self.generate_ts_files(statmech_dir, skip_rotors)
        run_arkane(statmech_dir)
        self.parse_arkane_kinetics_output(statmech_dir)
        for reaction in self.reactions:
            plotter.log_kinetics(reaction.ts_species.label, path=statmech_dir)
            clean_output_directory(species_path=os.path.join(self.output_directory, 'rxns', reaction.ts_species.label),
                                   is_ts=True)

    def set_reaction_dh_rxn(self, estimate_dh_rxn: bool):
        """
        Set the reaction enthalpy of reaction at 298 K (dh_rxn298) for the given reaction.

        Args:
            estimate_dh_rxn (bool): Whether to estimate the enthalpy of reaction.
                                    If True, uses electronic energies or E0 values if available.
        """
        for reaction in self.reactions:
            est_dh_rxn = estimate_dh_rxn \
                         or any(spc.thermo is None for spc in reaction.r_species + reaction.p_species)
            if not est_dh_rxn:
                reaction.dh_rxn298 = \
                    sum([product.thermo.get_enthalpy(298) * reaction.get_species_count(product, well=1)
                         for product in reaction.p_species]) \
                    - sum([reactant.thermo.get_enthalpy(298) * reaction.get_species_count(reactant, well=0)
                           for reactant in reaction.r_species])
            elif all([spc.e_elect is not None for spc in reaction.r_species + reaction.p_species]):
                reaction.dh_rxn298 = \
                    sum([product.e_elect * 1e3 * reaction.get_species_count(product, well=1)
                         for product in reaction.p_species]) \
                    - sum([reactant.e_elect * 1e3 * reaction.get_species_count(reactant, well=0)
                           for reactant in reaction.r_species])
            elif all([spc.e0 is not None for spc in reaction.r_species + reaction.p_species]):
                reaction.dh_rxn298 = \
                    sum([product.e0 * 1e3 * reaction.get_species_count(product, well=1)
                         for product in reaction.p_species]) \
                    - sum([reactant.e0 * 1e3 * reaction.get_species_count(reactant, well=0)
                           for reactant in reaction.r_species])

    def filter_out_unconverged_reactions(self, require_ts_convergence: bool = True) -> None:
        """
        Filter out reactions with unconverged transition states.

        Args:
            require_ts_convergence (bool): Whether to filter out reactions with unconverged TSs.
        """
        if require_ts_convergence:
            self.reactions = [reaction for reaction in self.reactions if
                              self.output_dict[reaction.ts_species.label]['convergence']]
            unconverged_rxns = [reaction for reaction in self.reactions if
                                not self.output_dict[reaction.ts_species.label]['convergence']]
            if unconverged_rxns:
                logger.warning(f'The following reactions have unconverged TSs and will not be computed:')
                for reaction in unconverged_rxns:
                    logger.warning(f'  {reaction.label} with TS {reaction.ts_species.label}')

    def generate_arkane_input(self, statmech_dir: str, skip_rotors: bool = False) -> None:
        """
        Generate the Arkane main input file.

        Args:
            statmech_dir (str): The path to the statmech directory.
            skip_rotors (bool, optional): Whether to skip internal rotor consideration. Default: ``False``.
        """
        input_path = os.path.join(statmech_dir, 'input.py')
        input_content = self.render_arkane_input_template(statmech_dir=statmech_dir, skip_rotors=skip_rotors)
        with open(input_path, 'w', encoding='utf-8') as f:
            f.write(input_content)

    def render_arkane_input_template(self, statmech_dir: str, skip_rotors: bool = False) -> str:
        """
        Render the Arkane main input template.

        Args:
            statmech_dir (str): The path to the statmech directory.
            skip_rotors (bool, optional): Whether to skip internal rotor consideration. Default: ``False``.
        """
        species_list = [{'label': spc.label,
                         'path': os.path.join(statmech_dir, 'species', f'{spc.label}.py'),
                         'smiles': spc.mol.copy(deep=True).to_smiles()}
                        for spc in self.species if spc.compute_thermo]

        model_chemistry = get_arkane_model_chemistry(sp_level=self.sp_level,
                                                     freq_level=self.freq_level,
                                                     freq_scale_factor=self.freq_scale_factor
                                                     ) or ''

        aec_dict = read_yaml_file(os.path.join(ARC_PATH, 'data', 'AEC.yml'))
        atom_energies = f'\natomEnergies = {aec_dict[self.sp_level.simple()]}' \
            if self.sp_level.simple() in aec_dict else ''

        freq_scale_factor = f'\nfrequencyScaleFactor = {self.freq_scale_factor}' \
            if self.freq_scale_factor is not None else ''

        return Template(main_input_template).render(
            title='Arkane thermo calculation',
            description='Generated by ARC.',
            model_chemistry=model_chemistry,
            atom_energies=atom_energies,
            freq_scale_factor=freq_scale_factor,
            use_hindered_rotors='True' if not skip_rotors else 'False',
            use_bac='True' if self.bac_type is not None else 'False',
            bac_type=self.bac_type or 'p',
            species_list=species_list,
            ts_list=[],
            reaction_list=[],
            compute_thermo=True,
            t_min=self.T_min,
            t_max=self.T_max,
            t_count=self.T_count,
        )

    def generate_species_files(self, statmech_dir: str,
                               skip_rotors: bool,
                               check_compute_thermo: bool = False,
                               ) -> None:
        """
        Generate individual species input files.

        Args:
            statmech_dir (str): The path to the statmech directory.
            skip_rotors (bool): Whether to skip internal rotor consideration.
            check_compute_thermo (bool, optional): Whether to check if species.compute_thermo is True.
                                                   Default: ``False``.
        """
        species_dir = os.path.join(statmech_dir, 'species')
        os.makedirs(species_dir, exist_ok=True)
        for spc in self.species:
            if check_compute_thermo and not spc.compute_thermo:
                continue
            self.generate_species_file(spc, species_dir, skip_rotors)

    def generate_ts_files(self, statmech_dir: str,
                          skip_rotors: bool,
                          ) -> None:
        """
        Generate individual species input files.

        Args:
            statmech_dir (str): The path to the statmech directory.
            skip_rotors (bool): Whether to skip internal rotor consideration.
        """
        ts_dir = os.path.join(statmech_dir, 'TSs')
        os.makedirs(ts_dir, exist_ok=True)
        for ts in [rxn.ts_species for rxn in self.reactions]:
            self.generate_species_file(ts, ts_dir, skip_rotors)

    def generate_species_file(self, species, species_dir: str, skip_rotors: bool) -> None:
        """
        Generate input file for a single species.

        Args:
            species (ARCSpecies): The species object to generate the file for.
            species_dir (str): The directory to save the species file in.
            skip_rotors (bool): Whether to skip internal rotor consideration.
        """
        file_path = os.path.join(species_dir, f'{species.label}.py')
        species.arkane_file = file_path
        if os.path.isfile(file_path):
            os.remove(file_path)
        rotors = [rotor for rotor in species.rotors_dict.values() if rotor['success']]
        use_rotors = not skip_rotors and bool(rotors)
        content = Template(species_input_template).render(
            linear=species.is_linear,
            spin_multiplicity=species.multiplicity,
            sp_path=self.output_dict[species.label]['paths']['composite']
                    or self.output_dict[species.label]['paths']['sp'],
            freq_path=self.output_dict[species.label]['paths']['freq'],
            use_hindered_rotors=use_rotors,
            rotors=rotors,
        )
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

    def parse_arkane_thermo_output(self, statmech_dir: str) -> None:
        """Parse Arkane thermodynamic output and assign results to species."""
        output_path = os.path.join(statmech_dir, 'output.py')
        if not os.path.isfile(output_path):
            logger.error(f'Missing Arkane output: {output_path}')
            return

        with open(output_path, 'r', encoding='utf-8') as f:
            output_content = f.read()

        for species in self.species:
            if not species.compute_thermo:
                continue
            parse_species_thermo(species, output_content)
            clean_output_directory(os.path.join(self.output_directory, 'Species', species.label))

    def parse_arkane_kinetics_output(self, statmech_dir: str) -> None:
        """Parse Arkane kinetic output and assign results to reactions."""
        output_path = os.path.join(statmech_dir, 'output.py')
        if not os.path.isfile(output_path):
            logger.error(f'Missing Arkane output: {output_path}')
            return

        with open(output_path, 'r', encoding='utf-8') as f:
            output_content = f.read()

        for rxn in self.reactions:
            parse_reaction_kinetics(rxn, output_content)











    # def run_statmech(self,
    #                  arkane_species: Union[Species, TransitionState],
    #                  arkane_file_path: str,
    #                  arkane_output_path: str = None,
    #                  bac_type: Optional[str] = None,
    #                  sp_level: Optional['Level'] = None,
    #                  plot: bool = False,
    #                  ) -> Tuple[Union[Species, TransitionState], bool]:
    #     """
    #     A helper function for running an Arkane statmech job.
    #
    #     Args:
    #         arkane_species (Union[Species, TransitionState]): An instance of an Arkane species() object.
    #         arkane_file_path (str): The path to the Arkane species file (either in .py or YAML form).
    #         arkane_output_path (str): The path to the folder in which the Arkane output.py file will be saved.
    #         bac_type (str, optional): The bond additivity correction type. 'p' for Petersson- or 'm' for Melius-type BAC.
    #                                   ``None`` to not use BAC.
    #         sp_level (Level, optional): The level of theory used for energy corrections.
    #         plot (bool): A flag indicating whether to plot a PDF of the calculated thermo properties (True to plot)
    #
    #     Returns:
    #         Tuple[Union[Species, TransitionState], bool]:
    #             - The Arkane Species or TransitionState object instance with an initialized ``Conformer`` object instance.
    #             - Whether the statmech job was successful.
    #     """
    #     success = True
    #     statmech_job = self.initialize_statmech_job_object(arkane_species=arkane_species,
    #                                                        arkane_file_path=arkane_file_path,
    #                                                        bac_type=bac_type,
    #                                                        sp_level=sp_level,
    #                                                        )
    #     try:
    #         statmech_job.execute(output_directory=arkane_output_path, plot=plot)
    #     except Exception as e:
    #         if 'could not be identified' in str(e):
    #             return self.run_statmech_using_molecular_properties(arkane_species=arkane_species,
    #                                                                 arkane_file_path=arkane_file_path,
    #                                                                 sp_level=sp_level,
    #                                                                 )
    #         logger.error(f'Arkane statmech job for species {arkane_species.label} failed with the error message:\n{e}')
    #         if statmech_job.applyBondEnergyCorrections \
    #                 and 'missing' in str(e).lower() and 'bac parameters for model chemistry' in str(e).lower():
    #             logger.warning('Trying to run Arkane without BACs')
    #             statmech_job.applyBondEnergyCorrections = False
    #             try:
    #                 statmech_job.execute(output_directory=arkane_output_path, plot=plot)
    #             except Exception as e:
    #                 logger.error(f'Arkane statmech job for {arkane_species.label} failed with the error message:\n{e}')
    #                 success = False
    #         else:
    #             success = False
    #     return arkane_species, success

    # def run_statmech_using_molecular_properties(self,
    #                                             arkane_species: Union[Species, TransitionState],
    #                                             arkane_file_path: str,
    #                                             sp_level: Optional['Level'] = None,
    #                                             ) -> Tuple[Union[Species, TransitionState], bool]:
    #     """
    #     A helper function for running an Arkane statmech job using directly entered molecular properties.
    #     Useful for computations that were done using software not supported by Arkane.
    #
    #     Args:
    #         arkane_species (Union[Species, TransitionState]): An instance of an Arkane species() object.
    #         arkane_file_path (str): The path to the Arkane species file (either in .py or YAML form).
    #         sp_level (Level, optional): The level of theory used for energy corrections.
    #
    #     Returns:
    #         Tuple[Union[Species, TransitionState], bool]:
    #             - The Arkane Species or TransitionState object instance with an initialized ``Conformer`` object instance.
    #             - Whether the statmech job was successful.
    #     """
    #     logger.warning(f'Running statmech for {arkane_species.label} using electronic energy only (no ZPE correction).')
    #     success = True
    #     species = self.species_dict[arkane_species.label] \
    #         if self.species_dict is not None and arkane_species.label in self.species_dict.keys() else self.species
    #     arkane_species = self.initialize_arkane_species_directly(arc_species=species,
    #                                                              path=os.path.dirname(arkane_file_path),
    #                                                              sp_level=sp_level,
    #                                                              )
    #     return arkane_species, success

    # def initialize_arkane_species_directly(self,
    #                                        arc_species: 'ARCSpecies',
    #                                        path: str,
    #                                        sp_level: Optional['Level'] = None,
    #                                        ) -> Union[Species, TransitionState]:
    #     """
    #     Write the species molecular properties Arkane input file.
    #
    #     Args:
    #         arc_species (ARCSpecies): The species object instance.
    #         path (str): The working folder path.
    #         sp_level (Level, optional): The level of theory used for energy corrections.
    #
    #     Returns:
    #         Union[Species, TransitionState]: An initialized Arkane species with a ``Conformer`` object instance.
    #     """
    #     negative_freqs = None
    #     arc_species.determine_symmetry()
    #     yml_out_path = os.path.join(path, 'output.yml')
    #     output_dict = read_yaml_file(yml_out_path) if os.path.isfile(yml_out_path) else None
    #     modes = [IdealGasTranslation(mass=(arc_species.mol.get_molecular_weight() * 1E3, 'g/mol'))]
    #     coords, z_list = xyz_to_coords_and_element_numbers(arc_species.get_xyz())
    #     moments_of_inertia = list(get_principal_moments_of_inertia(coords=np.array(coords, dtype=np.float64),
    #                                                                numbers=np.array(z_list, dtype=np.float64))[0])
    #     linear = any([moment <= 0.01 for moment in moments_of_inertia])
    #     if linear:
    #         moments_of_inertia = [moment for moment in moments_of_inertia if moment > 0.01]
    #         if len(moments_of_inertia):
    #             modes.append(LinearRotor(inertia=(moments_of_inertia[0], 'amu*angstrom^2'),
    #                                      symmetry=arc_species.external_symmetry))
    #         else:
    #             # This is a single atom molecule.
    #             pass
    #     else:
    #         modes.append(NonlinearRotor(inertia=(moments_of_inertia, 'amu*angstrom^2'),
    #                                     symmetry=arc_species.external_symmetry))
    #     if not arc_species.is_monoatomic():
    #         freqs = parse_frequencies(self.output_dict[arc_species.label]['paths']['freq'])
    #         negative_freqs = np.array([freq for freq in freqs if freq < 0], dtype=np.float64)
    #         freqs = np.array([freq for freq in freqs if freq > 0], dtype=np.float64)
    #         modes.append(HarmonicOscillator(frequencies=(freqs, 'cm^-1')))
    #
    #     corr = get_atomic_energy_corrections(arc_species=arc_species, sp_level=sp_level)
    #     conformer = Conformer(E0=(arc_species.e_elect + corr, 'kJ/mol'),
    #                           modes=modes,
    #                           spin_multiplicity=arc_species.multiplicity,
    #                           optical_isomers=arc_species.optical_isomers,
    #                           mass=([get_element_mass(s)[0] for s in arc_species.get_xyz()['symbols']], 'amu'),
    #                           number=z_list,
    #                           coordinates=(coords, 'angstrom'),
    #                           )
    #     for rotor in arc_species.rotors_dict.values():
    #         if not rotor['success']:
    #             continue
    #         inertia = conformer.get_internal_reduced_moment_of_inertia(pivots=rotor['pivots'], top1=rotor['top'])
    #         fourier_rotor = HinderedRotor(inertia=(inertia, 'amu*angstrom^2'), symmetry=rotor['symmetry'] or 1)
    #         energies, angles = parse_1d_scan_energies(rotor['scan_path'])
    #         fourier_rotor.fit_fourier_potential_to_data(np.array(angles, dtype=np.float64),
    #                                                     np.array(energies, dtype=np.float64))
    #         conformer.modes.append(fourier_rotor)
    #     hessian = output_dict['hessian'] if output_dict is not None and 'hessian' in output_dict.keys() else None
    #     if hessian is not None and len(z_list) > 1 and len(arc_species.rotors_dict.keys()) > 0:
    #         frequencies = np.array(project_rotors(conformer=conformer,
    #                                               hessian=np.array(hessian, np.float64),
    #                                               rotors=[[rotor['pivots'], rotor['top'], rotor['symmetry']]
    #                                                       for rotor in arc_species.rotors_dict.values()],
    #                                               linear=linear,
    #                                               is_ts=arc_species.is_ts,
    #                                               label=self.species.label,
    #                                               ))
    #         for mode in conformer.modes:
    #             if isinstance(mode, HarmonicOscillator):
    #                 mode.frequencies = (frequencies * self.freq_scale_factor, 'cm^-1')
    #     if arc_species.is_ts:
    #         if arc_species.label in arkane.input.transition_state_dict.keys():
    #             del arkane.input.transition_state_dict[arc_species.label]
    #         arkane_species = arkane_transition_state(label=arc_species.label)
    #         arkane_species.frequency = (negative_freqs[0] * self.freq_scale_factor, 'cm^-1')
    #     else:
    #         if arc_species.label in arkane.input.species_dict.keys():
    #             del arkane.input.species_dict[arc_species.label]
    #         arkane_species = arkane_input_species(label=arc_species.label)
    #         arkane_species.molecule = [arc_species.mol.copy(deep=True)]
    #         arkane_species.reactive = True
    #     arkane_species.conformer = conformer
    #     return arkane_species

    # def initialize_statmech_job_object(self,
    #                                    arkane_species: Species,
    #                                    arkane_file_path: str,
    #                                    bac_type: Optional[str] = None,
    #                                    sp_level: Optional['Level'] = None,
    #                                    ) -> 'StatMechJob':
    #     """
    #     Initialize a StatMechJob object instance.
    #
    #     Args:
    #         arkane_species (arkane_input_species): An instance of an Arkane species() object.
    #         arkane_file_path (str): The path to the Arkane species file (either in .py or YAML form).
    #         bac_type (str, optional): The bond additivity correction type. 'p' for Petersson- or 'm' for Melius-type BAC.
    #                                   ``None`` to not use BAC.
    #         sp_level (Level, optional): The level of theory used for energy corrections.
    #
    #     Returns:
    #         StatMechJob: THe initialized StatMechJob object instance.
    #     """
    #     # statmech_job = StatMechJob(arkane_species, arkane_file_path)
    #     # statmech_job.applyBondEnergyCorrections = bac_type is not None and sp_level is not None
    #     # if bac_type is not None:
    #     #     statmech_job.bondEnergyCorrectionType = bac_type
    #     # if sp_level is None or not self.arkane_has_en_corr():
    #     #     # If this is a kinetics computation and we don't have a valid model chemistry, don't bother about it.
    #     #     statmech_job.applyAtomEnergyCorrections = False
    #     # else:
    #     #     statmech_job.level_of_theory = sp_level.to_arkane_level_of_theory()
    #     # statmech_job.frequencyScaleFactor = self.freq_scale_factor
    #     # return statmech_job

    # def generate_arkane_species_file(self,
    #                                  species: 'ARCSpecies',
    #                                  bac_type: Optional[str],
    #                                  skip_rotors: bool = False,
    #                                  ) -> Optional[str]:
    #     """
    #     A helper function for generating an Arkane Python species file.
    #     Assigns the path of the generated file to the species.arkane_file attribute.
    #
    #     Args:
    #         species (ARCSpecies): The species to process.
    #         bac_type (str): The bond additivity correction type. 'p' for Petersson- or 'm' for Melius-type BAC.
    #                         ``None`` to not use BAC.
    #         skip_rotors (bool, optional): Whether to skip internal rotor consideration. Default: ``False``.
    #
    #     Returns:
    #         str: The path to the species arkane folder (Arkane's default output folder).
    #     """
    #     folder_name = 'rxns' if species.is_ts else 'Species'
    #     species_folder_path = os.path.join(self.output_directory, folder_name, species.label)
    #     arkane_output_path = os.path.join(species_folder_path, 'arkane')
    #     if not os.path.isdir(arkane_output_path):
    #         os.makedirs(arkane_output_path)
    #
    #     if species.yml_path is not None:
    #         species.arkane_file = species.yml_path
    #         return arkane_output_path
    #
    #     species.determine_symmetry()
    #
    #     sp_path = self.output_dict[species.label]['paths']['composite'] \
    #         or self.output_dict[species.label]['paths']['sp']
    #     if species.number_of_atoms == 1:
    #         freq_path = sp_path
    #         opt_path = sp_path
    #     else:
    #         freq_path = self.output_dict[species.label]['paths']['freq']
    #         opt_path = self.output_dict[species.label]['paths']['freq']
    #
    #     return_none_text = None
    #     if not sp_path:
    #         return_none_text = 'path to the sp calculation'
    #     if not freq_path:
    #         return_none_text = 'path to the freq calculation'
    #     if not os.path.isfile(freq_path):
    #         return_none_text = f'the freq file in path {freq_path}'
    #     if not os.path.isfile(sp_path):
    #         return_none_text = f'the sp file in path {sp_path}'
    #     if return_none_text is not None:
    #         logger.error(f'Could not find {return_none_text} for species {species.label}. Not calculating properties.')
    #         return None
    #
    #     rotors, rotors_description = '', ''
    #     if species.rotors_dict is not None and any([i_r_dict['pivots'] for i_r_dict in species.rotors_dict.values()]) \
    #             and not skip_rotors:
    #         rotors = '\n\nrotors = ['
    #         rotors_description = '1D rotors:\n'
    #         for i in range(species.number_of_rotors):
    #             pivots = str(species.rotors_dict[i]['pivots'])
    #             scan = str(species.rotors_dict[i]['scan'])
    #             if species.rotors_dict[i]['success']:
    #                 rotor_path = species.rotors_dict[i]['scan_path']
    #                 rotor_type = determine_rotor_type(rotor_path)
    #                 top = str(species.rotors_dict[i]['top'])
    #                 try:
    #                     rotor_symmetry, max_e, _ = determine_rotor_symmetry(species.label, pivots, rotor_path)
    #                 except RotorError:
    #                     logger.error(f'Could not determine rotor symmetry for species {species.label} between '
    #                                  f'pivots {pivots}. Setting the rotor symmetry to 1, '
    #                                  f'this could very well be WRONG.')
    #                     rotor_symmetry = 1
    #                     max_e = None
    #                 scan_trsh = ''
    #                 if 'trsh_methods' in species.rotors_dict[i]:
    #                     scan_res = 360
    #                     for scan_trsh_method in species.rotors_dict[i]['trsh_methods']:
    #                         if 'scan_trsh' in scan_trsh_method and len(scan_trsh) < len(scan_trsh_method['scan_trsh']):
    #                             scan_trsh = scan_trsh_method['scan_trsh']
    #                         if 'scan_res' in scan_trsh_method and scan_res > scan_trsh_method['scan_res']:
    #                             scan_res = scan_trsh_method['scan_res']
    #                     scan_trsh = f'Troubleshot with the following constraints and {scan_res} degrees ' \
    #                                 f'resolution:\n{scan_trsh}' if scan_trsh else ''
    #                 max_e = f', max scan energy: {max_e:.2f} kJ/mol' if max_e is not None else ''
    #                 free = ' (set as a FreeRotor)' if rotor_type == 'FreeRotor' else ''
    #                 rotors_description += f'pivots: {pivots}, dihedral: {scan}, ' \
    #                                       f'rotor symmetry: {rotor_symmetry}{max_e}{free}\n{scan_trsh}'
    #                 if rotor_type == 'HinderedRotor':
    #                     rotors += input_files['arkane_hindered_rotor'].format(rotor_path=rotor_path,
    #                                                                           pivots=pivots,
    #                                                                           top=top,
    #                                                                           symmetry=rotor_symmetry)
    #                 elif rotor_type == 'FreeRotor':
    #                     rotors += input_files['arkane_free_rotor'].format(rotor_path=rotor_path,
    #                                                                       pivots=pivots,
    #                                                                       top=top,
    #                                                                       symmetry=rotor_symmetry)
    #                 if i < species.number_of_rotors - 1:
    #                     rotors += ',\n          '
    #             else:
    #                 rotors_description += f'* Invalidated! pivots: {pivots}, dihedral: {scan}, ' \
    #                                       f'invalidation reason: {species.rotors_dict[i]["invalidation_reason"]}\n'
    #
    #         rotors += ']'
    #         if 'rotors' not in species.long_thermo_description:
    #             species.long_thermo_description += rotors_description + '\n'
    #
    #     # Write the Arkane species input file.
    #     bac_txt = '' if bac_type is not None else '_no_BAC'
    #     input_file_path = os.path.join(species_folder_path, f'{species.label}_arkane_input{bac_txt}.py')
    #     input_file = input_files['arkane_input_species'] if 'sp_sol' not in self.output_dict[species.label]['paths'] \
    #         else input_files['arkane_input_species_explicit_e']
    #     if bac_type is not None and not species.is_ts:
    #         logger.info(f'Using the following BAC (type {bac_type}) for {species.label}: {species.bond_corrections}')
    #         bonds = f'bonds = {species.bond_corrections}\n\n'
    #     else:
    #         logger.debug(f'NOT using BAC for {species.label}')
    #         bonds = ''
    #
    #     if 'sp_sol' not in self.output_dict[species.label]['paths']:
    #         input_file = input_file.format(bonds=bonds,
    #                                        symmetry=species.external_symmetry,
    #                                        multiplicity=species.multiplicity,
    #                                        optical=species.optical_isomers,
    #                                        sp_path=sp_path,
    #                                        opt_path=opt_path,
    #                                        freq_path=freq_path,
    #                                        rotors=rotors)
    #     else:
    #         # e_elect = e_original + sp_e_sol_corrected - sp_e_uncorrected
    #         original_log = ess_factory(self.output_dict[species.label]['paths']['sp'], check_for_errors=False)
    #         e_original = original_log.load_energy()
    #         e_sol_log = ess_factory(self.output_dict[species.label]['paths']['sp_sol'], check_for_errors=False)
    #         e_sol = e_sol_log.load_energy()
    #         e_no_sol_log = ess_factory(self.output_dict[species.label]['paths']['sp_no_sol'], check_for_errors=False)
    #         e_no_sol = e_no_sol_log.load_energy()
    #         e_elect = (e_original + e_sol - e_no_sol) / (constants.E_h * constants.Na)  # Convert J/mol to Hartree.
    #         logger.info(f'\nSolvation correction scheme for {species.label}:\n'
    #                     f'Original electronic energy: {e_original * 0.001} kJ/mol\n'
    #                     f'Solvation correction: {(e_sol - e_no_sol) * 0.001} kJ/mol\n'
    #                     f'New electronic energy: {(e_original + e_sol - e_no_sol) * 0.001} kJ/mol\n\n')
    #         input_file = input_files['arkane_input_species_explicit_e']
    #         input_file = input_file.format(bonds=bonds,
    #                                        symmetry=species.external_symmetry,
    #                                        multiplicity=species.multiplicity,
    #                                        optical=species.optical_isomers,
    #                                        sp_level=self.sp_level,
    #                                        e_elect=e_elect,
    #                                        opt_path=opt_path,
    #                                        freq_path=freq_path,
    #                                        rotors=rotors)
    #
    #     if freq_path:
    #         with open(input_file_path, 'w') as f:
    #             f.write(input_file)
    #         species.arkane_file = input_file_path
    #     else:
    #         species.arkane_file = None
    #
    #     return arkane_output_path


def run_arkane(statmech_dir: str) -> None:
    """
    Execute an Arkane calculation within statmech_dir that contains an 'input.py' file.

    Args:
        statmech_dir (str): The path to the statmech directory containing the 'input.py' file.
    """
    if not os.path.isdir(statmech_dir):
        logger.error(f'Cannot run Arkane in {statmech_dir} because it does not exist.')
        return
    input_file = os.path.join(statmech_dir, 'input.py')
    if not os.path.isfile(input_file):
        logger.error(f'Cannot run Arkane in {statmech_dir} because it does not contain an input.py file.')
        return
    env_name = 'rmg_env'
    shell_script = f"""cd "{statmech_dir}"
if command -v micromamba &> /dev/null; then
    eval "$(micromamba shell hook --shell=bash)"
    micromamba activate {env_name}
elif command -v mamba &> /dev/null; then
    eval "$(mamba shell hook --shell=bash)"
    mamba activate {env_name}
elif command -v conda &> /dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate {env_name}
else
    exit 1
fi
Arkane input.py
"""
    std_out, std_err = execute_command(command=shell_script,
                                       shell=True,
                                       no_fail=True,
                                       executable='/bin/bash')
    if std_err:
        logger.error(f'Arkane run failed:\n{std_err}')
    else:
        logger.debug(f'Arkane run completed:\n{std_out}')


def get_atomic_energy_corrections(arc_species: 'ARCSpecies',
                                  sp_level: Optional['Level'] = None,
                                  ) -> float:
    """
    Get the atomic energy corrections for the species.

    Args:
        arc_species (ARCSpecies): The species object instance.
        sp_level (Level, optional): The level of theory used for energy corrections.

    Returns:
        float: The atomic energy correction.
    """
    corr = 0.0
    atoms = dict()
    for symbol in arc_species.get_xyz()['symbols']:
        atoms[symbol] = atoms.get(symbol, 0) + 1
    level_aec = read_yaml_file(os.path.join(ARC_PATH, 'data', 'AEC.yml'))
    if sp_level.method in level_aec.keys():
        atom_energies = level_aec[sp_level.method]
        for symbol, count in atoms.items():
            corr -= count * atom_energies.get(symbol, 0)
    return corr


def clean_output_directory(species_path: str,  # todo
                           is_ts: bool = False,
                           ) -> None:
    """
    Relocate Arkane files.

    Args:
        species_path (str): The path to the species folder.
        is_ts (bool, optional): Whether the species represents a TS, in which case reaction files will be generated.
    """
    # 1. The YAML file
    species_yaml_base_path = os.path.join(species_path, 'arkane', 'species')
    if os.path.exists(species_yaml_base_path):
        species_yaml_files = os.listdir(species_yaml_base_path)
        if species_yaml_files:
            for yml_file in species_yaml_files:
                if '.yml' in yml_file:
                    shutil.move(src=os.path.join(species_yaml_base_path, yml_file),
                                dst=os.path.join(species_path, yml_file))
        shutil.rmtree(species_yaml_base_path)

    if is_ts:
        # 2. reaction files
        # 2.1. the reaction path (PES)
        paths_path = os.path.join(species_path, 'arkane', 'paths')
        if os.path.exists(paths_path):
            path_diagram_files = os.listdir(paths_path)
            if path_diagram_files:
                path_file = path_diagram_files[0]
                target_file = os.path.join(os.path.dirname(path_file), f'reaction_path.{path_file.split(".")[-1]}')
                shutil.move(src=os.path.join(paths_path, path_file),
                            dst=os.path.join(species_path, target_file))
            shutil.rmtree(paths_path)

        # 2.2. the Arrhenius plot
        plot_path = os.path.join(species_path, 'arkane', 'plots')
        if os.path.exists(plot_path):
            plot_files = os.listdir(plot_path)
            if plot_files:
                for plot_file in plot_files:
                    if '<=>' in plot_file:
                        target_file = os.path.join(os.path.dirname(plot_file), f'Arrhenius_plot.{plot_file.split(".")[-1]}')
                        shutil.move(src=os.path.join(plot_path, plot_file),
                                    dst=os.path.join(species_path, target_file))
                        if len(plot_files) == 1:
                            shutil.rmtree(plot_path)

    # 3. thermo plots
    plot_path = os.path.join(species_path, 'arkane', 'plots')
    if os.path.exists(plot_path):
        plot_files = os.listdir(plot_path)
        if plot_files:
            plot_file = plot_files[0]
            target_file = os.path.join(os.path.dirname(plot_file), f'thermo_properties_plot.{plot_file.split(".")[-1]}')
            shutil.move(src=os.path.join(plot_path, plot_file),
                        dst=os.path.join(species_path, target_file))
        shutil.rmtree(plot_path)


def create_statmech_dir(calcs_directory: str,
                        subdir: Optional[str] = None,
                        delete_existing_subdir: bool = True,
                        ) -> str:
    """
    Create the statmech directory in the calcs directory.

    Args:
        calcs_directory (str): The path to the ARC project calculations directory.
        subdir (str, optional): A subdirectory to create within the statmech directory. Default: ``None``.
        delete_existing_subdir (bool, optional): Whether to delete the existing subdirectory if it exists. Default: ``True``.

    Returns:
        str: The path to the statmech directory.
    """
    statmech_dir = os.path.join(calcs_directory, 'statmech')
    if subdir:
        statmech_dir = os.path.join(statmech_dir, subdir)
    if delete_existing_subdir and os.path.isdir(statmech_dir):
        shutil.rmtree(statmech_dir)
    if not os.path.isdir(statmech_dir):
        os.makedirs(statmech_dir, exist_ok=True)
    return statmech_dir







def _extract_section(file_path: str, section_start: str, section_end: str) -> Optional[str]:
    """
    Extract a section from a file between section_start and section_end.

    Args:
        file_path (str): Path to the file to read.
        section_start (str): String marking the start of the section.
        section_end (str): String marking the end of the section.

    Returns:
        Optional[str]: Extracted section as string, or None if not found.
    """
    if not os.path.isfile(file_path):
        return None
    with open(file_path, 'r') as f:
        text = f.read()
    start_idx = text.find(section_start)
    if start_idx == -1:
        return None
    end_idx = text.find(section_end, start_idx + len(section_start))
    if end_idx == -1:
        return None
    return text[start_idx:end_idx + len(section_end)]


def _section_contains_key(file_path: str, section_start: str, section_end: str, target: str) -> bool:
    """
    Check if target string appears in section with flexible attribute handling.

    Args:
        file_path (str): Path to the file.
        section_start (str): Section start marker.
        section_end (str): Section end marker.
        target (str): String to search for.

    Returns:
        bool: True if target found, False otherwise.
    """
    section = _extract_section(file_path, section_start, section_end)
    if section is None:
        return False
    if target in section:  # Check for exact match first
        return True
    if 'software=' in target:  # Check for partial match without software
        no_software = re.sub(r",\s*software='[^']*'", '', target)
        if no_software in section:
            return True
    return False


def _level_to_str(level: 'Level') -> str:
    """
    Convert Level to Arkane's LevelOfTheory string representation.

    Args:
        level (Level): Level object to convert.

    Returns:
        str: LevelOfTheory string representation.
    """
    parts = [f"method='{level.method}'"]
    if level.basis:
        parts.append(f"basis='{level.basis}'")
    if level.software:
        parts.append(f"software='{level.software}'")
    return f"LevelOfTheory({', '.join(parts)})"


def get_arkane_model_chemistry(sp_level: 'Level',
                               freq_level: Optional['Level'] = None,
                               freq_scale_factor: Optional[float] = None,
                               ) -> Optional[str]:
    """
    Get Arkane model chemistry string with database validation.

    Args:
        sp_level (Level): Level of theory for energy.
        freq_level (Optional[Level]): Level of theory for frequencies.
        freq_scale_factor (Optional[float]): Frequency scaling factor.

    Returns:
        Optional[str]: Arkane-compatible model chemistry string.
    """
    if sp_level.method_type == 'composite':
        return sp_level.method

    qm_corr_file = os.path.join(RMG_DB_PATH, 'input', 'quantum_corrections', 'data.py')

    atom_energies_start = "atom_energies = {"
    atom_energies_end = "}"
    freq_dict_start = "freq_dict = {"
    freq_dict_end = "}"

    sp_repr = _level_to_str(sp_level)
    quoted_sp_repr = f'"{sp_repr}"'

    if freq_scale_factor is not None:
        found = _section_contains_key(file_path=qm_corr_file,
                                      section_start=atom_energies_start,
                                      section_end=atom_energies_end,
                                      target=quoted_sp_repr)
        if not found:
            return None
        return sp_repr

    if freq_level is None:
        raise ValueError("freq_level required when freq_scale_factor isn't provided")

    freq_repr = _level_to_str(freq_level)
    quoted_freq_repr = f'"{freq_repr}"'

    found_sp = _section_contains_key(file_path=qm_corr_file,
                                     section_start=atom_energies_start,
                                     section_end=atom_energies_end,
                                     target=quoted_sp_repr)
    found_freq = _section_contains_key(file_path=qm_corr_file,
                                       section_start=freq_dict_start,
                                       section_end=freq_dict_end,
                                       target=quoted_freq_repr)

    if not found_sp or not found_freq:
        return None

    return (f"CompositeLevelOfTheory(\n"
            f"    freq={freq_repr},\n"
            f"    energy={sp_repr}\n"
            f")")


def parse_species_thermo(species, output_content: str) -> None:
    """Parse thermodynamic data for a single species."""
    # Parse E0
    e0 = parse_e0(species.label, output_content)
    if e0 is not None:
        species.e0 = e0

    # Parse thermo data
    thermo_match = re.search(
        rf"thermo\(\s*label\s*=\s*['\"]{re.escape(species.label)}['\"].*?thermo\s*=\s*ThermoData\((.*?)\)\s*\)",
        output_content,
        re.DOTALL
    )

    if thermo_match:
        thermo_block = thermo_match.group(1)
        species.thermo.update(parse_thermo_block(thermo_block))


def parse_reaction_kinetics(reaction, output_content: str) -> None:
    """
    Parse Arrhenius kinetics data for a single reaction from Arkane output.

    Args:
        reaction: The reaction object (must have a .label attribute).
        output_content (str): The full Arkane output file content.

    Populates:
        reaction.kinetics (dict): A dictionary with Arrhenius parameters and units.
    """
    e0 = parse_e0(reaction.ts_species.label, output_content)
    if e0 is not None:
        reaction.ts_species.e0 = e0
    kinetics_pattern = (rf"kinetics\(\s*label\s*=\s*['\"]{re.escape(reaction.label)}['\"].*?kinetics\s*=\s*Arrhenius\((.*?)\)\s*\)")
    kinetics_match = re.search(kinetics_pattern, output_content, re.DOTALL)
    if not kinetics_match:
        logger.warning(f"Kinetics block not found in the Arkane output file for reaction: {reaction.label}")
        return
    arrhenius_block = kinetics_match.group(1)
    arrhenius_data = dict()
    patterns = {
        'A': r"A\s*=\s*\(\s*([-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?),\s*['\"]([^'\"]+)['\"]\)",
        'n': r"n\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        'Ea': r"Ea\s*=\s*\(([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?),\s*['\"]([^'\"]+)['\"]\)",
        'T0': r"T0\s*=\s*\(([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?),\s*['\"]([^'\"]+)['\"]\)",
        'Tmin': r"Tmin\s*=\s*\(([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?),\s*['\"]([^'\"]+)['\"]\)",
        'Tmax': r"Tmax\s*=\s*\(([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?),\s*['\"]([^'\"]+)['\"]\)"}
    for key, pattern in patterns.items():
        match = re.search(pattern, arrhenius_block)
        if match:
            if key == 'n':
                arrhenius_data[key] = match.group(1)
            else:
                value, unit = match.groups()
                arrhenius_data[key] = (value, unit)
    reaction.kinetics = arrhenius_data


def parse_e0(label: str, content: str) -> float | None:
    """Parse E0 value for a species."""
    pattern = rf"conformer\(\s*label\s*=\s*['\"]{re.escape(label)}['\"].*?E0\s*=\s*\(([^)]*)\)"
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        return None

    e0_block = match.group(1)
    value_match = re.search(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?).*?['\"]kJ/mol['\"]", e0_block, re.DOTALL)
    return float(value_match.group(1)) if value_match else None


def parse_thermo_block(block: str) -> dict:
    """Parse thermo data block into dictionary, including full ThermoData."""
    thermo_data = {}

    # Parse H298 and S298
    h298_match = re.search(r"H298\s*=\s*\(\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?).*?['\"]kJ/mol['\"]", block,
                           re.DOTALL)
    s298_match = re.search(r"S298\s*=\s*\(\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?).*?['\"]J/\(mol\*K\)['\"]", block,
                           re.DOTALL)

    if h298_match:
        thermo_data['H298'] = float(h298_match.group(1))
    if s298_match:
        thermo_data['S298'] = float(s298_match.group(1))

    # Parse full thermo data
    thermo_data['thermo_data'] = parse_thermo_data_block(block)
    return thermo_data


def parse_thermo_data_block(block: str) -> dict:
    """
    Parse a ThermoData block from Arkane output.

    Args:
        block (str): The ThermoData block content of the species.

    Returns:
        dict: Parsed ThermoData parameters.
    """
    thermo_data = {}
    patterns = {'Tdata': r"Tdata\s*=\s*\((\[.*?\]),\s*['\"]([^'\"]+)['\"]\)",
                'Cpdata': r"Cpdata\s*=\s*\((\[.*?\]),\s*['\"]([^'\"]+)['\"]\)",
                'H298': r"H298\s*=\s*\(([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?),\s*['\"]([^'\"]+)['\"]\)",
                'S298': r"S298\s*=\s*\(([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?),\s*['\"]([^'\"]+)['\"]\)",
                'Tmin': r"Tmin\s*=\s*\(([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?),\s*['\"]([^'\"]+)['\"]\)",
                'Tmax': r"Tmax\s*=\s*\(([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?),\s*['\"]([^'\"]+)['\"]\)",
                'Cp0': r"Cp0\s*=\s*\(([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?),\s*['\"]([^'\"]+)['\"]\)",
                'CpInf': r"CpInf\s*=\s*\(([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?),\s*['\"]([^'\"]+)['\"]\)"}
    for key, pattern in patterns.items():
        match = re.search(pattern, block, re.DOTALL)
        if match:
            value_str, unit = match.groups()
            if key in ['Tdata', 'Cpdata']:  # Handle list values
                try:
                    value = eval(value_str, {'__builtins__': None}, {})
                except ValueError:
                    value = value_str
            else:  # Handle scalar values
                value = value_str
            thermo_data[key] = value
    return thermo_data


register_statmech_adapter('arkane', ArkaneAdapter)
