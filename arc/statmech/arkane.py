"""
An adapter for executing Arkane.
"""

import os
import re
import shutil
from abc import ABC
from typing import TYPE_CHECKING, List, Optional, Tuple

from mako.template import Template

import arc.plotter as plotter
from arc.common import ARC_PATH, get_logger, read_yaml_file
from arc.exceptions import InputError
from arc.imports import incore_commands, settings
from arc.job.local import execute_command
from arc.statmech.adapter import StatmechAdapter
from arc.statmech.factory import register_statmech_adapter

if TYPE_CHECKING:
    from arc.level import Level
    from arc.reaction import ARCReaction
    from arc.species.species import ARCSpecies


RMG_DB_PATH = settings['RMG_DB_PATH']
RMG_ENV_NAME = settings.get('RMG_ENV_NAME', 'rmg_env')
logger = get_logger()


main_input_template = """#!/usr/bin/env python
# -*- coding: utf-8 -*-

title = '${title}'
description = \"\"\"
${description}
\"\"\"

% if model_chemistry:
modelChemistry = ${model_chemistry}
% endif
${atom_energies}${freq_scale_factor}
useHinderedRotors = ${use_hindered_rotors}
useAtomCorrections = ${use_aec}
useBondCorrections = ${use_bac}
% if bool(use_bac):
bondCorrectionType = '${bac_type}'
% endif

% for spc in species_list:
% if spc['smiles']:
species('${spc['label']}', '${spc['path']}'${spc['pdep_data'] if 'pdep_data' in spc else ''},
        structure=SMILES('${spc['smiles']}'), spinMultiplicity=${spc['multiplicity']})
% else:
species('${spc['label']}', '${spc['path']}'${spc['pdep_data'] if 'pdep_data' in spc else ''},
        spinMultiplicity=${spc['multiplicity']})
% endif
% endfor

% for ts in ts_list:
transitionState('${ts['label']}', '${ts['path']}')
% endfor

% for rxn in reaction_list:
reaction(
    label='${rxn.label}',
    reactants=${rxn.reactants},
    products=${rxn.products},
    transitionState='${rxn.ts_species.label}',
    tunneling='Eckart',
)
% endfor

% if len(reaction_list):
% for rxn in reaction_list:
kinetics(label='${rxn.label}',
         Tmin=${t_min or (300, 'K')}, Tmax=${t_max or (3000, 'K')}, Tcount=${t_count or 25})
% endfor
% endif

% if compute_thermo:
% for spc in species_list:
thermo('${spc['label']}', 'NASA')
% endfor
% endif

"""

species_input_template = """#!/usr/bin/env python
# -*- coding: utf-8 -*-

linear = ${linear}
spinMultiplicity = ${spin_multiplicity}

energy = Log('${sp_path}')
geometry = Log('${freq_path}')
frequencies = Log('${freq_path}')

%if use_hindered_rotors:
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
        bac_type (Optional[str]): The bond additivity correction type. 'p' for Petersson- or 'm' for Melius-type BAC.
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
                 bac_type: Optional[str] = None,
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
        self.species = species if isinstance(species, list) else [species]
        self.reactions = reactions if isinstance(reactions, list) else [reactions] if reactions is not None else None
        self.sp_level = sp_level
        self.freq_level = freq_level
        self.freq_scale_factor = freq_scale_factor
        self.skip_nmd = skip_nmd
        self.species_dict = species_dict or {spc.label: spc for spc in self.species}
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
        self.generate_arkane_input(statmech_dir=statmech_dir, skip_rotors=skip_rotors, e0_only=e0_only)
        self.generate_species_files(statmech_dir, skip_rotors, check_compute_thermo=not e0_only)
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

    def generate_arkane_input(self,
                              statmech_dir: str,
                              skip_rotors: bool = False,
                              e0_only: bool = False,
                              ) -> None:
        """
        Generate the Arkane main input file.

        Args:
            statmech_dir (str): The path to the statmech directory.
            skip_rotors (bool, optional): Whether to skip internal rotor consideration. Default: ``False``.
        """
        input_path = os.path.join(statmech_dir, 'input.py')
        input_content = self.render_arkane_input_template(statmech_dir=statmech_dir,
                                                          skip_rotors=skip_rotors,
                                                          e0_only=e0_only)
        with open(input_path, 'w', encoding='utf-8') as f:
            f.write(input_content)

    def render_arkane_input_template(self,
                                     statmech_dir: str,
                                     skip_rotors: bool = False,
                                     e0_only: bool = False,
                                     ) -> str:
        """
        Render the Arkane main input template.

        Args:
            statmech_dir (str): The path to the statmech directory.
            skip_rotors (bool, optional): Whether to skip internal rotor consideration. Default: ``False``.
            e0_only (bool, optional): Whether to only run statmech (w/o thermo) to compute E0. Default: ``False``.
        """
        species_list = list()
        for spc in self.species:
            if e0_only or spc.compute_thermo:
                species_list.append({'label': spc.label,
                                     'path': spc.yml_path or os.path.join(statmech_dir, 'species', f'{spc.label}.py'),
                                     'smiles': spc.mol.copy(deep=True).to_smiles() if not spc.is_ts else '',
                                     'multiplicity': spc.multiplicity})
        ts_list = [{'label': rxn.ts_species.label,
                     'path': rxn.ts_species.yml_path or os.path.join(statmech_dir, 'TSs', f'{rxn.ts_species.label}.py')}
                     for rxn in self.reactions] if self.reactions else list()

        model_chemistry = get_arkane_model_chemistry(sp_level=self.sp_level,
                                                     freq_level=self.freq_level,
                                                     freq_scale_factor=self.freq_scale_factor
                                                     ) or ''

        aec_dict = read_yaml_file(os.path.join(ARC_PATH, 'data', 'AEC.yml'))
        atom_energies = f'\natomEnergies = {aec_dict[self.sp_level.simple()]}' \
            if self.sp_level.simple() in aec_dict else ''

        freq_scale_factor = f'\nfrequencyScaleFactor = {self.freq_scale_factor}' \
            if self.freq_scale_factor is not None else ''
        if self.reactions is not None:
            for rxn in self.reactions:
                reactants, products = rxn.get_reactants_and_products()
                rxn.reactants = [spc.label for spc in reactants]
                rxn.products = [spc.label for spc in products]
        calc_type = 'kinetics' if self.reactions else 'thermo'
        return Template(main_input_template).render(
            title=f'Arkane {calc_type} calculation',
            description='Generated by ARC.',
            model_chemistry=model_chemistry,
            atom_energies=atom_energies,
            freq_scale_factor=freq_scale_factor,
            use_hindered_rotors=True if not skip_rotors else False,
            use_aec=True,
            use_bac=True if self.bac_type is not None else False,
            bac_type=self.bac_type,
            species_list=species_list,
            ts_list=ts_list,
            reaction_list=self.reactions or list(),
            compute_thermo=calc_type == 'thermo' and not e0_only,
            t_min=self.T_min,
            t_max=self.T_max,
            t_count=self.T_count,
        )

    def generate_species_files(self,
                               statmech_dir: str,
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
            if spc.yml_path or check_compute_thermo and not spc.compute_thermo:
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
            if ts.yml_path:
                continue
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
        rotors = [rotor for rotor in species.rotors_dict.values() if rotor['success']] if species.rotors_dict else list()
        use_rotors = not skip_rotors and bool(rotors)
        content = Template(species_input_template).render(
            linear=species.is_linear,
            spin_multiplicity=species.multiplicity,
            sp_path=self.output_dict[species.label]['paths']['composite'] or self.output_dict[species.label]['paths']['sp'],
            freq_path=self.output_dict[species.label]['paths']['freq'] or self.output_dict[species.label]['paths']['sp'],
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
            parse_species_thermo(species, output_content)
            clean_output_directory(os.path.join(self.output_directory, 'Species', species.label))

        script_path = os.path.join(ARC_PATH, 'arc', 'scripts', 'save_arkane_thermo.py')
        rmg_db_path = RMG_DB_PATH or ""
        commands = [f'cd {statmech_dir}',
                    'bash -lc "set -euo pipefail; '
                    f'export RMG_DB_PATH=\\"{rmg_db_path}\\"; '
                    f'export RMG_DATABASE=\\"{rmg_db_path}\\"; '
                    'if command -v micromamba >/dev/null 2>&1; then '
                    f'    micromamba run -n {RMG_ENV_NAME} python {script_path}; '
                    'elif command -v conda >/dev/null 2>&1 || command -v mamba >/dev/null 2>&1; then '
                    f'    conda run -n {RMG_ENV_NAME} python {script_path}; '
                    'else '
                    '    echo \'❌ Micromamba/Mamba/Conda required\' >&2; exit 1; '
                    'fi"',
                    ]
        stdout, stderr = execute_command(command=commands, executable='/bin/bash')
        if len(stderr):
            logger.error(f'Error while running Arkane thermo script:\n{stderr}')
        thermo_yaml_path = os.path.join(statmech_dir, 'thermo.yaml')
        if os.path.isfile(thermo_yaml_path):
            content = read_yaml_file(thermo_yaml_path) or {}
            valid_labels = [lbl for lbl in content if lbl in self.species_dict]
            if valid_labels:
                h_header = "H298 (kJ/mol)"
                s_header = "S298 (J/mol/K)"
                label_width = max(len(lbl) for lbl in valid_labels + ["Label"])
                h_width     = max(len(h_header),
                                  max(len(f"{content[lbl]['H298']:.2f}") for lbl in valid_labels))
                s_width     = max(len(s_header),
                                  max(len(f"{content[lbl]['S298']:.2f}") for lbl in valid_labels))
                logger.info("\nComputed thermo data:")
                header = f"{'   Label':<{label_width}}     {h_header:>{h_width}}  {s_header:>{s_width}}"
                logger.info(header)
                for lbl in valid_labels:
                    spc = self.species_dict[lbl]
                    spc.thermo.H298 = content[lbl]['H298']
                    spc.thermo.S298 = content[lbl]['S298']
                    spc.thermo.data = content[lbl]['data']

                    line = (
                        f"   {lbl:<{label_width}}  "
                        f"{spc.thermo.H298:>{h_width}.2f}  "
                        f"{spc.thermo.S298:>{s_width}.2f}"
                    )
                    logger.info(line)

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
    rmg_db_path = RMG_DB_PATH or ""
    arkane_cmd = 'python -m arkane input.py'
    arkane_cmd += ' 2> >(tee -a stderr.log >&2) | tee -a stdout.log'
    shell_script = rf'''bash -lc 'set -euo pipefail
cd "{statmech_dir}"
export RMG_DB_PATH="{rmg_db_path}"
export RMG_DATABASE="{rmg_db_path}"

# Limit BLAS thread counts so OpenBLAS/MKL cannot explode into 48 threads
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MKL_DYNAMIC=FALSE

if command -v micromamba >/dev/null 2>&1; then
    micromamba run -n {RMG_ENV_NAME} {arkane_cmd}
elif command -v conda >/dev/null 2>&1 || command -v mamba >/dev/null 2>&1; then
    conda run -n {RMG_ENV_NAME} {arkane_cmd}
else
    echo "❌ Micromamba/Mamba/Conda required" >&2
    exit 1
fi' '''

    std_out, std_err = execute_command(command=shell_script,
                                       shell=True,
                                       no_fail=True,
                                       executable='/bin/bash')
    if std_err:
        logger.debug(f'Arkane run failed:\n{std_err}')
    else:
        logger.debug(f'Arkane run completed:\n{std_out}')


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
    Check if the target string appears in a section with flexible attribute handling.

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


def _normalize_method(method: str) -> str:
    """
    Normalize method names for comparison:
      - lowercase
      - remove all hyphens

    Examples:
        "DLPNO-CCSD(T)-F12"    -> "dlpnoccsd(t)f12"
        "dlpnoccsd(t)f122023"  -> "dlpnoccsd(t)f122023"
    """
    return method.lower().replace('-', '')


def _split_method_year(method_norm: str) -> "Tuple[str, Optional[int]]":
    """
    Split a normalized method into (base_method, year).

    Examples:
        "dlpnoccsd(t)f122023" -> ("dlpnoccsd(t)f12", 2023)
        "dlpnoccsd(t)f12"     -> ("dlpnoccsd(t)f12", None)
    """
    m = re.match(r"^(.*?)(\d{4})$", method_norm)
    if not m:
        return method_norm, None
    base, year_str = m.groups()
    return base, int(year_str)


def _normalize_basis(basis: Optional[str]) -> Optional[str]:
    """
    Normalize basis names for comparison:
      - lowercase
      - remove hyphens and spaces

    Examples:
        "cc-pVTZ-F12" -> "ccpvtzf12"
        "ccpvtzf12"   -> "ccpvtzf12"
    """
    if basis is None:
        return None
    return basis.replace('-', '').replace(' ', '').lower()


def _parse_lot_params(lot_str: str) -> dict:
    """
    Parse method, basis, and software from a LevelOfTheory(...) string.

    Example lot_str:
        "LevelOfTheory(method='dlpnoccsd(t)f122023',basis='ccpvtzf12',software='orca')"
    """
    params = {'method': None, 'basis': None, 'software': None}
    for key in params.keys():
        m = re.search(rf"{key}='([^']+)'", lot_str)
        if m:
            params[key] = m.group(1)
    return params


def _iter_level_keys_from_section(file_path: str,
                                  section_start: str,
                                  section_end: str) -> list[str]:
    """
    Return all LevelOfTheory(...) key strings that appear as dictionary keys
    in a given section of data.py.

    These look like:
        "LevelOfTheory(method='...',basis='...',software='...')" : { ... }
    """
    section = _extract_section(file_path, section_start, section_end)
    if section is None:
        return []

    # Match things like: "LevelOfTheory(...)" : { ... }
    pattern = r'"(LevelOfTheory\([^"]*\))"\s*:'
    return re.findall(pattern, section, flags=re.DOTALL)


def _available_years_for_level(level: "Level",
                               file_path: str,
                               section_start: str,
                               section_end: str) -> list[Optional[int]]:
    """
    Return a sorted list of available year suffixes for a given Level in a section.
    """
    if level is None or level.method is None:
        return []

    target_method_norm = _normalize_method(level.method)
    target_base, _ = _split_method_year(target_method_norm)
    target_basis_norm = _normalize_basis(level.basis)
    target_software = level.software.lower() if level.software else None

    years = set()
    for lot_str in _iter_level_keys_from_section(file_path, section_start, section_end):
        params = _parse_lot_params(lot_str)
        cand_method = params.get('method')
        cand_basis = params.get('basis')
        cand_sw = params.get('software')

        if cand_method is None:
            continue

        cand_method_norm = _normalize_method(cand_method)
        cand_base, cand_year = _split_method_year(cand_method_norm)

        if cand_base != target_base:
            continue
        if target_basis_norm is not None:
            cand_basis_norm = _normalize_basis(cand_basis)
            if cand_basis_norm != target_basis_norm:
                continue
        if target_software is not None and cand_sw is not None:
            if cand_sw.lower() != target_software:
                continue

        years.add(cand_year)

    # Sort with None first to represent "no year suffix"
    return sorted(years, key=lambda y: (-1 if y is None else y))


def _format_years(years: list[Optional[int]]) -> str:
    """
    Format a list of years for logging.
    """
    if not years:
        return "none"
    return ", ".join("none" if y is None else str(y) for y in years)


def _find_best_level_key_for_sp_level(level: "Level",
                                      file_path: str,
                                      section_start: str,
                                      section_end: str) -> Optional[str]:
    """
    Given an ARC Level and a data.py section, find the LevelOfTheory(...) key string
    that best matches the level's method/basis, allowing:
      - hyphen-insensitive comparison
      - an optional 4-digit year suffix in Arkane's method
    and choose the *no-year* entry when no year is specified.
    """
    if level is None or level.method is None:
        return None

    target_method_norm = _normalize_method(level.method)
    target_base, method_year = _split_method_year(target_method_norm)
    target_year = getattr(level, 'year', None) if getattr(level, 'year', None) is not None else method_year
    target_basis_norm = _normalize_basis(level.basis)
    target_software = level.software.lower() if level.software else None

    best_key = None

    for lot_str in _iter_level_keys_from_section(file_path, section_start, section_end):
        params = _parse_lot_params(lot_str)
        cand_method = params.get('method')
        cand_basis = params.get('basis')
        cand_sw = params.get('software')

        if cand_method is None:
            continue

        cand_method_norm = _normalize_method(cand_method)
        cand_base, cand_year = _split_method_year(cand_method_norm)

        # method base must match
        if cand_base != target_base:
            continue

        # basis must match (normalized), if we have one
        if target_basis_norm is not None:
            cand_basis_norm = _normalize_basis(cand_basis)
            if cand_basis_norm != target_basis_norm:
                continue

        # if user specified software, prefer matching software;
        # but don't *require* it to exist in data.py
        if target_software is not None and cand_sw is not None:
            if cand_sw.lower() != target_software:
                continue

        if target_year is not None:
            if cand_year != target_year:
                continue
            best_key = lot_str
            break
        else:
            if cand_year is None:
                best_key = lot_str
                break

    return best_key


def _level_to_str(level: 'Level') -> str:
    """
    Convert Level to Arkane's LevelOfTheory string representation.

    Args:
        level (Level): Level object to convert.

    Returns:
        str: LevelOfTheory string representation.
    """
    method = _normalize_method(level.method)
    if getattr(level, 'year', None) is not None and not method.endswith(str(level.year)):
        method = f"{method}{level.year}"

    parts = [f"method='{method}'"]
    if level.basis:
        parts.append(f"basis='{_normalize_basis(level.basis)}'")
    if level.software:
        parts.append(f"software='{level.software.lower()}'")
    return f"LevelOfTheory({','.join(parts)})".replace('-', '')


def get_arkane_model_chemistry(sp_level: 'Level',
                               freq_level: Optional['Level'] = None,
                               freq_scale_factor: Optional[float] = None,
                               ) -> Optional[str]:
    """
    Get Arkane model chemistry string with database validation.

    Reads RMG's quantum_corrections/data.py as plain text, searches for
    LevelOfTheory(...) keys, and matches:
      - method:   ignoring hyphens and optional 4-digit year suffix
      - basis:    ignoring hyphens and spaces

    If multiple entries only differ by year, the one with the *latest* year
    is chosen (year=0 if no year in that entry).

    Args:
        sp_level (Level): Level of theory for energy.
        freq_level (Optional[Level]): Level of theory for frequencies.
        freq_scale_factor (Optional[float]): Frequency scaling factor.

    Returns:
        Optional[str]: Arkane-compatible model chemistry string.
    """
    qm_corr_file = os.path.join(RMG_DB_PATH, 'input', 'quantum_corrections', 'data.py')
    if not os.path.isfile(qm_corr_file):
        qm_corr_file = os.path.join(RMG_DB_PATH, 'quantum_corrections', 'data.py')

    atom_energies_start = "atom_energies = {"
    atom_energies_end = "pbac = {"
    freq_dict_start = "freq_dict = {"
    freq_dict_end = "}"

    if sp_level.method_type == 'composite':
        # Composite Gaussian methods: prefer best year match from DB, fall back to normalized LevelOfTheory.
        best_energy = _find_best_level_key_for_sp_level(
            sp_level, qm_corr_file, atom_energies_start, atom_energies_end
        )
        if best_energy is None:
            years = _available_years_for_level(sp_level, qm_corr_file, atom_energies_start, atom_energies_end)
            if getattr(sp_level, 'year', None) is not None:
                logger.warning(
                    f"No Arkane AEC entry found for year {sp_level.year} at {sp_level.simple()}; "
                    f"available years: {_format_years(years)}"
                )
            elif years:
                logger.warning(
                    f"No Arkane AEC entry found for {sp_level.simple()} without a year; "
                    f"available years: {_format_years(years)}. "
                    f"Specify a year to select a matching entry."
                )
            return _level_to_str(sp_level)
        return best_energy

    # ---- Case 1: User supplied explicit frequency scale factor ----
    # We only need an energy level (AEC entry in atom_energies)
    if freq_scale_factor is not None:
        best_energy = _find_best_level_key_for_sp_level(
            sp_level, qm_corr_file, atom_energies_start, atom_energies_end
        )
        if best_energy is None:
            years = _available_years_for_level(sp_level, qm_corr_file, atom_energies_start, atom_energies_end)
            if getattr(sp_level, 'year', None) is not None:
                logger.warning(
                    f"No Arkane AEC entry found for year {sp_level.year} at {sp_level.simple()}; "
                    f"available years: {_format_years(years)}"
                )
            elif years:
                logger.warning(
                    f"No Arkane AEC entry found for {sp_level.simple()} without a year; "
                    f"available years: {_format_years(years)}. "
                    f"Specify a year to select a matching entry."
                )
            # No matching AEC level in Arkane DB
            return None
        # modelChemistry = LevelOfTheory(...)
        return best_energy

    # ---- Case 2: CompositeLevelOfTheory (separate freq and energy levels) ----
    if freq_level is None:
        raise ValueError("freq_level required when freq_scale_factor isn't provided")

    best_energy = _find_best_level_key_for_sp_level(
        sp_level, qm_corr_file, atom_energies_start, atom_energies_end
    )
    best_freq = _find_best_level_key_for_sp_level(
        freq_level, qm_corr_file, freq_dict_start, freq_dict_end
    )

    if best_energy is None or best_freq is None:
        if best_energy is None:
            years = _available_years_for_level(sp_level, qm_corr_file, atom_energies_start, atom_energies_end)
            if getattr(sp_level, 'year', None) is not None:
                logger.warning(
                    f"No Arkane AEC entry found for year {sp_level.year} at {sp_level.simple()}; "
                    f"available years: {_format_years(years)}"
                )
            elif years:
                logger.warning(
                    f"No Arkane AEC entry found for {sp_level.simple()} without a year; "
                    f"available years: {_format_years(years)}. "
                    f"Specify a year to select a matching entry."
                )
        if best_freq is None:
            years = _available_years_for_level(freq_level, qm_corr_file, freq_dict_start, freq_dict_end)
            if getattr(freq_level, 'year', None) is not None:
                logger.warning(
                    f"No Arkane frequency correction entry found for year {freq_level.year} at {freq_level.simple()}; "
                    f"available years: {_format_years(years)}"
                )
            elif years:
                logger.warning(
                    f"No Arkane frequency correction entry found for {freq_level.simple()} without a year; "
                    f"available years: {_format_years(years)}. "
                    f"Specify a year to select a matching entry."
                )
        # If either is missing, cannot construct a valid composite model chemistry
        return None

    # These strings are LevelOfTheory(...) expressions usable directly in Arkane input
    return (
        "CompositeLevelOfTheory(\n"
        f"    freq={best_freq},\n"
        f"    energy={best_energy}\n"
        ")"
    )


def check_arkane_bacs(sp_level: 'Level',
                      bac_type: str = 'p',
                      raise_error: bool = False,
                      ) -> bool:
    """
    Check that Arkane has AECs and BACs for the given sp level of theory.

    Uses plain-text parsing of quantum_corrections/data.py, matching LevelOfTheory
    keys by:
      - method base (ignore hyphens + optional year)
      - basis (normalized)
    and picking the latest year where multiple exist.
    """
    qm_corr_file = os.path.join(RMG_DB_PATH, 'input', 'quantum_corrections', 'data.py')
    if not os.path.isfile(qm_corr_file):
        qm_corr_file = os.path.join(RMG_DB_PATH, 'quantum_corrections', 'data.py')

    atom_energies_start = "atom_energies = {"
    atom_energies_end = "pbac = {"
    if bac_type.lower() == 'm':
        bac_section_start = "mbac = {"
        bac_section_end = "freq_dict ="
    else:
        bac_section_start = "pbac = {"
        bac_section_end = "mbac = {"

    best_aec_key = _find_best_level_key_for_sp_level(
        sp_level, qm_corr_file, atom_energies_start, atom_energies_end
    )
    best_bac_key = _find_best_level_key_for_sp_level(
        sp_level, qm_corr_file, bac_section_start, bac_section_end
    )

    has_aec = best_aec_key is not None
    has_bac = best_bac_key is not None
    has_encorr = bool(has_aec and has_bac)

    # For logging, prefer the matched key; fall back to the naive LevelOfTheory string
    repr_level = best_aec_key if best_aec_key is not None else _level_to_str(sp_level)

    if not has_encorr:
        year_note = ""
        aec_years = _available_years_for_level(sp_level, qm_corr_file, atom_energies_start, atom_energies_end)
        bac_years = _available_years_for_level(sp_level, qm_corr_file, bac_section_start, bac_section_end)
        if getattr(sp_level, 'year', None) is not None:
            year_note = (
                f" Available AEC years: {_format_years(aec_years)}; "
                f"available BAC years: {_format_years(bac_years)}."
            )
        elif aec_years or bac_years:
            year_note = (
                f" Available AEC years: {_format_years(aec_years)}; "
                f"available BAC years: {_format_years(bac_years)}. "
                f"Specify a year to select a matching entry."
            )
        mssg = (
            f"Arkane does not have the required energy corrections for {repr_level} "
            f"(AEC: {has_aec}, BAC: {has_bac}).{year_note}"
        )
        if raise_error:
            raise ValueError(mssg)
        else:
            logger.warning(mssg)
    return has_encorr



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
    label_pat = rf"kinetics\(\s*label\s*=\s*['\"]{re.escape(reaction.label)}['\"],"
    m_label = re.search(label_pat, output_content)
    if not m_label:
        logger.warning(f"Kinetics block not found for reaction: {reaction.label}")
        return

    arr_start = output_content.find("Arrhenius(", m_label.end())
    if arr_start == -1:
        logger.warning(f"'Arrhenius(' not found after label for reaction: {reaction.label}")
        return
    idx = arr_start + len("Arrhenius(")
    depth = 1

    while idx < len(output_content) and depth > 0:
        c = output_content[idx]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        idx += 1

    if depth != 0:
        logger.warning(f"Unbalanced parentheses in Arrhenius block for {reaction.label}")
        return

    arr_block = output_content[arr_start + len("Arrhenius(") : idx - 1]  # Extract the interior of Arrhenius( … )

    kinetics = dict()
    def find_tuple(key):
        pat = rf"{key}\s*=\s*\(\s*([-+]?[\d.eE+-]+)\s*,\s*['\"]([^'\"]+)['\"]"
        m = re.search(pat, arr_block)
        if m:
            return float(m.group(1)), m.group(2)
        return None

    def find_scalar(key):
        pat = rf"{key}\s*=\s*([-+]?[\d.eE+-]+)"
        m = re.search(pat, arr_block)
        return float(m.group(1)) if m else None

    kinetics["A"]    = find_tuple("A")
    kinetics["n"]    = find_scalar("n")
    kinetics["Ea"]   = find_tuple("Ea")
    kinetics["T0"]   = find_tuple("T0")
    kinetics["Tmin"] = find_tuple("Tmin")
    kinetics["Tmax"] = find_tuple("Tmax")
    m_comment = re.search(r"comment\s*=\s*['\"](.*?)['\"]", arr_block, re.DOTALL)
    if m_comment:
        kinetics["comment"] = m_comment.group(1).strip()
    reaction.kinetics = kinetics


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
