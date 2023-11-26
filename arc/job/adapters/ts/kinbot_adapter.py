
import datetime
from typing import TYPE_CHECKING, List, Optional, Tuple, Union
import subprocess
import os
import numpy as np

from arc.common import ARC_PATH, almost_equal_coords, get_logger, save_yaml_file, read_yaml_file
from arc.job.adapter import JobAdapter
from arc.job.adapters.common import _initialize_adapter
from arc.job.factory import register_job_adapter
from arc.plotter import save_geo
from arc.species.converter import xyz_from_data, xyz_to_kinbot_list
from arc.species.species import ARCSpecies, TSGuess, colliding_atoms
from arc.imports import settings

if TYPE_CHECKING:
    from rmgpy.molecule import Molecule
    from arc.level import Level
    from arc.reaction import ARCReaction
    from arc.species import ARCSpecies

logger = get_logger()

default_job_settings, global_ess_settings, input_filenames, output_filenames, servers, submit_filenames, KINBOT_PYTHON = \
    settings['default_job_settings'], settings['global_ess_settings'], settings['input_filenames'], \
    settings['output_filenames'], settings['servers'], settings['submit_filenames'], settings['KINBOT_PYTHON']

KINBOT_SCRIPT_PATH = os.path.join(ARC_PATH, 'arc', 'job', 'adapters', 'scripts', 'kinbot_script.py')

class KinBotAdapter(JobAdapter):

    """
    A class for executing KinBot Jobs
    """

    def __init__(self,
                 project: str,
                 project_directory: str,
                 job_type: Union[List[str], str],
                 args: Optional[dict] = None,
                 bath_gas: Optional[str] = None,
                 checkfile: Optional[str] = None,
                 conformer: Optional[int] = None,
                 constraints: Optional[List[Tuple[List[int], float]]] = None,
                 cpu_cores: Optional[str] = None,
                 dihedral_increment: Optional[float] = None,
                 dihedrals: Optional[List[float]] = None,
                 directed_scan_type: Optional[str] = None,
                 ess_settings: Optional[dict] = None,
                 ess_trsh_methods: Optional[List[str]] = None,
                 execution_type: Optional[str] = None,
                 fine: bool = False,
                 initial_time: Optional[Union['datetime.datetime', str]] = None,
                 irc_direction: Optional[str] = None,
                 job_id: Optional[int] = None,
                 job_memory_gb: float = 14.0,
                 job_name: Optional[str] = None,
                 job_num: Optional[int] = None,
                 job_server_name: Optional[str] = None,
                 job_status: Optional[List[Union[dict, str]]] = None,
                 level: Optional['Level'] = None,
                 max_job_time: Optional[float] = None,
                 reactions: Optional[List['ARCReaction']] = None,
                 rotor_index: Optional[int] = None,
                 server: Optional[str] = None,
                 server_nodes: Optional[list] = None,
                 species: Optional[List['ARCSpecies']] = None,
                 testing: bool = False,
                 times_rerun: int = 0,
                 torsions: Optional[List[List[int]]] = None,
                 tsg: Optional[int] = None,
                 xyz: Optional[dict] = None,) -> None:
        
        self.incore_capacity = 100
        self.job_adapter = 'kinbot'
        self.execution_type = execution_type or 'incore'
        self.command = None #No executable file
        self.url = 'https://github.com/zadorlab/KinBot'

        self.family_map = {'1+2_Cycloaddition': ['r12_cycloaddition'],
                           '1,2_Insertion_CO': ['r12_insertion_R'],
                           '1,2_Insertion_carbene': ['r12_insertion_R'],
                           '1,2_shiftS': ['12_shift_S_F', '12_shift_S_R'],
                           '1,3_Insertion_CO2': ['r13_insertion_CO2'],
                           '1,3_Insertion_ROR': ['r13_insertion_ROR'],
                           '1,3_Insertion_RSR': ['r13_insertion_RSR'],
                           '2+2_cycloaddition': ['r22_cycloaddition'],
                           'Cyclic_Ether_Formation': ['Cyclic_Ether_Formation'],
                           'Diels_alder_addition': ['Diels_alder_addition'],
                           'HO2_Elimination_from_PeroxyRadical': ['HO2_Elimination_from_PeroxyRadical'],
                           'Intra_Diels_alder_monocyclic': ['Intra_Diels_alder_R'],
                           'Intra_ene_reaction': ['cpd_H_migration'],
                           'intra_H_migration': ['intra_H_migration', 'intra_H_migration_suprafacial'],
                           'intra_OH_migration': ['intra_OH_migration'],
                           'Intra_R_Add_Endocyclic': ['Intra_R_Add_Endocyclic_F'],
                           'Intra_R_Add_Exocyclic': ['Intra_R_Add_Exocyclic_F'],
                           'Intra_R_Add_ExoTetCyclic': ['Intra_R_Add_ExoTetCyclic_F'],
                           'Intra_Retro_Diels_alder_bicyclic': ['Intra_Diels_alder_R'],  # not sure if these fit together
                           'Intra_RH_Add_Endocyclic': ['Intra_RH_Add_Endocyclic_F', 'Intra_RH_Add_Endocyclic_R'],
                           'Intra_RH_Add_Exocyclic': ['Intra_RH_Add_Exocyclic_F', 'Intra_RH_Add_Exocyclic_R'],
                           'ketoenol': ['ketoenol'],
                           'Korcek_step2': ['Korcek_step2'],
                           'R_Addition_COm': ['R_Addition_COm3_R'],
                           'R_Addition_CSm': ['R_Addition_CSm_R'],
                           'R_Addition_MultipleBond': ['R_Addition_MultipleBond'],
                           'Retroene': ['Retro_Ene'],
                           # '?': ['intra_R_migration'],  # unknown
                           }
        self.supported_families = list(self.family_map.keys())
        
        if reactions is None:
            raise ValueError('Cannot execute KinBot without ARCReaction object(s).')
        
        _initialize_adapter(obj=self,
                            is_ts=True,
                            project=project,
                            project_directory=project_directory,
                            job_type=job_type,
                            args=args,
                            bath_gas=bath_gas,
                            checkfile=checkfile,
                            conformer=conformer,
                            constraints=constraints,
                            cpu_cores=cpu_cores,
                            dihedral_increment=dihedral_increment,
                            dihedrals=dihedrals,
                            directed_scan_type=directed_scan_type,
                            ess_settings=ess_settings,
                            ess_trsh_methods=ess_trsh_methods,
                            fine=fine,
                            initial_time=initial_time,
                            irc_direction=irc_direction,
                            job_id=job_id,
                            job_memory_gb=job_memory_gb,
                            job_name=job_name,
                            job_num=job_num,
                            job_server_name=job_server_name,
                            job_status=job_status,
                            level=level,
                            max_job_time=max_job_time,
                            reactions=reactions,
                            rotor_index=rotor_index,
                            server=server,
                            server_nodes=server_nodes,
                            species=species,
                            testing=testing,
                            times_rerun=times_rerun,
                            torsions=torsions,
                            tsg=tsg,
                            xyz=xyz,
                            )
        
    def write_input_file(self,
                         mol,
                         families,
                         kinbot_xyz,
                         multiplicity,
                         charge) -> None:
        """
        Write the input file to execute the job on the server.
        """
        content = [{'mol':mol,
                          'families': families,
                          'kinbot_xyz': kinbot_xyz,
                          'multiplicity': multiplicity,
                          'charge': charge}]
        save_yaml_file(path=os.path.join(self.local_path, "input.yml"),content=content)

    def set_files(self) -> None:
        """
        Set files to be uploaded and downloaded. Writes the files if needed.
        Modifies the self.files_to_upload and self.files_to_download attributes.
        self.files_to_download is a list of remote paths.
        self.files_to_upload is a list of dictionaries, each with the following keys:
        ``'name'``, ``'source'``, ``'make_x'``, ``'local'``, and ``'remote'``.
        If ``'source'`` = ``'path'``, then the value in ``'local'`` is treated as a file path.
        Else if ``'source'`` = ``'input_files'``, then the value in ``'local'`` will be taken
        from the respective entry in inputs.py
        If ``'make_x'`` is ``True``, the file will be made executable.
        """
        pass
    
    def set_additional_file_paths(self) -> None:
        """
        Set additional file paths specific for the adapter.
        Called from set_file_paths() and extends it.
        """
        self.local_path_to_output_file = os.path.join(self.local_path, 'output.yml')

    def set_input_file_memory(self) -> None:
        """
        Set the input_file_memory attribute.
        """
        pass

    def execute_incore(self):
        """
        Execute job incore
        """

        self._log_job_execution()
        self.initial_time = self.initial_time if self.initial_time else datetime.datetime.now()

        self.reactions = [self.reactions] if not isinstance(self.reactions, list) else self.reactions
        for rxn in self.reactions:
            if rxn.family.label in self.supported_families:
                if rxn.ts_species is None:
                    # Mainly used for testing, in an ARC run the TS species should already exist.
                    rxn.ts_species = ARCSpecies(label='TS',
                                                is_ts=True,
                                                charge=rxn.charge,
                                                multiplicity=rxn.multiplicity,
                                                )
                species_to_explore = dict()
                if len(rxn.r_species) == 1:
                    species_to_explore['F'] = rxn.r_species[0]
                if len(rxn.p_species) == 1:
                    species_to_explore['R'] = rxn.p_species[0]

                if not species_to_explore:
                    logger.error(f'Cannot execute KinBot for a non-unimolecular reaction.\n'
                                 f'Got {len(rxn.r_species)} reactants and {rxn.p_species} products in\n{rxn}.')
                    continue

                method_index = 0
                for method_direction, spc in species_to_explore.items():
                    symbols = spc.get_xyz()['symbols']
                    for m, mol in enumerate(spc.mol_list):



                        self.write_input_file(mol= mol.to_smiles(),
                                              families = self.family_map[rxn.family.label],
                                              kinbot_xyz=xyz_to_kinbot_list(spc.get_xyz()),
                                              multiplicity=rxn.multiplicity,
                                              charge = rxn.charge)
                        
                        #self.input_file_path = str(self.local_path) + '/input.yml'
                        commands = ['source ~/.bashrc',
                                    f'{KINBOT_PYTHON} {KINBOT_SCRIPT_PATH} '
                                    f' {self.local_path} + /input.yml']
                        command = '; '.join(commands)
                        output = subprocess.run(command, shell=True, executable='/bin/bash')

                        try:
                            output = read_yaml_file(path=os.path.join(self.local_path, "output.yml"))
                            print(output)
                        except FileNotFoundError:
                            logger.error("Could not find output file")

                        for i in range(len(output)):

                            temp_output = output[list(output.keys())[i]]
                            ts_guess = TSGuess(method='KinBot',
                                                method_direction=method_direction,
                                                method_index=method_index,
                                                index=len(rxn.ts_species.ts_guesses),
                                                )
                            ts_guess.tic()




                            ts_guess.tok()
                            unique = True

                            if temp_output['success']:
                                ts_guess.success = True
                                xyz = xyz_from_data(coords=np.array(temp_output['coords']), symbols=symbols)
                            else:
                                for other_tsg in rxn.ts_species.ts_guesses:
                                    if other_tsg.success and almost_equal_coords(xyz, other_tsg.initial_xyz):
                                        if 'kinbot' not in other_tsg.method.lower():
                                            other_tsg.method += ' and KinBot'
                                        unique = False
                                        break
                                if unique:
                                        ts_guess.process_xyz(xyz)
                                        save_geo(xyz=xyz,
                                                 path=self.local_path,
                                                 filename=f'KinBot {method_direction} {method_index}',
                                                 format_='xyz',
                                                 comment=f'KinBot {method_direction} {method_index}'
                                                 )
                            if not temp_output['success']:
                                ts_guess.success = False
                            if unique:
                                rxn.ts_species.ts_guesses.append(ts_guess)
                                method_index += 1
            if len(self.reactions) < 5:
                successes = len([tsg for tsg in rxn.ts_species.ts_guesses if tsg.success and 'kinbot' in tsg.method])
                if successes:
                    logger.info(f'KinBot successfully found {successes} TS guesses for {rxn.label}.')
                else:
                    logger.info(f'KinBot did not find any successful TS guesses for {rxn.label}.')
        self.final_time = datetime.datetime.now()

    def execute_queue(self):
        """
        (Execute a job to the server's queue.)
        A single KinBot job will always be executed incore.
        """
        self.execute_incore()

register_job_adapter('kinbot', KinBotAdapter)
