"""
Utilities for running CREST within ARC.

Separated from heuristics so CREST can be conditionally imported and reused.
"""

import datetime
import os
import re
import time
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np
import pandas as pd

from arc.common import almost_equal_coords, get_logger
from arc.imports import settings, submit_scripts
from arc.job.adapter import JobAdapter
from arc.job.adapters.common import _initialize_adapter, ts_adapters_by_rmg_family
from arc.job.factory import register_job_adapter
from arc.job.local import check_job_status, submit_job
from arc.plotter import save_geo
from arc.species.converter import reorder_xyz_string, str_to_xyz, xyz_to_dmat, xyz_to_str
from arc.species.species import ARCSpecies, TSGuess
from arc.job.adapters.ts.heuristics import h_abstraction

if TYPE_CHECKING:
    from arc.level import Level
    from arc.reaction import ARCReaction

logger = get_logger()

MAX_CHECK_INTERVAL_SECONDS = 100

try:
    CREST_PATH = settings["CREST_PATH"]
    CREST_ENV_PATH = settings["CREST_ENV_PATH"]
    SERVERS = settings["servers"]
except KeyError:
    CREST_PATH = None
    CREST_ENV_PATH = None
    SERVERS = {}


def crest_available() -> bool:
    """
    Return whether CREST is configured for use.
    """
    return bool(SERVERS.get("local")) and bool(CREST_PATH or CREST_ENV_PATH)


class CrestAdapter(JobAdapter):
    """
    A class for executing CREST TS conformer searches based on heuristics-generated guesses.
    """

    def __init__(self,
                 project: str,
                 project_directory: str,
                 job_type: Union[List[str], str],
                 args: Optional[dict] = None,
                 bath_gas: Optional[str] = None,
                 checkfile: Optional[str] = None,
                 conformer: Optional[int] = None,
                 constraints: Optional[List] = None,
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
                 run_multi_species: bool = False,
                 reactions: Optional[List['ARCReaction']] = None,
                 rotor_index: Optional[int] = None,
                 server: Optional[str] = None,
                 server_nodes: Optional[list] = None,
                 queue: Optional[str] = None,
                 attempted_queues: Optional[List[str]] = None,
                 species: Optional[List[ARCSpecies]] = None,
                 testing: bool = False,
                 times_rerun: int = 0,
                 torsions: Optional[List[List[int]]] = None,
                 tsg: Optional[int] = None,
                 xyz: Optional[dict] = None,
                 ):

        self.incore_capacity = 50
        self.job_adapter = 'crest'
        self.command = None
        self.execution_type = execution_type or 'incore'

        if reactions is None:
            raise ValueError('Cannot execute TS CREST without ARCReaction object(s).')

        dihedral_increment = dihedral_increment or 30

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
                            run_multi_species=run_multi_species,
                            reactions=reactions,
                            rotor_index=rotor_index,
                            server=server,
                            server_nodes=server_nodes,
                            queue=queue,
                            attempted_queues=attempted_queues,
                            species=species,
                            testing=testing,
                            times_rerun=times_rerun,
                            torsions=torsions,
                            tsg=tsg,
                            xyz=xyz,
                            )

    def write_input_file(self) -> None:
        pass

    def set_files(self) -> None:
        pass

    def set_additional_file_paths(self) -> None:
        pass

    def set_input_file_memory(self) -> None:
        pass

    def execute_incore(self):
        self._log_job_execution()
        self.initial_time = self.initial_time if self.initial_time else datetime.datetime.now()

        supported_families = [key for key, val in ts_adapters_by_rmg_family.items() if 'crest' in val]

        self.reactions = [self.reactions] if not isinstance(self.reactions, list) else self.reactions
        for rxn in self.reactions:
            if rxn.family not in supported_families:
                logger.warning(f'The CREST TS search adapter does not support the {rxn.family} reaction family.')
                continue
            if any(spc.get_xyz() is None for spc in rxn.r_species + rxn.p_species):
                logger.warning(f'The CREST TS search adapter cannot process a reaction if 3D coordinates of '
                               f'some/all of its reactants/products are missing.\nNot processing {rxn}.')
                continue
            if not crest_available():
                logger.warning('CREST is not available. Skipping CREST TS search.')
                continue

            if rxn.ts_species is None:
                rxn.ts_species = ARCSpecies(label='TS',
                                            is_ts=True,
                                            charge=rxn.charge,
                                            multiplicity=rxn.multiplicity,
                                            )

            tsg = TSGuess(method='CREST')
            tsg.tic()

            crest_job_dirs = []
            xyz_guesses = h_abstraction(reaction=rxn,
                                        dihedral_increment=self.dihedral_increment,
                                        path=self.local_path)
            if not xyz_guesses:
                logger.warning(f'CREST TS search failed to generate any seed guesses for {rxn.label}.')
                tsg.tok()
                continue

            for iteration, xyz_entry in enumerate(xyz_guesses):
                if isinstance(xyz_entry, dict):
                    xyz_guess = xyz_entry.get("xyz")
                else:
                    xyz_guess = xyz_entry
                if xyz_guess is None:
                    continue

                df_dmat = convert_xyz_to_df(xyz_guess)
                try:
                    h_abs_atoms_dict = get_h_abs_atoms(df_dmat)
                    crest_job_dir = crest_ts_conformer_search(
                        xyz_guess,
                        h_abs_atoms_dict["A"],
                        h_abs_atoms_dict["H"],
                        h_abs_atoms_dict["B"],
                        path=self.local_path,
                        xyz_crest_int=iteration,
                    )
                    crest_job_dirs.append(crest_job_dir)
                except (ValueError, KeyError) as e:
                    logger.error(f"Could not determine the H abstraction atoms, got:\n{e}")

            if not crest_job_dirs:
                logger.warning(f'CREST TS search failed to prepare any jobs for {rxn.label}.')
                tsg.tok()
                continue

            crest_jobs = submit_crest_jobs(crest_job_dirs)
            monitor_crest_jobs(crest_jobs)
            xyz_guesses_crest = process_completed_jobs(crest_jobs)
            tsg.tok()

            for method_index, xyz in enumerate(xyz_guesses_crest):
                if xyz is None:
                    continue
                unique = True
                for other_tsg in rxn.ts_species.ts_guesses:
                    if almost_equal_coords(xyz, other_tsg.initial_xyz):
                        if 'crest' not in other_tsg.method.lower():
                            other_tsg.method += ' and CREST'
                        unique = False
                        break
                if unique:
                    ts_guess = TSGuess(method='CREST',
                                       index=len(rxn.ts_species.ts_guesses),
                                       method_index=method_index,
                                       t0=tsg.t0,
                                       execution_time=tsg.execution_time,
                                       success=True,
                                       family=rxn.family,
                                       xyz=xyz,
                                       )
                    rxn.ts_species.ts_guesses.append(ts_guess)
                    save_geo(xyz=xyz,
                             path=self.local_path,
                             filename=f'CREST_{method_index}',
                             format_='xyz',
                             comment=f'CREST {method_index}, family: {rxn.family}',
                             )

            if len(self.reactions) < 5:
                successes = [tsg for tsg in rxn.ts_species.ts_guesses if tsg.success and 'crest' in tsg.method.lower()]
                if successes:
                    logger.info(f'CREST successfully found {len(successes)} TS guesses for {rxn.label}.')
                else:
                    logger.info(f'CREST did not find any successful TS guesses for {rxn.label}.')

        self.final_time = datetime.datetime.now()

    def execute_queue(self):
        self.execute_incore()


def crest_ts_conformer_search(
    xyz_guess: dict,
    a_atom: int,
    h_atom: int,
    b_atom: int,
    path: str = "",
    xyz_crest_int: int = 0,
) -> str:
    """
    Prepare a CREST TS conformer search job:
    - Write coords.ref and constraints.inp
    - Write a PBS/HTCondor submit script using submit_scripts["local"]["crest"]
    - Return the CREST job directory path
    """
    path = os.path.join(path, f"crest_{xyz_crest_int}")
    os.makedirs(path, exist_ok=True)

    # --- coords.ref ---
    symbols = xyz_guess["symbols"]
    converted_coords = reorder_xyz_string(
        xyz_str=xyz_to_str(xyz_guess),
        reverse_atoms=True,
        convert_to="bohr",
    )
    coords_ref_content = f"$coord\n{converted_coords}\n$end\n"
    coords_ref_path = os.path.join(path, "coords.ref")
    with open(coords_ref_path, "w") as f:
        f.write(coords_ref_content)

    # --- constraints.inp ---
    num_atoms = len(symbols)
    # CREST uses 1-based indices
    a_atom += 1
    h_atom += 1
    b_atom += 1

    # All atoms not directly involved in A–H–B go into the metadynamics atom list
    list_of_atoms_numbers_not_participating_in_reaction = [
        i for i in range(1, num_atoms + 1) if i not in [a_atom, h_atom, b_atom]
    ]

    constraints_path = os.path.join(path, "constraints.inp")
    with open(constraints_path, "w") as f:
        f.write("$constrain\n")
        f.write(f"  atoms: {a_atom}, {h_atom}, {b_atom}\n")
        f.write("  force constant: 0.5\n")
        f.write("  reference=coords.ref\n")
        f.write(f"  distance: {a_atom}, {h_atom}, auto\n")
        f.write(f"  distance: {h_atom}, {b_atom}, auto\n")
        f.write("$metadyn\n")
        if list_of_atoms_numbers_not_participating_in_reaction:
            f.write(
                f'  atoms: {", ".join(map(str, list_of_atoms_numbers_not_participating_in_reaction))}\n'
            )
        f.write("$end\n")

    # --- build CREST command string ---
    # Example: crest coords.ref --cinp constraints.inp --noreftopo -T 8
    local_server = SERVERS.get("local", {})
    cpus = int(local_server.get("cpus", 8))
    if CREST_ENV_PATH:
        crest_exe = "crest"
    else:
        crest_exe = CREST_PATH if CREST_PATH is not None else "crest"

    commands = [
        crest_exe,
        "coords.ref",
        "--cinp constraints.inp",
        "--noreftopo",
        f"-T {cpus}",
    ]
    command = " ".join(commands)

    # --- activation line (optional) ---
    activation_line = CREST_ENV_PATH or ""

    if SERVERS.get("local") is not None:
        cluster_soft = SERVERS["local"]["cluster_soft"].lower()

        if cluster_soft in ["condor", "htcondor"]:
            # HTCondor branch (kept for completeness – you can delete if you don't use it)
            sub_job = submit_scripts["local"]["crest"]
            format_params = {
                "name": f"crest_{xyz_crest_int}",
                "cpus": cpus,
                "memory": int(SERVERS["local"].get("memory", 32.0) * 1024),
            }
            sub_job = sub_job.format(**format_params)

            with open(
                os.path.join(path, settings["submit_filenames"]["HTCondor"]), "w"
            ) as f:
                f.write(sub_job)

            crest_job = submit_scripts["local"]["crest_job"]
            crest_job = crest_job.format(
                path=path,
                activation_line=activation_line,
                commands=command,
            )

            with open(os.path.join(path, "job.sh"), "w") as f:
                f.write(crest_job)
            os.chmod(os.path.join(path, "job.sh"), 0o700)

            # Pre-create out/err for any status checkers that expect them
            for fname in ("out.txt", "err.txt"):
                fpath = os.path.join(path, fname)
                if not os.path.exists(fpath):
                    with open(fpath, "w") as f:
                        f.write("")
                    os.chmod(fpath, 0o600)

        elif cluster_soft == "pbs":
            # PBS branch that matches your 'crest' template above
            sub_job = submit_scripts["local"]["crest"]
            format_params = {
                "queue": SERVERS["local"].get("queue", "alon_q"),
                "name": f"crest_{xyz_crest_int}",
                "cpus": cpus,
                # 'memory' is in GB for the template: mem={memory}gb
                "memory": int(
                    SERVERS["local"].get("memory", 32)
                    if SERVERS["local"].get("memory", 32) < 60
                    else 40
                ),
                "activation_line": activation_line,
                "commands": command,
            }
            sub_job = sub_job.format(**format_params)

            submit_filename = settings["submit_filenames"]["PBS"]  # usually 'submit.sh'
            submit_path = os.path.join(path, submit_filename)
            with open(submit_path, "w") as f:
                f.write(sub_job)
            os.chmod(submit_path, 0o700)

        else:
            raise ValueError(f"Unsupported cluster_soft for CREST: {cluster_soft!r}")

    return path


def submit_crest_jobs(crest_paths: List[str]) -> dict:
    """
    Submit CREST jobs to the server.

    Args:
        crest_paths (List[str]): List of paths to the CREST directories.

    Returns:
        dict: A dictionary containing job IDs as keys and their statuses as values.
    """
    crest_jobs = {}
    for crest_path in crest_paths:
        job_status, job_id = submit_job(path=crest_path)
        logger.info(f"CREST job {job_id} submitted for {crest_path}")
        crest_jobs[job_id] = {"path": crest_path, "status": job_status}
    return crest_jobs


def monitor_crest_jobs(crest_jobs: dict, check_interval: int = 300) -> None:
    """
    Monitor CREST jobs until they are complete.

    Args:
        crest_jobs (dict): Dictionary containing job information (job ID, path, and status).
        check_interval (int): Time interval (in seconds) to wait between status checks.
    """
    while True:
        all_done = True
        for job_id, job_info in crest_jobs.items():
            if job_info["status"] not in ["done", "failed"]:
                try:
                    job_info["status"] = check_job_status(job_id)  # Update job status
                except Exception as e:
                    logger.error(f"Error checking job status for job {job_id}: {e}")
                    job_info["status"] = "failed"
                if job_info["status"] not in ["done", "failed"]:
                    all_done = False
        if all_done:
            break
        time.sleep(min(check_interval, MAX_CHECK_INTERVAL_SECONDS))


def process_completed_jobs(crest_jobs: dict) -> list:
    """
    Process the completed CREST jobs and update XYZ guesses.

    Args:
        crest_jobs (dict): Dictionary containing job information.
    """
    xyz_guesses = []
    for job_id, job_info in crest_jobs.items():
        crest_path = job_info["path"]
        if job_info["status"] == "done":
            crest_best_path = os.path.join(crest_path, "crest_best.xyz")
            if os.path.exists(crest_best_path):
                with open(crest_best_path, "r") as f:
                    content = f.read()
                xyz_guess = str_to_xyz(content)
                xyz_guesses.append(xyz_guess)
            else:
                logger.error(f"crest_best.xyz not found in {crest_path}")
        elif job_info["status"] == "failed":
            logger.error(f"CREST job failed for {crest_path}")

    return xyz_guesses


def extract_digits(s: str) -> int:
    """
    Extract the first integer from a string

    Args:
        s (str): The string to extract the integer from

    Returns:
        int: The first integer in the string

    """
    return int(re.sub(r"[^\d]", "", s))


def convert_xyz_to_df(xyz: dict) -> pd.DataFrame:
    """
    Convert a dictionary of xyz coords to a pandas DataFrame with bond distances

    Args:
        xyz (dict): The xyz coordinates of the molecule

    Return:
        pd.DataFrame: The xyz coordinates as a pandas DataFrame

    """
    symbols = xyz["symbols"]
    symbol_enum = [f"{symbol}{i}" for i, symbol in enumerate(symbols)]
    ts_dmat = xyz_to_dmat(xyz)

    return pd.DataFrame(ts_dmat, columns=symbol_enum, index=symbol_enum)


def get_h_abs_atoms(dataframe: pd.DataFrame) -> dict:
    """
    Get the donating/accepting hydrogen atom, and the two heavy atoms that are bonded to it

    Args:
        dataframe (pd.DataFrame): The dataframe of the bond distances, columns and index are the atom symbols

    Returns:
        dict: The hydrogen atom and the two heavy atoms. The keys are 'H', 'A', 'B'
    """

    closest_atoms = {}
    for index, row in dataframe.iterrows():

        row[index] = np.inf
        closest = row.nsmallest(2).index.tolist()
        closest_atoms[index] = closest

    hydrogen_keys = [key for key in dataframe.index if key.startswith("H")]
    condition_occurrences = []

    for hydrogen_key in hydrogen_keys:
        atom_neighbours = closest_atoms[hydrogen_key]
        is_heavy_present = any(
            atom for atom in atom_neighbours if not atom.startswith("H")
        )
        if_hydrogen_present = any(
            atom
            for atom in atom_neighbours
            if atom.startswith("H") and atom != hydrogen_key
        )

        if is_heavy_present and if_hydrogen_present:
            # Store the details of this occurrence
            condition_occurrences.append(
                {"H": hydrogen_key, "A": atom_neighbours[0], "B": atom_neighbours[1]}
            )

    # Check if the condition was met
    if condition_occurrences:
        if len(condition_occurrences) > 1:
            # Store distances to decide which occurrence to use
            occurrence_distances = []
            for occurrence in condition_occurrences:
                # Calculate the sum of distances to the two heavy atoms
                hydrogen_key = f"{occurrence['H']}"
                heavy_atoms = [f"{occurrence['A']}", f"{occurrence['B']}"]
                try:
                    distances = dataframe.loc[hydrogen_key, heavy_atoms].sum()
                    occurrence_distances.append((occurrence, distances))
                except KeyError as e:
                    logger.error(f"Error accessing distances for occurrence {occurrence}: {e}")

            # Select the occurrence with the smallest distance
            best_occurrence = min(occurrence_distances, key=lambda x: x[1])[0]
            return {
                "H": extract_digits(best_occurrence["H"]),
                "A": extract_digits(best_occurrence["A"]),
                "B": extract_digits(best_occurrence["B"]),
            }
        else:
            single_occurrence = condition_occurrences[0]
            return {
                "H": extract_digits(single_occurrence["H"]),
                "A": extract_digits(single_occurrence["A"]),
                "B": extract_digits(single_occurrence["B"]),
            }
    else:

        # Check the all the hydrogen atoms, and see the closest two heavy atoms and aggregate their distances to determine which Hyodrogen atom has the lowest distance aggregate
        min_distance = np.inf
        selected_hydrogen = None
        selected_heavy_atoms = None

        for hydrogen_key in hydrogen_keys:
            atom_neighbours = closest_atoms[hydrogen_key]
            heavy_atoms = [atom for atom in atom_neighbours if not atom.startswith("H")]

            if len(heavy_atoms) < 2:
                continue

            distances = dataframe.loc[hydrogen_key, heavy_atoms[:2]].sum()
            if distances < min_distance:
                min_distance = distances
                selected_hydrogen = hydrogen_key
                selected_heavy_atoms = heavy_atoms

        if selected_hydrogen:
            return {
                "H": extract_digits(selected_hydrogen),
                "A": extract_digits(selected_heavy_atoms[0]),
                "B": extract_digits(selected_heavy_atoms[1]),
            }
        else:
            raise ValueError("No valid hydrogen atom found.")


register_job_adapter('crest', CrestAdapter)
