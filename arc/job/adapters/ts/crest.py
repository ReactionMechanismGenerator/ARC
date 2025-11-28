"""
Utilities for running CREST within ARC.

Separated from heuristics so CREST can be conditionally imported and reused.
"""

import os
import re
import time
from typing import List

import numpy as np
import pandas as pd

from arc.common import get_logger
from arc.imports import settings, submit_scripts
from arc.job.local import check_job_status, submit_job
from arc.species.converter import reorder_xyz_string, str_to_xyz, xyz_to_dmat, xyz_to_str

logger = get_logger()

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
        f'-T {local_server.get("cpus", 8)}',
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
            os.chmod(submit_path, 0o750)

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
        time.sleep(min(check_interval, 100))


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
            atom for atom in closest_atoms if not atom.startswith("H")
        )
        if_hydrogen_present = any(
            atom
            for atom in closest_atoms
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
                    print(f"Error accessing distances for occurrence {occurrence}: {e}")

            # Select the occurrence with the smallest distance
            best_occurrence = min(occurrence_distances, key=lambda x: x[1])[0]
            return {
                "H": extract_digits(best_occurrence["H"]),
                "A": extract_digits(best_occurrence["A"]),
                "B": extract_digits(best_occurrence["B"]),
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

    return {}
