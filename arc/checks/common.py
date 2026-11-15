"""
A module with common functionalities used for checking the quality of calculations,
contains helper functions for Scheduler.
"""

import datetime
import re
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from arc.species.species import ARCSpecies

CONFORMER_JOB_TYPES = ('conf_opt', 'conf_sp')
TS_IRC_FAILED_MARKER = 'INVALID TS (failed IRC validation)'
NMD_NEAR_MISS_KEY = 'NMD_near_miss'
NMD_ACCEPTED_ON_IRC_WARNING = 'NMD below the 5% gate, accepted on IRC: '


def get_conformer_job_name(job_type: str, i: int) -> str:
    """
    Get the name of a conformer job, the inverse of ``get_i_from_job_name()``.
    This is the single definition of the conformer job name format,
    used both when spawning a job and when restoring one from a restart file.

    Args:
        job_type (str): The conformer job type, e.g., 'conf_opt' or 'conf_sp'.
        i (int): The conformer index.

    Returns:
        str: The conformer job name, e.g., 'conf_opt_3'.
    """
    return f'{job_type}_{i}'


def is_conformer_job(job_name: str) -> bool:
    """
    Check whether a job name represents a conformer job.

    Args:
        job_name (str): The job name, e.g., 'conf_opt_3' or 'conf_sp_0'.

    Returns:
        bool: ``True`` if the job name starts with a conformer job type prefix.
    """
    return job_name.startswith(CONFORMER_JOB_TYPES)


def sum_time_delta(timedelta_list: list[datetime.timedelta]) -> datetime.timedelta:
    """
    A helper function for summing datetime.timedelta objects.

    Args:
        timedelta_list (list): Time delta's to sum.

    Returns:
        datetime.timedelta: The timedelta sum.
    """
    result = datetime.timedelta(0)
    for timedelta in timedelta_list:
        if type(timedelta) == type(result):
            result += timedelta
    return result


def get_i_from_job_name(job_name: str) -> int | None:
    """
    Get the conformer or tsg index from the job name.

    Args:
        job_name (str): The job name, e.g., 'conf_opt_12', 'conf_sp_3', or 'tsg5'.

    Returns:
        int | None: The corresponding conformer or tsg index.
    """
    i = None
    for prefix in CONFORMER_JOB_TYPES:
        if job_name.startswith(prefix):
            i = int(job_name[len(prefix) + 1:])  # +1 for the '_' separator
            return i
    if job_name.startswith('tsg'):
        i = int(job_name[3:])
    return i


def get_index_of_abs_largest_neg_freq(freqs: np.ndarray | None) -> int | None:
    """
    Get the index of the |largest| negative frequency.

    Args:
        freqs (np.ndarray, optional): Entries are frequency values.

    Returns:
        int | None: The 0-index of the largest absolute negative frequency,
                    ``None`` if no frequencies were given or none of them is negative.
    """
    if freqs is None or not len(freqs) or all(freq > 0 for freq in freqs):
        return None
    return list(freqs).index(min(freqs))


def is_ts_check_exempt(check: str,
                       ts_checks: dict,
                       ) -> bool:
    """
    Determine whether a failed TS check is exempt, i.e., does not count as a TS validation failure.

    The only exemption is 'e_elect': the electronic-energy barrier check is superseded by the
    zero-point-corrected 'E0' check, so a failed 'e_elect' is excused once 'E0' passed.
    'E0' is looked up with ``dict.get()``, so a ``ts_checks`` dictionary that has no 'E0' key
    yields ``False`` (not exempt) instead of raising a ``KeyError``.

    Args:
        check (str): The key of the ``ts_checks`` entry being evaluated.
        ts_checks (dict): The ``ts_checks`` dictionary of the TS species.

    Returns:
        bool: Whether a failed ``check`` is exempt.
    """
    return check == 'e_elect' and bool(ts_checks.get('E0'))


def record_ts_check_warning(species: 'ARCSpecies',
                            warning: str,
                            ) -> None:
    """
    Append a diagnostic ``warning`` to a TS species' ``ts_checks['warnings']`` entry.

    The entry accumulates the reasons a TS check could not reach a verdict, and is written to the
    restart and output files, so a repeated check must not repeat its message. A ``warning`` that is
    already present is not appended again.

    Args:
        species (ARCSpecies): The TS species.
        warning (str): The message to record.
    """
    if warning not in species.ts_checks['warnings']:
        species.ts_checks['warnings'] += warning


def get_ts_validation_comment(ts_species: 'ARCSpecies | None') -> str | None:
    """
    Get a human-readable marker describing a positively-failed TS validation.

    Only a ``ts_checks['IRC']`` value of ``False`` (checked and failed) produces a marker.
    A value of ``None`` means the IRC check was not performed (e.g., IRC was not requested,
    or the reaction connectivity was unavailable) and is treated as unknown, not as a failure.

    The marker names any other TS check that also failed, applying the same exemption as
    ``ts.ts_passed_checks()``: a ``False`` 'e_elect' is not reported once 'E0' passed.

    Args:
        ts_species (ARCSpecies, optional): The TS species of the reaction the rate was computed for.

    Returns:
        str | None: The marker, or ``None`` if the TS did not positively fail the IRC check.
    """
    ts_checks = getattr(ts_species, 'ts_checks', None) or dict()
    if ts_checks.get('IRC', None) is not False:
        return None
    comment = f'{TS_IRC_FAILED_MARKER}: the optimized IRC endpoints of this TS do not correspond to the ' \
              f'reactants and products of this reaction, therefore this rate coefficient does not describe ' \
              f'this reaction and must not be used.'
    other_failed_checks = sorted(key for key, val in ts_checks.items()
                                 if key not in ['IRC', 'warnings'] and val is False
                                 and not is_ts_check_exempt(key, ts_checks))
    if other_failed_checks:
        comment += f' Additional TS checks that failed: {", ".join(other_failed_checks)}.'
    return comment


def get_nmd_near_miss(ts_checks: dict | None) -> dict | None:
    """
    Get the near-miss record of a TS whose normal mode displacement check failed only on the 5% reactive-bond gate.

    The record is the one ``check_normal_mode_displacement()`` stores under ``NMD_NEAR_MISS_KEY``. It is returned
    only beside an NMD verdict of ``False``, and only when it is a dict with a non-empty ``sub_gate_bonds`` list
    whose every entry passes ``is_sub_gate_bond()``.
    Anything else (no record, a malformed one, or an NMD verdict other than ``False``) reads as "not a near miss",
    so that a TS is never kept on a record that cannot be relied upon.

    Args:
        ts_checks (dict, optional): The ``ts_checks`` dictionary of a TS species.

    Returns:
        dict | None: The near-miss record, or ``None``.
    """
    if not isinstance(ts_checks, dict) or ts_checks.get('NMD') is not False:
        return None
    record = ts_checks.get(NMD_NEAR_MISS_KEY)
    if not isinstance(record, dict) or not isinstance(record.get('sub_gate_bonds'), list) \
            or not record['sub_gate_bonds'] or not all(is_sub_gate_bond(bond) for bond in record['sub_gate_bonds']):
        return None
    return record


def is_sub_gate_bond(bond) -> bool:
    """
    Check that an entry of a near-miss record's ``sub_gate_bonds`` has the shape the record is written in.

    Args:
        bond: The entry to check.

    Returns:
        bool: Whether the entry is a dict with ``atoms`` (two distinct non-negative integers), ``symbols`` (a
              string) and ``percent`` (a finite number from 0 to 5, since a change just under the 5% gate is
              stored rounded and can read 5.0).
    """
    if not isinstance(bond, dict):
        return False
    atoms, percent = bond.get('atoms'), bond.get('percent')
    return isinstance(atoms, list) and len(atoms) == 2 \
        and all(isinstance(i, int) and not isinstance(i, bool) and i >= 0 for i in atoms) \
        and atoms[0] != atoms[1] \
        and isinstance(bond.get('symbols'), str) \
        and isinstance(percent, (int, float)) and not isinstance(percent, bool) \
        and 0 <= percent <= 5.0


def get_nmd_near_miss_summary(record: dict) -> str:
    """
    Summarize the sub-gate bonds of a near-miss record in one line.

    Args:
        record (dict): A near-miss record, as returned by ``get_nmd_near_miss()``.

    Returns:
        str: The element symbols, 0-indexed atoms and percent change of each sub-gate bond, e.g.,
             ``'C-Cl (1, 2) 4.55%'``.
    """
    return ', '.join(f"{bond['symbols']} ({bond['atoms'][0]}, {bond['atoms'][1]}) {bond['percent']}%"
                     for bond in record['sub_gate_bonds'])


def remove_nmd_accepted_on_irc_warning(warnings: str) -> str:
    """
    Remove the warning that a TS was accepted on IRC despite a near-miss NMD failure.

    The warning describes one TS geometry, so it must not outlive a switch to another TS guess. Every other
    warning in ``warnings`` is kept as is.

    Args:
        warnings (str): Warnings separated by ``'; '``, as stored in the Scheduler output dict.

    Returns:
        str: ``warnings`` without any ``NMD_ACCEPTED_ON_IRC_WARNING`` entry.
    """
    return re.sub(re.escape(NMD_ACCEPTED_ON_IRC_WARNING) + r'[^;]*; ', '', warnings)
