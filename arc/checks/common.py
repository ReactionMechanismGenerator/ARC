"""
A module with common functionalities used for checking the quality of calculations,
contains helper functions for Scheduler.
"""

import datetime
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from arc.species.species import ARCSpecies

CONFORMER_JOB_TYPES = ('conf_opt', 'conf_sp')
TS_IRC_FAILED_MARKER = 'INVALID TS (failed IRC validation)'


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
        job_name (str): The job name, e.g., 'conformer12' or 'tsg5'.

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
