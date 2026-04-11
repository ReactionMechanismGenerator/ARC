"""
A module with common functionalities used for checking the quality of calculations,
contains helper functions for Scheduler.
"""

import datetime

CONFORMER_JOB_TYPES = ('conf_opt', 'conf_sp')


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
