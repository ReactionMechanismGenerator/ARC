"""
A module with common functionalities used for checking the quality of calculations,
contains helper functions for Scheduler.
"""

import datetime

from typing import List, Optional


def sum_time_delta(timedelta_list: List[datetime.timedelta]) -> datetime.timedelta:
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


def get_i_from_job_name(job_name: str) -> Optional[int]:
    """
    Get the conformer or tsg index from the job name.

    Args:
        job_name (str): The job name, e.g., 'conf_opt_0_a7575' or 'tsg5'.

    Returns:
        Optional[int]: The corresponding conformer or tsg index.
    """
    i = None
    if 'conf_opt' in job_name:
        i = int(job_name[9:])
    elif 'conf_sp' in job_name:
        i = int(job_name[8:])
    elif 'tsg' in job_name:
        i = int(job_name[3:])
    return i

