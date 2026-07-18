"""Zombie-job detection helpers.

A "zombie" is a queue-running job that has produced no output traffic by the
grace period: scheduler reports it as RUNNING, but the ESS process has wedged
or never started. The orchestration (kill + resubmit + per-(species, job_type)
cap) lives on the Scheduler; the pure decision logic and ESS classification
live here.
"""

import datetime
import os
from collections.abc import Collection
from typing import TYPE_CHECKING

from arc.common import get_logger
from arc.imports import settings
from arc.job.ssh import SSHClient

if TYPE_CHECKING:
    from arc.job.adapter import JobAdapter


logger = get_logger()


class RemoteStatError(RuntimeError):
    """Raised when the remote stat of a job's output file failed (e.g., a transient
    SSH error), as opposed to a successful stat that found no output file."""


# NOTE: submit templates that redirect ESS output to a node scratch dir and only
# copy it back to remote_path at job end (as on zeus) make the "no output traffic"
# heuristic blind to live progress, so any job past the grace looks like a zombie.
# Keep the grace well above the longest legitimate job (TS opt ~1 h, high-level
# single points up to several hours) to avoid killing healthy long jobs.
# Must exceed the longest legitimate quiet period; 1 h proved too short for
# slow TS opts during development.
# The effective grace is settings['zombie_grace_seconds'] with an optional
# per-server 'zombie_grace_seconds' override in the servers dict (see
# get_zombie_grace_seconds); this constant is the importable fallback.
ZOMBIE_GRACE_SECONDS = 21600  # 6 h

ZOMBIE_OUTPUT_FILENAME_FALLBACK = 'out.txt'

# ESS that flush login-visible output as the job runs (per SCF / per CC iter
# / per opt step). For these, absence of any output traffic after the grace
# period is a strong "zombie" signal. Incore-only or near-instant ESS
# (xtb / torchani / openbabel / mockter) are exempt.
ESS_PERIODIC_WRITERS = frozenset({
    'cfour', 'gaussian', 'molpro', 'orca', 'psi4', 'qchem', 'terachem',
})


def get_zombie_grace_seconds(server: str | None = None) -> int:
    """
    Resolve the effective zombie grace period for a server.

    The per-server 'zombie_grace_seconds' key in the servers settings dict takes
    precedence, then the global 'zombie_grace_seconds' setting, then the
    :data:`ZOMBIE_GRACE_SECONDS` fallback.

    Args:
        server (str, optional): The server name to resolve the override for.

    Returns:
        int: The grace period in seconds.
    """
    default = settings.get('zombie_grace_seconds', ZOMBIE_GRACE_SECONDS)
    if server:
        return settings.get('servers', {}).get(server, {}).get('zombie_grace_seconds', default)
    return default


def output_mtime(job: 'JobAdapter') -> datetime.datetime | None:
    """Return the latest mtime of the job's ESS output file.

    Tries the configured ESS output filename first and falls back to the
    wrapper log. Local jobs use ``os.path.getmtime``; remote jobs use
    ``SSHClient.get_last_modified_time`` against ``job.remote_path``.

    Args:
        job (JobAdapter): The job whose output file should be stat'ed. Required
            attributes: ``job_adapter``, ``server``, ``local_path``,
            ``local_path_to_output_file``, ``remote_path``, ``job_name``.

    Returns:
        datetime.datetime | None: The output file's mtime, or ``None`` if the
        stat succeeded but no candidate output file exists.

    Raises:
        RemoteStatError: If the remote stat itself failed (e.g., a transient
            SSH error), so callers can skip the check rather than treat the
            job as having no output.
    """
    out_filename = settings.get('output_filenames', {}).get(job.job_adapter)
    if job.server is None or job.server in ('', 'local'):
        candidates = [job.local_path_to_output_file]
        if out_filename:
            candidates.append(os.path.join(job.local_path, out_filename))
        for path in candidates:
            if path and os.path.isfile(path):
                return datetime.datetime.fromtimestamp(os.path.getmtime(path))
        return None
    try:
        with SSHClient(job.server) as ssh:
            p1 = os.path.join(job.remote_path, out_filename) if out_filename else None
            p2 = os.path.join(job.remote_path, ZOMBIE_OUTPUT_FILENAME_FALLBACK)
            return ssh.get_last_modified_time(remote_file_path_1=p1 or p2,
                                              remote_file_path_2=p2)
    except Exception as exc:
        logger.warning(
            f'Could not stat remote output for job {job.job_name} on '
            f'{job.server} ({type(exc).__name__}: {exc}); skipping zombie check.'
        )
        raise RemoteStatError(f'Failed to stat remote output for job {job.job_name} on {job.server}') from exc


def is_zombie(job: 'JobAdapter',
              server_job_ids: Collection[int | str],
              now: datetime.datetime | None = None,
              running_since: datetime.datetime | None = None,
              ) -> bool:
    """Decide whether a job is a zombie.

    Pure decision: takes the queue's running set rather than reaching into a
    ``Scheduler``. A job is a zombie iff all of these hold:

    * Its ``execution_type`` is not ``'incore'``.
    * Its ESS is in :data:`ESS_PERIODIC_WRITERS`.
    * The queue still reports it as running (``job.job_id in server_job_ids``).
    * It has been past the grace period (:func:`get_zombie_grace_seconds`,
      resolved for ``job.server``) since the clock start ``t0``, where ``t0``
      is ``running_since`` if given, else ``job.initial_time``.
    * Its output file is missing, or its mtime is at-or-before ``t0``.

    If the remote stat itself failed (:class:`RemoteStatError`), the check is
    skipped and ``False`` is returned: a transient SSH failure must not flag a
    healthy job as a zombie.

    Args:
        job (JobAdapter): The job to check. Required attributes:
            ``execution_type``, ``job_adapter``, ``job_id``, ``initial_time``,
            plus everything :func:`output_mtime` needs.
        server_job_ids (Collection[int | str]): The queue job IDs the scheduler
            currently considers running. Membership is tested with ``in``.
        now (datetime.datetime, optional): Reference "current time" for the
            grace-period check. Defaults to ``datetime.datetime.now()``;
            override in tests for determinism.
        running_since (datetime.datetime, optional): The first time the job was
            observed in a RUNNING queue state. Supersedes ``job.initial_time``
            as the grace-clock start, so time spent queued does not count.

    Returns:
        bool: ``True`` if the job is a zombie, ``False`` otherwise.
    """
    if job.execution_type == 'incore':
        return False
    adapter_name = (getattr(job, 'job_adapter', None) or '').lower()
    if adapter_name not in ESS_PERIODIC_WRITERS:
        return False
    if job.job_id is None or job.job_id not in server_job_ids:
        return False
    t0 = running_since or job.initial_time
    if t0 is None:
        return False
    now = now or datetime.datetime.now()
    if (now - t0).total_seconds() < get_zombie_grace_seconds(getattr(job, 'server', None)):
        return False
    try:
        mtime = output_mtime(job)
    except RemoteStatError:
        return False
    if mtime is None:
        return True
    return mtime <= t0
