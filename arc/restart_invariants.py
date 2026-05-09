"""
Restart-state invariants for the scheduler's ``running_jobs`` and
``job_dict`` data structures.

The two structures must stay in sync: every job name in ``running_jobs[label]``
must resolve to an entry in ``job_dict[label][...]``. When they drift,
``save_restart_dict`` crashes with a ``KeyError`` during the list-comprehension
that builds the restart YAML — see GitHub issues #632 and #624.

This module provides a pure check function plus a repair helper. The check
returns a list of ``InvariantViolation`` records (empty if all is well); the
repair drops names from ``running_jobs`` that have no backing in ``job_dict``,
producing a self-consistent state safe for serialization.
"""

from dataclasses import dataclass

from arc.checks.common import CONFORMER_JOB_TYPES, get_i_from_job_name, is_conformer_job


@dataclass(frozen=True)
class InvariantViolation:
    """
    One missing-backing finding from the consistency check.

    Attributes:
        label: The species label whose ``running_jobs`` entry has no backing.
        job_name: The unresolved job name (e.g. 'conf_opt_3', 'tsg2', 'opt_a4001').
        reason: Short human-readable description of why resolution failed.
    """
    label: str
    job_name: str
    reason: str


def resolve_job_dict_path(job_name: str) -> tuple[str, int | str] | None:
    """
    Compute the ``job_dict[label][...]`` lookup path for a given job name.

    Mirrors the lookup logic in ``Scheduler.save_restart_dict`` so the
    invariants check sees exactly what the save would.

    Args:
        job_name: The job name (e.g. ``'conf_opt_3'``, ``'tsg2'``, ``'opt_a4001'``).

    Returns:
        (job_type_key, sub_key) — for conformer/tsg jobs the sub_key is the
        integer index, for other jobs the sub_key is the full job name.
        Returns ``None`` if the job name is malformed.
    """
    if is_conformer_job(job_name):
        for prefix in CONFORMER_JOB_TYPES:
            if job_name.startswith(prefix):
                idx = get_i_from_job_name(job_name)
                if idx is None:
                    return None
                return prefix, idx
        return None
    if 'tsg' in job_name:
        idx = get_i_from_job_name(job_name)
        if idx is None:
            return None
        return 'tsg', idx
    if '_' not in job_name:
        return None
    return job_name.rsplit('_', 1)[0], job_name


def check_restart_dict_consistency(
    running_jobs: dict[str, list[str]],
    job_dict: dict[str, dict],
) -> list[InvariantViolation]:
    """
    Verify that every job name in ``running_jobs`` resolves to an entry in ``job_dict``.

    This is the precondition that ``Scheduler.save_restart_dict`` requires
    in order to not raise a ``KeyError`` during serialization.

    Args:
        running_jobs: Mapping label -> list of job names presumed running.
        job_dict: Mapping label -> {job_type -> {sub_key -> Job}}.

    Returns:
        A list of ``InvariantViolation`` records (empty when consistent).
    """
    violations: list[InvariantViolation] = []
    for label, names in (running_jobs or {}).items():
        spc_jobs = job_dict.get(label) if job_dict else None
        if spc_jobs is None:
            for name in names:
                violations.append(InvariantViolation(
                    label=label,
                    job_name=name,
                    reason=f'job_dict has no entry for label {label!r}',
                ))
            continue
        for name in names:
            path = resolve_job_dict_path(name)
            if path is None:
                violations.append(InvariantViolation(
                    label=label,
                    job_name=name,
                    reason=f'cannot derive job_dict path from job_name {name!r}',
                ))
                continue
            job_type_key, sub_key = path
            job_type_bucket = spc_jobs.get(job_type_key)
            if job_type_bucket is None:
                violations.append(InvariantViolation(
                    label=label,
                    job_name=name,
                    reason=f"job_dict[{label!r}] missing job_type key {job_type_key!r}",
                ))
                continue
            if sub_key not in job_type_bucket:
                violations.append(InvariantViolation(
                    label=label,
                    job_name=name,
                    reason=(
                        f"job_dict[{label!r}][{job_type_key!r}] missing sub-key {sub_key!r}"
                    ),
                ))
    return violations


def repair_running_jobs(
    running_jobs: dict[str, list[str]],
    job_dict: dict[str, dict],
) -> tuple[dict[str, list[str]], list[InvariantViolation]]:
    """
    Drop unresolved entries from ``running_jobs`` to make the snapshot self-consistent.

    The repair is non-destructive on the input: a fresh dict is returned. Names
    that fail to resolve in ``job_dict`` are removed; labels that end up with
    no remaining names are dropped from the output mapping. The list of
    violations encountered is returned alongside so callers can log them.

    Args:
        running_jobs: Mapping label -> list of job names.
        job_dict: Mapping label -> {job_type -> {sub_key -> Job}}.

    Returns:
        (repaired_running_jobs, removed_violations).
    """
    violations = check_restart_dict_consistency(running_jobs, job_dict)
    if not violations:
        return dict(running_jobs), []
    bad_pairs = {(v.label, v.job_name) for v in violations}
    repaired: dict[str, list[str]] = {}
    for label, names in running_jobs.items():
        kept = [name for name in names if (label, name) not in bad_pairs]
        if kept:
            repaired[label] = kept
    return repaired, violations
