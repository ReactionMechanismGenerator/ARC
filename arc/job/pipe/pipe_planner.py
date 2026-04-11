"""
Pipe task planner — family-specific routing from ARC objects to pipe task batches.

Translates scheduler-level decisions ("should we pipe these conformers?") into
homogeneous ``TaskSpec`` batches and submits them through a ``PipeCoordinator``.

Each ``try_pipe_*`` method returns the **exact subset of items it handled**
(e.g., rotor indices, species labels, conformer indices).  The scheduler
uses this to skip only the work that was actually piped, and immediately
falls back for the remainder.

This module owns the family-specific logic for:
  - choosing level / adapter
  - rejecting incore adapters
  - building TaskSpecs
  - deriving cluster software
  - checking pipe eligibility and submitting

The Scheduler decides *when* to try pipe mode; this module decides *how*.

Note on TSG:
  ``try_pipe_tsg`` is implemented but **not wired** into ``spawn_ts_jobs()``
  because TSG methods are typically few per reaction (3-5 adapters), rarely
  hitting ``min_tasks``.  Wire when workload stats justify it.
"""

from collections import Counter
from typing import TYPE_CHECKING
from collections.abc import Callable

from arc.common import get_logger
from arc.imports import settings
from arc.job.adapters.common import default_incore_adapters
from arc.level import Level

from arc.job.pipe.pipe_run import (
    build_conformer_pipe_tasks,
    build_rotor_scan_1d_tasks,
    build_species_leaf_task,
    build_ts_opt_tasks,
    build_tsg_tasks,
    derive_cluster_software,
)
from arc.job.pipe.pipe_state import TaskSpec

if TYPE_CHECKING:
    from arc.job.pipe.pipe_coordinator import PipeCoordinator
    from arc.reaction import ARCReaction
    from arc.scheduler import Scheduler

logger = get_logger()

pipe_settings = settings['pipe_settings']

class PipePlanner:
    """
    Family-specific pipe routing from ARC objects to pipe task batches.

    Each ``try_pipe_*`` method returns the handled subset so the scheduler
    can fall back only for the remainder.  The generic ``_try_pipe_job``
    captures the repeated routing pattern; individual methods supply the
    task-building callable and family-specific preconditions.

    Args:
        sched: The owning Scheduler instance.
        coordinator: The PipeCoordinator that owns active pipe runs.
    """

    def __init__(self, sched: 'Scheduler', coordinator: 'PipeCoordinator'):
        self.sched = sched
        self.coordinator = coordinator

    @property
    def _memory_mb(self) -> int:
        return int(self.sched.memory * 1024)

    def _level_dict(self, level) -> dict:
        return level.as_dict() if isinstance(level, Level) else Level(repr=level).as_dict()

    # ------------------------------------------------------------------
    # Generic routing helper
    # ------------------------------------------------------------------

    def _try_pipe_job(self,
                      run_id: str,
                      level,
                      job_type: str,
                      build_tasks_fn: Callable[..., list[TaskSpec]],
                      log_msg: str,
                      ) -> bool:
        """
        Generic pipe routing: deduce adapter, reject incore, build tasks,
        check eligibility, derive cluster software, log, and submit.

        Returns ``True`` if the batch was submitted, ``False`` otherwise.
        Family wrappers translate this bool into the appropriate handled-subset
        return value (all-or-nothing for families routed through this helper).
        """
        job_adapter = self.sched.deduce_job_adapter(level=Level(repr=level), job_type=job_type)
        if job_adapter in default_incore_adapters:
            return False
        tasks = build_tasks_fn(job_adapter)
        if not self.coordinator.should_use_pipe(tasks):
            return False
        cs = derive_cluster_software(self.sched.ess_settings, job_adapter)
        logger.info(f'{log_msg} (engine={job_adapter}, cluster={cs}).')
        self.coordinator.submit_pipe_run(run_id, tasks, cluster_software=cs)
        return True

    # ------------------------------------------------------------------
    # Family-specific routing — each returns the handled subset
    # ------------------------------------------------------------------

    def try_pipe_conformers(self, label: str) -> set[int]:
        """
        Route conformer optimization through pipe mode.

        Returns:
            set[int]: Conformer indices that were piped (all or empty).
        """
        level = self.sched.conformer_opt_level
        n_conformers = len(self.sched.species_dict[label].conformers)
        submitted = self._try_pipe_job(
            run_id=f'{label}_conf_opt',
            level=level,
            job_type='conf_opt',
            build_tasks_fn=lambda adapter: build_conformer_pipe_tasks(
                self.sched.species_dict[label], label, 'conf_opt',
                self._level_dict(level), adapter, self._memory_mb),
            log_msg=f'Routing {n_conformers} conformer optimizations for {label} to pipe mode',
        )
        return set(range(n_conformers)) if submitted else set()

    def try_pipe_conf_sp(self, label: str, conformer_indices: list[int]) -> set[int]:
        """
        Route conformer SP jobs through pipe mode for the given candidate indices.

        Args:
            label: The species label.
            conformer_indices: The exact conformer indices to consider for piping.
                Only these indices will be built into tasks; the returned handled
                set is always a subset of this input.

        Returns:
            set[int]: Conformer indices that were piped (all supplied or empty).
        """
        if not conformer_indices:
            return set()
        if not self.sched.job_types.get('conf_sp') or self.sched.conformer_sp_level is None:
            return set()
        if self.sched.conformer_sp_level == self.sched.conformer_opt_level:
            return set()
        level = self.sched.conformer_sp_level
        candidate_set = set(conformer_indices)
        submitted = self._try_pipe_job(
            run_id=f'{label}_conf_sp',
            level=level,
            job_type='conf_sp',
            build_tasks_fn=lambda adapter: build_conformer_pipe_tasks(
                self.sched.species_dict[label], label, 'conf_sp',
                self._level_dict(level), adapter, self._memory_mb,
                conformer_indices=sorted(candidate_set)),
            log_msg=f'Routing {len(candidate_set)} conformer SP jobs for {label} to pipe mode',
        )
        return candidate_set if submitted else set()

    def try_pipe_tsg(self, rxn: 'ARCReaction', methods: list[str]) -> set[str]:
        """
        Route TSG methods through pipe mode, grouped by method.

        TSG is a special case: it loops over methods and may create multiple
        pipe runs, so it does not use ``_try_pipe_job``.

        **Intentionally not wired** into ``Scheduler.spawn_ts_jobs()``.
        This is not an omission.  TSG methods are typically few per reaction
        (3-5 adapters), so per-method counts rarely reach ``min_tasks``.
        Future multi-reaction or global TSG batching could revisit this
        decision if workload statistics show enough same-method TSG tasks
        across reactions to justify pipe-mode submission.

        Args:
            rxn: The reaction whose TS guesses are being generated.
            methods: The exact list of TSG method names to consider.

        Returns:
            set[str]: Method names that were piped (subset of ``methods``).
        """
        ts_label = rxn.ts_label
        method_counts = Counter(methods)
        piped_methods = set()
        for method, count in method_counts.items():
            if count < pipe_settings.get('min_tasks', 10):
                continue
            tasks = build_tsg_tasks(ts_label, method, count, rxn.as_dict(), self._memory_mb)
            if not self.coordinator.should_use_pipe(tasks):
                continue
            cs = derive_cluster_software(self.sched.ess_settings, method)
            logger.info(f'Routing {count} TSG {method} tasks for {ts_label} to pipe mode.')
            self.coordinator.submit_pipe_run(f'{ts_label}_tsg_{method}', tasks, cluster_software=cs)
            piped_methods.add(method)
        return piped_methods

    def try_pipe_ts_opt(self, label: str, xyzs: list[dict], level) -> set[int]:
        """
        Route TS optimization jobs through pipe mode.

        Returns:
            set[int]: TS guess indices that were piped (all or empty).
        """
        submitted = self._try_pipe_job(
            run_id=f'{label}_ts_opt',
            level=level,
            job_type='opt',
            build_tasks_fn=lambda adapter: build_ts_opt_tasks(
                self.sched.species_dict[label], label, xyzs,
                self._level_dict(level), adapter, self._memory_mb),
            log_msg=f'Routing {len(xyzs)} TS opt jobs for {label} to pipe mode',
        )
        return set(range(len(xyzs))) if submitted else set()

    def try_pipe_species_sp(self, labels: list[str]) -> set[str]:
        """
        Batch species SP jobs through pipe mode.

        Returns:
            set[str]: Species labels that were piped (all or empty).
        """
        level = self.sched.sp_level
        submitted = self._try_pipe_job(
            run_id='species_sp_batch',
            level=level,
            job_type='sp',
            build_tasks_fn=lambda adapter: [
                build_species_leaf_task(self.sched.species_dict[lbl], lbl, 'species_sp',
                                        self._level_dict(level), adapter, self._memory_mb)
                for lbl in labels],
            log_msg=f'Routing {len(labels)} species SP jobs to pipe mode',
        )
        return set(labels) if submitted else set()

    def try_pipe_species_freq(self, labels: list[str]) -> set[str]:
        """
        Batch species freq jobs through pipe mode.

        Returns:
            set[str]: Species labels that were piped (all or empty).
        """
        level = self.sched.freq_level
        submitted = self._try_pipe_job(
            run_id='species_freq_batch',
            level=level,
            job_type='freq',
            build_tasks_fn=lambda adapter: [
                build_species_leaf_task(self.sched.species_dict[lbl], lbl, 'species_freq',
                                        self._level_dict(level), adapter, self._memory_mb)
                for lbl in labels],
            log_msg=f'Routing {len(labels)} species freq jobs to pipe mode',
        )
        return set(labels) if submitted else set()

    def try_pipe_irc(self, labels_and_directions: list[tuple[str, str]]) -> set[tuple[str, str]]:
        """
        Batch IRC jobs through pipe mode.

        Returns:
            set[tuple[str, str]]: ``(label, direction)`` pairs that were piped (all or empty).
        """
        level = self.sched.irc_level
        if not level:
            return set()

        def _build_irc_tasks(adapter):
            tasks = []
            for label, direction in labels_and_directions:
                task = build_species_leaf_task(
                    self.sched.species_dict[label], label, 'irc',
                    self._level_dict(level), adapter, self._memory_mb,
                    extra_ingestion={'irc_direction': direction})
                task.task_id = f'{label}_irc_{direction}'
                task.input_fingerprint = f'{label}_irc_{direction}'
                tasks.append(task)
            return tasks

        submitted = self._try_pipe_job(
            run_id='irc_batch',
            level=level,
            job_type='irc',
            build_tasks_fn=_build_irc_tasks,
            log_msg=f'Routing {len(labels_and_directions)} IRC jobs to pipe mode',
        )
        return set(labels_and_directions) if submitted else set()

    def try_pipe_rotor_scans_1d(self, label: str, rotor_indices: list[int]) -> set[int]:
        """
        Batch 1D rotor scan jobs through pipe mode.

        Returns:
            set[int]: Rotor indices that were piped (all or empty).
        """
        level = self.sched.scan_level
        if level is None:
            return set()
        submitted = self._try_pipe_job(
            run_id=f'{label}_scan_1d',
            level=level,
            job_type='scan',
            build_tasks_fn=lambda adapter: build_rotor_scan_1d_tasks(
                self.sched.species_dict[label], label, rotor_indices,
                self._level_dict(level), adapter, self._memory_mb),
            log_msg=f'Routing {len(rotor_indices)} 1D rotor scans for {label} to pipe mode',
        )
        return set(rotor_indices) if submitted else set()
