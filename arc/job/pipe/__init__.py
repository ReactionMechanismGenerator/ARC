"""
ARC pipe subpackage — distributed HPC execution via job arrays.

Submodules:
  - ``pipe_state``: task/run state primitives, data models, file-level locking
  - ``pipe_run``: PipeRun orchestrator, task builders, ingestion helpers
  - ``pipe_coordinator``: active pipe lifecycle management (eligibility, submission, polling, ingestion)
  - ``pipe_planner``: family-specific routing from ARC objects to pipe task batches
"""
