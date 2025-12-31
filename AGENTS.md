# Repository Guidelines

## Project Structure & Module Organization
- `arc/`: Core Python package (job orchestration, species/reaction models, schedulers, molecule utilities, scripts).
- `functional/`: Higher-level regression and workflow tests that exercise real ARC runs.
- `docs/source/`: Sphinx docs and tutorials; keep developer-facing changes reflected here.
- `examples/`: Minimal runnable inputs for contributors to validate new features.
- `devtools/`: Helper scripts invoked by the Makefile for installs, cleaning, and tooling.
- `Dockerfile`, `docker-compose.yml`, `environment.yml`: Reproducible environments; prefer these for fresh setups.

## Build, Test, and Development Commands
- **Always run scripts via the environment**: prefix any command with `conda run -n arc_env` (e.g., `conda run -n arc_env python arc/main.py`, `conda run -n arc_env make test`). This is required to match dependency pinning.
- `pip install -e .` or `make install`: Editable install and external dependencies (make pulls optional toolchains).
- `make compile`: Build the Cython extension `arc.molecule` in-place when touching that module.
- `make test` / `make test-unittests`: Run package tests with coverage (`arc/`).
- `make test-functional`: Run integration tests in `functional/`; slower but closer to production use.
- `make test-all`: Full suite with coverage; run before releasing changes.
- `make clean`: Remove build artifacts via `devtools/clean.sh` and `utilities.py clean`.

## Coding Style & Naming Conventions
- Follow PEP 8 (4-space indent); prefer type hints and short, single-purpose functions.
- Modules and functions use `snake_case`; classes use `CamelCase`; constants are `UPPER_SNAKE`.
- Tests live next to code as `*_test.py`; mirror the module name (`processor_test.py` for `processor.py`).
- Add docstrings for public interfaces and non-obvious chemistry logic; keep comments brief and targeted.

## Testing Guidelines
- Use `pytest`; structure new unit tests under `arc/` and broader workflow checks under `functional/`.
- Name tests `test_*` and group by behavior, not implementation details; prefer fixtures over ad hoc setup.
- Keep tests deterministic (no remote schedulers or long-running quantum jobs); stub I/O where possible.
- When adding features, extend coverage with edge cases (empty inputs, malformed geometries, scheduler failures).

## Commit & Pull Request Guidelines
- Commit messages: short, imperative summaries similar to history (`Fix consideration of no energy TS Guesses`); include scope where helpful.
- PRs: state the problem, the approach, and validation commands (`make test`, `make test-functional`); link issues and update docs/examples when user-facing.
- Include configuration or log snippets when touching job runners or external tools so reviewers can reason about reproducibility.

## Security & Configuration Tips
- Do not commit credentials or cluster host details; rely on environment variables or user config files outside the repo.
- Validate new external tool requirements in `environment.yml` or `devtools/install_*.sh` and mention them in docs.
- Prefer container or conda setups for contributors to reproduce your environment quickly.
