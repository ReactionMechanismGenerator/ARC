# Repository Guidelines

## Project Structure & Module Organization
- Core package lives in `arc/` (chemistry logic, schedulers, plotting utilities). Tests for these modules sit alongside as `*_test.py`.
- Higher-level integration checks live in `functional/`. Examples and reproducible inputs are in `examples/` and `data/`.
- Developer scripts are under `devtools/`; docs sources live in `docs/`; the CLI entry point is `ARC.py`.
- Build helpers such as the `Makefile` and `utilities.py` are at the repo root; Docker assets are in `dockerfiles/` and `docker-compose.yml`.

## Build, Test, and Development Commands
- `python -m pip install -e .` — editable install for local development.
- `make compile` — build the Cython extension (`arc.molecule`) in-place after dependency setup.
- `make test` — run unit tests with coverage over `arc/`.
- `make test-functional` — run functional/integration tests in `functional/`.
- `make test-all` — run both suites with coverage; default report is `coverage.xml`.
- `make clean` — remove build artifacts; `make check-env` prints Python path/version for debugging.

## Coding Style & Naming Conventions
- Follow PEP 8: 4-space indentation, readable line wraps (~100 cols), and descriptive variable names (species, reactions, jobs).
- Prefer f-strings for logging and user messages; keep logging via the shared `logger` in `arc.common`.
- Tests use `pytest` discovery; name files `*_test.py` and functions `test_*` near the code they cover.
- Keep modules import-safe to avoid circular deps (e.g., `arc/common.py` avoids importing other ARC modules).

## Testing Guidelines
- When running tests or any code, you must activate the conda environment called arc_env
- Primary framework: `pytest` (see `Makefile` targets above). Use `-ra -vv` locally when chasing failures.
- Add unit tests under `arc/` for module-level behavior and functional tests under `functional/` for end-to-end job flows.
- Aim to maintain existing coverage; `make test-all` produces `coverage.xml` for CI/codecov.
- Record any required external programs (quantum chemistry engines, schedulers) in test docstrings or skip markers.

## Commit & Pull Request Guidelines
- Commit messages should be concise and action-oriented (e.g., `Fix species thermo parsing`, `Improve scheduler resubmission logging`).
- Squash noisy WIP commits before raising a PR; keep each commit logically scoped (feature, fix, or refactor).
- PRs should include: a brief summary, linked issues, test results (`make test`/`make test-functional` output), and notes on external requirements (e.g., queue systems, ORCA/Gaussian availability).
- Add screenshots or sample log excerpts when changing plotting, logging, or job orchestration behavior to aid reviewers.

## Environment & Configuration Tips
- Use the provided `environment.yml` or `requirements.txt` to align dependencies; some features rely on external quantum chemistry backends configured via `arc/settings`.
- Before running remote jobs, verify scheduler and credentials in your local settings, and prefer `make check-env` to confirm Python tooling paths.
