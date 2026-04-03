# CLAUDE.md — ARC (Automated Rate Calculator)

This file guides Claude Code when working in the ARC repository.

---

## Project Overview

**ARC** automates electronic structure calculations to produce high-quality thermodynamic
and kinetic data for chemical kinetic modeling. It accepts 2D molecular representations
(SMILES, InChI, RMG adjacency lists) and autonomously executes, monitors, and post-processes
quantum chemistry jobs on remote or local compute servers.

**Primary outputs:** thermodynamic properties (H, S, Cp) and high-pressure limit rate
coefficients.

**Key dependencies:**
- RMG-Py and RMG-database (required, NOT used through API dependencies, RMG-database is read directly as text)
- Arkane (bundled with RMG) for statistical mechanics (running through a  subprocess)
- AutoTST, KinBot, TS-GCN (TS search engines, installed via `make install-all`)


**Docs:** https://reactionmechanismgenerator.github.io/ARC/

---

## Repository Layout

```
ARC/
├── arc/                  # Main package
│   ├── main.py           # ARC entry point / orchestration
│   ├── scheduler.py      # Job scheduling and monitoring loop
│   ├── reaction.py       # ARCReaction objects and TS logic
│   ├── species/          # ARCSpecies, conformers, converters, zmat
│   ├── job/              # Job adapters (local, SSH, troubleshooting)
│   ├── settings/         # settings.py, submit.py, input.py defaults
│   ├── parser.py         # ESS output parsing
│   ├── processor.py      # Post-processing and Arkane interfacing
│   ├── plotter.py        # Visualization utilities
│   ├── rmgdb.py          # RMG database interface
│   ├── common.py         # Shared utilities
│   └── level.py          # Level-of-theory representation
├── arc/tests/            # Unit tests (mirrors arc/ structure)
├── functional/           # Functional / integration tests
├── data/                 # Reference data, frequency scaling factors
├── examples/             # Example input files and notebooks
├── ipython/              # Standalone Jupyter notebooks / tools
├── devtools/             # install_rmg.sh and other dev scripts
├── ARC.py                # CLI entry point
├── utilities.py          # Top-level utility functions
├── Makefile              # Primary task runner (see commands below)
└── environment.yml       # Conda environment spec
```

Prefer working inside `arc/` submodules. Avoid modifying `ARC.py` or `utilities.py`
unless the change is genuinely top-level.

---

## Environment Setup

```bash
# 1. Clone
git clone https://github.com/ReactionMechanismGenerator/ARC.git
cd ARC

# 2. Add to PYTHONPATH (add to ~/.bashrc)
export PYTHONPATH=$PYTHONPATH:~/path/to/ARC/

# 3. Create and activate conda environment
conda env create -f environment.yml
conda activate arc_env

# 4. Install RMG (recommended: packaged release via Make)
make install-rmg

# 5. Install TS search engines (AutoTST, KinBot, TS-GCN)
make install-all

# 6. Configure servers and ESS
# Copy and edit settings files into ~/.arc/
cp arc/settings/settings.py ~/.arc/settings.py
cp arc/settings/submit.py   ~/.arc/submit.py
# Edit ~/.arc/settings.py: define 'servers' dict and 'global_ess_settings'
# Edit ~/.arc/submit.py: add Slurm submit script templates per server/ESS
```

**Active ESS in this group:** Gaussian, Orca, Molpro (all accessed via Slurm)

**Do not modify** `arc/settings/settings.py` or `arc/settings/submit.py` in the repo
directly — put overrides in `~/.arc/` to avoid merge conflicts.

---

## Common Commands

```bash
# Run unit tests
make test

# Run functional/integration tests (requires configured servers)
make test-functional

# Run ARC from an input file
python ARC.py input.yml

# Restart a project
python ARC.py restart.yml

# Useful aliases (add to ~/.bashrc)
alias arce='conda activate arc_env'
alias arc='python ~/path/to/ARC/ARC.py input.yml'
alias arcrestart='python ~/path/to/ARC/ARC.py restart.yml'
```

---

## Code Style & Conventions

- **PEP 8** strictly. No exceptions.
- **Docstrings:** Google style for all public functions, methods, and classes.
- **Type hints:** required on all function signatures (parameters and return types).
- Line length: follow PEP 8, longer lines are acceptable where breaking would harm readability, but keep it rare.

Example function signature:

```python
def compute_rate_coefficient(
    reaction: 'ARCReaction',
    temperature: float,
    method: str = 'arkane',
) -> float:
    """Compute the high-pressure limit rate coefficient at a given temperature.

    Args:
        reaction (ARCReaction): The reaction object with optimized TS.
        temperature (float): Temperature in Kelvin.
        method (str): Statistical mechanics backend. Currently only 'arkane'.

    Returns:
        float: Rate coefficient in cm^3/mol/s (bimolecular) or 1/s (unimolecular).
    """
```

---

## Testing

All new functions **must** have corresponding unit tests.

- Unit tests live in the same tree level as the module being tested.
- Functional tests live in `functional/` and require live server access — do not
  run these autonomously (see guardrails below).
- Avoid using MOCK classes when testing, always prefer real objects.
- When writing tests, start with a trivial check, then a normal case, then a wild-type, then check for extreme cases.
- Run unit tests during development:

```bash
make test
```

- CI runs automatically on every PR via GitHub Actions (`cont_int.yml`).
  A PR should not be merged if CI is failing.

---

## Branching & Contribution Workflow

- Branch off `main` for all new work:
  ```bash
  git checkout -b feature/your-feature-name
  ```
- Open a PR against `main` when ready.
- Do not push directly to `main`.
- Before opening a PR: run `make test`, ensure no regressions, and check coverage
  hasn't dropped significantly.
- Open a GitHub issue before starting non-trivial features to avoid duplicate work
  and align with the [roadmap](https://github.com/ReactionMechanismGenerator/ARC/wiki/Roadmap).

---

## Agent Guardrails — What Claude Must NOT Do

These actions require explicit human authorization every time:

1. **Do not submit ESS jobs to any compute server.** This incurs real HPC cost and
   can interfere with live research runs. Never call scheduler methods, SSH job
   submission functions, or `make test-functional` autonomously.

2. **Do not push to `main`.** All changes go through a PR. Never run
   `git push origin main` or force-push to any protected branch.

3. **Do not modify `~/.arc/settings.py` or `~/.arc/submit.py`.** These files contain
   server credentials, SSH key paths, and cluster configuration. Treat them as
   read-only unless explicitly asked.

4. **Do not hardcode server names, paths, or credentials** anywhere in source files.
   All server-specific config belongs in `settings.py` / `submit.py`.

---

## Domain Notes for AI Assistance

- ARC operates on **ARCSpecies** and **ARCReaction** objects — use these, not raw
  SMILES strings, when writing code that interfaces with ARC's internals.
- RMG-Py and RMG-database are **required runtime dependencies**, not optional.
  Any code that queries reaction families, thermo groups, or adjacency lists must
  go through the RMG interface (`arc/rmgdb.py`), not ad-hoc SMARTS matching.
- The scheduler loop in `scheduler.py` is stateful and long-running. Be careful
  with changes there — prefer adding hooks or handlers over rewriting core logic.
- Troubleshooting logic for crashed ESS jobs lives in `arc/job/trsh.py` — this is
  deliberately separate from job submission logic.
