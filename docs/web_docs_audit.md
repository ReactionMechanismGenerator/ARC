# Web Documentation Audit

This audit was performed while rebuilding the Sphinx documentation on the
`web_docs` branch.

## Major Findings

- `docs/source/index.rst` described ARC as Python 3.14, which is current, but
  the page used an old flat information architecture and did not guide new users
  through installation, execution, examples, and advanced controls.
- `docs/source/installation.rst` mixed current and old guidance, used typos such
  as `input.py` instead of the current `inputs.py`, and buried the important
  distinction between local, SSH, and HPC operation.
- `docs/source/running.rst` linked to examples but the docs page did not clearly
  distinguish top-level YAML examples, notebooks, test YAML fixtures, and
  restart fixtures.
- `docs/source/examples.rst` included a broken Python snippet:
  `sp: True` was not quoted in the `job_types` dictionary.
- `docs/source/advanced.rst` documented some stale names. The code currently
  normalizes `fine_grid` to `fine` and `lennard_jones` to `onedmin`, so new docs
  should prefer `fine` and `onedmin`.
- `docs/source/advanced.rst` had stale solvation guidance. Current `Level`
  fields are `solvation_method` and `solvent`, and support is adapter-dependent
  across Gaussian, ORCA, and xTB.
- `docs/source/advanced.rst` documented pipe mode as if it activated by default
  at 10 tasks. Current settings keep pipe mode disabled unless
  `pipe_settings['enabled']` is set to `True`; the default lease is 1 hour.
- Two YAML examples under `examples/Stationary` used obsolete top-level keys:
  `use_bac` and `scan_rotors`.
- The docs theme was the default Read the Docs theme with no project-specific
  visual treatment. Navigation was serviceable but not organized around the user
  journey.

## Source-of-Truth Checks

- Python target: `environment.yml` and `pyproject.toml` both require Python
  3.14.
- Current version: `arc/common.py` defines `VERSION = '1.1.0'`.
- Job types: `arc/common.py::initialize_job_types` defines current keys and
  aliases.
- API signatures: `arc/main.py::ARC.__init__`,
  `arc/species/species.py::ARCSpecies.__init__`, and
  `arc/reaction/reaction.py::ARCReaction.__init__` define the currently accepted
  Python arguments.
- Build/test commands: the repository `Makefile` defines `make compile`,
  `make test`, `make test-functional`, and external installer targets.
- Example inputs are passed directly to `ARC(**input_dict)`, so unsupported
  top-level keys fail rather than being ignored.

## Rebuild Strategy

- Rewrite the narrative docs from scratch instead of editing stale paragraphs in
  place.
- Keep generated API pages and specialized pages such as TS search, output, and
  Docker in the documentation tree.
- Present local, SSH, and HPC operation as first-class workflows.
- Keep examples compact, valid, and aligned with current class signatures.
- Improve the Sphinx site with a custom stylesheet while staying on the existing
  `sphinx_rtd_theme` dependency.
