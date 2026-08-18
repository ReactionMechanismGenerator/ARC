"""Standalone CLI for re-running the TCKDB upload sweep against an existing project.

Usage::

    python -m arc.tckdb.cli /path/to/input.yml
    python -m arc.tckdb.cli input.yml --offline
    python -m arc.tckdb.cli input.yml --upload-mode computed_reaction
    python -m arc.tckdb.cli input.yml --project-directory /elsewhere/proj

The CLI reads the same ``tckdb`` block from ``input.yml`` that ARC.py
parses post-``execute()``, and runs the same
:func:`arc.tckdb.sweep.run_upload_sweep` — so output is identical to
what you'd see at the end of an ARC run, minus any science work.

Project-directory resolution order:
1. ``--project-directory`` if passed
2. ``project_directory:`` value inside ``input.yml`` if present
3. ``dirname(input.yml)`` — matches ``ARC.py``'s default

Why this exists: when an ARC run already wrote ``output/output.yml``
but the upload sweep didn't fire (or fired in the wrong mode), you
shouldn't have to re-execute jobs to push payloads. Output.yml is the
contract; this CLI consumes it directly.
"""

import argparse
import os
import sys

from arc.common import read_yaml_file
from arc.tckdb.config import (
    TCKDBConfig,
    VALID_UPLOAD_MODES,
)
from arc.tckdb.sweep import run_upload_sweep


def parse_args(argv=None):
    """Parse CLI args. ``argv=None`` lets argparse read sys.argv directly."""
    parser = argparse.ArgumentParser(
        prog='python -m arc.tckdb.cli',
        description=(
            'Re-run the TCKDB upload sweep against an existing ARC project '
            "(reads <project>/output/output.yml; doesn't re-execute jobs)."
        ),
    )
    parser.add_argument(
        'input_file',
        help='Path to the ARC input.yml whose tckdb: block configures the upload.',
    )
    parser.add_argument(
        '-p', '--project-directory',
        default=None,
        help=(
            'Override project directory. Defaults to project_directory in '
            'input.yml, falling back to dirname(input.yml).'
        ),
    )
    parser.add_argument(
        '--offline',
        action='store_true',
        help=(
            'Force config.upload=False: write payloads + sidecars to disk '
            'but make no network calls. Useful for previewing what would '
            'be uploaded.'
        ),
    )
    parser.add_argument(
        '--upload-mode',
        default=None,
        choices=sorted(VALID_UPLOAD_MODES),
        help=(
            'Override tckdb.upload_mode from input.yml. Useful when the '
            'original run used the wrong mode (e.g. uploaded conformers '
            "but you wanted reactions) and you don't want to edit the "
            'input file.'
        ),
    )
    return parser.parse_args(argv)


def _resolve_project_directory(args, input_dict):
    """Apply the documented resolution order: --project-directory → input.yml → dirname(input)."""
    if args.project_directory:
        return os.path.abspath(args.project_directory)
    from_input = input_dict.get('project_directory')
    if from_input:
        return os.path.abspath(from_input)
    return os.path.abspath(os.path.dirname(args.input_file))


def _build_config(input_dict, *, offline, upload_mode_override):
    """Parse tckdb config from input_dict, applying CLI overrides.

    Returns ``None`` when no ``tckdb`` block exists or ``enabled: false`` —
    the caller treats this as a hard error (the CLI's whole point is to
    upload, so a no-op config is a misuse).
    """
    tckdb_raw = dict(input_dict.get('tckdb') or {})
    if not tckdb_raw:
        return None
    if upload_mode_override:
        tckdb_raw['upload_mode'] = upload_mode_override
    if offline:
        tckdb_raw['upload'] = False
    return TCKDBConfig.from_dict(tckdb_raw)


def main(argv=None, *, adapter_factory=None):
    """CLI entry point.

    ``adapter_factory`` is a test seam: when None, the real
    ``TCKDBAdapter`` is constructed from the config. Tests pass a stub
    that records calls without touching the network.

    Returns an exit code (0 success, non-zero failure) so callers can
    use ``sys.exit(main())`` cleanly.
    """
    args = parse_args(argv)

    if not os.path.exists(args.input_file):
        print(f'error: input file not found: {args.input_file}', file=sys.stderr)
        return 2

    input_dict = read_yaml_file(path=args.input_file)
    if not isinstance(input_dict, dict):
        print(f'error: {args.input_file} did not parse as a mapping.', file=sys.stderr)
        return 2

    project_directory = _resolve_project_directory(args, input_dict)
    if not os.path.isdir(project_directory):
        print(
            f'error: project directory does not exist: {project_directory}',
            file=sys.stderr,
        )
        return 2

    try:
        cfg = _build_config(
            input_dict, offline=args.offline,
            upload_mode_override=args.upload_mode,
        )
    except ValueError as exc:
        print(f'error: invalid tckdb config: {exc}', file=sys.stderr)
        return 2

    if cfg is None:
        print(
            'error: no tckdb block in input.yml (or enabled: false). '
            'Add a tckdb: block with enabled: true and base_url: ... to use this CLI.',
            file=sys.stderr,
        )
        return 2

    print(f'TCKDB CLI: project={project_directory}')
    print(f'TCKDB CLI: mode={cfg.upload_mode}  base_url={cfg.base_url}  upload={cfg.upload}')

    if adapter_factory is None:
        from arc.tckdb.adapter import TCKDBAdapter
        adapter = TCKDBAdapter(cfg, project_directory=project_directory)
    else:
        adapter = adapter_factory(cfg, project_directory)

    run_upload_sweep(
        adapter=adapter,
        project_directory=project_directory,
        tckdb_config=cfg,
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
