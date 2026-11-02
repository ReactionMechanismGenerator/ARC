"""
This module contains functionality to import user settings and fill in default values from ARC's settings.
"""

import logging
import os
import sys

import arc.settings.settings as arc_settings
from arc.settings.external_paths import (find_goflow_ckpt,
                                          find_goflow_feat_dict,
                                          find_rits_ckpt,
                                          queue_deferred_warning)
from arc.settings.inputs import input_files
from arc.settings.submit import ase_submit, incore_commands, pipe_submit, submit_scripts

logger = logging.getLogger('arc')


# (parent_key, dependent_key, finder_fn) triples: when a local ~/.arc/settings.py
# overrides parent_key without also overriding dependent_key, dependent_key was
# derived from the OLD parent_key at auto-discovery time and must be re-derived
# from the new one, or it silently keeps pointing at a mismatched checkout/artifact.
_OVERRIDDEN_DEPENDENTS = (
    ('GOFLOW_REPO_PATH', 'GOFLOW_CKPT_PATH', find_goflow_ckpt),
    ('GOFLOW_REPO_PATH', 'GOFLOW_FEAT_DICT_PATH', find_goflow_feat_dict),
    ('RITS_REPO_PATH', 'RITS_CKPT_PATH', find_rits_ckpt),
)


def resolve_overridden_dependents(settings: dict, local_settings_dict: dict) -> None:
    """
    Re-derive dependent settings whose parent was overridden without the dependent itself.

    A local ~/.arc/settings.py may override e.g. GOFLOW_REPO_PATH while leaving
    GOFLOW_CKPT_PATH unset; since arc/settings/settings.py derives GOFLOW_CKPT_PATH
    from GOFLOW_REPO_PATH at auto-discovery time (before the overlay), leaving it
    untouched here would mix the new repo with the old, auto-discovered checkpoint.
    An explicit override of the dependent itself always wins and is left alone.

    Args:
        settings (dict): The merged settings dict, updated in place.
        local_settings_dict (dict): The keys explicitly set by the local
            ~/.arc/settings.py overlay.

    Returns:
        None: `settings` is mutated in place.
    """
    for parent_key, dependent_key, finder in _OVERRIDDEN_DEPENDENTS:
        if parent_key in local_settings_dict and dependent_key not in local_settings_dict:
            settings[dependent_key] = finder(settings[parent_key])
            if settings[dependent_key] is None:
                msg = f'{parent_key} was overridden to "{settings[parent_key]}", but no valid ' \
                      f'{dependent_key} could be re-derived from it; {dependent_key} is unset.'
                logger.warning(msg)
                queue_deferred_warning(msg)


_UNUSABLE_OVERLAYS_REPORTED = set()


def _report_unusable_overlay(path: str, module: str, error: ImportError, what: str = '') -> None:
    """
    Report that a local ~/.arc overlay could not supply a setting, so ARC's default is used.

    An overlay that fails to load leaves ARC running on the repository defaults, which is a
    working configuration and therefore produces no other symptom: the submit script or the
    server definitions the user wrote are simply not the ones in use, and a run goes to the
    wrong cluster or with the wrong resources. Naming the file and the error is what makes that
    visible instead of silent.

    Whether the overlay file itself loaded decides how loud the report is, since the two cases
    mean opposite things. A file that loaded but does not define the name is a partial overlay,
    which is the ordinary way to override one setting and leave the rest alone, and is reported
    at the debug level. A file that did not load at all, most often because something it imports
    is not installed, loses every setting in it; that is reported once at the warning level, and
    queued so it survives the log being initialized later in the run.

    Note that a syntax error in an overlay is not reported here and never reaches this function:
    it is a ``SyntaxError`` rather than an ``ImportError``, and it propagates out of this module
    and stops ARC from starting at all.

    Args:
        path (str): The overlay file that could not be used.
        module (str): The module name the overlay is imported under.
        error (ImportError): The import failure.
        what (str, optional): The name that could not be imported, when the file itself loaded.
    """
    if module in sys.modules:
        logger.debug(f'{path} does not define "{what}", so ARC\'s default is used. '
                     f'Got {type(error).__name__}: {error}')
        return
    if path in _UNUSABLE_OVERLAYS_REPORTED:
        return
    _UNUSABLE_OVERLAYS_REPORTED.add(path)
    msg = f'Could not import {path}, so none of the settings in it are used and ARC\'s defaults ' \
          f'are used instead. Got {type(error).__name__}: {error}'
    logger.warning(msg)
    queue_deferred_warning(msg)


# Common imports where the user can optionally put a modified copy of settings.py or submit.py file under ~/.arc
home = os.getenv("HOME") or os.path.expanduser("~")
local_arc_path = os.path.join(home, '.arc')

local_arc_settings_path = os.path.join(local_arc_path, 'settings.py')
settings = {key: val for key, val in vars(arc_settings).items() if '__' not in key}
if os.path.isfile(local_arc_settings_path):
    local_settings = dict()
    if local_arc_path not in sys.path:
        sys.path.insert(1, local_arc_path)
    try:
        import settings as local_settings
    except ImportError as e:
        _report_unusable_overlay(local_arc_settings_path, 'settings', e)
    if local_settings:
        local_settings_dict = {key: val for key, val in vars(local_settings).items() if '__' not in key}
        settings.update(local_settings_dict)
        resolve_overridden_dependents(settings, local_settings_dict)
        # Set global_ess_settings to None if using a local settings file (ARC's defaults are dummies)
        settings['global_ess_settings'] = local_settings_dict['global_ess_settings'] \
            if 'global_ess_settings' in local_settings_dict and local_settings_dict['global_ess_settings'] else None

local_arc_submit_path = os.path.join(local_arc_path, 'submit.py')
if os.path.isfile(local_arc_submit_path):
    local_incore_commands, local_pipe_submit, local_submit_scripts = dict(), dict(), dict()
    local_ase_submit = dict()
    if local_arc_path not in sys.path:
        sys.path.insert(1, local_arc_path)
    try:
        from submit import incore_commands as local_incore_commands
    except ImportError as e:
        _report_unusable_overlay(local_arc_submit_path, 'submit', e, 'incore_commands')
    try:
        from submit import pipe_submit as local_pipe_submit
    except ImportError as e:
        _report_unusable_overlay(local_arc_submit_path, 'submit', e, 'pipe_submit')
    try:
        from submit import ase_submit as local_ase_submit
    except ImportError as e:
        _report_unusable_overlay(local_arc_submit_path, 'submit', e, 'ase_submit')
    try:
        from submit import submit_scripts as local_submit_scripts
    except ImportError as e:
        _report_unusable_overlay(local_arc_submit_path, 'submit', e, 'submit_scripts')
    if local_incore_commands:
        incore_commands.update(local_incore_commands)
    if local_pipe_submit:
        pipe_submit.update(local_pipe_submit)
    if local_ase_submit:
        ase_submit.update(local_ase_submit)
    if local_submit_scripts:
        submit_scripts.update(local_submit_scripts)

local_arc_inputs_path = os.path.join(local_arc_path, 'inputs.py')
if os.path.isfile(local_arc_inputs_path):
    local_input_files = dict()
    if local_arc_path not in sys.path:
        sys.path.insert(1, local_arc_path)
    try:
        from inputs import input_files as local_input_files
    except ImportError as e:
        _report_unusable_overlay(local_arc_inputs_path, 'inputs', e, 'input_files')
    if local_input_files:
        input_files.update(local_input_files)
