"""
This module contains functionality to import user settings and fill in default values from ARC's settings.
"""

import os
import sys

import arc.settings.settings as arc_settings
from arc.settings.inputs import input_files
from arc.settings.submit import incore_commands, pipe_submit, submit_scripts


# Common imports where the user can optionally put a modified copy of an ARC file un their ~/.arc folder
home = os.getenv("HOME") or os.path.expanduser("~")
local_arc_path = os.path.join(home, '.arc')

local_arc_settings_path = os.path.join(local_arc_path, 'settings.py')
settings = {key: val for key, val in vars(arc_settings).items() if '__' not in key}
if os.path.isfile(local_arc_settings_path):
    if local_arc_path not in sys.path:
        sys.path.insert(1, local_arc_path)
    import settings as local_settings
    local_settings_dict = {key: val for key, val in vars(local_settings).items() if '__' not in key}
    settings.update(local_settings_dict)
    # Set global_ess_settings to None if using a local settings file (ARC's defaults are dummies)
    settings['global_ess_settings'] = local_settings_dict['global_ess_settings'] or None

local_arc_submit_path = os.path.join(local_arc_path, 'submit.py')
if os.path.isfile(local_arc_submit_path):
    if local_arc_path not in sys.path:
        sys.path.insert(1, local_arc_path)
    try:
        from submit import incore_commands as local_incore_commands
        incore_commands.update(local_incore_commands)
    except ImportError:
        pass
        from submit import pipe_submit as local_pipe_submit
        pipe_submit.update(local_pipe_submit)
    except ImportError:
        pass
        from submit import submit_scripts as local_submit_scripts
        submit_scripts.update(local_submit_scripts)
    except ImportError:
        pass

local_arc_inputs_path = os.path.join(local_arc_path, 'inputs.py')
if os.path.isfile(local_arc_inputs_path):
    if local_arc_path not in sys.path:
        sys.path.insert(1, local_arc_path)
    from inputs import input_files as local_input_files
    input_files.update(local_input_files)
