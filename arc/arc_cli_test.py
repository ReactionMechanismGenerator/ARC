"""Tests for the optional standalone adapter hook in ARC.py."""

import importlib.util
import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

from arc.common import ARC_PATH


def _load_arc_cli():
    spec = importlib.util.spec_from_file_location('arc_cli_for_test', os.path.join(ARC_PATH, 'ARC.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestStandaloneTCKDBHook(unittest.TestCase):

    def test_disabled_block_returns_before_import(self):
        module = _load_arc_cli()
        with patch.dict(sys.modules, {'tckdb_arc': None}), patch.object(module.logger, 'warning') as warning:
            module.run_tckdb_upload({'enabled': False}, '/project')
        warning.assert_not_called()

    def test_missing_standalone_package_logs_once_and_is_a_noop(self):
        module = _load_arc_cli()
        with patch.dict(sys.modules, {'tckdb_arc': None}), patch.object(module.logger, 'warning') as warning:
            module.run_tckdb_upload({'enabled': True}, '/project')
            module.run_tckdb_upload({'enabled': True}, '/project')
        warning.assert_called_once()
        self.assertIn('continuing without upload', warning.call_args.args[0])

    def test_internal_dependency_import_error_is_not_hidden(self):
        module = _load_arc_cli()
        package = types.ModuleType('tckdb_arc')
        package.__path__ = []
        adapter_module = types.ModuleType('tckdb_arc.adapter')
        adapter_module.__getattr__ = lambda name: (_ for _ in ()).throw(
            ModuleNotFoundError("No module named 'missing_dependency'", name='missing_dependency')
        )
        with patch.dict(sys.modules, {
            'tckdb_arc': package,
            'tckdb_arc.adapter': adapter_module,
        }):
            with self.assertRaisesRegex(ModuleNotFoundError, 'missing_dependency'):
                module.run_tckdb_upload({'enabled': True}, '/project')

    def test_enabled_config_dispatches_standalone_sweep(self):
        module = _load_arc_cli()
        config = object()
        config_type = MagicMock()
        config_type.from_dict.return_value = config
        adapter_type = MagicMock()
        adapter = adapter_type.return_value
        sweep = MagicMock()

        package = types.ModuleType('tckdb_arc')
        package.__path__ = []
        adapter_module = types.ModuleType('tckdb_arc.adapter')
        adapter_module.TCKDBAdapter = adapter_type
        config_module = types.ModuleType('tckdb_arc.config')
        config_module.TCKDBConfig = config_type
        sweep_module = types.ModuleType('tckdb_arc.sweep')
        sweep_module.run_upload_sweep = sweep
        with patch.dict(sys.modules, {
            'tckdb_arc': package,
            'tckdb_arc.adapter': adapter_module,
            'tckdb_arc.config': config_module,
            'tckdb_arc.sweep': sweep_module,
        }):
            module.run_tckdb_upload({'enabled': True}, '/project')

        config_type.from_dict.assert_called_once_with({'enabled': True})
        adapter_type.assert_called_once_with(config, project_directory='/project')
        sweep.assert_called_once_with(
            adapter=adapter,
            project_directory='/project',
            tckdb_config=config,
        )


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
