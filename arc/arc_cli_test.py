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

    def test_upload_requires_enabled_true_explicitly(self):
        """A ``tckdb`` block that does not say ``enabled: true`` must not upload.

        Uploading publishes the run's full scientific record, so the gate
        defaults to deny: a block present for any other reason (a URL, a dry
        run, a commented-out toggle) must not ship the data by omission.
        """
        module = _load_arc_cli()
        for settings in ({}, {'url': 'https://example.invalid'}, {'enabled': 'yes'}, {'enabled': None}):
            with self.subTest(settings=settings):
                with patch.dict(sys.modules, {'tckdb_arc': None}), \
                        patch.object(module.logger, 'warning') as warning:
                    module.run_tckdb_upload(settings, '/project')
                # Never reached the import, so the missing-package path never fired.
                warning.assert_not_called()

    def test_enabled_upload_logs_its_destination(self):
        """The resolved endpoint is logged before any data leaves the machine."""
        module = _load_arc_cli()
        adapter, config_type, sweep = MagicMock(), MagicMock(), MagicMock()
        modules = {
            'tckdb_arc': MagicMock(),
            'tckdb_arc.adapter': MagicMock(TCKDBAdapter=adapter),
            'tckdb_arc.config': MagicMock(TCKDBConfig=config_type),
            'tckdb_arc.sweep': MagicMock(run_upload_sweep=sweep),
        }
        with patch.dict(sys.modules, modules), patch.object(module.logger, 'info') as info:
            module.run_tckdb_upload({'enabled': True, 'url': 'https://tckdb.example.invalid'}, '/project')
        logged = ' '.join(str(call.args[0]) for call in info.call_args_list)
        self.assertIn('tckdb.example.invalid', logged)

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

    def test_non_mapping_tckdb_entry_is_refused_not_raised(self):
        """``tckdb: true`` in an input file must not end the run with an AttributeError.

        The value under the ``tckdb`` key is whatever the user wrote. A bare
        ``true`` is the obvious mistake, and reaching ``.get`` on it would crash
        a finished multi-day run after its results are already on disk.
        """
        module = _load_arc_cli()
        for settings in (True, 'https://example.invalid', 3, ['enabled']):
            with self.subTest(settings=settings):
                with patch.dict(sys.modules, {'tckdb_arc': None}), \
                        patch.object(module.logger, 'warning') as warning:
                    module.run_tckdb_upload(settings, '/project')
                warning.assert_called_once()
                self.assertIn('not a settings block', warning.call_args.args[0])

    def test_an_upload_failure_does_not_fail_the_finished_run(self):
        """An adapter that raises is logged, not propagated out of ``main``.

        By the time the upload runs, the ARC results are written. Letting a
        ``ConnectionError`` — or an import error inside the adapter — escape
        turns a completed run into a non-zero exit and hides that fact.
        """
        module = _load_arc_cli()
        arc_object = MagicMock(project_directory='/project')
        with patch.object(module, 'ARC', return_value=arc_object), \
                patch.object(module, 'read_yaml_file',
                             return_value={'project': 'p', 'tckdb': {'enabled': True}}), \
                patch.object(module, 'parse_command_line_arguments',
                             return_value=MagicMock(file='/project/input.yml',
                                                    debug=False, quiet=False)), \
                patch.object(module, 'run_tckdb_upload',
                             side_effect=ConnectionError('endpoint refused')), \
                patch.object(module.logger, 'error') as error:
            module.main()
        arc_object.execute.assert_called_once()
        error.assert_called_once()
        self.assertIn('endpoint refused', str(error.call_args))

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
