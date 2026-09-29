"""Tests for the optional standalone adapter hook in ARC.py."""

import importlib.util
import io
import logging
import os
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

from arc.common import ARC_PATH, get_logger, initialize_log


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

    @staticmethod
    def _rendered(mock_logger_method) -> str:
        """Join every message the mocked logger method rendered, arguments applied."""
        messages = []
        for call in mock_logger_method.call_args_list:
            template, arguments = str(call.args[0]), call.args[1:]
            messages.append(template % arguments if arguments else template)
        return ' '.join(messages)

    def _run_enabled_upload(self, module, settings):
        """Run an enabled upload against a fully stubbed adapter; return the INFO text."""
        adapter, config_type, sweep = MagicMock(), MagicMock(), MagicMock()
        modules = {
            'tckdb_arc': MagicMock(),
            'tckdb_arc.adapter': MagicMock(TCKDBAdapter=adapter),
            'tckdb_arc.config': MagicMock(TCKDBConfig=config_type),
            'tckdb_arc.sweep': MagicMock(run_upload_sweep=sweep),
        }
        with patch.dict(sys.modules, modules), patch.object(module.logger, 'info') as info:
            module.run_tckdb_upload(settings, '/project')
        return self._rendered(info)

    def test_enabled_upload_logs_its_destination(self):
        """The resolved endpoint is logged before any data leaves the machine."""
        module = _load_arc_cli()
        logged = self._run_enabled_upload(
            module, {'enabled': True, 'url': 'https://tckdb.example.invalid'})
        self.assertIn('tckdb.example.invalid', logged)

    def test_the_logged_destination_carries_no_credentials(self):
        """A URL's userinfo is a credential and must never reach arc.log.

        ARC's own log is copied into issue reports and shared project
        directories, so a token written into the endpoint would travel with it.
        Only the host, and a port where one is given, are logged.
        """
        module = _load_arc_cli()
        logged = self._run_enabled_upload(
            module,
            {'enabled': True, 'url': 'https://tckdb_user:s3cr3t-token@tckdb.example.invalid:8443/api?key=abc'},
        )
        self.assertIn('tckdb.example.invalid:8443', logged)
        for secret in ('s3cr3t-token', 'tckdb_user', '@', 'key=abc', '/api'):
            self.assertNotIn(secret, logged)

    def test_the_destination_helper_reports_only_the_host(self):
        """Direct coverage of every shape the ``tckdb`` block can state a target in."""
        module = _load_arc_cli()
        cases = (
            ({'url': 'https://user:token@host.example:8443/p?q=1#f'}, 'host.example:8443'),
            ({'url': 'https://host.example'}, 'host.example'),
            ({'url': 'host.example'}, 'host.example'),
            ({'host': 'http://u:p@host.example'}, 'host.example'),
            ({'url': ''}, 'the configured endpoint'),
            ({'url': None}, 'the configured endpoint'),
            ({'url': 3}, 'the configured endpoint'),
            ({}, 'the configured endpoint'),
        )
        for settings, expected in cases:
            with self.subTest(settings=settings):
                self.assertEqual(module._tckdb_log_destination(settings), expected)

    def test_the_missing_package_warning_names_the_canonical_source(self):
        """The adapter is not on PyPI, so the warning must not invite a guessed install.

        ``pip install tckdb-arc`` would fetch an unrelated project; the warning
        names the repository the adapter actually comes from and says so.
        """
        module = _load_arc_cli()
        with patch.dict(sys.modules, {'tckdb_arc': None}), \
                patch.object(module.logger, 'warning') as warning:
            module.run_tckdb_upload({'enabled': True}, '/project')
        logged = self._rendered(warning)
        self.assertIn('https://github.com/calvinp0/tckdb-adapters', logged)
        self.assertIn('not published on PyPI', logged)

    def test_dropping_the_tckdb_block_before_arc_is_announced(self):
        """The block never reaches restart.yml, so a restarted run silently uploads nothing.

        ``ARC`` neither accepts nor records a ``tckdb`` key, which is what keeps
        any credential the block carries out of a file written into the project
        directory. The run log has to say so, or the absence of an upload after
        a restart looks like a failure with no trace.
        """
        module = _load_arc_cli()
        arc_object = MagicMock(project_directory='/project')
        with patch.object(module, 'ARC', return_value=arc_object) as arc_class, \
                patch.object(module, 'read_yaml_file',
                             return_value={'project': 'p', 'tckdb': {'enabled': True}}), \
                patch.object(module, 'parse_command_line_arguments',
                             return_value=MagicMock(file='/project/input.yml',
                                                    debug=False, quiet=False)), \
                patch.object(module, 'run_tckdb_upload'), \
                patch.object(module.logger, 'info') as info:
            module.main()
        self.assertNotIn('tckdb', arc_class.call_args.kwargs)
        self.assertIn('restart.yml', self._rendered(info))

    def test_no_tckdb_block_means_no_announcement(self):
        module = _load_arc_cli()
        arc_object = MagicMock(project_directory='/project')
        with patch.object(module, 'ARC', return_value=arc_object), \
                patch.object(module, 'read_yaml_file', return_value={'project': 'p'}), \
                patch.object(module, 'parse_command_line_arguments',
                             return_value=MagicMock(file='/project/input.yml',
                                                    debug=False, quiet=False)), \
                patch.object(module.logger, 'info') as info:
            module.main()
        self.assertNotIn('restart.yml', self._rendered(info))

    def test_missing_standalone_package_logs_once_and_is_a_noop(self):
        module = _load_arc_cli()
        with patch.dict(sys.modules, {'tckdb_arc': None}), patch.object(module.logger, 'warning') as warning:
            module.run_tckdb_upload({'enabled': True}, '/project')
            module.run_tckdb_upload({'enabled': True}, '/project')
        warning.assert_called_once()
        self.assertIn('continuing without upload', self._rendered(warning))

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
                self.assertIn('not a settings block', self._rendered(warning))

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


READYZ_RETRY = 'TCKDB /readyz not ready (attempt 1/5); retrying in 2.0s.'
READYZ_GAVE_UP = ('TCKDB server was not ready after 5 attempts; payloads were written to '
                  '/project/tckdb_payloads. Re-run the upload once it is up.')


def _emit_a_failed_preflight(**kwargs):
    """Log the records ``tckdb_arc`` logs when its readiness preflight gives up."""
    logging.getLogger('tckdb_arc.adapter').info(READYZ_RETRY)
    logging.getLogger('tckdb_arc').warning(READYZ_GAVE_UP)


class TestTCKDBLogRouting(unittest.TestCase):
    """The ``tckdb_arc`` records reach ``arc.log`` while the upload runs, and only then."""

    def setUp(self):
        arc_logger = get_logger()
        self.addCleanup(self._restore_logger, arc_logger, list(arc_logger.handlers),
                        arc_logger.level, arc_logger.propagate)
        tckdb_logger = logging.getLogger('tckdb_arc')
        self.tckdb_state = (list(tckdb_logger.handlers), tckdb_logger.level, tckdb_logger.propagate)
        self.addCleanup(self._restore_logger, tckdb_logger, *self.tckdb_state)
        self.root_stream = io.StringIO()
        root_handler = logging.StreamHandler(self.root_stream)
        logging.getLogger().addHandler(root_handler)
        self.addCleanup(logging.getLogger().removeHandler, root_handler)
        self.project = tempfile.mkdtemp(prefix='arc_tckdb_log_routing_')
        self.addCleanup(shutil.rmtree, self.project, ignore_errors=True)
        self.log_file = os.path.join(self.project, 'arc.log')
        self.stdout, self.stderr = io.StringIO(), io.StringIO()
        with patch('sys.stdout', self.stdout):
            initialize_log(log_file=self.log_file, project='tckdb_log_routing')

    @staticmethod
    def _restore_logger(named_logger, handlers, level, propagate):
        """Put back a logger's handlers, level and propagation, closing handlers this test opened."""
        for handler in list(named_logger.handlers):
            named_logger.removeHandler(handler)
            if handler not in handlers:
                handler.close()
        for handler in handlers:
            named_logger.addHandler(handler)
        named_logger.setLevel(level)
        named_logger.propagate = propagate

    def _read_log(self) -> str:
        with open(self.log_file, 'r') as f:
            return f.read()

    def _tckdb_logger_state(self):
        tckdb_logger = logging.getLogger('tckdb_arc')
        return list(tckdb_logger.handlers), tckdb_logger.level, tckdb_logger.propagate

    def _upload(self, sweep_side_effect):
        """Run an enabled upload whose stubbed sweep has ``sweep_side_effect``."""
        module = _load_arc_cli()
        modules = {
            'tckdb_arc': MagicMock(),
            'tckdb_arc.adapter': MagicMock(TCKDBAdapter=MagicMock()),
            'tckdb_arc.config': MagicMock(TCKDBConfig=MagicMock()),
            'tckdb_arc.sweep': MagicMock(run_upload_sweep=MagicMock(side_effect=sweep_side_effect)),
        }
        with patch.dict(sys.modules, modules), patch('sys.stderr', self.stderr):
            module.run_tckdb_upload({'enabled': True}, self.project)

    def test_adapter_records_are_written_to_arc_log_once(self):
        self._upload(_emit_a_failed_preflight)
        log = self._read_log()
        self.assertIn(READYZ_RETRY, log)
        self.assertIn(f'Warning: {READYZ_GAVE_UP}', log)
        self.assertEqual(self.stdout.getvalue().count(READYZ_GAVE_UP), 1)
        self.assertNotIn(READYZ_GAVE_UP, self.stderr.getvalue())
        self.assertNotIn(READYZ_GAVE_UP, self.root_stream.getvalue())

    def test_routing_ends_with_the_upload(self):
        self._upload(_emit_a_failed_preflight)
        self.assertEqual(self._tckdb_logger_state(), self.tckdb_state)
        with patch('sys.stderr', io.StringIO()):
            logging.getLogger('tckdb_arc.adapter').warning('logged after the upload returned')
        self.assertNotIn('logged after the upload returned', self._read_log())

    def test_routing_ends_when_the_upload_raises(self):
        def fail(**kwargs):
            _emit_a_failed_preflight()
            raise ConnectionError('endpoint refused')

        with self.assertRaises(ConnectionError):
            self._upload(fail)
        self.assertIn(READYZ_GAVE_UP, self._read_log())
        self.assertEqual(self._tckdb_logger_state(), self.tckdb_state)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
