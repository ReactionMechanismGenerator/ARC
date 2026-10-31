"""Tests of scheduler recognition and per-dictionary configuration keys."""

import os
import tempfile
import unittest
from unittest.mock import patch

import arc.common as common
import arc.settings.settings as default_settings
from arc.imports import settings, submit_scripts
from arc.job.adapters.gaussian import GaussianAdapter
from arc.level import Level
from arc.species import ARCSpecies


class TestClusterSoftValues(unittest.TestCase):
    """Scheduler support is independent of configured command dictionaries."""

    def test_supported_set_is_not_empty(self) -> None:
        self.assertEqual(set(common.CANONICAL_CLUSTER_SOFT.values()),
                         {'OGE', 'Slurm', 'PBS', 'HTCondor', 'local'})

    def test_values_and_server_diagnostics(self) -> None:
        for raw, expected in [('pbs', 'PBS'), (' SlUrM ', 'Slurm'), (' sGe ', 'OGE'),
                              ('HTCONDOR', 'HTCondor'), (' Local ', 'local')]:
            with self.subTest(raw=raw):
                self.assertEqual(common.canonicalize_cluster_soft_value(raw), expected)
                with patch.object(common, 'servers', {'test': {'cluster_soft': raw}}):
                    self.assertEqual(common.get_canonical_cluster_soft('test'), expected)
        with self.assertRaisesRegex(ValueError, 'test.*Torque'):
            common.get_canonical_cluster_soft('test', {'test': {'cluster_soft': 'Torque'}})
        for raw in (None, 42, ''):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                common.canonicalize_cluster_soft_value(raw)

    def test_unsupported_scheduler_is_not_enabled_by_an_overlay(self) -> None:
        with self.assertRaisesRegex(ValueError, 'Torque'):
            common.get_cluster_soft_key('Torque', {'Torque': 'qstat'}, 'check_status_command')


class TestClusterSoftConfigurationKeys(unittest.TestCase):
    """Each configuration dictionary resolves its own keys when it is used."""

    def test_each_dictionary_can_use_a_different_spelling(self) -> None:
        for name in ('check_status_command', 'submit_command', 'delete_command', 'submit_filenames',
                     't_max_format', 'list_available_nodes_command'):
            for raw, configured in [('PBS', 'pbs'), (' pbs ', 'PBS'), ('SLURM', 'Slurm'),
                                    ('sge', 'OGE'), ('OGE', 'SGE'), (' SGE ', 'SGE')]:
                with self.subTest(name=name, raw=raw, configured=configured):
                    self.assertEqual(common.get_cluster_soft_key(raw, {configured: 'value'}, name), configured)

    def test_direct_sge_key_precedes_alias_fallback(self) -> None:
        mapping = {'OGE': 'oge command', 'SGE': 'sge command'}
        self.assertEqual(common.get_cluster_soft_key('SGE', mapping, 'submit_command'), 'SGE')
        self.assertEqual(common.get_cluster_soft_key('sge', mapping, 'submit_command'), 'SGE')
        self.assertEqual(common.get_cluster_soft_key('oge', mapping, 'submit_command'), 'OGE')

    def test_exact_duplicate_keys_are_deterministic(self) -> None:
        for keys in [('PBS', 'pbs'), ('pbs', 'PBS')]:
            mapping = dict.fromkeys(keys, 'value')
            self.assertEqual(common.get_cluster_soft_key('PBS', mapping, 'submit_filenames'), 'PBS')
            self.assertEqual(common.get_cluster_soft_key('pbs', mapping, 'submit_filenames'), 'pbs')
            with self.assertRaisesRegex(ValueError, 'Ambiguous.*submit_filenames'):
                common.get_cluster_soft_key('Pbs', mapping, 'submit_filenames')

    def test_missing_entry_names_the_requested_dictionary(self) -> None:
        self.assertEqual(common.get_cluster_soft_key('PBS', {'pbs': 'qstat'}, 'check_status_command'), 'pbs')
        with self.assertRaisesRegex(ValueError, 'No entry in submit_command.*PBS'):
            common.get_cluster_soft_key('PBS', {'Slurm': 'sbatch'}, 'submit_command')

    def test_changed_keys_are_read_at_call_time(self) -> None:
        mapping = {'PBS': 'qstat'}
        self.assertEqual(common.get_cluster_soft_key('pbs', mapping, 'check_status_command'), 'PBS')
        mapping.clear()
        mapping['pbs'] = 'custom qstat'
        self.assertEqual(common.get_cluster_soft_key('PBS', mapping, 'check_status_command'), 'pbs')

    def test_local_execution_templates_are_allowed_explicitly(self) -> None:
        self.assertEqual(common.get_cluster_soft_key(' Local ', {'local': 'shell template'},
                                                    'pipe_submit', allow_local=True), 'local')

    def test_queue_less_server_has_a_specific_error(self) -> None:
        with self.assertRaisesRegex(ValueError, 'local.*no queueing system.*delete_command.*not applicable'):
            common.get_cluster_soft_key(' local ', {}, 'delete_command', "Server 'local'")


class TestQueueAdapterClusterSoft(unittest.TestCase):
    """Exercise the preparation path before a queued Gaussian job is submitted."""

    def test_queue_adapter_prepares_files_with_overlay_keys(self) -> None:
        for raw, command_key in [(' pbs ', 'pbs'), (' Slurm ', 'slurm'), ('SGE', 'SGE')]:
            with self.subTest(raw=raw), tempfile.TemporaryDirectory() as directory:
                server = dict(default_settings.servers['server1'], cluster_soft=raw)
                with patch.dict(settings['servers'], {'server1': server}), \
                        patch.dict(settings['submit_filenames'], {command_key: 'custom-submit.sh'}, clear=True), \
                        patch.dict(settings['t_max_format'], {command_key: 'hours'}, clear=True), \
                        patch.dict(submit_scripts, {'server1': {'gaussian': 'job {name} {t_max} {memory}'}}):
                    job = GaussianAdapter(project='cluster_keys', project_directory=directory,
                                          job_type='opt', level=Level(method='wb97xd', basis='def2tzvp'),
                                          species=ARCSpecies(label='oxygen', xyz='O 0 0 0', multiplicity=3),
                                          server='server1', execution_type='queue', testing=True)
                    self.assertTrue(os.path.isfile(os.path.join(job.local_path, 'custom-submit.sh')))
                    self.assertIn('custom-submit.sh', [entry['file_name'] for entry in job.files_to_upload])
                    with open(os.path.join(job.local_path, 'out.txt'), 'w') as stream:
                        stream.write('scheduler stdout')
                    job._get_additional_job_info()
                    self.assertIn('scheduler stdout', job.additional_job_info)
