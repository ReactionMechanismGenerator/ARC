#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests for the arc.job.pipe.queue_limit module
"""

import unittest
from unittest.mock import patch

import arc.job.pipe.queue_limit as queue_limit


PBS_QUEUE_FULL = """Queue: mafat_new_q
    queue_type = Execution
    total_jobs = 9
    state_count = Transit:0 Queued:3 Held:0 Waiting:0 Running:6 Exiting:0 Begun:0
    max_queued = [u:PBS_GENERIC=40]
    max_queued_res.ncpus = [u:PBS_GENERIC=600]
    acl_group_enable = True
    acl_groups = amouyal_prj,arad_prj,
	grinberg-dana_prj
    enabled = True
"""

QSTAT_USER = [
    '',
    'zeus.technion.ac.il:                                                            Req\'d  Req\'d   Elap',
    'Job ID                         Username Queue    Jobname    SessID NDS TSK Memory Time  S Time',
    '------------------------------ -------- -------- ---------- ------ --- --- ------ ----- - -----',
    '100.zeus                       alice    mafat_new_q ts0       1234   1   8   16gb 24:00 R 01:00',
    '101[].zeus                     alice    mafat_new_q pipe_x      --   1   8   16gb   --  B   --',
    '101[1].zeus                    alice    mafat_new_q pipe_x    4321   1   8   16gb   --  R 00:10',
    '102.zeus                       alice    other_q  ts1       1234   1   8   16gb 24:00 R 01:00',
    '103.zeus                       alice    mafat_new_q old       1234   1   8   16gb 24:00 F 01:00',
]


class TestPbsParsing(unittest.TestCase):

    def test_generic_per_user_limit(self):
        self.assertEqual(queue_limit.parse_pbs_per_user_limit(PBS_QUEUE_FULL, 'alice'), 40)

    def test_limit_for_the_user_by_name_wins(self):
        text = 'max_queued = [u:PBS_GENERIC=40]\n    max_queued = [u:alice=25]\n'
        self.assertEqual(queue_limit.parse_pbs_per_user_limit(text, 'alice'), 25)
        self.assertEqual(queue_limit.parse_pbs_per_user_limit(text, 'bob'), 40)

    def test_several_entries_on_one_line(self):
        text = '    max_queued = [u:PBS_GENERIC=40], [u:alice=60], [g:PBS_GENERIC=100]\n'
        self.assertEqual(queue_limit.parse_pbs_per_user_limit(text, 'alice'), 60)

    def test_non_user_and_resource_limits_are_ignored(self):
        text = 'max_queued = [o:PBS_ALL=500]\n    max_queued = [g:PBS_GENERIC=100]\n    max_queued_res.ncpus = [u:PBS_GENERIC=600]\n'
        self.assertIsNone(queue_limit.parse_pbs_per_user_limit(text, 'alice'))

    def test_malformed_text_gives_none(self):
        for text in ('', 'qstat: Unknown queue', 'max_queued = [u:PBS_GENERIC=forty]', 'max_queued = 40'):
            with self.subTest(text=text):
                self.assertIsNone(queue_limit.parse_pbs_per_user_limit(text, 'alice'))

    def test_wrapped_value_is_joined(self):
        text = '    max_queued = [u:PBS_GENERIC=\n\t40]\n'
        self.assertEqual(queue_limit.parse_pbs_per_user_limit(text, 'alice'), 40)

    def test_count_user_jobs_in_queue(self):
        self.assertEqual(queue_limit.count_pbs_user_jobs(QSTAT_USER, 'mafat_new_q'), 3)
        self.assertEqual(queue_limit.count_pbs_user_jobs(QSTAT_USER, 'other_q'), 1)
        self.assertEqual(queue_limit.count_pbs_user_jobs(['garbage', '', 'a b c'], 'mafat_new_q'), 0)


class TestSlurmParsing(unittest.TestCase):

    def test_max_submit_jobs(self):
        self.assertEqual(queue_limit.parse_slurm_max_submit_jobs('\n40\n\n100\n'), 40)
        self.assertIsNone(queue_limit.parse_slurm_max_submit_jobs('\n\n'))
        self.assertIsNone(queue_limit.parse_slurm_max_submit_jobs('sacctmgr: error'))

    def test_max_array_size(self):
        self.assertEqual(queue_limit.parse_slurm_max_array_size('MaxJobCount = 10000\nMaxArraySize = 1001\n'), 1000)
        self.assertIsNone(queue_limit.parse_slurm_max_array_size('nothing here'))


class TestDiscoverJobAllowance(unittest.TestCase):

    def _discover(self, software, outputs, **kwargs):
        with patch('arc.job.pipe.queue_limit.execute_command',
                   side_effect=lambda command: (outputs(command).splitlines(), [])), \
                patch('arc.job.pipe.queue_limit.getpass.getuser', return_value='alice'), \
                patch.dict(queue_limit.settings, {'check_status_command': {
                    'PBS': '/opt/pbs/bin/qstat -u $USER', 'Slurm': '/usr/bin/squeue -u $USER'}}):
            return queue_limit.discover_job_allowance(software, 'mafat_new_q', **kwargs)

    @staticmethod
    def _pbs_outputs(command):
        if ' -Qf ' in command:
            return PBS_QUEUE_FULL
        if command.endswith('-Bf'):
            return 'Server: zeus\n    max_queued = [u:PBS_GENERIC=100]\n'
        return '\n'.join(QSTAT_USER)

    def test_pbs_allowance_is_limit_minus_used(self):
        self.assertEqual(self._discover('pbs', self._pbs_outputs), 37)

    def test_pbs_override_replaces_the_read_limit(self):
        self.assertEqual(self._discover('pbs', self._pbs_outputs, limit_override=10), 7)

    def test_pbs_unparsable_limit_is_unknown(self):
        self.assertIsNone(self._discover('pbs', lambda command: 'garbage'))

    def test_failing_query_is_unknown(self):
        with patch('arc.job.pipe.queue_limit.execute_command', side_effect=OSError('no qstat')), \
                patch.dict(queue_limit.settings, {'check_status_command': {'PBS': '/opt/pbs/bin/qstat -u $USER'}}):
            self.assertIsNone(queue_limit.discover_job_allowance('pbs', 'q'))

    def test_unqueried_scheduler_is_unknown(self):
        self.assertIsNone(self._discover('htcondor', lambda command: ''))

    def test_slurm_allowance(self):
        def outputs(command):
            if 'sacctmgr' in command:
                return '40\n'
            if 'scontrol' in command:
                return 'MaxArraySize = 1001\n'
            return '1\n2\n3\n'
        self.assertEqual(self._discover('slurm', outputs), 37)

    def test_slurm_array_size_limit_applies_without_a_submit_limit(self):
        def outputs(command):
            return 'MaxArraySize = 21\n' if 'scontrol' in command else ''
        self.assertEqual(self._discover('slurm', outputs), 20)


class TestNextArraySize(unittest.TestCase):

    def test_halves_down_to_one(self):
        self.assertEqual([queue_limit.next_array_size(n) for n in (80, 5, 2, 1)], [40, 2, 1, 1])


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
