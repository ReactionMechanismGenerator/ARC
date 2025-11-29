#!/usr/bin/env python3
# encoding: utf-8

"""
Unit tests for CREST helper utilities within ts heuristics.
"""

import unittest
from unittest.mock import MagicMock, mock_open, patch

from arc.job.adapters.ts import heuristics


class TestCrestHelpers(unittest.TestCase):
    """Tests for CREST helper utilities."""

    def test_get_job_history_respects_scheduler(self):
        """Ensure the history command matches the configured local scheduler."""
        original_cluster = heuristics.settings['servers']['local'].get('cluster_soft')
        heuristics.settings['servers']['local']['cluster_soft'] = 'PBS'
        try:
            mocked_cp = MagicMock(returncode=0, stdout='line1\n', stderr='')
            with patch('subprocess.run', return_value=mocked_cp) as sub_run:
                stdout, stderr = heuristics.get_job_history('999')
            sub_run.assert_called_once()
            self.assertIn('qstat -JH 999', sub_run.call_args[0][0])
            self.assertEqual(stdout, ['line1'])
            self.assertEqual(stderr, [])
        finally:
            heuristics.settings['servers']['local']['cluster_soft'] = original_cluster

    def test_process_completed_jobs_resubmits_missing_best(self):
        """If crest_best.xyz is missing but history shows failure, ensure resubmission happens."""
        crest_jobs = {'1': {'path': '/tmp/crest_job', 'status': 'done', 'resubmitted': False}}
        with patch('arc.job.adapters.ts.heuristics.os.path.exists', side_effect=[False, True]), \
             patch('arc.job.adapters.ts.heuristics.get_job_history',
                   return_value=([" 1 some info F 00:00"], [])), \
             patch('arc.job.adapters.ts.heuristics.submit_job', return_value=('done', '2')) as submit_mock, \
             patch('arc.job.adapters.ts.heuristics.monitor_crest_jobs') as monitor_mock, \
             patch('arc.job.adapters.ts.heuristics.str_to_xyz', return_value={'symbols': ['H'], 'coords': [], 'isotopes': [1]}), \
             patch('builtins.open', mock_open(read_data='H 0 0 0\n')):
            xyzs = heuristics.process_completed_jobs(crest_jobs)

        submit_mock.assert_called_once()  # resubmitted once
        monitor_mock.assert_called_once()
        self.assertEqual(len(xyzs), 1)  # New job yielded one geometry


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
