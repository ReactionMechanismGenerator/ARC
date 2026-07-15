"""Focused scheduler tests for queue-based QST2 completion."""

import unittest
from unittest.mock import MagicMock, patch

from arc.scheduler import Scheduler


class TestQST2SchedulerCompletion(unittest.TestCase):

    def test_ingestion_requires_ess_success_and_is_idempotent(self):
        """An errored queue TSG is reported, never parsed, and the last completion advances once."""
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.running_jobs = {'TS0': ['tsg0', 'tsg1']}
        scheduler.run_conformer_jobs = MagicMock()

        successful_ts = MagicMock()
        successful_rxn = MagicMock()
        successful_ts.job_name = 'tsg0'
        successful_ts.job_adapter = 'qst2'
        successful_ts.local_path_to_output_file = '/tmp/success.log'
        successful_ts.job_status = ['done', {'status': 'done', 'keywords': [], 'error': '', 'line': ''}]
        successful_ts.reactions = [successful_rxn]

        errored_ts = MagicMock()
        errored_rxn = MagicMock()
        errored_ts.job_name = 'tsg1'
        errored_ts.job_adapter = 'qst2'
        errored_ts.local_path_to_output_file = '/tmp/errored.log'
        errored_ts.job_status = [
            'done',
            {'status': 'errored',
             'keywords': ['InternalCoordinateError', 'GL101', 'NoSymm'],
             'error': 'Endpoint interpolation failed in curvilinear coordinates.',
             'line': 'Error termination via Lnk1e in /usr/local/g16/l101.exe'},
        ]
        errored_ts.reactions = [errored_rxn]

        def end_job(job, label, job_name):
            scheduler.running_jobs[label].remove(job_name)
            return job.job_status[0] == 'done'

        scheduler.end_job = MagicMock(side_effect=end_job)
        with patch('arc.scheduler.logger.warning') as warning:
            # Complete the errored job first: it must not be parsed or advance while tsg0 remains.
            scheduler.process_completed_tsg_job(errored_ts, 'TS0', 'tsg1')
            scheduler.process_completed_tsg_job(errored_ts, 'TS0', 'tsg1')
            errored_rxn.ts_species.process_completed_tsg_queue_jobs.assert_not_called()
            scheduler.run_conformer_jobs.assert_not_called()
            self.assertEqual(warning.call_count, 1)
            self.assertIn('GL101', warning.call_args.args[0])
            self.assertIn('Endpoint interpolation failed', warning.call_args.args[0])

        # The successful job is ingested with qst2 provenance; as the final TSG it advances once.
        scheduler.process_completed_tsg_job(successful_ts, 'TS0', 'tsg0')
        scheduler.process_completed_tsg_job(successful_ts, 'TS0', 'tsg0')
        successful_rxn.ts_species.process_completed_tsg_queue_jobs.assert_called_once_with(
            path='/tmp/success.log', method='qst2')
        scheduler.run_conformer_jobs.assert_called_once_with(labels=['TS0'])
        self.assertEqual(scheduler.end_job.call_count, 2)

    def test_success_then_errored_qst2_ingests_once_and_advances_once(self):
        """A successful log finishing before the final errored QST2 job is retained exactly once."""
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.running_jobs = {'TS0': ['tsg0', 'tsg1']}
        scheduler.run_conformer_jobs = MagicMock()

        successful_ts = MagicMock()
        successful_rxn = MagicMock()
        successful_ts.job_name = 'tsg0'
        successful_ts.job_adapter = 'orca_neb'
        successful_ts.local_path_to_output_file = '/tmp/success.log'
        successful_ts.job_status = ['done', {'status': 'done', 'keywords': [], 'error': '', 'line': ''}]
        successful_ts.reactions = [successful_rxn]

        errored_ts = MagicMock()
        errored_rxn = MagicMock()
        errored_ts.job_name = 'tsg1'
        errored_ts.job_adapter = 'qst2'
        errored_ts.local_path_to_output_file = '/tmp/errored.log'
        errored_ts.job_status = [
            'done',
            {'status': 'errored',
             'keywords': ['InternalCoordinateError', 'GL101', 'NoSymm'],
             'error': 'Endpoint interpolation failed in curvilinear coordinates.',
             'line': 'Error termination via Lnk1e in /usr/local/g16/l101.exe'},
        ]
        errored_ts.reactions = [errored_rxn]

        def end_job(job, label, job_name):
            scheduler.running_jobs[label].remove(job_name)
            return job.job_status[0] == 'done'

        scheduler.end_job = MagicMock(side_effect=end_job)
        scheduler.process_completed_tsg_job(successful_ts, 'TS0', 'tsg0')
        scheduler.process_completed_tsg_job(successful_ts, 'TS0', 'tsg0')
        successful_rxn.ts_species.process_completed_tsg_queue_jobs.assert_called_once_with(
            path='/tmp/success.log', method='orca_neb')
        scheduler.run_conformer_jobs.assert_not_called()

        with patch('arc.scheduler.logger.warning') as warning:
            scheduler.process_completed_tsg_job(errored_ts, 'TS0', 'tsg1')
            scheduler.process_completed_tsg_job(errored_ts, 'TS0', 'tsg1')
            self.assertEqual(warning.call_count, 1)
        errored_rxn.ts_species.process_completed_tsg_queue_jobs.assert_not_called()
        successful_rxn.ts_species.process_completed_tsg_queue_jobs.assert_called_once()
        scheduler.run_conformer_jobs.assert_called_once_with(labels=['TS0'])
        self.assertEqual(scheduler.end_job.call_count, 2)

    def test_yaml_tsg_ingestion_does_not_require_ess_done(self):
        """Worker-produced YAML remains ingestible after a successful server termination."""
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.running_jobs = {'TS0': ['tsg0']}
        scheduler.run_conformer_jobs = MagicMock()
        scheduler.end_job = MagicMock(return_value=True)

        yaml_ts = MagicMock()
        yaml_rxn = MagicMock()
        yaml_ts.job_name = 'tsg0'
        yaml_ts.job_adapter = 'gcn'
        yaml_ts.local_path_to_output_file = '/tmp/output.yml'
        yaml_ts.job_status = [
            'done',
            {'status': 'errored', 'keywords': ['Unknown'], 'error': 'Not an ESS log.', 'line': ''},
        ]
        yaml_ts.reactions = [yaml_rxn]

        def end_job(job, label, job_name):
            scheduler.running_jobs[label].remove(job_name)
            return True

        scheduler.end_job.side_effect = end_job
        scheduler.process_completed_tsg_job(yaml_ts, 'TS0', 'tsg0')

        yaml_rxn.ts_species.process_completed_tsg_queue_jobs.assert_called_once_with(
            path='/tmp/output.yml', method='gcn')
        scheduler.run_conformer_jobs.assert_called_once_with(labels=['TS0'])


if __name__ == '__main__':
    unittest.main()
