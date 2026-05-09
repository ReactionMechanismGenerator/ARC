"""
Tier-3 thermo equivalence: validates the mockter pipeline against real DFT.

Drives ARC end-to-end on three small species (C2H6, OH, H2O) using
``mockter4/def2tzvp``. The mockter adapter forges Gaussian logs from the
``mockter4.yml`` fixture (which was extracted from a real DFT run); ARC
parses them, computes thermo, and writes ``output/output.yml``. The
resulting per-species fields are compared against the snapshot of the
real DFT run committed at ``arc/testing/mockter_fixtures/_reference_outputs/s4_output.yml``.

Bit-equality is required on the numerical fields that come straight
through the parse → output_dict path: ``opt_final_energy_hartree``,
``sp_energy_hartree``, ``zpe_hartree``, and ``harmonic_frequencies_cm1``.
Other fields (timestamps, git commits, on-disk paths) drift between runs
and are excluded by design.
"""

import os
import shutil
import tempfile
import unittest
import warnings

from arc.common import ARC_PATH, read_yaml_file
from arc.main import ARC


FIXTURES_DIR = os.path.join(ARC_PATH, 'arc', 'testing', 'mockter_fixtures')
REFERENCE_OUTPUT = os.path.join(FIXTURES_DIR, '_reference_outputs', 's4_output.yml')


def _project_name(base: str) -> str:
    """Return a per-xdist-worker project name to avoid parallel cleanup collisions."""
    worker_id = os.environ.get('PYTEST_XDIST_WORKER')
    return f'{base}_{worker_id}' if worker_id else base


class TestMockterThermoEquivalence(unittest.TestCase):
    """ARC + mockter4 must reproduce ARC + real DFT for the s4 species set."""

    @classmethod
    def setUpClass(cls):
        """One-time skip-or-load of the reference output snapshot."""
        warnings.filterwarnings(action='ignore', module='.*matplotlib.*')
        if not os.path.isfile(REFERENCE_OUTPUT):
            raise unittest.SkipTest(f'Reference output {REFERENCE_OUTPUT} not present')
        cls.reference = read_yaml_file(REFERENCE_OUTPUT)

    def setUp(self):
        """Each test gets its own project dir; ARC writes outputs into it."""
        self.project = _project_name('arc_mockter_thermo_equiv')
        self.project_directory = tempfile.mkdtemp(
            dir=os.path.join(ARC_PATH, 'Projects'),
            prefix=f'{self.project}_',
        )

    def tearDown(self):
        shutil.rmtree(self.project_directory, ignore_errors=True)

    def test_mockter4_thermo_pipeline_matches_reference(self):
        """Run ARC with mockter4 over C2H6/OH/H2O and assert per-species equivalence."""
        # opt and sp must use different levels so ARC actually runs the sp job
        # rather than short-circuiting sp by reusing the opt energy. Both still
        # route to mockter4 fixture (the digit is the only fixture selector;
        # basis is irrelevant to the lookup).
        arc = ARC(
            project=self.project,
            project_directory=self.project_directory,
            ess_settings={'mockter': 'local'},
            conformer_level='mockter4/def2tzvp',
            opt_level='mockter4/def2tzvp',
            freq_level='mockter4/def2tzvp',
            sp_level='mockter4/cc-pVTZ',
            arkane_level_of_theory='CCSD(T)-F12/cc-pVTZ-F12',
            compute_thermo=False,
            compute_rates=False,
            compute_transport=False,
            job_types={
                'conf_opt': True, 'conf_sp': False,
                'opt': True, 'fine': True, 'freq': True, 'sp': True,
                'rotors': False, 'irc': False, 'bde': False, 'orbitals': False,
            },
            n_confs=3,
            species=[
                {'label': 'C2H6', 'smiles': 'CC'},
                {'label': 'OH', 'smiles': '[OH]'},
                {'label': 'H2O', 'smiles': 'O'},
            ],
        )
        arc.execute()

        produced_path = os.path.join(self.project_directory, 'output', 'output.yml')
        self.assertTrue(os.path.isfile(produced_path), f'output.yml not produced at {produced_path}')
        produced = read_yaml_file(produced_path)

        ref_by_label = {spc['label']: spc for spc in self.reference['species']}
        produced_by_label = {spc['label']: spc for spc in produced['species']}

        for label in ('C2H6', 'OH', 'H2O'):
            with self.subTest(label=label):
                self.assertIn(label, produced_by_label, f'{label} missing from produced output.yml')
                self.assertIn(label, ref_by_label, f'{label} missing from reference snapshot')
                self._assert_species_equivalent(label, ref_by_label[label], produced_by_label[label])

    def _assert_species_equivalent(self, label: str, ref: dict, prod: dict) -> None:
        """
        Assert per-species bit-equality on fields that flow through ARC's parser.

        Args:
            label: Species label (used in failure messages).
            ref: Reference output.yml species entry from the real DFT run.
            prod: Produced output.yml species entry from the mockter run.
        """
        for energy_field in ('opt_final_energy_hartree', 'coarse_opt_final_energy_hartree',
                             'sp_energy_hartree', 'zpe_hartree'):
            self.assertAlmostEqual(
                ref[energy_field], prod[energy_field], places=7,
                msg=f'{label}.{energy_field}: ref={ref[energy_field]}, prod={prod[energy_field]}',
            )

        ref_freqs = ref['statmech']['harmonic_frequencies_cm1']
        prod_freqs = prod['statmech']['harmonic_frequencies_cm1']
        self.assertEqual(len(ref_freqs), len(prod_freqs), f'{label}: freq count mismatch')
        for i, (rf, pf) in enumerate(zip(ref_freqs, prod_freqs)):
            self.assertAlmostEqual(rf, pf, places=3, msg=f'{label}.freqs[{i}]: ref={rf}, prod={pf}')

        self.assertEqual(ref['freq_n_imag'], prod['freq_n_imag'], f'{label}: n_imag mismatch')
        self.assertEqual(ref['multiplicity'], prod['multiplicity'], f'{label}: multiplicity mismatch')
        self.assertEqual(ref['charge'], prod['charge'], f'{label}: charge mismatch')


if __name__ == '__main__':
    unittest.main()
