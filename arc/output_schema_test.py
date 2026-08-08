"""
Schema-conformance tests for the consolidated ``output.yml`` result contract.

``arc/schemas/output_yml_schema.json`` is the mechanical definition of the
contract that ``docs/output_yml_schema.md`` describes in prose. These tests
drive ARC's real writer over real ``ARCSpecies``/``ARCReaction`` objects and
validate the resulting document against that schema, so a field added,
renamed, or retyped in ``arc/output.py`` without a matching schema change
fails here rather than reaching a consumer.

The negative cases assert that the schema rejects documents it must reject:
without them the positive cases could pass against a schema that accepts
anything.
"""

import copy
import json
import os
import shutil
import tempfile
import unittest
from typing import Any, Iterator
from unittest.mock import patch

from jsonschema import Draft202012Validator
from jsonschema.validators import validator_for

from arc.common import ARC_PATH, ARC_TESTING_PATH, read_yaml_file
from arc.constants import E_h_kJmol
from arc.level import Level
from arc.output import EnergyCorrections, write_output_yml
from arc.reaction.reaction import ARCReaction
from arc.species.species import ARCSpecies, ThermoData, TSGuess
from arc.tckdb_evidence import OUTPUT_SCHEMA_VERSION


OUTPUT_SCHEMA_PATH = os.path.join(ARC_PATH, 'arc', 'schemas', 'output_yml_schema.json')
SCAN_LOG = os.path.join(ARC_TESTING_PATH, 'rotor_scans', 'sBuOH.out')
ORCA_SCAN_LOG = os.path.join(ARC_TESTING_PATH, 'rotor_scans', 'orca', 'cc.txt')
RESTART_YML = os.path.join(ARC_TESTING_PATH, 'restart', '2_restart_rate', 'restart.yml')
OPT_LOG = os.path.join(ARC_TESTING_PATH, 'opt', 'iC3H7.out')
FREQ_LOG = os.path.join(ARC_TESTING_PATH, 'freq', 'iC3H7.out')

GAUSSIAN_DECK = """%chk=check.chk
#P opt=(modredundant) wb97xd/def2tzvp

title

0 1
C 0.0 0.0 0.0

D 1 2 3 4 F
B 1 2 1.09 F
"""


def load_output_schema() -> dict:
    """Load the ``output.yml`` JSON Schema from ``arc/schemas``."""
    with open(OUTPUT_SCHEMA_PATH) as handle:
        return json.load(handle)


def petersson_corrections() -> EnergyCorrections:
    """The run-level tables a ``bac_type='p'`` run looks up from Arkane."""
    return EnergyCorrections(
        aec={'C': -37.8, 'H': -0.5, 'O': -74.9},
        bac={'C-H': -0.17, 'C-C': -0.2, 'C-O': -0.3},
        aec_key='CBS-QB3',
        bac_key='CBS-QB3',
    )


def melius_corrections() -> EnergyCorrections:
    """The run-level tables a ``bac_type='m'`` run looks up from Arkane.

    ``arkane.encorr.data.mbac`` stores a dict of dicts per level of theory, not
    a flat per-bond map, and ``arc/scripts/get_qm_corrections.py`` passes that
    nesting through.
    """
    return EnergyCorrections(
        aec={'C': -37.8, 'H': -0.5, 'O': -74.9},
        bac={'atom_corr': {'C': -0.6, 'H': -2.06, 'O': -2.49},
             'bond_corr_length': {'C': 0.056, 'H': 1.078, 'O': 120.004},
             'bond_corr_neighbor': {'C': -0.094, 'H': -0.179, 'O': -0.074},
             'mol_corr': -3.782},
        aec_key='CBS-QB3',
        bac_key='CBS-QB3',
    )


def build_output_document(project_directory: str,
                          species_dict: dict,
                          output_dict: dict,
                          reactions: list | None = None,
                          species_corrections: dict | None = None,
                          corrections: EnergyCorrections | None = None,
                          **writer_kwargs: Any,
                          ) -> dict:
    """Run ARC's real ``write_output_yml`` and return the document it wrote.

    The three helpers that shell out to the RMG conda environment
    (point groups, per-species corrections, and the run-level correction
    tables) are the only things replaced; every field in the returned
    document is built by the production code paths.
    """
    corrections = corrections or petersson_corrections()
    with patch('arc.output._compute_point_groups',
               return_value={label: 'C1' for label in species_dict}), \
            patch('arc.output._compute_species_corrections',
                  return_value=species_corrections or {}), \
            patch('arc.output._get_energy_corrections', return_value=corrections):
        write_output_yml(
            project='schema_test',
            project_directory=project_directory,
            species_dict=species_dict,
            reactions=reactions or [],
            output_dict=output_dict,
            **writer_kwargs,
        )
    return read_yaml_file(os.path.join(project_directory, 'output', 'output.yml'))


def make_rich_species() -> ARCSpecies:
    """A converged species carrying conformers, rotors, thermo, and statmech data.

    ``rotors_dict`` mixes two successful rotors with one ARC rejected, so the
    full document exercises both ``statmech.torsions`` and
    ``statmech.rejected_torsions`` at once.
    """
    spc = ARCSpecies(label='sBuOH', smiles='CCC(C)O')
    spc.initial_xyz = spc.get_xyz()
    spc.final_xyz = spc.get_xyz()
    spc.e_elect = -105236.6
    spc.e0 = -105136.6
    spc.freqs = [1300.0, 1500.0, 3000.0]
    spc.optical_isomers = 1
    spc.external_symmetry = 1
    spc._is_linear = False
    spc.conformers = [spc.get_xyz(), spc.get_xyz()]
    spc.conformer_energies = [0.0, 3.2]
    spc.rotors_dict = {
        0: {'success': True, 'scan': [1, 2, 3, 4], 'pivots': [2, 3], 'symmetry': 3,
            'type': 'HinderedRotor', 'scan_path': SCAN_LOG, 'dimensions': 1,
            'scan_software': 'gaussian'},
        1: {'success': True, 'scan': [2, 3, 4, 5], 'pivots': [3, 4], 'symmetry': 1,
            'type': 'FreeRotor', 'scan_path': '', 'dimensions': 1},
        2: {'success': False, 'scan': [3, 4, 5, 6], 'pivots': [4, 5], 'symmetry': 1,
            'type': 'HinderedRotor', 'scan_path': '', 'dimensions': 1,
            'invalidation_reason': 'rotor set too many (5) times'},
    }
    spc.thermo = ThermoData(
        H298=-292.8, S298=359.4, Tmin=(300.0, 'K'), Tmax=(3000.0, 'K'),
        thermo_points=[{'temperature_k': 300.0, 'cp_j_mol_k': 118.9, 'h_kj_mol': -290.4,
                        's_j_mol_k': 360.1, 'g_kj_mol': -398.4},
                       {'temperature_k': 500.0, 'cp_j_mol_k': 168.2, 'h_kj_mol': -261.7,
                        's_j_mol_k': 432.9, 'g_kj_mol': -478.2}],
        nasa_low={'tmin_k': 300.0, 'tmax_k': 1000.0, 'coeffs': [1.0] * 7},
        nasa_high={'tmin_k': 1000.0, 'tmax_k': 3000.0, 'coeffs': [2.0] * 7},
    )
    return spc


def make_directed_rotor_species() -> ARCSpecies:
    """A converged species carrying a 2D directed rotor beside a 1D rotor.

    A directed ND scan is marked successful by the scheduler and stores ``scan``
    and ``pivots`` as one entry per dimension, so ``_get_torsions`` exports a
    torsion whose index lists are lists of lists.
    """
    spc = ARCSpecies(label='nBuOH', smiles='CCCCO')
    spc.initial_xyz = spc.get_xyz()
    spc.final_xyz = spc.get_xyz()
    spc.e_elect = -105236.6
    spc.e0 = -105136.6
    spc.freqs = [1300.0, 1500.0, 3000.0]
    spc.optical_isomers = 1
    spc.external_symmetry = 1
    spc._is_linear = False
    spc.rotors_dict = {
        0: {'success': True, 'scan': [1, 2, 3, 4], 'pivots': [2, 3], 'symmetry': 3,
            'type': 'HinderedRotor', 'scan_path': SCAN_LOG, 'dimensions': 1,
            'scan_software': 'gaussian'},
        1: {'success': True, 'scan': [[1, 2, 3, 4], [2, 3, 4, 5]],
            'pivots': [[2, 3], [3, 4]], 'symmetry': 1, 'type': 'HinderedRotor',
            'scan_path': '', 'dimensions': 2},
    }
    return spc


def make_transition_state() -> ARCSpecies:
    """A converged TS with a chosen guess, imaginary mode, and IRC results."""
    ts = ARCSpecies(label='TS0', is_ts=True, xyz=make_rich_species().get_xyz(),
                    rxn_label='CH4 + OH <=> CH3 + H2O')
    ts.final_xyz = ts.get_xyz()
    ts.e_elect = -105000.0
    ts.e0 = -104900.0
    ts.freqs = [-1235.4, -18.2, 1300.0, 1500.0]
    ts.optical_isomers = 1
    ts.external_symmetry = 1
    ts._is_linear = False
    ts.chosen_ts = 0
    ts.chosen_ts_method = 'xtb_gsm'
    ts.successful_methods = ['xtb_gsm']
    guess = TSGuess(method='xtb_gsm', xyz=ts.get_xyz())
    guess.index = 0
    guess.method_sources = ['xtb_gsm']
    ts.ts_guesses = [guess]
    return ts


def make_reaction() -> ARCReaction:
    """A reaction with fitted Arkane kinetics and a long description."""
    rxn = ARCReaction(label='CH4 + OH <=> CH3 + H2O', ts_label='TS0')
    rxn.multiplicity = 2
    rxn.family = 'H_Abstraction'
    rxn.long_kinetic_description = 'Fitted by Arkane over 300-3000 K.'
    rxn.kinetics = {
        'A': (1.2e5, 'cm^3/(mol*s)'), 'n': 2.1, 'Ea': (12.3, 'kJ/mol'),
        'Tmin': (300.0, 'K'), 'Tmax': (3000.0, 'K'), 'dA': 1.48, 'dn': 0.05,
        'dEa': 0.29, 'dEa_units': 'kJ/mol', 'n_data_points': 50,
        'tunneling': 'Eckart',
    }
    return rxn


def make_species_corrections() -> dict:
    """The per-species AEC/BAC blocks ``get_species_corrections.py`` produces."""
    return {
        'sBuOH': {
            'aec': {
                'value': -0.0234,
                'value_unit': 'hartree',
                'components': [
                    {'component_kind': 'atom', 'key': 'C', 'multiplicity': 4,
                     'parameter_value': -37.847, 'parameter_unit': 'hartree',
                     'contribution_value': -0.0153},
                    {'component_kind': 'atom', 'key': 'O', 'multiplicity': 1,
                     'parameter_value': -74.9, 'parameter_unit': 'hartree',
                     'contribution_value': -0.0081},
                ],
            },
            'bac': {
                'value': -1.94,
                'value_unit': 'kcal_mol',
                'bac_type': 'p',
                'components': [
                    {'component_kind': 'bond', 'key': 'C-H', 'multiplicity': 9,
                     'parameter_value': -0.1735, 'parameter_unit': 'kcal_mol',
                     'contribution_value': -1.5615},
                    {'component_kind': 'bond', 'key': 'C-C', 'multiplicity': 3,
                     'parameter_value': -0.1262, 'parameter_unit': 'kcal_mol',
                     'contribution_value': -0.3786},
                ],
            },
        },
    }


def make_melius_species_corrections() -> dict:
    """The per-species blocks ``get_species_corrections.py`` produces for Melius.

    Melius BAC has no clean per-bond decomposition, so the helper emits a total
    with no ``components``.
    """
    corrections = make_species_corrections()
    corrections['nBuOH'] = {
        'aec': corrections['sBuOH']['aec'],
        'bac': {'value': -2.31, 'value_unit': 'kcal_mol', 'bac_type': 'm'},
    }
    del corrections['sBuOH']
    return corrections


def render_errors(errors: list) -> list[str]:
    """Flatten validation errors to one ``path: message`` line per leaf cause.

    ``oneOf``/``allOf`` failures carry their real cause in ``context``; without
    recursing into it the report says only "is not valid under any of the given
    schemas". Long messages echo the offending instance in the middle of the
    sentence, so they are elided from the middle rather than the end.
    """
    lines: list[str] = []
    for error in errors:
        if error.context:
            lines.extend(render_errors(error.context))
        else:
            message = error.message
            if len(message) > 300:
                message = f'{message[:180]} ... {message[-120:]}'
            lines.append(f'at {list(error.absolute_path)}: {message}')
    return lines


SCHEMA_APPLICATOR_KEYS = frozenset({'if', 'then', 'else', 'not'})
SCHEMA_LITERAL_KEYS = frozenset({'const', 'enum', 'default', 'examples', 'required', 'dependentRequired'})

OPEN_RECORD_ALLOWLIST = {
    '#/$defs/common_species_fields':
        'A base composed into species_record and ts_record through allOf. Closing it here would '
        'reject the TS-only fields on a ts_record; the two composed records each declare '
        'unevaluatedProperties: false, which is what closes this key set.',
    '#/$defs/common_species_fields/properties/opt_final_settings/oneOf/1':
        'A refinement branch that pins optimization_stage on the referenced '
        'optimization_stage_settings def, which declares the closure itself.',
    '#/$defs/common_species_fields/properties/coarse_opt_final_settings/oneOf/1':
        'The coarse-stage counterpart of the branch above.',
}

OPEN_OBJECT_ALLOWLIST = {
    '#/$defs/level_dict/properties/args':
        "Free-form ESS keyword blocks supplied by the user. The key set is the user's, not "
        "ARC's, so there is nothing for ARC to enumerate.",
}


def walk_subschemas(node: Any, pointer: str = '#') -> Iterator[tuple[str, dict]]:
    """Yield ``(json_pointer, subschema)`` for every subschema reachable from ``node``.

    ``if``/``then``/``else``/``not`` subtrees are skipped: they constrain a record
    declared elsewhere rather than declaring one, so they carry no key set of their
    own. Values of ``const``/``enum`` and the like are skipped because they are data,
    not schemas.
    """
    if isinstance(node, dict):
        yield pointer, node
        for key, value in node.items():
            if key in SCHEMA_APPLICATOR_KEYS or key in SCHEMA_LITERAL_KEYS:
                continue
            if key in ('properties', '$defs') and isinstance(value, dict):
                for name, child in value.items():
                    yield from walk_subschemas(child, f'{pointer}/{key}/{name}')
            elif isinstance(value, (dict, list)):
                yield from walk_subschemas(value, f'{pointer}/{key}')
    elif isinstance(node, list):
        for index, item in enumerate(node):
            yield from walk_subschemas(item, f'{pointer}/{index}')


def record_subschemas(schema: dict) -> dict[str, dict]:
    """Every subschema that declares a fixed key set, keyed by JSON pointer."""
    return {pointer: subschema for pointer, subschema in walk_subschemas(schema)
            if 'properties' in subschema}


def declares_closure(subschema: dict) -> bool:
    """Whether ``subschema`` forbids keys it does not itself declare."""
    return (subschema.get('additionalProperties') is False
            or subschema.get('unevaluatedProperties') is False)


def open_record_pointers(schema: dict) -> set[str]:
    """The pointers of record-shaped subschemas that accept unknown keys."""
    return {pointer for pointer, subschema in record_subschemas(schema).items()
            if not declares_closure(subschema)}


class SchemaAssertionMixin:
    """Assertions shared by the document-validation test cases."""

    schema: dict
    validator: Draft202012Validator

    def assert_valid(self, document: dict) -> None:
        """Fail with every schema violation listed, not just the first."""
        errors = list(self.validator.iter_errors(document))
        if errors:
            rendered = '\n'.join(f'  {line}' for line in sorted(set(render_errors(errors))))
            self.fail(f'Document does not satisfy output_yml_schema.json:\n{rendered}')

    def assert_invalid(self, document: dict, expected_in_report: str) -> None:
        """Assert the document is rejected, by an error naming ``expected_in_report``."""
        errors = list(self.validator.iter_errors(document))
        self.assertTrue(errors, 'Expected the schema to reject this document, but it passed.')
        rendered = '\n'.join(sorted(set(render_errors(errors))))
        self.assertIn(expected_in_report, rendered)


class TestOutputSchemaIsWellFormed(unittest.TestCase):
    """The schema file must itself be a legal JSON Schema."""

    def setUp(self):
        self.schema = load_output_schema()

    def test_schema_is_a_legal_json_schema(self):
        """The meta-schema check catches typos in keywords and broken $refs."""
        validator_class = validator_for(self.schema)
        self.assertIs(validator_class, Draft202012Validator)
        validator_class.check_schema(self.schema)

    def test_schema_version_const_matches_the_producer_constant(self):
        """A version bump in ARC must be accompanied by a schema update."""
        self.assertEqual(self.schema['properties']['schema_version']['const'],
                         OUTPUT_SCHEMA_VERSION)

    def test_every_ref_resolves(self):
        """No $ref may point at a missing $defs entry."""
        defined = set(self.schema['$defs'])
        referenced = set()

        def collect(node):
            if isinstance(node, dict):
                ref = node.get('$ref')
                if isinstance(ref, str) and ref.startswith('#/$defs/'):
                    referenced.add(ref[len('#/$defs/'):])
                for value in node.values():
                    collect(value)
            elif isinstance(node, list):
                for value in node:
                    collect(value)

        collect(self.schema)
        self.assertEqual(referenced - defined, set())


class TestSchemaClosureIsComplete(unittest.TestCase):
    """Every record ARC controls must reject keys the schema does not declare.

    Closure is what turns "a field was added to ``arc/output.py``" into a test
    failure. Asserting it structurally covers all of it at once, and covers
    records added later, which a per-record negative test cannot.
    """

    def setUp(self):
        self.schema = load_output_schema()

    def test_every_record_shaped_subschema_declares_closure(self):
        """The only open records are the ones the allowlist justifies."""
        self.assertEqual(open_record_pointers(self.schema), set(OPEN_RECORD_ALLOWLIST))

    def test_the_closure_check_covers_most_of_the_schema(self):
        """Guard the guard: an assertion over an empty walk would pass vacuously."""
        records = record_subschemas(self.schema)
        self.assertGreaterEqual(len(records) - len(OPEN_RECORD_ALLOWLIST), 25)

    def test_deleting_any_closure_constraint_fails_the_check(self):
        """Prove the check kills every closure mutant, not merely the average one."""
        closed = [pointer for pointer, subschema in record_subschemas(self.schema).items()
                  if declares_closure(subschema)]
        self.assertTrue(closed)
        for pointer in closed:
            with self.subTest(pointer=pointer):
                mutant = copy.deepcopy(self.schema)
                target = record_subschemas(mutant)[pointer]
                target.pop('additionalProperties', None)
                target.pop('unevaluatedProperties', None)
                self.assertIn(pointer, open_record_pointers(mutant))
                self.assertNotEqual(open_record_pointers(mutant), set(OPEN_RECORD_ALLOWLIST))

    def test_every_wide_open_object_is_allowlisted(self):
        """An object typed but otherwise unconstrained must be deliberate."""
        wide_open = set()
        for pointer, subschema in walk_subschemas(self.schema):
            declared_type = subschema.get('type')
            types = declared_type if isinstance(declared_type, list) else [declared_type]
            if ('object' in types and 'properties' not in subschema and '$ref' not in subschema
                    and 'additionalProperties' not in subschema
                    and 'unevaluatedProperties' not in subschema):
                wide_open.add(pointer)
        self.assertEqual(wide_open, set(OPEN_OBJECT_ALLOWLIST))

    def test_every_allowlisted_pointer_still_exists(self):
        """A renamed def must not leave a silently unused exemption behind."""
        pointers = {pointer for pointer, _ in walk_subschemas(self.schema)}
        for allowlist in (OPEN_RECORD_ALLOWLIST, OPEN_OBJECT_ALLOWLIST):
            for pointer, justification in allowlist.items():
                self.assertIn(pointer, pointers)
                self.assertTrue(justification.strip())


class TestRichDocumentValidates(SchemaAssertionMixin, unittest.TestCase):
    """A document built by the real writer, covering the contract's rich shapes."""

    @classmethod
    def setUpClass(cls):
        cls.schema = load_output_schema()
        cls.validator = Draft202012Validator(cls.schema)
        cls.tmp_dir = tempfile.mkdtemp()
        cls.addClassCleanup(shutil.rmtree, cls.tmp_dir, ignore_errors=True)

        calc_dir = os.path.join(cls.tmp_dir, 'calcs', 'Species', 'sBuOH', 'opt')
        os.makedirs(calc_dir, exist_ok=True)
        opt_log = os.path.join(calc_dir, 'output.out')
        shutil.copyfile(OPT_LOG, opt_log)
        with open(os.path.join(calc_dir, 'input.gjf'), 'w') as handle:
            handle.write(GAUSSIAN_DECK)

        species = make_rich_species()
        ts = make_transition_state()
        cls.document = build_output_document(
            project_directory=cls.tmp_dir,
            species_dict={'sBuOH': species, 'TS0': ts},
            output_dict={
                'sBuOH': {'convergence': True,
                          'paths': {'geo': opt_log, 'freq': FREQ_LOG, 'sp': OPT_LOG},
                          'job_types': {'opt': True}},
                'TS0': {'convergence': True,
                        'paths': {'geo': OPT_LOG, 'freq': FREQ_LOG, 'sp': OPT_LOG,
                                  'irc': [OPT_LOG, OPT_LOG],
                                  'irc_directions': ['forward', 'reverse']},
                        'job_types': {'opt': True, 'irc': True}},
            },
            reactions=[make_reaction()],
            species_corrections=make_species_corrections(),
            opt_level=Level(method='wb97xd', basis='def2tzvp', software='gaussian'),
            freq_level=Level(method='wb97xd', basis='def2tzvp', software='gaussian'),
            sp_level=Level(method='dlpno-ccsd(t)', basis='cc-pvtz', software='orca'),
            neb_level=Level(method='wb97x-d3', basis='def2-svp', software='orca'),
            arkane_level_of_theory=Level(method='cbs-qb3', software='gaussian'),
            freq_scale_factor=0.975,
            bac_type='p',
            t0=1700000000.0,
        )

    def test_document_validates(self):
        """The whole document satisfies the schema."""
        self.assert_valid(self.document)

    def test_the_rich_shapes_are_actually_present(self):
        """Guard the guard: a document missing these would validate vacuously."""
        species = self.document['species'][0]
        self.assertEqual(species['label'], 'sBuOH')
        self.assertIn('conformers', species)
        self.assertIn('conformer_energies', species)
        self.assertTrue(species['rotor_scans'])
        self.assertTrue(species['rotor_scans'][0]['result']['samples'])
        self.assertTrue(species['statmech']['torsions'])
        self.assertEqual(species['statmech']['rejected_torsions'], [
            {'rotor_index': 2, 'invalidation_reason': 'rotor set too many (5) times',
             'atom_indices': [3, 4, 5, 6], 'pivot_atoms': [4, 5], 'dimension': 1},
        ])
        self.assertTrue(species['thermo']['thermo_points'])
        self.assertIsNotNone(species['thermo']['nasa_low'])
        self.assertEqual({correction['correction_type'] for correction in species['energy_corrections']},
                         {'atom_energy', 'bond_additivity'})
        self.assertTrue(species['opt_constraints'])
        self.assertIsNotNone(species['sp_spin_diagnostic'])
        self.assertIn('neb_level', self.document)
        self.assertIn('tckdb_evidence', self.document)
        self.assertTrue(self.document['reactions'][0]['kinetics'])

    def test_transition_state_shapes_are_present(self):
        """The TS record carries the TS-only fields the schema requires."""
        ts = self.document['transition_states'][0]
        self.assertTrue(ts['is_ts'])
        self.assertIsNone(ts['thermo'])
        self.assertEqual(ts['freq_n_imag'], 2)
        self.assertEqual(len(ts['imaginary_frequencies_cm1']), 2)
        self.assertTrue(ts['ts_guesses'])
        self.assertEqual(len(ts['irc_logs']), 2)
        self.assertEqual(ts['irc_log_directions'], ['forward', 'reverse'])

    def test_species_record_carries_no_transition_state_fields(self):
        """``unevaluatedProperties: false`` is what enforces this; prove it holds."""
        species = self.document['species'][0]
        for field in ('rxn_label', 'irc_logs', 'ts_guesses', 'neb_log', 'gsm_log'):
            self.assertNotIn(field, species)


class TestOrcaRotorScanValidates(SchemaAssertionMixin, unittest.TestCase):
    """A rotor scan parsed from a real Orca log, not a Gaussian one.

    Every other scan case in this file runs a Gaussian log, and ``output_test.py``
    mocks the parser out for Orca. That left the Orca scan path uncovered end to
    end, which is how an adapter publishing absolute Hartree in a field named
    ``relative_energy_kj_mol`` reached a green suite.
    """

    @classmethod
    def setUpClass(cls):
        cls.schema = load_output_schema()
        cls.validator = Draft202012Validator(cls.schema)
        cls.tmp_dir = tempfile.mkdtemp()
        cls.addClassCleanup(shutil.rmtree, cls.tmp_dir, ignore_errors=True)

        spc = ARCSpecies(label='HONO', smiles='ON=O')
        spc.initial_xyz = spc.get_xyz()
        spc.final_xyz = spc.get_xyz()
        spc.e_elect = -539345.0
        spc.e0 = -539300.0
        spc.freqs = [640.0, 1265.0, 3590.0]
        spc.optical_isomers = 1
        spc.external_symmetry = 1
        spc._is_linear = False
        spc.rotors_dict = {
            0: {'success': True, 'scan': [1, 2, 3, 4], 'pivots': [2, 3], 'symmetry': 1,
                'type': 'HinderedRotor', 'scan_path': ORCA_SCAN_LOG, 'dimensions': 1,
                'scan_software': 'orca'},
        }
        cls.document = build_output_document(
            project_directory=cls.tmp_dir,
            species_dict={'HONO': spc},
            output_dict={'HONO': {'convergence': True, 'paths': {},
                                  'job_types': {'opt': True}}},
            opt_level=Level(method='wb97xd', basis='def2tzvp', software='orca'),
            freq_level=Level(method='wb97xd', basis='def2tzvp', software='orca'),
            sp_level=Level(method='wb97xd', basis='def2tzvp', software='orca'),
        )
        cls.scan_result = cls.document['species'][0]['rotor_scans'][0]['result']

    def test_document_validates(self):
        """The whole document satisfies the schema."""
        self.assert_valid(self.document)

    def test_relative_energies_are_relative_and_in_kj_mol(self):
        """The published curve is zeroed and in kJ/mol, not raw Hartree.

        Orca's log tabulates -205.45 to -205.43 Hartree. Published unconverted,
        every point would read as a large negative 'relative energy in kJ/mol'
        and the 47.5 kJ/mol torsional barrier would be invisible.
        """
        energies = [sample['relative_energy_kj_mol'] for sample in self.scan_result['samples']]
        self.assertEqual(len(energies), 45)
        self.assertAlmostEqual(min(energies), 0.0, places=9)
        self.assertAlmostEqual(max(energies), 47.5363, places=3)

    def test_the_absolute_energies_and_their_zero_reference_are_published(self):
        """Orca reaches the Hartree parser too, so the curve's origin is recoverable."""
        self.assertAlmostEqual(self.scan_result['zero_energy_reference_hartree'],
                               -205.45224799, places=6)
        samples = self.scan_result['samples']
        zero_reference = self.scan_result['zero_energy_reference_hartree']
        for sample in samples:
            self.assertAlmostEqual(
                (sample['electronic_energy_hartree'] - zero_reference) * E_h_kJmol,
                sample['relative_energy_kj_mol'], places=6)

    def test_the_published_angle_is_the_absolute_dihedral_of_each_geometry(self):
        """Every sample's angle is the dihedral of its own geometry, modulo a full turn.

        This is the check TCKDB runs on a deposit, so running it here is the same
        question asked of the same data.
        """
        from arc.species.converter import str_to_xyz
        from arc.species.vectors import calculate_dihedral_angle
        atoms = self.scan_result['coordinate']['atom_indices']
        for sample in self.scan_result['samples']:
            measured = calculate_dihedral_angle(
                coords=str_to_xyz(sample['geometry_xyz']), torsion=atoms, index=1)
            self.assertAlmostEqual(sample['angle_degrees'] % 360.0, measured % 360.0,
                                   places=4)


class TestRestartFixtureDocumentValidates(SchemaAssertionMixin, unittest.TestCase):
    """The real writer over a shipped restart file, with no hand-built state.

    Species reconstructed from ``restart.yml`` carry the shapes a live run
    actually produces — among them a ``conformer_energies`` list that is still
    all ``None`` because the conformer jobs had not reported yet. Hand-built
    fixtures never look like that, which is why this case exists.
    """

    @classmethod
    def setUpClass(cls):
        cls.schema = load_output_schema()
        cls.validator = Draft202012Validator(cls.schema)
        cls.tmp_dir = tempfile.mkdtemp()
        cls.addClassCleanup(shutil.rmtree, cls.tmp_dir, ignore_errors=True)

        cls.restart = read_yaml_file(RESTART_YML)
        species_dict = {entry['label']: ARCSpecies(species_dict=entry)
                        for entry in cls.restart['species']}
        cls.document = build_output_document(
            project_directory=cls.tmp_dir,
            species_dict=species_dict,
            output_dict=cls.restart['output'],
            corrections=EnergyCorrections(aec={}, bac={}, aec_key=None, bac_key=None),
        )

    def test_document_validates(self):
        """The whole document satisfies the schema."""
        self.assert_valid(self.document)

    def test_conformer_energies_carry_the_unreported_entries_they_really_have(self):
        """Guard the guard: this fixture must actually contain null energies."""
        with_conformers = [record for record in self.document['species']
                           if 'conformer_energies' in record]
        self.assertEqual(len(with_conformers), 4)
        for record in with_conformers:
            self.assertEqual(len(record['conformer_energies']), len(record['conformers']))
            self.assertIn(None, record['conformer_energies'])

    def test_the_transition_state_publishes_its_validation_verdicts(self):
        """``ts_checks`` reaches the document from the species' own state."""
        transition_states = self.document['transition_states']
        self.assertEqual(len(transition_states), 1)
        checks = transition_states[0]['ts_checks']
        self.assertEqual({key: checks[key] for key in ('E0', 'e_elect', 'IRC', 'freq', 'NMD')},
                         {'E0': True, 'e_elect': True, 'IRC': True, 'freq': True, 'NMD': True})
        self.assertEqual(checks['warnings'], '')


class TestSparseDocumentValidates(SchemaAssertionMixin, unittest.TestCase):
    """Non-converged and monoatomic species take the all-null branches."""

    @classmethod
    def setUpClass(cls):
        cls.schema = load_output_schema()
        cls.validator = Draft202012Validator(cls.schema)
        cls.tmp_dir = tempfile.mkdtemp()
        cls.addClassCleanup(shutil.rmtree, cls.tmp_dir, ignore_errors=True)

        failed = ARCSpecies(label='failed', smiles='CC')
        argon = ARCSpecies(label='Ar', smiles='[Ar]')
        cls.document = build_output_document(
            project_directory=cls.tmp_dir,
            species_dict={'failed': failed, 'Ar': argon},
            output_dict={
                'failed': {'convergence': False, 'paths': {}, 'job_types': {}},
                'Ar': {'convergence': True, 'paths': {}, 'job_types': {'opt': True}},
            },
        )

    def test_document_validates(self):
        """A run with no levels, no reactions, and no results still validates."""
        self.assert_valid(self.document)

    def test_omitted_and_null_fields_are_as_the_contract_states(self):
        """``neb_level`` is absent; the per-species result fields are null."""
        self.assertNotIn('neb_level', self.document)
        self.assertIsNone(self.document['opt_level'])
        by_label = {entry['label']: entry for entry in self.document['species']}
        self.assertIsNone(by_label['failed']['statmech'])
        self.assertIsNone(by_label['failed']['thermo'])
        self.assertEqual(by_label['failed']['rotor_scans'], [])
        self.assertNotIn('conformers', by_label['failed'])
        self.assertIsNone(by_label['Ar']['statmech'])
        self.assertIsNone(by_label['Ar']['zpe_hartree'])
        self.assertIsNone(by_label['Ar']['freq_n_imag'])


class TestDirectedRotorAndTwoStageOptValidate(SchemaAssertionMixin, unittest.TestCase):
    """Two shapes a real run produces that no other case here reaches.

    A 2D directed rotor exports a torsion whose index lists are lists of lists,
    and a coarse-then-fine opt is the only workflow that populates
    ``opt_final_settings``/``coarse_opt_final_settings`` and the ``coarse_opt_*``
    fields. A nested ``solvation_scheme_level`` rides along on the opt level.
    """

    @classmethod
    def setUpClass(cls):
        cls.schema = load_output_schema()
        cls.validator = Draft202012Validator(cls.schema)
        cls.tmp_dir = tempfile.mkdtemp()
        cls.addClassCleanup(shutil.rmtree, cls.tmp_dir, ignore_errors=True)
        cls.document = build_output_document(
            project_directory=cls.tmp_dir,
            species_dict={'nBuOH': make_directed_rotor_species()},
            output_dict={
                'nBuOH': {'convergence': True,
                          'paths': {'geo': OPT_LOG, 'geo_coarse': OPT_LOG,
                                    'freq': FREQ_LOG, 'sp': OPT_LOG},
                          'job_types': {'opt': True}},
            },
            opt_level=Level(method='wb97xd', basis='def2tzvp', software='gaussian',
                            solvation_method='smd', solvent='water',
                            solvation_scheme_level=Level(method='apfd', basis='6-311+g(2d,p)')),
            arkane_level_of_theory=Level(method='cbs-qb3', software='gaussian'),
            bac_type='p',
        )
        cls.species = cls.document['species'][0]

    def test_document_validates(self):
        """The schema must not reject a run that used a directed rotor."""
        self.assert_valid(self.document)

    def test_the_multi_dimensional_torsion_is_exported_as_lists_of_lists(self):
        """Guard the guard: this case is only meaningful with the ND shape present."""
        torsions = {tuple(torsion['pivot_atoms'][0]) if isinstance(torsion['pivot_atoms'][0], list)
                    else torsion['pivot_atoms'][0]: torsion
                    for torsion in self.species['statmech']['torsions']}
        self.assertEqual(len(self.species['statmech']['torsions']), 2)
        nd_torsion = torsions[(2, 3)]
        self.assertEqual(nd_torsion['atom_indices'], [[1, 2, 3, 4], [2, 3, 4, 5]])
        self.assertEqual(nd_torsion['pivot_atoms'], [[2, 3], [3, 4]])
        self.assertIsNone(nd_torsion['source_scan_key'])

    def test_only_the_one_dimensional_rotor_has_a_scan_record(self):
        """``rotor_scans`` stays 1D-only, so the ND torsion has no scan to point at."""
        self.assertEqual([entry['key'] for entry in self.species['rotor_scans']], ['scan_rotor_0'])
        for entry in self.species['rotor_scans']:
            self.assertEqual(entry['result']['dimension'], 1)

    def test_the_two_stage_optimization_settings_are_present(self):
        """This is the only case that reaches ``optimization_stage_settings``."""
        self.assertEqual(self.species['opt_final_settings'], {'optimization_stage': 'fine'})
        self.assertEqual(self.species['coarse_opt_final_settings'], {'optimization_stage': 'coarse'})
        self.assertIsNotNone(self.species['coarse_opt_log'])
        self.assertIsNotNone(self.species['coarse_opt_input_xyz'])
        self.assertIsNotNone(self.species['coarse_opt_output_xyz'])

    def test_the_nested_solvation_scheme_level_is_a_level_dict(self):
        """``_level_to_dict`` recurses, so the nested value is a full level dict."""
        nested = self.document['opt_level']['solvation_scheme_level']
        self.assertEqual(nested['method'], 'apfd')
        self.assertEqual(nested['basis'], '6-311+g(2d,p)')

    def test_a_bogus_key_in_the_nested_level_is_rejected(self):
        """The nested level is held to the same key set as a top-level one."""
        document = copy.deepcopy(self.document)
        document['opt_level']['solvation_scheme_level']['bogus_nested_key'] = 'x'
        self.assert_invalid(document, 'bogus_nested_key')

    def test_a_torsion_mixing_flat_and_nested_indices_is_rejected(self):
        """Accepting both shapes must not degrade into accepting anything."""
        document = copy.deepcopy(self.document)
        document['species'][0]['statmech']['torsions'][0]['atom_indices'] = [[1, 2, 3, 4], 5]
        self.assert_invalid(document, 'atom_indices')

    def test_a_coarse_stage_on_the_fine_opt_settings_is_rejected(self):
        """The two stage fields are pinned to opposite stages, never swapped."""
        document = copy.deepcopy(self.document)
        document['species'][0]['opt_final_settings'] = {'optimization_stage': 'coarse'}
        self.assert_invalid(document, 'optimization_stage')

    def test_an_unknown_key_in_the_stage_settings_is_rejected(self):
        """``optimization_stage_settings`` is closed like every other record."""
        document = copy.deepcopy(self.document)
        document['species'][0]['coarse_opt_final_settings']['grid'] = 'ultrafine'
        self.assert_invalid(document, 'grid')


class TestOnlyRejectedRotorsValidate(SchemaAssertionMixin, unittest.TestCase):
    """A species where every *decided* rotor was rejected, plus one still pending.

    Before rejected rotors were exported, this species would be
    indistinguishable, downstream, from one with no internal rotors
    whatsoever. One rejected rotor carries a real ``invalidation_reason``;
    another carries the ARC default empty string, which must survive
    verbatim rather than being fabricated into something more descriptive.
    A third rotor is still pending (``success`` is ``None`` -- never
    attempted, or mid-troubleshooting) and must not be published as a
    rejection: ARC has not decided its fate yet.
    """

    @classmethod
    def setUpClass(cls):
        cls.schema = load_output_schema()
        cls.validator = Draft202012Validator(cls.schema)
        cls.tmp_dir = tempfile.mkdtemp()
        cls.addClassCleanup(shutil.rmtree, cls.tmp_dir, ignore_errors=True)
        spc = ARCSpecies(label='allRejected', smiles='CCO')
        spc.initial_xyz = spc.get_xyz()
        spc.final_xyz = spc.get_xyz()
        spc.e_elect = -105236.6
        spc.e0 = -105136.6
        spc.freqs = [1300.0, 1500.0]
        spc.optical_isomers = 1
        spc.external_symmetry = 1
        spc._is_linear = False
        spc.rotors_dict = {
            0: {'success': False, 'scan': [1, 2, 3, 4], 'pivots': [2, 3], 'dimensions': 1,
                'invalidation_reason': 'rotor set too many (5) times'},
            1: {'success': False, 'scan': [2, 3, 4, 5], 'pivots': [3, 4], 'dimensions': 1,
                'invalidation_reason': ''},
            2: {'success': None, 'scan': [3, 4, 5, 6], 'pivots': [4, 5], 'dimensions': 1,
                'invalidation_reason': ''},
        }
        cls.document = build_output_document(
            project_directory=cls.tmp_dir,
            species_dict={'allRejected': spc},
            output_dict={'allRejected': {'convergence': True, 'paths': {}, 'job_types': {'opt': True}}},
        )
        cls.species = cls.document['species'][0]

    def test_document_validates(self):
        """A species with zero successful rotors still produces a valid document."""
        self.assert_valid(self.document)

    def test_no_successful_torsions_but_rejected_rotors_are_carried(self):
        """``torsions`` empties out; ``rejected_torsions`` is where the *rejected* rotors went."""
        self.assertEqual(self.species['statmech']['torsions'], [])
        rejected = {entry['rotor_index']: entry for entry in self.species['statmech']['rejected_torsions']}
        self.assertEqual(len(rejected), 2)
        self.assertEqual(rejected[0]['invalidation_reason'], 'rotor set too many (5) times')
        self.assertEqual(rejected[0]['atom_indices'], [1, 2, 3, 4])
        self.assertEqual(rejected[0]['pivot_atoms'], [2, 3])

    def test_empty_invalidation_reason_is_carried_verbatim(self):
        """A genuine rejection (``success is False``) with no recorded reason still carries the empty string as-is."""
        rejected = {entry['rotor_index']: entry for entry in self.species['statmech']['rejected_torsions']}
        self.assertEqual(rejected[1]['invalidation_reason'], '')

    def test_pending_rotor_is_not_published_as_a_rejection(self):
        """``success is None`` means pending/never-attempted, not rejected -- it must not appear here."""
        rejected_indices = {entry['rotor_index'] for entry in self.species['statmech']['rejected_torsions']}
        self.assertNotIn(2, rejected_indices)


class TestMeliusRunValidates(SchemaAssertionMixin, unittest.TestCase):
    """A ``bac_type='m'`` run, whose BAC table is nested rather than flat."""

    @classmethod
    def setUpClass(cls):
        cls.schema = load_output_schema()
        cls.validator = Draft202012Validator(cls.schema)
        cls.tmp_dir = tempfile.mkdtemp()
        cls.addClassCleanup(shutil.rmtree, cls.tmp_dir, ignore_errors=True)
        cls.document = build_output_document(
            project_directory=cls.tmp_dir,
            species_dict={'nBuOH': make_directed_rotor_species()},
            output_dict={
                'nBuOH': {'convergence': True,
                          'paths': {'geo': OPT_LOG, 'freq': FREQ_LOG, 'sp': OPT_LOG},
                          'job_types': {'opt': True}},
            },
            species_corrections=make_melius_species_corrections(),
            corrections=melius_corrections(),
            arkane_level_of_theory=Level(method='cbs-qb3', software='gaussian'),
            bac_type='m',
        )

    def test_document_validates(self):
        """The nested Melius table is a legal ``bond_additivity_corrections``."""
        self.assert_valid(self.document)

    def test_the_nested_melius_table_reaches_the_document_intact(self):
        """Guard the guard: the run-level table keeps Arkane's own nesting."""
        table = self.document['bond_additivity_corrections']
        self.assertEqual(sorted(table), ['atom_corr', 'bond_corr_length',
                                         'bond_corr_neighbor', 'mol_corr'])
        self.assertEqual(table['atom_corr']['C'], -0.6)
        self.assertIsInstance(table['mol_corr'], float)

    def test_the_atom_energy_corrections_survive_the_melius_run(self):
        """A Melius BAC lookup must not take the AEC table down with it."""
        self.assertEqual(sorted(self.document['atom_energy_corrections']), ['C', 'H', 'O'])
        species = self.document['species'][0]
        models = {correction['model'] for correction in species['energy_corrections']}
        self.assertEqual(models, {'arkane_atom_energy', 'melius'})

    def test_no_parameter_table_is_emitted_for_melius(self):
        """The producer emits ``parameter_table`` for Petersson only."""
        melius = next(correction for correction in self.document['species'][0]['energy_corrections']
                      if correction['model'] == 'melius')
        self.assertNotIn('parameter_table', melius)

    def test_a_flat_table_on_a_melius_run_is_rejected(self):
        """The pre-fix producer flattened the Melius table; the schema now catches it."""
        document = copy.deepcopy(self.document)
        document['bond_additivity_corrections'] = {'C-H': -0.17, 'C-C': -0.2}
        self.assert_invalid(document, 'bond_additivity_corrections')

    def test_a_nested_table_on_a_petersson_run_is_rejected(self):
        """The two table shapes are pinned to their own ``bac_type``."""
        document = copy.deepcopy(self.document)
        document['bac_type'] = 'p'
        self.assert_invalid(document, 'bond_additivity_corrections')

    def test_a_parameter_table_on_a_melius_correction_is_rejected(self):
        """``parameter_table`` is Petersson-only, not merely bond-additivity-only."""
        document = copy.deepcopy(self.document)
        melius = next(correction for correction in document['species'][0]['energy_corrections']
                      if correction['model'] == 'melius')
        melius['parameter_table'] = {'unit': 'kcal_mol', 'values': {'C-H': -0.17}}
        self.assert_invalid(document, "'parameter_table'")


class TestSchemaRejectsInvalidDocuments(SchemaAssertionMixin, unittest.TestCase):
    """The schema must reject the mistakes it exists to catch.

    Each case starts from the rich document that
    :class:`TestRichDocumentValidates` proves valid, applies one defect, and
    asserts the schema catches it. Without these, the positive cases above
    would pass just as happily against a schema of ``{}``.
    """

    @classmethod
    def setUpClass(cls):
        cls.schema = load_output_schema()
        cls.validator = Draft202012Validator(cls.schema)
        cls.tmp_dir = tempfile.mkdtemp()
        cls.addClassCleanup(shutil.rmtree, cls.tmp_dir, ignore_errors=True)
        cls.valid_document = build_output_document(
            project_directory=cls.tmp_dir,
            species_dict={'sBuOH': make_rich_species(), 'TS0': make_transition_state()},
            output_dict={
                'sBuOH': {'convergence': True,
                          'paths': {'geo': OPT_LOG, 'freq': FREQ_LOG, 'sp': OPT_LOG},
                          'job_types': {'opt': True}},
                'TS0': {'convergence': True,
                        'paths': {'geo': OPT_LOG, 'freq': FREQ_LOG, 'sp': OPT_LOG,
                                  'irc': [OPT_LOG, OPT_LOG],
                                  'irc_directions': ['forward', 'reverse']},
                        'job_types': {'opt': True, 'irc': True}},
            },
            reactions=[make_reaction()],
            species_corrections=make_species_corrections(),
            arkane_level_of_theory=Level(method='cbs-qb3', software='gaussian'),
            bac_type='p',
        )

    def setUp(self):
        self.document = copy.deepcopy(self.valid_document)
        self.species = self.document['species'][0]
        self.ts = self.document['transition_states'][0]

    def test_baseline_document_is_valid(self):
        """Every negative case below is only meaningful if the baseline passes."""
        self.assert_valid(self.document)

    def test_zero_based_scan_index_is_rejected(self):
        """``index_base`` is pinned to 1: a 0 would silently shift every atom."""
        self.species['rotor_scans'][0]['result']['coordinate']['index_base'] = 0
        self.assert_invalid(self.document, 'index_base')

    def test_an_unknown_irc_direction_is_rejected(self):
        """The direction vocabulary is closed because a wrong value swaps the reaction.

        ``irc_log_directions`` is index-aligned with ``irc_logs``; a typo here
        makes a consumer read the reverse branch as the forward one, which
        silently exchanges reactants and products.
        """
        self.ts['irc_log_directions'] = ['forward', 'backward']
        self.assert_invalid(self.document, 'backward')

    def test_an_unknown_constraint_unit_is_rejected(self):
        """``target_value_units`` is a closed vocabulary, not free text."""
        self.species['opt_constraints'] = [
            {'coordinate_type': 'distance', 'atom_indices': [1, 2], 'index_base': 1,
             'target_value': 1.09, 'target_value_units': 'Angstroms'},
        ]
        self.assert_invalid(self.document, 'Angstroms')

    def test_a_constraint_without_its_unit_is_rejected(self):
        """A length with no stated unit is exactly the ambiguity this field closes."""
        self.species['opt_constraints'] = [
            {'coordinate_type': 'distance', 'atom_indices': [1, 2], 'index_base': 1,
             'target_value': 1.09},
        ]
        self.assert_invalid(self.document, 'target_value_units')

    def test_a_scan_sample_named_by_the_displacement_field_is_rejected(self):
        """The samples' angle is the absolute dihedral and must be named ``angle_degrees``.

        A consumer keys on that name, so a document that offers the coordinate under
        ``scan_displacement_degrees`` must be refused rather than silently read as
        having no angle at all.
        """
        sample = self.species['rotor_scans'][0]['result']['samples'][0]
        sample['scan_displacement_degrees'] = sample.pop('angle_degrees')
        self.assert_invalid(self.document, 'angle_degrees')

    def test_misspelled_required_key_is_rejected(self):
        """A renamed field fails as both a missing key and an unexpected one."""
        self.species['sp_energy_hartre'] = self.species.pop('sp_energy_hartree')
        self.assert_invalid(self.document, 'sp_energy_hartree')

    def test_unknown_species_field_is_rejected(self):
        """This is the drift detector: a new emitted field must update the schema."""
        self.species['brand_new_field'] = 1.0
        self.assert_invalid(self.document, 'brand_new_field')

    def test_transition_state_field_on_a_species_is_rejected(self):
        """Species records must not carry the TS-only fields."""
        self.species['rxn_label'] = 'A <=> B'
        self.assert_invalid(self.document, 'rxn_label')

    def test_wrong_schema_version_is_rejected(self):
        """The version const is what lets a consumer gate on the contract."""
        self.document['schema_version'] = '1.0'
        self.assert_invalid(self.document, 'schema_version')

    def test_non_dihedral_scan_coordinate_is_rejected(self):
        """Only 1D dihedral rotor scans are emitted under ``rotor_scans``."""
        self.species['rotor_scans'][0]['result']['coordinate']['coordinate_type'] = 'distance'
        self.assert_invalid(self.document, 'coordinate_type')

    def test_non_degree_scan_unit_is_rejected(self):
        """``unit`` is pinned to degree; radians would rescale every angle."""
        self.species['rotor_scans'][0]['result']['coordinate']['unit'] = 'radian'
        self.assert_invalid(self.document, 'unit')

    def test_atom_rigid_rotor_kind_is_rejected(self):
        """``statmech`` is null for monoatomic species, so 'atom' cannot appear."""
        self.species['statmech']['rigid_rotor_kind'] = 'atom'
        self.assert_invalid(self.document, 'rigid_rotor_kind')

    def test_populated_freq_final_settings_is_rejected(self):
        """The contract states this is always null until a real signal exists."""
        self.species['freq_final_settings'] = {'grid': 'ultrafine'}
        self.assert_invalid(self.document, 'freq_final_settings')

    def test_null_neb_level_is_rejected(self):
        """``neb_level`` is omitted entirely, never emitted as null."""
        self.document['neb_level'] = None
        self.assert_invalid(self.document, 'neb_level')

    def test_null_optional_spin_diagnostic_field_is_rejected(self):
        """``s_squared_annihilated`` is omitted when absent, never null."""
        self.species['sp_spin_diagnostic']['s_squared_annihilated'] = None
        self.assert_invalid(self.document, 's_squared_annihilated')

    def test_conformers_without_energies_is_rejected(self):
        """The two conformer keys are emitted together or not at all."""
        del self.species['conformer_energies']
        self.assert_invalid(self.document, 'conformer_energies')

    def test_parameter_table_on_an_atom_energy_correction_is_rejected(self):
        """``parameter_table`` is BAC-only; ``reference_atom_energies`` is AEC-only."""
        aec = next(correction for correction in self.species['energy_corrections']
                   if correction['correction_type'] == 'atom_energy')
        aec['parameter_table'] = {'unit': 'kcal_mol', 'values': {'C-H': -0.17}}
        self.assert_invalid(self.document, "'parameter_table'")

    def test_wrong_bac_model_name_is_rejected(self):
        """Only the Arkane model names ARC emits are legal."""
        bac = next(correction for correction in self.species['energy_corrections']
                   if correction['correction_type'] == 'bond_additivity')
        bac['model'] = 'bac_petersson'
        self.assert_invalid(self.document, 'model')

    def test_missing_thermo_key_is_rejected(self):
        """Thermo keys are always present, so dropping one is a contract break."""
        del self.species['thermo']['s298_j_mol_k']
        self.assert_invalid(self.document, 's298_j_mol_k')

    def test_unknown_top_level_key_is_rejected(self):
        """Top-level drift is caught the same way as per-species drift."""
        self.document['new_top_level_section'] = []
        self.assert_invalid(self.document, 'new_top_level_section')

    def test_dangling_source_scan_key_shape_is_rejected(self):
        """``source_scan_key`` must look like a ``rotor_scans[].key``."""
        self.species['statmech']['torsions'][0]['source_scan_key'] = 'rotor-3'
        self.assert_invalid(self.document, 'source_scan_key')

    def test_ts_guess_not_marked_chosen_is_rejected(self):
        """Only the chosen guess is exported, so ``chosen`` is pinned to true."""
        self.ts['ts_guesses'][0]['chosen'] = False
        self.assert_invalid(self.document, 'chosen')

    def test_unknown_level_dict_key_is_rejected(self):
        """Level dicts are ``Level.as_dict()`` minus two keys; nothing else."""
        self.document['arkane_level_of_theory']['unexpected_attribute'] = 'x'
        self.assert_invalid(self.document, 'unexpected_attribute')

    def test_unknown_kinetics_key_is_rejected(self):
        """The kinetics block is a fixed key set built by ``_rxn_to_dict``."""
        self.document['reactions'][0]['kinetics']['T0'] = 1.0
        self.assert_invalid(self.document, 'T0')

    def test_unknown_transition_state_field_is_rejected(self):
        """TS records are closed by ``unevaluatedProperties``, like species records."""
        self.ts['brand_new_ts_field'] = 1.0
        self.assert_invalid(self.document, 'brand_new_ts_field')

    def test_unknown_reaction_field_is_rejected(self):
        """Reaction records are closed too; drift there is a contract break."""
        self.document['reactions'][0]['brand_new_reaction_field'] = 'x'
        self.assert_invalid(self.document, 'brand_new_reaction_field')

    def test_unknown_correction_total_unit_is_rejected(self):
        """The unit vocabulary is pinned: an unlisted unit is a silent rescale."""
        aec = next(correction for correction in self.species['energy_corrections']
                   if correction['correction_type'] == 'atom_energy')
        aec['total']['unit'] = 'kj_mol'
        self.assert_invalid(self.document, 'unit')

    def test_atom_energy_total_reported_in_kcal_is_rejected(self):
        """AEC totals are Hartree; kcal/mol here is the historical drift-1 class."""
        aec = next(correction for correction in self.species['energy_corrections']
                   if correction['correction_type'] == 'atom_energy')
        aec['total']['unit'] = 'kcal_mol'
        self.assert_invalid(self.document, 'unit')

    def test_bond_additivity_total_reported_in_hartree_is_rejected(self):
        """BAC totals are kcal/mol; Hartree here is the same drift in reverse."""
        bac = next(correction for correction in self.species['energy_corrections']
                   if correction['correction_type'] == 'bond_additivity')
        bac['total']['unit'] = 'hartree'
        self.assert_invalid(self.document, 'unit')

    def test_unknown_component_parameter_unit_is_rejected(self):
        """Per-component units are pinned to the same vocabulary as the totals."""
        aec = next(correction for correction in self.species['energy_corrections']
                   if correction['correction_type'] == 'atom_energy')
        aec['components'][0]['parameter_unit'] = 'kJ/mol'
        self.assert_invalid(self.document, 'parameter_unit')

    def test_atom_component_parameter_reported_in_kcal_is_rejected(self):
        """Atom parameters are Hartree and bond parameters kcal/mol, never swapped."""
        aec = next(correction for correction in self.species['energy_corrections']
                   if correction['correction_type'] == 'atom_energy')
        aec['components'][0]['parameter_unit'] = 'kcal_mol'
        self.assert_invalid(self.document, 'parameter_unit')

    def test_bond_component_parameter_reported_in_hartree_is_rejected(self):
        """The bond half of the same pinning."""
        bac = next(correction for correction in self.species['energy_corrections']
                   if correction['correction_type'] == 'bond_additivity')
        bac['components'][0]['parameter_unit'] = 'hartree'
        self.assert_invalid(self.document, 'parameter_unit')

    def test_conformer_lists_are_index_aligned(self):
        """The alignment the schema cannot express, asserted against the producer."""
        self.assertEqual(len(self.species['conformers']),
                         len(self.species['conformer_energies']))


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
