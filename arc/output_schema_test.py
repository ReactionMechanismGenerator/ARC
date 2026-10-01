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
from arc.output import ArkaneProvenance, EnergyCorrections, write_output_yml
from arc.reaction.reaction import ARCReaction
from arc.species.species import ARCSpecies, ThermoData, TSGuess
from arc.parser_evidence import OUTPUT_SCHEMA_VERSION


OUTPUT_SCHEMA_PATH = os.path.join(ARC_PATH, 'arc', 'schemas', 'output_yml_schema.json')
SCAN_LOG = os.path.join(ARC_TESTING_PATH, 'rotor_scans', 'sBuOH.out')
ORCA_SCAN_LOG = os.path.join(ARC_TESTING_PATH, 'rotor_scans', 'orca', 'cc.txt')
RESTART_YML = os.path.join(ARC_TESTING_PATH, 'restart', '2_restart_rate', 'restart.yml')
OPT_LOG = os.path.join(ARC_TESTING_PATH, 'opt', 'iC3H7.out')
FREQ_LOG = os.path.join(ARC_TESTING_PATH, 'freq', 'iC3H7.out')
ARKANE_PROVENANCE = ArkaneProvenance('3.3.0', '6b1368de6c19204c7ce4fda6fbecb05da4a0fe0e')
RMG_DATABASE_IDENTITY = {'path_kind': 'package', 'git_commit': None, 'version': '4.0.0',
                         'quantum_corrections_path': '/prefix/share/rmgdatabase/input/quantum_corrections/data.py',
                         'quantum_corrections_sha256': 'a' * 64, 'matches_arc_rmg_db_path': True}

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

    Five helpers are replaced: the three that shell out to the RMG conda environment
    (point groups, per-species corrections, and the run-level correction tables), the
    Arkane provenance resolver, which otherwise reads whatever ``RMG_PATH`` the
    machine running the tests happens to export and yields ``ARKANE_PROVENANCE``
    instead, and the RMG database identity, which reads ``RMG_DB_PATH`` and yields
    ``RMG_DATABASE_IDENTITY``. Every other field in the returned document is built by the production
    code paths.
    """
    corrections = corrections or petersson_corrections()
    with patch('arc.output._compute_point_groups',
               return_value={label: 'C1' for label in species_dict}), \
            patch('arc.output._compute_species_corrections',
                  return_value=species_corrections or {}), \
            patch('arc.output._get_arkane_provenance', return_value=ARKANE_PROVENANCE), \
            patch('arc.output._get_rmg_database_identity', return_value=dict(RMG_DATABASE_IDENTITY)), \
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
    spc.conformer_energies = [-105236.6, -105233.4]
    spc.conformer_levels = [{'method': 'wb97xd', 'basis': 'def2-svp', 'software': 'gaussian'}] * 2
    spc.conformer_energy_sources = [{'kind': 'electronic_kj_mol',
                                     'level': {'method': 'dlpno-ccsd(t)', 'basis': 'cc-pvtz', 'software': 'orca'}}] * 2
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
    spc.thermo.atom_corrections_applied = True
    spc.thermo.bond_corrections_applied = True
    spc.thermo.atom_corrections_level = Level(method='cbs-qb3', software='gaussian')
    spc.e0_atom_corrections_applied = True
    spc.e0_bond_corrections_applied = True
    spc.arkane_rotor_modes = ['HinderedRotor', 'FreeRotor']
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
    ts.e0_atom_corrections_applied = True
    ts.e0_bond_corrections_applied = False
    ts.arkane_rotor_modes = []
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
        'tunneling': 'Eckart', 'atom_corrections_applied': True,
        'comment': 'Fitted to 50 data points; dA = *|/ 1.48', 'ts_validation': None,
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
    '#/$defs/species_record/properties/levels':
        'A refinement of common_species_fields.levels that pins irc to null on a non-TS record; the '
        'levels object itself is closed where it is declared.',
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
                                  'irc_directions': ['forward', 'reverse'],
                                  'irc_levels': [{'method': 'wb97xd', 'basis': 'def2svp', 'software': 'gaussian'},
                                                 {'method': 'b3lyp', 'basis': '6-31g', 'software': 'gaussian'}]},
                        'job_types': {'opt': True, 'irc': True}},
            },
            reactions=[make_reaction()],
            species_corrections=make_species_corrections(),
            opt_level=Level(method='wb97xd', basis='def2tzvp', software='gaussian'),
            freq_level=Level(method='wb97xd', basis='def2tzvp', software='gaussian'),
            sp_level=Level(method='dlpno-ccsd(t)', basis='cc-pvtz', software='orca'),
            neb_level=Level(method='wb97x-d3', basis='def2-svp', software='orca'),
            scan_level=Level(method='b3lyp', basis='6-31g', software='gaussian'),
            irc_level=Level(method='wb97xd', basis='def2svp', software='gaussian'),
            conformer_opt_level=Level(method='wb97xd', basis='def2svp', software='gaussian'),
            conformer_sp_level=Level(method='dlpno-ccsd(t)', basis='cc-pvtz', software='orca'),
            ts_guess_level=Level(method='pm7', software='mopac'),
            arkane_level_of_theory=Level(method='cbs-qb3', software='gaussian'),
            freq_scale_factor=0.975,
            bac_type='p',
            t0=1700000000.0,
        )

    def test_document_validates(self):
        """The whole document satisfies the schema."""
        self.assert_valid(self.document)

    def test_the_latent_exports_are_present_with_the_producer_values(self):
        """The writer's own values for the T1, isotope, route, property, reversibility and kinetics keys."""
        species = self.document['species'][0]
        n_atoms = len(species['xyz'].splitlines())
        self.assertEqual(len(species['xyz_isotopes']), n_atoms)
        self.assertEqual(len(species['opt_input_xyz_isotopes']), n_atoms)
        self.assertEqual(species['conformers_isotopes'], [species['xyz_isotopes']] * len(species['conformers']))
        self.assertIsNone(species['coarse_opt_input_xyz_isotopes'])
        self.assertIsNone(species['coarse_opt_output_xyz_isotopes'])
        self.assertEqual(species['opt_route'], '#P opt=(calcfc) guess=mix uhf/3-21g IOp(2/9=2000) scf=xqc')
        self.assertIn(' freq ', species['freq_route'])
        self.assertAlmostEqual(species['opt_dipole_moment_debye'], 0.2355)
        self.assertEqual(species['opt_dipole_moment_density'], 'SCF')
        self.assertIsNone(species['sp_t1_diagnostic'])
        sample = species['rotor_scans'][0]['result']['samples'][0]
        self.assertIn('geometry_isotopes', sample)
        ts = self.document['transition_states'][0]
        self.assertEqual(len(ts['irc_log_routes']), len(ts['irc_logs']))
        reaction = self.document['reactions'][0]
        self.assertIs(reaction['reversible'], True)
        self.assertEqual(reaction['kinetics']['comment'], 'Fitted to 50 data points; dA = *|/ 1.48')
        self.assertIsNone(reaction['kinetics']['ts_validation'])

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
        self.assertIs(species['thermo']['atom_corrections_applied'], True)
        self.assertIs(species['thermo']['bond_corrections_applied'], True)
        self.assertEqual(species['thermo']['atom_corrections_level']['method'], 'cbs-qb3')
        self.assertEqual({correction['correction_type'] for correction in species['energy_corrections']},
                         {'atom_energy', 'bond_additivity'})
        self.assertTrue(species['opt_constraints'])
        self.assertIsNotNone(species['sp_spin_diagnostic'])
        self.assertIn('neb_level', self.document)
        for key in ('scan_level', 'irc_level', 'conformer_opt_level', 'conformer_sp_level', 'ts_guess_level'):
            self.assertIsNotNone(self.document[key])
        self.assertEqual(species['conformer_levels'][0]['method'], 'wb97xd')
        self.assertEqual(species['conformer_energy_kind'], 'electronic_kj_mol')
        self.assertEqual(species['conformer_energy_level']['method'], 'dlpno-ccsd(t)')
        self.assertNotIn('software', species['conformer_energy_level'])
        self.assertIsNone(species['conformer_force_field'])
        self.assertIn('parser_evidence', self.document)
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
        self.assertEqual(ts['irc_log_levels'], [{'method': 'wb97xd', 'basis': 'def2svp'},
                                                {'method': 'b3lyp', 'basis': '6-31g'}])
        self.assertEqual(ts['ess_software']['irc'], 'gaussian')
        self.assertIn('Gaussian', ts['ess_versions']['irc'])
        self.assertIsNone(ts['composite_log'])
        self.assertIsNone(ts['composite_input'])
        self.assertIsNone(self.document['gsm_level'])
        self.assertEqual(self.document['species'][0]['rotor_scans'][0]['ess_software'], 'gaussian')

    def test_the_arkane_provenance_comes_from_the_resolver_not_the_environment(self):
        """The header pair is whatever ``_get_arkane_provenance`` returned, so it is deterministic."""
        self.assertEqual(self.document['arkane_version'], ARKANE_PROVENANCE.version)
        self.assertEqual(self.document['arkane_git_commit'], ARKANE_PROVENANCE.git_commit)

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
        calcs = os.path.join(cls.tmp_dir, 'calcs', 'Species', 'nBuOH')
        os.makedirs(calcs, exist_ok=True)
        opt_log = os.path.join(calcs, 'opt.out')
        freq_log = os.path.join(calcs, 'freq.out')
        shutil.copy(OPT_LOG, opt_log)
        shutil.copy(FREQ_LOG, freq_log)
        cls.document = build_output_document(
            project_directory=cls.tmp_dir,
            species_dict={'nBuOH': make_directed_rotor_species()},
            output_dict={
                'nBuOH': {'convergence': True,
                          'paths': {'geo': opt_log, 'geo_coarse': opt_log,
                                    'freq': freq_log, 'sp': opt_log},
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


class TestArkaneAppliedExportsValidate(SchemaAssertionMixin, unittest.TestCase):
    """The E0 and kinetics correction switches, the Arkane treatment, the database identity and the applied
    Petersson components, as the real writer emits them."""

    @classmethod
    def setUpClass(cls):
        cls.schema = load_output_schema()
        cls.validator = Draft202012Validator(cls.schema)
        cls.tmp_dir = tempfile.mkdtemp()
        cls.addClassCleanup(shutil.rmtree, cls.tmp_dir, ignore_errors=True)

    def build(self, reaction=None, species_corrections=None):
        """The document of the rich species and TS, with the given reaction and per-species corrections."""
        return build_output_document(
            project_directory=self.tmp_dir,
            species_dict={'sBuOH': make_rich_species(), 'TS0': make_transition_state()},
            output_dict={
                'sBuOH': {'convergence': True, 'paths': {'geo': OPT_LOG, 'freq': FREQ_LOG, 'sp': OPT_LOG},
                          'job_types': {'opt': True}},
                'TS0': {'convergence': True, 'paths': {'geo': OPT_LOG, 'freq': FREQ_LOG, 'sp': OPT_LOG},
                        'job_types': {'opt': True}},
            },
            reactions=[reaction or make_reaction()],
            species_corrections=species_corrections or make_species_corrections(),
            arkane_level_of_theory=Level(method='cbs-qb3', software='gaussian'),
            bac_type='p',
        )

    def test_the_statmech_block_states_the_e0_switches_and_what_arkane_applied(self):
        """A species whose Arkane run kept both rotors is a one-dimensional rotor treatment with per-rotor
        treatments; a TS whose run kept none is a rigid rotor and harmonic oscillator."""
        document = self.build()
        self.assert_valid(document)
        species, ts = document['species'][0]['statmech'], document['transition_states'][0]['statmech']
        self.assertIs(species['e0_atom_corrections_applied'], True)
        self.assertIs(species['e0_bond_corrections_applied'], True)
        self.assertEqual(species['arkane_rotors_applied'], 2)
        self.assertEqual(species['arkane_treatment'], 'rrho_1d')
        self.assertEqual([torsion['treatment'] for torsion in species['torsions']], ['hindered_rotor', 'free_rotor'])
        self.assertIs(ts['e0_atom_corrections_applied'], True)
        self.assertIs(ts['e0_bond_corrections_applied'], False)
        self.assertEqual(ts['arkane_rotors_applied'], 0)
        self.assertEqual(ts['arkane_treatment'], 'rrho')

    def test_the_statmech_switches_and_treatment_are_null_when_unknown(self):
        """Nothing is guessed: a species with no recorded Arkane run exports null, and its torsions have no
        treatment even though ARC's own rotor record names a type."""
        species = make_rich_species()
        species.e0_atom_corrections_applied = None
        species.e0_bond_corrections_applied = None
        species.arkane_rotor_modes = None
        document = build_output_document(
            project_directory=self.tmp_dir,
            species_dict={'sBuOH': species},
            output_dict={'sBuOH': {'convergence': True, 'paths': {'geo': OPT_LOG, 'freq': FREQ_LOG, 'sp': OPT_LOG},
                                   'job_types': {'opt': True}}},
        )
        self.assert_valid(document)
        statmech = document['species'][0]['statmech']
        for key in ('e0_atom_corrections_applied', 'e0_bond_corrections_applied', 'arkane_rotors_applied',
                    'arkane_treatment'):
            self.assertIn(key, statmech)
            self.assertIsNone(statmech[key])
        self.assertEqual([torsion['treatment'] for torsion in statmech['torsions']], [None, None])

    def test_the_kinetics_block_states_the_atom_correction_switch_of_its_run(self):
        """The switch is exported as recorded, and null when the kinetics did not come from an Arkane run."""
        for switch in (True, False, None):
            with self.subTest(switch=switch):
                reaction = make_reaction()
                reaction.kinetics['atom_corrections_applied'] = switch
                document = self.build(reaction=reaction)
                self.assert_valid(document)
                self.assertIs(document['reactions'][0]['kinetics']['atom_corrections_applied'], switch)

    def test_the_ts_atom_energy_record_follows_the_switch_of_the_run_that_wrote_its_e0(self):
        """The TS's atom_energy record is dropped when its own E0 run did not apply atom corrections, kept when it
        did, and left alone when that is not known."""
        corrections = {'TS0': make_species_corrections()['sBuOH']}
        for switch, expected in ((False, []), (True, ['atom_energy']), (None, ['atom_energy'])):
            with self.subTest(switch=switch):
                species = make_rich_species()
                ts = make_transition_state()
                ts.e0_atom_corrections_applied = switch
                document = build_output_document(
                    project_directory=self.tmp_dir,
                    species_dict={'sBuOH': species, 'TS0': ts},
                    output_dict={label: {'convergence': True, 'job_types': {'opt': True},
                                         'paths': {'geo': OPT_LOG, 'freq': FREQ_LOG, 'sp': OPT_LOG}}
                                 for label in ('sBuOH', 'TS0')},
                    reactions=[make_reaction()], species_corrections=corrections)
                self.assert_valid(document)
                records = document['transition_states'][0]['energy_corrections']
                self.assertEqual([record['correction_type'] for record in records], expected)

    def test_an_e0_only_species_exports_the_corrections_of_its_e0_run(self):
        """A BDE fragment is computed by the E0-only run, which applies the run's BAC, so its bond_additivity
        record is exported and follows the E0 switches, not the thermo ones."""
        corrections = {'sBuOH': make_species_corrections()['sBuOH']}
        for bond_switch, expected in ((True, ['atom_energy', 'bond_additivity']), (False, ['atom_energy']),
                                      (None, ['atom_energy', 'bond_additivity'])):
            with self.subTest(bond_switch=bond_switch):
                fragment = make_rich_species()
                fragment.compute_thermo = False
                fragment.e0_only = True
                fragment.e0_bond_corrections_applied = bond_switch
                fragment.thermo = ThermoData()
                document = build_output_document(
                    project_directory=self.tmp_dir, species_dict={'sBuOH': fragment},
                    output_dict={'sBuOH': {'convergence': True, 'job_types': {'opt': True},
                                           'paths': {'geo': OPT_LOG, 'freq': FREQ_LOG, 'sp': OPT_LOG}}},
                    species_corrections=corrections, bac_type='p',
                    arkane_level_of_theory=Level(method='cbs-qb3', software='gaussian'))
                self.assert_valid(document)
                records = document['species'][0]['energy_corrections']
                self.assertEqual([record['correction_type'] for record in records], expected)

    def test_the_header_carries_the_database_identity(self):
        """The identity the resolver returned is exported whole, in the closed shape."""
        document = self.build()
        self.assert_valid(document)
        self.assertEqual(document['rmg_database'], RMG_DATABASE_IDENTITY)

    def test_a_partly_parametrized_petersson_bac_keeps_its_applied_components(self):
        """Arkane skips a bond it has no parameter for, so the applied components sum to the total and the
        skipped bond is listed, not dropped with them."""
        corrections = make_species_corrections()
        bac = corrections['sBuOH']['bac']
        bac['value'] = -1.5615
        bac['components'][1].update(parameter_value=None, contribution_value=None)
        document = self.build(species_corrections=corrections)
        self.assert_valid(document)
        record = next(entry for entry in document['species'][0]['energy_corrections']
                      if entry['correction_type'] == 'bond_additivity')
        self.assertEqual([component['key'] for component in record['components']], ['C-H'])
        self.assertAlmostEqual(sum(component['contribution_value'] for component in record['components']),
                               record['total']['value'])
        self.assertEqual(record['skipped_components'], [{'bond': 'C-C', 'count': 3}])
        atom_record = next(entry for entry in document['species'][0]['energy_corrections']
                           if entry['correction_type'] == 'atom_energy')
        self.assertIsNone(atom_record['skipped_components'])

    def test_the_new_keys_are_required_and_closed(self):
        """Each new key is required but nullable where it says so, and an unknown value or key is refused."""
        valid = self.build()
        for path, key in ((('species', 0, 'statmech'), 'e0_atom_corrections_applied'),
                          (('species', 0, 'statmech'), 'e0_bond_corrections_applied'),
                          (('species', 0, 'statmech'), 'arkane_rotors_applied'),
                          (('species', 0, 'statmech'), 'arkane_treatment'),
                          (('reactions', 0, 'kinetics'), 'atom_corrections_applied'),
                          (('rmg_database',), 'quantum_corrections_path'),
                          (('rmg_database',), 'matches_arc_rmg_db_path'),
                          ((), 'arc_aec_yml_sha256'),
                          ((), 'rmg_database')):
            with self.subTest(key=key):
                document = copy.deepcopy(valid)
                node = document
                for step in path:
                    node = node[step]
                del node[key]
                self.assert_invalid(document, key)
        document = copy.deepcopy(valid)
        del document['species'][0]['energy_corrections'][0]['skipped_components']
        self.assert_invalid(document, 'skipped_components')
        for path, key, bad in ((('species', 0, 'statmech'), 'arkane_treatment', 'hindered_rotor'),
                               (('species', 0, 'statmech'), 'arkane_rotors_applied', -1),
                               (('species', 0, 'statmech', 'torsions', 0), 'treatment', 'rotor'),
                               (('rmg_database',), 'path_kind', 'tarball'),
                               (('rmg_database',), 'quantum_corrections_sha256', 'abc'),
                               (('rmg_database',), 'extra', 1)):
            with self.subTest(key=key, bad=bad):
                document = copy.deepcopy(valid)
                node = document
                for step in path:
                    node = node[step]
                node[key] = bad
                self.assert_invalid(document, key if key == 'extra' else str(bad))
        document = copy.deepcopy(valid)
        document['species'][0]['statmech']['torsions'][0]['treatment'] = None
        document['species'][0]['statmech']['arkane_treatment'] = 'rrho_1d_nd'
        document['rmg_database'] = {'path_kind': 'unknown', 'git_commit': None, 'version': None,
                                    'quantum_corrections_path': None, 'quantum_corrections_sha256': None,
                                    'matches_arc_rmg_db_path': None}
        document['arc_aec_yml_sha256'] = 'b' * 64
        self.assert_valid(document)

    def test_the_couplings_between_the_new_keys_are_enforced(self):
        """An unparsed Arkane output has no treatment, a rigid rotor has no rotor, and only a Petersson
        bond-additivity record lists skipped bonds."""
        valid = self.build()
        for rotors, treatment in ((None, 'rrho'), (None, 'rrho_1d'), (2, 'rrho')):
            with self.subTest(rotors=rotors, treatment=treatment):
                document = copy.deepcopy(valid)
                document['species'][0]['statmech']['arkane_rotors_applied'] = rotors
                document['species'][0]['statmech']['arkane_treatment'] = treatment
                self.assert_invalid(document, 'arkane_')
        document = copy.deepcopy(valid)
        document['species'][0]['statmech']['arkane_rotors_applied'] = 2
        document['species'][0]['statmech']['arkane_treatment'] = None
        self.assert_valid(document)
        skipped = [{'bond': 'C-C', 'count': 1}]
        document = copy.deepcopy(valid)
        atom_record = next(entry for entry in document['species'][0]['energy_corrections']
                           if entry['correction_type'] == 'atom_energy')
        atom_record['skipped_components'] = skipped
        self.assert_invalid(document, 'skipped_components')
        bond_record = next(entry for entry in document['species'][0]['energy_corrections']
                           if entry['correction_type'] == 'bond_additivity')
        bond_record['skipped_components'] = skipped
        atom_record['skipped_components'] = None
        self.assert_valid(document)
        bond_record['model'] = 'melius'
        self.assert_invalid(document, 'skipped_components')

    def test_a_rotor_treatment_needs_a_rotor_and_a_petersson_record_needs_its_list(self):
        """A rotor treatment implies at least one rotor, a petersson record always lists its skipped bonds (possibly
        none), and any other record has none."""
        valid = self.build()
        for treatment in ('rrho_1d', 'rrho_nd', 'rrho_1d_nd'):
            for rotors in (0, None):
                with self.subTest(treatment=treatment, rotors=rotors):
                    document = copy.deepcopy(valid)
                    document['species'][0]['statmech']['arkane_rotors_applied'] = rotors
                    document['species'][0]['statmech']['arkane_treatment'] = treatment
                    self.assert_invalid(document, 'arkane_')
            document = copy.deepcopy(valid)
            document['species'][0]['statmech']['arkane_rotors_applied'] = 1
            document['species'][0]['statmech']['arkane_treatment'] = treatment
            self.assert_valid(document)
        document = copy.deepcopy(valid)
        bond_record = next(entry for entry in document['species'][0]['energy_corrections']
                           if entry['correction_type'] == 'bond_additivity')
        self.assertEqual(bond_record['model'], 'petersson')
        bond_record['skipped_components'] = []
        self.assert_valid(document)
        bond_record['skipped_components'] = None
        self.assert_invalid(document, 'skipped_components')
        bond_record['skipped_components'] = []
        atom_record = next(entry for entry in document['species'][0]['energy_corrections']
                           if entry['correction_type'] == 'atom_energy')
        atom_record['skipped_components'] = []
        self.assert_invalid(document, 'skipped_components')

    def test_the_ts_frequency_keys_are_written_by_the_real_writer(self):
        document = self.build()
        ts = document['transition_states'][0]
        self.assertEqual(ts['freq_frequencies_cm1_ess_order'], [-1235.4, -18.2, 1300.0, 1500.0])
        self.assertIsNone(ts['reaction_coordinate_mode_index'])
        self.assertNotIn('freq_frequencies_cm1_ess_order', document['species'][0])
        self.assertNotIn('reaction_coordinate_mode_index', document['species'][0])
        self.assert_valid(document)


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
                                  'irc_directions': ['forward', 'reverse'],
                                  'irc_levels': [{'method': 'wb97xd', 'basis': 'def2svp', 'software': 'gaussian'},
                                                 {'method': 'b3lyp', 'basis': '6-31g', 'software': 'gaussian'}]},
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

    def test_thermo_without_the_atom_corrections_level_key_is_rejected(self):
        """The key is required, so a producer that forgets it cannot pass as a null."""
        del self.species['thermo']['atom_corrections_level']
        self.assert_invalid(self.document, 'atom_corrections_level')

    def test_uncorrected_and_unknown_thermo_validate(self):
        """``false`` with no level, and all three ``null``, are both legal thermo blocks."""
        for applied, bond in ((False, False), (None, None)):
            with self.subTest(atom_corrections_applied=applied):
                thermo = self.species['thermo']
                thermo['atom_corrections_applied'] = applied
                thermo['bond_corrections_applied'] = bond
                thermo['atom_corrections_level'] = None
                self.assert_valid(self.document)

    def test_bond_corrections_without_atom_corrections_are_rejected(self):
        """ARC never turns BAC on with AEC off, so a document saying so is wrong."""
        self.species['thermo']['atom_corrections_applied'] = False
        self.species['thermo']['atom_corrections_level'] = None
        self.assert_invalid(self.document, 'bond_corrections_applied')

    def test_uncorrected_thermo_naming_an_atom_corrections_level_is_rejected(self):
        """A level is named only for atom energies that were actually subtracted."""
        self.species['thermo']['atom_corrections_applied'] = False
        self.species['thermo']['bond_corrections_applied'] = False
        self.assert_invalid(self.document, 'atom_corrections_level')

    def test_corrected_thermo_without_an_atom_corrections_level_is_rejected(self):
        """``true`` alone would leave a consumer unable to check which level was subtracted."""
        self.species['thermo']['atom_corrections_level'] = None
        self.assert_invalid(self.document, 'atom_corrections_level')

    def test_unknown_atom_corrections_with_known_bond_corrections_is_rejected(self):
        """The two flags come from one Arkane run, so they are unknown together."""
        self.species['thermo']['atom_corrections_applied'] = None
        self.species['thermo']['atom_corrections_level'] = None
        self.assert_invalid(self.document, 'bond_corrections_applied')

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

    def test_irc_log_levels_carry_no_software_and_are_ts_only(self):
        """The per-log IRC levels state no deduced software, and an ordinary species has no such key"""
        self.ts['irc_log_levels'][0]['software'] = 'gaussian'
        self.assert_invalid(self.document, 'software')
        document = copy.deepcopy(self.valid_document)
        document['species'][0]['irc_log_levels'] = []
        self.assert_invalid(document, 'irc_log_levels')
        document = copy.deepcopy(self.valid_document)
        del document['transition_states'][0]['irc_log_levels']
        self.assert_invalid(document, 'irc_log_levels')
        document = copy.deepcopy(self.valid_document)
        document['transition_states'][0]['irc_log_levels'] = [None, None]
        self.assert_valid(document)

    def test_the_new_program_keys_are_required_and_closed(self):
        """composite_log, composite_input, scan programs and gsm_level are required but nullable"""
        for record, key in (('species', 'composite_log'), ('species', 'composite_input'),
                            ('transition_states', 'composite_log')):
            document = copy.deepcopy(self.valid_document)
            del document[record][0][key]
            self.assert_invalid(document, key)
        for key in ('ess_software', 'ess_version'):
            document = copy.deepcopy(self.valid_document)
            del document['species'][0]['rotor_scans'][0][key]
            self.assert_invalid(document, key)
            document = copy.deepcopy(self.valid_document)
            document['species'][0]['rotor_scans'][0][key] = None
            self.assert_valid(document)
        document = copy.deepcopy(self.valid_document)
        del document['gsm_level']
        self.assert_invalid(document, 'gsm_level')
        document = copy.deepcopy(self.valid_document)
        document['gsm_level'] = {'method': 'gfn2', 'software': 'xtb'}
        self.assert_valid(document)
        document['ts_unknown'] = 1
        self.assert_invalid(document, 'ts_unknown')
        document = copy.deepcopy(self.valid_document)
        document['species'][0]['ess_software'] = {'irc': 'gaussian', 'scan': 'gaussian'}
        self.assert_invalid(document, 'scan')
        document['species'][0]['ess_software'] = {'irc': 'gaussian', 'composite': 'gaussian'}
        self.assert_valid(document)

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

    def test_the_symmetric_and_spherical_top_kinds_are_accepted(self):
        for kind in ('linear', 'symmetric_top', 'spherical_top', 'asymmetric_top'):
            with self.subTest(kind=kind):
                self.species['statmech']['rigid_rotor_kind'] = kind
                self.assert_valid(self.document)
        self.species['statmech']['rigid_rotor_kind'] = None
        self.assert_valid(self.document)
        del self.species['statmech']['rigid_rotor_kind']
        self.assert_invalid(self.document, 'rigid_rotor_kind')
        self.species['statmech']['rigid_rotor_kind'] = 'oblate'
        self.assert_invalid(self.document, 'rigid_rotor_kind')

    def test_the_ts_frequency_keys_are_required_nullable_and_ts_only(self):
        for key in ('freq_frequencies_cm1_ess_order', 'reaction_coordinate_mode_index'):
            with self.subTest(key=key):
                document = copy.deepcopy(self.document)
                del document['transition_states'][0][key]
                self.assert_invalid(document, key)
                document = copy.deepcopy(self.document)
                document['transition_states'][0][key] = None
                self.assert_valid(document)
                document = copy.deepcopy(self.document)
                document['species'][0][key] = None
                self.assert_invalid(document, key)
        self.ts['freq_frequencies_cm1_ess_order'] = [-1235.4, -18.2, 1300.0]
        self.ts['reaction_coordinate_mode_index'] = 1
        self.assert_valid(self.document)
        for bad in (0, -1, 1.5, '1'):
            with self.subTest(index=bad):
                self.ts['reaction_coordinate_mode_index'] = bad
                self.assert_invalid(self.document, 'reaction_coordinate_mode_index')
        self.ts['reaction_coordinate_mode_index'] = None
        self.ts['freq_frequencies_cm1_ess_order'] = ['-1235.4']
        self.assert_invalid(self.document, 'freq_frequencies_cm1_ess_order')

    def test_the_atom_map_keys_are_required_nullable_and_coupled(self):
        reaction = self.document['reactions'][0]
        keys = ('atom_map', 'atom_map_reactant_labels', 'atom_map_product_labels', 'atom_map_source',
                'atom_map_method')
        for key in keys:
            with self.subTest(key=key):
                self.assertIsNone(reaction[key])
                document = copy.deepcopy(self.document)
                del document['reactions'][0][key]
                self.assert_invalid(document, key)

        def fill(source, method):
            reaction.update(atom_map=[2, 0, 1], atom_map_reactant_labels=['H2O'], atom_map_product_labels=['H2O'],
                            atom_map_source=source, atom_map_method=method)

        for source, method in (('inferred', 'arc.mapping.driver.map_reaction'), ('inferred', None),
                               ('declared', None), (None, None)):
            with self.subTest(source=source, method=method):
                fill(source, method)
                self.assert_valid(self.document)
        for source in ('computed', 1, True):
            with self.subTest(source=source):
                fill(source, None)
                self.assert_invalid(self.document, 'atom_map_source')
        for source in ('declared', None):
            with self.subTest(method_without_inferred_source=source):
                fill(source, 'arc.mapping.driver.map_reaction')
                self.assert_invalid(self.document, 'atom_map_method')
        for key, value in (('atom_map_reactant_labels', None), ('atom_map_product_labels', None),
                           ('atom_map_reactant_labels', [1]), ('atom_map_product_labels', 'H2O')):
            with self.subTest(key=key, value=value):
                fill('inferred', None)
                reaction[key] = value
                self.assert_invalid(self.document, key)
        for key, value in (('atom_map_reactant_labels', ['H2O']), ('atom_map_product_labels', ['H2O']),
                           ('atom_map_source', 'declared'), ('atom_map_method', 'x')):
            with self.subTest(null_map_with=key):
                reaction.update({k: None for k in keys})
                reaction[key] = value
                self.assert_invalid(self.document, key)
        for bad in ([-1, 0], [0.5], ['0'], 'x'):
            with self.subTest(atom_map=bad):
                fill('inferred', None)
                reaction['atom_map'] = bad
                self.assert_invalid(self.document, 'atom_map')

    def test_the_irc_participant_mapping_is_required_nullable_closed_and_ts_only(self):
        document = copy.deepcopy(self.document)
        del document['transition_states'][0]['irc_participant_mapping']
        self.assert_invalid(document, 'irc_participant_mapping')
        self.assertIsNone(self.ts['irc_participant_mapping'])
        document = copy.deepcopy(self.document)
        document['species'][0]['irc_participant_mapping'] = None
        self.assert_invalid(document, 'irc_participant_mapping')
        side = {'endpoint': 1, 'endpoint_label': 'IRC_TS0_1',
                'participants': [{'label': 'CH3', 'position': 1, 'occurrence': 1, 'atom_indices': [0, 1, 2, 3]}]}
        self.ts['irc_participant_mapping'] = {'reactants': copy.deepcopy(side),
                                              'products': dict(copy.deepcopy(side), endpoint=2, endpoint_label=None),
                                              'sides_distinguishable': True, 'atom_order_matches_ts': None}
        self.assert_valid(self.document)
        for order in (True, False):
            self.ts['irc_participant_mapping']['atom_order_matches_ts'] = order
            self.assert_valid(self.document)
        self.ts['irc_participant_mapping']['atom_order_matches_ts'] = None
        mapping = self.ts['irc_participant_mapping']
        for description, mutate in (
                ('endpoint 3', lambda m: m['reactants'].update(endpoint=3)),
                ('no products', lambda m: m.pop('products')),
                ('no sides_distinguishable', lambda m: m.pop('sides_distinguishable')),
                ('no atom_order_matches_ts', lambda m: m.pop('atom_order_matches_ts')),
                ('null sides_distinguishable', lambda m: m.update(sides_distinguishable=None)),
                ('string atom_order_matches_ts', lambda m: m.update(atom_order_matches_ts='yes')),
                ('0 occurrence', lambda m: m['reactants']['participants'][0].update(occurrence=0)),
                ('extra key', lambda m: m.update(extra=1)),
                ('extra side key', lambda m: m['reactants'].update(extra=1)),
                ('1-based position', lambda m: m['reactants']['participants'][0].update(position=0)),
                ('negative atom', lambda m: m['reactants']['participants'][0].update(atom_indices=[-1])),
                ('repeated atom', lambda m: m['reactants']['participants'][0].update(atom_indices=[1, 1])),
                ('missing occurrence', lambda m: m['reactants']['participants'][0].pop('occurrence')),
                ('extra participant key', lambda m: m['reactants']['participants'][0].update(extra=1))):
            with self.subTest(description):
                self.ts['irc_participant_mapping'] = copy.deepcopy(mapping)
                mutate(self.ts['irc_participant_mapping'])
                self.assert_invalid(self.document, 'irc_participant_mapping')

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

    def test_records_naming_different_arkane_blocks_are_accepted(self):
        """The AEC and BAC sections are matched independently, so the keys may differ."""
        aec = next(correction for correction in self.species['energy_corrections']
                   if correction['correction_type'] == 'atom_energy')
        bac = next(correction for correction in self.species['energy_corrections']
                   if correction['correction_type'] == 'bond_additivity')
        aec['matched_arkane_key'] = "LevelOfTheory(method='wb97mv',basis='def2tzvpd',software='qchem')"
        bac['matched_arkane_key'] = "LevelOfTheory(method='wb97mv2023',basis='def2tzvpd',software='qchem')"
        self.assert_valid(self.document)

    def test_a_null_matched_arkane_key_is_accepted_but_a_missing_one_is_not(self):
        """An unmatched section is reported as null, never by dropping the field."""
        bac = next(correction for correction in self.species['energy_corrections']
                   if correction['correction_type'] == 'bond_additivity')
        bac['matched_arkane_key'] = None
        self.assert_valid(self.document)
        del bac['matched_arkane_key']
        self.assert_invalid(self.document, 'matched_arkane_key')

    def test_dropping_the_freq_scale_factor_key_is_rejected(self):
        """The block the scale factor was read from is always reported, null included."""
        self.document['freq_scale_factor_key'] = None
        self.assert_valid(self.document)
        del self.document['freq_scale_factor_key']
        self.assert_invalid(self.document, 'freq_scale_factor_key')

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
        self.assertEqual(len(self.species['conformers']),
                         len(self.species['conformer_levels']))

    def test_the_new_header_levels_and_conformer_keys_are_required(self):
        """A producer that forgets any of them cannot pass as a null."""
        for key in ('scan_level', 'irc_level', 'conformer_opt_level', 'conformer_sp_level', 'ts_guess_level'):
            with self.subTest(key=key):
                document = copy.deepcopy(self.document)
                del document[key]
                self.assert_invalid(document, key)
        for record_key in ('species', 'transition_states'):
            for key in ('conformer_levels', 'conformer_energy_kind', 'conformer_energy_level',
                        'conformer_force_field'):
                with self.subTest(record=record_key, key=key):
                    document = copy.deepcopy(self.document)
                    del document[record_key][0][key]
                    self.assert_invalid(document, key)

    def test_null_header_levels_validate(self):
        """Each header level is nullable: the job type was not requested."""
        for key in ('scan_level', 'irc_level', 'conformer_opt_level', 'conformer_sp_level', 'ts_guess_level'):
            self.document[key] = None
        self.assert_valid(self.document)

    def test_a_header_level_that_is_not_a_level_dict_is_rejected(self):
        """A level is the level_dict shape, not a repr string."""
        self.document['scan_level'] = 'b3lyp/6-31g'
        self.assert_invalid(self.document, 'scan_level')

    def test_conformer_levels_may_be_null_per_entry_and_every_kind_validates(self):
        """Null entries (force-field geometries), a null list, and each kind with its level rule validate."""
        self.species['conformer_levels'] = [None, self.species['conformer_levels'][0]]
        self.assert_valid(self.document)
        self.species['conformer_energy_kind'] = 'force_field_kcal_mol'
        self.species['conformer_energy_level'] = None
        self.assert_valid(self.document)
        self.species['conformer_energy_kind'] = None
        self.assert_valid(self.document)

    def test_conformer_keys_must_be_null_without_conformers(self):
        """A record that exports no conformers has null conformer_levels, kind and energy level."""
        del self.species['conformers']
        del self.species['conformer_energies']
        del self.species['conformers_isotopes']
        self.assert_invalid(self.document, 'conformer_levels')
        self.species['conformer_levels'] = None
        self.assert_invalid(self.document, 'conformer_energy_kind')
        self.species['conformer_energy_kind'] = None
        self.assert_invalid(self.document, 'conformer_energy_level')
        self.species['conformer_energy_level'] = None
        self.species['conformer_force_field'] = 'MMFF94s (rdkit)'
        self.assert_invalid(self.document, 'conformer_force_field')
        self.species['conformer_force_field'] = None
        self.assert_valid(self.document)

    def test_the_latent_export_keys_are_required_and_may_be_null(self):
        """A producer that forgets a latent key cannot pass as a null, and a null of each is valid."""
        record_keys = ('xyz_isotopes', 'coarse_opt_input_xyz_isotopes', 'coarse_opt_output_xyz_isotopes',
                       'opt_input_xyz_isotopes', 'opt_route', 'freq_route', 'sp_route', 'composite_route',
                       'sp_t1_diagnostic', 'opt_dipole_moment_debye', 'opt_dipole_moment_density',
                       'freq_polarizability_angstrom3')
        for record in (self.species, self.ts):
            for key in record_keys:
                with self.subTest(key=key, record=record['label']):
                    value = record.pop(key)
                    self.assert_invalid(self.document, key)
                    record[key] = None
                    self.assert_valid(self.document)
                    record[key] = value
        with self.subTest(key='irc_log_routes'):
            value = self.ts.pop('irc_log_routes')
            self.assert_invalid(self.document, 'irc_log_routes')
            self.ts['irc_log_routes'] = value
            self.assert_valid(self.document)
        reaction = self.document['reactions'][0]
        for container, keys in ((reaction, ('reversible',)), (reaction['kinetics'], ('comment', 'ts_validation'))):
            for key in keys:
                with self.subTest(key=key):
                    value = container.pop(key)
                    self.assert_invalid(self.document, key)
                    container[key] = None
                    self.assert_valid(self.document)
                    container[key] = value

    def test_the_latent_export_keys_reject_a_wrong_type(self):
        """Isotopes are positive integers, routes and kinetics text are strings, numbers are numbers."""
        cases = (('xyz_isotopes', [12, 'H']), ('xyz_isotopes', [0, 1]), ('xyz_isotopes', 12),
                 ('opt_input_xyz_isotopes', [1.5]), ('opt_route', 7), ('sp_t1_diagnostic', 'high'),
                 ('opt_dipole_moment_debye', '0.2'), ('opt_dipole_moment_density', 3),
                 ('freq_polarizability_angstrom3', [1.0]))
        for key, value in cases:
            with self.subTest(key=key, value=value):
                original = self.species[key]
                self.species[key] = value
                self.assert_invalid(self.document, key)
                self.species[key] = original
        self.ts['irc_log_routes'] = [1]
        self.assert_invalid(self.document, 'irc_log_routes')
        self.ts['irc_log_routes'] = [None, None]
        reaction = self.document['reactions'][0]
        for container, key, value in ((reaction, 'reversible', 'yes'), (reaction['kinetics'], 'comment', 5),
                                      (reaction['kinetics'], 'ts_validation', False)):
            with self.subTest(key=key):
                original = container[key]
                container[key] = value
                self.assert_invalid(self.document, key)
                container[key] = original
        self.assert_valid(self.document)

    def test_irc_log_routes_belong_to_transition_states_only(self):
        """A species record must not carry the TS-only IRC routes."""
        self.species['irc_log_routes'] = []
        self.assert_invalid(self.document, 'irc_log_routes')

    def test_conformers_isotopes_are_emitted_with_the_conformers_only(self):
        """The isotope lists of the conformers come and go with ``conformers``, and their entries may be null."""
        self.species['conformers_isotopes'] = [None, self.species['xyz_isotopes']]
        self.assert_valid(self.document)
        del self.species['conformers_isotopes']
        self.assert_invalid(self.document, 'conformers_isotopes')

    def test_a_scan_sample_isotope_list_is_a_positive_integer_list(self):
        """``geometry_isotopes`` is null or positive integers."""
        sample = self.species['rotor_scans'][0]['result']['samples'][0]
        sample['geometry_isotopes'] = [0]
        self.assert_invalid(self.document, 'geometry_isotopes')
        sample['geometry_isotopes'] = None
        self.assert_valid(self.document)

    def test_a_scan_sample_states_its_geometry_and_isotopes_together(self):
        """``geometry_xyz`` and ``geometry_isotopes`` are emitted and omitted together."""
        sample = self.species['rotor_scans'][0]['result']['samples'][0]
        isotopes = sample.pop('geometry_isotopes')
        self.assert_invalid(self.document, 'geometry_isotopes')
        geometry = sample.pop('geometry_xyz')
        self.assert_valid(self.document)
        sample['geometry_isotopes'] = isotopes
        self.assert_invalid(self.document, 'geometry_xyz')
        sample['geometry_xyz'] = geometry
        self.assert_valid(self.document)

    def test_a_null_conformer_levels_with_conformers_is_rejected(self):
        """When conformers are exported, conformer_levels is a list."""
        self.species['conformer_levels'] = None
        self.assert_invalid(self.document, 'conformer_levels')

    def test_an_energy_level_without_the_electronic_kind_is_rejected(self):
        """Force-field energies have no level, and an unknown or mixed kind cannot state one."""
        level = self.species['conformer_energy_level']
        self.assertIsNotNone(level)
        for kind in ('force_field_kcal_mol', None):
            with self.subTest(kind=kind):
                document = copy.deepcopy(self.document)
                document['species'][0]['conformer_energy_kind'] = kind
                self.assert_invalid(document, 'conformer_energy_level')

    def test_an_unknown_conformer_energy_kind_is_rejected(self):
        """Only the two documented kinds, or null."""
        self.species['conformer_energy_kind'] = 'electronic_hartree'
        self.assert_invalid(self.document, 'conformer_energy_kind')

    def test_the_writer_emits_levels_and_endpoint_markers_that_validate(self):
        """The baseline carries a levels object on both record types and null endpoint markers on the species."""
        self.assertEqual(set(self.species['levels']), {'opt', 'freq', 'sp', 'composite', 'irc'})
        self.assertEqual(set(self.ts['levels']), {'opt', 'freq', 'sp', 'composite', 'irc'})
        self.assertIsNone(self.species['irc_endpoint_of'])
        self.assertIsNone(self.species['irc_endpoint_direction'])
        self.assertIsNone(self.document['adaptive_levels'])

    def test_recorded_levels_validate_and_a_malformed_level_is_rejected(self):
        """A level dict validates under any job key, a non-level value does not."""
        level = {'method': 'wb97xd', 'basis': 'def2-tzvp'}
        for record in (self.species, self.ts):
            record['levels'].update({'opt': level, 'freq': level, 'sp': level, 'composite': level})
        self.ts['levels']['irc'] = level
        self.assert_valid(self.document)
        self.species['levels']['opt'] = 'wb97xd/def2-tzvp'
        self.assert_invalid(self.document, 'levels')

    def test_a_recorded_level_that_states_a_software_is_rejected(self):
        """The program that ran a job is stated by ess_software, so a per-record level carries no software."""
        for key in ('opt', 'freq', 'sp', 'composite'):
            with self.subTest(key=key):
                document = copy.deepcopy(self.document)
                document['species'][0]['levels'][key] = {'method': 'wb97xd', 'basis': 'def2-tzvp',
                                                         'software': 'gaussian'}
                self.assert_invalid(document, 'levels')
        self.species['conformer_energy_level'] = {'method': 'wb97xd', 'software': 'gaussian'}
        self.assert_invalid(self.document, 'conformer_energy_level')
        self.species['conformer_energy_level'] = {'method': 'wb97xd'}
        self.species['conformer_levels'] = [{'method': 'wb97xd', 'software': 'gaussian'}, None]
        self.assert_invalid(self.document, 'conformer_levels')

    def test_the_conformer_force_field_is_a_string_or_null(self):
        """A force field with a kind, a force field without one (kJ/mol), and null validate; a non-string does not."""
        self.species['conformer_energy_kind'] = 'force_field_kcal_mol'
        self.species['conformer_energy_level'] = None
        for value in ('MMFF94s (rdkit)', None):
            with self.subTest(value=value):
                self.species['conformer_force_field'] = value
                self.assert_valid(self.document)
        self.species['conformer_energy_kind'] = None
        self.species['conformer_force_field'] = 'UFF (openbabel)'
        self.assert_valid(self.document)
        self.species['conformer_force_field'] = 7
        self.assert_invalid(self.document, 'conformer_force_field')

    def test_a_record_without_the_levels_key_is_rejected(self):
        """The key is required, so a producer that forgets it cannot pass as a null."""
        for record in (self.species, self.ts):
            with self.subTest(is_ts=record['is_ts']):
                document = copy.deepcopy(self.document)
                key = 'species' if not record['is_ts'] else 'transition_states'
                del document[key][0]['levels']
                self.assert_invalid(document, 'levels')

    def test_levels_with_a_missing_or_unknown_job_key_are_rejected(self):
        """All five job keys are required and no other key is allowed."""
        del self.species['levels']['composite']
        self.assert_invalid(self.document, 'composite')
        document = copy.deepcopy(self.valid_document)
        document['transition_states'][0]['levels']['scan'] = None
        self.assert_invalid(document, 'scan')

    def test_a_non_ts_record_with_an_irc_level_is_rejected(self):
        """Only a transition state has IRC jobs."""
        self.species['levels']['irc'] = {'method': 'wb97xd', 'basis': 'def2-tzvp'}
        self.assert_invalid(self.document, 'irc')

    def test_an_irc_endpoint_species_validates(self):
        """A species marked as an IRC endpoint with a direction is valid, with or without a recorded direction."""
        self.species['irc_endpoint_of'] = 'TS0'
        for direction in ('forward', 'reverse', None):
            with self.subTest(direction=direction):
                self.species['irc_endpoint_direction'] = direction
                self.assert_valid(self.document)

    def test_an_endpoint_direction_outside_the_vocabulary_is_rejected(self):
        """Only forward and reverse name a direction."""
        self.species['irc_endpoint_of'] = 'TS0'
        self.species['irc_endpoint_direction'] = 'sideways'
        self.assert_invalid(self.document, 'irc_endpoint_direction')

    def test_a_direction_without_an_endpoint_is_rejected(self):
        """An ordinary species has no IRC direction."""
        self.species['irc_endpoint_direction'] = 'forward'
        self.assert_invalid(self.document, 'irc_endpoint_direction')

    def test_a_species_without_the_endpoint_keys_is_rejected(self):
        """Both keys are required on a species record, so a producer that forgets them cannot pass as a null."""
        for key in ('irc_endpoint_of', 'irc_endpoint_direction'):
            with self.subTest(key=key):
                document = copy.deepcopy(self.document)
                del document['species'][0][key]
                self.assert_invalid(document, key)

    def test_the_endpoint_keys_are_not_allowed_on_a_transition_state(self):
        """The TS record does not carry the species-only endpoint markers, whatever its own irc_label holds."""
        self.ts['irc_endpoint_of'] = 'TS0'
        self.assert_invalid(self.document, 'irc_endpoint_of')

    def test_adaptive_levels_in_the_restart_list_form_validate(self):
        """The header list form: heavy-atom ranges ending in 'inf', with job types joined by a space."""
        self.document['adaptive_levels'] = [
            {'atom_range': [1, 5], 'levels': {'opt freq': {'method': 'wb97xd', 'basis': 'def2-svp'},
                                               'sp': {'method': 'dlpno-ccsd(t)', 'basis': 'cc-pvtz'}}},
            {'atom_range': [6, 'inf'], 'levels': {'opt freq': {'method': 'b3lyp', 'basis': '6-31g'}}}]
        self.assert_valid(self.document)

    def test_malformed_adaptive_levels_are_rejected(self):
        """A bad range, a string level, an unknown entry key, and an absent header key are all rejected."""
        good = {'atom_range': [1, 'inf'], 'levels': {'opt': {'method': 'wb97xd'}}}
        cases = {'atom_range': {**good, 'atom_range': [1, 'many']},
                 'levels': {**good, 'levels': {'opt': 'wb97xd/def2-svp'}},
                 'extra': {**good, 'extra': 1}}
        for name, entry in cases.items():
            with self.subTest(case=name):
                document = copy.deepcopy(self.document)
                document['adaptive_levels'] = [entry]
                self.assert_invalid(document, 'adaptive_levels')
        del self.document['adaptive_levels']
        self.assert_invalid(self.document, 'adaptive_levels')


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
