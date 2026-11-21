#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests for the arc.schema module
"""

import copy
import glob
import importlib
import inspect
import logging
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import yaml
from pydantic import ValidationError

from arc.common import ARC_INPUT_SCHEMA_VERSION, ARC_PATH, VERSION, get_logger, read_yaml_file
from arc.exceptions import InputError
from arc.imports import settings
from arc.main import ARC, check_rotor_scan_resolution
from arc.reaction import ARCReaction
from arc.schema import (ARCInput, BACTypeEnum, JobTypes, Reaction, Species, get_arc_input_keys,
                        validate_input_dict)
from arc.settings.settings import default_job_types
from arc.species.species import ARCSpecies


def _normalize(value):
    """
    Recursively convert tuples to lists for order-insensitive structural comparison.

    Args:
        value: The value to normalize.

    Returns:
        The normalized value with all tuples converted to lists.
    """
    if isinstance(value, (list, tuple)):
        return [_normalize(v) for v in value]
    if isinstance(value, dict):
        return {k: _normalize(v) for k, v in value.items()}
    return value


# Keys excluded from the ARC.as_dict() equivalence comparison in TestSeam,
# each with the concrete, investigated reason it is inherently non-deterministic between two separate
# ARC constructions (as opposed to a real behavioral difference introduced by the schema seam).
_AS_DICT_EXCLUDED_KEYS = {
    # Absolute path embedding the per-construction temp directory, which necessarily differs between
    # the 'raw' and 'schema' temp directories.
    'project_directory',
    # Derived from project_directory (output/, etc.), same reason as above.
    'output_directory',
    # Wall-clock timestamp captured at construction time (self.t0 = time.time()); the two
    # constructions run microseconds apart and will not match.
    'execution_time',
}


def _normalize_transient_mol_atom_ids(obj):
    """
    Recursively renumber RMG ``Molecule`` atom ``id`` fields (and their ``edges`` cross-references)
    to a 0-based positional index.

    Each ``ARCSpecies``/RMG ``Molecule`` atom is stamped with an ``id`` drawn from a single
    process-wide auto-incrementing (negative) counter at construction time. Two separate ARC
    constructions in the same test process therefore get different, but internally consistent,
    id numbering for otherwise-identical molecules; the ids themselves carry no chemical
    information, only their consistent use as edge references within one molecule does. This
    normalizes that transient counter away without touching any other data.

    Args:
        obj: An arbitrarily nested structure (as produced by ``ARC.as_dict()``) to normalize.

    Returns:
        A structurally-equal copy of ``obj`` with any molecule atom ``id``/``edges`` fields
        renumbered to positional indices; other data is left untouched.
    """
    if isinstance(obj, dict):
        if isinstance(obj.get('atoms'), list) and all(isinstance(a, dict) and 'id' in a for a in obj['atoms']):
            id_map = {atom['id']: i for i, atom in enumerate(obj['atoms'])}
            new_atoms = []
            for atom in obj['atoms']:
                new_atom = dict(atom)
                new_atom['id'] = id_map[atom['id']]
                if isinstance(new_atom.get('edges'), dict):
                    new_atom['edges'] = {id_map.get(k, k): v for k, v in new_atom['edges'].items()}
                new_atoms.append(_normalize_transient_mol_atom_ids(new_atom))
            new_obj = {k: (new_atoms if k == 'atoms' else v) for k, v in obj.items()}
            if isinstance(new_obj.get('atom_order'), list):
                new_obj['atom_order'] = [id_map.get(x, x) for x in new_obj['atom_order']]
            return {k: _normalize_transient_mol_atom_ids(v) for k, v in new_obj.items()}
        return {k: _normalize_transient_mol_atom_ids(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalize_transient_mol_atom_ids(v) for v in obj]
    return obj


class TmpDirTestCase(unittest.TestCase):
    """
    A TestCase that can create temporary directories which are removed after each test.
    """

    def make_tmp_dir(self) -> str:
        """
        Create a temporary directory that is removed when the test finishes.

        Returns:
            str: The path to the new directory.
        """
        path = tempfile.mkdtemp(prefix='arc_test_schema_')
        self.addCleanup(shutil.rmtree, path, ignore_errors=True)
        return path


class TestARCInput(TmpDirTestCase):
    """
    Tests for the top-level ARCInput model.
    """

    def test_arc_input_minimal(self):
        """Test creating a minimal ARCInput instance"""
        arc_input = ARCInput(project='test_schema')
        self.assertEqual(arc_input.project, 'test_schema')
        self.assertEqual(arc_input.verbose, 20)
        self.assertEqual(arc_input.bac_type, 'p')
        self.assertEqual(arc_input.thermo_adapter, 'arkane')
        self.assertEqual(arc_input.kinetics_adapter, 'arkane')
        self.assertEqual(arc_input.n_confs, 10)
        self.assertEqual(arc_input.e_confs, 5.0)
        self.assertEqual(arc_input.T_count, 50)
        self.assertEqual(arc_input.level_of_theory, '')
        self.assertEqual(arc_input.specific_job_type, '')
        self.assertIsNone(arc_input.species)
        self.assertIsNone(arc_input.reactions)
        self.assertIs(arc_input.compute_thermo, True)
        self.assertIs(arc_input.compute_rates, True)
        self.assertIs(arc_input.compute_transport, False)
        self.assertIs(arc_input.allow_nonisomorphic_2d, False)
        self.assertIs(arc_input.trsh_ess_jobs, True)
        self.assertIs(arc_input.trsh_rotors, True)

    def test_arc_input_requires_project(self):
        """Test that a project name is required"""
        with self.assertRaises(ValidationError) as cm:
            ARCInput()
        self.assertIn('project', str(cm.exception))

    def test_arc_input_bac_type(self):
        """Test the bac_type field"""
        arc_input = ARCInput(project='p', bac_type='m')
        self.assertEqual(arc_input.bac_type, 'm')
        arc_input = ARCInput(project='p', bac_type=None)
        self.assertIsNone(arc_input.bac_type)
        with self.assertRaises(ValidationError) as cm:
            ARCInput(project='p', bac_type='x')
        self.assertIn('bac_type', str(cm.exception))
        self.assertEqual([e.value for e in BACTypeEnum], ['p', 'm'])

    def test_arc_input_bac_type_is_case_insensitive(self):
        """Test that bac_type is lowercased before enum validation (arc/arkane.py compares
        bac_type.lower() == 'm'), but the spelled-out name is still rejected since arc/arkane.py
        writes the raw value straight into the Arkane input as bondCorrectionType"""
        arc_input = ARCInput(project='p', bac_type='P')
        self.assertEqual(arc_input.bac_type, 'p')
        arc_input = ARCInput(project='p', bac_type='M')
        self.assertEqual(arc_input.bac_type, 'm')
        with self.assertRaises(ValidationError) as cm:
            ARCInput(project='p', bac_type='petersson')
        self.assertIn('bac_type', str(cm.exception))
        with self.assertRaises(ValidationError) as cm:
            ARCInput(project='p', bac_type='PETERSSON')
        self.assertIn('bac_type', str(cm.exception))

    def test_arc_input_statmech_adapters(self):
        """Test the thermo_adapter and kinetics_adapter fields"""
        arc_input = ARCInput(project='p', thermo_adapter='Arkane', kinetics_adapter='Arkane')
        self.assertEqual(arc_input.thermo_adapter, 'arkane')
        self.assertEqual(arc_input.kinetics_adapter, 'arkane')
        with self.assertRaises(ValidationError) as cm:
            ARCInput(project='p', thermo_adapter='mess')
        self.assertIn('thermo_adapter', str(cm.exception))
        with self.assertRaises(ValidationError) as cm:
            ARCInput(project='p', kinetics_adapter='unsupported')
        self.assertIn('kinetics_adapter', str(cm.exception))

    def test_arc_input_ts_adapters(self):
        """Test the ts_adapters field validator"""
        arc_input = ARCInput(project='p', ts_adapters=['heuristics', 'GCN'])
        self.assertEqual(arc_input.ts_adapters, ['heuristics', 'GCN'])
        arc_input = ARCInput(project='p', ts_adapters=[])
        self.assertEqual(arc_input.ts_adapters, list())
        with self.assertRaises(ValidationError) as cm:
            ARCInput(project='p', ts_adapters=['no_such_adapter'])
        self.assertIn('ts_adapters', str(cm.exception))

    def test_arc_input_numeric_constraints(self):
        """Test numeric field constraints"""
        with self.assertRaises(ValidationError) as cm:
            ARCInput(project='p', n_confs=0)
        self.assertIn('n_confs', str(cm.exception))
        with self.assertRaises(ValidationError) as cm:
            ARCInput(project='p', e_confs=-1.0)
        self.assertIn('e_confs', str(cm.exception))
        with self.assertRaises(ValidationError) as cm:
            ARCInput(project='p', T_count=-5)
        self.assertIn('T_count', str(cm.exception))
        with self.assertRaises(ValidationError) as cm:
            ARCInput(project='p', max_job_time=-1)
        self.assertIn('max_job_time', str(cm.exception))
        with self.assertRaises(ValidationError) as cm:
            ARCInput(project='p', job_memory=-14)
        self.assertIn('job_memory', str(cm.exception))
        with self.assertRaises(ValidationError) as cm:
            ARCInput(project='p', freq_scale_factor=0)
        self.assertIn('freq_scale_factor', str(cm.exception))
        with self.assertRaises(ValidationError) as cm:
            ARCInput(project='p', verbose=-10)
        self.assertIn('verbose', str(cm.exception))
        self.assertEqual(ARCInput(project='p', max_job_time=0).max_job_time, 0)

    def test_arc_input_optional_string_fields_accept_none(self):
        """Test that level_of_theory and specific_job_type accept an explicit null, as real restart
        files and a bare input can carry, rather than only a string"""
        arc_input = ARCInput(project='p', level_of_theory=None, specific_job_type=None)
        self.assertIsNone(arc_input.level_of_theory)
        self.assertIsNone(arc_input.specific_job_type)

    def test_job_types_and_species_flags_accept_int_0_1(self):
        """Test that job_types and species flags treat 0/1 like the top-level flags do"""
        arc_input = ARCInput(project='p', job_types={'rotors': 0, 'freq': 1},
                             species=[{'label': 'H2O', 'smiles': 'O', 'compute_thermo': 0, 'is_ts': 0}])
        self.assertIs(arc_input.job_types.rotors, False)
        self.assertIs(arc_input.job_types.freq, True)
        self.assertIs(arc_input.species[0].compute_thermo, False)
        for bad in ('False', 2):
            with self.assertRaises(ValidationError) as cm:
                ARCInput(project='p', job_types={'rotors': bad})
            self.assertIn('rotors', str(cm.exception))
            with self.assertRaises(ValidationError) as cm:
                ARCInput(project='p', species=[{'label': 'H2O', 'smiles': 'O', 'compute_thermo': bad}])
            self.assertIn('compute_thermo', str(cm.exception))

    def test_arc_input_bool_flags_accept_int_0_1(self):
        """Test that a run flag spelled as an integer 0/1 (a previously-working raw ARC input) is
        coerced to a bool, while a quoted string or an out-of-range int still fails strict-bool
        validation and cannot silently invert the flag"""
        arc_input = ARCInput(project='p', compute_thermo=1, trsh_rotors=0)
        self.assertIs(arc_input.compute_thermo, True)
        self.assertIs(arc_input.trsh_rotors, False)
        for bad in ('False', 2):
            with self.assertRaises(ValidationError) as cm:
                ARCInput(project='p', compute_thermo=bad)
            self.assertIn('compute_thermo', str(cm.exception))

    def test_arc_input_integral_float_coercion(self):
        """Test that job_memory, n_confs, and T_count accept an integral float (a previously-working
        raw ARC input value) while still rejecting a non-integral float and a numeric string, so
        strict-int validation is preserved for genuinely invalid input"""
        arc_input = ARCInput(project='p', job_memory=14.0, n_confs=14.0, T_count=14.0)
        self.assertEqual(arc_input.job_memory, 14)
        self.assertEqual(arc_input.n_confs, 14)
        self.assertEqual(arc_input.T_count, 14)
        for field in ('job_memory', 'n_confs', 'T_count'):
            with self.assertRaises(ValidationError) as cm:
                ARCInput(project='p', **{field: 14.7})
            self.assertIn(field, str(cm.exception))
            with self.assertRaises(ValidationError) as cm:
                ARCInput(project='p', **{field: '14'})
            self.assertIn(field, str(cm.exception))

    def test_arc_input_verbose_accepts_any_logging_level(self):
        """Test that verbose accepts any non-negative integer logging level, and rejects bools and strings"""
        for level in (0, 10, 15, 20, 25, 50, 60):
            self.assertEqual(ARCInput(project='p', verbose=level).verbose, level)
        for level in (True, '20', 20.5):
            with self.assertRaises(ValidationError) as cm:
                ARCInput(project='p', verbose=level)
            self.assertIn('verbose', str(cm.exception))

    def test_arc_input_verbose_rejects_explicit_none_but_allows_absent(self):
        """Test that verbose=None raises (logger.setLevel(None) crashes ARC), while an absent
        verbose still validates and keeps its default"""
        with self.assertRaises(ValidationError) as cm:
            ARCInput(project='p', verbose=None)
        self.assertIn('verbose', str(cm.exception))
        arc_input = ARCInput(project='p')
        self.assertEqual(arc_input.verbose, 20)
        self.assertNotIn('verbose', arc_input.as_input_dict())

    def test_arc_input_t_min_t_max(self):
        """Test the T_min and T_max fields"""
        arc_input = ARCInput(project='p', T_min=(300, 'K'), T_max=[3000, 'K'])
        self.assertEqual(arc_input.T_min, (300.0, 'K'))
        self.assertEqual(arc_input.T_max, (3000.0, 'K'))
        with self.assertRaises(ValidationError) as cm:
            ARCInput(project='p', T_min=('K', 300))
        self.assertIn('T_min', str(cm.exception))

    def test_arc_input_t_min_t_max_unit_is_case_insensitive(self):
        """Test that a lowercase 'k' unit is accepted and normalized to 'K'"""
        arc_input = ARCInput(project='p', T_min=[300, 'k'], T_max=(3000, 'K'))
        self.assertEqual(arc_input.T_min, (300, 'K'))
        self.assertEqual(arc_input.as_input_dict()['T_min'], [300, 'K'])

    def test_arc_input_t_min_t_max_errors_are_readable(self):
        """Test that a malformed T_min gives one plain error without pydantic's internal union branch names"""
        for bad in (-5, [300, 'C'], 'hot', [300, 400], True):
            with self.assertRaises(ValidationError) as cm:
                ARCInput(project='p', T_min=bad)
            message = str(cm.exception)
            self.assertIn('T_min', message)
            self.assertEqual(len(cm.exception.errors()), 1)
            for internal_name in ('tuple[', 'constrained-int', 'constrained-float', 'function-before'):
                self.assertNotIn(internal_name, message)

    def test_arc_input_numbers_keep_their_type(self):
        """Test that integer values are not converted to floats, so restart.yml keeps its shape"""
        arc_input = ARCInput(project='p', e_confs=5, T_min=300, T_max=[3000, 'K'], max_job_time=24,
                             freq_scale_factor=1, rotor_scan_resolution=8)
        dumped = arc_input.as_input_dict()
        for key, value in (('e_confs', 5), ('T_min', 300), ('max_job_time', 24),
                           ('freq_scale_factor', 1), ('rotor_scan_resolution', 8)):
            self.assertIs(type(dumped[key]), int, key)
            self.assertEqual(dumped[key], value)
        self.assertIs(type(dumped['T_max'][0]), int)
        self.assertEqual(yaml.dump(dumped), yaml.dump(validate_input_dict(yaml.safe_load(yaml.dump(dumped)))))

    def test_arc_input_numbers_reject_strings_and_bools(self):
        """Test that numeric fields are strict: a quoted number or a bool is rejected rather than coerced"""
        for field in ('e_confs', 'max_job_time', 'freq_scale_factor', 'rotor_scan_resolution', 'T_min'):
            for bad in ('10', True):
                with self.assertRaises(ValidationError) as cm:
                    ARCInput(project='p', **{field: bad})
                self.assertIn(field, str(cm.exception))

    def test_arc_input_t_min_t_max_dumps_as_plain_list(self):
        """Test that T_min/T_max are plain lists (not tuples) after as_input_dict(), for yaml.dump safety"""
        arc_input = ARCInput(project='p', T_min=[300, 'K'], T_max=[3000, 'K'])
        dumped = arc_input.as_input_dict()
        self.assertIsInstance(dumped['T_min'], list)
        self.assertIsInstance(dumped['T_max'], list)
        self.assertEqual(dumped['T_min'], [300.0, 'K'])
        dumped_yaml = yaml.dump({'T_min': dumped['T_min'], 'T_max': dumped['T_max']})
        self.assertNotIn('python/tuple', dumped_yaml)

    def test_arc_input_t_min_t_max_accept_bare_scalar(self):
        """Test that a bare scalar T_min/T_max (raw ARC's shorthand for Kelvin) validates"""
        arc_input = ARCInput(project='p', T_min=500, T_max=3000)
        self.assertEqual(arc_input.T_min, 500)
        self.assertEqual(arc_input.T_max, 3000)
        dumped = arc_input.as_input_dict()
        # A scalar in must stay a scalar out; processor.py normalizes it downstream.
        self.assertEqual(dumped['T_min'], 500)
        self.assertIsInstance(dumped['T_min'], float | int)
        self.assertEqual(dumped['T_max'], 3000)
        self.assertIsInstance(dumped['T_max'], float | int)
        dumped_yaml = yaml.dump({'T_min': dumped['T_min'], 'T_max': dumped['T_max']})
        self.assertNotIn('python/tuple', dumped_yaml)

    def test_arc_input_t_min_t_max_pair_form_still_works(self):
        """Regression check: the (value, unit) pair form must still validate and dump as a plain list"""
        arc_input = ARCInput(project='p', T_min=[300, 'K'], T_max=(3000, 'K'))
        self.assertEqual(arc_input.T_min, (300.0, 'K'))
        self.assertEqual(arc_input.T_max, (3000.0, 'K'))
        dumped = arc_input.as_input_dict()
        self.assertIsInstance(dumped['T_min'], list)
        self.assertEqual(dumped['T_min'], [300.0, 'K'])
        self.assertIsInstance(dumped['T_max'], list)
        self.assertEqual(dumped['T_max'], [3000.0, 'K'])

    def test_arc_input_t_min_t_max_reject_non_kelvin_unit(self):
        """Test that a non-Kelvin unit raises, since no consumer ever reads or converts the unit
        (plotter.py/processor.py silently treat T_min[0]/T_max[0] as Kelvin regardless of the label)"""
        with self.assertRaises(ValidationError) as cm:
            ARCInput(project='p', T_min=[300, 'C'])
        self.assertIn('T_min', str(cm.exception))
        with self.assertRaises(ValidationError) as cm:
            ARCInput(project='p', T_max=[3000, 'F'])
        self.assertIn('T_max', str(cm.exception))
        arc_input = ARCInput(project='p', T_min=[300, 'K'])
        self.assertEqual(arc_input.T_min, (300.0, 'K'))

    def test_arc_input_t_min_t_max_reject_non_positive_temperature(self):
        """Test that a non-positive absolute temperature raises, in both scalar and pair form"""
        with self.assertRaises(ValidationError) as cm:
            ARCInput(project='p', T_min=-5)
        self.assertIn('T_min', str(cm.exception))
        with self.assertRaises(ValidationError) as cm:
            ARCInput(project='p', T_min=0)
        self.assertIn('T_min', str(cm.exception))
        with self.assertRaises(ValidationError) as cm:
            ARCInput(project='p', T_min=[0, 'K'])
        self.assertIn('T_min', str(cm.exception))
        arc_input = ARCInput(project='p', T_min=500)
        self.assertEqual(arc_input.T_min, 500)
        self.assertEqual(arc_input.as_input_dict()['T_min'], 500)

    def test_arc_input_t_min_t_max_accept_none(self):
        """Test that T_min: null / T_max: null still validate, since real restart files have them"""
        arc_input = ARCInput(project='p', T_min=None, T_max=None)
        self.assertIsNone(arc_input.T_min)
        self.assertIsNone(arc_input.T_max)

    def test_arc_input_species_and_reactions_accept_none(self):
        """Test that explicit species: null / reactions: null validate, mirroring ARC.__init__"""
        arc_input = ARCInput(project='p', species=None, reactions=None)
        self.assertIsNone(arc_input.species)
        self.assertIsNone(arc_input.reactions)

    def test_arc_input_wrong_job_types(self):
        """Test that a wrongly-typed job_types raises"""
        with self.assertRaises(ValidationError) as cm:
            ARCInput(project='p', job_types='rotors')
        self.assertIn('job_types', str(cm.exception))

    def test_arc_input_extra_keys_forbidden(self):
        """Test that an unknown top-level key is rejected with an error citing the key"""
        with self.assertRaises(ValidationError) as cm:
            ARCInput(project='p', some_unknown_key='value')
        self.assertIn('some_unknown_key', str(cm.exception))

    def test_strict_bool_fields_reject_quoted_strings(self):
        """Test that bool fields reject quoted string/int values instead of laxly coercing them"""
        with self.assertRaises(ValidationError) as cm:
            ARCInput(project='p', compute_thermo='False')
        self.assertIn('compute_thermo', str(cm.exception))
        with self.assertRaises(ValidationError) as cm:
            ARCInput(project='p', job_types={'rotors': 'False'})
        self.assertIn('rotors', str(cm.exception))
        with self.assertRaises(ValidationError) as cm:
            Species(label='spc', smiles='CC', is_ts='False')
        self.assertIn('is_ts', str(cm.exception))

    def test_strict_bool_fields_accept_unquoted_yaml_bools(self):
        """Test that ordinary (unquoted) YAML booleans still validate after strictness is added"""
        yaml_str = """
        project: p
        compute_thermo: false
        compare_to_rmg: no
        """
        data = yaml.safe_load(yaml_str)
        arc_input = ARCInput(**data)
        self.assertIs(arc_input.compute_thermo, False)
        self.assertIs(arc_input.compare_to_rmg, False)

    def test_e_confs_accepts_none_and_round_trips_through_arcs_own_restart(self):
        """Test that e_confs: None is accepted, since ARC writes it and uses it to disable the cutoff."""
        tmp_dir = self.make_tmp_dir()
        self.assertIsNone(ARCInput(project='p', e_confs=None).e_confs)
        self.assertEqual(ARCInput(project='p').e_confs, 5.0)
        with self.assertRaises(ValidationError):
            ARCInput(project='p', e_confs=-1.0)

        arc_object = ARC(project='e_confs_round_trip', e_confs=None, project_directory=os.path.join(tmp_dir, 'first'))
        restart_dict = arc_object.as_dict()
        self.assertIsNone(restart_dict['e_confs'])
        validated = validate_input_dict(copy.deepcopy(restart_dict))
        self.assertIsNone(validated['e_confs'])

    def test_specific_job_type_must_be_a_known_job_type(self):
        """Test that specific_job_type is validated against the job-type vocabulary at load time.

        Its vocabulary is the keys of default_job_types, otherwise enforced only by an InputError from
        initialize_job_types() inside ARC.__init__, which ARC.py calls outside its handled block.
        """
        self.assertEqual(ARCInput(project='p', specific_job_type='opt').specific_job_type, 'opt')
        self.assertEqual(ARCInput(project='p', specific_job_type='').specific_job_type, '')
        self.assertEqual(ARCInput(project='p').specific_job_type, '')
        for job_type in default_job_types.keys():
            self.assertEqual(ARCInput(project='p', specific_job_type=job_type).specific_job_type, job_type)

        with self.assertRaises(ValidationError) as cm:
            ARCInput(project='p', specific_job_type='bogus')
        self.assertIn('bogus', str(cm.exception))

    def test_as_input_dict_preserves_explicit_none_and_omits_unset(self):
        """Test that as_input_dict reproduces exactly the user-provided keys"""
        arc_input = ARCInput(project='p', bac_type=None, species=[{'label': 'H2', 'smiles': '[H][H]'}])
        dumped = arc_input.as_input_dict()
        self.assertEqual(set(dumped.keys()), {'project', 'bac_type', 'species'})
        self.assertIsNone(dumped['bac_type'])
        self.assertEqual(dumped['species'], [{'label': 'H2', 'smiles': '[H][H]'}])


class TestJobTypes(unittest.TestCase):
    """
    Tests for the JobTypes model.
    """

    def test_job_types_extra_keys_forbidden(self):
        """Test that a typo inside job_types is rejected at load rather than silently ignored"""
        with self.assertRaises(ValidationError) as cm:
            ARCInput(project='p', job_types={'fine_grd': True})
        self.assertIn('fine_grd', str(cm.exception))

    def test_job_types_schema(self):
        """Test creating an instance of JobTypes"""
        job_types = JobTypes()
        self.assertIs(job_types.conf_opt, True)
        self.assertIs(job_types.conf_sp, False)
        self.assertIs(job_types.opt, True)
        self.assertIs(job_types.fine_grid, True)
        self.assertIs(job_types.freq, True)
        self.assertIs(job_types.sp, True)
        self.assertIs(job_types.rotors, True)
        self.assertIs(job_types.irc, True)
        self.assertIs(job_types.orbitals, False)
        self.assertIs(job_types.lennard_jones, False)
        self.assertIs(job_types.bde, False)

        job_types = JobTypes(fine=True, onedmin=False)
        self.assertIs(job_types.fine, True)
        self.assertIs(job_types.onedmin, False)

        with self.assertRaises(ValidationError) as cm:
            JobTypes(rotors='sometimes')
        self.assertIn('rotors', str(cm.exception))

    def test_job_types_restart_form_keys_validated(self):
        """Test that the restart-form job_types keys (fine, onedmin) are typed and validated"""
        with self.assertRaises(ValidationError) as cm:
            ARCInput(project='p', job_types={'fine': 'definitely'})
        self.assertIn('fine', str(cm.exception))
        with self.assertRaises(ValidationError) as cm:
            ARCInput(project='p', job_types={'onedmin': 'maybe'})
        self.assertIn('onedmin', str(cm.exception))
        arc_input = ARCInput(project='p', job_types={'fine': True, 'onedmin': False})
        self.assertIs(arc_input.job_types.fine, True)
        self.assertIs(arc_input.job_types.onedmin, False)

    def test_job_types_fine_reject_explicit_none_but_allow_absent(self):
        """Test that an explicit fine null raises, since initialize_job_types only fills fine's default
        (True) when the key is absent, so an explicit null would silently disable fine-grid opt. An
        explicit onedmin null is allowed: onedmin defaults to False, so null is identical to absent"""
        with self.assertRaises(ValidationError) as cm:
            ARCInput(project='p', job_types={'fine': None})
        self.assertIn('fine', str(cm.exception))

        arc_input = ARCInput(project='p', job_types={'onedmin': None})
        self.assertIsNone(arc_input.job_types.onedmin)

        arc_input = ARCInput(project='p', job_types={'fine': True})
        self.assertIs(arc_input.job_types.fine, True)

        arc_input = ARCInput(project='p', job_types={})
        self.assertIsNone(arc_input.job_types.fine)
        self.assertNotIn('fine', arc_input.as_input_dict()['job_types'])

    def test_job_types_contradictory_fine_aliases_raise(self):
        """Test that contradictory fine_grid/fine values raise, since fine_grid silently wins in
        ``initialize_job_types`` and the user's explicit ``fine`` value would otherwise be dropped"""
        with self.assertRaises(ValidationError) as cm:
            JobTypes(fine_grid=True, fine=False)
        self.assertIn('fine_grid', str(cm.exception))
        self.assertIn('fine', str(cm.exception))

    def test_job_types_contradictory_lennard_jones_aliases_raise(self):
        """Test that contradictory lennard_jones/onedmin values raise, since lennard_jones silently
        wins in ``initialize_job_types`` and the user's explicit ``onedmin`` value would otherwise be dropped"""
        with self.assertRaises(ValidationError) as cm:
            JobTypes(lennard_jones=True, onedmin=False)
        self.assertIn('lennard_jones', str(cm.exception))
        self.assertIn('onedmin', str(cm.exception))

    def test_job_types_agreeing_aliases_allowed(self):
        """Test that both aliases of a pair set to the same value is allowed (no information lost)"""
        job_types = JobTypes(fine_grid=True, fine=True)
        self.assertIs(job_types.fine_grid, True)
        self.assertIs(job_types.fine, True)
        job_types = JobTypes(lennard_jones=False, onedmin=False)
        self.assertIs(job_types.lennard_jones, False)
        self.assertIs(job_types.onedmin, False)

    def test_job_types_single_alias_only_still_allowed(self):
        """Test that specifying only one of a pair's aliases still validates unaffected"""
        job_types = JobTypes(fine_grid=False)
        self.assertIs(job_types.fine_grid, False)
        self.assertIsNone(job_types.fine)
        job_types = JobTypes(fine=False)
        self.assertIs(job_types.fine, False)
        job_types = JobTypes(lennard_jones=True)
        self.assertIs(job_types.lennard_jones, True)
        self.assertIsNone(job_types.onedmin)
        job_types = JobTypes(onedmin=True)
        self.assertIs(job_types.onedmin, True)


class TestSpecies(unittest.TestCase):
    """
    Tests for the Species model.
    """

    def test_species_schema(self):
        """Test creating an instance of Species"""
        spc = Species(label='H2', smiles='[H][H]')
        self.assertEqual(spc.label, 'H2')
        self.assertEqual(spc.smiles, '[H][H]')
        self.assertIs(spc.is_ts, False)

        spc = Species(label='vinoxy', smiles='C=C[O]', multiplicity=2, charge=0,
                      external_symmetry=1, optical_isomers=1, bdes=[(1, 2), 'all_h'])
        self.assertEqual(spc.multiplicity, 2)
        self.assertEqual(spc.charge, 0)

        with self.assertRaises(ValidationError) as cm:
            Species(smiles='CC')
        self.assertIn('label', str(cm.exception))

        with self.assertRaises(ValidationError) as cm:
            Species(label='spc', smiles='CC', multiplicity=0)
        self.assertIn('multiplicity', str(cm.exception))

        with self.assertRaises(ValidationError) as cm:
            Species(label='spc', smiles='CC', external_symmetry=0)
        self.assertIn('external_symmetry', str(cm.exception))

        with self.assertRaises(ValidationError) as cm:
            Species(label='spc', smiles='CC', optical_isomers=-1)
        self.assertIn('optical_isomers', str(cm.exception))

        with self.assertRaises(ValidationError) as cm:
            Species(label='spc', smiles='CC', charge='neutral')
        self.assertIn('charge', str(cm.exception))

    def test_strict_int_fields_reject_quoted_strings(self):
        """Test that int fields reject quoted numeric strings instead of laxly coercing them"""
        with self.assertRaises(ValidationError) as cm:
            Species(label='spc', smiles='CC', multiplicity='3')
        self.assertIn('multiplicity', str(cm.exception))
        with self.assertRaises(ValidationError) as cm:
            Species(label='spc', smiles='CC', charge='0')
        self.assertIn('charge', str(cm.exception))
        spc = Species(label='spc', smiles='CC', multiplicity=3, charge=0)
        self.assertEqual(spc.multiplicity, 3)
        self.assertEqual(spc.charge, 0)

    def test_species_structure_check(self):
        """Test that a non-TS species with no structure specification raises"""
        with self.assertRaises(ValidationError):
            Species(label='no_structure')
        ts = Species(label='TS0', is_ts=True)
        self.assertIs(ts.is_ts, True)
        yml_spc = Species(label='from_yml', yml_path='spc.yml')
        self.assertEqual(yml_spc.yml_path, 'spc.yml')
        # most_stable_conformer is an index into conformers, not a structure source, so it alone must
        # not satisfy the structure check.
        with self.assertRaises(ValidationError):
            Species(label='only_index', most_stable_conformer=0)

    def test_species_runtime_keys_modeled(self):
        """Test that runtime/restart species keys are modeled fields rather than extras"""
        spc = Species(label='H2', smiles='[H][H]', neg_freqs_trshed=[], ts_methods=None)
        self.assertEqual(spc.neg_freqs_trshed, list())
        self.assertIn('neg_freqs_trshed', Species.model_fields)
        self.assertIn('ts_methods', Species.model_fields)

    def test_species_typo_key_rejected(self):
        """Test that a typo in a species key is rejected instead of being silently swallowed
        (a swallowed 'mutliplicity' typo previously led to a wrong auto-determined multiplicity)"""
        with self.assertRaises(ValidationError) as cm:
            Species(label='CH2', smiles='[CH2]', mutliplicity=1)
        self.assertIn('mutliplicity', str(cm.exception))

    def test_a_restart_species_carrying_stability_state_is_accepted(self):
        """Test that a restart dict written after a stability analysis validates under extra='forbid'"""
        spc = ARCSpecies(label='vinoxy', smiles='C=C[O]')
        spc.derived_stability_verdict = {'stable': False, 'adopted': True}
        spc.scf_references = {'opt_a1234': 'uhf'}
        spc.stability_analysis_ran = True
        spc.stability_pending_opt_job = 'opt_a1234'
        spc.stability_reoptimized = True
        validated = Species(**spc.as_dict())
        self.assertEqual(validated.derived_stability_verdict, {'stable': False, 'adopted': True})
        self.assertEqual(validated.scf_references, {'opt_a1234': 'uhf'})
        self.assertIs(validated.stability_analysis_ran, True)
        self.assertEqual(validated.stability_pending_opt_job, 'opt_a1234')
        self.assertIs(validated.stability_reoptimized, True)

    def test_a_species_with_only_a_cheap_conformer_is_rejected(self):
        """Test that a cheap_conformer alone is not accepted as a structure specification.

        from_dict() records it, but the structure check at arc/species/species.py:1035-1038 counts only
        mol, initial_xyz, final_xyz and conformers, so such a species raises SpeciesError on load --
        past the seam where the CLI can report it cleanly.
        """
        with self.assertRaises(ValidationError) as cm:
            Species(label='spc', cheap_conformer='C 0 0 0')
        self.assertIn('structure specification', str(cm.exception))
        # Every other source the validator accepts does load; this one was the only outlier.
        self.assertEqual(Species(label='spc', smiles='C').smiles, 'C')
        conformer = {'symbols': ('C',), 'isotopes': (12,), 'coords': ((0.0, 0.0, 0.0),)}
        self.assertIsNotNone(Species(label='spc', conformers=[conformer]).conformers)

    def test_a_yml_path_only_species_needs_no_label(self):
        """Test that a species given only a yml_path is accepted, since ARCSpecies derives its label"""
        spc = Species(yml_path='/path/to/vinoxy.yml')
        self.assertIsNone(spc.label)
        self.assertEqual(spc.yml_path, '/path/to/vinoxy.yml')
        self.assertEqual(Species(label='vinoxy', yml_path='/path/to/vinoxy.yml').label, 'vinoxy')

        with self.assertRaises(ValidationError) as cm:
            Species(smiles='C=C[O]')
        self.assertIn('label', str(cm.exception))


class TestReaction(unittest.TestCase):
    """
    Tests for the Reaction model.
    """

    def test_reaction_typo_key_rejected(self):
        """Test that a typo in a reaction key is rejected instead of being silently swallowed"""
        with self.assertRaises(ValidationError) as cm:
            Reaction(label='H + O2 <=> HO2', ts_xyz_gues=['H 0 0 0'])
        self.assertIn('ts_xyz_gues', str(cm.exception))

    def test_reaction_identification_requires_label_or_reactants_and_products(self):
        """Test that a reaction is identified by a label or by both reactants and products, and that
        r_species/p_species dicts alone are rejected: ARCReaction.from_dict runs
        set_label_reactants_products before converting those dicts, so they cannot identify a reaction
        at load time and ARC would raise"""
        self.assertEqual(Reaction(label='H + O2 <=> HO2').label, 'H + O2 <=> HO2')
        self.assertEqual(Reaction(reactants=['H', 'O2'], products=['HO2']).reactants, ['H', 'O2'])
        with self.assertRaises(ValidationError):
            Reaction(r_species=[{'label': 'H', 'smiles': '[H]'}],
                     p_species=[{'label': 'HO2', 'smiles': 'O[O]'}])
        with self.assertRaises(ValidationError):
            Reaction(reactants=['H', 'O2'])

    def test_family_own_reverse_rejects_an_integer_outside_the_legacy_zero_one_range(self):
        """Test that only the legacy 0/1 integers are accepted, since the setter collapses any int"""
        self.assertIs(Reaction(label='rxn1', family_own_reverse=1).family_own_reverse, True)
        self.assertIs(Reaction(label='rxn1', family_own_reverse=0).family_own_reverse, False)
        self.assertIs(Reaction(label='rxn1', family_own_reverse=True).family_own_reverse, True)
        for value in (-1, 2, 42):
            with self.assertRaises(ValidationError) as cm:
                Reaction(label='rxn1', family_own_reverse=value)
            self.assertIn('family_own_reverse', str(cm.exception))

    def test_a_nested_reaction_species_typo_is_rejected(self):
        """Test that r_species/p_species/ts_species receive the same validation as top-level species"""
        for key in ('r_species', 'p_species'):
            with self.assertRaises(ValidationError) as cm:
                Reaction(label='H + O2 <=> HO2', reactants=['H', 'O2'], products=['HO2'],
                         **{key: [{'label': 'H', 'smiles': '[H]', 'mutliplicity': 1}]})
            self.assertIn('mutliplicity', str(cm.exception))
        with self.assertRaises(ValidationError) as cm:
            Reaction(label='H + O2 <=> HO2', ts_species={'label': 'TS0', 'is_ts': True,
                                                         'ts_nmber': 0})
        self.assertIn('ts_nmber', str(cm.exception))

    def test_typed_nested_species_round_trip_preserves_the_exact_keys(self):
        """Test that typing the nested species lists does not inject keys ARCSpecies.from_dict gates on"""
        rxn = ARCReaction(r_species=[ARCSpecies(label='H', smiles='[H]'),
                                     ARCSpecies(label='O2', smiles='[O][O]')],
                          p_species=[ARCSpecies(label='HO2', smiles='O[O]')])
        rxn.ts_species = ARCSpecies(label='TS0', is_ts=True, xyz='O 0 0 0\nO 0 0 1.3\nH 0 0 2.3')
        rxn_dict = rxn.as_dict()
        validated = ARCInput(project='nested_round_trip',
                             reactions=[copy.deepcopy(rxn_dict)]).as_input_dict()['reactions'][0]
        for key in ('r_species', 'p_species'):
            self.assertEqual(len(validated[key]), len(rxn_dict[key]))
            for validated_spc, raw_spc in zip(validated[key], rxn_dict[key]):
                self.assertEqual(set(validated_spc.keys()), set(raw_spc.keys()))
        self.assertEqual(set(validated['ts_species'].keys()), set(rxn_dict['ts_species'].keys()))

    def test_reaction_schema(self):
        """Test creating an instance of Reaction"""
        rxn = Reaction(label='H + O2 <=> HO2')
        self.assertEqual(rxn.label, 'H + O2 <=> HO2')
        rxn = Reaction(label='N2H4 + NH <=> N2H3 + NH2', multiplicity=3,
                       ts_xyz_guess=['N 0 0 0\nH 0 0 1'])
        self.assertEqual(rxn.multiplicity, 3)
        rxn = Reaction(reactants=['H', 'O2'], products=['HO2'])
        self.assertEqual(rxn.reactants, ['H', 'O2'])

        with self.assertRaises(ValidationError) as cm:
            Reaction(label='rxn1', multiplicity=0)
        self.assertIn('multiplicity', str(cm.exception))

        with self.assertRaises(ValidationError) as cm:
            Reaction(label='rxn1', ts_xyz_guess=5)
        self.assertIn('ts_xyz_guess', str(cm.exception))

        with self.assertRaises(ValidationError):
            Reaction(multiplicity=1)

    def test_extensions_blocks_validate_and_pass_through_verbatim(self):
        """Test that species and reaction extensions blocks are accepted as opaque mappings and kept as given"""
        t3_block = {'qm_label': 's12_CH3', 'rmg_label': {0: '[CH3](15)'}, 'anything_new': [1, 2]}
        species = [{'label': 'CH3', 'smiles': '[CH3]', 'extensions': {'t3': t3_block}}, {'label': 'H', 'smiles': '[H]'}]
        reactions = [{'label': 'CH3 + H <=> CH4', 'extensions': {'t3': {'t3_index': 3}}}]
        validated = validate_input_dict({'project': 'p', 'species': species, 'reactions': reactions})
        self.assertEqual(validated['species'][0]['extensions'], {'t3': t3_block})
        self.assertNotIn('extensions', validated['species'][1])
        self.assertEqual(validated['reactions'][0]['extensions'], {'t3': {'t3_index': 3}})
        for bad in ({'t3': 5}, ['t3'], 't3'):
            with self.assertRaises(ValidationError) as cm:
                validate_input_dict({'project': 'p', 'species': [{'label': 'H', 'smiles': '[H]', 'extensions': bad}]})
            self.assertIn('extensions', str(cm.exception))
            with self.assertRaises(ValidationError) as cm:
                validate_input_dict({'project': 'p', 'species': [{'label': 'H', 'smiles': '[H]', 'qm_label': 'a',
                                                                  'extensions': bad}]})
            self.assertIn('extensions', str(cm.exception))

    def test_legacy_flat_t3_species_keys_are_lifted_into_extensions(self):
        """Test that the frozen flat T3 keys are moved into extensions['t3'], while other unknown keys still fail"""
        flat = {'key': 12, 'qm_label': 's12_CH3', 't3_status': 'pending', 'thermo_method': None}
        validated = validate_input_dict({'project': 'p',
                                         'species': [{'label': 'CH3', 'smiles': '[CH3]', **flat}],
                                         'reactions': [{'label': 'CH3 <=> CH3',
                                                        'r_species': [{'label': 'CH3', **flat}]}]})
        self.assertEqual(validated['species'][0], {'label': 'CH3', 'smiles': '[CH3]', 'extensions': {'t3': flat}})
        self.assertEqual(validated['reactions'][0]['r_species'][0], {'label': 'CH3', 'extensions': {'t3': flat}})
        with self.assertRaises(ValidationError) as cm:
            validate_input_dict({'project': 'p', 'species': [{'label': 'CH3', 'smiles': '[CH3]', 'qm_lable': 'x'}]})
        self.assertIn('qm_lable', str(cm.exception))
        with self.assertRaises(ValidationError) as cm:
            validate_input_dict({'project': 'p', 'species': [{'label': 'CH3', 'smiles': '[CH3]', 'key': 12,
                                                              'extensions': {'t3': {'key': 13}}}]})
        self.assertIn('key', str(cm.exception))

    def test_reaction_species_may_be_label_only_references(self):
        """Test that label-only r_species/p_species entries validate, as ARC resolves them against the species list"""
        def make_input():
            return {'project': 'label_only_stubs',
                    'species': [{'label': 'CHO2', 'smiles': '[O]C=O'}, {'label': 'CO2', 'smiles': 'O=C=O'},
                                {'label': 'H', 'smiles': '[H]'}],
                    'reactions': [{'label': 'CHO2 <=> CO2 + H', 'r_species': [{'label': 'CHO2'}],
                                   'p_species': [{'label': 'CO2'}, {'label': 'H'}]}]}
        validated = validate_input_dict(make_input())
        self.assertEqual(validated['reactions'][0]['r_species'], [{'label': 'CHO2'}])
        raw = make_input()
        raw['project_directory'], raw['calc_freq_factor'] = tempfile.mkdtemp(prefix='arc_test_schema_'), False
        self.addCleanup(shutil.rmtree, raw['project_directory'], ignore_errors=True)
        arc_object = ARC(**raw)
        self.assertEqual(arc_object.reactions[0].r_species[0].mol.to_smiles(), '[O]C=O')
        with self.assertRaises(ValidationError) as cm:
            validate_input_dict({'project': 'p', 'species': [{'label': 'CHO2'}]})
        self.assertIn('structure specification', str(cm.exception))

    def test_reaction_accepts_a_single_xyz_dict(self):
        """Test that a single xyz dict is accepted for ts_xyz_guess and xyz, as ARCReaction wraps it in a list"""
        xyz = {'symbols': ('H', 'H'), 'isotopes': (1, 1), 'coords': ((0.0, 0.0, 0.0), (0.0, 0.0, 0.74))}
        rxn = Reaction(label='H + H <=> H2', ts_xyz_guess=xyz, xyz=xyz)
        self.assertEqual(rxn.ts_xyz_guess, xyz)
        arc_rxn = ARCReaction(label='H + H <=> H2', ts_xyz_guess=xyz)
        self.assertEqual(arc_rxn.ts_xyz_guess, [xyz])


class TestSchemaDrift(unittest.TestCase):
    """
    Drift guards between the schema and the ARC objects it mirrors.
    """

    def test_arc_input_fields_match_arc_init_kwargs(self):
        """Drift guard: ARCInput's modeled fields must exactly mirror ARC.__init__'s kwargs"""
        init_params = set(inspect.signature(ARC.__init__).parameters) - {'self'}
        self.assertEqual(set(ARCInput.model_fields), init_params,
                         ('ARCInput.model_fields and ARC.__init__ kwargs diverged. Since ARCInput forbids unknown '
                          'top-level keys, adding an ARC.__init__ kwarg without modeling it in arc/schema.py would '
                          'make valid inputs fail validation (and vice versa) — update both together.'))

    def test_get_arc_input_keys_matches_arc_init_kwargs(self):
        """Test that get_arc_input_keys() returns exactly the keyword arguments ARC.__init__ accepts"""
        self.assertEqual(get_arc_input_keys(), set(inspect.signature(ARC.__init__).parameters) - {'self'})
        self.assertIn('project', get_arc_input_keys())

    def test_job_types_covers_all_default_job_types(self):
        """Drift guard: every key in settings.default_job_types must be a JobTypes field, so that
        adding a job type to arc/settings/settings.py cannot silently leave it untyped"""
        self.assertLessEqual(set(default_job_types), set(JobTypes.model_fields))

    def test_job_types_defaults_follow_the_settings_overlay(self):
        """Test that the schema reads default_job_types from arc.imports.settings, the dict ARC itself uses"""
        overlaid = dict(settings['default_job_types'], conf_opt=not settings['default_job_types']['conf_opt'])
        schema_module = importlib.import_module('arc.schema')
        self.addCleanup(importlib.reload, schema_module)
        with patch.dict(settings, {'default_job_types': overlaid}):
            reloaded = importlib.reload(schema_module)
        self.assertEqual(reloaded.JobTypes().conf_opt, overlaid['conf_opt'])

    def test_species_as_dict_keys_are_modeled(self):
        """Drift guard: every key ARCSpecies.as_dict() writes must be modeled by Species.

        ``ARCSpecies.as_dict()`` emits most keys conditionally, so a default-valued species exercises
        only part of the restart shape. The stability attributes below are therefore populated before
        collecting the keys: left at their defaults they are omitted, and the guard would pass while
        the schema silently failed to model them.
        """
        spc = ARCSpecies(label='vinoxy', smiles='C=C[O]')
        spc.derived_stability_verdict = {'stable': False}
        spc.scf_references = {'opt_a1234': 'uhf'}
        spc.stability_analysis_ran = True
        spc.stability_pending_opt_job = 'opt_a1234'
        spc.stability_reoptimized = True
        ts = ARCSpecies(label='TS0', is_ts=True, xyz='O 0 0 0\nH 0 0 1\nH 0 1 0')
        emitted_keys = set(spc.as_dict().keys()) | set(ts.as_dict().keys())
        self.assertTrue(emitted_keys.issubset(set(Species.model_fields)),
                        (f'ARC writes a species key the schema does not model — add it to arc/schema.py or '
                         f'restarts will break. Unmodeled: {sorted(emitted_keys - set(Species.model_fields))}'))

    def test_reaction_as_dict_keys_are_modeled(self):
        """Drift guard: every key ARCReaction.as_dict() writes must be modeled by Reaction"""
        rxn = ARCReaction(r_species=[ARCSpecies(label='H', smiles='[H]'),
                                     ARCSpecies(label='O2', smiles='[O][O]')],
                          p_species=[ARCSpecies(label='HO2', smiles='O[O]')])
        emitted_keys = set(rxn.as_dict().keys())
        self.assertTrue(emitted_keys.issubset(set(Reaction.model_fields)),
                        (f'ARC writes a reaction key the schema does not model — add it to arc/schema.py or '
                         f'restarts will break. Unmodeled: {sorted(emitted_keys - set(Reaction.model_fields))}'))

    def test_rotor_scan_resolution_schema_matches_arcs_own_check(self):
        """Test that the schema and arc.main.check_rotor_scan_resolution() agree on every value, incl. non-numbers"""
        for value in (None, 1, 2, 4, 4.0, 5, 10, 20, 7, 0.5, 21, 30, 45, 90, 180, 360, 361, True, False, '10', [10]):
            try:
                check_rotor_scan_resolution(value)
                arc_accepts = True
            except InputError:
                arc_accepts = False
            try:
                ARCInput(project='p', rotor_scan_resolution=value)
                schema_accepts = True
            except ValidationError:
                schema_accepts = False
            self.assertEqual(schema_accepts, arc_accepts,
                             (f'rotor_scan_resolution={value}: the schema {"accepts" if schema_accepts else "rejects"} '
                              f'it but arc.main.check_rotor_scan_resolution() '
                              f'{"accepts" if arc_accepts else "rejects"} it.'))


class TestRoundTrip(TmpDirTestCase):
    """
    Tests that real input and restart files validate and round-trip through the schema.
    """

    def test_all_examples_validate(self):
        """Test that all input.yml files under examples/ validate through ARCInput"""
        example_files = glob.glob(os.path.join(ARC_PATH, 'examples', '**', 'input.yml'), recursive=True)
        self.assertGreaterEqual(len(example_files), 8)
        for example_file in example_files:
            input_dict = read_yaml_file(example_file)
            arc_input = ARCInput(**input_dict)
            self.assertEqual(arc_input.project, input_dict['project'])

    def test_restart_files_round_trip(self):
        """Test that real restart.yml files validate and round-trip through ARCInput"""
        restart_files = glob.glob(os.path.join(ARC_PATH, 'arc', 'testing', 'restart', '**', 'restart.yml'),
                                  recursive=True)
        self.assertGreaterEqual(len(restart_files), 4)
        for restart_file in restart_files:
            restart_dict = read_yaml_file(restart_file)
            arc_input = ARCInput(**restart_dict)
            dumped = arc_input.as_input_dict()
            self.assertEqual(set(dumped.keys()), set(restart_dict.keys()), restart_file)
            for key, original in restart_dict.items():
                if key in ('thermo_adapter', 'kinetics_adapter'):
                    # The schema normalizes these to lowercase, mirroring ARC.__init__.
                    self.assertEqual(dumped[key], original.lower())
                else:
                    self.assertEqual(_normalize(dumped[key]), _normalize(original), f'{restart_file}: {key}')

    def test_a_restart_species_with_skipped_methods_validates(self):
        """Test that a TS whose skipped_methods is recorded validates on restart and restores the map"""
        ts = ARCSpecies(label='TS0', is_ts=True)
        ts.skipped_methods = {'autotst': 'AUTOTST_PYTHON is not set'}
        species_dict = yaml.safe_load(yaml.dump(ts.as_dict()))
        self.assertIn('skipped_methods', species_dict)
        validated = validate_input_dict({'project': 'p', 'species': [species_dict]})
        self.assertEqual(validated['species'][0]['skipped_methods'], {'autotst': 'AUTOTST_PYTHON is not set'})
        restored = ARCSpecies(species_dict=validated['species'][0])
        self.assertEqual(restored.skipped_methods, {'autotst': 'AUTOTST_PYTHON is not set'})

    def test_restart_thermo_runtime_keys_preserved(self):
        """Test that runtime restart keys survive validation as modeled fields"""
        restart_file = os.path.join(ARC_PATH, 'arc', 'testing', 'restart', '1_restart_thermo', 'restart.yml')
        restart_dict = read_yaml_file(restart_file)
        arc_input = ARCInput(**restart_dict)
        dumped = arc_input.as_input_dict()
        self.assertEqual(dumped['output'], restart_dict['output'])
        self.assertEqual(dumped['running_jobs'], restart_dict['running_jobs'])
        self.assertEqual(dumped['ess_settings'], restart_dict['ess_settings'])
        self.assertEqual(dumped['job_types'], restart_dict['job_types'])
        for original_spc, dumped_spc in zip(restart_dict['species'], dumped['species']):
            self.assertEqual(set(dumped_spc.keys()), set(original_spc.keys()))

    def test_arc_object_as_dict_round_trip(self):
        """Test that an ARC object's as_dict() output validates through ARCInput"""
        tmp_dir = self.make_tmp_dir()
        arc_object = ARC(project='arc_schema_test',
                         project_directory=tmp_dir,
                         species=[{'label': 'H2', 'smiles': '[H][H]'}],
                         )
        arc_dict = arc_object.as_dict()
        self.assertEqual(arc_dict['schema_version'], ARC_INPUT_SCHEMA_VERSION)
        self.assertEqual(arc_dict['arc_version'], VERSION)
        arc_input = ARCInput(**arc_dict)
        dumped = arc_input.as_input_dict()
        self.assertEqual(set(dumped.keys()), set(arc_dict.keys()) - {'schema_version', 'arc_version'})
        self.assertEqual(dumped['project'], 'arc_schema_test')
        self.assertEqual(dumped['job_types'], arc_dict['job_types'])
        self.assertEqual(_normalize(dumped['species']), _normalize(arc_dict['species']))

    def test_a_t3_written_restart_file_loads_and_keeps_t3_data(self):
        """Test that a restart file written inside a T3 run (flat T3 keys, a python/tuple tag) validates,
        builds ARC, and is written back with the T3 data nested under extensions['t3']"""
        restart_text = (
            "project: t3_restart\n"
            "calc_freq_factor: false\n"
            "T_min: !!python/tuple [300, K]\n"
            "species:\n"
            "- label: s12_CH3\n"
            "  smiles: '[CH3]'\n"
            "  key: 12\n"
            "  qm_label: s12_CH3\n"
            "  rmg_label: {0: '[CH3](15)'}\n"
            "  thermo_method: GAV\n"
            "  thermo_source: null\n"
            "  thermo_comment: ''\n"
            "  t3_status: pending\n"
            "  created_at_iteration: 1\n"
            "  reasons: ['(i 1) the 2nd most sensitive species thermo for OH(3)']\n"
            "- label: TS0\n"
            "  is_ts: true\n")
        directory = self.make_tmp_dir()
        restart_path = os.path.join(directory, 'restart.yml')
        with open(restart_path, 'w') as f:
            f.write(restart_text)
        validated = validate_input_dict(read_yaml_file(restart_path))
        self.assertEqual(validated['species'][0]['extensions']['t3']['rmg_label'], {0: '[CH3](15)'})
        self.assertNotIn('extensions', validated['species'][1])
        validated['project_directory'] = directory
        written = ARC(**validated).as_dict()['species'][0]
        self.assertEqual(written['extensions']['t3']['qm_label'], 's12_CH3')
        self.assertEqual(written['extensions']['t3']['reasons'],
                         ['(i 1) the 2nd most sensitive species thermo for OH(3)'])
        self.assertNotIn('qm_label', written)


class TestSeam(TmpDirTestCase):
    """
    Tests that constructing ARC through the schema seam preserves behavior.
    """

    def test_schema_seam_is_behavior_preserving(self):
        """Test that constructing ARC through the schema seam is equivalent to constructing it
        directly from the raw dict, for real example input files"""
        example_files = [
            os.path.join(ARC_PATH, 'examples', 'minimal', 'input.yml'),
            os.path.join(ARC_PATH, 'examples', 'Stationary', 'thermo_demo', 'input.yml'),
            os.path.join(ARC_PATH, 'examples', 'Reactions', 'rates_demo', 'input.yml'),
            os.path.join(ARC_PATH, 'arc', 'testing', 'restart', '2_restart_rate', 'restart.yml'),
            os.path.join(ARC_PATH, 'arc', 'testing', 'restart', '5_TS1', 'restart.yml'),
        ]
        for example_file in example_files:
            with self.subTest(example_file=example_file):
                tmp_dir = self.make_tmp_dir()
                raw_dict = read_yaml_file(example_file)

                # ARC.__init__ mutates the 'species'/'reactions' lists it is given in place (converting
                # dicts to ARCSpecies/ARCReaction objects), so each construction needs its own copy of raw_dict.
                arc_raw = ARC(**copy.deepcopy(raw_dict), project_directory=os.path.join(tmp_dir, 'raw'))
                schema_input_dict = ARCInput(**copy.deepcopy(raw_dict)).as_input_dict()
                schema_input_dict['project_directory'] = os.path.join(tmp_dir, 'schema')
                arc_schema = ARC(**schema_input_dict)

                raw_as_dict = arc_raw.as_dict()
                schema_as_dict = arc_schema.as_dict()

                self.assertEqual(set(raw_as_dict.keys()), set(schema_as_dict.keys()))
                for key in raw_as_dict:
                    if key in _AS_DICT_EXCLUDED_KEYS:
                        continue
                    raw_value = _normalize_transient_mol_atom_ids(raw_as_dict[key])
                    schema_value = _normalize_transient_mol_atom_ids(schema_as_dict[key])
                    self.assertEqual(raw_value, schema_value,
                                     f'{example_file}: key {key!r} differs: '
                                     f'{raw_as_dict[key]!r} != {schema_as_dict[key]!r}')


class TestSchemaVersion(TmpDirTestCase):
    """
    Tests for schema versioning and validate_input_dict().
    """

    def test_schema_version_absent_accepted(self):
        """Test that an input without a schema_version key validates silently"""
        arc_input = ARCInput(project='p')
        self.assertIsNone(arc_input.schema_version)
        self.assertNotIn('schema_version', arc_input.as_input_dict())

    def test_schema_version_current_accepted(self):
        """Test that the current schema version is accepted"""
        arc_input = ARCInput(project='p', schema_version=ARC_INPUT_SCHEMA_VERSION)
        self.assertEqual(arc_input.schema_version, ARC_INPUT_SCHEMA_VERSION)

    def test_schema_version_explicit_null_rejected(self):
        """Test that an explicit schema_version: null raises"""
        with self.assertRaises(ValidationError) as cm:
            ARCInput(project='p', schema_version=None)
        self.assertIn('schema_version', str(cm.exception))

    def test_schema_version_newer_rejected(self):
        """Test that a schema_version from a future ARC raises with both versions named"""
        with self.assertRaises(ValidationError) as cm:
            ARCInput(project='p', schema_version=ARC_INPUT_SCHEMA_VERSION + 1)
        self.assertIn('schema_version', str(cm.exception))
        self.assertIn(str(ARC_INPUT_SCHEMA_VERSION + 1), str(cm.exception))
        self.assertIn(str(ARC_INPUT_SCHEMA_VERSION), str(cm.exception))

    def test_schema_version_older_warns_and_proceeds(self):
        """Test that an older schema_version logs a warning and still validates"""
        with self.assertLogs(get_logger(), level=logging.WARNING) as logs:
            arc_input = ARCInput(project='p', schema_version=0)
        self.assertEqual(arc_input.schema_version, 0)
        self.assertTrue(any(record.levelno == logging.WARNING and 'schema_version 0' in record.getMessage()
                            and str(ARC_INPUT_SCHEMA_VERSION) in record.getMessage()
                            for record in logs.records))

    def test_schema_version_and_arc_version_stripped_from_input_dict(self):
        """Test that as_input_dict() strips schema_version and arc_version"""
        arc_input = ARCInput(project='p', schema_version=ARC_INPUT_SCHEMA_VERSION, arc_version=VERSION)
        dumped = arc_input.as_input_dict()
        self.assertNotIn('schema_version', dumped)
        self.assertNotIn('arc_version', dumped)
        self.assertEqual(dumped, {'project': 'p'})

    def test_validate_input_dict_minimal_example(self):
        """Test that ARC constructs from validate_input_dict() on a real example input file"""
        tmp_dir = self.make_tmp_dir()
        input_dict = read_yaml_file(os.path.join(ARC_PATH, 'examples', 'minimal', 'input.yml'))
        input_dict['project_directory'] = tmp_dir
        arc_object = ARC(**validate_input_dict(input_dict))
        self.assertEqual(arc_object.project, input_dict['project'])

    def test_validate_input_dict_versioned_round_trip(self):
        """Test a full round trip: construct ARC, as_dict() it (now versioned), validate, reconstruct.
        Also test that raw ARC(**restart_dict) tolerates the version keys directly"""
        tmp_dir = self.make_tmp_dir()
        arc_object = ARC(project='arc_schema_version_round_trip',
                         project_directory=os.path.join(tmp_dir, 'first'),
                         species=[{'label': 'H2', 'smiles': '[H][H]'}],
                         )
        restart_dict = arc_object.as_dict()
        self.assertEqual(restart_dict['schema_version'], ARC_INPUT_SCHEMA_VERSION)
        self.assertEqual(restart_dict['arc_version'], VERSION)

        validated = validate_input_dict(copy.deepcopy(restart_dict))
        self.assertNotIn('schema_version', validated)
        self.assertNotIn('arc_version', validated)
        validated['project_directory'] = os.path.join(tmp_dir, 'second')
        arc_object_2 = ARC(**validated)
        self.assertEqual(arc_object_2.project, 'arc_schema_version_round_trip')
        self.assertEqual(arc_object_2.__version__, VERSION)

        raw_dict = copy.deepcopy(restart_dict)
        raw_dict['project_directory'] = os.path.join(tmp_dir, 'third')
        arc_object_3 = ARC(**raw_dict)
        self.assertEqual(arc_object_3.project, 'arc_schema_version_round_trip')
        self.assertEqual(arc_object_3.__version__, VERSION)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
