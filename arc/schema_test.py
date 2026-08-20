#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests for the arc.schema module
"""

import copy
import glob
import inspect
import logging
import os

import pytest
import yaml
from pydantic import ValidationError

from arc.common import ARC_INPUT_SCHEMA_VERSION, ARC_PATH, VERSION, get_logger, read_yaml_file
from arc.main import ARC
from arc.reaction import ARCReaction
from arc.schema import ARCInput, BACTypeEnum, JobTypes, Reaction, Species, validate_input_dict
from arc.settings.settings import default_job_types
from arc.species.species import ARCSpecies


def test_arc_input_minimal():
    """Test creating a minimal ARCInput instance"""
    arc_input = ARCInput(project='test_schema')
    assert arc_input.project == 'test_schema'
    assert arc_input.verbose == 20
    assert arc_input.bac_type == 'p'
    assert arc_input.thermo_adapter == 'arkane'
    assert arc_input.kinetics_adapter == 'arkane'
    assert arc_input.n_confs == 10
    assert arc_input.e_confs == 5.0
    assert arc_input.T_count == 50
    assert arc_input.level_of_theory == ''
    assert arc_input.specific_job_type == ''
    assert arc_input.species is None
    assert arc_input.reactions is None
    assert arc_input.compute_thermo is True
    assert arc_input.compute_rates is True
    assert arc_input.compute_transport is False
    assert arc_input.allow_nonisomorphic_2d is False
    assert arc_input.trsh_ess_jobs is True
    assert arc_input.trsh_rotors is True


def test_arc_input_requires_project():
    """Test that a project name is required"""
    with pytest.raises(ValidationError) as excinfo:
        ARCInput()
    assert 'project' in str(excinfo.value)


def test_arc_input_bac_type():
    """Test the bac_type field"""
    arc_input = ARCInput(project='p', bac_type='m')
    assert arc_input.bac_type == 'm'
    arc_input = ARCInput(project='p', bac_type=None)
    assert arc_input.bac_type is None
    with pytest.raises(ValidationError) as excinfo:
        ARCInput(project='p', bac_type='x')
    assert 'bac_type' in str(excinfo.value)
    assert [e.value for e in BACTypeEnum] == ['p', 'm']


def test_arc_input_bac_type_is_case_insensitive():
    """Test that bac_type is lowercased before enum validation (arc/arkane.py compares
    bac_type.lower() == 'm'), but the spelled-out name is still rejected since arc/arkane.py
    writes the raw value straight into the Arkane input as bondCorrectionType"""
    arc_input = ARCInput(project='p', bac_type='P')
    assert arc_input.bac_type == 'p'
    arc_input = ARCInput(project='p', bac_type='M')
    assert arc_input.bac_type == 'm'
    with pytest.raises(ValidationError) as excinfo:
        ARCInput(project='p', bac_type='petersson')
    assert 'bac_type' in str(excinfo.value)
    with pytest.raises(ValidationError) as excinfo:
        ARCInput(project='p', bac_type='PETERSSON')
    assert 'bac_type' in str(excinfo.value)


def test_arc_input_statmech_adapters():
    """Test the thermo_adapter and kinetics_adapter fields"""
    arc_input = ARCInput(project='p', thermo_adapter='Arkane', kinetics_adapter='Arkane')
    assert arc_input.thermo_adapter == 'arkane'
    assert arc_input.kinetics_adapter == 'arkane'
    with pytest.raises(ValidationError) as excinfo:
        ARCInput(project='p', thermo_adapter='mess')
    assert 'thermo_adapter' in str(excinfo.value)
    with pytest.raises(ValidationError) as excinfo:
        ARCInput(project='p', kinetics_adapter='unsupported')
    assert 'kinetics_adapter' in str(excinfo.value)


def test_arc_input_ts_adapters():
    """Test the ts_adapters field validator"""
    arc_input = ARCInput(project='p', ts_adapters=['heuristics', 'GCN'])
    assert arc_input.ts_adapters == ['heuristics', 'GCN']
    arc_input = ARCInput(project='p', ts_adapters=[])
    assert arc_input.ts_adapters == list()
    with pytest.raises(ValidationError) as excinfo:
        ARCInput(project='p', ts_adapters=['no_such_adapter'])
    assert 'ts_adapters' in str(excinfo.value)


def test_arc_input_numeric_constraints():
    """Test numeric field constraints"""
    with pytest.raises(ValidationError) as excinfo:
        ARCInput(project='p', n_confs=0)
    assert 'n_confs' in str(excinfo.value)
    with pytest.raises(ValidationError) as excinfo:
        ARCInput(project='p', e_confs=-1.0)
    assert 'e_confs' in str(excinfo.value)
    with pytest.raises(ValidationError) as excinfo:
        ARCInput(project='p', T_count=-5)
    assert 'T_count' in str(excinfo.value)
    with pytest.raises(ValidationError) as excinfo:
        ARCInput(project='p', max_job_time=-1)
    assert 'max_job_time' in str(excinfo.value)
    with pytest.raises(ValidationError) as excinfo:
        ARCInput(project='p', job_memory=-14)
    assert 'job_memory' in str(excinfo.value)
    with pytest.raises(ValidationError) as excinfo:
        ARCInput(project='p', freq_scale_factor=0)
    assert 'freq_scale_factor' in str(excinfo.value)
    with pytest.raises(ValidationError) as excinfo:
        ARCInput(project='p', verbose=25)
    assert 'verbose' in str(excinfo.value)
    # max_job_time=0 is falsy and treated by ARC as "use the default", so it must be accepted.
    assert ARCInput(project='p', max_job_time=0).max_job_time == 0


def test_arc_input_optional_string_fields_accept_none():
    """Test that level_of_theory and specific_job_type accept an explicit null, as real restart
    files and a bare input can carry, rather than only a string"""
    arc_input = ARCInput(project='p', level_of_theory=None, specific_job_type=None)
    assert arc_input.level_of_theory is None
    assert arc_input.specific_job_type is None


def test_arc_input_bool_flags_accept_int_0_1():
    """Test that a run flag spelled as an integer 0/1 (a previously-working raw ARC input) is
    coerced to a bool, while a quoted string or an out-of-range int still fails strict-bool
    validation and cannot silently invert the flag"""
    arc_input = ARCInput(project='p', compute_thermo=1, trsh_rotors=0)
    assert arc_input.compute_thermo is True
    assert arc_input.trsh_rotors is False
    for bad in ('False', 2):
        with pytest.raises(ValidationError) as excinfo:
            ARCInput(project='p', compute_thermo=bad)
        assert 'compute_thermo' in str(excinfo.value)


def test_arc_input_integral_float_coercion():
    """Test that job_memory, n_confs, and T_count accept an integral float (a previously-working
    raw ARC input value) while still rejecting a non-integral float and a numeric string, so
    strict-int validation is preserved for genuinely invalid input"""
    arc_input = ARCInput(project='p', job_memory=14.0, n_confs=14.0, T_count=14.0)
    assert arc_input.job_memory == 14
    assert arc_input.n_confs == 14
    assert arc_input.T_count == 14
    for field in ('job_memory', 'n_confs', 'T_count'):
        with pytest.raises(ValidationError) as excinfo:
            ARCInput(project='p', **{field: 14.7})
        assert field in str(excinfo.value)
        with pytest.raises(ValidationError) as excinfo:
            ARCInput(project='p', **{field: '14'})
        assert field in str(excinfo.value)


def test_arc_input_verbose_accepts_full_logging_level_range():
    """Test that verbose accepts all standard logging levels, including ERROR/CRITICAL"""
    for level in (10, 20, 30, 40, 50):
        arc_input = ARCInput(project='p', verbose=level)
        assert arc_input.verbose == level
    for level in (15, 35, 60):
        with pytest.raises(ValidationError) as excinfo:
            ARCInput(project='p', verbose=level)
        assert 'verbose' in str(excinfo.value)


def test_arc_input_verbose_rejects_explicit_none_but_allows_absent():
    """Test that verbose=None raises (logger.setLevel(None) crashes ARC), while an absent
    verbose still validates and keeps its default"""
    with pytest.raises(ValidationError) as excinfo:
        ARCInput(project='p', verbose=None)
    assert 'verbose' in str(excinfo.value)
    arc_input = ARCInput(project='p')
    assert arc_input.verbose == 20
    assert 'verbose' not in arc_input.as_input_dict()


def test_arc_input_t_min_t_max():
    """Test the T_min and T_max fields"""
    arc_input = ARCInput(project='p', T_min=(300, 'K'), T_max=[3000, 'K'])
    assert arc_input.T_min == (300.0, 'K')
    assert arc_input.T_max == (3000.0, 'K')
    with pytest.raises(ValidationError) as excinfo:
        ARCInput(project='p', T_min=('K', 300))
    assert 'T_min' in str(excinfo.value)


def test_arc_input_t_min_t_max_dumps_as_plain_list():
    """Test that T_min/T_max are plain lists (not tuples) after as_input_dict(), for yaml.dump safety"""
    arc_input = ARCInput(project='p', T_min=[300, 'K'], T_max=[3000, 'K'])
    dumped = arc_input.as_input_dict()
    assert isinstance(dumped['T_min'], list)
    assert isinstance(dumped['T_max'], list)
    assert dumped['T_min'] == [300.0, 'K']
    dumped_yaml = yaml.dump({'T_min': dumped['T_min'], 'T_max': dumped['T_max']})
    assert 'python/tuple' not in dumped_yaml


def test_arc_input_t_min_t_max_accept_bare_scalar():
    """Test that a bare scalar T_min/T_max (raw ARC's shorthand for Kelvin) validates"""
    arc_input = ARCInput(project='p', T_min=500, T_max=3000)
    assert arc_input.T_min == 500
    assert arc_input.T_max == 3000
    dumped = arc_input.as_input_dict()
    # A scalar in must stay a scalar out; processor.py normalizes it downstream.
    assert dumped['T_min'] == 500
    assert isinstance(dumped['T_min'], float | int)
    assert dumped['T_max'] == 3000
    assert isinstance(dumped['T_max'], float | int)
    dumped_yaml = yaml.dump({'T_min': dumped['T_min'], 'T_max': dumped['T_max']})
    assert 'python/tuple' not in dumped_yaml


def test_arc_input_t_min_t_max_pair_form_still_works():
    """Regression check: the (value, unit) pair form must still validate and dump as a plain list"""
    arc_input = ARCInput(project='p', T_min=[300, 'K'], T_max=(3000, 'K'))
    assert arc_input.T_min == (300.0, 'K')
    assert arc_input.T_max == (3000.0, 'K')
    dumped = arc_input.as_input_dict()
    assert isinstance(dumped['T_min'], list)
    assert dumped['T_min'] == [300.0, 'K']
    assert isinstance(dumped['T_max'], list)
    assert dumped['T_max'] == [3000.0, 'K']


def test_arc_input_t_min_t_max_reject_non_kelvin_unit():
    """Test that a non-Kelvin unit raises, since no consumer ever reads or converts the unit
    (plotter.py/processor.py silently treat T_min[0]/T_max[0] as Kelvin regardless of the label)"""
    with pytest.raises(ValidationError) as excinfo:
        ARCInput(project='p', T_min=[300, 'C'])
    assert 'T_min' in str(excinfo.value)
    with pytest.raises(ValidationError) as excinfo:
        ARCInput(project='p', T_max=[3000, 'F'])
    assert 'T_max' in str(excinfo.value)
    arc_input = ARCInput(project='p', T_min=[300, 'K'])
    assert arc_input.T_min == (300.0, 'K')


def test_arc_input_t_min_t_max_reject_non_positive_temperature():
    """Test that a non-positive absolute temperature raises, in both scalar and pair form"""
    with pytest.raises(ValidationError) as excinfo:
        ARCInput(project='p', T_min=-5)
    assert 'T_min' in str(excinfo.value)
    with pytest.raises(ValidationError) as excinfo:
        ARCInput(project='p', T_min=0)
    assert 'T_min' in str(excinfo.value)
    with pytest.raises(ValidationError) as excinfo:
        ARCInput(project='p', T_min=[0, 'K'])
    assert 'T_min' in str(excinfo.value)
    arc_input = ARCInput(project='p', T_min=500)
    assert arc_input.T_min == 500
    assert arc_input.as_input_dict()['T_min'] == 500


def test_arc_input_t_min_t_max_accept_none():
    """Test that T_min: null / T_max: null still validate, since real restart files have them"""
    arc_input = ARCInput(project='p', T_min=None, T_max=None)
    assert arc_input.T_min is None
    assert arc_input.T_max is None


def test_arc_input_species_and_reactions_accept_none():
    """Test that explicit species: null / reactions: null validate, mirroring ARC.__init__"""
    arc_input = ARCInput(project='p', species=None, reactions=None)
    assert arc_input.species is None
    assert arc_input.reactions is None


def test_arc_input_wrong_job_types():
    """Test that a wrongly-typed job_types raises"""
    with pytest.raises(ValidationError) as excinfo:
        ARCInput(project='p', job_types='rotors')
    assert 'job_types' in str(excinfo.value)


def test_arc_input_extra_keys_forbidden():
    """Test that an unknown top-level key is rejected with an error citing the key"""
    with pytest.raises(ValidationError) as excinfo:
        ARCInput(project='p', some_unknown_key='value')
    assert 'some_unknown_key' in str(excinfo.value)


def test_arc_input_fields_match_arc_init_kwargs():
    """Drift guard: ARCInput's modeled fields must exactly mirror ARC.__init__'s kwargs"""
    init_params = set(inspect.signature(ARC.__init__).parameters) - {'self'}
    assert set(ARCInput.model_fields) == init_params, \
        ('ARCInput.model_fields and ARC.__init__ kwargs diverged. Since ARCInput forbids unknown '
         'top-level keys, adding an ARC.__init__ kwarg without modeling it in arc/schema.py would '
         'make valid inputs fail validation (and vice versa) — update both together.')


def test_job_types_extra_keys_forbidden():
    """Test that a typo inside job_types is rejected at load rather than silently ignored"""
    with pytest.raises(ValidationError) as excinfo:
        ARCInput(project='p', job_types={'fine_grd': True})
    assert 'fine_grd' in str(excinfo.value)


def test_job_types_covers_all_default_job_types():
    """Drift guard: every key in settings.default_job_types must be a JobTypes field, so that
    adding a job type to arc/settings/settings.py cannot silently leave it untyped"""
    assert set(default_job_types) <= set(JobTypes.model_fields)


def test_job_types_schema():
    """Test creating an instance of JobTypes"""
    job_types = JobTypes()
    assert job_types.conf_opt is True
    assert job_types.conf_sp is False
    assert job_types.opt is True
    assert job_types.fine_grid is True
    assert job_types.freq is True
    assert job_types.sp is True
    assert job_types.rotors is True
    assert job_types.irc is True
    assert job_types.orbitals is False
    assert job_types.lennard_jones is False
    assert job_types.bde is False

    job_types = JobTypes(fine=True, onedmin=False)
    assert job_types.fine is True
    assert job_types.onedmin is False

    with pytest.raises(ValidationError) as excinfo:
        JobTypes(rotors='sometimes')
    assert 'rotors' in str(excinfo.value)


def test_job_types_restart_form_keys_validated():
    """Test that the restart-form job_types keys (fine, onedmin) are typed and validated"""
    with pytest.raises(ValidationError) as excinfo:
        ARCInput(project='p', job_types={'fine': 'definitely'})
    assert 'fine' in str(excinfo.value)
    with pytest.raises(ValidationError) as excinfo:
        ARCInput(project='p', job_types={'onedmin': 'maybe'})
    assert 'onedmin' in str(excinfo.value)
    arc_input = ARCInput(project='p', job_types={'fine': True, 'onedmin': False})
    assert arc_input.job_types.fine is True
    assert arc_input.job_types.onedmin is False


def test_job_types_fine_reject_explicit_none_but_allow_absent():
    """Test that an explicit fine null raises, since initialize_job_types only fills fine's default
    (True) when the key is absent, so an explicit null would silently disable fine-grid opt. An
    explicit onedmin null is allowed: onedmin defaults to False, so null is identical to absent"""
    with pytest.raises(ValidationError) as excinfo:
        ARCInput(project='p', job_types={'fine': None})
    assert 'fine' in str(excinfo.value)

    arc_input = ARCInput(project='p', job_types={'onedmin': None})
    assert arc_input.job_types.onedmin is None

    arc_input = ARCInput(project='p', job_types={'fine': True})
    assert arc_input.job_types.fine is True

    arc_input = ARCInput(project='p', job_types={})
    assert arc_input.job_types.fine is None
    assert 'fine' not in arc_input.as_input_dict()['job_types']


def test_job_types_contradictory_fine_aliases_raise():
    """Test that contradictory fine_grid/fine values raise, since fine_grid silently wins in
    ``initialize_job_types`` and the user's explicit ``fine`` value would otherwise be dropped"""
    with pytest.raises(ValidationError) as excinfo:
        JobTypes(fine_grid=True, fine=False)
    assert 'fine_grid' in str(excinfo.value)
    assert 'fine' in str(excinfo.value)


def test_job_types_contradictory_lennard_jones_aliases_raise():
    """Test that contradictory lennard_jones/onedmin values raise, since lennard_jones silently
    wins in ``initialize_job_types`` and the user's explicit ``onedmin`` value would otherwise be dropped"""
    with pytest.raises(ValidationError) as excinfo:
        JobTypes(lennard_jones=True, onedmin=False)
    assert 'lennard_jones' in str(excinfo.value)
    assert 'onedmin' in str(excinfo.value)


def test_job_types_agreeing_aliases_allowed():
    """Test that both aliases of a pair set to the same value is allowed (no information lost)"""
    job_types = JobTypes(fine_grid=True, fine=True)
    assert job_types.fine_grid is True
    assert job_types.fine is True
    job_types = JobTypes(lennard_jones=False, onedmin=False)
    assert job_types.lennard_jones is False
    assert job_types.onedmin is False


def test_job_types_single_alias_only_still_allowed():
    """Test that specifying only one of a pair's aliases still validates unaffected"""
    job_types = JobTypes(fine_grid=False)
    assert job_types.fine_grid is False
    assert job_types.fine is None
    job_types = JobTypes(fine=False)
    assert job_types.fine is False
    job_types = JobTypes(lennard_jones=True)
    assert job_types.lennard_jones is True
    assert job_types.onedmin is None
    job_types = JobTypes(onedmin=True)
    assert job_types.onedmin is True


def test_strict_bool_fields_reject_quoted_strings():
    """Test that bool fields reject quoted string/int values instead of laxly coercing them"""
    with pytest.raises(ValidationError) as excinfo:
        ARCInput(project='p', compute_thermo='False')
    assert 'compute_thermo' in str(excinfo.value)
    with pytest.raises(ValidationError) as excinfo:
        ARCInput(project='p', job_types={'rotors': 'False'})
    assert 'rotors' in str(excinfo.value)
    with pytest.raises(ValidationError) as excinfo:
        Species(label='spc', smiles='CC', is_ts='False')
    assert 'is_ts' in str(excinfo.value)


def test_strict_bool_fields_accept_unquoted_yaml_bools():
    """Test that ordinary (unquoted) YAML booleans still validate after strictness is added"""
    yaml_str = """
    project: p
    compute_thermo: false
    compare_to_rmg: no
    """
    data = yaml.safe_load(yaml_str)
    arc_input = ARCInput(**data)
    assert arc_input.compute_thermo is False
    assert arc_input.compare_to_rmg is False


def test_species_schema():
    """Test creating an instance of Species"""
    spc = Species(label='H2', smiles='[H][H]')
    assert spc.label == 'H2'
    assert spc.smiles == '[H][H]'
    assert spc.is_ts is False

    spc = Species(label='vinoxy', smiles='C=C[O]', multiplicity=2, charge=0,
                  external_symmetry=1, optical_isomers=1, bdes=[(1, 2), 'all_h'])
    assert spc.multiplicity == 2
    assert spc.charge == 0

    with pytest.raises(ValidationError) as excinfo:
        Species(smiles='CC')
    assert 'label' in str(excinfo.value)

    with pytest.raises(ValidationError) as excinfo:
        Species(label='spc', smiles='CC', multiplicity=0)
    assert 'multiplicity' in str(excinfo.value)

    with pytest.raises(ValidationError) as excinfo:
        Species(label='spc', smiles='CC', external_symmetry=0)
    assert 'external_symmetry' in str(excinfo.value)

    with pytest.raises(ValidationError) as excinfo:
        Species(label='spc', smiles='CC', optical_isomers=-1)
    assert 'optical_isomers' in str(excinfo.value)

    with pytest.raises(ValidationError) as excinfo:
        Species(label='spc', smiles='CC', charge='neutral')
    assert 'charge' in str(excinfo.value)


def test_strict_int_fields_reject_quoted_strings():
    """Test that int fields reject quoted numeric strings instead of laxly coercing them"""
    with pytest.raises(ValidationError) as excinfo:
        Species(label='spc', smiles='CC', multiplicity='3')
    assert 'multiplicity' in str(excinfo.value)
    with pytest.raises(ValidationError) as excinfo:
        Species(label='spc', smiles='CC', charge='0')
    assert 'charge' in str(excinfo.value)
    spc = Species(label='spc', smiles='CC', multiplicity=3, charge=0)
    assert spc.multiplicity == 3
    assert spc.charge == 0


def test_species_structure_check():
    """Test that a non-TS species with no structure specification raises"""
    with pytest.raises(ValidationError):
        Species(label='no_structure')
    ts = Species(label='TS0', is_ts=True)
    assert ts.is_ts is True
    yml_spc = Species(label='from_yml', yml_path='spc.yml')
    assert yml_spc.yml_path == 'spc.yml'
    # most_stable_conformer is an index into conformers, not a structure source, so it alone must
    # not satisfy the structure check.
    with pytest.raises(ValidationError):
        Species(label='only_index', most_stable_conformer=0)


def test_species_runtime_keys_modeled():
    """Test that runtime/restart species keys are modeled fields rather than extras"""
    spc = Species(label='H2', smiles='[H][H]', neg_freqs_trshed=[], ts_methods=None)
    assert spc.neg_freqs_trshed == list()
    assert 'neg_freqs_trshed' in Species.model_fields
    assert 'ts_methods' in Species.model_fields


def test_species_typo_key_rejected():
    """Test that a typo in a species key is rejected instead of being silently swallowed
    (a swallowed 'mutliplicity' typo previously led to a wrong auto-determined multiplicity)"""
    with pytest.raises(ValidationError) as excinfo:
        Species(label='CH2', smiles='[CH2]', mutliplicity=1)
    assert 'mutliplicity' in str(excinfo.value)


def test_reaction_typo_key_rejected():
    """Test that a typo in a reaction key is rejected instead of being silently swallowed"""
    with pytest.raises(ValidationError) as excinfo:
        Reaction(label='H + O2 <=> HO2', ts_xyz_gues=['H 0 0 0'])
    assert 'ts_xyz_gues' in str(excinfo.value)


def test_reaction_identification_requires_label_or_reactants_and_products():
    """Test that a reaction is identified by a label or by both reactants and products, and that
    r_species/p_species dicts alone are rejected: ARCReaction.from_dict runs
    set_label_reactants_products before converting those dicts, so they cannot identify a reaction
    at load time and ARC would raise"""
    assert Reaction(label='H + O2 <=> HO2').label == 'H + O2 <=> HO2'
    assert Reaction(reactants=['H', 'O2'], products=['HO2']).reactants == ['H', 'O2']
    with pytest.raises(ValidationError):
        Reaction(r_species=[{'label': 'H', 'smiles': '[H]'}],
                 p_species=[{'label': 'HO2', 'smiles': 'O[O]'}])
    with pytest.raises(ValidationError):
        Reaction(reactants=['H', 'O2'])


def test_species_as_dict_keys_are_modeled():
    """Drift guard: every key ARCSpecies.as_dict() writes must be modeled by Species"""
    spc = ARCSpecies(label='vinoxy', smiles='C=C[O]')
    ts = ARCSpecies(label='TS0', is_ts=True, xyz='O 0 0 0\nH 0 0 1\nH 0 1 0')
    emitted_keys = set(spc.as_dict().keys()) | set(ts.as_dict().keys())
    assert emitted_keys.issubset(set(Species.model_fields)), \
        (f'ARC writes a species key the schema does not model — add it to arc/schema.py or '
         f'restarts will break. Unmodeled: {sorted(emitted_keys - set(Species.model_fields))}')


def test_reaction_as_dict_keys_are_modeled():
    """Drift guard: every key ARCReaction.as_dict() writes must be modeled by Reaction"""
    rxn = ARCReaction(r_species=[ARCSpecies(label='H', smiles='[H]'),
                                 ARCSpecies(label='O2', smiles='[O][O]')],
                      p_species=[ARCSpecies(label='HO2', smiles='O[O]')])
    emitted_keys = set(rxn.as_dict().keys())
    assert emitted_keys.issubset(set(Reaction.model_fields)), \
        (f'ARC writes a reaction key the schema does not model — add it to arc/schema.py or '
         f'restarts will break. Unmodeled: {sorted(emitted_keys - set(Reaction.model_fields))}')


def test_reaction_schema():
    """Test creating an instance of Reaction"""
    rxn = Reaction(label='H + O2 <=> HO2')
    assert rxn.label == 'H + O2 <=> HO2'
    rxn = Reaction(label='N2H4 + NH <=> N2H3 + NH2', multiplicity=3,
                   ts_xyz_guess=['N 0 0 0\nH 0 0 1'])
    assert rxn.multiplicity == 3
    rxn = Reaction(reactants=['H', 'O2'], products=['HO2'])
    assert rxn.reactants == ['H', 'O2']

    with pytest.raises(ValidationError) as excinfo:
        Reaction(label='rxn1', multiplicity=0)
    assert 'multiplicity' in str(excinfo.value)

    with pytest.raises(ValidationError) as excinfo:
        Reaction(label='rxn1', ts_xyz_guess=5)
    assert 'ts_xyz_guess' in str(excinfo.value)

    with pytest.raises(ValidationError):
        Reaction(multiplicity=1)


def test_all_examples_validate():
    """Test that all input.yml files under examples/ validate through ARCInput"""
    example_files = glob.glob(os.path.join(ARC_PATH, 'examples', '**', 'input.yml'), recursive=True)
    assert len(example_files) >= 8
    for example_file in example_files:
        input_dict = read_yaml_file(example_file)
        arc_input = ARCInput(**input_dict)
        assert arc_input.project == input_dict['project']


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


def test_restart_files_round_trip():
    """Test that real restart.yml files validate and round-trip through ARCInput"""
    restart_files = glob.glob(os.path.join(ARC_PATH, 'arc', 'testing', 'restart', '**', 'restart.yml'),
                              recursive=True)
    assert len(restart_files) >= 4
    for restart_file in restart_files:
        restart_dict = read_yaml_file(restart_file)
        arc_input = ARCInput(**restart_dict)
        dumped = arc_input.as_input_dict()
        assert set(dumped.keys()) == set(restart_dict.keys()), restart_file
        for key, original in restart_dict.items():
            if key in ('thermo_adapter', 'kinetics_adapter'):
                # The schema normalizes these to lowercase, mirroring ARC.__init__.
                assert dumped[key] == original.lower()
            else:
                assert _normalize(dumped[key]) == _normalize(original), f'{restart_file}: {key}'


def test_restart_thermo_runtime_keys_preserved():
    """Test that runtime restart keys survive validation as modeled fields"""
    restart_file = os.path.join(ARC_PATH, 'arc', 'testing', 'restart', '1_restart_thermo', 'restart.yml')
    restart_dict = read_yaml_file(restart_file)
    arc_input = ARCInput(**restart_dict)
    dumped = arc_input.as_input_dict()
    assert dumped['output'] == restart_dict['output']
    assert dumped['running_jobs'] == restart_dict['running_jobs']
    assert dumped['ess_settings'] == restart_dict['ess_settings']
    assert dumped['job_types'] == restart_dict['job_types']
    for original_spc, dumped_spc in zip(restart_dict['species'], dumped['species']):
        assert set(dumped_spc.keys()) == set(original_spc.keys())


def test_arc_object_as_dict_round_trip(tmp_path):
    """Test that an ARC object's as_dict() output validates through ARCInput"""
    arc_object = ARC(project='arc_schema_test',
                     project_directory=str(tmp_path),
                     species=[{'label': 'H2', 'smiles': '[H][H]'}],
                     )
    arc_dict = arc_object.as_dict()
    assert arc_dict['schema_version'] == ARC_INPUT_SCHEMA_VERSION
    assert arc_dict['arc_version'] == VERSION
    arc_input = ARCInput(**arc_dict)
    dumped = arc_input.as_input_dict()
    assert set(dumped.keys()) == set(arc_dict.keys()) - {'schema_version', 'arc_version'}
    assert dumped['project'] == 'arc_schema_test'
    assert dumped['job_types'] == arc_dict['job_types']
    assert _normalize(dumped['species']) == _normalize(arc_dict['species'])


def test_as_input_dict_preserves_explicit_none_and_omits_unset():
    """Test that as_input_dict reproduces exactly the user-provided keys"""
    arc_input = ARCInput(project='p', bac_type=None, species=[{'label': 'H2', 'smiles': '[H][H]'}])
    dumped = arc_input.as_input_dict()
    assert set(dumped.keys()) == {'project', 'bac_type', 'species'}
    assert dumped['bac_type'] is None
    assert dumped['species'] == [{'label': 'H2', 'smiles': '[H][H]'}]


# Keys excluded from the ARC.as_dict() equivalence comparison in test_schema_seam_is_behavior_preserving,
# each with the concrete, investigated reason it is inherently non-deterministic between two separate
# ARC constructions (as opposed to a real behavioral difference introduced by the schema seam).
_AS_DICT_EXCLUDED_KEYS = {
    # Absolute path embedding the per-construction tmp_path, which necessarily differs between
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


@pytest.mark.parametrize('example_file', [
    os.path.join(ARC_PATH, 'examples', 'minimal', 'input.yml'),
    os.path.join(ARC_PATH, 'examples', 'Stationary', 'thermo_demo', 'input.yml'),
    os.path.join(ARC_PATH, 'examples', 'Reactions', 'rates_demo', 'input.yml'),
    os.path.join(ARC_PATH, 'arc', 'testing', 'restart', '2_restart_rate', 'restart.yml'),
    os.path.join(ARC_PATH, 'arc', 'testing', 'restart', '5_TS1', 'restart.yml'),
])
def test_schema_seam_is_behavior_preserving(example_file, tmp_path):
    """Test that constructing ARC through the schema seam is equivalent to constructing it
    directly from the raw dict, for real example input files"""
    raw_dict = read_yaml_file(example_file)

    # ARC.__init__ mutates the 'species'/'reactions' lists it is given in place (converting
    # dicts to ARCSpecies/ARCReaction objects), so each construction needs its own copy of raw_dict.
    arc_raw = ARC(**copy.deepcopy(raw_dict), project_directory=str(tmp_path / 'raw'))
    schema_input_dict = ARCInput(**copy.deepcopy(raw_dict)).as_input_dict()
    schema_input_dict['project_directory'] = str(tmp_path / 'schema')
    arc_schema = ARC(**schema_input_dict)

    raw_as_dict = arc_raw.as_dict()
    schema_as_dict = arc_schema.as_dict()

    assert set(raw_as_dict.keys()) == set(schema_as_dict.keys())
    for key in raw_as_dict:
        if key in _AS_DICT_EXCLUDED_KEYS:
            continue
        raw_value = _normalize_transient_mol_atom_ids(raw_as_dict[key])
        schema_value = _normalize_transient_mol_atom_ids(schema_as_dict[key])
        assert raw_value == schema_value, \
            f'{example_file}: key {key!r} differs: {raw_as_dict[key]!r} != {schema_as_dict[key]!r}'


def test_schema_version_absent_accepted():
    """Test that an input without a schema_version key validates silently"""
    arc_input = ARCInput(project='p')
    assert arc_input.schema_version is None
    assert 'schema_version' not in arc_input.as_input_dict()


def test_schema_version_current_accepted():
    """Test that the current schema version is accepted"""
    arc_input = ARCInput(project='p', schema_version=ARC_INPUT_SCHEMA_VERSION)
    assert arc_input.schema_version == ARC_INPUT_SCHEMA_VERSION


def test_schema_version_explicit_null_rejected():
    """Test that an explicit schema_version: null raises"""
    with pytest.raises(ValidationError) as excinfo:
        ARCInput(project='p', schema_version=None)
    assert 'schema_version' in str(excinfo.value)


def test_schema_version_newer_rejected():
    """Test that a schema_version from a future ARC raises with both versions named"""
    with pytest.raises(ValidationError) as excinfo:
        ARCInput(project='p', schema_version=ARC_INPUT_SCHEMA_VERSION + 1)
    assert 'schema_version' in str(excinfo.value)
    assert str(ARC_INPUT_SCHEMA_VERSION + 1) in str(excinfo.value)
    assert str(ARC_INPUT_SCHEMA_VERSION) in str(excinfo.value)


def test_schema_version_older_warns_and_proceeds():
    """Test that an older schema_version logs a warning and still validates"""
    arc_logger = get_logger()
    records = list()
    handler = logging.Handler()
    handler.emit = records.append
    arc_logger.addHandler(handler)
    try:
        arc_input = ARCInput(project='p', schema_version=0)
    finally:
        arc_logger.removeHandler(handler)
    assert arc_input.schema_version == 0
    assert any(record.levelno == logging.WARNING and 'schema_version 0' in record.getMessage()
               and str(ARC_INPUT_SCHEMA_VERSION) in record.getMessage()
               for record in records)


def test_schema_version_and_arc_version_stripped_from_input_dict():
    """Test that as_input_dict() strips schema_version and arc_version"""
    arc_input = ARCInput(project='p', schema_version=ARC_INPUT_SCHEMA_VERSION, arc_version=VERSION)
    dumped = arc_input.as_input_dict()
    assert 'schema_version' not in dumped
    assert 'arc_version' not in dumped
    assert dumped == {'project': 'p'}


def test_validate_input_dict_minimal_example(tmp_path):
    """Test that ARC constructs from validate_input_dict() on a real example input file"""
    input_dict = read_yaml_file(os.path.join(ARC_PATH, 'examples', 'minimal', 'input.yml'))
    input_dict['project_directory'] = str(tmp_path)
    arc_object = ARC(**validate_input_dict(input_dict))
    assert arc_object.project == input_dict['project']


def test_validate_input_dict_versioned_round_trip(tmp_path):
    """Test a full round trip: construct ARC, as_dict() it (now versioned), validate, reconstruct.
    Also test that raw ARC(**restart_dict) tolerates the version keys directly"""
    arc_object = ARC(project='arc_schema_version_round_trip',
                     project_directory=str(tmp_path / 'first'),
                     species=[{'label': 'H2', 'smiles': '[H][H]'}],
                     )
    restart_dict = arc_object.as_dict()
    assert restart_dict['schema_version'] == ARC_INPUT_SCHEMA_VERSION
    assert restart_dict['arc_version'] == VERSION

    validated = validate_input_dict(copy.deepcopy(restart_dict))
    assert 'schema_version' not in validated
    assert 'arc_version' not in validated
    validated['project_directory'] = str(tmp_path / 'second')
    arc_object_2 = ARC(**validated)
    assert arc_object_2.project == 'arc_schema_version_round_trip'
    assert arc_object_2.__version__ == VERSION

    raw_dict = copy.deepcopy(restart_dict)
    raw_dict['project_directory'] = str(tmp_path / 'third')
    arc_object_3 = ARC(**raw_dict)
    assert arc_object_3.project == 'arc_schema_version_round_trip'
    assert arc_object_3.__version__ == VERSION
