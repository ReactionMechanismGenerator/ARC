"""
ARC's schema module for input validation.

Validates an ARC input (or restart) dictionary at load time, before instantiating ARC.
Field names and defaults mirror the ``ARC.__init__`` keyword arguments and the key names
emitted by ``ARC.as_dict()`` into ``restart.yml``.
"""

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field, StrictBool, StrictInt, field_serializer, field_validator, model_validator

from arc.common import ARC_INPUT_SCHEMA_VERSION, get_logger
from arc.job.factory import get_registered_job_adapters
from arc.settings.settings import default_job_types
from arc.statmech.adapter import StatmechEnum


logger = get_logger()


class BACTypeEnum(str, Enum):
    """
    The supported bond additivity correction types.
    'p' for Petersson-type and 'm' for Melius-type BAC.
    """
    petersson = 'p'
    melius = 'm'


class TemperatureUnitEnum(str, Enum):
    """
    The supported units for a T_min/T_max (value, unit) pair.
    Only Kelvin is supported: neither ``plotter.py`` nor ``processor.py`` convert the unit
    before using the value, so any other unit would be silently treated as Kelvin.
    """
    kelvin = 'K'


class JobTypes(BaseModel):
    """
    A class for validating the ``job_types`` argument of an ARC input.
    """
    conf_opt: StrictBool = default_job_types['conf_opt']
    conf_sp: StrictBool = default_job_types['conf_sp']
    opt: StrictBool = default_job_types['opt']
    fine_grid: StrictBool = default_job_types['fine_grid']
    freq: StrictBool = default_job_types['freq']
    sp: StrictBool = default_job_types['sp']
    rotors: StrictBool = default_job_types['rotors']
    irc: StrictBool = default_job_types['irc']
    orbitals: StrictBool = default_job_types['orbitals']
    lennard_jones: StrictBool = default_job_types['lennard_jones']
    bde: StrictBool = default_job_types['bde']
    fine: StrictBool | None = None
    onedmin: StrictBool | None = None

    class Config:
        extra = "forbid"

    @model_validator(mode='after')
    def check_fine_not_explicit_none(self) -> 'JobTypes':
        """
        Reject an explicit null for ``fine`` while still allowing it to be absent.

        ``initialize_job_types`` in ``arc/common.py`` fills ``fine``'s default (``True``) only when
        the key is missing from ``job_types``. An explicit ``fine: null`` is present, so it skips
        default-filling and stays ``None``, which is falsy and silently disables the job type. An
        absent ``fine`` is unaffected and still receives the default. ``onedmin`` is not guarded:
        it defaults to ``False``, so an explicit ``onedmin: null`` is behaviorally identical to
        omitting it.

        Returns:
            JobTypes: This instance, unchanged, once validated.
        """
        if 'fine' in self.model_fields_set and self.fine is None:
            raise ValueError(
                "job_types.fine was explicitly set to null. ARC only fills the default value for "
                "'fine' when it is absent from job_types; an explicit null is kept as None "
                "(falsy), which would silently disable fine-grid optimization. Omit 'fine' "
                "entirely to use the default, or set it to true/false explicitly.")
        return self

    @model_validator(mode='after')
    def check_alias_pairs_not_contradictory(self) -> 'JobTypes':
        """
        Reject contradictory values between a legacy/input alias and its ARC-internal counterpart.

        ``initialize_job_types`` in ``arc/common.py`` unconditionally overwrites the internal key
        (``fine``, ``onedmin``) with the legacy/input key's value (``fine_grid``, ``lennard_jones``)
        whenever both are present, silently discarding the internal key's value with no warning.
        Raise instead of allowing that silent data loss when the two explicitly disagree.

        Returns:
            JobTypes: This instance, unchanged, once validated.
        """
        fields_set = self.model_fields_set
        for input_name, internal_name in (('fine_grid', 'fine'), ('lennard_jones', 'onedmin')):
            if input_name in fields_set and internal_name in fields_set:
                input_value = getattr(self, input_name)
                internal_value = getattr(self, internal_name)
                if input_value != internal_value:
                    raise ValueError(
                        f"Contradictory job_types values: '{input_name}'={input_value} but "
                        f"'{internal_name}'={internal_value}. '{input_name}' takes precedence over "
                        f"'{internal_name}' in ARC, so '{internal_name}' would be silently discarded; "
                        f"set both to the same value or specify only one of them.")
        return self


class Species(BaseModel):
    """
    A class for validating a species entry of an ARC input.
    Mirrors the ``ARCSpecies.__init__`` input surface plus every runtime/restart key that
    ``ARCSpecies.as_dict()`` writes; unknown keys are rejected.
    """
    label: str
    active: dict | None = None
    adjlist: str | None = None
    adaptive_lot_n_heavy: StrictInt | None = None
    arkane_file: str | None = None
    bdes: list | None = None
    bond_corrections: dict | None = None
    charge: StrictInt | None = None
    cheap_conformer: dict | str | None = None
    checkfile: str | None = None
    chosen_ts: StrictInt | None = None
    chosen_ts_list: list | None = None
    chosen_ts_method: str | None = None
    compute_thermo: StrictBool | None = None
    conf_is_isomorphic: StrictBool | None = None
    conformer_energies: list | None = None
    conformers: list | None = None
    conformers_before_opt: list | None = None
    consider_all_diastereomers: StrictBool = True
    directed_rotors: dict | None = None
    e0: float | None = None
    e0_only: StrictBool = False
    e_elect: float | None = None
    external_symmetry: Annotated[int, Field(ge=1, strict=True)] | None = None
    final_xyz: dict | str | None = None
    force_field: str = 'MMFF94s'
    fragments: list | None = None
    freqs: list | None = None
    inchi: str | None = None
    include_in_thermo_lib: StrictBool | None = True
    index: StrictInt | None = None
    initial_xyz: dict | str | None = None
    irc_label: str | None = None
    is_ts: StrictBool = False
    keep_mol: StrictBool = False
    long_thermo_description: str | None = None
    mol: str | dict | None = None
    mol_list: list | None = None
    most_stable_conformer: StrictInt | None = None
    multi_species: str | None = None
    multiplicity: Annotated[int, Field(ge=1, strict=True)] | None = None
    neg_freqs_trshed: list | None = None
    number_of_radicals: Annotated[int, Field(ge=0, strict=True)] | None = None
    number_of_rotors: StrictInt | None = None
    opt_level: str | None = None
    optical_isomers: Annotated[int, Field(ge=1, strict=True)] | None = None
    original_label: str | None = None
    preserve_param_in_scan: list | None = None
    project_directory: str | None = None
    radius: float | None = None
    recent_md_conformer: dict | str | None = None
    rotors_dict: dict | None = None
    run_time: float | None = None
    rxn_index: StrictInt | None = None
    rxn_label: str | None = None
    rxn_zone_atom_indices: list | None = None
    smiles: str | None = None
    successful_methods: list | None = None
    t1: float | None = None
    thermo_at_own_level: StrictBool = False
    ts_checks: dict | None = None
    ts_conf_spawned: StrictBool | None = None
    ts_guesses: list | None = None
    ts_guesses_exhausted: StrictBool | None = None
    ts_methods: list | None = None
    ts_number: StrictInt | None = None
    ts_report: str | None = None
    tsg_spawned: StrictBool | None = None
    unsuccessful_methods: list | None = None
    xyz: list | dict | str | None = None
    yml_path: str | None = None
    zmat: dict | None = None

    class Config:
        extra = "forbid"

    @model_validator(mode='after')
    def check_structure_specified(self) -> 'Species':
        """
        Ensure a non-TS species has at least one structure specification.

        Returns:
            Species: The validated Species instance.
        """
        if not self.is_ts:
            sources = (self.smiles, self.inchi, self.adjlist, self.xyz, self.mol, self.yml_path,
                       self.initial_xyz, self.final_xyz, self.conformers, self.cheap_conformer)
            if not any(sources):
                raise ValueError(f'A non-TS species must have at least one structure specification '
                                 f'(SMILES, InChI, adjlist, xyz, mol, or a yml_path). '
                                 f'Got none for species "{self.label}".')
        return self


class Reaction(BaseModel):
    """
    A class for validating a reaction entry of an ARC input.
    Mirrors the ``ARCReaction.__init__`` input surface plus every runtime/restart key that
    ``ARCReaction.as_dict()`` writes; unknown keys are rejected.
    """
    label: str = ''
    atom_map: list | None = None
    charge: StrictInt | None = None
    done_opt_r_n_p: StrictBool | None = None
    family: str | None = None
    family_own_reverse: StrictBool | StrictInt | None = None
    index: StrictInt | None = None
    kinetics: dict | None = None
    long_kinetic_description: str | None = None
    multiplicity: Annotated[int, Field(ge=1, strict=True)] | None = None
    p_species: list | None = None
    preserve_param_in_scan: list | None = None
    products: list[str] | None = None
    r_species: list | None = None
    reactants: list[str] | None = None
    ts_label: str | None = None
    ts_methods: list | None = None
    ts_species: dict | None = None
    ts_xyz_guess: list | str | None = None
    xyz: list | str | None = None

    class Config:
        extra = "forbid"

    @model_validator(mode='after')
    def check_reaction_identified(self) -> 'Reaction':
        """
        Ensure the reaction can be identified via a label, or both reactants and products.

        ``ARCReaction.from_dict`` calls ``set_label_reactants_products`` before it converts the
        ``r_species``/``p_species`` dicts into species objects, so those dicts alone cannot identify
        a reaction at load time and ARC would raise. Only a ``label`` or explicit ``reactants`` and
        ``products`` label lists satisfy ARC.

        Returns:
            Reaction: The validated Reaction instance.
        """
        if not self.label and not (self.reactants and self.products):
            raise ValueError('A reaction must have either a label, or both reactants and products.')
        return self


class ARCInput(BaseModel):
    """
    A class for validating an ARC input (or restart) dictionary.
    Field names and defaults mirror ``ARC.__init__``; unknown keys are rejected.
    """
    project: str
    adaptive_levels: list | dict | None = None
    allow_nonisomorphic_2d: StrictBool = False
    arc_version: str | None = None
    arkane_level_of_theory: str | dict | None = None
    bac_type: BACTypeEnum | None = 'p'
    bath_gas: str | None = None
    calc_freq_factor: StrictBool = True
    compare_to_rmg: StrictBool = True
    composite_method: str | dict | None = None
    compute_rates: StrictBool = True
    compute_thermo: StrictBool = True
    compute_transport: StrictBool = False
    conformer_level: str | dict | None = None
    conformer_opt_level: str | dict | None = None
    conformer_sp_level: str | dict | None = None
    dont_gen_confs: list[str] | None = None
    e_confs: Annotated[float, Field(ge=0)] = 5.0
    ess_settings: dict | None = None
    freq_level: str | dict | None = None
    freq_scale_factor: Annotated[float, Field(gt=0)] | None = None
    irc_level: str | dict | None = None
    job_memory: Annotated[int, Field(gt=0, strict=True)] | None = None
    job_types: JobTypes | None = None
    keep_checks: StrictBool = False
    kinetics_adapter: StatmechEnum = 'arkane'
    level_of_theory: str | None = ''
    max_job_time: Annotated[float, Field(ge=0)] | None = None
    n_confs: Annotated[int, Field(gt=0, strict=True)] | None = 10
    opt_level: str | dict | None = None
    orbitals_level: str | dict | None = None
    output: dict | None = None
    output_multi_spc: dict | None = None
    project_directory: str | None = None
    reactions: list[Reaction] | None = None
    report_e_elect: StrictBool | None = False
    running_jobs: dict | None = None
    scan_level: str | dict | None = None
    schema_version: StrictInt | None = None
    skip_nmd: StrictBool | None = False
    sp_level: str | dict | None = None
    species: list[Species] | None = None
    specific_job_type: str | None = ''
    T_count: Annotated[int, Field(gt=0, strict=True)] | None = 50
    T_max: Annotated[float, Field(gt=0)] | tuple[Annotated[float, Field(gt=0)], TemperatureUnitEnum] | None = None
    T_min: Annotated[float, Field(gt=0)] | tuple[Annotated[float, Field(gt=0)], TemperatureUnitEnum] | None = None
    thermo_adapter: StatmechEnum = 'arkane'
    trsh_ess_jobs: StrictBool = True
    trsh_rotors: StrictBool = True
    ts_adapters: list[str] | None = None
    ts_guess_level: str | dict | None = None
    verbose: Annotated[int, Field(ge=10, le=50, multiple_of=10, strict=True)] = 20

    class Config:
        extra = "forbid"
        use_enum_values = True

    @model_validator(mode='after')
    def check_schema_version(self) -> 'ARCInput':
        """
        Validate the ``schema_version`` key of an input/restart file against the running ARC.

        An absent ``schema_version`` is accepted silently (files written before versioning).
        An explicit null is rejected, a version newer than the running ARC's
        ``ARC_INPUT_SCHEMA_VERSION`` is rejected, and an older version logs a warning.

        Returns:
            ARCInput: This instance, unchanged, once validated.
        """
        if 'schema_version' in self.model_fields_set:
            if self.schema_version is None:
                raise ValueError(
                    f'schema_version was explicitly set to null. Omit the key entirely, or set it '
                    f'to the input schema version this ARC writes ({ARC_INPUT_SCHEMA_VERSION}).')
            if self.schema_version > ARC_INPUT_SCHEMA_VERSION:
                raise ValueError(
                    f'schema_version {self.schema_version} is newer than the input schema version '
                    f'{ARC_INPUT_SCHEMA_VERSION} supported by this ARC. This file was written by a '
                    f'newer ARC; upgrade ARC to run it.')
            if self.schema_version < ARC_INPUT_SCHEMA_VERSION:
                logger.warning(f'schema_version {self.schema_version} is older than the input schema '
                               f'version {ARC_INPUT_SCHEMA_VERSION} supported by this ARC, '
                               f'proceeding anyway.')
        return self

    @field_serializer('T_min', 'T_max')
    def serialize_temperature_tuple(self, value: float | tuple[float, str] | None) -> float | list | None:
        """
        Serialize a (value, unit) temperature tuple as a plain list, leaving a bare scalar as-is.

        Plain ``yaml.dump`` serializes a Python tuple with a ``!!python/tuple`` tag, which
        corrupts restart.yml round-trips; a plain list dumps as an ordinary YAML sequence. A bare
        scalar (raw ARC's shorthand for a Kelvin temperature) must round-trip as a scalar, since
        ``processor.py`` normalizes it downstream and normalizing it here would change the
        restart file shape.

        Args:
            value: The validated (value, unit) tuple, a bare scalar, or None.

        Returns:
            The (value, unit) pair as a list, the bare scalar unchanged, or None.
        """
        if isinstance(value, tuple):
            return list(value)
        return value

    @field_validator('thermo_adapter', 'kinetics_adapter', 'bac_type', mode='before')
    @classmethod
    def lowercase_adapter_and_bac_type(cls, value):
        """
        Lowercase a statmech adapter/BAC type name before enum validation, mirroring ARC.__init__
        (``arc/arkane.py`` compares ``bac_type.lower()`` against single-letter codes).

        Args:
            value: The raw field value.

        Returns:
            The lowercased value if it is a string, otherwise the original value.
        """
        return value.lower() if isinstance(value, str) else value

    @field_validator('job_memory', 'n_confs', 'T_count', mode='before')
    @classmethod
    def coerce_integral_float(cls, value):
        """
        Convert an integral float (e.g., ``14.0``) to an int before strict-int validation.

        Raw ARC accepts a float memory/count value, so a previously-working input file must
        not hard-fail at load merely for spelling an integer as a float. Non-integral floats
        and strings are left untouched, so they still fail the field's strict-int validation.

        Args:
            value: The raw field value.

        Returns:
            The value as an int if it was an integral float, otherwise the original value.
        """
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return value

    @field_validator('allow_nonisomorphic_2d', 'calc_freq_factor', 'compare_to_rmg', 'compute_rates',
                     'compute_thermo', 'compute_transport', 'keep_checks', 'report_e_elect',
                     'skip_nmd', 'trsh_ess_jobs', 'trsh_rotors', mode='before')
    @classmethod
    def coerce_int_flag(cls, value):
        """
        Convert an integer ``0``/``1`` to a bool before strict-bool validation of a run flag.

        Raw ARC treats these flags with plain truthiness, so a previously-working input file that
        spells a flag as ``0``/``1`` must not hard-fail at load. Any other value (including a
        quoted ``"False"`` or an out-of-range int like ``2``) is left untouched, so it still fails
        the field's strict-bool validation and cannot silently invert the flag.

        Args:
            value: The raw field value.

        Returns:
            ``False``/``True`` if the value was ``0``/``1``, otherwise the original value.
        """
        if isinstance(value, int) and not isinstance(value, bool) and value in (0, 1):
            return bool(value)
        return value

    @field_validator('ts_adapters')
    @classmethod
    def check_ts_adapters(cls, value):
        """
        ARCInput.ts_adapters validator.

        Args:
            value: The ts_adapters list to validate.

        Returns:
            The validated ts_adapters list.
        """
        if value is not None:
            registered_job_adapters = get_registered_job_adapters()
            for ts_adapter in value:
                if ts_adapter.lower() not in registered_job_adapters.keys():
                    raise ValueError(f'Unknown TS adapter: "{ts_adapter}". Registered job adapters are: '
                                     f'{[adapter.value for adapter in registered_job_adapters.keys()]}')
        return value

    def as_input_dict(self) -> dict:
        """
        Return a plain dict suitable for ARC(**...), enums dumped as their string values.

        Dumps with ``exclude_unset=True`` so the result reproduces exactly the keys the user
        provided (including explicit ``None`` values, which ARC treats differently from absent
        keys, e.g. ``bac_type: None`` disables BAC while an absent ``bac_type`` defaults to 'p';
        ``ARCSpecies.from_dict()`` gates on key presence, so injecting default/None-valued keys
        would change behavior).

        Strips ``schema_version`` and ``arc_version`` so they never reach ``ARC(**...)``.

        Returns:
            dict: The dictionary representation of the validated input.
        """
        dumped = self.model_dump(exclude_unset=True)
        dumped.pop('schema_version', None)
        dumped.pop('arc_version', None)
        return dumped


def validate_input_dict(input_dict: dict) -> dict:
    """
    Validate an ARC input/restart dictionary and return kwargs safe for ARC(**...).

    Args:
        input_dict (dict): The raw ARC input (or restart) dictionary to validate.

    Returns:
        dict: The validated dictionary, safe to unpack into ``ARC(**...)``.
    """
    return ARCInput(**input_dict).as_input_dict()
